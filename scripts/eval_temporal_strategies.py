r"""First temporal-vs-DVC-strategy evaluation on synthetic noise textures.

Wraps :func:`mamba_dvc.validate.series_error.evaluate_synthetic` with a
fixed, opinionated experiment shape:

- band-limited noise reference (:func:`make_texture`),
- a moderate time-parameterized field combining a per-step rigid shift
  with a small uniform dilation, scaled linearly in time,
- a sweep over ``{REFERENCE_ANCHORED, SEQUENTIAL}`` x ``lags``,
- prints a per-pair + cumulative-drift summary, and optionally writes
  an NPZ + a markdown report.

The defaults target a fast iteration loop (192³, 16 frames, lag set
``(1, 2, 4)``) — expected runtime is order of minutes on one A6000.

Run with::

    uv run python scripts/eval_temporal_strategies.py
    uv run python scripts/eval_temporal_strategies.py --ephemeral
    uv run python scripts/eval_temporal_strategies.py --shape 256 --timesteps 21

The ``--ephemeral`` flag skips all filesystem writes (no NPZ, no
markdown, no output directory created). Use it when you only want to
smoke-test the pipeline or compare two configurations interactively.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from mamba_dvc.run.eventlog import DispatchLogger, SeriesPairLogger, SessionScope
from mamba_dvc.types import PairingStrategy, SeriesPairStatus
from mamba_dvc.validate.series_error import (
    CumulativeDriftTable,
    PerPairErrorTable,
    SyntheticEvalReport,
    evaluate_synthetic,
)
from mamba_dvc.validate.synthetic import (
    compose,
    linear_motion,
    make_texture,
    rigid_shift,
    uniform_dilation,
)

_DEFAULT_STRATEGIES: tuple[PairingStrategy, ...] = (
    PairingStrategy.REFERENCE_ANCHORED,
    PairingStrategy.SEQUENTIAL,
)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(",") if x.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the synthetic evaluation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        default=192,
        help="Cubic volume side length in voxels. Default: 192.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=16,
        help="Number of timesteps (range(0, N)). Default: 16.",
    )
    parser.add_argument(
        "--lags",
        type=_parse_csv_ints,
        default=(1, 2, 4),
        help="Comma-separated lag values. Default: 1,2,4.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=64,
        help=(
            "Subvolume window size. Default 64 fits ~125 POIs in a 192**3"
            " volume at overlap=0.5; for shape >= 256 the W=96 plan default"
            " is appropriate."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap fraction in [0, 1). Default: 0.5.",
    )
    parser.add_argument(
        "--shift",
        type=_parse_csv_floats,
        default=(0.5, 0.8, 0.2),
        help=(
            "Per-step rigid shift in voxels, (dz,dy,dx). Default: 0.5,0.8,0.2."
            " Scaled by t via linear_motion."
        ),
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=5e-4,
        help=(
            "Per-step uniform-dilation strain about the volume center."
            " Default: 5e-4. Scaled by t via linear_motion."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the noise-texture RNG. Default: 0.",
    )
    parser.add_argument(
        "--texture-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for make_texture (voxels). Default: 1.5.",
    )
    parser.add_argument(
        "--warp-order",
        type=int,
        default=3,
        help="Spline order for frame synthesis (1 or 3). Default: 3.",
    )
    parser.add_argument(
        "--devices",
        type=_parse_csv_ints,
        default=None,
        help=(
            "Comma-separated CUDA device ids to dispatch correlation across."
            " When set, evaluate_synthetic opens one MultiGPUDispatcher for"
            " the whole sweep so the spawn + CUDA-init cost is paid once."
            " Default: unset (host-only CPU path)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/temporal-eval"),
        help="Output directory for NPZ + markdown report. Ignored with --ephemeral.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Skip all filesystem writes; print summary only.",
    )
    return parser.parse_args(argv)


def _row_stats(arr: np.ndarray) -> str:
    finite_mask = np.isfinite(arr)
    if not bool(finite_mask.any()):
        return "n/a (no finite rows)"
    finite = arr[finite_mask]
    return (
        f"min={float(finite.min()):.4f}  "
        f"med={float(np.median(finite)):.4f}  "
        f"max={float(finite.max()):.4f}  "
        f"n={int(finite_mask.sum())}"
    )


def _per_pair_block(strategy: PairingStrategy, lag: int, table: PerPairErrorTable) -> str:
    n_fail = int(np.sum(table.pair_status == int(SeriesPairStatus.FAILED)))
    header = f"[{strategy.value}, lag={lag}]  pairs={len(table.mae)}  failed={n_fail}"
    lines = [
        header,
        f"  mae  : {_row_stats(table.mae)}",
        f"  rmse : {_row_stats(table.rmse)}",
        f"  p95  : {_row_stats(table.p95)}",
    ]
    return "\n".join(lines)


def _cumulative_block(strategy: PairingStrategy, lag: int, table: CumulativeDriftTable) -> str:
    if len(table.mae) == 0:
        return f"[{strategy.value}, lag={lag}]  steps=0 (no OK pairs)"
    first_mae = float(table.mae[0]) if np.isfinite(table.mae[0]) else float("nan")
    last_mae = float(table.mae[-1]) if np.isfinite(table.mae[-1]) else float("nan")
    growth = last_mae - first_mae
    return (
        f"[{strategy.value}, lag={lag}]  steps={len(table.mae)}\n"
        f"  drift mae : first={first_mae:.4f}  last={last_mae:.4f}  "
        f"growth={growth:+.4f}\n"
        f"  drift p95 : last={float(table.p95[-1]):.4f}"
    )


def format_summary(report: SyntheticEvalReport) -> str:
    """Render the per-pair and cumulative-drift tables as a printable summary."""
    rule = "=" * 72
    sections: list[str] = []

    sections.append(rule)
    sections.append("Per-pair error (estimated minus analytical inter-frame GT)")
    sections.append(rule)
    for strategy, lag in sorted(report.per_pair.keys(), key=lambda k: (k[0].value, k[1])):
        sections.append(_per_pair_block(strategy, lag, report.per_pair[(strategy, lag)]))

    if report.cumulative:
        sections.append("")
        sections.append(rule)
        sections.append("Cumulative drift (SEQUENTIAL only; composed vs absolute GT)")
        sections.append(rule)
        for strategy, lag in sorted(
            report.cumulative.keys(), key=lambda k: (k[0].value, k[1])
        ):
            sections.append(
                _cumulative_block(strategy, lag, report.cumulative[(strategy, lag)])
            )

    return "\n".join(sections)


def _flatten_for_npz(report: SyntheticEvalReport) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for (strategy, lag), table in report.per_pair.items():
        prefix = f"per_pair__{strategy.value}__lag{lag}"
        for name in table.__dataclass_fields__:
            payload[f"{prefix}__{name}"] = np.asarray(getattr(table, name))
    for (strategy, lag), table in report.cumulative.items():
        prefix = f"cumulative__{strategy.value}__lag{lag}"
        for name in table.__dataclass_fields__:
            payload[f"{prefix}__{name}"] = np.asarray(getattr(table, name))
    payload["timesteps"] = np.asarray(report.timesteps, dtype=np.int64)
    payload["lags"] = np.asarray(report.lags, dtype=np.int64)
    payload["strategies"] = np.asarray([s.value for s in report.strategies])
    return payload


def write_outputs(
    args: argparse.Namespace,
    report: SyntheticEvalReport,
    summary: str,
    elapsed_seconds: float,
) -> tuple[Path, Path]:
    """Persist the NPZ tables and markdown report; return the two paths."""
    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / "tables.npz"
    md_path = args.out / "report.md"

    np.savez(npz_path, **_flatten_for_npz(report))  # pyright: ignore[reportArgumentType]

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Temporal strategy evaluation\n\n")
        fh.write("## Config\n\n")
        fh.write(f"- shape: ({args.shape}, {args.shape}, {args.shape})\n")
        fh.write(f"- timesteps: range(0, {args.timesteps})\n")
        fh.write(f"- lags: {tuple(args.lags)}\n")
        fh.write(f"- window: {args.window}  overlap: {args.overlap}\n")
        fh.write(f"- shift/step (dz,dy,dx): {tuple(args.shift)}\n")
        fh.write(f"- dilation strain/step: {args.strain}\n")
        fh.write(f"- texture sigma: {args.texture_sigma}  seed: {args.seed}\n")
        fh.write(f"- warp order: {args.warp_order}\n")
        fh.write(f"- elapsed: {elapsed_seconds:.1f} s\n\n")
        fh.write("## Summary\n\n```\n")
        fh.write(summary)
        fh.write("\n```\n")

    return npz_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: build the texture, run the sweep, summarize, optionally persist."""
    args = parse_args(argv)
    shape: tuple[int, int, int] = (args.shape, args.shape, args.shape)
    timesteps = tuple(range(args.timesteps))

    if len(args.shift) != 3:
        print(f"--shift must have 3 entries (dz,dy,dx), got {args.shift}", file=sys.stderr)
        return 2
    shift = (float(args.shift[0]), float(args.shift[1]), float(args.shift[2]))

    print(
        f"Building texture shape={shape} sigma={args.texture_sigma} seed={args.seed} ...",
        flush=True,
    )
    reference = make_texture(shape, sigma=args.texture_sigma, seed=args.seed)

    center = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
    velocity = compose(rigid_shift(shift), uniform_dilation(args.strain, center))
    u_of_t = linear_motion(velocity)

    print(
        f"Field: linear_motion(rigid_shift({shift}) + uniform_dilation(strain={args.strain},"
        f" center={center}))",
        flush=True,
    )
    print(
        f"Sweep: strategies={[s.value for s in _DEFAULT_STRATEGIES]}  "
        f"lags={tuple(args.lags)}  timesteps={timesteps[0]}..{timesteps[-1]}",
        flush=True,
    )
    if args.devices is not None:
        print(f"Dispatcher: MultiGPUDispatcher device_ids={tuple(args.devices)}", flush=True)
    else:
        print("Dispatcher: none (host-only CPU path)", flush=True)
    if args.ephemeral:
        print("Mode: ephemeral (no filesystem writes)", flush=True)
    else:
        print(f"Mode: persistent  --out={args.out}", flush=True)

    # Persistent runs: open a SessionScope around the whole sweep so
    # every event (sweep_start / pair_start / pair_end / dispatch_* /
    # phase records / warnings) lands in events.jsonl under args.out.
    # Ephemeral runs skip the scope so no file is created. The
    # ``series`` binding describes the synthetic experiment shape so
    # multiple runs against the same out dir can be split apart by
    # df.groupby("series") at analysis time.
    series_id = f"synthetic-shape{args.shape}-seed{args.seed}-ts{args.timesteps}"
    if args.ephemeral:
        scope_cm: Any = nullcontext()
        pair_observer = None
        dispatch_observer = None
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        scope_cm = SessionScope(args.out, series=series_id)
        pair_observer = SeriesPairLogger()
        # Only thread DispatchLogger through when device_ids is set;
        # evaluate_synthetic warns (and skips) when it is not.
        dispatch_observer = DispatchLogger() if args.devices is not None else None

    t0 = time.perf_counter()
    with scope_cm:
        # sweep_start / sweep_end are only meaningful when an
        # events.jsonl is open underneath; in ephemeral mode they
        # would just leak structlog lines to stderr, duplicating the
        # print() summary above. Gate on the persistent branch.
        if not args.ephemeral:
            eventlog = structlog.get_logger("mamba_dvc.eventlog")
            eventlog.info(
                "sweep_start",
                strategies=[s.value for s in _DEFAULT_STRATEGIES],
                lags=list(args.lags),
                timesteps=list(timesteps),
                shape=list(shape),
                window=args.window,
                overlap=args.overlap,
                devices=list(args.devices) if args.devices is not None else None,
            )
        report = evaluate_synthetic(
            reference,
            u_of_t,
            timesteps,
            strategies=_DEFAULT_STRATEGIES,
            lags=args.lags,
            device_ids=args.devices,
            window=args.window,
            overlap=args.overlap,
            warp_order=args.warp_order,
            pair_observer=pair_observer,
            dispatch_observer=dispatch_observer,
        )
        elapsed = time.perf_counter() - t0
        if not args.ephemeral:
            eventlog.info(
                "sweep_end",
                duration_s=round(elapsed, 3),
                n_per_pair_tables=len(report.per_pair),
                n_cumulative_tables=len(report.cumulative),
            )

    summary = format_summary(report)
    print()
    print(summary)
    print()
    print(f"Elapsed: {elapsed:.1f} s")

    if args.ephemeral:
        print("[ephemeral] skipping filesystem writes.")
        return 0

    npz_path, md_path = write_outputs(args, report, summary, elapsed)
    print(f"Wrote {npz_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {args.out / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
