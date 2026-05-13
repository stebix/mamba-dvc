r"""End-to-end driver for the v1 multi-GPU DVC pipeline against a zarr store.

Quick-and-dirty manual perf harness, not part of the test suite. Opens a
:class:`mamba_dvc.io.dataset.DvcDataset`, materializes one
:class:`EvaluationPair` (reference + deformed + mask + optional GT field),
runs :func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`, and emits a
plain-text report with phase timings, host-RSS deltas, per-device VRAM
deltas, and a status / displacement summary. When the deformation has a
ground-truth flow, an :class:`ErrorReport` is appended.

Run with::

    uv run python scripts/run_e2e_zarr.py \\
        --store /path/scan.zarr \\
        --deformation fs104 \\
        --report report.txt --out displacements.npz

For first runs use ``--dry-run-shape 192,256,256`` to slice a centered
subblock that finishes in well under a minute. Iterate by re-running with
different ``--batch-size``.

The new CLI (``--store / --deformation / --mask / --manifest``) replaces
the legacy ``--reference-path / --deformed-path / --mask-path /
--mask-store`` triplet from before the profile + manifest contract
landed.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import numpy as np
import psutil
import zarr
from mamba_dvc.core.ncc import NCCMode, NCCNormalization
from mamba_dvc.gpu.budget import is_cupy_available, probe_free_vram
from mamba_dvc.gpu.dispatch import correlate_multi_gpu
from mamba_dvc.instrument import accumulating
from mamba_dvc.io.dataset import NO_MASK, DvcDataset, EvaluationPair
from mamba_dvc.io.field import FieldConvention
from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.types import DisplacementField, POIStatus
from mamba_dvc.validate.known_fields import ErrorReport, evaluate_pair

_FLOW_CONVENTIONS: tuple[str, ...] = get_args(FieldConvention)

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:
    _cp = None  # type: ignore[assignment]


# --------------------------------------------------------------------- args


def _parse_devices(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    if not spec.strip():
        raise argparse.ArgumentTypeError("--devices must not be empty")
    return [int(p) for p in spec.split(",")]


def _parse_window(spec: str) -> int | tuple[int, int, int]:
    if "," not in spec:
        return int(spec)
    parts = tuple(int(p) for p in spec.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--window tuple must have 3 entries")
    return parts  # type: ignore[return-value]


def _parse_dry_shape(spec: str | None) -> tuple[int, int, int] | None:
    if spec is None:
        return None
    parts = tuple(int(p) for p in spec.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--dry-run-shape must be z,y,x")
    return parts  # type: ignore[return-value]


def _parse_batch_size(spec: str) -> int | str:
    """Accept a positive integer or the literal string ``"auto"``.

    Mirrors the contract on
    :func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`. ``"auto"`` is
    forwarded as-is and resolved by the budget oracle inside dispatch.
    """
    if spec == "auto":
        return "auto"
    try:
        value = int(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--batch-size must be a positive int or 'auto', got {spec!r}"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"--batch-size must be a positive int or 'auto', got {value}"
        )
    return value


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run mamba-dvc multi-GPU pipeline against a zarr store."
    )
    p.add_argument("--store", required=True, type=Path, help="Path to the zarr store.")
    p.add_argument(
        "--deformation",
        required=True,
        help="Deformation entry name (real iterations/* or synthetic synthetic/*).",
    )
    p.add_argument(
        "--mask",
        default=None,
        help=(
            "Mask name to use. Omit to use the profile / manifest default. "
            "Pass 'none' to skip the mask entirely."
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional StoreManifest YAML; else discovered from sidecar / .attrs.",
    )
    p.add_argument(
        "--flow-convention",
        choices=_FLOW_CONVENTIONS,
        default=None,
        help=(
            "Override the synthetic-flow sign convention. Default: defer to "
            "the manifest / profile (which currently declares 'pull_back', "
            "though typical zarr stores are actually authored as "
            "'push_forward'). Pass 'push_forward' here to flip the sign "
            "interpretation without editing the manifest."
        ),
    )
    p.add_argument(
        "--devices",
        type=_parse_devices,
        default=None,
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. Default: all visible.",
    )
    p.add_argument("--window", type=_parse_window, default=96, help="Subvolume size.")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.9)
    p.add_argument("--tukey-alpha", type=float, default=None)
    p.add_argument("--search-radius", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default=64,
        help=(
            "FFT batch per shard. Positive int or 'auto' (probes free VRAM and "
            "asks the budget oracle for the largest fit). Default 64 keeps "
            "historical numeric reproducibility for one PR cycle; flip to "
            "'auto' once production runs validate the cost model."
        ),
    )
    p.add_argument(
        "--ncc-mode",
        type=NCCMode,
        choices=list(NCCMode),
        default=NCCMode.LINEAR,
    )
    p.add_argument(
        "--ncc-normalization",
        type=NCCNormalization,
        choices=list(NCCNormalization),
        default=NCCNormalization.OVERLAP,
    )
    p.add_argument(
        "--dry-run-shape",
        type=_parse_dry_shape,
        default=None,
        help="z,y,x. Slice a centered subblock from each volume for quick smoke runs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional .npz path: positions, displacements, confidence, status.",
    )
    p.add_argument(
        "--report", type=Path, default=None, help="Optional plain-text report path."
    )
    p.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Accumulate per-phase wall times (mamba_dvc.instrument) and append a "
            "breakdown to the report. Fullest with --devices 0 — multi-GPU worker "
            "subprocesses do not propagate their NCC sub-phase records to the parent."
        ),
    )
    return p


# ----------------------------------------------------------------- timing


@dataclass
class PhaseRecord:
    """Single timed section: wall seconds and host-RSS delta in bytes."""

    name: str
    seconds: float
    rss_delta_bytes: int


@dataclass
class RunMetrics:
    """All timings, memory deltas, and input metadata for one run."""

    phases: list[PhaseRecord] = field(default_factory=list)
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    vram_before: list[tuple[int, int, int]] = field(default_factory=list)
    vram_after: list[tuple[int, int, int]] = field(default_factory=list)


@contextmanager
def phase(name: str, metrics: RunMetrics):
    """Time a block and record its host-RSS delta."""
    proc = psutil.Process()
    rss_before = int(proc.memory_info().rss)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        rss_after = int(proc.memory_info().rss)
        metrics.phases.append(PhaseRecord(name, dt, rss_after - rss_before))


# --------------------------------------------------------------- vram probe


def _vram_snapshot(device_ids: list[int]) -> list[tuple[int, int, int]]:
    """Return ``(device_id, free_bytes, total_bytes)`` per device.

    Delegates to :func:`mamba_dvc.gpu.budget.probe_free_vram` so the
    script and the budget oracle share one CuPy seam (plan §12).
    Returns an empty list when CuPy is missing -- the script will fail
    later in dispatch with a clear error.
    """
    if not is_cupy_available():
        return []
    return [(d, *probe_free_vram(d)) for d in device_ids]


# ---------------------------------------------------------------- reporting


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{sign}{n:6.2f} {unit}"
        n /= 1024
    return f"{sign}{n:.2f} TB"


def _system_info() -> dict[str, str]:
    info = {
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "ram_total": _fmt_bytes(psutil.virtual_memory().total),
    }
    if _cp is not None:
        try:
            info["cupy"] = _cp.__version__
            info["cuda_runtime"] = str(_cp.cuda.runtime.runtimeGetVersion())
            n = int(_cp.cuda.runtime.getDeviceCount())
            names = []
            for i in range(n):
                props = _cp.cuda.runtime.getDeviceProperties(i)
                names.append(
                    props["name"].decode()
                    if isinstance(props["name"], bytes)
                    else props["name"]
                )
            info["gpus"] = f"{n}x ({', '.join(names)})"
        except Exception as exc:  # pragma: no cover - best-effort
            info["cupy"] = f"present, probe failed: {exc!r}"
    else:
        info["cupy"] = "MISSING"
    return info


def _summarize_field(field: DisplacementField) -> dict[str, Any]:
    """Status histogram + percentile stats for valid POIs."""
    status = field.status
    counts = {
        name: int(np.sum(status == int(member)))
        for name, member in POIStatus.__members__.items()
    }
    valid = field.valid
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {"counts": counts, "n_total": int(status.size), "valid_stats": None}
    disp = field.displacements[valid]
    mag = np.linalg.norm(disp, axis=1)
    conf = field.confidence[valid]
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return {
        "counts": counts,
        "n_total": int(status.size),
        "valid_stats": {
            "mag_p50": pct(mag, 50),
            "mag_p90": pct(mag, 90),
            "mag_p99": pct(mag, 99),
            "mag_max": float(mag.max()),
            "conf_p50": pct(conf, 50),
            "conf_p10": pct(conf, 10),
            "conf_min": float(conf.min()),
        },
    }


def _render_report(
    args: argparse.Namespace,
    metrics: RunMetrics,
    summary: dict[str, Any] | None,
    error_report: ErrorReport | None,
    error: str | None,
) -> str:
    sysinfo = _system_info()
    lines: list[str] = []
    now = datetime.now(tz=UTC).astimezone().isoformat(timespec="seconds")
    lines.append(f"=== mamba-dvc e2e run @ {now} ===")
    lines.append(f"invocation: {' '.join(sys.argv)}")
    lines.append("")
    lines.append("== system ==")
    for k, v in sysinfo.items():
        lines.append(f"  {k:14s} {v}")
    lines.append("")
    lines.append("== input ==")
    for name, meta in metrics.inputs.items():
        lines.append(
            f"  {name:11s} shape={meta.get('shape')} dtype={meta.get('dtype')} "
            f"nbytes={_fmt_bytes(meta.get('nbytes', 0))}"
        )
    lines.append("")
    lines.append("== phases ==")
    lines.append(f"  {'name':24s} {'seconds':>10s}  {'dRSS':>11s}")
    for ph in metrics.phases:
        lines.append(
            f"  {ph.name:24s} {ph.seconds:10.3f}  {_fmt_bytes(ph.rss_delta_bytes):>11s}"
        )
    total = sum(p.seconds for p in metrics.phases)
    lines.append(f"  {'TOTAL':24s} {total:10.3f}")
    lines.append("")
    if metrics.vram_before and metrics.vram_after:
        lines.append("== per-device VRAM (free, MB) ==")
        lines.append(f"  {'gpu':>3s} {'before':>10s} {'after':>10s} {'d used':>10s}  total")
        for (d, fb, tot), (_, fa, _) in zip(
            metrics.vram_before, metrics.vram_after, strict=True
        ):
            used_delta = fb - fa
            lines.append(
                f"  {d:3d} {fb / 1e6:10.1f} {fa / 1e6:10.1f} "
                f"{used_delta / 1e6:10.1f}  {_fmt_bytes(tot)}"
            )
        lines.append("")
    if summary is not None:
        lines.append("== POI status ==")
        n_total = summary["n_total"]
        for name, c in summary["counts"].items():
            pct = (100.0 * c / n_total) if n_total else 0.0
            lines.append(f"  {name:14s} {c:8d} ({pct:5.1f}%)")
        stats = summary["valid_stats"]
        if stats is None:
            lines.append("  (no valid POIs — skipping percentile summary)")
        else:
            lines.append("")
            lines.append("== |displacement| (voxels), valid POIs only ==")
            lines.append(
                f"  p50 {stats['mag_p50']:.3f}   p90 {stats['mag_p90']:.3f}   "
                f"p99 {stats['mag_p99']:.3f}   max {stats['mag_max']:.3f}"
            )
            lines.append("")
            lines.append("== confidence, valid POIs only ==")
            lines.append(
                f"  p50 {stats['conf_p50']:.3f}   p10 {stats['conf_p10']:.3f}   "
                f"min {stats['conf_min']:.3f}"
            )
    if error_report is not None:
        lines.append("")
        lines.append(f"== error vs ground truth ({error_report.name}) ==")
        lines.append(
            f"  mae  {error_report.mae:.4f}   rmse {error_report.rmse:.4f}   "
            f"p95 {error_report.p95:.4f}"
        )
        lines.append(
            f"  per-axis MAE: dz={error_report.per_axis_mae[0]:.4f} "
            f"dy={error_report.per_axis_mae[1]:.4f} "
            f"dx={error_report.per_axis_mae[2]:.4f}"
        )
        if error_report.by_distance is not None:
            tbl = error_report.by_distance
            lines.append("  by distance from boundary (voxels):")
            lines.append(f"    edges: {tbl.edges}")
            lines.append(f"    counts: {tbl.counts.tolist()}")
            lines.append(f"    mae:    {tbl.mae.tolist()}")
    if error is not None:
        lines.append("")
        lines.append("== ERROR ==")
        lines.append(error)
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def _resolve_mask_arg(mask: str | None) -> Any:
    """Translate the CLI ``--mask`` value into a load_pair selector."""
    if mask is None:
        return None
    if mask.lower() == "none":
        return NO_MASK
    return mask


def _apply_flow_convention_override(
    manifest: StoreManifest | None, convention: str | None
) -> StoreManifest | None:
    """Patch ``manifest.synthetic.flow.convention`` from the CLI override.

    Returns ``manifest`` unchanged when no override is requested. When
    an override is requested but no manifest was loaded, materializes a
    minimal :class:`StoreManifest` carrying just the flow convention so
    the dataset's resolution chain (manifest > profile) sees it.
    """
    if convention is None:
        return manifest
    base = manifest if manifest is not None else StoreManifest()
    new_flow = replace(base.synthetic.flow, convention=convention)  # type: ignore[arg-type]
    new_synthetic = replace(base.synthetic, flow=new_flow)
    return replace(base, synthetic=new_synthetic)


def _run_pipeline(
    args: argparse.Namespace, metrics: RunMetrics
) -> tuple[dict[str, Any] | None, ErrorReport | None, str | None]:
    """Run open -> load -> correlate -> evaluate -> serialize, recording ``metrics``.

    Returns ``(summary, error_report, error)``; ``error`` is the formatted
    traceback string when any phase raised, else ``None``. Kept separate
    from :func:`main` so the optional timing accumulator can wrap the
    whole pipeline without re-indenting it.
    """
    summary: dict[str, Any] | None = None
    error: str | None = None
    field_result: DisplacementField | None = None
    err_report: ErrorReport | None = None

    try:
        manifest: StoreManifest | None = None
        if args.manifest is not None:
            manifest = StoreManifest.from_yaml(args.manifest)
        manifest = _apply_flow_convention_override(manifest, args.flow_convention)

        with phase("open_dataset", metrics):
            ds = DvcDataset.open(args.store, manifest=manifest)

        with phase("load_pair", metrics):
            pair: EvaluationPair = ds.load_pair(
                args.deformation,
                mask=_resolve_mask_arg(args.mask),
                dry_shape=args.dry_run_shape,
            )

        for label, arr in (
            ("reference", pair.reference),
            ("deformed", pair.deformed),
        ):
            metrics.inputs[label] = {
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": arr.nbytes,
            }
        if pair.mask is not None:
            metrics.inputs["mask"] = {
                "shape": tuple(pair.mask.shape),
                "dtype": str(pair.mask.dtype),
                "nbytes": pair.mask.nbytes,
            }

        device_ids_for_probe: list[int] = []
        if _cp is not None:
            device_ids_for_probe = (
                args.devices
                if args.devices is not None
                else list(range(int(_cp.cuda.runtime.getDeviceCount())))
            )
        metrics.vram_before = _vram_snapshot(device_ids_for_probe)

        with phase("correlate_multi_gpu", metrics):
            field_result = correlate_multi_gpu(
                pair.reference,
                pair.deformed,
                mask=pair.mask,
                device_ids=args.devices,
                window=args.window,
                overlap=args.overlap,
                mask_threshold=args.mask_threshold,
                tukey_alpha=args.tukey_alpha,
                search_radius=args.search_radius,
                batch_size=args.batch_size,
                ncc_mode=args.ncc_mode,
                ncc_normalization=args.ncc_normalization,
            )

        metrics.vram_after = _vram_snapshot(device_ids_for_probe)
        summary = _summarize_field(field_result)

        if pair.gt_field is not None:
            with phase("evaluate_pair", metrics):
                err_report = evaluate_pair(pair, field_result, distance_bins=())

        if args.out is not None:
            with phase("serialize", metrics):
                np.savez(
                    args.out,
                    positions=field_result.positions,
                    displacements=field_result.displacements,
                    valid=field_result.valid,
                    confidence=field_result.confidence,
                    status=field_result.status,
                    grid_shape=np.array(field_result.grid_shape),
                    spacing=np.array(field_result.spacing),
                    window=np.array(field_result.window),
                )
    except Exception:
        error = traceback.format_exc()

    return summary, err_report, error


def main(argv: list[str] | None = None) -> int:
    """Run the harness and return a process exit code (0 = ok, 1 = exception)."""
    args = _build_argparser().parse_args(argv)
    metrics = RunMetrics()

    timing_cm = accumulating() if args.timing else nullcontext(None)
    with timing_cm as acc:
        summary, err_report, error = _run_pipeline(args, metrics)

    report = _render_report(args, metrics, summary, err_report, error)
    if acc is not None:
        report += "\n" + acc.render(title="== phase breakdown — wall seconds ==") + "\n"
    sys.stdout.write(report)
    if args.report is not None:
        args.report.write_text(report, encoding="utf-8")

    return 1 if error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
