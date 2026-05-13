r"""Spot-check the KDTree boundary-distance change against a real campaign.

Re-runs ``evaluate_pair`` -- with the new ``BoundaryDistanceIndex`` -- on
a displacement field that an earlier ``mamba-dvc run`` already produced
(back when ``known_fields.py`` used the full-volume
``distance_transform_edt``), and diffs the resulting ``by_distance``
table against that run's on-disk sidecar. The field and the mask are
taken straight from the campaign output, so GPU-FFT non-determinism is
out of the picture: the only thing that varies between the two runs is
the boundary-distance computation, which this confirms is numerically
unchanged -- and hundreds of times cheaper. It also times the new
once-per-pair ``BoundaryDistanceIndex.from_mask`` (still one full-volume
``binary_dilation``) so a new bottleneck would not slip through.

This is the manual gate referenced in ``docs/plans/kdtree-impl-plan.md``:
run it on RECO-4xA6000 before launching the next campaign.

Examples
--------
::

    uv run python scripts/spot_check_kdtree.py
    uv run python scripts/spot_check_kdtree.py \
        --campaign-dir D:\runs\results_v1 --store 5L_PEEK_... --deformation fs204
    uv run python scripts/spot_check_kdtree.py --variant mt0.9-<hash>
    uv run python scripts/spot_check_kdtree.py --dry-shape 192,256,256   # fast subblock run

Exit code is 0 on a clean check, 1 on a numerical mismatch or a setup
problem (missing campaign dir, store, sidecar, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np

# These ``run.batch`` helpers are reused on purpose: they reproduce the
# campaign's exact dataset wiring (manifest patching, ``strict`` flag,
# ``flow_convention`` override, mask selector). Re-implementing them
# risks the very mismatch the perf doc warns about -- a wrong
# ``flow_convention`` silently inflates synthetic error ~50x.
from mamba_dvc.run.batch import _make_default_opener, _mask_selector
from mamba_dvc.run.config import BatchSpec
from mamba_dvc.types import DisplacementField
from mamba_dvc.validate.known_fields import BoundaryDistanceIndex, evaluate_pair

# Directly-runnable defaults: the single store / first scored deformation
# of the `results_v1` config template, under its default `out_dir`.
DEFAULT_CAMPAIGN_DIR = Path(r"C:\Users\jstebani\storage\mamba-test-runs\results_v1")
DEFAULT_STORE = "103L_Mg5Gd_4w_000"
DEFAULT_DEFORMATION = "fs104"
PREFERRED_MASK_THRESHOLD = 0.9  # which variant to auto-pick when --variant is omitted

# Above this the per-pair index build is itself a problem (R3's bbox crop).
SLOW_BUILD_SECONDS = 180.0


def _die(msg: str) -> NoReturn:
    """Print ``msg`` to stderr and exit non-zero."""
    print(f"spot_check_kdtree: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=DEFAULT_CAMPAIGN_DIR,
        help="Campaign output dir (contains config.snapshot.yaml). Default: %(default)s",
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help="Store stem or .zarr name to spot-check. Default: %(default)s",
    )
    parser.add_argument(
        "--deformation",
        default=DEFAULT_DEFORMATION,
        help="Synthetic deformation name (must carry GT). Default: %(default)s",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Variant id to check; default auto-picks the "
            f"mask_threshold={PREFERRED_MASK_THRESHOLD:g} variant, else the first on disk."
        ),
    )
    parser.add_argument(
        "--dry-shape",
        default=None,
        help="Override dry_shape with 'z,y,x' (a centered subblock) for a fast run.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for the by_distance mae/rmse check. Default: %(default)s",
    )
    return parser.parse_args(argv)


def _store_stem(name: str) -> str:
    """Strip a trailing ``.zarr`` (mirrors ``run.batch._store_stem``)."""
    return name[:-5] if name.endswith(".zarr") else name


def _resolve_store_path(spec: BatchSpec, query: str) -> Path:
    """Return the campaign store whose path or ``.zarr`` name matches ``query``."""
    wanted = _store_stem(query)
    for store in spec.stores:
        if _store_stem(store.name) == wanted or store.name == query or str(store) == query:
            return store
    _die(f"--store {query!r} matches no store in {[s.name for s in spec.stores]}")


def _pick_variant(deform_dir: Path, requested: str | None) -> tuple[str, dict]:
    """Choose the variant to check; return ``(variant_id, resolved_params)``.

    With ``--variant`` it is that exact id. Otherwise prefer the
    ``mask_threshold == PREFERRED_MASK_THRESHOLD`` variant, falling back
    to the first sidecar in the directory.
    """
    if requested is not None:
        sidecar = deform_dir / f"{requested}.json"
        if not sidecar.exists():
            _die(f"variant {requested!r}: {sidecar} not found")
        return requested, json.loads(sidecar.read_text(encoding="utf-8")).get("params", {})

    sidecars = sorted(deform_dir.glob("*.json"))
    if not sidecars:
        _die(f"no sidecars under {deform_dir} -- has the campaign run there?")
    parsed = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sidecars]
    for path, data in parsed:
        if data.get("params", {}).get("mask_threshold") == PREFERRED_MASK_THRESHOLD:
            return data.get("variant_id", path.stem), data.get("params", {})
    path, data = parsed[0]
    return data.get("variant_id", path.stem), data.get("params", {})


def _coerce_dry_shape(value: object) -> tuple[int, int, int] | None:
    """Normalize a dry_shape (CLI ``'z,y,x'`` string, JSON list, or ``None``) to a 3-tuple."""
    if value is None or value == "":
        return None
    parts = (
        [int(x) for x in value.split(",")]
        if isinstance(value, str)
        else [int(x) for x in value]
    )
    if len(parts) != 3:
        _die(f"dry_shape must have 3 entries, got {parts}")
    return (parts[0], parts[1], parts[2])


def _fmt(value: object) -> str:
    """Compact human-readable repr for scalars and small arrays in the report."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(v) for v in np.asarray(value).tolist()) + "]"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def _scalar_eq(a: object, b: object) -> bool:
    """Compare two JSON-roundtripped scalars; ``nan == nan``, otherwise ``math.isclose``."""
    fa, fb = float(a), float(b)  # type: ignore[arg-type]
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)


def _array_max_abs_diff(new: np.ndarray, old: np.ndarray) -> float:
    """Largest ``|new - old|`` over the non-NaN slots; ``0.0`` if all NaN or empty."""
    diff = np.abs(np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64))
    finite = diff[~np.isnan(diff)]
    return float(finite.max()) if finite.size else 0.0


def main(argv: list[str] | None = None) -> int:
    """Run the spot check; return a process exit code (0 = clean, 1 = problem)."""
    args = _parse_args(argv)
    campaign_dir: Path = args.campaign_dir

    snapshot = campaign_dir / "config.snapshot.yaml"
    if not snapshot.exists():
        _die(f"{snapshot} not found -- point --campaign-dir at a finished campaign output dir")
    spec = BatchSpec.from_yaml(snapshot)
    if not spec.distance_bins:
        print(
            f"campaign {campaign_dir} ran with distance_bins: [] -- no by_distance table was "
            "produced, so the KDTree change has nothing to affect here. Nothing to check."
        )
        return 0

    store = _resolve_store_path(spec, args.store)
    store_stem = _store_stem(store.name)
    deform_dir = campaign_dir / store_stem / args.deformation
    if not deform_dir.is_dir():
        _die(
            f"{deform_dir} not found -- did the campaign run "
            f"{args.deformation!r} for store {store_stem!r}?"
        )

    variant_id, params = _pick_variant(deform_dir, args.variant)
    sidecar_path = deform_dir / f"{variant_id}.json"
    npz_path = deform_dir / f"{variant_id}.npz"
    if not npz_path.exists():
        _die(f"{npz_path} not found")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    old_summary = sidecar.get("summary", {})
    old_bd = old_summary.get("by_distance")
    if old_bd is None:
        _die(
            f"sidecar {sidecar_path} has no summary.by_distance -- pick a scored variant "
            "(synthetic deformation, non-empty distance_bins)"
        )

    field = DisplacementField.load_npz(npz_path)
    flow_convention = params.get("flow_convention")
    mask_selector = _mask_selector(params.get("mask"))
    dry_shape = _coerce_dry_shape(
        args.dry_shape if args.dry_shape is not None else params.get("dry_shape")
    )

    print("=" * 78)
    print("KDTree boundary-distance spot check")
    print("-" * 78)
    print(f"campaign dir   {campaign_dir}")
    print(f"store          {store_stem}   ({store})")
    print(f"deformation    {args.deformation}")
    print(
        f"variant        {variant_id}   "
        f"mask_threshold={params.get('mask_threshold')}  flow_convention={flow_convention}"
    )
    print(f"distance_bins  {list(spec.distance_bins)}")
    if dry_shape is not None:
        print(f"dry_shape      {dry_shape}  (centered subblock -- timing not representative)")
    print("-" * 78)

    try:
        dataset = _make_default_opener(spec)(store, flow_convention)
        t0 = time.perf_counter()
        pair = dataset.load_pair(args.deformation, mask=mask_selector, dry_shape=dry_shape)
        t_load = time.perf_counter() - t0
    except Exception as exc:  # manual harness -- surface whatever broke
        _die(f"could not materialize the pair: {exc!r}")

    if pair.gt_field is None:
        _die(f"{args.deformation!r} has no ground truth -- evaluate_pair cannot score it")
    if pair.mask is None:
        _die(
            "pair.mask is None -- but the campaign produced a by_distance table, so it had one"
        )

    background = int(np.count_nonzero(~pair.mask))
    pct_bg = 0.0 if pair.mask.size == 0 else 100.0 * background / pair.mask.size
    print(
        f"load_pair      {t_load:6.1f}s   volume={tuple(pair.mask.shape)}  "
        f"background voxels={background:,} ({pct_bg:.1f}% of volume)"
    )

    t0 = time.perf_counter()
    index = BoundaryDistanceIndex.from_mask(pair.mask)
    t_build = time.perf_counter() - t0
    shell_voxels = 0 if index.tree is None else int(index.tree.n)
    empty_note = "  (empty fg shell -> all distances 0 or inf)" if index.tree is None else ""
    print(f"from_mask      {t_build:6.1f}s   fg-shell voxels={shell_voxels:,}{empty_note}")

    t0 = time.perf_counter()
    report = evaluate_pair(
        pair, field, distance_bins=tuple(spec.distance_bins), boundary_index=index
    )
    t_eval = time.perf_counter() - t0
    new_bd = report.by_distance
    assert new_bd is not None  # distance_bins non-empty and pair.mask present
    print(
        f"evaluate_pair  {t_eval:6.1f}s   "
        f"(campaign sidecar wall_evaluate_s={_fmt(old_summary.get('wall_evaluate_s'))})"
    )
    print("-" * 78)

    ok = True

    def check(label: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")

    # --- the boundary table: the only thing this change touches ---
    old_edges = tuple(old_bd["edges"])
    check(
        "by_distance.edges",
        tuple(new_bd.edges) == old_edges,
        f"new={_fmt(new_bd.edges)}  old={_fmt(old_edges)}",
    )

    old_counts = np.asarray(old_bd["counts"], dtype=np.float32)
    check(
        "by_distance.counts (bin membership)",
        np.array_equal(new_bd.counts, old_counts),
        f"new={_fmt(new_bd.counts)}  old={_fmt(old_counts)}",
    )

    for name, new_arr, old_list in (
        ("mae", new_bd.mae, old_bd["mae"]),
        ("rmse", new_bd.rmse, old_bd["rmse"]),
    ):
        old_arr = np.asarray(old_list, dtype=np.float32)
        passed = np.allclose(new_arr, old_arr, rtol=args.rtol, atol=1e-6, equal_nan=True)
        check(
            f"by_distance.{name}",
            passed,
            f"max|delta|={_array_max_abs_diff(new_arr, old_arr):.3g}  "
            f"new={_fmt(new_arr)}  old={_fmt(old_arr)}",
        )

    # --- aggregates the change must NOT touch (no mask involved, same field) ---
    for name, new_val, old_val in (
        ("mae", report.mae, old_summary.get("mae")),
        ("rmse", report.rmse, old_summary.get("rmse")),
        ("p95", report.p95, old_summary.get("p95")),
        (
            "confidence_vs_error",
            report.confidence_vs_error,
            old_summary.get("confidence_vs_error"),
        ),
    ):
        passed = old_val is not None and _scalar_eq(new_val, old_val)
        check(f"summary.{name} unchanged", passed, f"new={_fmt(new_val)}  old={_fmt(old_val)}")
    old_pa = old_summary.get("per_axis_mae")
    passed = (
        isinstance(old_pa, list)
        and len(old_pa) == 3
        and all(_scalar_eq(a, b) for a, b in zip(report.per_axis_mae, old_pa, strict=True))
    )
    check(
        "summary.per_axis_mae unchanged",
        passed,
        f"new={_fmt(report.per_axis_mae)}  old={_fmt(old_pa)}",
    )

    # --- perf observations (informational; only flagged when egregious) ---
    print("-" * 78)
    old_eval = old_summary.get("wall_evaluate_s")
    if isinstance(old_eval, (int, float)) and old_eval > 0:
        speedup = old_eval / max(t_eval, 1e-6)
        collapsed = t_eval < max(30.0, old_eval / 2.0)
        tail = "" if collapsed else "   <-- did NOT collapse, investigate"
        print(
            f"  [INFO] evaluate_pair {old_eval:.1f}s -> {t_eval:.2f}s  ({speedup:.0f}x){tail}"
        )
    if t_build > SLOW_BUILD_SECONDS:
        print(
            f"  [WARN] from_mask took {t_build:.0f}s -- the full-volume binary_dilation "
            "dominates; see the R3 bbox crop in docs/plans/general-perf-improvements.md"
        )

    print("-" * 78)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
