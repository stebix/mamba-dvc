r"""End-to-end driver for the v1 multi-GPU DVC pipeline against a zarr store.

Quick-and-dirty manual perf harness, not part of the test suite. Loads
reference, deformed, and mask arrays from one (or more) zarr stores,
materializes them to host NumPy buffers, runs
:func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`, and emits a plain-text
report with phase timings, host-RSS deltas, per-device VRAM deltas, and a
status / displacement summary.

Run with::

    uv run python scripts/run_e2e_zarr.py \\
        --store /path/scan.zarr \\
        --reference-path /t0/raw \\
        --deformed-path /t1/raw \\
        --mask-path /screw/mask \\
        --report report.txt --out displacements.npz

For first runs use ``--dry-run-shape 192,256,256`` to slice a centered
subblock that finishes in well under a minute. Iterate by re-running with
different ``--batch-size``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import zarr
from mamba_dvc.gpu.dispatch import correlate_multi_gpu
from mamba_dvc.types import DisplacementField, POIStatus

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


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run mamba-dvc multi-GPU pipeline against a zarr store."
    )
    p.add_argument("--store", required=True, type=Path, help="Path to the zarr store.")
    p.add_argument("--reference-path", required=True, help="Internal zarr path for reference.")
    p.add_argument("--deformed-path", required=True, help="Internal zarr path for deformed.")
    p.add_argument(
        "--mask-store",
        type=Path,
        default=None,
        help="Optional separate zarr store for the mask. Defaults to --store.",
    )
    p.add_argument(
        "--mask-path",
        default=None,
        help="Internal zarr path for the mask. Omit to run with no mask (all-True).",
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ncc-mode", choices=("linear", "cyclic"), default="linear")
    p.add_argument("--ncc-normalization", choices=("overlap", "global"), default="overlap")
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


# --------------------------------------------------------------------- io


def _open_zarr_array(store_path: Path, internal_path: str) -> Any:
    """Open a zarr array via store path + internal path.

    Works for both v3 and v2 stores via the high-level ``zarr.open`` entry
    point; if the user points at a v3 group, we resolve the internal path.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"zarr store not found: {store_path}")
    root = zarr.open(str(store_path), mode="r")
    arr = root[internal_path]
    return arr


def _center_slice(
    shape: tuple[int, ...], dry: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """Centered (z,y,x) slice of size ``dry`` inside ``shape``."""
    out: list[slice] = []
    for full, want in zip(shape, dry, strict=True):
        if want > full:
            raise ValueError(f"--dry-run-shape entry {want} exceeds source shape {full}")
        lo = (full - want) // 2
        out.append(slice(lo, lo + want))
    return tuple(out)  # type: ignore[return-value]


def load_volume(
    store_path: Path,
    internal_path: str,
    *,
    dry_shape: tuple[int, int, int] | None,
    as_float32: bool,
) -> np.ndarray:
    """Materialize a 3D zarr array into a contiguous host buffer."""
    arr = _open_zarr_array(store_path, internal_path)
    if arr.ndim != 3:
        raise ValueError(f"{store_path}:{internal_path} has ndim={arr.ndim}, expected 3")
    if dry_shape is not None:
        sl = _center_slice(arr.shape, dry_shape)
        data = np.asarray(arr[sl])
    else:
        data = np.asarray(arr[:])
    if as_float32:
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
    else:
        if data.dtype != np.bool_:
            data = data.astype(np.bool_, copy=False)
    return np.ascontiguousarray(data)


# --------------------------------------------------------------- vram probe


def _vram_snapshot(device_ids: list[int]) -> list[tuple[int, int, int]]:
    """Return ``(device_id, free_bytes, total_bytes)`` per device.

    No-op (empty list) when CuPy is missing -- the script will fail later
    in dispatch with a clear error.
    """
    if _cp is None:
        return []
    snap: list[tuple[int, int, int]] = []
    for d in device_ids:
        with _cp.cuda.Device(d):
            free, total = _cp.cuda.runtime.memGetInfo()
            snap.append((d, int(free), int(total)))
    return snap


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
    if error is not None:
        lines.append("")
        lines.append("== ERROR ==")
        lines.append(error)
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    """Run the harness and return a process exit code (0 = ok, 1 = exception)."""
    args = _build_argparser().parse_args(argv)
    metrics = RunMetrics()
    summary: dict[str, Any] | None = None
    error: str | None = None
    field_result: DisplacementField | None = None

    mask_store = args.mask_store if args.mask_store is not None else args.store

    try:
        with phase("load_reference", metrics):
            reference = load_volume(
                args.store, args.reference_path, dry_shape=args.dry_run_shape, as_float32=True
            )
        metrics.inputs["reference"] = {
            "shape": tuple(reference.shape),
            "dtype": str(reference.dtype),
            "nbytes": reference.nbytes,
        }

        with phase("load_deformed", metrics):
            deformed = load_volume(
                args.store, args.deformed_path, dry_shape=args.dry_run_shape, as_float32=True
            )
        metrics.inputs["deformed"] = {
            "shape": tuple(deformed.shape),
            "dtype": str(deformed.dtype),
            "nbytes": deformed.nbytes,
        }

        mask: np.ndarray | None = None
        if args.mask_path is not None:
            with phase("load_mask", metrics):
                mask = load_volume(
                    mask_store, args.mask_path, dry_shape=args.dry_run_shape, as_float32=False
                )
            metrics.inputs["mask"] = {
                "shape": tuple(mask.shape),
                "dtype": str(mask.dtype),
                "nbytes": mask.nbytes,
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
                reference,
                deformed,
                mask=mask,
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

    report = _render_report(args, metrics, summary, error)
    sys.stdout.write(report)
    if args.report is not None:
        args.report.write_text(report, encoding="utf-8")

    return 1 if error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
