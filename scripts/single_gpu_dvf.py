"""Single-GPU happy path: compute one displacement field + GT error metrics.

Bypasses the multi-process dispatcher (and therefore the
``cudaErrorAlreadyMapped`` failure observed on the 4-GPU campaign path).
Routes through ``correlate_multi_gpu(..., device_ids=[device])`` which
falls into ``_run_single_process`` -- in-process, no SHM, no spawn.

Usage
-----
::

    uv run python scripts/single_gpu_dvf.py \
        --store C:/.../dvc-data-storage/103L_Mg5Gd_4w_000.zarr \
        --deformation fs004 \
        --out-dir C:/.../out/103L_fs004_mt09 \
        --window 96 --overlap 0.8 \
        --mask mask_fill --mask-threshold 0.9 \
        --flow-convention push_forward

Outputs (under ``--out-dir``):
- ``displacement_field.npz`` -- positions, displacements, valid, confidence,
  status, grid metadata, GT samples at admitted positions, residuals.
- ``error_metrics.json`` -- per-axis and magnitude MAE/RMSE/percentiles.
- ``params.yaml`` -- resolved knobs.
- ``events.jsonl`` -- structured phase timings (via SessionScope).
- ``run.log`` -- human-readable log (INFO+).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mamba_dvc.core.ncc import NCCMode, NCCNormalization
from mamba_dvc.gpu.dispatch import correlate_multi_gpu
from mamba_dvc.instrument import timed
from mamba_dvc.io.dataset import NO_MASK, DvcDataset
from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.run.batch import _apply_flow_convention
from mamba_dvc.run.eventlog import SessionScope

_LOG = logging.getLogger("mamba_dvc.scripts.single_gpu_dvf")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", type=Path, required=True, help="path to .zarr store")
    p.add_argument("--deformation", required=True, help="deformation entry name (e.g. fs004)")
    p.add_argument("--out-dir", type=Path, required=True, help="output directory (created)")
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--overlap", type=float, default=0.8)
    p.add_argument(
        "--mask",
        default="mask_fill",
        help="mask name; 'none' skips the mask (uses NO_MASK sentinel)",
    )
    p.add_argument("--mask-threshold", type=float, default=0.9)
    p.add_argument(
        "--flow-convention",
        choices=["push_forward", "pull_back"],
        default="push_forward",
        help="override the store's manifest convention for synthetic GT",
    )
    p.add_argument("--ncc-mode", choices=["linear", "cyclic"], default="linear")
    p.add_argument("--ncc-normalization", choices=["overlap", "global"], default="overlap")
    p.add_argument(
        "--search-radius",
        type=int,
        default=None,
        help="None -> window // 2 (CuPy default)",
    )
    p.add_argument(
        "--tukey-alpha",
        type=float,
        default=None,
        help="None -> per-ncc-mode default",
    )
    p.add_argument(
        "--batch-size",
        default="auto",
        help="'auto' (probe VRAM) or a positive int",
    )
    p.add_argument(
        "--gt-interpolation",
        type=int,
        default=1,
        help="spline order for GroundTruthField sampling (0..5)",
    )
    p.add_argument("--device", type=int, default=0, help="single GPU id")
    p.add_argument(
        "--dry-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="optional centered subblock size for smoke runs",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="strict store verification (default: lenient)",
    )
    return p.parse_args()


def _setup_logging(out_dir: Path) -> logging.FileHandler:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Also mirror to stderr.
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    stderr.setLevel(logging.INFO)
    root.addHandler(stderr)
    return handler


def _resolve_batch_size(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--batch-size must be 'auto' or a positive int, got {value!r}"
        ) from exc
    if n <= 0:
        raise argparse.ArgumentTypeError(f"--batch-size must be > 0, got {n}")
    return n


def _open_dataset(store: Path, *, flow_convention: str, strict: bool) -> DvcDataset:
    base = StoreManifest.discover(store)
    patched = _apply_flow_convention(base, flow_convention)
    return DvcDataset.open(store, manifest=patched, strict=strict)


def _summarise_status(status: np.ndarray) -> dict[str, int]:
    from mamba_dvc.types import POIStatus  # local: avoid hard dep at import

    out: dict[str, int] = {}
    for member in POIStatus:
        n = int(np.count_nonzero(status == member.value))
        if n > 0:
            out[member.name] = n
    return out


def _error_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Per-axis and magnitude error stats over `pred - gt`."""
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.size == 0:
        return {"n": 0}
    resid = pred - gt
    out: dict[str, Any] = {"n": int(pred.shape[0])}
    for axis_idx, axis_name in enumerate(("dz", "dy", "dx")):
        r = resid[:, axis_idx]
        out[axis_name] = {
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(np.mean(r * r))),
            "p50": float(np.percentile(np.abs(r), 50)),
            "p95": float(np.percentile(np.abs(r), 95)),
            "p99": float(np.percentile(np.abs(r), 99)),
            "bias": float(np.mean(r)),
        }
    mag_pred = np.linalg.norm(pred, axis=1)
    mag_gt = np.linalg.norm(gt, axis=1)
    mag_resid = np.linalg.norm(resid, axis=1)
    out["magnitude"] = {
        "mae": float(np.mean(mag_resid)),
        "rmse": float(np.sqrt(np.mean(mag_resid * mag_resid))),
        "p50": float(np.percentile(mag_resid, 50)),
        "p95": float(np.percentile(mag_resid, 95)),
        "p99": float(np.percentile(mag_resid, 99)),
    }
    out["gt_magnitude"] = {
        "mean": float(np.mean(mag_gt)),
        "p50": float(np.percentile(mag_gt, 50)),
        "p95": float(np.percentile(mag_gt, 95)),
        "max": float(np.max(mag_gt)),
    }
    out["pred_magnitude"] = {
        "mean": float(np.mean(mag_pred)),
        "p50": float(np.percentile(mag_pred, 50)),
        "p95": float(np.percentile(mag_pred, 95)),
        "max": float(np.max(mag_pred)),
    }
    return out


def main() -> int:
    """Run one single-GPU correlation pass and write artifacts to ``--out-dir``."""
    args = _parse_args()
    batch_size = _resolve_batch_size(args.batch_size)
    out_dir: Path = args.out_dir
    _setup_logging(out_dir)
    _LOG.info(
        "single_gpu_dvf starting: store=%s deformation=%s out=%s",
        args.store,
        args.deformation,
        out_dir,
    )

    params: dict[str, Any] = {
        "store": str(args.store),
        "deformation": args.deformation,
        "window": args.window,
        "overlap": args.overlap,
        "mask": args.mask,
        "mask_threshold": args.mask_threshold,
        "flow_convention": args.flow_convention,
        "ncc_mode": args.ncc_mode,
        "ncc_normalization": args.ncc_normalization,
        "search_radius": args.search_radius,
        "tukey_alpha": args.tukey_alpha,
        "batch_size": batch_size,
        "gt_interpolation": args.gt_interpolation,
        "device": args.device,
        "dry_shape": list(args.dry_shape) if args.dry_shape else None,
        "strict": args.strict,
    }
    (out_dir / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))

    with SessionScope(
        out_dir,
        campaign="single_gpu_dvf",
        store=args.store.name,
        deformation=args.deformation,
    ):
        with timed("script.open_dataset"):
            dataset = _open_dataset(
                args.store,
                flow_convention=args.flow_convention,
                strict=args.strict,
            )
        _LOG.info("dataset volume_shape=%s", dataset.volume_shape)

        mask_selector: Any
        if isinstance(args.mask, str) and args.mask.lower() == "none":
            mask_selector = NO_MASK
        else:
            mask_selector = args.mask

        with timed(
            "script.load_pair",
            store=args.store.name,
            deformation=args.deformation,
        ):
            pair = dataset.load_pair(
                args.deformation,
                mask=mask_selector,
                dry_shape=tuple(args.dry_shape) if args.dry_shape else None,
                gt_interpolation=args.gt_interpolation,
            )
        _LOG.info(
            "loaded pair: ref %s def %s mask %s gt %s",
            pair.reference.shape,
            pair.deformed.shape,
            pair.mask.shape if pair.mask is not None else None,
            "yes" if pair.gt_field is not None else "no",
        )

        with timed("script.correlate", device=args.device):
            field = correlate_multi_gpu(
                pair.reference,
                pair.deformed,
                mask=pair.mask,
                deformed_mask=pair.mask,
                device_ids=[args.device],
                window=args.window,
                overlap=args.overlap,
                mask_threshold=args.mask_threshold,
                tukey_alpha=args.tukey_alpha,
                search_radius=args.search_radius,
                batch_size=batch_size,
                ncc_mode=NCCMode(args.ncc_mode),
                ncc_normalization=NCCNormalization(args.ncc_normalization),
            )

        valid_mask = np.asarray(field.valid, dtype=bool)
        n_points = int(field.displacements.shape[0])
        n_valid = int(valid_mask.sum())
        _LOG.info(
            "correlation done: %d/%d POIs valid (%.1f%%)",
            n_valid,
            n_points,
            100.0 * n_valid / max(n_points, 1),
        )

        # Score against GT if available. Sample at every POI position; rows
        # without a valid result get NaN in the saved residuals so downstream
        # consumers can mask uniformly. Metrics computed over valid rows only.
        gt_at_positions = np.full_like(field.displacements, np.nan, dtype=np.float32)
        residuals = np.full_like(field.displacements, np.nan, dtype=np.float32)
        metrics: dict[str, Any] = {
            "n_points": n_points,
            "n_valid": n_valid,
            "valid_fraction": n_valid / max(n_points, 1),
            "status_counts": _summarise_status(np.asarray(field.status)),
            "has_gt": pair.gt_field is not None,
        }
        if pair.gt_field is not None and n_valid > 0:
            gt_field: GroundTruthField = pair.gt_field
            with timed("script.gt_sample", n_valid=n_valid):
                valid_positions = np.asarray(field.positions[valid_mask], dtype=np.float32)
                gt_valid = gt_field(valid_positions)
            gt_at_positions[valid_mask] = gt_valid
            residuals[valid_mask] = np.asarray(field.displacements[valid_mask]) - gt_valid
            metrics["error"] = _error_stats(
                np.asarray(field.displacements[valid_mask]),
                gt_valid,
            )

        npz_path = out_dir / "displacement_field.npz"
        np.savez_compressed(
            npz_path,
            positions=np.asarray(field.positions),
            displacements=np.asarray(field.displacements),
            valid=valid_mask,
            confidence=np.asarray(field.confidence),
            status=np.asarray(field.status),
            grid_shape=np.asarray(field.grid_shape, dtype=np.int64),
            spacing=np.asarray(field.spacing, dtype=np.int64),
            window=np.asarray(field.window, dtype=np.int64),
            volume_shape=np.asarray(dataset.volume_shape, dtype=np.int64),
            gt_at_positions=gt_at_positions,
            residuals=residuals,
        )
        _LOG.info("wrote %s (%.1f MB)", npz_path, npz_path.stat().st_size / 1e6)

        metrics_path = out_dir / "error_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        _LOG.info("wrote %s", metrics_path)

        if "error" in metrics:
            err = metrics["error"]["magnitude"]
            _LOG.info(
                "magnitude error: MAE=%.4f RMSE=%.4f p95=%.4f vox  (|GT| mean=%.4f)",
                err["mae"],
                err["rmse"],
                err["p95"],
                metrics["error"]["gt_magnitude"]["mean"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
