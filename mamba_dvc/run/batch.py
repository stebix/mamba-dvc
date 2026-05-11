"""Execution loop for batch displacement-field campaigns.

Consumes a :class:`~mamba_dvc.run.config.BatchSpec`, iterates the
``(store x deformation x variant)`` product with a *materialize-once /
iterate-many* loop (open the dataset and ``load_pair`` once per
load-tier slice; loop the cheaper compute-tier variants inside), runs
``correlate_multi_gpu`` + ``evaluate_pair``, and persists each result as
a ``.npz`` field plus a JSON sidecar. A ``manifest.jsonl`` aggregates
the run; ``variants.json`` decodes the hash-suffixed variant ids.

See ``docs/plans/run-interface.md`` §2 (invariants) and §6 (flow).

Two deliberate deviations from the plan's §5 sketch:

- ``run_batch`` does not take a ``dry_run`` flag and never prints; the
  preview is :func:`plan_jobs` plus the CLI's own rendering. Keeping
  the library side print-free is the reason.
- The pinned-memory OOM retry is a single fallback to a 2-GPU subset
  (only when ``devices`` was given explicitly), not a multi-step ladder.
  The real fix — non-pinned host staging in ``gpu.dispatch`` — is
  tracked separately (plan §9 deferrals); a bare same-args retry is
  omitted because a deterministic OOM just fails twice as slowly.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import platform
import subprocess
import time
import traceback
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from mamba_dvc.gpu.dispatch import correlate_multi_gpu
from mamba_dvc.io.dataset import NO_MASK, DvcDataset, EvaluationPair
from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.run.config import BatchSpec, Variant, knob_names, slug_value
from mamba_dvc.types import DisplacementField, POIStatus
from mamba_dvc.validate.known_fields import ErrorReport, evaluate_pair

__all__ = ["DatasetOpener", "Job", "JobResult", "plan_jobs", "run_batch"]


# An opener maps ``(store_path, flow_convention)`` to an open dataset.
# The default (:func:`_make_default_opener`) wraps ``DvcDataset.open``
# with the campaign's shared manifest (if any) and the convention
# override; tests substitute a fake that returns a duck-typed dataset.
DatasetOpener = Callable[[Path, str | None], DvcDataset]

# A correlate function has ``correlate_multi_gpu``'s call shape:
# ``(reference, deformed, mask=..., device_ids=..., **grid, **compute)``.
_CorrelateFn = Callable[..., DisplacementField]

# Sentinel deformation name used for a store-level failure (the store
# could not be opened, so no per-deformation jobs could be enumerated).
_STORE_LEVEL = "*"


# ----------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class Job:
    """One unit of work: a ``(store, deformation, variant)`` triple."""

    store: Path
    deformation: str
    kind: str  # "real" | "synthetic" | "?" for a store-level failure
    variant: Variant


@dataclass(frozen=True)
class JobResult:
    """Outcome of one :class:`Job` (or a store-level failure)."""

    job: Job
    status: str  # "ok" | "failed" | "skipped"
    npz_path: Path | None
    sidecar_path: Path | None
    error: str | None  # traceback when ``status == "failed"``
    summary: Mapping[str, Any]


# ----------------------------------------------------------------- planning


def plan_jobs(
    spec: BatchSpec,
    *,
    force: bool = False,
    only: Mapping[str, str] | None = None,
    stores: Sequence[Path] | None = None,
    open_dataset: DatasetOpener | None = None,
) -> list[Job]:
    """Enumerate the campaign's pending jobs.

    Opens each (selected) store, lists its deformations, crosses them
    with the spec's variants, applies the ``only`` filter, and drops
    variants already on disk (unless ``force``). A store that fails to
    open is skipped with a warning rather than aborting the preview —
    :func:`run_batch` records such a store as a failed job instead.

    Parameters
    ----------
    spec
        The parsed campaign.
    force
        If true, include variants whose result already exists on disk.
    only
        Optional ``{knob: value}`` filter (string values, compared via
        :func:`~mamba_dvc.run.config.slug_value`); only variants whose
        resolved params match every entry are kept.
    stores
        Optional subset of ``spec.stores`` to plan for (matched by path
        or by ``.zarr`` directory name).
    open_dataset
        Optional opener override; defaults to the campaign's standard
        manifest-aware opener.

    Returns
    -------
    list of Job
        In ``(store, deformation, variant)`` order.

    Raises
    ------
    ValueError
        If ``only`` names a knob the driver does not understand.
    """
    _validate_only(only)
    opener = open_dataset or _make_default_opener(spec)
    target_stores = _select_target_stores(spec.stores, stores)
    campaign_dir = spec.campaign_dir

    jobs: list[Job] = []
    for store in target_stores:
        try:
            ds = opener(store, None)
        except Exception as exc:
            # Preview is best-effort: a broken store drops out of the plan.
            warnings.warn(
                f"store {store} failed to open, skipping in plan: {exc!r}", stacklevel=2
            )
            continue
        jobs.extend(_store_jobs(spec, store, ds, campaign_dir, force=force, only=only))
    return jobs


def _store_jobs(
    spec: BatchSpec,
    store: Path,
    ds: DvcDataset,
    campaign_dir: Path,
    *,
    force: bool,
    only: Mapping[str, str] | None,
) -> list[Job]:
    """Pending jobs for one already-open store (entry listing is convention-free)."""
    selected = spec.select_names(ds.list_real(), ds.list_synthetic())
    out: list[Job] = []
    for name, kind in selected:
        for variant in spec.variants:
            if only is not None and not _matches_only(variant, only):
                continue
            job = Job(store=store, deformation=name, kind=kind, variant=variant)
            if not force and _is_done(campaign_dir, job):
                continue
            out.append(job)
    return out


# ----------------------------------------------------------------- execution


def run_batch(
    spec: BatchSpec,
    *,
    force: bool = False,
    only: Mapping[str, str] | None = None,
    stores: Sequence[Path] | None = None,
    correlate_fn: _CorrelateFn | None = None,
    open_dataset: DatasetOpener | None = None,
) -> list[JobResult]:
    """Execute the campaign with the materialize-once / iterate-many loop.

    Writes ``config.snapshot.yaml`` (or a serialised copy of the parsed
    config), then for every pending job: ``correlate`` → ``evaluate``
    (synthetic only) → write ``<variant_id>.npz`` + ``.json``, appending
    a row to ``manifest.jsonl`` as it goes. ``variants.json`` is
    refreshed at the end. Already-done jobs are skipped (resume); a
    failed job (store open, ``load_pair``, ``correlate``, ``evaluate``,
    or write) is recorded and the run continues.

    Parameters
    ----------
    spec
        The parsed campaign.
    force, only, stores
        As for :func:`plan_jobs`.
    correlate_fn
        Optional callable with ``correlate_multi_gpu``'s call shape;
        defaults to it. Lets tests / custom dispatch paths slot in.
    open_dataset
        Optional opener override; see :data:`DatasetOpener`.

    Returns
    -------
    list of JobResult
        One per job *run this invocation* (skipped/resumed jobs are not
        included). Failures carry ``status == "failed"`` and a traceback.

    Raises
    ------
    ValueError
        If ``only`` names an unknown knob.
    """
    _validate_only(only)
    correlate = correlate_fn or correlate_multi_gpu
    opener = open_dataset or _make_default_opener(spec)
    target_stores = _select_target_stores(spec.stores, stores)
    campaign_dir = spec.campaign_dir

    _warn_if_convention_unpinned(spec)

    campaign_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_dir / "manifest.jsonl"
    _write_config_snapshot(spec, campaign_dir)

    results: list[JobResult] = []

    def _record(jr: JobResult) -> None:
        results.append(jr)
        _append_manifest_row(manifest_path, spec, jr)

    for store in target_stores:
        # Enumerate this store's pending jobs (entry listing does not
        # depend on flow_convention, so a single open suffices here).
        try:
            ds_probe = opener(store, None)
            jobs = _store_jobs(spec, store, ds_probe, campaign_dir, force=force, only=only)
        except Exception:
            _record(
                JobResult(
                    job=Job(
                        store=store,
                        deformation=_STORE_LEVEL,
                        kind="?",
                        variant=spec.variants[0],
                    ),
                    status="failed",
                    npz_path=None,
                    sidecar_path=None,
                    error=traceback.format_exc(),
                    summary={"phase": "open_store"},
                )
            )
            continue
        if not jobs:
            continue

        # Group by flow_convention → one (possibly re-opened) dataset each.
        for convention, conv_jobs in _group_by(
            jobs, lambda j: j.variant.load_params["flow_convention"]
        ):
            try:
                ds = opener(store, convention)
            except Exception:
                for job in conv_jobs:
                    _record(_failed(job, "open_store", traceback.format_exc()))
                continue

            # Group by (deformation, mask, dry_shape) → one load_pair each.
            for load_key, load_jobs in _group_by(conv_jobs, _load_group_key):
                deformation, mask_sel, dry_shape = load_key
                try:
                    pair = ds.load_pair(
                        deformation,
                        mask=_mask_selector(mask_sel),
                        dry_shape=dry_shape,
                    )
                except Exception:
                    for job in load_jobs:
                        _record(_failed(job, "load_pair", traceback.format_exc()))
                    continue

                truth_cache: dict[tuple[Any, ...], np.ndarray] = {}
                # Sort by grid params so adjacent variants reuse the cache.
                for job in sorted(load_jobs, key=lambda j: _grid_key(j.variant)):
                    _record(
                        _run_variant(spec, pair, job, correlate, truth_cache, campaign_dir)
                    )

                # The materialized pair is ~30 GB on production volumes;
                # release it before the next load_pair (plan §6).
                del pair
                gc.collect()

    _write_variants_index(spec, campaign_dir)
    return results


def _run_variant(
    spec: BatchSpec,
    pair: EvaluationPair,
    job: Job,
    correlate: _CorrelateFn,
    truth_cache: dict[tuple[Any, ...], np.ndarray],
    campaign_dir: Path,
) -> JobResult:
    """Correlate → (evaluate) → persist one variant against a materialized pair."""
    variant = job.variant

    t0 = time.perf_counter()
    try:
        field = _correlate_with_fallback(correlate, pair, variant, spec.devices)
    except Exception:
        return _failed(job, "correlate", traceback.format_exc())
    wall_correlate = time.perf_counter() - t0

    try:
        report: ErrorReport | None = None
        wall_evaluate = 0.0
        if pair.gt_field is not None:
            te0 = time.perf_counter()
            gk = _grid_key(variant)
            if gk not in truth_cache:
                truth_cache[gk] = np.ascontiguousarray(
                    pair.gt_field(field.positions), dtype=np.float32
                )
            report = evaluate_pair(
                pair, field, distance_bins=spec.distance_bins, truth=truth_cache[gk]
            )
            wall_evaluate = time.perf_counter() - te0

        npz_path, sidecar_path = _result_paths(campaign_dir, job)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        field.save_npz(npz_path)
        summary = _summarize(
            field,
            report,
            timings={
                "wall_correlate_s": round(wall_correlate, 3),
                "wall_evaluate_s": round(wall_evaluate, 3),
            },
        )
        _write_sidecar(sidecar_path, spec, job, summary, scored=report is not None)
    except Exception:
        return _failed(job, "evaluate_or_write", traceback.format_exc())

    return JobResult(
        job=job,
        status="ok",
        npz_path=npz_path,
        sidecar_path=sidecar_path,
        error=None,
        summary=summary,
    )


def _correlate_with_fallback(
    correlate: _CorrelateFn,
    pair: EvaluationPair,
    variant: Variant,
    devices: tuple[int, ...] | None,
) -> DisplacementField:
    kwargs: dict[str, Any] = {**variant.grid_params, **variant.compute_params}
    try:
        return correlate(
            pair.reference, pair.deformed, mask=pair.mask, device_ids=devices, **kwargs
        )
    except Exception:
        # Host pinned-memory exhaustion scales with the worker count;
        # if the user pinned >2 GPUs, retry on the first two before
        # giving up. (No fallback when ``devices`` is None — we won't
        # enumerate the fleet here.)
        if devices is not None and len(devices) > 2:
            return correlate(
                pair.reference, pair.deformed, mask=pair.mask, device_ids=devices[:2], **kwargs
            )
        raise


# ----------------------------------------------------------------- grouping


def _group_by(items: Iterable[Job], key: Callable[[Job], Any]) -> list[tuple[Any, list[Job]]]:
    """Stable group-by preserving first-seen key order."""
    groups: dict[Any, list[Job]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return list(groups.items())


def _load_group_key(job: Job) -> tuple[str, Any, tuple[int, int, int] | None]:
    lp = job.variant.load_params
    dry = lp["dry_shape"]
    dry_tuple = (int(dry[0]), int(dry[1]), int(dry[2])) if dry is not None else None
    return (job.deformation, lp["mask"], dry_tuple)


def _grid_key(variant: Variant) -> tuple[Any, ...]:
    return tuple((k, variant.grid_params[k]) for k in sorted(variant.grid_params))


# --------------------------------------------------------------- opener / paths


def _make_default_opener(spec: BatchSpec) -> DatasetOpener:
    """Build the standard manifest-aware dataset opener for ``spec``.

    Uses the campaign's shared manifest when set, else per-store
    discovery. When a ``flow_convention`` override is requested it is
    patched onto the resolved (or a freshly synthesised) manifest, the
    way ``scripts/run_e2e_zarr.py`` does it.
    """
    shared_manifest = (
        StoreManifest.from_yaml(spec.manifest) if spec.manifest is not None else None
    )

    def _open(store: Path, flow_convention: str | None) -> DvcDataset:
        if flow_convention is None:
            return DvcDataset.open(store, manifest=shared_manifest)
        base = (
            shared_manifest if shared_manifest is not None else StoreManifest.discover(store)
        )
        patched = _apply_flow_convention(base, flow_convention)
        return DvcDataset.open(store, manifest=patched)

    return _open


def _apply_flow_convention(manifest: StoreManifest | None, convention: str) -> StoreManifest:
    """Return ``manifest`` with ``synthetic.flow.convention`` set to ``convention``."""
    base = manifest if manifest is not None else StoreManifest()
    new_flow = replace(base.synthetic.flow, convention=convention)  # type: ignore[arg-type]
    return replace(base, synthetic=replace(base.synthetic, flow=new_flow))


def _mask_selector(mask: Any) -> Any:
    """Translate a config ``mask`` value into a ``load_pair`` selector.

    ``None`` → profile/manifest default; ``"none"`` → :data:`NO_MASK`
    (skip the mask); any other string → that mask name.
    """
    if mask is None:
        return None
    if isinstance(mask, str) and mask.lower() == "none":
        return NO_MASK
    return mask


def _store_stem(store: Path) -> str:
    name = store.name
    return name[:-5] if name.endswith(".zarr") else name


def _result_paths(campaign_dir: Path, job: Job) -> tuple[Path, Path]:
    # ``variant_id`` carries dots (``mt0.7-...``), so build the suffixes by
    # string concat — ``Path.with_suffix`` would mangle ``.7-<hash>``.
    parent = campaign_dir / _store_stem(job.store) / job.deformation
    vid = job.variant.variant_id
    return parent / f"{vid}.npz", parent / f"{vid}.json"


def _rel_npz(job: Job) -> str:
    return f"{_store_stem(job.store)}/{job.deformation}/{job.variant.variant_id}.npz"


def _select_target_stores(
    all_stores: Sequence[Path], subset: Sequence[Path] | None
) -> list[Path]:
    if subset is None:
        return list(all_stores)
    wanted = list(subset)
    out: list[Path] = []
    for store in all_stores:
        if any(_store_matches(store, w) for w in wanted):
            out.append(store)
    for w in wanted:
        if not any(_store_matches(store, w) for store in all_stores):
            warnings.warn(f"--store {w} matches no store in the campaign config", stacklevel=2)
    return out


def _store_matches(store: Path, query: Path) -> bool:
    if store == query or store.name == query.name:
        return True
    try:
        return store.resolve() == query.resolve()
    except OSError:  # pragma: no cover - resolve on a vanished path
        return False


# ----------------------------------------------------------------- resume


def _is_done(campaign_dir: Path, job: Job) -> bool:
    """Report whether ``job``'s result is on disk with a matching params hash."""
    npz_path, sidecar_path = _result_paths(campaign_dir, job)
    if not npz_path.exists() or not sidecar_path.exists():
        return False
    try:
        with sidecar_path.open("r", encoding="utf-8") as fh:
            sidecar = json.load(fh)
    except OSError, json.JSONDecodeError:
        return False
    if not isinstance(sidecar, Mapping):
        return False
    return cast("Mapping[str, Any]", sidecar).get("params_hash") == job.variant.params_hash


# ----------------------------------------------------------------- only filter


def _validate_only(only: Mapping[str, str] | None) -> None:
    if only is None:
        return
    unknown = set(only) - knob_names()
    if unknown:
        raise ValueError(
            f"--only references unknown knob(s) {sorted(unknown)}; "
            f"known knobs are {sorted(knob_names())}"
        )


def _matches_only(variant: Variant, only: Mapping[str, str]) -> bool:
    return all(
        slug_value(variant.params.get(key)) == str(value) for key, value in only.items()
    )


# ----------------------------------------------------------------- warnings


def _warn_if_convention_unpinned(spec: BatchSpec) -> None:
    """Emit one warning if any variant leaves ``flow_convention`` unset.

    Per plan invariant §2.6: an unpinned convention falls back to each
    store's manifest/profile, and the canonical transduced stores were
    authored ``push_forward`` while the current profile default is
    ``pull_back`` — a mismatch silently inflates synthetic error ~50x.
    """
    unpinned = sorted(
        v.variant_id for v in spec.variants if v.load_params["flow_convention"] is None
    )
    if unpinned:
        warnings.warn(
            "flow_convention is unset for variant(s) "
            f"{unpinned}; the resolved convention will come from each store's "
            "manifest/profile. Verify it matches how the synthetic flow was authored — "
            "a mismatch silently inflates synthetic error ~50x "
            "(see docs/plans/run-interface.md).",
            stacklevel=3,
        )


# ----------------------------------------------------------------- summaries


def _summarize(
    field: DisplacementField,
    report: ErrorReport | None,
    *,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    status = field.status
    counts = {
        name: int(np.count_nonzero(status == int(member)))
        for name, member in POIStatus.__members__.items()
    }
    out: dict[str, Any] = {
        "n_total": int(status.size),
        "n_ok": counts.get("OK", 0),
        "n_masked": counts.get("MASKED", 0),
        "n_outlier": counts.get("OUTLIER", 0),
        "status_counts": counts,
        "grid_shape": list(field.grid_shape),
        **dict(timings),
    }
    if report is not None:
        out["mae"] = report.mae
        out["rmse"] = report.rmse
        out["p95"] = report.p95
        out["per_axis_mae"] = list(report.per_axis_mae)
        out["confidence_vs_error"] = report.confidence_vs_error
        if report.by_distance is not None:
            bd = report.by_distance
            out["by_distance"] = {
                "edges": list(bd.edges),
                "counts": bd.counts.tolist(),
                "mae": bd.mae.tolist(),
                "rmse": bd.rmse.tolist(),
            }
    return out


# ----------------------------------------------------------------- sidecar / manifest


def _write_sidecar(
    path: Path, spec: BatchSpec, job: Job, summary: Mapping[str, Any], *, scored: bool
) -> None:
    payload = {
        "campaign": spec.campaign,
        "store": _store_stem(job.store),
        "store_path": str(job.store),
        "deformation": job.deformation,
        "kind": job.kind,
        "variant_id": job.variant.variant_id,
        "name": job.variant.name,
        "params_hash": job.variant.params_hash,
        "params": dict(job.variant.params),
        "scored": scored,
        "summary": dict(summary),
        "provenance": _provenance(),
        "npz": _rel_npz(job),
    }
    _dump_json(path, payload)


def _append_manifest_row(path: Path, spec: BatchSpec, jr: JobResult) -> None:
    job = jr.job
    row: dict[str, Any] = {
        "campaign": spec.campaign,
        "store": _store_stem(job.store),
        "deformation": job.deformation,
        "kind": job.kind,
        "variant_id": job.variant.variant_id,
        "name": job.variant.name,
        "params_hash": job.variant.params_hash,
        "status": jr.status,
        "npz": _rel_npz(job) if jr.npz_path is not None else None,
    }
    # Flatten the resolved params into columns so a DataFrame groups cleanly.
    for key, value in job.variant.params.items():
        row[key] = value
    # Lift the scalar summary fields up to the row level.
    for key in (
        "n_total",
        "n_ok",
        "n_masked",
        "n_outlier",
        "mae",
        "rmse",
        "p95",
        "per_axis_mae",
        "confidence_vs_error",
        "wall_correlate_s",
        "wall_evaluate_s",
    ):
        if key in jr.summary:
            row[key] = jr.summary[key]
    if jr.status == "failed" and jr.error is not None:
        # Sidecars are only written for ``ok`` jobs, so a failed row
        # carries (the tail of) its traceback for triage.
        row["error"] = jr.error[-2000:]
        if "phase" in jr.summary:
            row["phase"] = jr.summary["phase"]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_jsonable(row), allow_nan=False) + "\n")


def _write_variants_index(spec: BatchSpec, campaign_dir: Path) -> None:
    index = {
        v.variant_id: {"name": v.name, "params_hash": v.params_hash, "params": dict(v.params)}
        for v in spec.variants
    }
    _dump_json(campaign_dir / "variants.json", index)


def _write_config_snapshot(spec: BatchSpec, campaign_dir: Path) -> None:
    target = campaign_dir / "config.snapshot.yaml"
    if spec.source_path is not None and spec.source_path.exists():
        target.write_text(spec.source_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    import yaml

    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(spec.raw), fh, sort_keys=False)


# ----------------------------------------------------------------- provenance


def _provenance() -> dict[str, Any]:
    return {
        "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "package_version": _package_version(),
        "git_sha": _git_sha(),
    }


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("mamba-dvc")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - editable/uninstalled
        return None


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:  # pragma: no cover - git missing
        return None
    sha = out.stdout.strip()
    return sha or None


# ----------------------------------------------------------------- json io


def _failed(job: Job, phase: str, error: str) -> JobResult:
    return JobResult(
        job=job,
        status="failed",
        npz_path=None,
        sidecar_path=None,
        error=error,
        summary={"phase": phase},
    )


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays, tuples, and NaNs for JSON."""
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in cast("Mapping[Any, Any]", obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in cast("Sequence[Any]", obj)]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(payload), fh, indent=2, allow_nan=False)
        fh.write("\n")
