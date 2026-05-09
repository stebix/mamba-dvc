"""Multi-GPU dispatch for the single-pair DVC pipeline.

Dispatch is the multi-GPU sibling of
:mod:`mamba_dvc.pipeline.correlate`; both modules import the shared
inner-loop / validation surface from
:mod:`mamba_dvc.pipeline._internal` so there is one source of truth.

Implements plan §6: replicate the reference / deformed / mask volumes
on every GPU, shard the admitted POIs along lattice-Z into contiguous
slabs, run :func:`mamba_dvc.pipeline._internal.correlate_admitted_subset`
in a per-GPU worker process, gather full-length sparse partials at the
parent, scatter them into one merged set of buffers, and run the
outlier test on the assembled lattice.

Architectural invariants
------------------------

- One Python process per GPU via :class:`multiprocessing.spawn` (the
  Windows-safe context). Each worker pins itself to its assigned device
  before any CuPy import-side allocation.
- No inter-GPU communication during NCC. Every worker sees the *full*
  volume on its own device and only its share of the POI indices.
- The outlier test (:mod:`mamba_dvc.core.outlier`) runs on the parent
  after gather. Its 3x3x3 lattice neighborhood has no halo semantics,
  but it would see invalid data if it ran inside a worker on a partial
  lattice -- so it stays at the seam.
- IPC for the volumes uses :mod:`multiprocessing.shared_memory` via
  :mod:`mamba_dvc.gpu.shm`. When the I/O layer (zarr / tensorstore)
  lands, dispatch can pivot to per-worker independent opens with no
  change to the contract below.

Public entry point: :func:`correlate_multi_gpu`. The single-process
fallback (``len(device_ids) == 1``) skips spawning altogether and
simply uploads the volumes to the single device in-process; that path
is the cheap baseline against which the multi-process path is
regression-tested.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from multiprocessing.connection import Connection, wait
from typing import Any, Literal

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.core.grid import build_grid, filter_by_mask
from mamba_dvc.core.outlier import detect_outliers
from mamba_dvc.gpu.budget import (
    BudgetInputs,
    estimate_max_batch,
    kernel_footprint,
    probe_free_vram,
    resident_bytes,
)
from mamba_dvc.gpu.shm import SharedArrayHandle, attach, published
from mamba_dvc.pipeline._internal import (
    TUKEY_DEFAULTS,
    NCCMode,
    NCCNormalization,
    correlate_admitted_subset,
    normalize_window,
    resolve_masks,
)
from mamba_dvc.types import DisplacementField, GridSpec, POIStatus

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = ["correlate_multi_gpu"]


# Headroom reserved on the free-VRAM pool when ``batch_size="auto"``.
# Same value the budget module uses by default; pinned here so the
# resolution log line in :func:`_resolve_auto_batch` cannot drift from
# the figure passed to :func:`estimate_max_batch`.
_AUTO_HEADROOM_FRACTION: float = 0.15


# Worker -> parent message protocol over a unidirectional Pipe.
# ("ok", displacements, confidence, status) on success;
# ("err", repr_exc, traceback, gpu_id) on failure. The parent demuxes
# on the first tuple element.


def _shard_admitted_indices(
    admitted_idx: np.ndarray,
    grid: GridSpec,
    n_shards: int,
) -> list[np.ndarray]:
    """Partition ``admitted_idx`` into ``n_shards`` contiguous lattice-Z slabs.

    Each shard's index set is the subset of ``admitted_idx`` whose flat
    index, decoded against ``grid.grid_shape``, falls into that shard's
    Z range. Shards may be empty (e.g. a Z-slab fully covered by the
    screw mask) -- the helper handles a zero-length ``admitted_idx``
    cleanly, so empty shards are valid work units.

    Parameters
    ----------
    admitted_idx
        ``(k,)`` int64 flat-lattice indices for POIs that survived
        mask admission.
    grid
        Lattice descriptor; only ``grid.grid_shape`` is consulted.
    n_shards
        Number of shards to produce. Must be positive.

    Returns
    -------
    list of numpy.ndarray
        ``n_shards`` int64 arrays whose union (concatenated, sorted)
        equals ``admitted_idx``.

    Raises
    ------
    ValueError
        If ``n_shards`` is non-positive.
    """
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    nz, ny, nx = grid.grid_shape
    z_lat = admitted_idx // (ny * nx)

    z_chunks = np.array_split(np.arange(nz, dtype=np.int64), n_shards)
    shards: list[np.ndarray] = []
    for chunk in z_chunks:
        if chunk.size == 0:
            shards.append(np.empty(0, dtype=np.int64))
            continue
        z_lo, z_hi = int(chunk[0]), int(chunk[-1] + 1)
        in_slab = (z_lat >= z_lo) & (z_lat < z_hi)
        shards.append(admitted_idx[in_slab].astype(np.int64, copy=False))
    return shards


def _resolve_auto_batch(
    inputs: BudgetInputs,
    device_ids: Sequence[int],
) -> int:
    """Probe each device cold, pick a cross-device batch, log the decision.

    Mirrors :func:`mamba_dvc.gpu.budget.recommend_batch_size` but exposes
    the per-device free-bytes vector so the resolution log is greppable
    in heterogeneous-fleet debugging. Plan §11.1, Option A: emit a
    structured ``key=value`` line on stderr rather than churning the
    return type of :func:`correlate_multi_gpu`.

    The probe must be cold -- run *before* :func:`mamba_dvc.gpu.shm.published`
    publishes the volumes and *before* any worker spawns. Probing after
    the upload would undercount available headroom.
    """
    free_per_device = [(int(d), int(probe_free_vram(int(d))[0])) for d in device_ids]
    min_free = min(free for _, free in free_per_device)

    headroom = _AUTO_HEADROOM_FRACTION
    batch = estimate_max_batch(min_free, inputs, headroom_fraction=headroom)

    fp = kernel_footprint(inputs.window, inputs.mode, inputs.normalization)
    res = resident_bytes(inputs)
    per_dev = ",".join(f"{d}:{free}" for d, free in free_per_device)
    print(
        f"auto.batch_size={batch} "
        f"auto.free_bytes_min={min_free} "
        f"auto.free_bytes_per_device={per_dev} "
        f"auto.resident_bytes={res} "
        f"auto.per_poi_bytes={fp.per_poi_bytes} "
        f"auto.fixed_bytes={fp.fixed_bytes} "
        f"auto.headroom={headroom}",
        file=sys.stderr,
    )
    return batch


def _resolve_device_ids(device_ids: Sequence[int] | None) -> list[int]:
    """Normalize the user's ``device_ids`` argument into a concrete list.

    ``None`` resolves to all visible GPUs (the production deployment).
    An empty sequence is rejected; explicit single-element lists are
    honored (the common dev case).
    """
    if _cp is None:
        raise RuntimeError(
            "correlate_multi_gpu requires CuPy; for host-only execution use "
            "mamba_dvc.pipeline.correlate.correlate"
        )
    if device_ids is None:
        n_visible = int(_cp.cuda.runtime.getDeviceCount())
        if n_visible <= 0:
            raise RuntimeError("no CUDA devices visible to CuPy")
        return list(range(n_visible))
    resolved = [int(d) for d in device_ids]
    if not resolved:
        raise ValueError("device_ids must be non-empty")
    return resolved


def _to_device(grid: GridSpec, admitted_idx: np.ndarray) -> tuple[GridSpec, Any]:
    """Move grid-side host arrays onto the current CuPy device.

    The helper requires that every array it touches lives in the same
    module. Here we promote ``grid.starts`` and ``admitted_idx`` to
    CuPy. ``positions`` is also promoted for symmetry, even though the
    helper never reads it -- this keeps :class:`GridSpec` invariants
    consistent on the device side.
    """
    assert _cp is not None  # guarded by caller
    grid_dev = replace(
        grid,
        positions=_cp.asarray(grid.positions),
        starts=_cp.asarray(grid.starts),
    )
    admitted_dev = _cp.asarray(admitted_idx)
    return grid_dev, admitted_dev


def _run_helper_on_device(
    reference: Float32[np.ndarray, "z y x"],
    deformed: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"],
    deformed_mask: Bool[np.ndarray, "z y x"],
    grid: GridSpec,
    admitted_idx: np.ndarray,
    helper_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upload host arrays, run the helper on the current device, D2H result.

    Used by both the single-GPU in-process path and the worker
    entrypoint. The current CuPy device must already be set by the
    caller.
    """
    assert _cp is not None  # guarded by caller
    ref_dev = _cp.asarray(reference)
    def_dev = _cp.asarray(deformed)
    mask_dev = _cp.asarray(mask)
    def_mask_dev = _cp.asarray(deformed_mask)
    grid_dev, admitted_dev = _to_device(grid, admitted_idx)

    disp_dev, conf_dev, stat_dev = correlate_admitted_subset(
        ref_dev,
        def_dev,
        mask_dev,
        def_mask_dev,
        grid_dev,
        admitted_dev,
        **helper_kwargs,
    )
    # `cp.asnumpy` is a synchronization point -- it forces the device
    # work queue to drain before we touch the host buffers.
    return _cp.asnumpy(disp_dev), _cp.asnumpy(conf_dev), _cp.asnumpy(stat_dev)


def _worker_entry(
    gpu_id: int,
    handles: dict[str, SharedArrayHandle],
    grid: GridSpec,
    admitted_idx: np.ndarray,
    helper_kwargs: dict[str, Any],
    result_send: Connection,
) -> None:
    """Worker process entrypoint -- top-level for ``spawn`` picklability.

    Pins to ``gpu_id``, attaches the four shared-memory volumes, runs
    the helper on the current device for this shard's admitted POIs,
    and sends the D2H'd full-length sparse arrays back to the parent.
    Any exception is captured and shipped over the same pipe so the
    parent can re-raise with context instead of hanging on ``recv``.
    """
    shm_objects: list[Any] = []
    try:
        import cupy as cp  # pyright: ignore[reportMissingImports]

        cp.cuda.Device(gpu_id).use()

        host_arrays: dict[str, np.ndarray] = {}
        for name, handle in handles.items():
            shm, view = attach(handle)
            shm_objects.append(shm)
            host_arrays[name] = view

        disp_host, conf_host, stat_host = _run_helper_on_device(
            host_arrays["reference"],
            host_arrays["deformed"],
            host_arrays["mask"],
            host_arrays["deformed_mask"],
            grid,
            admitted_idx,
            helper_kwargs,
        )
        result_send.send(("ok", disp_host, conf_host, stat_host))
    except Exception as exc:
        # Broad catch is intentional: workers must always send a
        # message back so the parent's `recv` doesn't block forever.
        result_send.send(("err", repr(exc), traceback.format_exc(), gpu_id))
    finally:
        for shm in shm_objects:
            with suppress(Exception):
                shm.close()
        result_send.close()


def _gather_shards(
    device_ids: Sequence[int],
    recv_pipes: list[Connection],
    shards: list[np.ndarray],
    processes: list[Any],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block on workers, scatter each shard's slots into parent buffers.

    Failed workers (signalled via the ``"err"`` message or via a closed
    pipe with a non-zero exit code) raise a :class:`RuntimeError`
    annotated with the GPU id and the shard size, so the failure mode
    is greppable in logs.
    """
    merged_disp = np.zeros((n_points, 3), dtype=np.float32)
    merged_conf = np.zeros(n_points, dtype=np.float32)
    merged_stat = np.full(n_points, POIStatus.MASKED, dtype=np.uint8)

    pipe_to_meta: dict[Connection, tuple[int, np.ndarray, Any]] = {
        recv: (gpu_id, shard_idx, proc)
        for recv, gpu_id, shard_idx, proc in zip(
            recv_pipes, device_ids, shards, processes, strict=True
        )
    }

    while pipe_to_meta:
        # `wait` returns whichever connections are ready. On Windows
        # those are `PipeConnection` instances; on POSIX they are
        # `Connection` instances. They share no common base class in
        # Python's typing, so we cannot type-filter here -- the
        # runtime invariant is `wait()` only returns objects we passed
        # in, and the `recv in pipe_to_meta` check below is the
        # defensive backstop.
        ready_objects = wait(list(pipe_to_meta.keys()))
        for recv in ready_objects:  # pyright: ignore[reportGeneralTypeIssues]
            if recv not in pipe_to_meta:  # pragma: no cover - defensive
                continue
            gpu_id, shard_idx, proc = pipe_to_meta.pop(recv)  # pyright: ignore[reportArgumentType]
            try:
                msg = recv.recv()  # pyright: ignore[reportAttributeAccessIssue]
            except EOFError as exc:
                proc.join(timeout=5)
                raise RuntimeError(
                    f"worker on GPU {gpu_id} (shard size {int(shard_idx.size)}) "
                    f"died with exitcode {proc.exitcode} before sending result"
                ) from exc
            finally:
                recv.close()  # pyright: ignore[reportAttributeAccessIssue]

            if msg[0] == "err":
                _, repr_exc, tb, gid = msg
                raise RuntimeError(
                    f"worker on GPU {gid} (shard size {int(shard_idx.size)}) "
                    f"raised: {repr_exc}\n{tb}"
                )
            _, s_disp, s_conf, s_stat = msg
            merged_disp[shard_idx] = s_disp[shard_idx]
            merged_conf[shard_idx] = s_conf[shard_idx]
            merged_stat[shard_idx] = s_stat[shard_idx]

    for proc in processes:
        proc.join(timeout=10)
    return merged_disp, merged_conf, merged_stat


def _run_multi_process(
    reference: np.ndarray,
    deformed: np.ndarray,
    eff_mask: np.ndarray,
    eff_def_mask: np.ndarray,
    grid: GridSpec,
    admitted_idx: np.ndarray,
    device_ids: list[int],
    helper_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spawn one worker per device, dispatch shards, gather and scatter."""
    n_points = int(np.prod(grid.grid_shape))
    shards = _shard_admitted_indices(admitted_idx, grid, len(device_ids))
    ctx = mp.get_context("spawn")

    with (
        published(reference) as h_ref,
        published(deformed) as h_def,
        published(eff_mask) as h_mask,
        published(eff_def_mask) as h_def_mask,
    ):
        handles = {
            "reference": h_ref,
            "deformed": h_def,
            "mask": h_mask,
            "deformed_mask": h_def_mask,
        }

        processes: list[Any] = []
        recv_pipes: list[Connection] = []
        for gpu_id, shard_idx in zip(device_ids, shards, strict=True):
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_worker_entry,
                args=(gpu_id, handles, grid, shard_idx, helper_kwargs, send),
            )
            proc.start()
            send.close()  # parent-side handle no longer needed
            processes.append(proc)
            # `Pipe(duplex=False)` returns `PipeConnection` on Windows,
            # which is API-compatible with `Connection` -- pyright's
            # nominal split between the two is the only friction here.
            recv_pipes.append(recv)  # pyright: ignore[reportArgumentType]

        try:
            merged = _gather_shards(device_ids, recv_pipes, shards, processes, n_points)
        except BaseException:
            # On any error, terminate stragglers so we don't leak
            # workers (and so the SHM cleanup below isn't blocked).
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in processes:
                proc.join(timeout=5)
            raise

    return merged


def _run_single_process(
    reference: np.ndarray,
    deformed: np.ndarray,
    eff_mask: np.ndarray,
    eff_def_mask: np.ndarray,
    grid: GridSpec,
    admitted_idx: np.ndarray,
    device_id: int,
    helper_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-process single-GPU happy path: upload, run helper, D2H."""
    assert _cp is not None  # guarded by caller
    with _cp.cuda.Device(device_id):
        return _run_helper_on_device(
            reference,
            deformed,
            eff_mask,
            eff_def_mask,
            grid,
            admitted_idx,
            helper_kwargs,
        )


def correlate_multi_gpu(
    reference: Float32[np.ndarray, "z y x"],
    deformed: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    deformed_mask: Bool[np.ndarray, "z y x"] | None = None,
    *,
    device_ids: Sequence[int] | None = None,
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float | None = None,
    search_radius: int | None = None,
    batch_size: int | Literal["auto"] = "auto",
    eps: float = 1e-12,
    ncc_mode: NCCMode = NCCMode.LINEAR,
    ncc_normalization: NCCNormalization = NCCNormalization.OVERLAP,
) -> DisplacementField:
    """Run a single-pair v1 DVC pass across one or more GPUs.

    Mirrors :func:`mamba_dvc.pipeline.correlate.correlate` but executes
    the FFT + peakfit stage on GPU(s) via CuPy. The grid construction,
    mask admission, and outlier test continue to run on the parent
    (host); only the per-POI batched workhorse is pushed to device.

    Parameters
    ----------
    reference, deformed
        ``(z, y, x)`` float32 host volumes. Must share shape.
    mask, deformed_mask
        Optional ``(z, y, x)`` bool masks; same fallback semantics as
        :func:`correlate`.
    device_ids
        GPU indices to dispatch across. ``None`` (default) selects
        every visible CUDA device (production behaviour). A single-id
        list runs in-process without spawning -- useful for dev hosts
        and the single-GPU regression test.
    window, overlap, mask_threshold, tukey_alpha, search_radius
        See :func:`correlate`. Identical semantics.
    batch_size
        Per-shard FFT batch. ``"auto"`` (default) probes free VRAM on
        every device in ``device_ids`` and picks the largest batch that
        fits the resolved window/mode/normalization config; see
        :mod:`mamba_dvc.gpu.budget` for the cost model and plan
        ``docs/plans/batchsize-oracle-v1.md`` for the contract. Pass an
        integer to bypass the recommender entirely (the existing
        manual-tuning workflow).
    eps, ncc_mode, ncc_normalization
        See :func:`correlate`.

    Returns
    -------
    DisplacementField
        Same shape as :func:`correlate`'s output. The two functions are
        bit-identical on equivalent inputs (within float32 noise);
        ``test_multi_gpu_matches_single_gpu`` pins this in CI.

    Raises
    ------
    ValueError
        For malformed inputs (shape, dtype, parameter range), an
        ``batch_size`` that is neither a positive ``int`` nor the
        string ``"auto"``, or an empty ``device_ids`` sequence.
    RuntimeError
        If CuPy is unavailable, if no CUDA device is visible, if the
        ``"auto"`` budget is too tight for the resolved config (the
        message names ``free``, ``resident``, ``per_poi``, and the
        offending ``headroom_fraction``), or if a worker process fails.
        Worker failures are re-raised on the parent with the offending
        ``gpu_id`` and the shard size, plus the worker's traceback for
        diagnosis.

    Notes
    -----
    For ``len(device_ids) > 1`` the parent spawns one
    :class:`multiprocessing.spawn` child per GPU. Volumes ship via
    :mod:`mamba_dvc.gpu.shm` shared memory; per-shard partial results
    return through a unidirectional pipe (~150 KB each). Process
    startup is ~1.5s and CUDA context init is ~3s per device --
    significant for one pair, amortizable in v2's time-series driver.
    """
    if reference.shape != deformed.shape:
        raise ValueError(
            f"reference shape {reference.shape} does not match deformed shape {deformed.shape}"
        )
    if reference.dtype != np.float32 or deformed.dtype != np.float32:
        raise ValueError(
            f"reference and deformed must be float32, "
            f"got {reference.dtype} and {deformed.dtype}"
        )
    if reference.ndim != 3:
        raise ValueError(f"reference must be 3D, got ndim={reference.ndim}")
    # ``Literal["auto"]`` widens batch_size to ``int | str`` at runtime;
    # split the validation so the int branch keeps the existing
    # positivity guard and the string branch only accepts "auto".
    if isinstance(batch_size, int):
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive int or 'auto', got {batch_size!r}"
            )
    elif batch_size != "auto":
        raise ValueError(f"batch_size must be a positive int or 'auto', got {batch_size!r}")

    win = normalize_window(window)
    if search_radius is None:
        search_radius = min(win) // 2
    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")
    try:
        ncc_mode = NCCMode(ncc_mode)
    except ValueError as exc:
        raise ValueError(
            f"ncc_mode must be one of {[m.value for m in NCCMode]}, got {ncc_mode!r}"
        ) from exc
    try:
        ncc_normalization = NCCNormalization(ncc_normalization)
    except ValueError as exc:
        raise ValueError(
            f"ncc_normalization must be one of {[m.value for m in NCCNormalization]}, "
            f"got {ncc_normalization!r}"
        ) from exc

    resolved_tukey = TUKEY_DEFAULTS[ncc_mode] if tukey_alpha is None else tukey_alpha

    volume_shape: tuple[int, int, int] = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
    )

    grid = build_grid(volume_shape, window=win, overlap=overlap)
    n_points = int(np.prod(grid.grid_shape))

    eff_mask, eff_def_mask = resolve_masks(mask, deformed_mask, volume_shape)

    admitted = (
        filter_by_mask(grid, eff_mask, mask_threshold)
        if mask is not None
        else np.ones(n_points, dtype=np.bool_)
    )
    admitted_idx = np.flatnonzero(admitted).astype(np.int64)

    # Volumes / masks must be C-contiguous to publish into shared
    # memory without an extra copy on the worker side.
    reference_c = np.ascontiguousarray(reference)
    deformed_c = np.ascontiguousarray(deformed)
    eff_mask_c = np.ascontiguousarray(eff_mask)
    # When deformed_mask falls back to mask, ascontiguousarray returns
    # the same object -- avoid publishing the same buffer twice.
    eff_def_mask_c = (
        eff_mask_c if eff_def_mask is eff_mask else np.ascontiguousarray(eff_def_mask)
    )

    resolved_device_ids = _resolve_device_ids(device_ids)

    # Resolve "auto" *after* device_ids are known and *before* any SHM
    # publish or worker spawn -- the probe must see free VRAM as the
    # workers will see it. Plan §5.
    if batch_size == "auto":
        budget_inputs = BudgetInputs(
            volume_shape=volume_shape,
            window=win,
            mode=ncc_mode,
            normalization=ncc_normalization,
            has_mask=mask is not None,
            deformed_mask_distinct=eff_def_mask_c is not eff_mask_c,
        )
        resolved_batch = _resolve_auto_batch(budget_inputs, resolved_device_ids)
    else:
        resolved_batch = batch_size

    helper_kwargs: dict[str, Any] = {
        "search_radius": search_radius,
        "batch_size": resolved_batch,
        "eps": eps,
        "ncc_mode": ncc_mode,
        "ncc_normalization": ncc_normalization,
        "tukey_alpha": resolved_tukey,
    }

    if len(resolved_device_ids) == 1:
        displacements, confidence, status = _run_single_process(
            reference_c,
            deformed_c,
            eff_mask_c,
            eff_def_mask_c,
            grid,
            admitted_idx,
            resolved_device_ids[0],
            helper_kwargs,
        )
    else:
        # When mask falls back, dispatch publishes the same buffer
        # twice (once as 'mask', once as 'deformed_mask'). That is
        # fine -- workers attach by name and read; the duplicate is a
        # rounding error against the 1.6 GB mask itself.
        displacements, confidence, status = _run_multi_process(
            reference_c,
            deformed_c,
            eff_mask_c,
            eff_def_mask_c,
            grid,
            admitted_idx,
            resolved_device_ids,
            helper_kwargs,
        )

    # Outlier rejection on the assembled lattice. This is the step
    # that REQUIRES a single host-side sweep; running it inside a
    # worker would see a 3x3x3 neighborhood with cross-shard slots
    # masked out and produce false positives at the slab boundaries.
    valid_pre = status == POIStatus.OK
    outlier_flag = detect_outliers(grid, displacements, valid_pre)
    status[outlier_flag] = POIStatus.OUTLIER
    displacements[outlier_flag] = 0.0
    confidence[outlier_flag] = 0.0

    valid = status == POIStatus.OK

    return DisplacementField(
        positions=grid.positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=grid.grid_shape,
        spacing=grid.spacing,
        window=grid.window,
    )
