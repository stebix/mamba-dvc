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
import time
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
from mamba_dvc.gpu.shm import SharedArrayHandle, attach, publish, published
from mamba_dvc.instrument import log_phase, timed
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

__all__ = ["MultiGPUDispatcher", "correlate_multi_gpu"]


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
    with timed("dispatch.h2d", sync=True):
        ref_dev = _cp.asarray(reference)
        def_dev = _cp.asarray(deformed)
        mask_dev = _cp.asarray(mask)
        def_mask_dev = _cp.asarray(deformed_mask)
    grid_dev, admitted_dev = _to_device(grid, admitted_idx)

    # ``sync=True`` on the helper block: the inner batched loop only
    # launches kernels, so without an exit sync this would measure launch
    # time, not compute. The sync drains the device queue at the seam --
    # the same drain ``cp.asnumpy`` does below, just attributed here.
    with timed("dispatch.helper", sync=True):
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
    # work queue to drain before we touch the host buffers (already drained
    # by the helper block's exit sync, so this measures just the D2H copy).
    with timed("dispatch.d2h", sync=True):
        result = (_cp.asnumpy(disp_dev), _cp.asnumpy(conf_dev), _cp.asnumpy(stat_dev))
    return result


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

    t_total = time.perf_counter()

    with timed("dispatch.build_grid"):
        grid = build_grid(volume_shape, window=win, overlap=overlap)
    n_points = int(np.prod(grid.grid_shape))

    with timed("dispatch.resolve_masks"):
        eff_mask, eff_def_mask = resolve_masks(mask, deformed_mask, volume_shape)

    if mask is not None:
        with timed("dispatch.filter_by_mask", n_points=n_points):
            admitted = filter_by_mask(grid, eff_mask, mask_threshold)
    else:
        admitted = np.ones(n_points, dtype=np.bool_)
    admitted_idx = np.flatnonzero(admitted).astype(np.int64)

    # Volumes / masks must be C-contiguous to publish into shared
    # memory without an extra copy on the worker side. This is the
    # ~30 GB host-side materialisation flagged in
    # ``docs/plans/general-perf-improvements.md`` R3.
    with timed("dispatch.materialize_contiguous"):
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

    with timed(
        "dispatch.compute",
        n_devices=len(resolved_device_ids),
        n_admitted=int(admitted_idx.shape[0]),
        batch_size=resolved_batch,
    ):
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
    with timed("dispatch.outlier"):
        valid_pre = status == POIStatus.OK
        outlier_flag = detect_outliers(grid, displacements, valid_pre)
        status[outlier_flag] = POIStatus.OUTLIER
        displacements[outlier_flag] = 0.0
        confidence[outlier_flag] = 0.0

    valid = status == POIStatus.OK

    log_phase(
        "dispatch.total",
        time.perf_counter() - t_total,
        n_points=n_points,
        n_admitted=int(admitted_idx.shape[0]),
        n_devices=len(resolved_device_ids),
    )
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


# ============================================================================
# Persistent multi-GPU dispatcher (time-series amortization)
# ============================================================================
#
# Buildout doc: docs/buildout/timestep-pipeline.md sections 3 and 6.
#
# ``correlate_multi_gpu`` above pays spawn (~1.5 s x N) + CUDA init (~3 s x N)
# once per call. For a 100-frame series that wastes minutes on setup that
# could be paid once. :class:`MultiGPUDispatcher` hoists per-series state
# (workers, mask, optional anchored reference, grid, admitted shards, auto
# batch) into ``__enter__`` so each ``.correlate(...)`` call is compute-only.
#
# Worker protocol on the duplex pipes:
#     parent -> worker  ("pair", deformed_handle, ref_handle | None)
#     parent -> worker  ("shutdown",)
#     worker -> parent  ("ready",)                       (one-time, on enter)
#     worker -> parent  ("ok", disp_host, conf_host, stat_host)
#     worker -> parent  ("err", repr_exc, traceback)
#
# Per-pair payload only ships the new SHM handles; mask, anchored reference
# (if any), grid, shard_idx and helper_kwargs are bound at worker startup.


def _serve_worker(
    gpu_id: int,
    mask_handle: SharedArrayHandle,
    anchored_ref_handle: SharedArrayHandle | None,
    grid: GridSpec,
    shard_idx: np.ndarray,
    helper_kwargs: dict[str, Any],
    request_recv: Connection,
    result_send: Connection,
) -> None:
    """Serve dispatcher pair requests on ``gpu_id`` until shutdown.

    Pins to ``gpu_id``, attaches the resident mask (and optional
    anchored reference) once, sends ``("ready",)`` to the parent, then
    services ``("pair", ...)`` requests until ``("shutdown",)``. Pre-
    loop failures send a single ``("err", ...)`` and exit; per-pair
    failures send ``("err", ...)`` and continue serving so the parent
    can surface a single pair failure without tearing down the pool.
    """
    resident_shm: list[Any] = []
    try:
        import cupy as cp  # pyright: ignore[reportMissingImports]

        cp.cuda.Device(gpu_id).use()

        mask_shm, mask_view = attach(mask_handle)
        resident_shm.append(mask_shm)
        mask_dev = cp.asarray(mask_view)
        # v1 has no separate deformed_mask through the dispatcher (per-
        # frame deformed masks are a v2 feature); reuse the same device
        # buffer.
        def_mask_dev = mask_dev

        anchored_ref_dev: Any = None
        if anchored_ref_handle is not None:
            anch_shm, anch_view = attach(anchored_ref_handle)
            resident_shm.append(anch_shm)
            anchored_ref_dev = cp.asarray(anch_view)

        grid_dev, shard_dev = _to_device(grid, shard_idx)
    except Exception as exc:
        with suppress(Exception):
            result_send.send(("err", repr(exc), traceback.format_exc()))
        for shm in resident_shm:
            with suppress(Exception):
                shm.close()
        with suppress(Exception):
            request_recv.close()
        with suppress(Exception):
            result_send.close()
        return

    try:
        result_send.send(("ready",))
    except Exception:
        # Parent died before handshake. Bail without serving.
        for shm in resident_shm:
            with suppress(Exception):
                shm.close()
        with suppress(Exception):
            request_recv.close()
        return

    try:
        while True:
            try:
                msg = request_recv.recv()
            except EOFError:
                # Parent closed its send-pipe without an explicit
                # shutdown -- treat as graceful exit.
                break

            if msg[0] == "shutdown":
                break

            if msg[0] != "pair":
                result_send.send(("err", f"unknown message type {msg[0]!r}", ""))
                continue

            _, deformed_handle, ref_handle = msg
            per_pair_shm: list[Any] = []
            try:
                def_shm, def_view = attach(deformed_handle)
                per_pair_shm.append(def_shm)
                def_dev = cp.asarray(def_view)

                if ref_handle is None:
                    if anchored_ref_dev is None:
                        raise RuntimeError(
                            "dispatcher worker received ref_handle=None but "
                            "no anchored reference was published at startup"
                        )
                    ref_dev = anchored_ref_dev
                else:
                    ref_shm, ref_view = attach(ref_handle)
                    per_pair_shm.append(ref_shm)
                    ref_dev = cp.asarray(ref_view)

                disp_dev, conf_dev, stat_dev = correlate_admitted_subset(
                    ref_dev,
                    def_dev,
                    mask_dev,
                    def_mask_dev,
                    grid_dev,
                    shard_dev,
                    **helper_kwargs,
                )
                disp_host = cp.asnumpy(disp_dev)
                conf_host = cp.asnumpy(conf_dev)
                stat_host = cp.asnumpy(stat_dev)
                result_send.send(("ok", disp_host, conf_host, stat_host))
            except Exception as exc:
                # Per-pair failure: stay in the serve loop. The parent
                # surfaces this as a single failed pair to the driver.
                result_send.send(("err", repr(exc), traceback.format_exc()))
            finally:
                for shm in per_pair_shm:
                    with suppress(Exception):
                        shm.close()
    finally:
        for shm in resident_shm:
            with suppress(Exception):
                shm.close()
        with suppress(Exception):
            request_recv.close()
        with suppress(Exception):
            result_send.close()


class MultiGPUDispatcher:
    """Persistent multi-GPU worker pool for the time-series driver.

    Replaces the per-call spawn + CUDA-init cost of
    :func:`correlate_multi_gpu` with a once-per-series cost paid in
    :meth:`__enter__`. Resident on every device for the dispatcher's
    lifetime:

    - the validity mask (and, for ``REFERENCE_ANCHORED`` series, the
      anchored reference volume);
    - the grid and the device's shard of admitted POI indices;
    - the resolved auto-batch size and the constant helper kwargs.

    Per-pair, only the new ``deformed`` volume (and the new
    ``reference`` for SEQUENTIAL series) is published to shared memory
    and uploaded to device -- the resident buffers are reused.

    Construction does not spawn workers or import CuPy; that happens
    on :meth:`__enter__`. The constructor only stores config and
    validates what is decidable without a device.

    Usage
    -----
    Use as a context manager::

        with MultiGPUDispatcher(
            device_ids=[0, 1, 2, 3],
            mask=screw_mask,
            anchored_reference=frame_0,  # REFERENCE_ANCHORED only
            window=96,
            overlap=0.5,
        ) as dispatcher:
            for t, frame in frames:
                field = dispatcher.correlate(None, frame)

    For SEQUENTIAL series, omit ``anchored_reference`` and pass the
    previous frame as the reference on each call.

    Single-device fast path
    -----------------------
    When ``len(device_ids) == 1``, the dispatcher skips
    :mod:`multiprocessing` entirely and uploads the resident state to
    the parent's in-process CuPy context. This is the cheap baseline
    against which the multi-process path is regression-tested.

    Failure model
    -------------
    Per-pair failures (CUDA OOM, helper exceptions) surface as a
    :class:`RuntimeError` raised from :meth:`correlate`; the dispatcher
    stays usable for subsequent pairs because the workers stay in their
    serve loop. A worker dying outright makes subsequent :meth:`correlate`
    calls raise -- the driver should treat that as series-aborting.

    Parameters
    ----------
    device_ids
        GPU indices to dispatch across. ``None`` (default) selects
        every visible CUDA device at :meth:`__enter__` time.
    mask
        Optional ``(z, y, x)`` boolean validity mask shared across all
        pairs in v1.
    anchored_reference
        Optional ``(z, y, x)`` float32 volume resident on every device
        for the dispatcher's lifetime. Required for
        ``REFERENCE_ANCHORED`` series; omit for SEQUENTIAL. When
        supplied, :meth:`correlate` accepts ``reference=None`` to mean
        "use the resident anchored reference".
    volume_shape
        ``(z, y, x)`` shape every reference / deformed volume must
        match. Inferred from ``mask`` or ``anchored_reference`` when
        not supplied; required otherwise.
    window, overlap, mask_threshold, tukey_alpha, search_radius
        Forwarded to the per-pair correlator. Identical semantics to
        :func:`correlate_multi_gpu`. Bound at construction and reused
        across every pair.
    batch_size
        Per-shard FFT batch. ``"auto"`` probes free VRAM at
        :meth:`__enter__` once and reuses the result for every pair;
        an int bypasses the recommender.
    eps, ncc_mode, ncc_normalization
        Forwarded to the helper unchanged.

    Raises
    ------
    ValueError
        For invalid construction inputs (bad shapes / dtypes, no way
        to infer ``volume_shape``, malformed ``batch_size`` /
        ``ncc_mode`` / ``ncc_normalization``).
    RuntimeError
        On :meth:`__enter__` if CuPy is unavailable or no CUDA device
        is visible, or a worker fails to start.
    """

    def __init__(
        self,
        device_ids: Sequence[int] | None = None,
        *,
        mask: Bool[np.ndarray, "z y x"] | None = None,
        anchored_reference: Float32[np.ndarray, "z y x"] | None = None,
        volume_shape: tuple[int, int, int] | None = None,
        window: int | tuple[int, int, int] = 96,
        overlap: float = 0.5,
        mask_threshold: float = 0.9,
        tukey_alpha: float | None = None,
        search_radius: int | None = None,
        batch_size: int | Literal["auto"] = "auto",
        eps: float = 1e-12,
        ncc_mode: NCCMode = NCCMode.LINEAR,
        ncc_normalization: NCCNormalization = NCCNormalization.OVERLAP,
    ) -> None:
        # Resolve volume shape: explicit > mask > anchored_reference.
        inferred_shape: tuple[int, int, int] | None = None
        if volume_shape is not None:
            if len(volume_shape) != 3:
                raise ValueError(f"volume_shape must have length 3, got {volume_shape}")
            inferred_shape = (
                int(volume_shape[0]),
                int(volume_shape[1]),
                int(volume_shape[2]),
            )
        elif mask is not None:
            if mask.ndim != 3:
                raise ValueError(f"mask must be 3D, got ndim={mask.ndim}")
            inferred_shape = (
                int(mask.shape[0]),
                int(mask.shape[1]),
                int(mask.shape[2]),
            )
        elif anchored_reference is not None:
            if anchored_reference.ndim != 3:
                raise ValueError(
                    f"anchored_reference must be 3D, got ndim={anchored_reference.ndim}"
                )
            inferred_shape = (
                int(anchored_reference.shape[0]),
                int(anchored_reference.shape[1]),
                int(anchored_reference.shape[2]),
            )
        if inferred_shape is None:
            raise ValueError(
                "MultiGPUDispatcher needs a volume_shape (or a mask, or an "
                "anchored_reference) to bind the grid"
            )

        if mask is not None:
            if mask.shape != inferred_shape:
                raise ValueError(
                    f"mask shape {mask.shape} does not match volume_shape {inferred_shape}"
                )
            if mask.dtype != np.bool_:
                raise ValueError(f"mask must have bool dtype, got {mask.dtype}")
        if anchored_reference is not None:
            if anchored_reference.shape != inferred_shape:
                raise ValueError(
                    f"anchored_reference shape {anchored_reference.shape} does not match "
                    f"volume_shape {inferred_shape}"
                )
            if anchored_reference.dtype != np.float32:
                raise ValueError(
                    f"anchored_reference must be float32, got {anchored_reference.dtype}"
                )

        if isinstance(batch_size, int):
            if batch_size <= 0:
                raise ValueError(
                    f"batch_size must be a positive int or 'auto', got {batch_size!r}"
                )
        elif batch_size != "auto":
            raise ValueError(
                f"batch_size must be a positive int or 'auto', got {batch_size!r}"
            )

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

        win = normalize_window(window)
        resolved_search_radius = min(win) // 2 if search_radius is None else int(search_radius)
        if resolved_search_radius <= 0:
            raise ValueError(f"search_radius must be positive, got {resolved_search_radius}")

        self._device_ids_arg = device_ids
        self._mask = mask
        self._anchored_reference = anchored_reference
        self._volume_shape: tuple[int, int, int] = inferred_shape
        self._window = win
        self._overlap = float(overlap)
        self._mask_threshold = float(mask_threshold)
        self._tukey_alpha = tukey_alpha
        self._search_radius = resolved_search_radius
        self._batch_size = batch_size
        self._eps = float(eps)
        self._ncc_mode = ncc_mode
        self._ncc_normalization = ncc_normalization

        # Lifecycle flag.
        self._opened = False
        # State populated in __enter__:
        self._device_ids: list[int] = []
        self._is_multiprocess: bool = False
        self._grid: GridSpec | None = None
        self._n_points: int = 0
        self._helper_kwargs: dict[str, Any] = {}
        self._admitted_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._shards_idx: list[np.ndarray] = []
        self._effective_mask_c: np.ndarray | None = None
        self._anchored_reference_c: np.ndarray | None = None
        # Multi-process plumbing:
        self._workers: list[Any] = []
        self._send_pipes: list[Connection] = []
        self._recv_pipes: list[Connection] = []
        self._mask_shm_obj: Any = None
        self._anch_shm_obj: Any = None
        # In-process plumbing:
        self._inproc_mask_dev: Any = None
        self._inproc_def_mask_dev: Any = None
        self._inproc_anchored_dev: Any = None
        self._inproc_grid_dev: GridSpec | None = None
        self._inproc_admitted_dev: Any = None

    @property
    def has_anchored_reference(self) -> bool:
        """Whether an anchored reference was supplied at construction."""
        return self._anchored_reference is not None

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Bound volume shape every pair must conform to."""
        return self._volume_shape

    @property
    def device_ids(self) -> tuple[int, ...]:
        """Resolved device ids; only meaningful after :meth:`__enter__`."""
        return tuple(self._device_ids)

    @property
    def is_multiprocess(self) -> bool:
        """Whether the dispatcher will spawn workers (``len(device_ids) > 1``)."""
        return self._is_multiprocess

    def __enter__(self) -> "MultiGPUDispatcher":
        """Spawn workers (or upload in-process state) and bind the grid.

        Validates that CuPy is available, resolves ``device_ids``,
        builds the series grid + admitted POI set, resolves the
        auto-batch, publishes the resident mask (+ optional anchored
        reference) into shared memory, and either spawns one serve-loop
        worker per device (multi-GPU) or uploads the resident buffers
        to the parent's in-process CuPy context (single-GPU). Returns
        ``self`` so the dispatcher can be bound by ``with ... as``.
        """
        if self._opened:
            raise RuntimeError("MultiGPUDispatcher is already open")

        self._device_ids = _resolve_device_ids(self._device_ids_arg)

        eff_mask, _eff_def_mask = resolve_masks(self._mask, None, self._volume_shape)
        self._grid = build_grid(self._volume_shape, window=self._window, overlap=self._overlap)
        n_points = int(np.prod(self._grid.grid_shape))
        self._n_points = n_points
        if self._mask is not None:
            admitted = filter_by_mask(self._grid, eff_mask, self._mask_threshold)
        else:
            admitted = np.ones(n_points, dtype=np.bool_)
        admitted_idx = np.flatnonzero(admitted).astype(np.int64)
        self._admitted_idx = admitted_idx

        if self._batch_size == "auto":
            budget_inputs = BudgetInputs(
                volume_shape=self._volume_shape,
                window=self._window,
                mode=self._ncc_mode,
                normalization=self._ncc_normalization,
                has_mask=self._mask is not None,
                deformed_mask_distinct=False,
            )
            resolved_batch = _resolve_auto_batch(budget_inputs, self._device_ids)
        else:
            resolved_batch = int(self._batch_size)

        resolved_tukey = (
            TUKEY_DEFAULTS[self._ncc_mode]
            if self._tukey_alpha is None
            else float(self._tukey_alpha)
        )
        self._helper_kwargs = {
            "search_radius": self._search_radius,
            "batch_size": resolved_batch,
            "eps": self._eps,
            "ncc_mode": self._ncc_mode,
            "ncc_normalization": self._ncc_normalization,
            "tukey_alpha": resolved_tukey,
        }

        self._effective_mask_c = np.ascontiguousarray(eff_mask)
        if self._anchored_reference is not None:
            self._anchored_reference_c = np.ascontiguousarray(self._anchored_reference)

        try:
            if len(self._device_ids) == 1:
                self._is_multiprocess = False
                self._enter_inproc()
                self._shards_idx = [admitted_idx]
            else:
                self._is_multiprocess = True
                self._enter_multiprocess(admitted_idx)
        except BaseException:
            # Workers may have been partially started; tear them down
            # before re-raising.
            if self._is_multiprocess:
                self._exit_multiprocess()
            self._cleanup_state()
            self._opened = False
            raise

        self._opened = True
        return self

    def _enter_inproc(self) -> None:
        """In-process single-device path: upload resident buffers."""
        assert _cp is not None
        assert self._grid is not None
        assert self._effective_mask_c is not None
        device_id = self._device_ids[0]
        with _cp.cuda.Device(device_id):
            self._inproc_mask_dev = _cp.asarray(self._effective_mask_c)
            self._inproc_def_mask_dev = self._inproc_mask_dev
            if self._anchored_reference_c is not None:
                self._inproc_anchored_dev = _cp.asarray(self._anchored_reference_c)
            grid_dev, admitted_dev = _to_device(self._grid, self._admitted_idx)
            self._inproc_grid_dev = grid_dev
            self._inproc_admitted_dev = admitted_dev

    def _enter_multiprocess(self, admitted_idx: np.ndarray) -> None:
        """Multi-device path: publish SHM, spawn serve-loop workers, handshake."""
        assert self._grid is not None
        assert self._effective_mask_c is not None
        shards = _shard_admitted_indices(admitted_idx, self._grid, len(self._device_ids))
        self._shards_idx = shards

        mask_shm, mask_handle = publish(self._effective_mask_c)
        self._mask_shm_obj = mask_shm
        anchored_handle: SharedArrayHandle | None = None
        if self._anchored_reference_c is not None:
            anch_shm, anchored_handle = publish(self._anchored_reference_c)
            self._anch_shm_obj = anch_shm

        ctx = mp.get_context("spawn")
        for gpu_id, shard_idx in zip(self._device_ids, shards, strict=True):
            req_recv, req_send = ctx.Pipe(duplex=False)
            res_recv, res_send = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_serve_worker,
                args=(
                    gpu_id,
                    mask_handle,
                    anchored_handle,
                    self._grid,
                    shard_idx,
                    dict(self._helper_kwargs),
                    req_recv,
                    res_send,
                ),
            )
            proc.start()
            req_recv.close()
            res_send.close()
            self._workers.append(proc)
            self._send_pipes.append(req_send)  # pyright: ignore[reportArgumentType]
            self._recv_pipes.append(res_recv)  # pyright: ignore[reportArgumentType]

        # Handshake: every worker must send ("ready",) before we let
        # callers issue pair requests. Per-worker failures surface here
        # as ("err", ...) and abort the open.
        errors: list[str] = []
        for gpu_id, recv in zip(self._device_ids, self._recv_pipes, strict=True):
            try:
                msg = recv.recv()
            except EOFError:
                errors.append(f"worker on GPU {gpu_id} closed pipe before sending ready")
                continue
            if msg[0] == "ready":
                continue
            if msg[0] == "err":
                _, repr_exc, tb = msg
                errors.append(f"worker on GPU {gpu_id} failed startup: {repr_exc}\n{tb}")
            else:
                errors.append(f"worker on GPU {gpu_id} sent unexpected handshake {msg[0]!r}")
        if errors:
            raise RuntimeError(
                "MultiGPUDispatcher worker startup failed:\n" + "\n".join(errors)
            )

    def __exit__(self, *exc_info: Any) -> None:
        """Shut down workers, release SHM, and clear resident state.

        Idempotent: a second call (or one on a dispatcher that never
        opened) is a no-op. Exceptions during worker join or SHM
        unlink are suppressed so a partial failure during teardown
        cannot mask the original ``with``-body exception.
        """
        if not self._opened:
            return
        try:
            if self._is_multiprocess:
                self._exit_multiprocess()
            else:
                self._exit_inproc()
        finally:
            self._cleanup_state()
            self._opened = False

    def _exit_inproc(self) -> None:
        """Release in-process device references; CuPy pool reclaims memory."""
        self._inproc_mask_dev = None
        self._inproc_def_mask_dev = None
        self._inproc_anchored_dev = None
        self._inproc_grid_dev = None
        self._inproc_admitted_dev = None

    def _exit_multiprocess(self) -> None:
        """Send shutdown to every worker, join with timeout, terminate on hang."""
        for pipe in self._send_pipes:
            with suppress(Exception):
                pipe.send(("shutdown",))
        for pipe in self._send_pipes:
            with suppress(Exception):
                pipe.close()
        for pipe in self._recv_pipes:
            with suppress(Exception):
                pipe.close()
        for proc in self._workers:
            proc.join(timeout=10)
            if proc.is_alive():
                with suppress(Exception):
                    proc.terminate()
                proc.join(timeout=5)

    def _cleanup_state(self) -> None:
        """Unpublish SHM and clear plumbing state. Idempotent."""
        if self._mask_shm_obj is not None:
            with suppress(Exception):
                self._mask_shm_obj.close()
            with suppress(OSError):
                self._mask_shm_obj.unlink()
            self._mask_shm_obj = None
        if self._anch_shm_obj is not None:
            with suppress(Exception):
                self._anch_shm_obj.close()
            with suppress(OSError):
                self._anch_shm_obj.unlink()
            self._anch_shm_obj = None
        self._workers = []
        self._send_pipes = []
        self._recv_pipes = []

    def correlate(
        self,
        reference: Float32[np.ndarray, "z y x"] | None,
        deformed: Float32[np.ndarray, "z y x"],
    ) -> DisplacementField:
        """Run one pair across the pool and return a :class:`DisplacementField`.

        Parameters
        ----------
        reference
            ``(z, y, x)`` float32 reference volume, or ``None`` to use
            the resident anchored reference (requires
            ``anchored_reference`` to have been supplied at construction).
        deformed
            ``(z, y, x)`` float32 deformed volume.

        Returns
        -------
        DisplacementField
            Same shape and semantics as :func:`correlate_multi_gpu` on
            equivalent inputs (within float32 noise).

        Raises
        ------
        RuntimeError
            If called outside a ``with`` block, or a worker raised
            mid-pair.
        ValueError
            If shapes or dtypes do not match the dispatcher's bound
            configuration, or ``reference=None`` is passed without an
            anchored reference.
        """
        if not self._opened:
            raise RuntimeError(
                "MultiGPUDispatcher.correlate must be called inside the with block"
            )
        if reference is None and not self.has_anchored_reference:
            raise ValueError(
                "reference=None requires the dispatcher to have an anchored_reference"
            )
        if deformed.shape != self._volume_shape:
            raise ValueError(
                f"deformed shape {deformed.shape} does not match "
                f"dispatcher volume_shape {self._volume_shape}"
            )
        if deformed.dtype != np.float32:
            raise ValueError(f"deformed must be float32, got {deformed.dtype}")
        if reference is not None:
            if reference.shape != self._volume_shape:
                raise ValueError(
                    f"reference shape {reference.shape} does not match "
                    f"dispatcher volume_shape {self._volume_shape}"
                )
            if reference.dtype != np.float32:
                raise ValueError(f"reference must be float32, got {reference.dtype}")

        deformed_c = np.ascontiguousarray(deformed)
        reference_c = None if reference is None else np.ascontiguousarray(reference)

        if self._is_multiprocess:
            disp, conf, stat = self._dispatch_pair_mp(reference_c, deformed_c)
        else:
            disp, conf, stat = self._dispatch_pair_inproc(reference_c, deformed_c)

        # Outlier rejection on the assembled lattice -- same step the
        # one-shot ``correlate_multi_gpu`` runs at the parent.
        assert self._grid is not None
        valid_pre = stat == POIStatus.OK
        outlier_flag = detect_outliers(self._grid, disp, valid_pre)
        stat[outlier_flag] = POIStatus.OUTLIER
        disp[outlier_flag] = 0.0
        conf[outlier_flag] = 0.0
        valid = stat == POIStatus.OK

        return DisplacementField(
            positions=self._grid.positions,
            displacements=disp,
            valid=valid,
            confidence=conf,
            status=stat,
            grid_shape=self._grid.grid_shape,
            spacing=self._grid.spacing,
            window=self._grid.window,
        )

    def _dispatch_pair_inproc(
        self,
        reference_c: np.ndarray | None,
        deformed_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert _cp is not None
        # All ``_inproc_*_dev`` slots are populated by :meth:`_enter_inproc`,
        # so the dispatcher is guaranteed to be in a usable state here.
        assert self._inproc_grid_dev is not None
        device_id = self._device_ids[0]
        with _cp.cuda.Device(device_id):
            def_dev = _cp.asarray(deformed_c)
            if reference_c is None:
                ref_dev = self._inproc_anchored_dev
                if ref_dev is None:
                    raise RuntimeError(
                        "in-process dispatcher: reference=None requires anchored_reference"
                    )
            else:
                ref_dev = _cp.asarray(reference_c)

            disp_dev, conf_dev, stat_dev = correlate_admitted_subset(
                ref_dev,
                def_dev,
                self._inproc_mask_dev,
                self._inproc_def_mask_dev,
                self._inproc_grid_dev,
                self._inproc_admitted_dev,
                **self._helper_kwargs,
            )
            return (
                _cp.asnumpy(disp_dev),
                _cp.asnumpy(conf_dev),
                _cp.asnumpy(stat_dev),
            )

    def _dispatch_pair_mp(
        self,
        reference_c: np.ndarray | None,
        deformed_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_points = self._n_points
        merged_disp = np.zeros((n_points, 3), dtype=np.float32)
        merged_conf = np.zeros(n_points, dtype=np.float32)
        merged_stat = np.full(n_points, POIStatus.MASKED, dtype=np.uint8)

        # Per-pair SHM lifetime: bracket the publish around send + gather.
        def_shm_obj, def_handle = publish(deformed_c)
        ref_shm_obj: Any = None
        ref_handle: SharedArrayHandle | None = None
        try:
            if reference_c is not None:
                ref_shm_obj, ref_handle = publish(reference_c)

            for pipe in self._send_pipes:
                pipe.send(("pair", def_handle, ref_handle))

            pipe_to_meta: dict[Connection, tuple[int, np.ndarray]] = {
                recv: (gid, shard)
                for recv, gid, shard in zip(
                    self._recv_pipes, self._device_ids, self._shards_idx, strict=True
                )
            }
            pending: set[Connection] = set(pipe_to_meta.keys())
            errors: list[str] = []
            while pending:
                ready = wait(list(pending))
                for recv in ready:  # pyright: ignore[reportGeneralTypeIssues]
                    if recv not in pending:
                        continue
                    pending.discard(recv)  # pyright: ignore[reportArgumentType]
                    gpu_id, shard_idx = pipe_to_meta[recv]  # pyright: ignore[reportArgumentType]
                    try:
                        msg = recv.recv()  # pyright: ignore[reportAttributeAccessIssue]
                    except EOFError as exc:
                        raise RuntimeError(
                            f"dispatcher worker on GPU {gpu_id} (shard size "
                            f"{int(shard_idx.size)}) died before sending result"
                        ) from exc

                    if msg[0] == "err":
                        _, repr_exc, tb = msg
                        errors.append(
                            f"worker on GPU {gpu_id} (shard size "
                            f"{int(shard_idx.size)}) raised: {repr_exc}\n{tb}"
                        )
                        continue
                    _, s_disp, s_conf, s_stat = msg
                    merged_disp[shard_idx] = s_disp[shard_idx]
                    merged_conf[shard_idx] = s_conf[shard_idx]
                    merged_stat[shard_idx] = s_stat[shard_idx]
            if errors:
                raise RuntimeError("MultiGPUDispatcher pair failed:\n" + "\n".join(errors))
        finally:
            with suppress(Exception):
                def_shm_obj.close()
            with suppress(OSError):
                def_shm_obj.unlink()
            if ref_shm_obj is not None:
                with suppress(Exception):
                    ref_shm_obj.close()
                with suppress(OSError):
                    ref_shm_obj.unlink()

        return merged_disp, merged_conf, merged_stat
