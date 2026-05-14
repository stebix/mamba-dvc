"""Pure single-device driver lifting :func:`correlate` to a time series.

Implements the time-series hook reserved in ``docs/plans/overview.md``
§10 and detailed in ``docs/buildout/timestep-pipeline.md``. The driver
walks an iterator of ``(t, frame)`` tuples, applies a pluggable
:class:`mamba_dvc.types.PairingStrategy`, and assembles per-pair
:class:`mamba_dvc.types.DisplacementField` results into a single
:class:`mamba_dvc.types.DisplacementSeries`.

The function is **pure** in the same sense as
:func:`mamba_dvc.pipeline.correlate.correlate`: it takes an iterator
of frames already in memory, holds no state across calls, and never
touches the filesystem. Frame supply, prefetch, and disk streaming
live in higher-level wrappers (``io/series.py``, CLI) consuming this
driver. The persistent multi-GPU pool reserved in
``docs/plans/dispatch.md`` is not wired in here; v1 routes every pair
through the single-device :func:`correlate` path. A dispatcher branch
will land alongside the ``MultiGPUDispatcher`` context manager.

Failure isolation
-----------------
A pair that raises (CUDA OOM, propagated worker exception, malformed
input) records a :data:`SeriesPairStatus.FAILED` row with a
zero-filled :class:`DisplacementField` and the driver keeps consuming
the iterator. Downstream cumulative composition halts at the first
failed boundary (see :meth:`DisplacementSeries.cumulative`). A
``RuntimeWarning`` is emitted so the failure is visible without
turning the function impure or coupling it to a logger.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.core.grid import build_grid
from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
from mamba_dvc.pipeline._internal import NCCMode, NCCNormalization
from mamba_dvc.pipeline.correlate import correlate
from mamba_dvc.types import (
    DisplacementField,
    DisplacementSeries,
    PairingStrategy,
    POIStatus,
    SeriesPairObserver,
    SeriesPairStatus,
)

__all__ = ["correlate_series"]


def correlate_series(
    frames: Iterable[tuple[int, Float32[np.ndarray, "z y x"]]],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    *,
    strategy: PairingStrategy = PairingStrategy.SEQUENTIAL,
    dispatcher: MultiGPUDispatcher | None = None,
    on_pair: Callable[[int, DisplacementField], None] | None = None,
    pair_observer: SeriesPairObserver | None = None,
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float | None = None,
    search_radius: int | None = None,
    batch_size: int | Literal["auto"] = "auto",
    eps: float = 1e-12,
    ncc_mode: NCCMode = NCCMode.LINEAR,
    ncc_normalization: NCCNormalization = NCCNormalization.OVERLAP,
) -> DisplacementSeries:
    """Run a single-device DVC pass over an ordered iterator of frames.

    Parameters
    ----------
    frames
        Iterable of ``(t, frame)`` tuples. ``t`` is an integer frame
        index (not a timestamp); ``frame`` is a ``(z, y, x)`` float32
        volume. The driver consumes the iterator exactly once; the
        first tuple seeds the reference (or the previous-frame slot
        for :data:`PairingStrategy.SEQUENTIAL`) and the grid is built
        from its shape. Rewinding is the caller's job.
    mask
        Optional shared validity mask. ``None`` means "every voxel is
        valid". The same mask is reused for every pair — per-frame
        masks are a v2 feature.
    strategy
        Pairing topology. :data:`PairingStrategy.SEQUENTIAL` (default)
        pairs ``(prev, next)``; :data:`PairingStrategy.REFERENCE_ANCHORED`
        pairs ``(first, next)``;
        :data:`PairingStrategy.UPDATED_REFERENCE` raises until the v2
        warper lands.
    dispatcher
        Optional opened :class:`mamba_dvc.gpu.dispatch.MultiGPUDispatcher`.
        When supplied, every pair routes through the persistent pool so
        the spawn + CUDA-init cost is paid once for the whole series
        instead of per pair. The dispatcher must be opened by the caller
        (``with MultiGPUDispatcher(...) as d:``) and its
        ``volume_shape`` must match the seed frame. For
        :data:`PairingStrategy.REFERENCE_ANCHORED` series, if the
        dispatcher was built with an ``anchored_reference``, the driver
        skips re-uploading the reference each call. When ``None``
        (default), pairs go through the host-only
        :func:`mamba_dvc.pipeline.correlate.correlate` -- useful for
        tests and CPU-only dev machines.
    on_pair
        Optional callback ``(t_def, field)`` fired after each pair
        completes (success or failure). The sole side-channel out of
        the driver — used for streaming writes, progress bars, or
        per-frame error capture. Receives the deformed-frame ``t``,
        not the ``(t_ref, t_def)`` pair index.
    pair_observer
        Optional :class:`SeriesPairObserver` fired with
        ``on_pair_start`` *before* dispatch and ``on_pair_end`` *after*
        the pair completes. Carries the full ``(t_ref, t_def)`` pair
        index plus the resulting :class:`SeriesPairStatus` and
        :class:`DisplacementField`. Used by
        :class:`mamba_dvc.run.eventlog.SeriesPairLogger` to bind
        ``t_ref`` / ``t_def`` contextvars so every phase record fired
        inside a pair (``dispatch.*``, ``ncc.*``) inherits them, plus
        emit explicit ``kind:"pair_start"`` / ``kind:"pair_end"`` rows
        in ``events.jsonl``. The protocol is the structured analogue
        to ``on_pair``; both may be supplied -- they fire independently
        in a well-defined order (``pair_observer.on_pair_start`` →
        dispatch → ``pair_observer.on_pair_end`` → ``on_pair``).
    window, overlap, mask_threshold, tukey_alpha, search_radius
        Forwarded unchanged to :func:`correlate`. The driver introduces
        no new tuning knobs in v1.
    batch_size, eps, ncc_mode, ncc_normalization
        Forwarded unchanged to :func:`correlate`. See its docstring for
        per-mode semantics.

    Returns
    -------
    DisplacementSeries
        Ordered container of per-pair fields with shared grid metadata
        and a ``pair_status`` array recording any per-pair failures.
        Empty series (``len(fields) == 0``) is returned when the
        iterator yields zero or one frame.

    Raises
    ------
    ValueError
        If the iterator is empty (no first frame to seed the grid).
    NotImplementedError
        If ``strategy`` is :data:`PairingStrategy.UPDATED_REFERENCE`.

    Warnings
    --------
    Emits a :class:`RuntimeWarning` for each per-pair failure. The
    failure is otherwise silent on the return value beyond the
    :data:`SeriesPairStatus.FAILED` row — pair-level failure reasons
    are not stored on :class:`DisplacementSeries` in v1 (deferred to
    follow-on work on the IO + checkpoint layer).
    """
    if strategy is PairingStrategy.UPDATED_REFERENCE:
        raise NotImplementedError(
            "PairingStrategy.UPDATED_REFERENCE is reserved for v2 (warm-start with"
            " iterative warp); the v1 driver does not implement it."
        )

    frames_iter = iter(frames)
    try:
        t0, seed_frame = next(frames_iter)
    except StopIteration as exc:
        raise ValueError(
            "frames iterator is empty — cannot build a series without a first frame"
        ) from exc

    if seed_frame.ndim != 3:
        raise ValueError(f"first frame must be 3D, got ndim={seed_frame.ndim}")
    if seed_frame.dtype != np.float32:
        raise ValueError(f"first frame must be float32, got {seed_frame.dtype}")

    volume_shape: tuple[int, int, int] = (
        int(seed_frame.shape[0]),
        int(seed_frame.shape[1]),
        int(seed_frame.shape[2]),
    )

    # Build the series-level grid once. correlate() builds its own grid
    # internally per call (cheap, ~O(volume) once setup), but we record
    # ours on the DisplacementSeries so cumulative composition and
    # downstream error reports share a single source of truth.
    grid = build_grid(volume_shape, window=window, overlap=overlap)
    n_points = int(np.prod(grid.grid_shape))

    def _failed_field() -> DisplacementField:
        return DisplacementField(
            positions=grid.positions,
            displacements=np.zeros((n_points, 3), dtype=np.float32),
            valid=np.zeros(n_points, dtype=np.bool_),
            confidence=np.zeros(n_points, dtype=np.float32),
            status=np.full(n_points, np.uint8(POIStatus.MASKED), dtype=np.uint8),
            grid_shape=grid.grid_shape,
            spacing=grid.spacing,
            window=grid.window,
        )

    # Resolve dispatcher path once: when one is supplied, every pair
    # routes through the pool; when not, every pair calls the pure
    # host-only ``correlate()``. The branch lives outside the per-pair
    # try/except so the dispatcher's per-pair RuntimeError is caught
    # alongside any other failure mode.
    anchored_via_dispatcher = (
        dispatcher is not None
        and dispatcher.has_anchored_reference
        and strategy is PairingStrategy.REFERENCE_ANCHORED
    )

    def _run_pair(
        reference: Float32[np.ndarray, "z y x"],
        deformed: Float32[np.ndarray, "z y x"],
        t_ref: int,
        t_def: int,
    ) -> tuple[DisplacementField, SeriesPairStatus]:
        # Broad ``except Exception`` is intentional: per-pair failure
        # isolation is the buildout-doc contract. KeyboardInterrupt /
        # SystemExit derive from BaseException and bypass this branch.
        if pair_observer is not None:
            pair_observer.on_pair_start(t_ref=t_ref, t_def=t_def)
        field: DisplacementField
        status: SeriesPairStatus
        try:
            if dispatcher is not None:
                ref_arg = None if anchored_via_dispatcher else reference
                field = dispatcher.correlate(ref_arg, deformed)
            else:
                field = correlate(
                    reference,
                    deformed,
                    mask=mask,
                    window=window,
                    overlap=overlap,
                    mask_threshold=mask_threshold,
                    tukey_alpha=tukey_alpha,
                    search_radius=search_radius,
                    batch_size=batch_size,
                    eps=eps,
                    ncc_mode=ncc_mode,
                    ncc_normalization=ncc_normalization,
                )
            status = SeriesPairStatus.OK
        except Exception as exc:
            warnings.warn(
                f"correlate_series: pair ({t_ref}, {t_def}) failed with"
                f" {type(exc).__name__}: {exc}; recording as SeriesPairStatus.FAILED",
                RuntimeWarning,
                stacklevel=3,
            )
            field = _failed_field()
            status = SeriesPairStatus.FAILED
        if pair_observer is not None:
            pair_observer.on_pair_end(t_ref=t_ref, t_def=t_def, status=status, field=field)
        return field, status

    fields: list[DisplacementField] = []
    pair_index_rows: list[tuple[int, int]] = []
    pair_status_rows: list[int] = []

    if strategy is PairingStrategy.REFERENCE_ANCHORED:
        anchored_frame = seed_frame
        for t, frame in frames_iter:
            field, status = _run_pair(anchored_frame, frame, t0, t)
            fields.append(field)
            pair_index_rows.append((t0, t))
            pair_status_rows.append(int(status))
            if on_pair is not None:
                on_pair(t, field)
    else:
        # SEQUENTIAL — pair (prev, next) and promote next as the new prev.
        prev_t, prev_frame = t0, seed_frame
        for t, frame in frames_iter:
            field, status = _run_pair(prev_frame, frame, prev_t, t)
            fields.append(field)
            pair_index_rows.append((prev_t, t))
            pair_status_rows.append(int(status))
            if on_pair is not None:
                on_pair(t, field)
            prev_t, prev_frame = t, frame

    if pair_index_rows:
        pair_indices = np.asarray(pair_index_rows, dtype=np.int64)
        pair_status = np.asarray(pair_status_rows, dtype=np.uint8)
    else:
        pair_indices = np.empty((0, 2), dtype=np.int64)
        pair_status = np.empty(0, dtype=np.uint8)

    return DisplacementSeries(
        fields=tuple(fields),
        pair_indices=pair_indices,
        pair_status=pair_status,
        grid=grid,
        strategy=strategy,
    )
