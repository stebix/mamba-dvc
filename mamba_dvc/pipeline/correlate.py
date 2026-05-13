"""Single-pair DVC orchestrator (v1).

Implements the pure-function contract described in
``docs/plans/correlate.md``: wraps the six landed core verbs --
``build_grid``, ``filter_by_mask``, ``extract_subvolumes``,
``preprocess_subvolumes``, batched FFT NCC, and Gaussian subvoxel fit
-- into a single call returning a :class:`mamba_dvc.types.DisplacementField`.

The function is intentionally small and stateless. Multi-GPU dispatch
(plan §6) and the time-series / iterative-warp drivers (plan §10) layer
on top of it; future v2 features replace individual core verbs without
touching this glue.

Dispatch seam
-------------
The batched NCC + peakfit + search-radius gate lives in
:mod:`mamba_dvc.pipeline._internal` as
:func:`correlate_admitted_subset`. That helper is shared with
:mod:`mamba_dvc.gpu.dispatch` so there is one source of truth for the
inner loop, mask resolution, and parameter validation. The dispatch
parent shards ``admitted_idx`` across GPUs, calls the helper inside
each worker on device arrays, and scatters the per-shard outputs back
into the parent's full-length buffers.
"""

from __future__ import annotations

import time
import warnings
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.core.grid import build_grid, filter_by_mask
from mamba_dvc.core.outlier import detect_outliers
from mamba_dvc.gpu.budget import BudgetInputs, is_cupy_available, recommend_batch_size
from mamba_dvc.instrument import log_phase, timed
from mamba_dvc.pipeline._internal import (
    TUKEY_DEFAULTS,
    NCCMode,
    NCCNormalization,
    correlate_admitted_subset,
    normalize_window,
    resolve_masks,
)
from mamba_dvc.types import DisplacementField, POIStatus

__all__ = ["correlate"]


def correlate(
    reference: Float32[np.ndarray, "z y x"],
    deformed: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    deformed_mask: Bool[np.ndarray, "z y x"] | None = None,
    *,
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
    """Run a single-pair v1 DVC pass and return a :class:`DisplacementField`.

    Parameters
    ----------
    reference, deformed
        ``(z, y, x)`` float32 volumes. Must share shape and array
        module (NumPy or CuPy).
    mask
        Optional ``(z, y, x)`` boolean validity mask shared between
        the two frames. ``True`` = valid bone tissue, ``False`` =
        excluded (screw / artifact). When ``None``, every voxel is
        treated as valid.
    deformed_mask
        Optional per-frame validity mask for the deformed volume. Plan
        §3 reserves this for v2's per-frame masking; in v1 it falls
        back to ``mask`` when not supplied.
    window
        Subvolume size in voxels. Scalar applies isotropically.
    overlap
        Fraction of the window shared between adjacent POIs along each
        axis. Default ``0.5`` (50% overlap).
    mask_threshold
        Minimum valid-voxel fraction for a POI to be admitted. POIs
        below this threshold are flagged ``MASKED`` in the output.
    tukey_alpha
        Taper fraction of the 3D Tukey window applied during
        preprocessing. ``None`` (default) resolves to a per-mode
        default: ``0.0`` (rectangular) for ``ncc_mode="linear"`` --
        canonical Lewis NCC needs no taper because zero-padding kills
        the wrap-around -- and ``0.25`` for ``ncc_mode="cyclic"`` to
        match plan §2's leakage-suppression convention.
    search_radius
        Maximum permissible integer-lag displacement along any axis.
        POIs whose integer peak exceeds this radius are flagged
        ``OUT_OF_RANGE`` and have their displacement zeroed. Defaults
        to ``min(window) // 2``, the largest unambiguous lag in either
        mode.
    batch_size
        Number of POIs processed per FFT batch. Influences peak GPU
        memory but not the result. ``"auto"`` (default) probes device
        ``0`` and asks :mod:`mamba_dvc.gpu.budget` for the largest
        batch that fits the resolved config; on CPU-only hosts (no
        CuPy) ``"auto"`` falls back to ``64`` with a
        :class:`RuntimeWarning`. Pass an integer to bypass the
        recommender entirely (the function then makes no probe and
        remains side-effect-free).
    eps
        Safety floor passed through to
        :func:`mamba_dvc.core.ncc.correlate`.
    ncc_mode
        ``"linear"`` (default) selects :func:`correlate_linear` --
        zero-pads each subvolume to ``2W`` to remove the cyclic
        wrap-around bias documented in
        ``docs/insights/error-minimization.md``. ``"cyclic"`` selects
        :func:`correlate_cyclic`, kept side-by-side for A/B comparison
        and regression testing.
    ncc_normalization
        ``"overlap"`` (default) uses the per-lag Lewis denominator;
        ``"global"`` uses the whole-window L2. Only meaningful when
        ``ncc_mode="linear"`` -- ``ncc_mode="cyclic"`` requires
        ``"global"``.

    Returns
    -------
    DisplacementField
        Frozen dataclass with one entry per POI in the lattice. Failed
        POIs (``status != POIStatus.OK``) carry zero displacement and
        zero confidence.

    Raises
    ------
    ValueError
        For malformed inputs: shape mismatch, dtype mismatch,
        host/device mixing, ``batch_size`` other than a positive int or
        ``"auto"``, or out-of-range parameters.
    RuntimeError
        If ``batch_size="auto"`` and the resolved per-device budget is
        too tight for the requested config (the message names ``free``,
        ``resident``, ``per_poi``, and the offending
        ``headroom_fraction``).

    Notes
    -----
    The function is pure when ``batch_size`` is an explicit integer.
    ``batch_size="auto"`` performs one cold ``cp.cuda.runtime.memGetInfo``
    probe via :mod:`mamba_dvc.gpu.budget`; that probe is the only side
    effect, and it does not allocate or modify device state. v2
    multi-GPU dispatch replicates the volumes on each device and calls
    :func:`mamba_dvc.pipeline._internal.correlate_admitted_subset`
    once per shard of POIs, sharing the same algorithmic core; v2's
    iterative warp driver calls this function inside a fixed-point
    loop after warping the deformed volume by the previous estimate.
    Both require that no orchestration state survives a return.
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

    with timed("correlate.build_grid"):
        grid = build_grid(volume_shape, window=win, overlap=overlap)
    n_points = int(np.prod(grid.grid_shape))

    with timed("correlate.resolve_masks"):
        effective_mask, effective_def_mask = resolve_masks(mask, deformed_mask, volume_shape)

    if mask is not None:
        with timed("correlate.filter_by_mask", n_points=n_points):
            admitted = filter_by_mask(grid, effective_mask, mask_threshold)
    else:
        admitted = np.ones(n_points, dtype=np.bool_)
    admitted_idx = np.flatnonzero(admitted).astype(np.int64)

    # Resolve "auto" via :mod:`mamba_dvc.gpu.budget`. Probes device 0
    # because ``correlate()`` has no device_ids surface; multi-device
    # callers should use :func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`,
    # which probes the actual device set. CPU-only hosts fall back to a
    # fixed default with a warning so this entrypoint stays usable on
    # dev machines without CuPy.
    if batch_size == "auto":
        if is_cupy_available():
            budget_inputs = BudgetInputs(
                volume_shape=volume_shape,
                window=win,
                mode=ncc_mode,
                normalization=ncc_normalization,
                has_mask=mask is not None,
                deformed_mask_distinct=effective_def_mask is not effective_mask,
            )
            resolved_batch = recommend_batch_size(budget_inputs, [0])
        else:
            warnings.warn(
                "auto-batch requires CuPy; falling back to batch_size=64",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_batch = 64
    else:
        resolved_batch = batch_size

    with timed(
        "correlate.ncc",
        n_admitted=int(admitted_idx.shape[0]),
        batch_size=resolved_batch,
    ):
        displacements, confidence, status = correlate_admitted_subset(
            reference,
            deformed,
            effective_mask,
            effective_def_mask,
            grid,
            admitted_idx,
            search_radius=search_radius,
            batch_size=resolved_batch,
            eps=eps,
            ncc_mode=ncc_mode,
            ncc_normalization=ncc_normalization,
            tukey_alpha=resolved_tukey,
        )

    # Outlier rejection (plan §2 step 7). Run after the search-radius
    # gate (folded into the helper) so out-of-range POIs do not
    # participate in any neighborhood; the precedence MASKED ->
    # OUT_OF_RANGE -> OUTLIER preserves the original failure mode for
    # diagnostic stratification.
    with timed("correlate.outlier"):
        valid_pre = status == POIStatus.OK
        outlier_flag = detect_outliers(grid, displacements, valid_pre)
        status[outlier_flag] = POIStatus.OUTLIER
        displacements[outlier_flag] = 0.0
        confidence[outlier_flag] = 0.0

    valid = status == POIStatus.OK

    log_phase(
        "correlate.total",
        time.perf_counter() - t_total,
        n_points=n_points,
        n_admitted=int(admitted_idx.shape[0]),
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
