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

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.core.grid import build_grid, filter_by_mask
from mamba_dvc.core.outlier import detect_outliers
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
    batch_size: int = 256,
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
        memory but not the result. The linear kernel allocates
        ``(batch, 2W, 2W, 2W)`` float32 buffers (~8x the cyclic
        kernel's transient footprint); at production ``W=96`` expect
        to drop ``batch_size`` to ~64 to fit on a single A6000. A
        future helper in :mod:`mamba_dvc.gpu.dispatch` will recommend
        batch sizes programmatically.
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
        host/device mixing, or out-of-range parameters.

    Notes
    -----
    The function is pure. v2 multi-GPU dispatch replicates the volumes
    on each device and calls
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
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

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

    effective_mask, effective_def_mask = resolve_masks(mask, deformed_mask, volume_shape)

    admitted = (
        filter_by_mask(grid, effective_mask, mask_threshold)
        if mask is not None
        else np.ones(n_points, dtype=np.bool_)
    )
    admitted_idx = np.flatnonzero(admitted).astype(np.int64)

    displacements, confidence, status = correlate_admitted_subset(
        reference,
        deformed,
        effective_mask,
        effective_def_mask,
        grid,
        admitted_idx,
        search_radius=search_radius,
        batch_size=batch_size,
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
