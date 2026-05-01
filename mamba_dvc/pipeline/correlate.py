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
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.core.extract import extract_subvolumes
from mamba_dvc.core.grid import build_grid, filter_by_mask
from mamba_dvc.core.ncc import correlate as correlate_ncc
from mamba_dvc.core.ncc import peak_displacement
from mamba_dvc.core.outlier import detect_outliers
from mamba_dvc.core.peakfit import gaussian_subvoxel_fit
from mamba_dvc.core.window import preprocess_subvolumes
from mamba_dvc.types import DisplacementField, POIStatus

__all__ = ["correlate"]

NCCMode = Literal["linear", "cyclic"]
NCCNormalization = Literal["overlap", "global"]

# Per-mode Tukey defaults. Linear NCC does not need spectral-leakage
# suppression from a window because zero-padding eliminates the cyclic
# wrap; the box itself plays the role of the window in the Lewis
# denominator. Cyclic NCC retains the conventional 0.25 taper from plan
# §2 to suppress the leakage that the wrap would otherwise inject.
_TUKEY_DEFAULTS: dict[NCCMode, float] = {"linear": 0.0, "cyclic": 0.25}


def _normalize_window(window: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Coerce a scalar or 3-tuple window into ``(wz, wy, wx)``."""
    if isinstance(window, int):
        return (window, window, window)
    if len(window) != 3:
        raise ValueError(f"window tuple must have length 3, got {len(window)}")
    wz, wy, wx = (int(w) for w in window)
    return (wz, wy, wx)


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
    ncc_mode: NCCMode = "linear",
    ncc_normalization: NCCNormalization = "overlap",
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
    on each device and calls this function once per shard of POIs;
    v2's iterative warp driver calls it inside a fixed-point loop
    after warping the deformed volume by the previous estimate. Both
    require that no orchestration state survives a return.
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

    win = _normalize_window(window)
    if search_radius is None:
        search_radius = min(win) // 2
    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")

    if ncc_mode not in ("linear", "cyclic"):
        raise ValueError(f"ncc_mode must be 'linear' or 'cyclic', got {ncc_mode!r}")
    if ncc_normalization not in ("overlap", "global"):
        raise ValueError(
            f"ncc_normalization must be 'overlap' or 'global', got {ncc_normalization!r}"
        )

    resolved_tukey = _TUKEY_DEFAULTS[ncc_mode] if tukey_alpha is None else tukey_alpha

    volume_shape: tuple[int, int, int] = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
    )

    grid = build_grid(volume_shape, window=win, overlap=overlap)
    n_points = grid.starts.shape[0]

    # Mask handling. In v1 the deformed mask falls back to the
    # reference mask; the parameter is in the signature today so v2
    # can switch on a per-frame mask without an API change.
    if mask is None:
        effective_mask = np.ones(volume_shape, dtype=np.bool_)
    else:
        if mask.shape != volume_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {volume_shape}"
            )
        if mask.dtype != np.bool_:
            raise ValueError(f"mask must have bool dtype, got {mask.dtype}")
        effective_mask = mask

    if deformed_mask is None:
        effective_def_mask = effective_mask
    else:
        if deformed_mask.shape != volume_shape:
            raise ValueError(
                f"deformed_mask shape {deformed_mask.shape} does not match "
                f"volume shape {volume_shape}"
            )
        if deformed_mask.dtype != np.bool_:
            raise ValueError(f"deformed_mask must have bool dtype, got {deformed_mask.dtype}")
        effective_def_mask = deformed_mask

    admitted = (
        filter_by_mask(grid, effective_mask, mask_threshold)
        if mask is not None
        else np.ones(n_points, dtype=np.bool_)
    )

    displacements = np.zeros((n_points, 3), dtype=np.float32)
    confidence = np.zeros(n_points, dtype=np.float32)
    status = np.full(n_points, POIStatus.OK, dtype=np.uint8)
    status[~admitted] = POIStatus.MASKED

    admitted_idx = np.flatnonzero(admitted)

    for batch_start in range(0, admitted_idx.size, batch_size):
        batch_slice = admitted_idx[batch_start : batch_start + batch_size]
        chunk = grid.starts[batch_slice]

        ref_sv = extract_subvolumes(reference, chunk, grid.window)
        def_sv = extract_subvolumes(deformed, chunk, grid.window)
        mref_sv = extract_subvolumes(effective_mask, chunk, grid.window)
        mdef_sv = extract_subvolumes(effective_def_mask, chunk, grid.window)

        ref_pp = preprocess_subvolumes(ref_sv, mref_sv, tukey_alpha=resolved_tukey)
        def_pp = preprocess_subvolumes(def_sv, mdef_sv, tukey_alpha=resolved_tukey)

        corr = correlate_ncc(
            ref_pp,
            def_pp,
            mode=ncc_mode,
            normalization=ncc_normalization,
            eps=eps,
        )
        integer, peak = peak_displacement(corr)
        fractional = gaussian_subvoxel_fit(corr, integer)

        displacements[batch_slice] = integer.astype(np.float32) + fractional
        confidence[batch_slice] = peak

    # search_radius enforcement. integer-peak magnitude is what
    # determines whether the lag is unambiguous; the fractional
    # refinement only nudges by < 1 voxel and cannot push a borderline
    # POI from valid to invalid in a way that matters at our budget.
    out_of_range = admitted & (np.abs(displacements).max(axis=1) > float(search_radius))
    status[out_of_range] = POIStatus.OUT_OF_RANGE
    displacements[out_of_range] = 0.0
    confidence[out_of_range] = 0.0

    # Outlier rejection (plan §2 step 7). Run after the search-radius
    # gate so out-of-range POIs do not participate in any neighborhood;
    # the precedence MASKED -> OUT_OF_RANGE -> OUTLIER preserves the
    # original failure mode for diagnostic stratification.
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
