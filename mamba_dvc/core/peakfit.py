"""Separable Gaussian subvoxel peak refinement.

Implements step 6 of the single-pass DVC pipeline (plan
``docs/plans/overview.md`` §2): given the per-POI correlation volumes
from :mod:`mamba_dvc.core.ncc` and their integer-lag peak indices, fit
a 1D log-parabola along each axis through the three samples
``corr[i-1], corr[i], corr[i+1]`` to recover a fractional offset in
``(-0.5, 0.5)`` per axis.

The closed-form 3-point log-parabolic fit is exact for any signal that
is locally a Gaussian along that axis — which an FFT-NCC peak from a
band-limited textured volume is, to a very good approximation. This is
what gets the v1 pipeline to the ~0.1 voxel accuracy target without an
iterative IC-GN refinement (plan §2).

Output convention matches :func:`mamba_dvc.core.ncc.peak_displacement`:
the returned fractional offset is **added** to the integer
displacement, producing a final displacement in voxel units along
``(dz, dy, dx)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float32, Int64

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = ["gaussian_subvoxel_fit"]


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``."""
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def _gather_neighbors(
    correlation: Float32[np.ndarray, "batch w w w"],
    integer_peak: Int64[np.ndarray, "batch 3"],
    axis: int,
) -> tuple[
    Float32[np.ndarray, "batch"],
    Float32[np.ndarray, "batch"],
    Float32[np.ndarray, "batch"],
]:
    """Return correlation samples at ``i-1``, ``i``, ``i+1`` along ``axis``.

    Indices wrap modulo the axis length, matching the cyclic convention
    of :func:`mamba_dvc.core.ncc.correlate`. The integer peak index per
    batch is interpreted as the *unsigned* index into the correlation
    volume (i.e., ``integer_peak`` is what
    :func:`mamba_dvc.core.ncc.peak_displacement` would return *before*
    its negative-lag wrap, but we accept the wrapped form and undo the
    wrap implicitly via modular arithmetic).
    """
    xp = _array_module(correlation)
    batch, wz, wy, wx = correlation.shape
    sizes = (wz, wy, wx)
    axis_size = sizes[axis]

    # Recover the *unsigned* index from the signed displacement: a
    # displacement in [-w/2, w/2) is congruent modulo w to an index in
    # [0, w). Apply the wrap to all three axes; we only refine along
    # `axis` but indexing requires all of them.
    iz = integer_peak[:, 0] % wz
    iy = integer_peak[:, 1] % wy
    ix = integer_peak[:, 2] % wx

    if axis == 0:
        center_idx = iz
    elif axis == 1:
        center_idx = iy
    elif axis == 2:
        center_idx = ix
    else:  # pragma: no cover - guarded at the public entry point
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    minus = (center_idx - 1) % axis_size
    plus = (center_idx + 1) % axis_size

    batch_idx = xp.arange(batch, dtype=xp.int64)

    if axis == 0:
        c_minus = correlation[batch_idx, minus, iy, ix]
        c_zero = correlation[batch_idx, iz, iy, ix]
        c_plus = correlation[batch_idx, plus, iy, ix]
    elif axis == 1:
        c_minus = correlation[batch_idx, iz, minus, ix]
        c_zero = correlation[batch_idx, iz, iy, ix]
        c_plus = correlation[batch_idx, iz, plus, ix]
    else:
        c_minus = correlation[batch_idx, iz, iy, minus]
        c_zero = correlation[batch_idx, iz, iy, ix]
        c_plus = correlation[batch_idx, iz, iy, plus]

    return (
        c_minus.astype(xp.float32),
        c_zero.astype(xp.float32),
        c_plus.astype(xp.float32),
    )


def _log_parabola_offset(
    c_minus: Float32[np.ndarray, "batch"],
    c_zero: Float32[np.ndarray, "batch"],
    c_plus: Float32[np.ndarray, "batch"],
    *,
    eps: float = 1e-12,
) -> Float32[np.ndarray, "batch"]:
    """Return the closed-form log-parabolic peak offset in ``(-0.5, 0.5)``.

    The fit assumes the three samples are positive (the NCC peak lobe
    is positive at and adjacent to the maximum). When any sample is
    non-positive — possible for very low-contrast subvolumes near the
    masked region — the log is undefined; we fall back to a plain
    parabolic fit on the raw values, which is the de-facto standard
    fallback in DIC literature for low-confidence peaks.

    The denominator is clamped with ``eps`` to defend against the
    degenerate case where all three samples are equal (a flat peak),
    which would otherwise produce a 0 / 0.
    """
    xp = _array_module(c_zero)

    safe = (c_minus > 0.0) & (c_zero > 0.0) & (c_plus > 0.0)

    log_minus = xp.log(xp.where(safe, c_minus, xp.float32(1.0)))
    log_zero = xp.log(xp.where(safe, c_zero, xp.float32(1.0)))
    log_plus = xp.log(xp.where(safe, c_plus, xp.float32(1.0)))

    # Standard log-parabolic peak formula, derived by fitting
    # log y = a (x - x0)^2 + b through (-1, log_minus), (0, log_zero), (1, log_plus):
    #   x0 = (log_minus - log_plus) / (2 * (log_minus - 2 log_zero + log_plus))
    log_num = log_minus - log_plus
    log_den = 2.0 * (log_minus - 2.0 * log_zero + log_plus)
    log_offset = log_num / xp.where(xp.abs(log_den) < eps, xp.float32(eps), log_den)

    # Plain-parabolic fallback in linear space.
    lin_num = c_minus - c_plus
    lin_den = 2.0 * (c_minus - 2.0 * c_zero + c_plus)
    lin_offset = lin_num / xp.where(xp.abs(lin_den) < eps, xp.float32(eps), lin_den)

    offset = xp.where(safe, log_offset, lin_offset)

    # Clamp to (-0.5, 0.5). Values outside that range mean the integer
    # peak was wrong (the true peak lies in the next bin); clamping
    # leaves the integer choice authoritative — the outlier filter
    # downstream will catch the anomaly via low confidence.
    return xp.clip(offset, xp.float32(-0.5), xp.float32(0.5)).astype(xp.float32)


def gaussian_subvoxel_fit(
    correlation: Float32[np.ndarray, "batch w w w"],
    integer_displacement: Int64[np.ndarray, "batch 3"],
) -> Float32[np.ndarray, "batch 3"]:
    """Refine integer-lag peak indices to fractional voxel offsets.

    Parameters
    ----------
    correlation
        ``(batch, wz, wy, wx)`` float32 correlation volume from
        :func:`mamba_dvc.core.ncc.correlate`.
    integer_displacement
        ``(batch, 3)`` int64 ``(dz, dy, dx)`` integer-voxel
        displacements as returned by
        :func:`mamba_dvc.core.ncc.peak_displacement` (signed, wrapped
        to ``[-w/2, w/2)``). The fit interprets each entry modulo the
        axis length, so the wrap is undone implicitly — callers do not
        need to reverse it.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, 3)`` float32 fractional ``(δz, δy, δx)`` offsets in
        ``[-0.5, 0.5]``. The final per-POI displacement in voxel units
        is ``integer_displacement.astype(float32) + return_value``.

    Raises
    ------
    ValueError
        For malformed inputs: wrong ndim, wrong dtype, batch
        mismatch, or mixed host/device arrays.

    Notes
    -----
    Fits a 1D log-parabola through the three samples
    ``corr[i-1, j, k], corr[i, j, k], corr[i+1, j, k]`` along each axis
    independently. The 3D Gaussian peak that arises from a separable
    window applied to a band-limited reference is itself
    axis-separable, so the per-axis 1D fit is exact for the model and
    competitive with a full 3D fit at a fraction of the cost.

    Indices ``i ± 1`` wrap cyclically around the correlation volume,
    consistent with the cyclic convention of
    :func:`mamba_dvc.core.ncc.correlate`. POIs whose true subvoxel
    offset exceeds half a voxel — which would mean the integer-peak
    detector picked the wrong bin — are clamped to the bin boundary
    rather than allowed to extrapolate; the outlier filter downstream
    (plan §2 step 7) is the right place to flag those.
    """
    if correlation.ndim != 4:
        raise ValueError(
            f"correlation must be 4D (batch, wz, wy, wx), got ndim={correlation.ndim}"
        )
    if correlation.dtype != np.float32:
        raise ValueError(f"correlation must be float32, got {correlation.dtype}")
    if integer_displacement.ndim != 2 or integer_displacement.shape[1] != 3:
        raise ValueError(
            f"integer_displacement must have shape (batch, 3), "
            f"got {integer_displacement.shape}"
        )
    if integer_displacement.dtype != np.int64:
        raise ValueError(
            f"integer_displacement must be int64, got {integer_displacement.dtype}"
        )
    if integer_displacement.shape[0] != correlation.shape[0]:
        raise ValueError(
            f"batch size mismatch: correlation has {correlation.shape[0]}, "
            f"integer_displacement has {integer_displacement.shape[0]}"
        )

    xp = _array_module(correlation)
    if _array_module(integer_displacement) is not xp:
        raise ValueError(
            "correlation and integer_displacement must live in the same array module "
            "(both NumPy or both CuPy); no implicit transfer is performed"
        )

    offsets = xp.empty(integer_displacement.shape, dtype=xp.float32)
    for axis in range(3):
        c_minus, c_zero, c_plus = _gather_neighbors(correlation, integer_displacement, axis)
        offsets[:, axis] = _log_parabola_offset(c_minus, c_zero, c_plus)

    return offsets
