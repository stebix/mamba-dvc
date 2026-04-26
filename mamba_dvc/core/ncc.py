"""Batched FFT normalized cross-correlation and integer peak localization.

Implements steps 4-5 of the single-pass DVC pipeline (plan
``docs/plans/overview.md`` §2 and ``docs/plans/ncc.md``): one batched
forward ``rfftn`` per stack, conjugate multiply, batched ``irfftn``,
followed by per-POI ``argmax`` with the lag wrapped into
``[-w/2, w/2)``.

Design choices (plan §2 and ``docs/plans/ncc.md``):

- Two pure verbs: :func:`correlate` (returns the full normalized
  correlation map) and :func:`peak_displacement` (argmax + circular
  wrap). Splitting them keeps :func:`correlate` reusable for v2's
  Padfield masked FFT NCC, and lets :mod:`mamba_dvc.core.peakfit`
  consume the correlation map directly.
- Whole-window L2 normalization. ``mamba_dvc.core.window`` already
  zeroed masked voxels, so the L2 norm summed over the full window is
  identical to the norm summed over the valid support. The mask is
  consulted nowhere here.
- Real-to-complex transforms via ``rfftn``/``irfftn``. The inputs are
  real float32 and cuFFT's R2C plan cuts spectral memory roughly in
  half along the last axis.
- Cyclic correlation, no zero padding. The plan caps the displacement
  budget at ``|d| ≤ w/3``; aliasing does not occur.
- Duck-typed across NumPy and CuPy via ``cupy.get_array_module``. No
  implicit host/device transfer; both inputs must agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float32, Int64

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = ["correlate", "peak_displacement"]


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``."""
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def correlate(
    reference: Float32[np.ndarray, "batch w w w"],
    deformed: Float32[np.ndarray, "batch w w w"],
    *,
    eps: float = 1e-12,
) -> Float32[np.ndarray, "batch w w w"]:
    """Return the per-POI normalized cross-correlation volume.

    Parameters
    ----------
    reference, deformed
        ``(batch, wz, wy, wx)`` float32 subvolumes from
        :func:`mamba_dvc.core.window.preprocess_subvolumes`. Both must
        live in the same array module (NumPy or CuPy) and have
        identical shape.
    eps
        Small floor added to the per-POI L2-norm product before the
        square root, so an entirely zero subvolume produces a zero (not
        NaN) correlation map. The valid-support admission rule in
        :func:`mamba_dvc.core.grid.filter_by_mask` should already have
        rejected such POIs upstream; ``eps`` is the safety net.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 C-contiguous correlation map.
        Lag ``(0, 0, 0)`` is at index ``(0, 0, 0)``; positive lags
        occupy indices ``[0, w/2)``; negative lags occupy ``[w/2, w)``
        in cyclic-FFT order.

    Raises
    ------
    ValueError
        For malformed inputs: wrong ndim, wrong dtype, shape mismatch,
        mixed host/device arrays, or non-positive ``eps``.

    Notes
    -----
    Implements the cyclic cross-correlation theorem

    ``corr = irfftn( conj(rfftn(reference)) * rfftn(deformed) )``

    with per-POI normalization by ``sqrt(‖f‖² · ‖g‖² + eps)``. The
    convention "lag = deformed - reference" makes the recovered peak
    coincide with the displacement of the deformed volume relative to
    the reference, matching the sign used by
    :mod:`mamba_dvc.validate.synthetic`.
    """
    if reference.ndim != 4:
        raise ValueError(
            f"reference must be 4D (batch, wz, wy, wx), got ndim={reference.ndim}"
        )
    if deformed.ndim != 4:
        raise ValueError(f"deformed must be 4D (batch, wz, wy, wx), got ndim={deformed.ndim}")
    if reference.shape != deformed.shape:
        raise ValueError(
            f"reference shape {reference.shape} does not match deformed shape {deformed.shape}"
        )
    if reference.dtype != np.float32 or deformed.dtype != np.float32:
        raise ValueError(
            f"reference and deformed must be float32, "
            f"got {reference.dtype} and {deformed.dtype}"
        )
    if not eps > 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    xp = _array_module(reference)
    if _array_module(deformed) is not xp:
        raise ValueError(
            "reference and deformed must live in the same array module "
            "(both NumPy or both CuPy); no implicit transfer is performed"
        )

    fft_axes = (1, 2, 3)
    spectrum_ref = xp.fft.rfftn(reference, axes=fft_axes)
    spectrum_def = xp.fft.rfftn(deformed, axes=fft_axes)

    cross_power = xp.conj(spectrum_ref) * spectrum_def
    corr = xp.fft.irfftn(cross_power, s=reference.shape[1:], axes=fft_axes)

    # L2 norms over the full window. Preprocessing has already zeroed
    # masked voxels, so this equals the norm over the valid support.
    ref_sq = (reference.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
    def_sq = (deformed.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
    norms = xp.sqrt(ref_sq * def_sq + xp.float64(eps))
    norms_f32 = norms.astype(xp.float32)

    corr_normalized = corr.astype(xp.float32) / norms_f32[:, None, None, None]
    return xp.ascontiguousarray(corr_normalized, dtype=xp.float32)


def peak_displacement(
    correlation: Float32[np.ndarray, "batch w w w"],
) -> tuple[Int64[np.ndarray, "batch 3"], Float32[np.ndarray, "batch"]]:
    """Return the integer-lag peak per POI plus the peak NCC value.

    Parameters
    ----------
    correlation
        ``(batch, wz, wy, wx)`` correlation volume from
        :func:`correlate`. May be NumPy (host) or CuPy (device).

    Returns
    -------
    displacement
        ``(batch, 3)`` int64 ``(dz, dy, dx)`` integer-voxel
        displacements. Each axis is wrapped to the signed range
        ``[-w/2, w/2)``: indices in ``[0, w/2)`` are returned as-is;
        indices in ``[w/2, w)`` are returned as ``i - w``.
    peak_value
        ``(batch,)`` float32 NCC value at the integer peak. Used as the
        ``confidence`` channel of the final
        :class:`mamba_dvc.types.DisplacementField`.

    Raises
    ------
    ValueError
        If ``correlation`` is not 4D float32.

    Notes
    -----
    The wrap convention assumes ``correlate`` produced the correlation
    map via ``rfftn``/``irfftn`` without zero padding, so lag ``0`` is
    at index ``0`` and the half-window symmetry is exact for even
    window sizes (the only case used in v1 — defaults are ``w = 96``
    and ``w = 128``).
    """
    if correlation.ndim != 4:
        raise ValueError(
            f"correlation must be 4D (batch, wz, wy, wx), got ndim={correlation.ndim}"
        )
    if correlation.dtype != np.float32:
        raise ValueError(f"correlation must be float32, got {correlation.dtype}")

    xp = _array_module(correlation)
    batch, wz, wy, wx = correlation.shape

    flat = correlation.reshape(batch, wz * wy * wx)
    flat_idx = flat.argmax(axis=1)
    peak_value = flat.max(axis=1).astype(xp.float32)

    iz, rem = xp.divmod(flat_idx, wy * wx)
    iy, ix = xp.divmod(rem, wx)

    half_z, half_y, half_x = wz // 2, wy // 2, wx // 2
    dz = xp.where(iz < half_z, iz, iz - wz)
    dy = xp.where(iy < half_y, iy, iy - wy)
    dx = xp.where(ix < half_x, ix, ix - wx)

    displacement = xp.stack([dz, dy, dx], axis=1).astype(xp.int64)
    return displacement, peak_value
