"""Batched FFT normalized cross-correlation with cyclic and linear modes.

Implements steps 4-5 of the single-pass DVC pipeline (plan
``docs/plans/overview.md`` §2 and ``docs/plans/ncc.md``).

Two kernels live side-by-side:

- :func:`correlate_cyclic` -- the original v0 kernel: one batched
  ``rfftn`` per stack, conjugate-multiply, batched ``irfftn`` at window
  size ``W``, with a single whole-window L2 normalization per POI. Fast
  but exhibits a systematic shrinkage bias for non-zero shifts (see
  ``docs/insights/error-minimization.md``).
- :func:`correlate_linear` -- zero-pads each subvolume to ``2W`` along
  every spatial axis, performs the cross-correlation in the padded
  domain, then crops the central ``W`` block of valid lags and repacks
  it back into cyclic-FFT layout (lag 0 at index 0). This converts
  cyclic correlation into linear correlation, eliminating the
  wrap-around contribution to the peak shape. Optionally pairs with a
  Lewis (1995) per-lag overlap-aware denominator (``normalization="overlap"``)
  that uses two extra FFT-based correlations of ``ref²`` and ``def²``
  against an indicator box, so the denominator tracks the true overlap
  size at each lag.

:func:`correlate` is a thin dispatcher that requires the caller to
specify both ``mode`` and ``normalization``. There are intentionally no
defaults at the core level: the choice is load-bearing for accuracy,
and we want the call site to record it explicitly. The pipeline driver
in :mod:`mamba_dvc.pipeline.correlate` supplies the defaults at its
own layer.

Layout note
-----------
Both kernels return a ``(batch, wz, wy, wx)`` correlation map in the
cyclic-FFT layout that :func:`peak_displacement` and
:mod:`mamba_dvc.core.peakfit` already consume: lag ``(0, 0, 0)`` lives
at index ``(0, 0, 0)``; positive lags fill ``[0, w/2)``; negative lags
fill ``[w/2, w)``. The linear kernel achieves this by cropping the
central block of the ``2W``-FFT output and re-shifting -- so downstream
code is unchanged. A future PR may switch the linear kernel to a
centered layout (lag 0 at the centre) and update ``peakfit`` /
``peak_displacement`` accordingly; that is listed as a v2 cleanup in
``docs/plans/overview.md`` §10.

Memory footprint
----------------
The linear kernel allocates several ``(batch, 2W, 2W, 2W)`` float32
buffers; for ``W=96`` that is ~28 MB per buffer per POI, scaling
linearly with ``batch``. At the production batch size of 256 this puts
the linear+overlap path well above 50 GB transient on a single A6000;
expect to pick a smaller batch size (~64) when running this mode at
``W=96``. A future preprocessing heuristic in
:mod:`mamba_dvc.gpu.dispatch` will recommend batch sizes
programmatically.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from jaxtyping import Float32, Int64

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = [
    "NCCMode",
    "NCCNormalization",
    "correlate",
    "correlate_cyclic",
    "correlate_linear",
    "peak_displacement",
]


class NCCMode(StrEnum):
    """FFT-NCC kernel selection."""

    CYCLIC = "cyclic"
    LINEAR = "linear"


class NCCNormalization(StrEnum):
    """Per-POI denominator convention for the FFT-NCC kernel."""

    GLOBAL = "global"
    OVERLAP = "overlap"


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``."""
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def _validate_pair(
    reference: Float32[np.ndarray, "batch w w w"],
    deformed: Float32[np.ndarray, "batch w w w"],
    eps: float,
) -> Any:
    """Shared validation; returns the array module of the inputs."""
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
    return xp


def correlate_cyclic(
    reference: Float32[np.ndarray, "batch w w w"],
    deformed: Float32[np.ndarray, "batch w w w"],
    *,
    eps: float = 1e-12,
) -> Float32[np.ndarray, "batch w w w"]:
    """Cyclic FFT NCC at window size with whole-window L2 normalization.

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
        NaN) correlation map.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 C-contiguous correlation map in
        cyclic-FFT layout: lag ``(0, 0, 0)`` is at index ``(0, 0, 0)``;
        positive lags occupy ``[0, w/2)``; negative lags occupy
        ``[w/2, w)``.

    Notes
    -----
    Implements ``corr = irfftn(conj(rfftn(ref)) * rfftn(def))`` with
    per-POI normalization by ``sqrt(‖f‖² · ‖g‖² + eps)``. The
    convention "lag = deformed - reference" makes the recovered peak
    coincide with the displacement of the deformed volume relative to
    the reference, matching the sign used by
    :mod:`mamba_dvc.validate.synthetic`.

    The cyclic kernel is biased toward zero for non-zero shifts: at
    true lag ``u`` the cyclic sum pairs ``W - |u|`` voxels correctly and
    ``|u|`` voxels via wrap-around, so the peak shape is skewed and the
    Gaussian sub-voxel fit pulls the recovered displacement toward
    ``0``. The bias scales linearly with ``|u|/W``. See
    ``docs/insights/error-minimization.md`` for the full derivation;
    use :func:`correlate_linear` for unbiased estimation.
    """
    xp = _validate_pair(reference, deformed, eps)

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


def _crop_padded_to_cyclic(
    arr_padded: Any,
    window: tuple[int, int, int],
    fft_axes: tuple[int, int, int],
    xp: Any,
) -> Any:
    """Crop the central ``W`` block of valid lags from a ``2W`` map.

    The output is repacked to cyclic-FFT layout (lag 0 at index 0) so
    that downstream consumers (:func:`peak_displacement`,
    :mod:`mamba_dvc.core.peakfit`) treat it identically to a cyclic-W
    correlation map.
    """
    wz, wy, wx = window
    shifted = xp.fft.fftshift(arr_padded, axes=fft_axes)
    center = shifted[
        :,
        wz // 2 : wz // 2 + wz,
        wy // 2 : wy // 2 + wy,
        wx // 2 : wx // 2 + wx,
    ]
    return xp.fft.ifftshift(center, axes=fft_axes)


def correlate_linear(
    reference: Float32[np.ndarray, "batch w w w"],
    deformed: Float32[np.ndarray, "batch w w w"],
    *,
    normalization: NCCNormalization,
    eps: float = 1e-12,
) -> Float32[np.ndarray, "batch w w w"]:
    """Zero-padded linear FFT NCC; output repacked to cyclic-FFT layout.

    Parameters
    ----------
    reference, deformed
        ``(batch, wz, wy, wx)`` float32 subvolumes. Same constraints as
        :func:`correlate_cyclic`.
    normalization
        ``"overlap"`` (Lewis 1995): per-lag denominator
        ``sqrt(S_ref(k) · S_def(k))`` where ``S(k)`` is the sum of
        squared signal over the in-bounds intersection at lag ``k``.
        Computed via two extra FFT-based correlations of ``ref²`` and
        ``def²`` against an indicator box. ``"global"``: same
        whole-window L2 denominator as :func:`correlate_cyclic`, kept
        as an A/B knob to isolate the contribution of each fix.
    eps
        Small floor added to the denominator before the square root.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 C-contiguous correlation map in
        cyclic-FFT layout (lag 0 at index 0; positive lags in
        ``[0, w/2)``; negative lags in ``[w/2, w)``). Drop-in compatible
        with :func:`peak_displacement` and
        :func:`mamba_dvc.core.peakfit.gaussian_subvoxel_fit`.

    Notes
    -----
    Linear correlation eliminates the cyclic wrap-around. Combined with
    the overlap-aware Lewis denominator it removes the ``|u|/W``
    shrinkage bias that the cyclic kernel exhibits.

    Memory: the kernel allocates several ``(batch, 2*wz, 2*wy, 2*wx)``
    float32 buffers (~8x the cyclic kernel's transient footprint).
    Tune ``batch_size`` accordingly when invoking from the pipeline at
    production window sizes.
    """
    xp = _validate_pair(reference, deformed, eps)
    try:
        normalization = NCCNormalization(normalization)
    except ValueError as exc:
        raise ValueError(
            f"normalization must be one of {[m.value for m in NCCNormalization]}, "
            f"got {normalization!r}"
        ) from exc

    batch, wz, wy, wx = reference.shape
    window = (wz, wy, wx)
    padded_shape = (2 * wz, 2 * wy, 2 * wx)
    fft_axes = (1, 2, 3)

    ref_padded = xp.zeros((batch, *padded_shape), dtype=xp.float32)
    ref_padded[:, :wz, :wy, :wx] = reference
    def_padded = xp.zeros((batch, *padded_shape), dtype=xp.float32)
    def_padded[:, :wz, :wy, :wx] = deformed

    spectrum_ref = xp.fft.rfftn(ref_padded, axes=fft_axes)
    spectrum_def = xp.fft.rfftn(def_padded, axes=fft_axes)
    cross_power = xp.conj(spectrum_ref) * spectrum_def
    numerator_full = xp.fft.irfftn(cross_power, s=padded_shape, axes=fft_axes)
    numerator = _crop_padded_to_cyclic(numerator_full, window, fft_axes, xp)

    if normalization is NCCNormalization.GLOBAL:
        ref_sq = (reference.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
        def_sq = (deformed.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
        denom = xp.sqrt(ref_sq * def_sq + xp.float64(eps)).astype(xp.float32)
        corr = numerator / denom[:, None, None, None]
    else:  # NCCNormalization.OVERLAP
        ref_sq_padded = ref_padded * ref_padded
        def_sq_padded = def_padded * def_padded
        spectrum_ref_sq = xp.fft.rfftn(ref_sq_padded, axes=fft_axes)
        spectrum_def_sq = xp.fft.rfftn(def_sq_padded, axes=fft_axes)

        # Indicator box of size W within the 2W cube. One copy is
        # shared across the batch (rfft of an unbatched array).
        box = xp.zeros(padded_shape, dtype=xp.float32)
        box[:wz, :wy, :wx] = xp.float32(1.0)
        spectrum_box = xp.fft.rfftn(box)

        s_ref_sq_full = xp.fft.irfftn(
            xp.conj(spectrum_ref_sq) * spectrum_box,
            s=padded_shape,
            axes=fft_axes,
        )
        s_def_sq_full = xp.fft.irfftn(
            xp.conj(spectrum_box) * spectrum_def_sq,
            s=padded_shape,
            axes=fft_axes,
        )

        # Sums-of-squares are non-negative analytically; FFT round-off
        # can produce small negative values that would NaN the sqrt.
        s_ref_sq = xp.maximum(
            _crop_padded_to_cyclic(s_ref_sq_full, window, fft_axes, xp),
            xp.float32(0.0),
        )
        s_def_sq = xp.maximum(
            _crop_padded_to_cyclic(s_def_sq_full, window, fft_axes, xp),
            xp.float32(0.0),
        )

        denom = xp.sqrt(s_ref_sq * s_def_sq + xp.float32(eps))
        corr = numerator / denom

    return xp.ascontiguousarray(corr, dtype=xp.float32)


def correlate(
    reference: Float32[np.ndarray, "batch w w w"],
    deformed: Float32[np.ndarray, "batch w w w"],
    *,
    mode: NCCMode,
    normalization: NCCNormalization,
    eps: float = 1e-12,
) -> Float32[np.ndarray, "batch w w w"]:
    """Dispatch to the cyclic or linear NCC kernel.

    Both ``mode`` and ``normalization`` are required keyword arguments
    -- the choice has a measurable accuracy impact (see
    ``docs/insights/error-minimization.md``) and we want call sites to
    record it explicitly rather than picking up a default. The
    pipeline driver in :mod:`mamba_dvc.pipeline.correlate` is the
    single place that selects defaults for downstream callers.

    Parameters
    ----------
    reference, deformed
        ``(batch, wz, wy, wx)`` float32 preprocessed subvolumes.
    mode
        ``"cyclic"`` to use :func:`correlate_cyclic`, ``"linear"`` to
        use :func:`correlate_linear`.
    normalization
        ``"global"`` for whole-window L2; ``"overlap"`` for the
        per-lag Lewis denominator. Only meaningful in ``mode="linear"``;
        for ``mode="cyclic"`` the value must be ``"global"``.
    eps
        Denominator floor.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 correlation map in cyclic-FFT
        layout (see module docstring).

    Raises
    ------
    ValueError
        For malformed inputs (wrong ndim, dtype, mismatched shape,
        mixed host/device arrays, non-positive ``eps``), unknown
        ``mode`` or ``normalization`` values, or the unsupported
        combination ``mode="cyclic"`` with ``normalization="overlap"``.
    """
    try:
        mode = NCCMode(mode)
    except ValueError as exc:
        raise ValueError(
            f"mode must be one of {[m.value for m in NCCMode]}, got {mode!r}"
        ) from exc
    try:
        normalization = NCCNormalization(normalization)
    except ValueError as exc:
        raise ValueError(
            f"normalization must be one of {[m.value for m in NCCNormalization]}, "
            f"got {normalization!r}"
        ) from exc

    if mode is NCCMode.CYCLIC:
        if normalization is not NCCNormalization.GLOBAL:
            raise ValueError(
                f"mode='cyclic' supports only normalization='global', "
                f"got normalization={normalization.value!r}; use mode='linear' "
                f"for overlap-aware Lewis normalization"
            )
        return correlate_cyclic(reference, deformed, eps=eps)
    return correlate_linear(reference, deformed, normalization=normalization, eps=eps)


def peak_displacement(
    correlation: Float32[np.ndarray, "batch w w w"],
) -> tuple[Int64[np.ndarray, "batch 3"], Float32[np.ndarray, "batch"]]:
    """Return the integer-lag peak per POI plus the peak NCC value.

    Parameters
    ----------
    correlation
        ``(batch, wz, wy, wx)`` correlation volume from either
        :func:`correlate_cyclic` or :func:`correlate_linear` (both
        return the same cyclic-FFT layout). May be NumPy (host) or
        CuPy (device).

    Returns
    -------
    displacement
        ``(batch, 3)`` int64 ``(dz, dy, dx)`` integer-voxel
        displacements. Each axis is wrapped to the signed range
        ``[-w/2, w/2)``: indices in ``[0, w/2)`` are returned as-is;
        indices in ``[w/2, w)`` are returned as ``i - w``.
    peak_value
        ``(batch,)`` float32 NCC value at the integer peak.

    Raises
    ------
    ValueError
        If ``correlation`` is not 4D float32.
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
