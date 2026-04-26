"""Per-POI subvolume preprocessing prior to batched FFT NCC.

Implements step 3 of the DVC pipeline (plan
``docs/plans/overview.md`` §2): for each subvolume in a ``(batch, wz, wy, wx)``
tensor, this module

1. subtracts the mean computed over valid (unmasked) voxels only,
2. multiplies the centered subvolume by a separable 3D Tukey window to
   suppress spectral leakage from the implicit periodic continuation
   that the FFT-based cross-correlation assumes,
3. zeroes out masked voxels so they contribute nothing to the NCC's
   valid-support normalization in step 4.

The mask input is optional. When no mask is supplied, every voxel is
treated as valid: the plain subvolume mean is subtracted, the Tukey
window is still applied, and no post-window zeroing is needed. This
matches the signature symmetry with :func:`mamba_dvc.core.extract.extract_subvolumes`
— callers that load volumes without a mask get the same preprocessing
pipeline as the masked case.

Design mirrors :mod:`mamba_dvc.core.extract`:

- Duck-typed across NumPy and CuPy via ``cupy.get_array_module``.
- Pure function; no state, no mutation of inputs.
- Float32 C-contiguous output, matching cuFFT's layout requirements
  for the next stage (:mod:`mamba_dvc.core.ncc`).

The Tukey builder is exposed as :func:`tukey_window_3d` so callers that
drive the pipeline in a hot loop can precompute and cache the window
across batches. Building the window each call is cheap (a few thousand
voxels for a ``96**3`` window) but avoidable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Bool, Float32

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = ["preprocess_subvolumes", "tukey_window_1d", "tukey_window_3d"]


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``."""
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def tukey_window_1d(
    n: int,
    alpha: float,
    *,
    xp: Any = np,
) -> Float32[np.ndarray, "w"]:
    """Return a length-``n`` Tukey window with taper fraction ``alpha``.

    Parameters
    ----------
    n
        Number of samples. Must be positive.
    alpha
        Fraction of the window that is tapered, split equally between
        the two ends. ``alpha == 0`` gives a rectangular window;
        ``alpha == 1`` gives a Hann window. Must lie in ``[0, 1]``.
    xp
        Array module. Pass ``cupy`` (or the value returned by
        ``cupy.get_array_module`` on a device array) to build the
        window directly on GPU.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(n,)`` float32 window. Symmetric about the midpoint; endpoints
        equal ``0`` whenever ``alpha > 0`` and ``n > 1``.

    Notes
    -----
    For ``x = i / (n - 1)`` in ``[0, 1]`` the window is

    ``w(x) = 0.5 * (1 + cos(2*pi/alpha * (x - alpha/2)))`` on the left
    taper ``0 <= x < alpha/2``, ``w(x) = 1`` on the flat interior
    ``[alpha/2, 1 - alpha/2]``, and the right taper is the mirror
    image. The taper is a raised cosine — continuous in value and
    first derivative at the taper-to-flat transition, which is what
    keeps the spectral leakage suppression clean without widening the
    mainlobe more than necessary.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if n == 1:
        return xp.ones(1, dtype=xp.float32)
    if alpha == 0.0:
        return xp.ones(n, dtype=xp.float32)

    x = xp.arange(n, dtype=xp.float64) / (n - 1)
    two_pi_over_alpha = 2.0 * np.pi / alpha

    left_taper = 0.5 * (1.0 + xp.cos(two_pi_over_alpha * (x - alpha / 2.0)))
    right_taper = 0.5 * (1.0 + xp.cos(two_pi_over_alpha * (x - 1.0 + alpha / 2.0)))

    in_flat = (x >= alpha / 2.0) & (x <= 1.0 - alpha / 2.0)
    in_right = x > 1.0 - alpha / 2.0

    w = xp.where(in_flat, xp.float64(1.0), left_taper)
    w = xp.where(in_right, right_taper, w)
    return w.astype(xp.float32)


def tukey_window_3d(
    shape: tuple[int, int, int],
    alpha: float,
    *,
    xp: Any = np,
) -> Float32[np.ndarray, "w w w"]:
    """Return a separable 3D Tukey window of the given shape.

    Parameters
    ----------
    shape
        ``(wz, wy, wx)`` window extent per axis.
    alpha
        Taper fraction; see :func:`tukey_window_1d`. The same
        ``alpha`` is used along each axis, matching plan §2's
        convention.
    xp
        Array module.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(wz, wy, wx)`` float32 window, equal to the outer product of
        three 1D Tukey windows. Separability is a mathematical
        consequence of applying an independent raised-cosine taper per
        axis — there is no approximation involved.

    Notes
    -----
    The window is *not* normalized. Callers that need a unit L2 norm
    (e.g., for a specific normalization convention in a v2 NCC
    variant) should scale the return value themselves; the default NCC
    in :mod:`mamba_dvc.core.ncc` absorbs the window's energy into the
    per-subvolume L2 normalization computed over the valid-voxel
    support, so an unnormalized window is the correct input.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have length 3, got {len(shape)}")
    wz, wy, wx = (int(s) for s in shape)
    if wz <= 0 or wy <= 0 or wx <= 0:
        raise ValueError(f"shape entries must be positive, got {shape}")

    w_z = tukey_window_1d(wz, alpha, xp=xp)
    w_y = tukey_window_1d(wy, alpha, xp=xp)
    w_x = tukey_window_1d(wx, alpha, xp=xp)

    return (w_z[:, None, None] * w_y[None, :, None] * w_x[None, None, :]).astype(xp.float32)


def preprocess_subvolumes(
    subvolumes: Float32[np.ndarray, "batch w w w"],
    mask_subvolumes: Bool[np.ndarray, "batch w w w"] | None = None,
    tukey_alpha: float = 0.25,
) -> Float32[np.ndarray, "batch w w w"]:
    """Masked mean-subtract → Tukey window → masked zeroing.

    Parameters
    ----------
    subvolumes
        ``(batch, wz, wy, wx)`` float32 subvolumes from
        :func:`mamba_dvc.core.extract.extract_subvolumes`. May be NumPy
        (host) or CuPy (device); output uses the same array module.
    mask_subvolumes
        ``(batch, wz, wy, wx)`` boolean mask subvolumes, one per
        corresponding entry in ``subvolumes``. ``True`` marks valid
        voxels (bone tissue); ``False`` marks excluded voxels (screw,
        artifacts). Must live in the same array module as
        ``subvolumes``. When ``None``, every voxel is treated as
        valid — the plain mean is subtracted and no post-window
        zeroing is applied.
    tukey_alpha
        Taper fraction of the 3D Tukey window; see
        :func:`tukey_window_1d`. Default ``0.25`` matches plan §2.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 C-contiguous preprocessed
        subvolumes. Array module matches ``subvolumes``.

    Raises
    ------
    ValueError
        For malformed inputs: wrong ndim, wrong dtype, mask shape
        mismatch, mixed host/device arrays, or ``tukey_alpha`` out of
        ``[0, 1]``.

    Notes
    -----
    Operation order matters. Subtracting the masked mean **before** the
    window ensures that the windowed signal tapers a zero-mean signal
    to zero at every face — which is the only state in which the FFT's
    periodic continuation is actually continuous. Windowing before
    mean-subtracting would attenuate a constant DC offset non-uniformly
    and introduce a new low-frequency artifact.

    Subvolumes with *zero* valid voxels would produce a division by
    zero in the mean computation. We clamp the denominator to 1 in
    that degenerate case; the final output for those subvolumes is
    identically zero anyway, because every voxel is zeroed in step 3.
    Upstream, :func:`mamba_dvc.core.grid.filter_by_mask` should have
    already rejected such POIs via the ``mask_threshold`` admission
    rule (plan §3), so this clamp only exists as a safety net.
    """
    if subvolumes.ndim != 4:
        raise ValueError(
            f"subvolumes must be 4D (batch, wz, wy, wx), got ndim={subvolumes.ndim}"
        )
    if subvolumes.dtype != np.float32:
        raise ValueError(f"subvolumes must be float32, got {subvolumes.dtype}")
    if not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError(f"tukey_alpha must be in [0, 1], got {tukey_alpha}")

    xp = _array_module(subvolumes)
    _, wz, wy, wx = subvolumes.shape

    if mask_subvolumes is not None:
        if mask_subvolumes.shape != subvolumes.shape:
            raise ValueError(
                f"mask_subvolumes shape {mask_subvolumes.shape} does not match "
                f"subvolumes shape {subvolumes.shape}"
            )
        if mask_subvolumes.dtype != np.bool_:
            raise ValueError(
                f"mask_subvolumes must have bool dtype, got {mask_subvolumes.dtype}"
            )
        if _array_module(mask_subvolumes) is not xp:
            raise ValueError(
                "subvolumes and mask_subvolumes must live in the same array module "
                "(both NumPy or both CuPy); no implicit transfer is performed"
            )

        mask_f = mask_subvolumes.astype(xp.float32)
        valid_counts = mask_f.sum(axis=(1, 2, 3), keepdims=True)
        safe_counts = xp.maximum(valid_counts, xp.float32(1.0))
        means = (subvolumes * mask_f).sum(axis=(1, 2, 3), keepdims=True) / safe_counts
        centered = subvolumes - means.astype(xp.float32)
    else:
        mask_f = None
        means = subvolumes.mean(axis=(1, 2, 3), keepdims=True)
        centered = subvolumes - means.astype(xp.float32)

    window = tukey_window_3d((wz, wy, wx), tukey_alpha, xp=xp)
    windowed = centered * window[None, :, :, :]

    if mask_f is not None:
        windowed = windowed * mask_f

    return xp.ascontiguousarray(windowed, dtype=xp.float32)
