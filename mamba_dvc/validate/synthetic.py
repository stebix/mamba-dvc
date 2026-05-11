"""Analytical synthetic pairs for DVC validation.

This module is the test-bed scaffolding called out in plan
``docs/plans/overview.md`` §9, tier 1. It emits:

1. a textured reference volume (band-limited noise),
2. a closed-form displacement field callable at arbitrary coordinates,
3. a deformed volume produced by resampling the reference under the
   field.

The three are decoupled on purpose — POI-filter tests only need the
reference, correlator tests need all three. Everything is host-side
NumPy; no CuPy.

Warp convention
---------------
The default DIC/DVC convention is **pull-back**:
``deformed(x) = reference(x - u(x))``. ``correlate()`` recovers the
same ``u`` that appears here, so test assertions can compare directly
without sign-flipping.

:func:`warp` also accepts ``convention="push_forward"``, which samples
at ``coords + u(x)`` instead. This is a cheap sign-flip — equivalent
to pull-back of ``-u`` — *not* a true forward scatter. Use it to model
inputs whose stored field follows the ``deformed(x) = reference(x + u(x))``
sign convention; remember that any displacement recovered by
``correlate()`` from such a pair will be the negation of the field
that produced it.

Coordinate convention
---------------------
Displacement fields are evaluated on ``(points, 3)`` arrays of voxel
coordinates in ``(z, y, x)`` order and return ``(points, 3)``
displacement vectors in the same order. All fields are callables rather
than pre-sampled arrays so the ground truth at any POI center is exact
— no interpolation error leaks into the expected-error budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float32
from scipy.ndimage import gaussian_filter, map_coordinates

from mamba_dvc.types import GridSpec

WarpConvention = Literal["pull_back", "push_forward"]

__all__ = [
    "DisplacementFunction",
    "SyntheticPair",
    "WarpConvention",
    "compose",
    "make_pair",
    "make_texture",
    "rigid_shift",
    "sample_on_grid",
    "simple_shear",
    "sinusoidal",
    "uniform_dilation",
    "warp",
]


@runtime_checkable
class DisplacementFunction(Protocol):
    """Closed-form displacement field, evaluable at arbitrary coordinates.

    Implementations accept a ``(points, 3)`` array of voxel coordinates in
    ``(z, y, x)`` order and return a ``(points, 3)`` displacement array in
    the same order.

    Marked ``@runtime_checkable`` so ``beartype`` can validate it when
    ``jaxtyped(beartype)`` wraps functions that accept a field.
    """

    def __call__(
        self, coords: Float32[np.ndarray, "points 3"]
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate the field at ``coords`` and return matching displacements."""
        ...


@dataclass(frozen=True)
class SyntheticPair:
    """Container for a synthesized reference/deformed pair."""

    reference: Float32[np.ndarray, "z y x"]
    deformed: Float32[np.ndarray, "z y x"]
    field: DisplacementFunction
    mask: Bool[np.ndarray, "z y x"] | None


def make_texture(
    shape: tuple[int, int, int],
    *,
    sigma: float = 1.5,
    seed: int = 0,
) -> Float32[np.ndarray, "z y x"]:
    """Generate band-limited noise texture with zero mean and unit variance.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape.
    sigma
        Gaussian low-pass sigma, in voxels. ``~1.5`` gives spectral
        content across the band the FFT correlator uses while keeping
        the texture smooth enough that cubic resampling is faithful.
    seed
        Seed for the local RNG; no global RNG state is touched.

    Returns
    -------
    numpy.ndarray
        ``(z, y, x)`` float32 texture, standardized to zero mean and
        unit variance across the full volume.

    Raises
    ------
    ValueError
        If ``shape`` does not have length 3 with positive entries, or
        if ``sigma`` is not positive.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have length 3, got {len(shape)}")
    if any(s <= 0 for s in shape):
        raise ValueError(f"shape entries must be positive, got {shape}")
    if not sigma > 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(size=shape, dtype=np.float32)
    smoothed = gaussian_filter(white, sigma=sigma, mode="reflect").astype(
        np.float32, copy=False
    )

    mean = smoothed.mean(dtype=np.float64)
    std = smoothed.std(dtype=np.float64)
    if std == 0.0:
        # Degenerate case — constant field. Return zeros rather than
        # divide by zero; tests using such a texture would fail anyway
        # because the correlator has no signal to lock onto.
        return np.zeros(shape, dtype=np.float32)
    return ((smoothed - mean) / std).astype(np.float32, copy=False)


def rigid_shift(shift: tuple[float, float, float]) -> DisplacementFunction:
    """Return a constant displacement field ``u(x) = shift``.

    Parameters
    ----------
    shift
        ``(dz, dy, dx)`` rigid translation in voxels.

    Returns
    -------
    DisplacementFunction
        Callable evaluating to ``shift`` at every coordinate.
    """
    shift_arr = np.asarray(shift, dtype=np.float32)
    if shift_arr.shape != (3,):
        raise ValueError(f"shift must have length 3, got {shift}")

    def _field(
        coords: Float32[np.ndarray, "points 3"],
    ) -> Float32[np.ndarray, "points 3"]:
        out = np.empty_like(coords, dtype=np.float32)
        out[...] = shift_arr
        return out

    return _field


def uniform_dilation(
    strain: float,
    center: tuple[float, float, float],
) -> DisplacementFunction:
    """Return a uniform dilation field ``u(x) = strain * (x - center)``.

    Parameters
    ----------
    strain
        Scalar dilation strain; the displacement at voxel offset ``r``
        from ``center`` is ``strain * r``.
    center
        ``(z, y, x)`` coordinates of the dilation center (usually the
        volume centroid).
    """
    center_arr = np.asarray(center, dtype=np.float32)
    if center_arr.shape != (3,):
        raise ValueError(f"center must have length 3, got {center}")
    strain_f = float(strain)

    def _field(
        coords: Float32[np.ndarray, "points 3"],
    ) -> Float32[np.ndarray, "points 3"]:
        return ((coords - center_arr) * strain_f).astype(np.float32, copy=False)

    return _field


def simple_shear(
    gamma: float,
    shear_axis: int,
    gradient_axis: int,
) -> DisplacementFunction:
    """Return a simple-shear field ``u[shear_axis] = gamma * x[gradient_axis]``.

    Parameters
    ----------
    gamma
        Shear strain. The displacement along ``shear_axis`` is
        ``gamma * coord[gradient_axis]``; all other components are
        zero.
    shear_axis
        Axis along which displacement occurs. ``0``/``1``/``2`` →
        ``z``/``y``/``x``.
    gradient_axis
        Axis whose coordinate drives the shear amplitude.

    Raises
    ------
    ValueError
        If ``shear_axis == gradient_axis`` (that is a dilation along
        one axis, not a shear), or if either axis is out of range.
    """
    if shear_axis not in (0, 1, 2) or gradient_axis not in (0, 1, 2):
        raise ValueError(
            f"shear_axis/gradient_axis must be in {{0, 1, 2}}, "
            f"got {shear_axis}, {gradient_axis}"
        )
    if shear_axis == gradient_axis:
        raise ValueError("shear_axis must differ from gradient_axis")
    gamma_f = float(gamma)

    def _field(
        coords: Float32[np.ndarray, "points 3"],
    ) -> Float32[np.ndarray, "points 3"]:
        out = np.zeros_like(coords, dtype=np.float32)
        out[:, shear_axis] = (gamma_f * coords[:, gradient_axis]).astype(
            np.float32, copy=False
        )
        return out

    return _field


def sinusoidal(
    amplitude: tuple[float, float, float],
    wavelength: tuple[float, float, float],
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DisplacementFunction:
    """Return a per-axis sinusoidal displacement field.

    The displacement along axis ``i`` is
    ``A_i * sin(2π * coord[i] / L_i + φ_i)`` (independent per axis).

    Parameters
    ----------
    amplitude
        ``(az, ay, ax)`` peak displacement per axis, in voxels.
    wavelength
        ``(lz, ly, lx)`` spatial wavelength per axis, in voxels. Each
        entry must be positive and non-zero.
    phase
        ``(φz, φy, φx)`` phase offsets in radians.

    Raises
    ------
    ValueError
        If any wavelength is non-positive.
    """
    amp_arr = np.asarray(amplitude, dtype=np.float32)
    wl_arr = np.asarray(wavelength, dtype=np.float32)
    phase_arr = np.asarray(phase, dtype=np.float32)
    if amp_arr.shape != (3,) or wl_arr.shape != (3,) or phase_arr.shape != (3,):
        raise ValueError("amplitude, wavelength, phase must each have length 3")
    if not np.all(wl_arr > 0.0):
        raise ValueError(f"wavelength entries must be positive, got {wavelength}")

    two_pi_over_wl = (2.0 * np.pi / wl_arr).astype(np.float32)

    def _field(
        coords: Float32[np.ndarray, "points 3"],
    ) -> Float32[np.ndarray, "points 3"]:
        angles = coords * two_pi_over_wl + phase_arr
        return (amp_arr * np.sin(angles)).astype(np.float32, copy=False)

    return _field


def compose(*fields: DisplacementFunction) -> DisplacementFunction:
    """Return the pointwise sum of displacement fields.

    Composition here is **additive at coordinates**, not function
    composition: ``compose(f, g)(x) == f(x) + g(x)``. This matches how
    linear superpositions of kinematic modes (shift + dilation, etc.)
    are usually expressed in DVC validation.

    Raises
    ------
    ValueError
        If called with no fields.
    """
    if not fields:
        raise ValueError("compose requires at least one field")

    fields_tuple = tuple(fields)

    def _field(
        coords: Float32[np.ndarray, "points 3"],
    ) -> Float32[np.ndarray, "points 3"]:
        total = fields_tuple[0](coords).astype(np.float32, copy=True)
        for f in fields_tuple[1:]:
            total += f(coords)
        return total

    return _field


def warp(
    reference: Float32[np.ndarray, "z y x"],
    field: DisplacementFunction,
    *,
    order: int = 3,
    mode: str = "reflect",
    convention: WarpConvention = "pull_back",
) -> Float32[np.ndarray, "z y x"]:
    """Warp ``reference`` under ``field`` via cubic spline resampling.

    Two sign conventions are supported:

    - ``"pull_back"`` (default): ``deformed(x) = reference(x - u(x))``.
      This is the DIC/DVC convention used by the rest of the pipeline;
      ``correlate()`` recovers the same ``u`` that appears here.
    - ``"push_forward"``: ``deformed(x) = reference(x + u(x))``.
      Implemented as a sign flip — *not* a true forward scatter. It is
      equivalent to pull-back of ``-u``. Use this when reproducing
      inputs whose stored displacement follows the opposite sign
      convention; the recovered displacement from ``correlate()`` will
      then be the negation of ``field``.

    Parameters
    ----------
    reference
        ``(z, y, x)`` float32 source volume.
    field
        Analytical displacement field.
    order
        Spline order for ``scipy.ndimage.map_coordinates``. Default 3
        (cubic); order 1 biases the Gaussian subvoxel fit enough to
        break the 0.1-voxel assertion in the plan.
    mode
        Boundary mode forwarded to ``map_coordinates``. Default
        ``"reflect"``; tests should still exclude a near-boundary band
        when stratifying error.
    convention
        ``"pull_back"`` (default) or ``"push_forward"``. See module
        docstring for the sign-convention discussion.

    Returns
    -------
    numpy.ndarray
        ``(z, y, x)`` float32 deformed volume.

    Raises
    ------
    ValueError
        If ``reference`` is not 3D, or ``convention`` is not one of
        ``"pull_back"`` / ``"push_forward"``.
    """
    if reference.ndim != 3:
        raise ValueError(f"reference must be 3D, got ndim={reference.ndim}")
    if convention not in ("pull_back", "push_forward"):
        raise ValueError(
            f"convention must be 'pull_back' or 'push_forward', got {convention!r}"
        )

    shape = reference.shape
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    disp = field(coords)
    # pull_back samples at x - u(x); push_forward sign-flips to x + u(x).
    # `.T` puts axes first because map_coordinates expects (3, N).
    signed_disp = -disp if convention == "pull_back" else disp
    sample_coords = (coords + signed_disp).T

    warped = map_coordinates(
        reference,
        sample_coords,
        order=order,
        mode=mode,
        prefilter=True,
    ).reshape(shape)

    return warped.astype(np.float32, copy=False)


def sample_on_grid(
    field: DisplacementFunction,
    grid: GridSpec,
) -> Float32[np.ndarray, "points 3"]:
    """Evaluate an analytical field at the POI centers of ``grid``.

    Convenience wrapper so tests don't re-implement the coordinate
    conversion when comparing a recovered displacement to ground
    truth.
    """
    return field(grid.positions)


def make_pair(
    shape: tuple[int, int, int],
    field: DisplacementFunction,
    *,
    texture_sigma: float = 1.5,
    seed: int = 0,
    mask: Bool[np.ndarray, "z y x"] | None = None,
    order: int = 3,
    reference: Float32[np.ndarray, "z y x"] | None = None,
    convention: WarpConvention = "pull_back",
) -> SyntheticPair:
    """Produce a ``(reference, deformed, field, mask)`` bundle.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape for both volumes.
    field
        Analytical displacement to apply.
    texture_sigma
        Gaussian sigma for :func:`make_texture`. Ignored when
        ``reference`` is supplied.
    seed
        RNG seed for :func:`make_texture`. Ignored when ``reference``
        is supplied.
    mask
        Optional boolean mask, passed through to the result; this
        module does not generate masks itself.
    order
        Spline order for the warp.
    reference
        Optional pre-built reference volume (e.g. a phantom from
        :mod:`mamba_dvc.validate.phantoms`). When ``None`` (default),
        ``make_texture`` generates a band-limited noise reference.
    convention
        Warp sign convention; see :func:`warp`. Default ``"pull_back"``.

    Returns
    -------
    SyntheticPair
        Bundle carrying both volumes, the field callable, and the
        optional mask.

    Raises
    ------
    ValueError
        If ``mask`` is provided but its shape does not match
        ``shape``, or if ``reference`` is provided with the wrong
        shape / dtype.
    """
    if mask is not None and mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match requested shape {shape}")

    if reference is None:
        reference_arr = make_texture(shape, sigma=texture_sigma, seed=seed)
    else:
        if reference.shape != shape:
            raise ValueError(
                f"reference shape {reference.shape} does not match requested shape {shape}"
            )
        if reference.dtype != np.float32:
            raise ValueError(f"reference must be float32, got {reference.dtype}")
        reference_arr = reference

    deformed = warp(reference_arr, field, order=order, convention=convention)
    return SyntheticPair(
        reference=reference_arr,
        deformed=deformed,
        field=field,
        mask=mask,
    )
