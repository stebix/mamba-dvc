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

import inspect
import itertools
from collections.abc import Callable, Sequence
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
    "SyntheticSeries",
    "TemporalDisplacementFunction",
    "WarpConvention",
    "compose",
    "linear_motion",
    "make_pair",
    "make_series",
    "make_texture",
    "normalize_temporal_form",
    "rigid_shift",
    "sample_on_grid",
    "simple_shear",
    "sinusoidal",
    "temporal_from_curried",
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


@runtime_checkable
class TemporalDisplacementFunction(Protocol):
    """Time-parameterized displacement field, evaluable at arbitrary coords.

    Implementations accept a ``(points, 3)`` coordinate array in
    ``(z, y, x)`` order plus a scalar time ``t`` and return a
    ``(points, 3)`` displacement array in the same order. Sign
    convention follows :class:`DisplacementFunction`: the displacement
    is the field that :func:`warp` consumes at the given ``t``.

    The canonical convention used by the time-series test bed is that
    ``u(coords, 0)`` is the zero field — i.e. ``frame_0`` equals the
    reference modulo cubic-spline prefilter quantization. This is not
    enforced; analytical fields that start non-zero at ``t = 0`` are
    legal but the consumer must understand the implications for the
    per-pair vs. cumulative error split.
    """

    def __call__(
        self,
        coords: Float32[np.ndarray, "points 3"],
        t: float,
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate the field at ``coords`` at time ``t`` and return displacements."""
        ...


def temporal_from_curried(
    u_of_t: Callable[[float], DisplacementFunction],
) -> TemporalDisplacementFunction:
    """Wrap a curried ``t -> DisplacementFunction`` into the pointwise form.

    The pointwise form ``(coords, t) -> displacements`` is the canonical
    internal representation used by :class:`TemporalDisplacementFunction`
    and :func:`make_series`. Use this helper to adapt a closure-style
    factory that returns a fresh :class:`DisplacementFunction` per time
    step:

    >>> u_curried = lambda t: rigid_shift((0.0, t, 0.0))
    >>> u_temporal = temporal_from_curried(u_curried)
    >>> u_temporal(coords, t=2.0)  # equivalent to rigid_shift((0.0, 2.0, 0.0))(coords)

    Parameters
    ----------
    u_of_t
        Callable taking a scalar ``t`` and returning a
        :class:`DisplacementFunction` evaluable at that time.

    Returns
    -------
    TemporalDisplacementFunction
        ``(coords, t) -> displacements`` view of the same field.
    """

    def _temporal(
        coords: Float32[np.ndarray, "points 3"], t: float
    ) -> Float32[np.ndarray, "points 3"]:
        return u_of_t(float(t))(coords)

    return _temporal


def linear_motion(velocity: DisplacementFunction) -> TemporalDisplacementFunction:
    """Build a temporal field ``u(coords, t) = t * velocity(coords)``.

    Scales a spatial velocity pattern
    linearly with time. The most common shape for synthetic series
    eval — every analytical kinematic mode in this module
    (:func:`rigid_shift`, :func:`uniform_dilation`, :func:`simple_shear`,
    :func:`sinusoidal`) becomes a time-evolving field by passing it
    through :func:`linear_motion`.

    Parameters
    ----------
    velocity
        Spatial pattern. Interpreted as displacement-per-unit-time;
        the units are whatever ``t`` carries in the caller's series
        (typically integer frame index).

    Returns
    -------
    TemporalDisplacementFunction
        ``u(coords, t) = t * velocity(coords)``.
    """

    def _temporal(
        coords: Float32[np.ndarray, "points 3"], t: float
    ) -> Float32[np.ndarray, "points 3"]:
        scaled = velocity(coords) * np.float32(t)
        return scaled.astype(np.float32, copy=False)

    return _temporal


def normalize_temporal_form(
    u_of_t: Callable[..., object],
) -> TemporalDisplacementFunction:
    """Coerce a 1-arg (curried) or 2-arg (pointwise) callable into the pointwise form.

    Dispatch is by *total positional parameter count* (required +
    defaulted). A 1-positional callable is interpreted as the curried
    form ``t -> DisplacementFunction``; a 2-positional callable as the
    pointwise form ``(coords, t) -> displacements``.

    Known ambiguity: a pointwise function written as
    ``def f(coords, t=0.0)`` still has 2 positional parameters and is
    correctly identified. A curried function written with extra
    keyword-only parameters keeps its single positional and is also
    correctly identified. The pathological case is a curried function
    with ``*args``; signature introspection rejects it with a
    :class:`ValueError` rather than guessing.

    Parameters
    ----------
    u_of_t
        Either a :class:`TemporalDisplacementFunction` (pointwise) or
        a ``Callable[[float], DisplacementFunction]`` (curried).

    Returns
    -------
    TemporalDisplacementFunction
        Pointwise form. Returned identically when ``u_of_t`` is
        already pointwise.

    Raises
    ------
    ValueError
        If the positional arity is not 1 or 2, or the callable's
        signature cannot be introspected.
    """
    try:
        sig = inspect.signature(u_of_t)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "u_of_t signature is not introspectable; pass either a function with"
            " 1 (curried) or 2 (pointwise) positional parameters, or wrap with"
            " temporal_from_curried"
        ) from exc

    positional_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    positional = [p for p in sig.parameters.values() if p.kind in positional_kinds]
    has_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if has_var_positional:
        raise ValueError(
            "u_of_t signature uses *args, which is ambiguous between curried and"
            " pointwise forms; wrap it with temporal_from_curried to disambiguate"
        )

    n_pos = len(positional)
    if n_pos == 1:
        return temporal_from_curried(u_of_t)  # type: ignore[arg-type]
    if n_pos == 2:
        return u_of_t  # type: ignore[return-value]
    raise ValueError(
        f"u_of_t must take 1 (curried) or 2 (pointwise) positional arguments, got {n_pos}"
    )


@dataclass(frozen=True)
class SyntheticSeries:
    """Container for a synthesized reference + warped-frame series.

    Parameters
    ----------
    reference
        ``(z, y, x)`` float32 reference volume (frame at the series
        anchor point — typically ``t = timesteps[0]``).
    frames
        Tuple of ``(z, y, x)`` float32 volumes, one per entry of
        :attr:`timesteps`. ``frames[i] = warp(reference, u(·, timesteps[i]))``.
    timesteps
        Integer frame indices aligned with :attr:`frames`. Monotonic
        increasing, no duplicates.
    field_at
        Pointwise temporal field used to synthesize :attr:`frames`.
        Always stored in the normalized ``(coords, t) -> displacements``
        form; downstream eval code calls it directly to evaluate GT at
        POI centers.
    mask
        Optional shared validity mask carried through verbatim; this
        module does not generate masks itself.
    """

    reference: Float32[np.ndarray, "z y x"]
    frames: tuple[Float32[np.ndarray, "z y x"], ...]
    timesteps: tuple[int, ...]
    field_at: TemporalDisplacementFunction
    mask: Bool[np.ndarray, "z y x"] | None


def make_series(
    shape: tuple[int, int, int],
    u_of_t: Callable[..., object],
    timesteps: Sequence[int],
    *,
    reference: Float32[np.ndarray, "z y x"] | None = None,
    texture_sigma: float = 1.5,
    seed: int = 0,
    mask: Bool[np.ndarray, "z y x"] | None = None,
    order: int = 3,
    convention: WarpConvention = "pull_back",
) -> SyntheticSeries:
    """Produce ``(reference, frames, field_at, mask)`` for a time-series eval.

    Time-series sibling of :func:`make_pair`. Synthesizes one warped
    frame per entry of ``timesteps`` against a shared reference
    volume. The reference is generated by :func:`make_texture` when
    not supplied.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape shared by the reference and every
        synthesized frame.
    u_of_t
        Temporal displacement field. Either the pointwise form
        ``(coords, t) -> displacements`` (canonical) or the curried
        form ``t -> DisplacementFunction``. Dispatch is by positional
        arity (see :func:`normalize_temporal_form`).
    timesteps
        Integer frame indices to synthesize. Must be non-empty,
        monotonic increasing, and have no duplicates. The convention
        is that ``timesteps[0]`` (typically ``0``) anchors the series:
        ``frames[0] = warp(reference, u(·, timesteps[0]))``, which
        equals the reference modulo cubic-spline prefilter noise when
        ``u(·, timesteps[0])`` is the zero field.
    reference
        Optional pre-built reference volume. When ``None``,
        :func:`make_texture` generates a band-limited noise reference
        from ``shape`` + ``texture_sigma`` + ``seed``.
    texture_sigma, seed
        Forwarded to :func:`make_texture`; ignored when ``reference``
        is supplied.
    mask
        Optional shared validity mask. Must match ``shape``.
    order
        Spline order for :func:`scipy.ndimage.map_coordinates` inside
        :func:`warp`. Default ``3`` (cubic) matches :func:`make_pair`
        — order ``1`` (trilinear) biases the subvoxel fit enough to
        inflate the measurable error floor below ~0.2 vx and should
        only be used for fast iteration where the absolute floor is
        not being reported.
    convention
        Warp sign convention; see :func:`warp`. Default
        ``"pull_back"``.

    Returns
    -------
    SyntheticSeries
        Frozen bundle with the reference, frames, timesteps, the
        normalized pointwise field, and the optional mask.

    Raises
    ------
    ValueError
        If ``timesteps`` is empty, non-monotonic, or contains
        duplicates; if ``reference`` is supplied with wrong shape or
        dtype; if ``mask`` is supplied with the wrong shape; if
        ``u_of_t`` cannot be dispatched.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must have length 3, got {shape}")
    if mask is not None and mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match requested shape {shape}")

    ts = tuple(int(t) for t in timesteps)
    if not ts:
        raise ValueError("timesteps must be non-empty")
    if len(set(ts)) != len(ts):
        raise ValueError(f"timesteps must have no duplicates, got {ts}")
    if any(b <= a for a, b in itertools.pairwise(ts)):
        raise ValueError(f"timesteps must be strictly increasing, got {ts}")

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

    temporal = normalize_temporal_form(u_of_t)

    def _field_at_t(t_value: float) -> DisplacementFunction:
        def _field(
            coords: Float32[np.ndarray, "points 3"],
        ) -> Float32[np.ndarray, "points 3"]:
            return temporal(coords, t_value)

        return _field

    frames_out: list[Float32[np.ndarray, "z y x"]] = []
    for t in ts:
        # Always run the warp pipeline — including at t=0 — so every
        # frame shares the same spectral content. Skipping the warp
        # for the anchor frame would give frame_0 a different prefilter
        # signature than t > 0 frames and inject a systematic NCC bias.
        warped = warp(reference_arr, _field_at_t(float(t)), order=order, convention=convention)
        frames_out.append(warped)

    return SyntheticSeries(
        reference=reference_arr,
        frames=tuple(frames_out),
        timesteps=ts,
        field_at=temporal,
        mask=mask,
    )
