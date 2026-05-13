"""Operations on :class:`DisplacementField` instances.

This module owns combinators over already-computed displacement
fields: composition, semi-Lagrangian resampling, and friends. The
inputs are :class:`mamba_dvc.types.DisplacementField` dataclasses
produced by :func:`mamba_dvc.pipeline.correlate.correlate` or
:func:`mamba_dvc.pipeline.series.correlate_series`; the outputs share
the same lattice metadata.

Composition is the time-series counterpart to
:func:`mamba_dvc.validate.synthetic.compose`, which adds analytical
``DisplacementFunction`` callables. The two are *different operations*:

* ``synthetic.compose(f, g)(x) = f(x) + g(x)`` — pointwise sum of
  callables, used to build a kinematic superposition (e.g. rigid
  shift + dilation) before a pair is warped.
* :func:`compose_displacement_fields` — semi-Lagrangian temporal
  composition of arrays already sampled on a POI lattice. Used to
  chain inter-frame increments from a ``SEQUENTIAL`` series into an
  absolute field across the series.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from jaxtyping import Float32
from scipy.interpolate import RegularGridInterpolator

from mamba_dvc.types import DisplacementField, POIStatus

__all__ = ["compose_displacement_fields", "sample_field_at"]


def compose_displacement_fields(
    u_total: DisplacementField,
    u_inc: DisplacementField,
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> DisplacementField:
    """Semi-Lagrangian composition of two displacement fields.

    Implements::

        u_total_new(x) = u_total(x) + u_inc(x + u_total(x))

    where ``x`` are the POI positions of both fields (the driver builds
    the grid once for the whole series, so ``u_total`` and ``u_inc``
    share lattice). The sample ``u_inc(x + u_total(x))`` is computed by
    interpolating :attr:`u_inc.displacements` on the regular lattice at
    the offset coordinates.

    The "increment" interpretation: ``u_total`` is the cumulative
    displacement from the series reference frame to some intermediate
    frame ``t_i``; ``u_inc`` is the inter-frame displacement from
    ``t_i`` to ``t_{i+1}`` recovered by the correlator. The result is
    the cumulative displacement from the series reference to
    ``t_{i+1}``.

    Parameters
    ----------
    u_total
        Cumulative field up to the previous frame.
    u_inc
        Inter-frame incremental field from the previous frame to the
        next. Must share lattice metadata with ``u_total``.
    interpolation
        Interpolation order for the semi-Lagrangian sample.
        ``"linear"`` (default) is the small-displacement regime;
        ``"cubic"`` is a higher-order alternative when per-step
        displacements approach the grid spacing.

    Returns
    -------
    DisplacementField
        A new frozen field on the shared lattice. POIs whose query
        coordinates ``x + u_total(x)`` fall outside the lattice extent,
        or whose source POIs were invalid in either input, are flagged
        :data:`POIStatus.MASKED` with zero displacement and zero
        confidence. Output confidence is the per-POI minimum of the
        two inputs (conservative: the chain is as good as its weakest
        link).

    Raises
    ------
    ValueError
        If the two fields disagree on ``positions``, ``grid_shape``,
        ``spacing``, or ``window``, or if ``interpolation`` is not one
        of ``"linear" | "cubic"``.

    Notes
    -----
    The interpolation only supports a regular lattice in voxel
    coordinates: positions are expected to be the canonical lattice
    produced by :func:`mamba_dvc.core.grid.build_grid` (``starts``
    plus half-window). Non-regular POI sets would require a different
    sampler.

    The cumulative output remains MASKED-or-OK; the more granular
    :class:`POIStatus` values (``OUTLIER``, ``LOW_CONF``,
    ``OUT_OF_RANGE``) do not compose meaningfully across time, so the
    result collapses them into ``MASKED`` for downstream consumers.
    """
    if interpolation not in ("linear", "cubic"):
        raise ValueError(f"interpolation must be 'linear' or 'cubic', got {interpolation!r}")
    if u_total.grid_shape != u_inc.grid_shape:
        raise ValueError(
            f"grid_shape mismatch: u_total {u_total.grid_shape} vs u_inc {u_inc.grid_shape}"
        )
    if u_total.spacing != u_inc.spacing:
        raise ValueError(
            f"spacing mismatch: u_total {u_total.spacing} vs u_inc {u_inc.spacing}"
        )
    if u_total.window != u_inc.window:
        raise ValueError(f"window mismatch: u_total {u_total.window} vs u_inc {u_inc.window}")
    if u_total.positions.shape != u_inc.positions.shape:
        raise ValueError(
            f"positions shape mismatch: u_total {u_total.positions.shape} "
            f"vs u_inc {u_inc.positions.shape}"
        )
    # Positions are produced by ``build_grid`` so they should be bit-identical
    # when the lattice metadata matches, but verify because downstream sampling
    # would silently misalign if a caller stitched two unrelated fields.
    if not np.array_equal(u_total.positions, u_inc.positions):
        raise ValueError(
            "u_total.positions and u_inc.positions differ — fields are not on the same lattice"
        )

    nz, ny, nx = u_total.grid_shape

    # Per-axis 1D coordinate vectors. The lattice is regular so we read
    # the first slab along each axis instead of recomputing strides.
    positions_lattice = u_total.positions.reshape(nz, ny, nx, 3)
    z_axis = positions_lattice[:, 0, 0, 0].astype(np.float64)
    y_axis = positions_lattice[0, :, 0, 1].astype(np.float64)
    x_axis = positions_lattice[0, 0, :, 2].astype(np.float64)

    inc_lattice = u_inc.displacements.reshape(nz, ny, nx, 3)

    query_points = (u_total.positions + u_total.displacements).astype(np.float64, copy=False)

    sampled = np.empty_like(u_total.displacements, dtype=np.float32)
    for component in range(3):
        interp = RegularGridInterpolator(
            (z_axis, y_axis, x_axis),
            inc_lattice[..., component].astype(np.float64),
            method=interpolation,
            bounds_error=False,
            fill_value=np.nan,
        )
        sampled[:, component] = interp(query_points).astype(np.float32)

    out_of_bounds = np.any(np.isnan(sampled), axis=1)
    sampled = np.where(np.isnan(sampled), np.float32(0.0), sampled).astype(
        np.float32, copy=False
    )

    new_displacements = (u_total.displacements + sampled).astype(np.float32, copy=False)
    new_valid = u_total.valid & u_inc.valid & ~out_of_bounds
    new_displacements = np.where(
        new_valid[:, None], new_displacements, np.float32(0.0)
    ).astype(np.float32, copy=False)
    new_confidence = np.where(
        new_valid,
        np.minimum(u_total.confidence, u_inc.confidence),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    new_status = np.where(
        new_valid, np.uint8(POIStatus.OK), np.uint8(POIStatus.MASKED)
    ).astype(np.uint8, copy=False)

    return DisplacementField(
        positions=u_total.positions,
        displacements=new_displacements,
        valid=new_valid,
        confidence=new_confidence,
        status=new_status,
        grid_shape=u_total.grid_shape,
        spacing=u_total.spacing,
        window=u_total.window,
    )


def sample_field_at(
    field: DisplacementField,
    coords: Float32[np.ndarray, "n 3"],
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
    fill_value: float = 0.0,
) -> Float32[np.ndarray, "n 3"]:
    """Resample a :class:`DisplacementField` at arbitrary voxel coordinates.

    Convenience wrapper around the same regular-grid interpolator used
    inside :func:`compose_displacement_fields`. Out-of-lattice queries
    are filled with ``fill_value`` (default ``0.0``) rather than
    raising.

    Parameters
    ----------
    field
        Source field; its POI positions must form the canonical regular
        lattice from :func:`mamba_dvc.core.grid.build_grid`.
    coords
        ``(n, 3)`` query coordinates in voxel space, ``(z, y, x)`` order.
    interpolation
        ``"linear"`` (default) or ``"cubic"``.
    fill_value
        Value used at coordinates outside the lattice extent.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` float32 sampled displacements.

    Raises
    ------
    ValueError
        If ``interpolation`` is not one of ``"linear" | "cubic"`` or
        ``coords`` is not a ``(n, 3)`` array.
    """
    if interpolation not in ("linear", "cubic"):
        raise ValueError(f"interpolation must be 'linear' or 'cubic', got {interpolation!r}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3), got {coords.shape}")

    nz, ny, nx = field.grid_shape
    positions_lattice = field.positions.reshape(nz, ny, nx, 3)
    z_axis = positions_lattice[:, 0, 0, 0].astype(np.float64)
    y_axis = positions_lattice[0, :, 0, 1].astype(np.float64)
    x_axis = positions_lattice[0, 0, :, 2].astype(np.float64)

    disp_lattice = field.displacements.reshape(nz, ny, nx, 3)
    query = coords.astype(np.float64, copy=False)

    out = np.empty_like(coords, dtype=np.float32)
    for component in range(3):
        interp = RegularGridInterpolator(
            (z_axis, y_axis, x_axis),
            disp_lattice[..., component].astype(np.float64),
            method=interpolation,
            bounds_error=False,
            fill_value=float(fill_value),
        )
        out[:, component] = interp(query).astype(np.float32)
    return out
