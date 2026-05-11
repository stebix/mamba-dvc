"""Sampled ground-truth displacement fields for experimental zarr stores.

Wraps a ``(Z, Y, X, 3)`` array of stored displacements as a
:class:`mamba_dvc.validate.synthetic.DisplacementFunction`-conformant
callable. Once normalized, downstream tooling (``correlate``,
``validate.synthetic.sample_on_grid``) treats experimental ground truth
identically to analytical fields.

Storage normalization
---------------------
On-disk layouts vary across collaborators:

- Axis order: leading vector axis (``"3_zyx"`` ⇒ ``(3, Z, Y, X)``) or
  trailing vector axis (``"zyx_3"`` ⇒ ``(Z, Y, X, 3)``).
- Vector order: ``(dz, dy, dx)`` is the project convention; ``"xyz_3"``
  is reserved for the (rare) case where the on-disk vector order is
  ``(dx, dy, dz)``.
- Convention: ``"pull_back"`` matches
  :func:`mamba_dvc.validate.synthetic.warp` and the sign
  ``correlate()`` returns; ``"push_forward"`` is the opposite sign.

Construction normalizes every variant to internal ``(Z, Y, X, 3)``
pull-back with vector order ``(dz, dy, dx)``. After that, everything
downstream branches on nothing.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import zarr
from jaxtyping import Float32
from scipy.ndimage import map_coordinates, spline_filter

from mamba_dvc.io.volume import center_slice

__all__ = ["FieldAxisOrder", "FieldConvention", "GroundTruthField"]


FieldConvention = Literal["pull_back", "push_forward"]
FieldAxisOrder = Literal["zyx_3", "3_zyx"]

# Spline orders below this need no prefilter (and ``spline_filter`` rejects
# them); at and above it the B-spline prefilter is applied once at
# construction so ``__call__`` doesn't re-run it on every evaluation.
_PREFILTER_MIN_ORDER = 2
_BOUNDARY_MODE = "reflect"


class GroundTruthField:
    """Sampled ``(Z, Y, X, 3)`` deformation array exposed as a callable.

    Parameters
    ----------
    array
        Sampled displacement field. Internal storage is normalized to
        ``(Z, Y, X, 3)`` with last-axis order ``(dz, dy, dx)``;
        construct via :meth:`from_zarr` to handle on-disk variants.
    convention
        ``"pull_back"`` matches
        :func:`mamba_dvc.validate.synthetic.warp`:
        ``deformed(x) = reference(x - u(x))``. ``"push_forward"`` is
        the opposite sign and is converted to pull-back at
        construction time so downstream code never branches on
        convention.
    interpolation
        Spline order for off-grid evaluation. Default 3 (cubic) — same
        order :func:`mamba_dvc.validate.synthetic.warp` uses, so
        synthetic-vs-experimental comparisons share the same
        interpolation error budget.

    Notes
    -----
    Storage is normalized at construction. After ``__init__`` returns,
    ``self._array`` is always ``(Z, Y, X, 3)`` float32 in pull-back
    convention with ``(dz, dy, dx)`` vector order. Comparisons against
    :attr:`mamba_dvc.types.DisplacementField.displacements` work
    without sign-flipping.

    For ``interpolation >= 2`` the B-spline prefilter is applied once
    here (per spatial component, with the same ``"reflect"`` boundary
    mode :meth:`__call__` uses) and ``map_coordinates`` is then called
    with ``prefilter=False``. ``scipy.ndimage.map_coordinates(...,
    prefilter=True)`` otherwise re-runs the full-volume recursive
    filter on every call, which dominates evaluation cost for
    production-sized flow arrays. The input array is never mutated.
    """

    _array: Float32[np.ndarray, "z y x 3"]
    _interpolation: int
    _prefiltered: bool

    def __init__(
        self,
        array: Float32[np.ndarray, "z y x 3"],
        *,
        convention: FieldConvention = "pull_back",
        interpolation: int = 3,
    ) -> None:
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"array must have shape (Z, Y, X, 3), got {array.shape}")
        if interpolation < 0 or interpolation > 5:
            raise ValueError(f"interpolation must be in [0, 5], got {interpolation}")

        # Own a private contiguous float32 buffer: we negate (push_forward)
        # and/or prefilter in place below, and must not touch the caller's array.
        normalized = np.array(array, dtype=np.float32, order="C")
        if convention == "push_forward":
            np.negative(normalized, out=normalized)

        if interpolation >= _PREFILTER_MIN_ORDER:
            for axis in range(3):
                # ``output=np.float32`` keeps the prefilter coefficients (and
                # the transient buffer) in float32, matching what
                # ``map_coordinates(prefilter=True)`` does internally for a
                # float32 input. The scipy stub types ``output`` too narrowly
                # (``type[float64]``); the runtime accepts any dtype.
                normalized[..., axis] = spline_filter(
                    normalized[..., axis],
                    order=interpolation,
                    output=np.float32,  # pyright: ignore[reportArgumentType]
                    mode=_BOUNDARY_MODE,
                )
            self._prefiltered = True
        else:
            self._prefiltered = False

        self._array = normalized
        self._interpolation = interpolation

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape ``(Z, Y, X)`` of the underlying field."""
        return (
            int(self._array.shape[0]),
            int(self._array.shape[1]),
            int(self._array.shape[2]),
        )

    @classmethod
    def from_zarr(
        cls,
        zarr_array: zarr.Array[Any],
        *,
        axis_order: FieldAxisOrder,
        convention: FieldConvention,
        dry_shape: tuple[int, int, int] | None = None,
        interpolation: int = 3,
    ) -> GroundTruthField:
        """Materialize a stored flow array and wrap it as a callable.

        Parameters
        ----------
        zarr_array
            Lazy zarr array. Must be 4D; the leading or trailing axis
            (per ``axis_order``) carries the 3 vector components.
        axis_order
            On-disk axis layout. ``"3_zyx"`` ⇒ ``(3, Z, Y, X)`` (vector
            axis first); ``"zyx_3"`` ⇒ ``(Z, Y, X, 3)`` (vector axis
            last).
        convention
            On-disk sign convention; see class docstring.
        dry_shape
            Optional ``(z, y, x)`` size of a centered subblock.
            Applied to the *spatial* axes, leaving the vector axis
            intact.
        interpolation
            Spline order for off-grid evaluation.

        Raises
        ------
        ValueError
            If ``zarr_array`` is not 4D, the vector axis size is not
            3, or ``dry_shape`` exceeds the spatial shape.
        """
        if zarr_array.ndim != 4:
            raise ValueError(f"flow array must be 4D, got ndim={zarr_array.ndim}")

        full_shape = tuple(int(s) for s in zarr_array.shape)
        if axis_order == "3_zyx":
            vector_axis_size = full_shape[0]
            spatial_shape = full_shape[1:4]
        else:
            vector_axis_size = full_shape[3]
            spatial_shape = full_shape[0:3]

        if vector_axis_size != 3:
            raise ValueError(f"flow array vector axis must be size 3, got {vector_axis_size}")

        if dry_shape is not None:
            spatial_slice = center_slice(spatial_shape, dry_shape)
            if axis_order == "3_zyx":
                full_slice: tuple[slice, ...] = (slice(None), *spatial_slice)
            else:
                full_slice = (*spatial_slice, slice(None))
            data = np.asarray(zarr_array[full_slice])
        else:
            data = np.asarray(zarr_array[:])

        if axis_order == "3_zyx":
            data = np.moveaxis(data, 0, -1)

        data = np.ascontiguousarray(data, dtype=np.float32)
        return cls(data, convention=convention, interpolation=interpolation)

    def __call__(
        self, coords: Float32[np.ndarray, "points 3"]
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate ``u`` at arbitrary voxel coordinates ``(z, y, x)``.

        Parameters
        ----------
        coords
            ``(points, 3)`` float32 voxel coordinates in ``(z, y, x)``
            order.

        Returns
        -------
        numpy.ndarray
            ``(points, 3)`` displacements ``(dz, dy, dx)`` at each
            input coordinate, in pull-back convention.
        """
        coords_arr = np.ascontiguousarray(coords, dtype=np.float32)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (points, 3), got {coords_arr.shape}")

        sample_coords = coords_arr.T  # (3, points)
        out = np.empty_like(coords_arr)
        for axis in range(3):
            out[:, axis] = map_coordinates(
                self._array[..., axis],
                sample_coords,
                order=self._interpolation,
                mode=_BOUNDARY_MODE,
                prefilter=not self._prefiltered,
            ).astype(np.float32, copy=False)
        return out
