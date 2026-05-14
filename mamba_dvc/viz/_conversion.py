"""Numpy ↔ VTK type bridges and unit math for the viz submodule.

Centralizes every conversion between project arrays / dataclasses and
the VTK objects PyVista consumes, so the rest of :mod:`mamba_dvc.viz`
stays pure-Python. Also the only place voxel-unit arithmetic happens:
:func:`unify_spacing` is the chokepoint listed in
``docs/plans/visualization.md`` §7.

Imports of :mod:`pyvista` are deferred into the function bodies so this
module can be imported (and unit-tested for unit math / spacing
normalization) without the optional ``viz`` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.types import (
    DisplacementField,
    GridSpec,
    PhysicalUnit,
    VoxelSpacing,
)

if TYPE_CHECKING:
    import pyvista as pv


__all__ = [
    "downsample_for_render",
    "field_to_polydata",
    "grid_to_structured",
    "mask_to_image_data",
    "normalize_spacing",
    "unify_spacing",
    "volume_to_image_data",
]


# Conversion factors to a canonical millimeter base. ``"voxel"`` is
# intentionally absent: voxel-unit data has no physical scale and must
# not be mixed with physical units silently.
_TO_MM: dict[PhysicalUnit, float] = {
    "nm": 1e-6,
    "um": 1e-3,
    "mm": 1.0,
}


def normalize_spacing(
    spacing: VoxelSpacing | tuple[float, float, float] | None,
) -> VoxelSpacing:
    """Coerce a spacing argument into a :class:`VoxelSpacing`.

    Parameters
    ----------
    spacing
        :class:`VoxelSpacing`, a bare ``(z, y, x)`` tuple (treated as
        ``unit="voxel"``), or ``None`` (treated as
        ``VoxelSpacing((1.0, 1.0, 1.0), "voxel")``).

    Returns
    -------
    VoxelSpacing
        Always a fully constructed dataclass — downstream code never
        has to branch on type.
    """
    if isinstance(spacing, VoxelSpacing):
        return spacing
    if spacing is None:
        return VoxelSpacing((1.0, 1.0, 1.0), "voxel")
    values = tuple(float(v) for v in spacing)
    if len(values) != 3:
        raise ValueError(f"spacing tuple must have length 3, got {len(values)}")
    return VoxelSpacing(values, "voxel")


def unify_spacing(*specs: VoxelSpacing, target: PhysicalUnit = "mm") -> VoxelSpacing:
    """Convert a set of spacings to a single canonical unit.

    Parameters
    ----------
    *specs
        One or more :class:`VoxelSpacing` instances, typically gathered
        from different inputs of a multi-object plot.
    target
        Canonical unit to convert into. Defaults to ``"mm"``. Must be a
        physical unit; ``"voxel"`` is rejected because it is unitless.

    Returns
    -------
    VoxelSpacing
        A spacing in ``target`` units. If all inputs already share the
        target unit the values are returned unchanged.

    Raises
    ------
    ValueError
        If ``specs`` is empty, if any spec carries ``unit="voxel"``
        while another carries a physical unit (silent conversion would
        be meaningless), if ``target == "voxel"``, or if all inputs do
        not share identical numeric values after conversion.

    Notes
    -----
    The "all inputs must agree after conversion" rule reflects how the
    function is used: callers compose objects sampled on a common
    physical grid (recovered field + ground truth + volume). Mismatched
    physical spacings indicate a dataset bug, so the function raises
    rather than picking one arbitrarily.
    """
    if not specs:
        raise ValueError("unify_spacing requires at least one VoxelSpacing")
    if target == "voxel":
        raise ValueError("target unit must be a physical unit, got 'voxel'")
    if target not in get_args(PhysicalUnit):
        raise ValueError(f"target unit {target!r} is not a known PhysicalUnit")

    units = {s.unit for s in specs}
    if "voxel" in units and units != {"voxel"}:
        raise ValueError(f"cannot unify mixed voxel-unit and physical-unit spacings: {units}")

    if units == {"voxel"}:
        # Caller asked for a physical target but only has voxel-unit
        # inputs; that is the same meaningless mix, just inverted.
        raise ValueError(f"cannot convert voxel-unit spacings to physical unit {target!r}")

    target_factor = _TO_MM[target]
    converted: list[tuple[float, float, float]] = []
    for spec in specs:
        factor = _TO_MM[spec.unit] / target_factor
        converted.append(tuple(v * factor for v in spec.values))  # type: ignore[arg-type]

    first = converted[0]
    for other in converted[1:]:
        if not all(
            np.isclose(a, b, rtol=1e-6, atol=0.0) for a, b in zip(first, other, strict=True)
        ):
            raise ValueError(f"spacings disagree after conversion to {target!r}: {converted}")
    return VoxelSpacing(first, target)


def volume_to_image_data(
    volume: Float32[np.ndarray, "z y x"],
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
) -> pv.ImageData:
    """Wrap a ``(z, y, x)`` float32 volume as a :class:`pyvista.ImageData`.

    The PyVista convention is ``(x, y, z)`` for ``dimensions`` and
    ``spacing``; this function handles the axis swap so callers can
    keep the project-wide ``(z, y, x)`` ordering.

    Parameters
    ----------
    volume
        ``(z, y, x)`` float32 array. Made C-contiguous if necessary;
        callers should pass C-contiguous arrays for zero-copy.
    spacing
        Voxel spacing; see :func:`normalize_spacing`. Anisotropic
        spacings are honored.

    Returns
    -------
    pyvista.ImageData
        Grid carrying the volume as a point-data scalar named
        ``"intensity"``.
    """
    import pyvista as pv

    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got ndim={volume.ndim}")
    if volume.dtype != np.float32:
        raise TypeError(f"volume must be float32, got {volume.dtype}")

    arr = np.ascontiguousarray(volume)
    sp = normalize_spacing(spacing)

    nz, ny, nx = arr.shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(sp.values[2], sp.values[1], sp.values[0]),
        origin=(0.0, 0.0, 0.0),
    )
    # PyVista's point_data expects a flat array in Fortran order over
    # ``(x, y, z)``. ``ravel(order="F")`` on a ``(z, y, x)`` array gives
    # the wrong axis order; transpose first so the fastest-varying axis
    # in the flat layout matches the grid's x axis.
    grid.point_data["intensity"] = arr.transpose(2, 1, 0).ravel(order="F")
    return grid


def mask_to_image_data(
    mask: Bool[np.ndarray, "z y x"],
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
) -> pv.ImageData:
    """Wrap a boolean mask as a :class:`pyvista.ImageData` (uint8 cast).

    Stored under the scalar name ``"mask"`` with values 0/1. Useful as
    a transfer-function gate or a translucent overlay.
    """
    import pyvista as pv

    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got ndim={mask.ndim}")
    if mask.dtype != np.bool_:
        raise TypeError(f"mask must be bool, got {mask.dtype}")

    arr = np.ascontiguousarray(mask).astype(np.uint8, copy=False)
    sp = normalize_spacing(spacing)

    nz, ny, nx = arr.shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(sp.values[2], sp.values[1], sp.values[0]),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["mask"] = arr.transpose(2, 1, 0).ravel(order="F")
    return grid


def field_to_polydata(
    field: DisplacementField,
    *,
    only_valid: bool = True,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    stride: int | None = None,
) -> pv.PolyData:
    """Build a :class:`pyvista.PolyData` carrying POI points + displacements.

    Points are placed at :attr:`DisplacementField.positions` (converted
    from voxel indices to physical coordinates via ``spacing``). The
    displacement vector becomes the active vectors array; ``valid``,
    ``status``, and ``confidence`` are attached as scalar arrays so a
    single conversion feeds every glyph / status / confidence
    renderer.

    Parameters
    ----------
    field
        Recovered displacement field.
    only_valid
        If True, drop POIs with ``valid == False`` from the output.
    spacing
        Voxel spacing; coordinates are scaled by ``spacing.values``.
        ``positions`` are stored ``(z, y, x)`` in voxel units; the
        polydata is written ``(x, y, z)`` in spacing units.
    stride
        Optional uniform decimation applied *after* the ``only_valid``
        filter: keep every ``stride``-th remaining POI. ``None`` or
        ``1`` keeps every POI. Must be a positive integer.

    Returns
    -------
    pyvista.PolyData
        Point cloud with arrays ``"displacement"`` (vectors),
        ``"magnitude"``, ``"valid"``, ``"status"``, ``"confidence"``.

    Raises
    ------
    ValueError
        If ``stride`` is provided and is not a positive integer.
    """
    import pyvista as pv

    if stride is not None and stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride}")

    sp = normalize_spacing(spacing)
    sx, sy, sz = sp.values[2], sp.values[1], sp.values[0]

    positions = field.positions
    displacements = field.displacements
    valid = field.valid
    status = field.status
    confidence = field.confidence

    if only_valid:
        keep = valid
        positions = positions[keep]
        displacements = displacements[keep]
        status = status[keep]
        confidence = confidence[keep]
        valid = valid[keep]

    if stride is not None and stride > 1:
        positions = positions[::stride]
        displacements = displacements[::stride]
        status = status[::stride]
        confidence = confidence[::stride]
        valid = valid[::stride]

    # (z, y, x) -> (x, y, z) and scale into spacing units.
    points = np.empty_like(positions)
    points[:, 0] = positions[:, 2] * sx
    points[:, 1] = positions[:, 1] * sy
    points[:, 2] = positions[:, 0] * sz

    vectors = np.empty_like(displacements)
    vectors[:, 0] = displacements[:, 2] * sx
    vectors[:, 1] = displacements[:, 1] * sy
    vectors[:, 2] = displacements[:, 0] * sz

    poly = pv.PolyData(points.astype(np.float32, copy=False))
    poly.point_data["displacement"] = vectors.astype(np.float32, copy=False)
    poly.point_data["magnitude"] = np.linalg.norm(vectors, axis=1).astype(
        np.float32, copy=False
    )
    poly.point_data["valid"] = valid.astype(np.uint8, copy=False)
    poly.point_data["status"] = status.astype(np.uint8, copy=False)
    poly.point_data["confidence"] = confidence.astype(np.float32, copy=False)
    poly.set_active_vectors("displacement")
    return poly


def grid_to_structured(
    grid: GridSpec,
    *,
    scalars: Float32[np.ndarray, "points"] | None = None,
    scalar_name: str = "scalar",
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
) -> pv.StructuredGrid:
    """Build a :class:`pyvista.StructuredGrid` over the regular POI lattice.

    Useful for scalar overlays (confidence, error magnitude) drawn as a
    colored grid rather than glyphs.

    Parameters
    ----------
    grid
        Source POI grid; ``grid.grid_shape`` defines the lattice.
    scalars
        Optional ``(points,)`` scalar array, attached under
        ``scalar_name``.
    scalar_name
        Name to register the scalar under.
    spacing
        Voxel spacing; positions are converted ``(z, y, x)`` voxel
        coords to ``(x, y, z)`` physical coords.

    Returns
    -------
    pyvista.StructuredGrid
        Lattice with the same point order as ``grid.positions``.
    """
    import pyvista as pv

    sp = normalize_spacing(spacing)
    sx, sy, sz = sp.values[2], sp.values[1], sp.values[0]

    nz, ny, nx = grid.grid_shape
    positions = grid.positions

    points = np.empty_like(positions)
    points[:, 0] = positions[:, 2] * sx
    points[:, 1] = positions[:, 1] * sy
    points[:, 2] = positions[:, 0] * sz

    structured = pv.StructuredGrid()
    structured.points = points.astype(np.float32, copy=False)
    structured.dimensions = (nx, ny, nz)
    if scalars is not None:
        if scalars.shape != (positions.shape[0],):
            raise ValueError(
                f"scalars shape {scalars.shape} does not match grid points "
                f"({positions.shape[0]},)"
            )
        structured.point_data[scalar_name] = scalars.astype(np.float32, copy=False)
    return structured


def downsample_for_render(
    volume: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    spacing: VoxelSpacing | None = None,
    factor: int = 2,
) -> tuple[
    Float32[np.ndarray, "z y x"],
    Bool[np.ndarray, "z y x"] | None,
    VoxelSpacing,
]:
    """Strided downsample of a host volume + mask, with matching scaled spacing.

    Strided slicing only (no resampling): the first voxel center stays
    at ``(0, 0, 0)`` so a :class:`DisplacementField` whose ``positions``
    are voxel indices into the *original* volume aligns physically with
    the returned ``volume_ds`` when both are passed to PyVista with their
    respective :class:`VoxelSpacing` — the field with the *original*
    spacing, the volume with the returned ``spacing_ds``.

    Parameters
    ----------
    volume
        ``(z, y, x)`` float32 host volume.
    mask
        Optional same-shape boolean mask; downsampled with the same
        stride so it stays voxel-aligned with ``volume_ds``.
    spacing
        Voxel spacing of ``volume``. ``None`` is treated as the
        unit-voxel spacing (see :func:`normalize_spacing`).
    factor
        Strided-downsample factor. Must be a positive integer; ``1``
        is a no-op pass-through.

    Returns
    -------
    tuple
        ``(volume_ds, mask_ds, spacing_ds)``. ``mask_ds`` is ``None``
        when ``mask`` was ``None``. ``spacing_ds`` carries values
        scaled by ``factor`` along every axis and the same ``unit`` as
        the input spacing.

    Raises
    ------
    ValueError
        If ``factor`` is less than 1, ``volume`` is not 3D, or
        ``mask.shape`` does not match ``volume.shape``.

    Notes
    -----
    Strided slicing is intentional. ``scipy.ndimage.zoom`` and similar
    resamplers can shift the first voxel center by up to half a voxel,
    which puts the field glyphs and the bone subtly out of register —
    exactly the alignment failure mode this primitive is supposed to
    rule out. The cost is no anti-aliasing: when ``factor`` is large
    (≥ 4) and the volume carries high-frequency content, the render
    will alias, but never drift.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got ndim={volume.ndim}")
    if mask is not None and mask.shape != volume.shape:
        raise ValueError(f"mask shape {mask.shape} does not match volume shape {volume.shape}")

    sp = normalize_spacing(spacing)

    if factor == 1:
        return volume, mask, sp

    vol_ds = volume[::factor, ::factor, ::factor]
    mask_ds = mask[::factor, ::factor, ::factor] if mask is not None else None
    spacing_ds = VoxelSpacing(
        (sp.values[0] * factor, sp.values[1] * factor, sp.values[2] * factor),
        sp.unit,
    )
    return vol_ds, mask_ds, spacing_ds
