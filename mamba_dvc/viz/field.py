"""Displacement-field renderers for the viz submodule.

POI-level rendering primitives that consume a
:class:`~mamba_dvc.types.DisplacementField` (and optionally a sampled
ground-truth array) and register actors on a caller-supplied
:class:`pyvista.Plotter`. Match the contract of the volume renderers
in :mod:`mamba_dvc.viz.volume`: take the plotter, mutate it, return it
for chaining; never call ``.show()`` internally.

The three primitives split by visual idiom rather than by data shape:

- :func:`render_field` draws displacement glyphs (arrows) at every
  surviving POI.
- :func:`render_error_glyphs` draws ``recovered - truth`` glyphs at
  every surviving POI. Kept distinct from :func:`render_field`
  because the reasonable colormap, glyph factor, and color limits
  are an order of magnitude different — overloading them on a
  ``mode=`` flag would invite mistakes.
- :func:`render_field_lattice` colors the regular POI lattice as a
  :class:`pyvista.StructuredGrid`, useful for confidence / status /
  magnitude maps where the direction of the displacement is not the
  thing being communicated.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import numpy as np
from jaxtyping import Float32

from mamba_dvc.types import DisplacementField, VoxelSpacing
from mamba_dvc.viz._conversion import field_to_polydata, normalize_spacing

if TYPE_CHECKING:
    import pyvista as pv


__all__ = ["render_error_glyphs", "render_field", "render_field_lattice"]


_LatticeScalar = Literal["magnitude", "confidence", "status", "valid"]


def render_field(
    plot: pv.Plotter,
    field: DisplacementField,
    *,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    only_valid: bool = True,
    scale: Literal["magnitude", "uniform"] = "magnitude",
    factor: float = 1.0,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    stride: int | None = None,
    scalar_bar_kw: dict[str, object] | None = None,
) -> pv.Plotter:
    """Add displacement-vector glyphs to ``plot``.

    Parameters
    ----------
    plot
        Existing Plotter; mutated in place.
    field
        Recovered displacement field. Only POIs with
        ``status == POIStatus.OK`` are drawn by default.
    spacing
        Voxel spacing for converting POI voxel indices to physical
        coordinates and scaling displacement vectors. See
        :func:`mamba_dvc.viz._conversion.normalize_spacing`.
    only_valid
        Drop ``valid == False`` POIs before glyphing. Default ``True``.
    scale
        ``"magnitude"`` scales glyph length by ``||u||`` (default);
        ``"uniform"`` draws every glyph at the same length. Both are
        further amplified by ``factor``.
    factor
        Visual amplification applied on top of ``scale``. Held at
        ``1.0`` so screenshots are reproducible — callers pick a value
        appropriate to the typical displacement magnitude.
    cmap
        Color map name used for the magnitude scalar.
    clim
        Optional ``(low, high)`` clipping range for the colormap.
    stride
        Optional uniform decimation applied *after* the ``only_valid``
        filter (every ``stride``-th POI is kept). Useful on dense
        lattices where glyph overdraw hides structure.
    scalar_bar_kw
        Forwarded to PyVista as ``scalar_bar_args``. ``None`` lets
        PyVista pick defaults.

    Returns
    -------
    pyvista.Plotter
        The same plotter, for chaining.
    """
    poly = field_to_polydata(field, only_valid=only_valid, spacing=spacing, stride=stride)
    glyph = poly.glyph(
        orient="displacement",
        scale="magnitude" if scale == "magnitude" else False,
        factor=float(factor),
    )
    plot.add_mesh(
        glyph,
        scalars="magnitude",
        cmap=cmap,
        clim=clim,
        scalar_bar_args=scalar_bar_kw,
    )
    return plot


def render_error_glyphs(
    plot: pv.Plotter,
    field: DisplacementField,
    truth: Float32[np.ndarray, "points 3"],
    *,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    only_valid: bool = True,
    factor: float = 1.0,
    cmap: str = "magma",
    clim: tuple[float, float] | None = None,
    stride: int | None = None,
    scalar_bar_kw: dict[str, object] | None = None,
) -> pv.Plotter:
    """Add ``recovered - truth`` glyphs to ``plot``.

    Parameters
    ----------
    plot
        Existing Plotter; mutated in place.
    field
        Recovered displacement field.
    truth
        ``(points, 3)`` ``(dz, dy, dx)`` array sampled at
        ``field.positions``. Analytical fields call
        ``truth_fn(field.positions)`` once before invoking this
        primitive — keeps the contract free of a
        :class:`mamba_dvc.validate.synthetic.DisplacementFunction`
        dependency.
    spacing
        Voxel spacing; see :func:`render_field`.
    only_valid
        Drop ``valid == False`` POIs before glyphing.
    factor
        Visual amplification. Error magnitudes are typically two
        decades smaller than displacement magnitudes, so reasonable
        values here are correspondingly larger.
    cmap
        Color map name. Defaults to ``"magma"`` to contrast against
        the ``"viridis"`` default of :func:`render_field`.
    clim
        Optional ``(low, high)`` clipping range for the colormap.
    stride
        Optional uniform decimation; see :func:`render_field`.
    scalar_bar_kw
        Forwarded as ``scalar_bar_args``.

    Returns
    -------
    pyvista.Plotter
        The same plotter, for chaining.

    Raises
    ------
    ValueError
        If ``truth.shape`` does not match ``field.displacements.shape``.
    """
    if truth.shape != field.displacements.shape:
        raise ValueError(
            f"truth shape {truth.shape} does not match field.displacements "
            f"shape {field.displacements.shape}"
        )
    error = (field.displacements - truth.astype(np.float32, copy=False)).astype(
        np.float32, copy=False
    )
    # The dataclass is frozen but ``replace`` returns a fresh instance.
    # __post_init__ only checks ``valid == (status == OK)`` — untouched here.
    error_field = dataclasses.replace(field, displacements=error)
    poly = field_to_polydata(
        error_field, only_valid=only_valid, spacing=spacing, stride=stride
    )
    glyph = poly.glyph(
        orient="displacement",
        scale="magnitude",
        factor=float(factor),
    )
    plot.add_mesh(
        glyph,
        scalars="magnitude",
        cmap=cmap,
        clim=clim,
        scalar_bar_args=scalar_bar_kw,
    )
    return plot


def render_field_lattice(
    plot: pv.Plotter,
    field: DisplacementField,
    *,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    scalar: _LatticeScalar | None = "magnitude",
    scalar_array: Float32[np.ndarray, "points"] | None = None,
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    opacity: float = 1.0,
) -> pv.Plotter:
    """Add a colored :class:`pyvista.StructuredGrid` over the POI lattice.

    A different visual idiom than :func:`render_field`: instead of
    per-POI glyphs, the regular lattice is rendered as a continuous
    colored surface. Useful when the *direction* of the displacement
    is not the point — confidence maps, status overlays, or per-POI
    error scalars.

    Parameters
    ----------
    plot
        Existing Plotter; mutated in place.
    field
        Source field. Its ``grid_shape`` must agree with
        ``positions.shape[0]``.
    spacing
        Voxel spacing for converting POI voxel indices to physical
        coordinates.
    scalar
        Which per-POI attribute of ``field`` to color by:
        ``"magnitude"`` (default), ``"confidence"``, ``"status"``,
        or ``"valid"``. Set to ``None`` together with
        ``scalar_array=None`` to draw the bare lattice in a single
        flat color.
    scalar_array
        Escape hatch for derived scalars (e.g. per-POI error magnitude).
        Mutually exclusive with ``scalar``; pass ``scalar=None`` when
        using this. Must have shape ``(positions,)``.
    cmap
        Color map name. Ignored when neither ``scalar`` nor
        ``scalar_array`` is supplied.
    clim
        Optional ``(low, high)`` clipping range for the colormap.
    opacity
        Surface opacity in ``[0, 1]``.

    Returns
    -------
    pyvista.Plotter
        The same plotter, for chaining.

    Raises
    ------
    ValueError
        If ``scalar`` and ``scalar_array`` are both supplied, if
        ``scalar_array`` has the wrong shape, or if
        ``field.grid_shape`` and ``field.positions`` disagree.
    """
    import pyvista as pv

    if scalar_array is not None and scalar is not None:
        raise ValueError(
            "pass either `scalar` (named field attribute) or `scalar_array` "
            "(precomputed values), not both"
        )

    nz, ny, nx = field.grid_shape
    positions = field.positions
    expected = nz * ny * nx
    if positions.shape[0] != expected:
        raise ValueError(
            f"field.positions has {positions.shape[0]} entries but grid_shape "
            f"{field.grid_shape} requires {expected}"
        )

    sp = normalize_spacing(spacing)
    sx, sy, sz = sp.values[2], sp.values[1], sp.values[0]

    points = np.empty_like(positions)
    points[:, 0] = positions[:, 2] * sx
    points[:, 1] = positions[:, 1] * sy
    points[:, 2] = positions[:, 0] * sz

    structured = pv.StructuredGrid()
    structured.points = points.astype(np.float32, copy=False)
    structured.dimensions = (nx, ny, nz)

    name: str | None
    values: np.ndarray | None
    if scalar_array is not None:
        if scalar_array.shape != (positions.shape[0],):
            raise ValueError(
                f"scalar_array shape {scalar_array.shape} does not match "
                f"({positions.shape[0]},)"
            )
        values = np.ascontiguousarray(scalar_array, dtype=np.float32)
        name = "scalar"
    elif scalar == "magnitude":
        values = np.linalg.norm(field.displacements, axis=1).astype(np.float32, copy=False)
        name = "magnitude"
    elif scalar == "confidence":
        values = field.confidence.astype(np.float32, copy=False)
        name = "confidence"
    elif scalar == "status":
        values = field.status.astype(np.float32, copy=False)
        name = "status"
    elif scalar == "valid":
        values = field.valid.astype(np.float32, copy=False)
        name = "valid"
    else:
        values = None
        name = None

    if values is None or name is None:
        plot.add_mesh(structured, color="lightgray", opacity=opacity)
        return plot

    structured.point_data[name] = values
    plot.add_mesh(structured, scalars=name, cmap=cmap, clim=clim, opacity=opacity)
    return plot
