"""Volume rendering of µCT volumes.

Renderers add to a caller-supplied :class:`pyvista.Plotter` and return
it for composition. They never call ``.show()`` themselves; that stays
the caller's responsibility so the same code works for interactive
notebook use and headless screenshot export.

The two entry points cover the v1 needs from
``docs/plans/visualization.md`` §3:

- :func:`render_volume` for GPU raycast rendering with a transfer
  function.
- :func:`render_isosurface` for surface extraction at a chosen
  intensity threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.types import VoxelSpacing
from mamba_dvc.viz._conversion import volume_to_image_data

if TYPE_CHECKING:
    import pyvista as pv


__all__ = ["render_isosurface", "render_volume"]


def render_volume(
    plot: pv.Plotter,
    volume: Float32[np.ndarray, "z y x"],
    *,
    mask: Bool[np.ndarray, "z y x"] | None = None,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    opacity: str | Sequence[float] = "sigmoid",
    cmap: str = "bone",
    clim: tuple[float, float] | None = None,
) -> pv.Plotter:
    """Add a volume raycast rendering of ``volume`` to ``plot``.

    Parameters
    ----------
    plot
        Existing Plotter; mutated in place.
    volume
        ``(z, y, x)`` float32 reference or deformed volume.
    mask
        Optional ``(z, y, x)`` boolean mask. When provided, voxels with
        ``mask == False`` are zeroed before upload so the transfer
        function does not light them up. The original array is not
        mutated.
    spacing
        Voxel spacing; see
        :func:`mamba_dvc.viz._conversion.normalize_spacing`.
    opacity
        Transfer function passed through to PyVista's ``add_volume``.
        Strings (e.g. ``"sigmoid"``, ``"linear"``) and per-bin
        sequences both work.
    cmap
        Color map name.
    clim
        Optional ``(low, high)`` clipping range for the transfer
        function. ``None`` lets PyVista pick from the data extents.

    Returns
    -------
    pyvista.Plotter
        The same plotter, for chaining.
    """
    if mask is not None:
        if mask.shape != volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {volume.shape}"
            )
        rendered = np.where(mask, volume, np.float32(0.0)).astype(np.float32, copy=False)
    else:
        rendered = volume

    grid = volume_to_image_data(rendered, spacing=spacing)
    plot.add_volume(
        grid,
        scalars="intensity",
        cmap=cmap,
        opacity=opacity,
        clim=clim,
    )
    return plot


def render_isosurface(
    plot: pv.Plotter,
    volume: Float32[np.ndarray, "z y x"],
    *,
    level: float,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    color: str = "lightgray",
    opacity: float = 1.0,
) -> pv.Plotter:
    """Add an isosurface of ``volume`` at intensity ``level`` to ``plot``.

    Parameters
    ----------
    plot
        Existing Plotter; mutated in place.
    volume
        ``(z, y, x)`` float32 source.
    level
        Iso-value to extract.
    spacing
        Voxel spacing; see
        :func:`mamba_dvc.viz._conversion.normalize_spacing`.
    color
        Surface color.
    opacity
        Surface opacity in ``[0, 1]``.

    Returns
    -------
    pyvista.Plotter
        The same plotter, for chaining.
    """
    grid = volume_to_image_data(volume, spacing=spacing)
    surface = grid.contour(isosurfaces=[float(level)], scalars="intensity")
    plot.add_mesh(surface, color=color, opacity=opacity)
    return plot
