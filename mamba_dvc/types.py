"""Core dataclasses and shared shape-symbol vocabulary.

Shape symbols used throughout ``mamba_dvc`` public signatures (via
``jaxtyping``):

- ``z``, ``y``, ``x`` : voxel axes of a full volume.
- ``points`` : number of points-of-interest (POIs) or coordinate entries.
- ``w`` : subvolume window size (may be anisotropic in practice; ``w`` is
  the shared symbol for a per-axis window extent in annotations).

Keep this vocabulary stable — downstream modules assume these symbols
carry the meanings above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Int64


@dataclass(frozen=True)
class GridSpec:
    """Regular lattice of POI centers over a reference volume.

    Parameters
    ----------
    positions
        POI centers in voxel coordinates ``(z, y, x)``, float32. Centers
        are the geometric midpoint of each subvolume:
        ``position = start + (window - 1) / 2``.
    starts
        Integer start indices ``(z, y, x)`` of each POI subvolume. Kept
        alongside ``positions`` so downstream extraction does not have
        to recompute them.
    grid_shape
        ``(nz, ny, nx)`` POI counts per axis. ``prod(grid_shape) == points``.
    spacing
        Grid stride per axis, in voxels. Derived from ``window`` and
        ``overlap`` at construction time.
    window
        Subvolume size per axis, in voxels.
    volume_shape
        Shape of the source reference volume, for downstream bounds
        checks.

    Notes
    -----
    ``grid_shape`` + ``spacing`` + ``positions`` are redundant by design:
    together they make v2 densification (reshape to a regular lattice
    and fit a B-spline) a one-liner.
    """

    positions: Float32[np.ndarray, "points 3"]
    starts: Int64[np.ndarray, "points 3"]
    grid_shape: tuple[int, int, int]
    spacing: tuple[int, int, int]
    window: tuple[int, int, int]
    volume_shape: tuple[int, int, int]
