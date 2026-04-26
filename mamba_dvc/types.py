"""Core dataclasses and shared shape-symbol vocabulary.

Shape symbols used throughout ``mamba_dvc`` public signatures (via
``jaxtyping``):

- ``z``, ``y``, ``x`` : voxel axes of a full volume.
- ``points`` : number of points-of-interest (POIs) or coordinate entries.
- ``w`` : subvolume window size (may be anisotropic in practice; ``w`` is
  the shared symbol for a per-axis window extent in annotations).

Keep this vocabulary stable -- downstream modules assume these symbols
carry the meanings above.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from jaxtyping import Bool, Float32, Int64, UInt8

__all__ = ["DisplacementField", "GridSpec", "POIStatus"]


class POIStatus(IntEnum):
    """Categorical reason a POI ended up in its final state.

    Stored per-POI in :attr:`DisplacementField.status` as ``uint8``.
    ``OK`` is ``0`` so a default-zero array initializes every POI as
    valid; downstream stages mark deviations.

    Members
    -------
    OK
        POI passed every gate and carries a usable displacement.
    MASKED
        POI was rejected by :func:`mamba_dvc.core.grid.filter_by_mask`;
        its subvolume did not have enough valid-mask support.
    OUTLIER
        Reserved for the normalized-median outlier test
        (``core/outlier.py``, plan §2 step 7) -- not populated yet in
        v1 until that module lands.
    LOW_CONF
        Reserved for confidence-threshold rejection by future driver
        layers; not populated yet in v1.
    OUT_OF_RANGE
        POI's integer-lag peak exceeded the configured
        ``search_radius``; the wrapped displacement is unreliable, so
        the orchestrator zeroes it and flags the POI.
    """

    OK = 0
    MASKED = 1
    OUTLIER = 2
    LOW_CONF = 3
    OUT_OF_RANGE = 4


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


@dataclass(frozen=True)
class DisplacementField:
    """Result of a single-pair DVC run on a regular POI lattice.

    Parameters
    ----------
    positions
        POI centers in voxel coordinates ``(z, y, x)``, float32.
        Identical to the corresponding :class:`GridSpec` field.
    displacements
        ``(points, 3)`` float32 ``(dz, dy, dx)`` displacement in
        voxels. Non-OK POIs (``status != POIStatus.OK``) carry zero
        displacement -- callers stratifying error must gate on
        :attr:`valid` first.
    valid
        ``(points,)`` boolean convenience view of
        ``status == POIStatus.OK``. Stored alongside ``status`` for
        ergonomics; the two never disagree.
    confidence
        ``(points,)`` float32 peak NCC value. Non-OK POIs carry zero
        confidence; meaningful only when ``valid`` is True.
    status
        ``(points,)`` uint8 :class:`POIStatus` value per POI. Use this
        to stratify failure modes for diagnostic plots.
    grid_shape
        ``(nz, ny, nx)`` POI counts per axis -- the lattice
        :func:`mamba_dvc.core.grid.build_grid` produced.
    spacing
        Grid stride per axis, in voxels.
    window
        Subvolume size per axis, in voxels.

    Notes
    -----
    ``displacements.reshape(*grid_shape, 3)`` recovers the regular
    lattice form expected by densification / visualization code.

    The ``frozen=True`` decoration keeps the result immutable; tests
    and validation reports build new arrays rather than mutating the
    return value.
    """

    positions: Float32[np.ndarray, "points 3"]
    displacements: Float32[np.ndarray, "points 3"]
    valid: Bool[np.ndarray, "points"]
    confidence: Float32[np.ndarray, "points"]
    status: UInt8[np.ndarray, "points"]
    grid_shape: tuple[int, int, int]
    spacing: tuple[int, int, int]
    window: tuple[int, int, int]
