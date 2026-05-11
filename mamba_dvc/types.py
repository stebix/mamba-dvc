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
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32, Int64, UInt8

__all__ = ["DisplacementField", "GridSpec", "POIStatus", "PhysicalUnit", "VoxelSpacing"]


PhysicalUnit = Literal["voxel", "nm", "um", "mm"]
"""Physical unit attached to a :class:`VoxelSpacing`.

``"voxel"`` is the unit-agnostic default for synthetic / pre-IO data;
``"nm" | "um" | "mm"`` are the physical units that flow in once
``mamba_dvc.io`` lands. Visualization unit math (the only place this is
read at runtime) lives in :mod:`mamba_dvc.viz._conversion`.
"""


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

    def save_npz(self, path: Path | str) -> None:
        """Persist this field to a ``.npz`` archive.

        Writes the per-POI arrays (``positions``, ``displacements``,
        ``valid``, ``confidence``, ``status``) plus the lattice scalars
        (``grid_shape``, ``spacing``, ``window``) as small int arrays.
        The schema matches what ``scripts/run_e2e_zarr.py --out`` emits,
        so existing notebooks load both. ``numpy.savez`` appends the
        ``.npz`` suffix if ``path`` lacks it.

        Parameters
        ----------
        path
            Destination path. The parent directory must already exist.
        """
        np.savez(
            path,
            positions=self.positions,
            displacements=self.displacements,
            valid=self.valid,
            confidence=self.confidence,
            status=self.status,
            grid_shape=np.asarray(self.grid_shape, dtype=np.int64),
            spacing=np.asarray(self.spacing, dtype=np.int64),
            window=np.asarray(self.window, dtype=np.int64),
        )

    @classmethod
    def load_npz(cls, path: Path | str) -> DisplacementField:
        """Reconstruct a :class:`DisplacementField` from :meth:`save_npz` output.

        Parameters
        ----------
        path
            Path to a ``.npz`` archive written by :meth:`save_npz` (or
            the equivalent layout from ``scripts/run_e2e_zarr.py``).

        Returns
        -------
        DisplacementField
            The frozen field. Array dtypes are coerced back to the
            canonical types (float32 / bool / uint8); the lattice
            scalars are returned as plain ``int`` 3-tuples.

        Raises
        ------
        KeyError
            If the archive is missing one of the expected entries.
        """
        with np.load(path) as data:
            gs = data["grid_shape"]
            sp = data["spacing"]
            win = data["window"]
            return cls(
                positions=np.ascontiguousarray(data["positions"], dtype=np.float32),
                displacements=np.ascontiguousarray(data["displacements"], dtype=np.float32),
                valid=np.ascontiguousarray(data["valid"], dtype=np.bool_),
                confidence=np.ascontiguousarray(data["confidence"], dtype=np.float32),
                status=np.ascontiguousarray(data["status"], dtype=np.uint8),
                grid_shape=(int(gs[0]), int(gs[1]), int(gs[2])),
                spacing=(int(sp[0]), int(sp[1]), int(sp[2])),
                window=(int(win[0]), int(win[1]), int(win[2])),
            )


@dataclass(frozen=True)
class VoxelSpacing:
    """Per-axis voxel spacing with an explicit physical unit.

    Carried alongside volumes (and emitted by ``mamba_dvc.io.volume``
    once it lands). The numeric :attr:`values` are unitless; :attr:`unit`
    names the physical interpretation. The default ``unit="voxel"``
    keeps synthetic / pre-IO data unit-agnostic.

    Parameters
    ----------
    values
        ``(z, y, x)`` spacings, strictly positive. Same dtype as the
        host ``float`` (Python floats, kept as a tuple for hashability).
    unit
        Physical interpretation of :attr:`values`. One of
        ``"voxel" | "nm" | "um" | "mm"``. Defaults to ``"voxel"``.

    Raises
    ------
    ValueError
        If :attr:`values` does not have length 3 or contains a
        non-positive entry.

    Notes
    -----
    All unit arithmetic is centralized in
    :func:`mamba_dvc.viz._conversion.unify_spacing`; renderers consume
    :class:`VoxelSpacing` opaquely. Mixed ``"voxel"`` + physical-unit
    composition is a hard error there, not a silent conversion.

    Per-axis differing units (e.g. nm in xy, µm in z on legacy TIFF
    stacks) are deferred. The dataclass shape leaves room to extend
    :attr:`unit` to a 3-tuple without breaking signatures.
    """

    values: tuple[float, float, float]
    unit: PhysicalUnit = "voxel"

    def __post_init__(self) -> None:
        """Validate spacing length and positivity."""
        if len(self.values) != 3:
            raise ValueError(f"values must have length 3, got {len(self.values)}")
        if any(v <= 0.0 for v in self.values):
            raise ValueError(f"values entries must be positive, got {self.values}")
