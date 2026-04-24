"""POI lattice construction and mask-aware admission.

This module owns the very first step of the DVC pipeline (plan
``docs/plans/overview.md`` §2, step 1): emit points-of-interest (POIs)
on a regular lattice over the reference volume, and drop candidates
whose subvolume does not carry enough valid-mask support.

Everything here is host-side NumPy. POI construction runs once per
pair, before any GPU sharding, so there is no benefit to moving it to
CuPy.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool

from mamba_dvc.types import GridSpec

__all__ = ["build_grid", "filter_by_mask"]


def _normalize_window(window: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Return an ``(wz, wy, wx)`` tuple from a scalar or 3-tuple input."""
    if isinstance(window, int):
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        return (window, window, window)

    if len(window) != 3:
        raise ValueError(f"window tuple must have length 3, got {len(window)}")
    wz, wy, wx = (int(w) for w in window)
    if wz <= 0 or wy <= 0 or wx <= 0:
        raise ValueError(f"window entries must be positive, got {window}")
    return (wz, wy, wx)


def _strides_from_overlap(
    window: tuple[int, int, int], overlap: float
) -> tuple[int, int, int]:
    """Compute integer per-axis strides from window size and overlap fraction."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")
    strides = tuple(max(1, round(w * (1.0 - overlap))) for w in window)
    return (strides[0], strides[1], strides[2])


def build_grid(
    volume_shape: tuple[int, int, int],
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
) -> GridSpec:
    """Build a regular POI lattice over a volume of the given shape.

    Parameters
    ----------
    volume_shape
        ``(z, y, x)`` shape of the reference volume in voxels.
    window
        Subvolume size in voxels. Scalar applies isotropically;
        otherwise a per-axis tuple ``(wz, wy, wx)``.
    overlap
        Fraction of the window shared between neighboring POIs along
        each axis. ``overlap=0`` gives non-overlapping subvolumes;
        ``overlap=0.5`` is the pipeline default. Must lie in
        ``[0.0, 1.0)``.

    Returns
    -------
    GridSpec
        Lattice descriptor with ``positions``, ``starts``, ``grid_shape``,
        ``spacing``, ``window``, and ``volume_shape`` populated.

    Raises
    ------
    ValueError
        If ``volume_shape`` is not 3-tuple with positive entries, if
        ``window`` has a non-positive entry, if ``overlap`` is out of
        range, or if ``window`` exceeds ``volume_shape`` on any axis.

    Notes
    -----
    POI counts per axis follow ``n_i = floor((S_i - W_i) / stride_i) + 1``
    with ``stride_i = max(1, round(W_i * (1 - overlap)))``. POI centers
    use the geometric midpoint convention ``start + (W - 1) / 2``.
    """
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must have length 3, got {len(volume_shape)}")
    vz, vy, vx = (int(s) for s in volume_shape)
    if vz <= 0 or vy <= 0 or vx <= 0:
        raise ValueError(f"volume_shape entries must be positive, got {volume_shape}")

    win = _normalize_window(window)
    if any(w > s for w, s in zip(win, (vz, vy, vx), strict=True)):
        raise ValueError(
            f"window {win} exceeds volume_shape {(vz, vy, vx)} on at least one axis"
        )

    stride = _strides_from_overlap(win, overlap)

    counts = tuple(
        ((s - w) // st) + 1 for s, w, st in zip((vz, vy, vx), win, stride, strict=True)
    )
    nz, ny, nx = counts

    starts_z = np.arange(nz, dtype=np.int64) * stride[0]
    starts_y = np.arange(ny, dtype=np.int64) * stride[1]
    starts_x = np.arange(nx, dtype=np.int64) * stride[2]

    zz, yy, xx = np.meshgrid(starts_z, starts_y, starts_x, indexing="ij")
    starts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    half = np.asarray(
        [(win[0] - 1) / 2.0, (win[1] - 1) / 2.0, (win[2] - 1) / 2.0],
        dtype=np.float32,
    )
    positions = starts.astype(np.float32) + half

    return GridSpec(
        positions=positions,
        starts=starts,
        grid_shape=(nz, ny, nx),
        spacing=stride,
        window=win,
        volume_shape=(vz, vy, vx),
    )


def filter_by_mask(
    grid: GridSpec,
    mask: Bool[np.ndarray, "z y x"],
    threshold: float = 0.9,
) -> Bool[np.ndarray, "points"]:
    """Admit POIs whose subvolume has enough valid-mask support.

    Parameters
    ----------
    grid
        Lattice produced by :func:`build_grid`.
    mask
        Shared binary mask over the volume (``True`` = valid tissue,
        ``False`` = screw / excluded). Shape must match
        ``grid.volume_shape``. Must have boolean dtype.
    threshold
        Minimum valid-voxel fraction for a POI to be admitted. A POI is
        valid iff ``fraction >= threshold``. Must lie in ``[0.0, 1.0]``.

    Returns
    -------
    numpy.ndarray
        ``(points,)`` boolean array, aligned with ``grid.positions``.

    Raises
    ------
    ValueError
        If ``mask.shape`` does not match ``grid.volume_shape``, if
        ``mask.dtype`` is not boolean, or if ``threshold`` is out of
        range.

    Notes
    -----
    Uses a 3D summed-area table (integral image) so the per-POI
    subvolume sum is an eight-corner inclusion-exclusion lookup,
    vectorized over all POIs. Build cost is ``O(z*y*x)`` once; lookup
    cost is ``O(points)``.
    """
    if mask.shape != grid.volume_shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match grid.volume_shape {grid.volume_shape}"
        )
    if mask.dtype != np.bool_:
        raise ValueError(f"mask must have bool dtype, got {mask.dtype}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    sat = _integral_image_3d(mask)

    starts = grid.starts
    wz, wy, wx = grid.window
    z0 = starts[:, 0]
    y0 = starts[:, 1]
    x0 = starts[:, 2]
    z1 = z0 + wz
    y1 = y0 + wy
    x1 = x0 + wx

    counts = _subvolume_sums(sat, z0, y0, x0, z1, y1, x1)
    total = np.float64(wz) * np.float64(wy) * np.float64(wx)
    fraction = counts.astype(np.float64) / total

    return fraction >= threshold


def _integral_image_3d(mask: Bool[np.ndarray, "z y x"]) -> np.ndarray:
    """Return a zero-padded 3D summed-area table of ``mask`` as ``uint32``.

    The returned array has shape ``(Z+1, Y+1, X+1)`` with the first
    slab along every axis set to zero, so the inclusion-exclusion
    lookup at a subvolume ``[z0:z1, y0:y1, x0:x1]`` reduces to eight
    indexed reads without any boundary branching.

    ``uint32`` holds voxel counts up to ``2**32 - 1``, comfortably
    above the production volume's ``Z*Y*X`` (``~1.57e9``).
    """
    z, y, x = mask.shape
    inner = mask.astype(np.uint32, copy=True)
    np.cumsum(inner, axis=0, out=inner)
    np.cumsum(inner, axis=1, out=inner)
    np.cumsum(inner, axis=2, out=inner)
    sat = np.zeros((z + 1, y + 1, x + 1), dtype=np.uint32)
    sat[1:, 1:, 1:] = inner
    return sat


def _subvolume_sums(
    sat: np.ndarray,
    z0: np.ndarray,
    y0: np.ndarray,
    x0: np.ndarray,
    z1: np.ndarray,
    y1: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    """Compute subvolume sums from a zero-padded 3D integral image."""
    return (
        sat[z1, y1, x1]
        - sat[z0, y1, x1]
        - sat[z1, y0, x1]
        - sat[z1, y1, x0]
        + sat[z0, y0, x1]
        + sat[z0, y1, x0]
        + sat[z1, y0, x0]
        - sat[z0, y0, x0]
    )
