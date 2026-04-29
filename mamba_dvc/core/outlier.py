"""Normalized-median outlier detection on a regular POI lattice.

Implements step 7 of the single-pass DVC pipeline (plan
``docs/plans/overview.md`` §2, ``docs/plans/outlier.md``). Consumes the
flat ``(points,)`` displacement and validity arrays produced by
:mod:`mamba_dvc.pipeline.correlate` after the search-radius gate, and
returns a flag mask of POIs whose displacement is statistically
inconsistent with their lattice neighbors under the
Westerweel & Scarano (2005) normalized median test.

Host-side NumPy only. The lattice is small (~12 k POIs at v1 defaults)
and a vectorized ``np.nanmedian`` over a 3x3x3 sliding window beats any
GPU offload after accounting for transfer overhead.
"""

from __future__ import annotations

import warnings

import numpy as np
from jaxtyping import Bool, Float32
from numpy.lib.stride_tricks import sliding_window_view

from mamba_dvc.types import GridSpec

__all__ = ["detect_outliers"]


def detect_outliers(
    grid: GridSpec,
    displacements: Float32[np.ndarray, "points 3"],
    valid: Bool[np.ndarray, "points"],
    *,
    epsilon: float = 2.0,
    noise_floor: float = 0.1,
    min_neighbors: int = 4,
) -> Bool[np.ndarray, "points"]:
    """Flag POIs whose displacement fails the normalized median test.

    Parameters
    ----------
    grid
        Lattice descriptor produced by
        :func:`mamba_dvc.core.grid.build_grid`. Only ``grid_shape`` is
        consulted; the C-order ``(nz, ny, nx)`` reshape recovers the
        regular lattice.
    displacements
        ``(points, 3)`` float32 ``(dz, dy, dx)`` per-POI displacement.
        Non-OK POIs are expected to carry zero displacement, but the
        verb does not depend on that — it consults ``valid`` for
        admission.
    valid
        ``(points,)`` boolean validity mask at the time of the test.
        ``True`` iff the POI is currently ``POIStatus.OK`` (i.e. it
        passed mask admission and the search-radius gate). Invalid
        POIs are excluded from every neighborhood and never flagged.
    epsilon
        Threshold on the normalized residual; default ``2.0`` is the
        Westerweel & Scarano (2005) canonical value used throughout the
        PIV/DVC literature.
    noise_floor
        Constant added to the residual MAD before division, in voxel
        units. Default ``0.1`` matches the canonical reference and
        defends against flat regions where the local MAD is zero.
    min_neighbors
        Minimum number of valid neighbors required for the test to be
        applied. POIs with fewer valid neighbors are left ``False``
        regardless of displacement, so the lattice boundary is not
        biased toward "outlier" by an under-populated sample.

    Returns
    -------
    numpy.ndarray
        ``(points,)`` boolean array. ``True`` iff the POI is flagged
        as an outlier. Only POIs with ``valid[p] == True`` and a
        sufficiently populated neighborhood can be flagged.

    Raises
    ------
    ValueError
        If ``displacements`` does not have shape ``(points, 3)``, if
        ``valid`` does not have shape ``(points,)``, if dtypes are
        wrong, if ``np.prod(grid.grid_shape)`` does not equal
        ``points``, or if any of ``epsilon``, ``noise_floor``,
        ``min_neighbors`` is non-positive.

    Notes
    -----
    For each lattice index ``p`` and each component ``c in {z, y, x}``::

        u_med(p)  = median { d[q, c]           : q in N(p), valid[q] }
        r_i(p)    =        | d[q_i, c] - u_med | for each q_i in N(p)
        r_med(p)  = median r_i(p)
        rho_c(p)  = | d[p, c] - u_med | / ( r_med + noise_floor )

    where ``N(p)`` is the 3x3x3 neighborhood of ``p`` minus the centre
    voxel itself. A POI is flagged iff ``max_c rho_c(p) > epsilon``
    *and* its neighborhood has at least ``min_neighbors`` valid
    entries.

    The vectorized implementation pads the lattice with NaN, uses
    :func:`numpy.lib.stride_tricks.sliding_window_view` to materialize
    a ``(nz, ny, nx, 3, 3, 3, 3)`` view (the trailing 3 is the
    component axis), masks invalid neighbors to NaN, and computes
    medians via :func:`numpy.nanmedian` along the three neighborhood
    axes. Memory: a few MB at v1 lattice scales. Runtime: single-digit
    milliseconds.

    References
    ----------
    Westerweel, J. & Scarano, F. (2005), "Universal outlier detection
    for PIV data", Experiments in Fluids 39:1096-1100.
    """
    if displacements.ndim != 2 or displacements.shape[1] != 3:
        raise ValueError(
            f"displacements must have shape (points, 3), got {displacements.shape}"
        )
    if displacements.dtype != np.float32:
        raise ValueError(f"displacements must be float32, got {displacements.dtype}")
    if valid.ndim != 1:
        raise ValueError(f"valid must be 1D, got ndim={valid.ndim}")
    if valid.dtype != np.bool_:
        raise ValueError(f"valid must have bool dtype, got {valid.dtype}")
    if valid.shape[0] != displacements.shape[0]:
        raise ValueError(
            f"valid length {valid.shape[0]} does not match displacements length "
            f"{displacements.shape[0]}"
        )
    expected_points = int(np.prod(grid.grid_shape))
    if displacements.shape[0] != expected_points:
        raise ValueError(
            f"displacements length {displacements.shape[0]} does not match "
            f"prod(grid.grid_shape)={expected_points}"
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if noise_floor <= 0.0:
        raise ValueError(f"noise_floor must be positive, got {noise_floor}")
    if min_neighbors <= 0:
        raise ValueError(f"min_neighbors must be positive, got {min_neighbors}")

    nz, ny, nx = grid.grid_shape
    n_points = nz * ny * nx

    # Reshape to the regular lattice. C-order over (nz, ny, nx) matches
    # the order build_grid emits.
    d_lat = displacements.reshape(nz, ny, nx, 3)
    v_lat = valid.reshape(nz, ny, nx)

    # Neighbors of an invalid POI must not contribute to any other
    # POI's median. We achieve this by masking invalid entries to NaN
    # in a working copy *before* building the sliding view.
    d_masked = np.where(v_lat[..., None], d_lat, np.float32(np.nan))

    # Pad with NaN on the spatial axes so the sliding view has length
    # nz, ny, nx along the outer axes. Pad shape: pad widths are 1 on
    # each spatial axis, 0 on the component axis.
    d_padded = np.pad(
        d_masked,
        pad_width=((1, 1), (1, 1), (1, 1), (0, 0)),
        mode="constant",
        constant_values=np.float32(np.nan),
    )

    # Sliding window over the three spatial axes only. Resulting shape:
    # (nz, ny, nx, 3, 3, 3, 3) -- the last "3" is the component axis,
    # the three preceding 3s are the neighborhood window axes (in
    # z, y, x order).
    win = sliding_window_view(d_padded, window_shape=(3, 3, 3), axis=(0, 1, 2))  # pyright: ignore[reportCallIssue, reportArgumentType]
    # win.shape == (nz, ny, nx, 3 components, 3, 3, 3 window axes)

    # Reorder so the component axis is last for ergonomic axis args:
    # (nz, ny, nx, 3, 3, 3, 3) -> (nz, ny, nx, 3, 3, 3, 3) where the
    # last axis is component. moveaxis(source=3, destination=-1).
    win = np.moveaxis(win, 3, -1)
    # win.shape == (nz, ny, nx, 3 win_z, 3 win_y, 3 win_x, 3 component)

    # Drop the centre (self) from each neighborhood. The centre maps to
    # window index (1, 1, 1) regardless of POI position. Set to NaN so
    # nanmedian ignores it.
    win = win.copy()  # sliding_window_view is read-only
    win[:, :, :, 1, 1, 1, :] = np.float32(np.nan)

    # All-NaN neighborhoods (every neighbor masked out) make
    # np.nanmedian / np.nanmax emit RuntimeWarnings. Those POIs are
    # filtered by the min_neighbors guard below, so the NaN propagation
    # here is intentional -- silence the noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="All-NaN axis encountered")

        # Per-component median across the 3x3x3 neighborhood (axes 3, 4, 5).
        u_med = np.nanmedian(win, axis=(3, 4, 5))
        # u_med.shape == (nz, ny, nx, 3 component)

        # Per-component absolute residuals over the same neighborhood,
        # then the median of those (the residual MAD).
        residuals = np.abs(win - u_med[:, :, :, None, None, None, :])
        r_med = np.nanmedian(residuals, axis=(3, 4, 5))
        # r_med.shape == (nz, ny, nx, 3 component)

        # Normalized residual per component.
        rho = np.abs(d_lat - u_med) / (r_med + np.float32(noise_floor))
        rho_max = np.nanmax(rho, axis=-1)  # (nz, ny, nx)

    # Count valid neighbors (excluding self) using a parallel sliding
    # view over the validity lattice.
    v_padded = np.pad(
        v_lat,
        pad_width=((1, 1), (1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )
    v_win = sliding_window_view(v_padded, window_shape=(3, 3, 3), axis=(0, 1, 2))  # pyright: ignore[reportCallIssue, reportArgumentType]
    # v_win.shape == (nz, ny, nx, 3, 3, 3); count and subtract self.
    n_neighbors = v_win.sum(axis=(3, 4, 5)) - v_lat.astype(np.int64)

    # Flag iff: POI itself is valid, has enough neighbors, and the
    # normalized residual exceeds the threshold. The np.where on rho_max
    # filters NaNs (which arise when r_med is NaN, i.e. fewer than two
    # valid neighbors per component) — those are guarded by the
    # min_neighbors test anyway, but the explicit nan_to_num avoids a
    # spurious comparison warning.
    rho_finite = np.where(np.isnan(rho_max), 0.0, rho_max)
    flag_lat = v_lat & (n_neighbors >= min_neighbors) & (rho_finite > epsilon)

    return flag_lat.reshape(n_points)
