"""Tests for ``mamba_dvc.core.outlier``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.grid import build_grid
from mamba_dvc.core.outlier import detect_outliers
from mamba_dvc.types import GridSpec

detect_outliers_checked = jaxtyped(typechecker=beartype)(detect_outliers)


def _scalar_reference(
    grid: GridSpec,
    displacements: np.ndarray,
    valid: np.ndarray,
    *,
    epsilon: float = 2.0,
    noise_floor: float = 0.1,
    min_neighbors: int = 4,
) -> np.ndarray:
    """Plain Python scalar implementation of Westerweel & Scarano (2005).

    Used to verify the vectorized version on small randomized lattices.
    """
    nz, ny, nx = grid.grid_shape
    d_lat = displacements.reshape(nz, ny, nx, 3)
    v_lat = valid.reshape(nz, ny, nx)
    flag = np.zeros((nz, ny, nx), dtype=np.bool_)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not v_lat[z, y, x]:
                    continue
                neighbors = []
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            zz, yy, xx = z + dz, y + dy, x + dx
                            if not (0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx):
                                continue
                            if not v_lat[zz, yy, xx]:
                                continue
                            neighbors.append(d_lat[zz, yy, xx])
                if len(neighbors) < min_neighbors:
                    continue
                neighbors_arr = np.array(neighbors, dtype=np.float32)
                rho_max = 0.0
                for c in range(3):
                    u_med = float(np.median(neighbors_arr[:, c]))
                    r_med = float(np.median(np.abs(neighbors_arr[:, c] - u_med)))
                    rho = abs(float(d_lat[z, y, x, c]) - u_med) / (r_med + noise_floor)
                    rho_max = max(rho_max, rho)
                if rho_max > epsilon:
                    flag[z, y, x] = True

    return flag.reshape(nz * ny * nx)


def _smooth_grid(nz: int = 5, ny: int = 5, nx: int = 5) -> GridSpec:
    """Build a small lattice with a known volume_shape compatible window."""
    # Pick volume_shape and window so build_grid emits exactly
    # (nz, ny, nx). With window=W and overlap=0.5 -> stride=W/2,
    # volume_shape = (n - 1) * stride + W.
    window = 8
    stride = window // 2
    volume_shape = (
        (nz - 1) * stride + window,
        (ny - 1) * stride + window,
        (nx - 1) * stride + window,
    )
    return build_grid(volume_shape, window=window, overlap=0.5)


class TestReferenceParity:
    def test_matches_scalar_loop_on_random_lattice(self):
        rng = np.random.default_rng(0)
        grid = _smooth_grid(5, 5, 5)
        n = int(np.prod(grid.grid_shape))
        # Mostly smooth, a few injected spikes to exercise both branches.
        d = rng.standard_normal((n, 3)).astype(np.float32) * 0.1
        spike_idx = rng.choice(n, size=3, replace=False)
        d[spike_idx] += np.float32(5.0)
        v = np.ones(n, dtype=np.bool_)
        # Mark a handful as invalid to exercise neighbor exclusion.
        v[rng.choice(n, size=5, replace=False)] = False

        got = detect_outliers(grid, d, v)
        expected = _scalar_reference(grid, d, v)

        np.testing.assert_array_equal(got, expected)


class TestSingleBadPoi:
    def test_one_spike_in_zero_field_is_flagged(self):
        grid = _smooth_grid(5, 5, 5)
        n = int(np.prod(grid.grid_shape))
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.ones(n, dtype=np.bool_)

        # Place a spike at an interior POI.
        bad = (2, 2, 2)
        bad_flat = np.ravel_multi_index(bad, grid.grid_shape)
        d[bad_flat] = np.float32(5.0)

        flag = detect_outliers(grid, d, v)

        assert bool(flag[bad_flat]) is True
        # Every other POI should remain unflagged.
        flag[bad_flat] = False
        assert not bool(flag.any())


class TestSmoothFields:
    def test_linear_gradient_is_not_flagged(self):
        grid = _smooth_grid(5, 5, 5)
        nz, ny, nx = grid.grid_shape
        # Linear field d(p) = 0.05 * (z, y, x) — well below ε at default
        # noise_floor.
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        d = 0.05 * np.stack([zz, yy, xx], axis=-1).reshape(-1, 3).astype(np.float32)
        v = np.ones(d.shape[0], dtype=np.bool_)

        flag = detect_outliers(grid, d, v)

        assert not bool(flag.any())

    def test_constant_field_is_not_flagged(self):
        grid = _smooth_grid(4, 4, 4)
        n = int(np.prod(grid.grid_shape))
        d = np.full((n, 3), 1.5, dtype=np.float32)
        v = np.ones(n, dtype=np.bool_)

        flag = detect_outliers(grid, d, v)

        assert not bool(flag.any())


class TestBoundary:
    def test_corner_poi_is_not_flagged_when_local_trend_holds(self):
        # 5x5x5 lattice with a smooth linear field; corner POIs have 7
        # valid neighbors (above min_neighbors), and their displacement
        # matches the local trend, so they must not be flagged.
        grid = _smooth_grid(5, 5, 5)
        nz, ny, nx = grid.grid_shape
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        d = 0.05 * np.stack([zz, yy, xx], axis=-1).reshape(-1, 3).astype(np.float32)
        v = np.ones(d.shape[0], dtype=np.bool_)

        flag = detect_outliers(grid, d, v)

        # All eight corners.
        for corner in [
            (0, 0, 0),
            (0, 0, nx - 1),
            (0, ny - 1, 0),
            (0, ny - 1, nx - 1),
            (nz - 1, 0, 0),
            (nz - 1, 0, nx - 1),
            (nz - 1, ny - 1, 0),
            (nz - 1, ny - 1, nx - 1),
        ]:
            i = np.ravel_multi_index(corner, grid.grid_shape)
            assert not bool(flag[i]), f"corner {corner} unexpectedly flagged"


class TestMaskAwareness:
    def test_invalid_neighbors_not_used_in_median(self):
        # Half-lattice valid=False on the +x side. POIs at the validity
        # boundary see neighbors only on the valid side; their median is
        # computed from genuine displacements, not zero-padded ghosts.
        grid = _smooth_grid(5, 5, 6)
        nz, ny, nx = grid.grid_shape
        n = nz * ny * nx
        d = np.zeros((n, 3), dtype=np.float32)
        # Constant non-zero displacement on the valid side; zeros on the
        # invalid side. If the verb honored the invalid side as zero
        # neighbors, the boundary POIs' median would shift toward zero
        # and they would be flagged. With proper exclusion they match
        # the local constant.
        d_lat = d.reshape(nz, ny, nx, 3)
        d_lat[:, :, :3, :] = np.float32(2.0)
        v_lat = np.ones((nz, ny, nx), dtype=np.bool_)
        v_lat[:, :, 3:] = False
        d = d_lat.reshape(-1, 3)
        v = v_lat.reshape(-1)

        flag = detect_outliers(grid, d, v)

        # No valid POI should be flagged: the valid side is constant.
        assert not bool(flag.any())


class TestMinNeighborsGuard:
    def test_three_valid_neighbors_never_flagged(self):
        grid = _smooth_grid(3, 3, 3)
        nz, ny, nx = grid.grid_shape
        n = nz * ny * nx
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.zeros(n, dtype=np.bool_)

        # Configure: target POI at (1, 1, 1) with absurd displacement,
        # exactly 3 valid neighbors around it. Below min_neighbors=4 so
        # the test must not flag it.
        target = np.ravel_multi_index((1, 1, 1), grid.grid_shape)
        v[target] = True
        d[target] = np.float32([100.0, 0.0, 0.0])
        for nb in [(0, 1, 1), (2, 1, 1), (1, 0, 1)]:
            v[np.ravel_multi_index(nb, grid.grid_shape)] = True

        flag = detect_outliers(grid, d, v, min_neighbors=4)

        assert not bool(flag[target])

    def test_four_valid_neighbors_can_flag(self):
        # Same setup but with one additional valid neighbor — now the
        # test has just enough sample to fire.
        grid = _smooth_grid(3, 3, 3)
        nz, ny, nx = grid.grid_shape
        n = nz * ny * nx
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.zeros(n, dtype=np.bool_)

        target = np.ravel_multi_index((1, 1, 1), grid.grid_shape)
        v[target] = True
        d[target] = np.float32([100.0, 0.0, 0.0])
        for nb in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1)]:
            v[np.ravel_multi_index(nb, grid.grid_shape)] = True

        flag = detect_outliers(grid, d, v, min_neighbors=4)

        assert bool(flag[target])


class TestValidPreFilter:
    def test_invalid_poi_never_returns_true(self):
        grid = _smooth_grid(4, 4, 4)
        n = int(np.prod(grid.grid_shape))
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.ones(n, dtype=np.bool_)

        # Pick an interior POI, mark it invalid, and give it an absurd
        # displacement. The verb must return False for it.
        target = np.ravel_multi_index((2, 2, 2), grid.grid_shape)
        v[target] = False
        d[target] = np.float32(1e6)

        flag = detect_outliers(grid, d, v)

        assert not bool(flag[target])


class TestContract:
    def test_output_shape_dtype(self):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.ones(n, dtype=np.bool_)

        flag = detect_outliers(grid, d, v)

        assert flag.shape == (n,)
        assert flag.dtype == np.bool_

    def test_determinism(self):
        rng = np.random.default_rng(42)
        grid = _smooth_grid(4, 4, 4)
        n = int(np.prod(grid.grid_shape))
        d = rng.standard_normal((n, 3)).astype(np.float32) * 0.1
        d[5] = np.float32(8.0)  # arbitrary spike
        v = np.ones(n, dtype=np.bool_)

        a = detect_outliers(grid, d, v)
        b = detect_outliers(grid, d, v)

        np.testing.assert_array_equal(a, b)

    def test_runtime_shape_check_via_jaxtyping(self):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        d = np.zeros((n, 3), dtype=np.float32)
        v = np.ones(n, dtype=np.bool_)

        flag = detect_outliers_checked(grid, d, v)

        assert flag.shape == (n,)


class TestInputValidation:
    def test_displacement_shape_mismatch_raises(self):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        with pytest.raises(ValueError, match="displacements"):
            detect_outliers(
                grid,
                np.zeros((n, 2), dtype=np.float32),  # type: ignore[arg-type]
                np.ones(n, dtype=np.bool_),
            )

    def test_displacement_dtype_raises(self):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        with pytest.raises(ValueError, match="float32"):
            detect_outliers(
                grid,
                np.zeros((n, 3), dtype=np.float64),  # type: ignore[arg-type]
                np.ones(n, dtype=np.bool_),
            )

    def test_valid_dtype_raises(self):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        with pytest.raises(ValueError, match="bool"):
            detect_outliers(
                grid,
                np.zeros((n, 3), dtype=np.float32),
                np.ones(n, dtype=np.uint8),  # type: ignore[arg-type]
            )

    def test_grid_points_mismatch_raises(self):
        grid = _smooth_grid(3, 3, 3)
        with pytest.raises(ValueError, match="grid_shape"):
            detect_outliers(
                grid,
                np.zeros((2, 3), dtype=np.float32),
                np.ones(2, dtype=np.bool_),
            )

    @pytest.mark.parametrize(
        ("epsilon", "noise_floor", "min_neighbors"),
        [
            (0.0, 0.1, 4),
            (-1.0, 0.1, 4),
            (2.0, 0.0, 4),
            (2.0, -0.1, 4),
            (2.0, 0.1, 0),
            (2.0, 0.1, -2),
        ],
    )
    def test_non_positive_params_raise(
        self, epsilon: float, noise_floor: float, min_neighbors: int
    ):
        grid = _smooth_grid(3, 3, 3)
        n = int(np.prod(grid.grid_shape))
        with pytest.raises(ValueError):
            detect_outliers(
                grid,
                np.zeros((n, 3), dtype=np.float32),
                np.ones(n, dtype=np.bool_),
                epsilon=epsilon,
                noise_floor=noise_floor,
                min_neighbors=min_neighbors,
            )
