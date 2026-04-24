"""Tests for ``mamba_dvc.core.grid``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core import grid as grid_mod
from mamba_dvc.core.grid import build_grid, filter_by_mask
from mamba_dvc.types import GridSpec

build_grid_checked = jaxtyped(typechecker=beartype)(build_grid)
filter_by_mask_checked = jaxtyped(typechecker=beartype)(filter_by_mask)


def _naive_fractions(
    mask: np.ndarray,
    starts: np.ndarray,
    window: tuple[int, int, int],
) -> np.ndarray:
    wz, wy, wx = window
    total = wz * wy * wx
    out = np.empty(starts.shape[0], dtype=np.float64)
    for i, (z0, y0, x0) in enumerate(starts):
        sub = mask[z0 : z0 + wz, y0 : y0 + wy, x0 : x0 + wx]
        out[i] = sub.sum() / total
    return out


class TestBuildGrid:
    def test_production_defaults_give_documented_grid(self):
        grid = build_grid((960, 1280, 1280), window=96, overlap=0.5)

        assert grid.grid_shape == (19, 25, 25)
        assert grid.positions.shape == (19 * 25 * 25, 3)
        assert grid.starts.shape == (19 * 25 * 25, 3)
        assert grid.spacing == (48, 48, 48)
        assert grid.window == (96, 96, 96)
        assert grid.volume_shape == (960, 1280, 1280)

    def test_positions_are_float32_and_starts_int64(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        assert grid.positions.dtype == np.float32
        assert grid.starts.dtype == np.int64

    def test_small_volume_spot_check(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)

        assert grid.grid_shape == (3, 3, 3)
        assert grid.spacing == (16, 16, 16)

        expected_starts_axis = np.array([0, 16, 32], dtype=np.int64)
        zz, yy, xx = np.meshgrid(
            expected_starts_axis,
            expected_starts_axis,
            expected_starts_axis,
            indexing="ij",
        )
        expected_starts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        np.testing.assert_array_equal(grid.starts, expected_starts)

        expected_centers = expected_starts.astype(np.float32) + np.float32(15.5)
        np.testing.assert_allclose(grid.positions, expected_centers)

    def test_subvolumes_stay_inside_volume(self):
        grid = build_grid((100, 80, 70), window=(32, 24, 16), overlap=0.25)
        z1 = grid.starts[:, 0] + grid.window[0]
        y1 = grid.starts[:, 1] + grid.window[1]
        x1 = grid.starts[:, 2] + grid.window[2]

        assert int(grid.starts.min()) >= 0
        assert int(z1.max()) <= 100
        assert int(y1.max()) <= 80
        assert int(x1.max()) <= 70

    def test_scalar_and_tuple_window_agree(self):
        a = build_grid((64, 64, 64), window=32, overlap=0.5)
        b = build_grid((64, 64, 64), window=(32, 32, 32), overlap=0.5)

        assert a.grid_shape == b.grid_shape
        assert a.spacing == b.spacing
        np.testing.assert_array_equal(a.starts, b.starts)
        np.testing.assert_array_equal(a.positions, b.positions)

    def test_overlap_zero_gives_non_overlapping_stride(self):
        grid = build_grid((96, 96, 96), window=32, overlap=0.0)
        assert grid.spacing == (32, 32, 32)
        assert grid.grid_shape == (3, 3, 3)

    def test_window_equals_volume_gives_single_poi(self):
        grid = build_grid((32, 48, 64), window=(32, 48, 64), overlap=0.5)
        assert grid.grid_shape == (1, 1, 1)
        np.testing.assert_array_equal(grid.starts, np.array([[0, 0, 0]]))

    def test_returns_gridspec_instance(self):
        grid = build_grid_checked((64, 64, 64), window=32, overlap=0.5)
        assert isinstance(grid, GridSpec)

    @pytest.mark.parametrize(
        ("volume_shape", "window", "overlap"),
        [
            ((64, 64, 64), 0, 0.5),
            ((64, 64, 64), -8, 0.5),
            ((64, 64, 64), 32, -0.1),
            ((64, 64, 64), 32, 1.0),
            ((64, 64, 64), 32, 1.5),
            ((64, 64, 64), (32, 32), 0.5),
            ((64, 64, 64), (32, 0, 32), 0.5),
            ((64, 64, 64), 128, 0.5),
            ((64, 64), 32, 0.5),
            ((64, 64, 0), 32, 0.5),
        ],
    )
    def test_invalid_inputs_raise(
        self,
        volume_shape: tuple[int, ...],
        window: int | tuple[int, ...],
        overlap: float,
    ):
        with pytest.raises(ValueError):
            build_grid(volume_shape, window=window, overlap=overlap)  # type: ignore[arg-type]


class TestFilterByMask:
    def test_all_true_mask_admits_every_poi(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.bool_)

        valid = filter_by_mask(grid, mask, threshold=0.9)

        assert valid.dtype == np.bool_
        assert valid.shape == (grid.positions.shape[0],)
        assert bool(valid.all())

    def test_all_false_mask_admits_nothing(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.zeros((64, 64, 64), dtype=np.bool_)

        # Any positive threshold rejects zero fraction; threshold=0.0 would
        # admit everything by the `fraction >= threshold` rule.
        valid = filter_by_mask(grid, mask, threshold=0.01)

        assert not bool(valid.any())

    def test_threshold_zero_admits_all_when_mask_all_true(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.bool_)

        valid = filter_by_mask(grid, mask, threshold=0.0)

        assert bool(valid.all())

    def test_threshold_one_requires_fully_valid_subvolume(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.bool_)
        mask[0, 0, 0] = False

        valid = filter_by_mask(grid, mask, threshold=1.0)

        # Only the POI at (0, 0, 0) sees the hole.
        expected = np.ones(grid.positions.shape[0], dtype=np.bool_)
        expected[0] = False
        np.testing.assert_array_equal(valid, expected)

    def test_sat_matches_naive_reference_on_random_mask(self):
        rng = np.random.default_rng(0)
        volume_shape = (48, 56, 64)
        mask = rng.random(volume_shape) > 0.3

        grid = build_grid(volume_shape, window=(16, 16, 16), overlap=0.5)
        fractions_naive = _naive_fractions(mask, grid.starts, grid.window)

        for threshold in (0.0, 0.25, 0.5, 0.7, 0.9, 1.0):
            expected = fractions_naive >= threshold
            got = filter_by_mask(grid, mask, threshold=threshold)
            np.testing.assert_array_equal(got, expected, err_msg=f"{threshold=}")

    def test_half_plane_mask_has_expected_boundary(self):
        volume_shape = (32, 32, 64)
        mask = np.zeros(volume_shape, dtype=np.bool_)
        mask[:, :, 32:] = True  # right half of X is valid

        grid = build_grid(volume_shape, window=16, overlap=0.0)
        # spacing=16 → starts along X: 0, 16, 32, 48 → 4 POIs on X axis.
        # Windows starting at x0=0,16 are fully invalid; x0=32,48 fully valid.
        fractions = _naive_fractions(mask, grid.starts, grid.window)
        x0 = grid.starts[:, 2]
        np.testing.assert_allclose(fractions[x0 < 32], 0.0)
        np.testing.assert_allclose(fractions[x0 >= 32], 1.0)

        valid = filter_by_mask(grid, mask, threshold=0.5)
        np.testing.assert_array_equal(valid, x0 >= 32)

    def test_runtime_shape_check_via_jaxtyping(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.bool_)

        # Correct call goes through the checked wrapper without error.
        valid = filter_by_mask_checked(grid, mask, threshold=0.5)
        assert bool(valid.all())

    def test_shape_mismatch_raises(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 32), dtype=np.bool_)
        with pytest.raises(ValueError, match="mask shape"):
            filter_by_mask(grid, mask)

    def test_non_bool_dtype_raises(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool dtype"):
            filter_by_mask(grid, mask)  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [-0.01, 1.01, 2.0, -1.0])
    def test_threshold_out_of_range_raises(self, threshold: float):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        mask = np.ones((64, 64, 64), dtype=np.bool_)
        with pytest.raises(ValueError, match="threshold"):
            filter_by_mask(grid, mask, threshold=threshold)


class TestInternals:
    def test_integral_image_matches_numpy_cumsum(self):
        rng = np.random.default_rng(1)
        mask = rng.random((12, 10, 14)) > 0.4
        sat = grid_mod._integral_image_3d(mask)

        assert sat.shape == (13, 11, 15)
        assert sat.dtype == np.uint32
        assert int(sat[0].sum()) == 0
        assert int(sat[:, 0].sum()) == 0
        assert int(sat[:, :, 0].sum()) == 0

        # Full-volume count at the far corner.
        assert int(sat[-1, -1, -1]) == int(mask.sum())

    def test_subvolume_sums_agree_with_slicing(self):
        rng = np.random.default_rng(2)
        mask = rng.random((20, 18, 22)) > 0.5
        sat = grid_mod._integral_image_3d(mask)

        starts = np.array([[0, 0, 0], [3, 4, 5], [10, 9, 11], [15, 13, 17]], dtype=np.int64)
        window = (5, 5, 5)
        z0 = starts[:, 0]
        y0 = starts[:, 1]
        x0 = starts[:, 2]
        z1 = z0 + window[0]
        y1 = y0 + window[1]
        x1 = x0 + window[2]

        got = grid_mod._subvolume_sums(sat, z0, y0, x0, z1, y1, x1)
        for i, (a, b, c) in enumerate(starts):
            expected = int(mask[a : a + window[0], b : b + window[1], c : c + window[2]].sum())
            assert int(got[i]) == expected
