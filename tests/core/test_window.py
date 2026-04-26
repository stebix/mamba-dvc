"""Tests for ``mamba_dvc.core.window``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.window import (
    preprocess_subvolumes,
    tukey_window_1d,
    tukey_window_3d,
)
from scipy.signal.windows import tukey as scipy_tukey

preprocess_subvolumes_checked = jaxtyped(typechecker=beartype)(preprocess_subvolumes)


class TestTukeyWindow1D:
    def test_all_ones_for_alpha_zero(self):
        w = tukey_window_1d(16, 0.0)
        np.testing.assert_array_equal(w, np.ones(16, dtype=np.float32))

    def test_hann_for_alpha_one(self):
        n = 32
        w = tukey_window_1d(n, 1.0)
        expected = np.hanning(n).astype(np.float32)
        np.testing.assert_allclose(w, expected, atol=1e-6)

    def test_endpoints_are_zero_for_positive_alpha(self):
        for alpha in (0.1, 0.25, 0.5, 0.9, 1.0):
            w = tukey_window_1d(64, alpha)
            assert float(w[0]) == pytest.approx(0.0, abs=1e-6)
            assert float(w[-1]) == pytest.approx(0.0, abs=1e-6)

    def test_flat_center_has_unit_value(self):
        n, alpha = 100, 0.25
        w = tukey_window_1d(n, alpha)
        # Flat interior is x in [alpha/2, 1 - alpha/2] → indices [n*alpha/2, n*(1 - alpha/2)]
        # Pick a point well inside the flat region.
        center = n // 2
        assert float(w[center]) == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_about_midpoint(self):
        w = tukey_window_1d(65, 0.3)
        np.testing.assert_allclose(w, w[::-1], atol=1e-6)

    def test_matches_scipy_reference(self):
        for n in (16, 31, 64, 96):
            for alpha in (0.1, 0.25, 0.5, 0.75):
                got = tukey_window_1d(n, alpha)
                expected = scipy_tukey(n, alpha=alpha).astype(np.float32)
                np.testing.assert_allclose(got, expected, atol=1e-6, err_msg=f"{n=}, {alpha=}")

    def test_n_one_edge_case(self):
        w = tukey_window_1d(1, 0.25)
        np.testing.assert_array_equal(w, np.ones(1, dtype=np.float32))

    def test_output_dtype(self):
        assert tukey_window_1d(16, 0.25).dtype == np.float32

    @pytest.mark.parametrize("n", [0, -1, -16])
    def test_invalid_n_raises(self, n: int):
        with pytest.raises(ValueError, match="n must be positive"):
            tukey_window_1d(n, 0.25)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0, -1.0])
    def test_invalid_alpha_raises(self, alpha: float):
        with pytest.raises(ValueError, match="alpha must be in"):
            tukey_window_1d(16, alpha)


class TestTukeyWindow3D:
    def test_shape_and_dtype(self):
        w = tukey_window_3d((8, 10, 12), 0.25)
        assert w.shape == (8, 10, 12)
        assert w.dtype == np.float32

    def test_is_separable_outer_product(self):
        shape = (8, 10, 12)
        alpha = 0.3
        w_3d = tukey_window_3d(shape, alpha)
        w_z = tukey_window_1d(shape[0], alpha)
        w_y = tukey_window_1d(shape[1], alpha)
        w_x = tukey_window_1d(shape[2], alpha)
        expected = w_z[:, None, None] * w_y[None, :, None] * w_x[None, None, :]
        np.testing.assert_allclose(w_3d, expected, atol=1e-6)

    def test_symmetric_per_axis(self):
        w = tukey_window_3d((9, 9, 9), 0.25)
        np.testing.assert_allclose(w, w[::-1, :, :], atol=1e-6)
        np.testing.assert_allclose(w, w[:, ::-1, :], atol=1e-6)
        np.testing.assert_allclose(w, w[:, :, ::-1], atol=1e-6)

    def test_all_ones_for_alpha_zero(self):
        w = tukey_window_3d((6, 6, 6), 0.0)
        np.testing.assert_array_equal(w, np.ones((6, 6, 6), dtype=np.float32))

    def test_center_voxel_is_unity(self):
        w = tukey_window_3d((17, 17, 17), 0.25)
        assert float(w[8, 8, 8]) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("shape", [(8, 8), (8,), (8, 8, 8, 8), (0, 8, 8)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]):
        with pytest.raises(ValueError):
            tukey_window_3d(shape, 0.25)  # type: ignore[arg-type]


class TestPreprocessSubvolumes:
    def test_preserves_shape_dtype_and_contiguity(self):
        rng = np.random.default_rng(0)
        subvolumes = rng.standard_normal((4, 8, 8, 8), dtype=np.float32)

        out = preprocess_subvolumes(subvolumes, tukey_alpha=0.25)

        assert out.shape == (4, 8, 8, 8)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_alpha_zero_is_pure_mean_subtract(self):
        # With alpha=0 (rectangular window) and no mask, output should be
        # the input with its per-subvolume mean subtracted.
        rng = np.random.default_rng(1)
        subvolumes = rng.standard_normal((3, 6, 6, 6), dtype=np.float32)

        out = preprocess_subvolumes(subvolumes, tukey_alpha=0.0)
        expected = subvolumes - subvolumes.mean(axis=(1, 2, 3), keepdims=True)

        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_unmasked_output_has_zero_mean_at_alpha_zero(self):
        rng = np.random.default_rng(2)
        subvolumes = rng.standard_normal((5, 8, 8, 8), dtype=np.float32) + 7.0

        out = preprocess_subvolumes(subvolumes, tukey_alpha=0.0)

        means = out.mean(axis=(1, 2, 3))
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_output_vanishes_at_boundary_with_positive_alpha(self):
        # Tukey window forces the outer shell of each subvolume to zero
        # whenever alpha > 0 and the mean-subtracted signal is finite.
        rng = np.random.default_rng(3)
        subvolumes = rng.standard_normal((2, 10, 10, 10), dtype=np.float32)

        out = preprocess_subvolumes(subvolumes, tukey_alpha=0.5)

        # Every face voxel is zero.
        np.testing.assert_allclose(out[:, 0, :, :], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, -1, :, :], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, :, 0, :], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, :, -1, :], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, :, :, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, :, :, -1], 0.0, atol=1e-5)

    def test_masked_voxels_are_exactly_zero(self):
        rng = np.random.default_rng(4)
        subvolumes = rng.standard_normal((2, 8, 8, 8), dtype=np.float32)
        mask = rng.random((2, 8, 8, 8)) > 0.3

        out = preprocess_subvolumes(subvolumes, mask, tukey_alpha=0.25)

        invalid = ~mask
        np.testing.assert_array_equal(out[invalid], 0.0)

    def test_masked_mean_excludes_invalid_voxels(self):
        # Construct a subvolume where valid voxels sum to a known mean
        # and invalid voxels carry a large bias. The masked mean should
        # ignore the bias.
        subvolumes = np.zeros((1, 4, 4, 4), dtype=np.float32)
        # Valid region: the first half along z has value 2.0.
        subvolumes[0, :2, :, :] = 2.0
        # Invalid region: the second half has value 100.0 (would
        # dominate an unmasked mean).
        subvolumes[0, 2:, :, :] = 100.0

        mask = np.zeros((1, 4, 4, 4), dtype=np.bool_)
        mask[0, :2, :, :] = True

        # With alpha=0 and the mask, step 1 subtracts the masked mean
        # (2.0 across the whole subvolume), step 2 is a no-op, step 3
        # zeroes the invalid half. So the valid half should become 0.0.
        out = preprocess_subvolumes(subvolumes, mask, tukey_alpha=0.0)

        np.testing.assert_allclose(out[0, :2, :, :], 0.0, atol=1e-5)
        np.testing.assert_array_equal(out[0, 2:, :, :], 0.0)

    def test_all_false_mask_yields_all_zero_output(self):
        subvolumes = np.ones((2, 6, 6, 6), dtype=np.float32)
        mask = np.zeros((2, 6, 6, 6), dtype=np.bool_)

        out = preprocess_subvolumes(subvolumes, mask, tukey_alpha=0.25)

        np.testing.assert_array_equal(out, np.zeros_like(out))
        assert not np.any(np.isnan(out))

    def test_does_not_mutate_inputs(self):
        subvolumes = np.ones((2, 6, 6, 6), dtype=np.float32) * 3.0
        mask = np.ones((2, 6, 6, 6), dtype=np.bool_)
        subvolumes_copy = subvolumes.copy()
        mask_copy = mask.copy()

        _ = preprocess_subvolumes(subvolumes, mask, tukey_alpha=0.25)

        np.testing.assert_array_equal(subvolumes, subvolumes_copy)
        np.testing.assert_array_equal(mask, mask_copy)

    def test_runtime_shape_check_via_jaxtyping(self):
        subvolumes = np.zeros((2, 6, 6, 6), dtype=np.float32)
        out = preprocess_subvolumes_checked(subvolumes, None, 0.25)
        assert out.shape == (2, 6, 6, 6)

    def test_pipeline_round_trip_with_extract(self):
        # End-to-end: build a grid, extract subvolumes, preprocess. The
        # output must be shape-compatible and finite.
        from mamba_dvc.core.extract import extract_subvolumes
        from mamba_dvc.core.grid import build_grid

        rng = np.random.default_rng(5)
        volume = rng.standard_normal((32, 32, 32), dtype=np.float32)
        grid = build_grid(volume.shape, window=16, overlap=0.5)

        subvolumes = extract_subvolumes(volume, grid.starts, grid.window)
        out = preprocess_subvolumes(subvolumes, None, tukey_alpha=0.25)

        assert out.shape == subvolumes.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0, -1.0])
    def test_alpha_out_of_range_raises(self, alpha: float):
        subvolumes = np.zeros((1, 4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="tukey_alpha"):
            preprocess_subvolumes(subvolumes, None, alpha)

    def test_non_float32_subvolumes_raise(self):
        subvolumes = np.zeros((1, 4, 4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            preprocess_subvolumes(subvolumes, None, 0.25)  # type: ignore[arg-type]

    def test_wrong_ndim_raises(self):
        subvolumes = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="4D"):
            preprocess_subvolumes(subvolumes, None, 0.25)  # type: ignore[arg-type]

    def test_mask_shape_mismatch_raises(self):
        subvolumes = np.zeros((2, 4, 4, 4), dtype=np.float32)
        mask = np.ones((2, 4, 4, 5), dtype=np.bool_)
        with pytest.raises(ValueError, match="mask_subvolumes shape"):
            preprocess_subvolumes(subvolumes, mask, 0.25)

    def test_non_bool_mask_raises(self):
        subvolumes = np.zeros((2, 4, 4, 4), dtype=np.float32)
        mask = np.ones((2, 4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool dtype"):
            preprocess_subvolumes(subvolumes, mask, 0.25)  # type: ignore[arg-type]


@pytest.mark.gpu
class TestPreprocessSubvolumesGPU:
    def test_matches_cpu_on_cupy_inputs(self):
        cp = pytest.importorskip("cupy")

        rng = np.random.default_rng(6)
        subvolumes_host = rng.standard_normal((3, 8, 8, 8), dtype=np.float32)
        mask_host = rng.random((3, 8, 8, 8)) > 0.3

        subvolumes_dev = cp.asarray(subvolumes_host)
        mask_dev = cp.asarray(mask_host)

        out_dev = preprocess_subvolumes(subvolumes_dev, mask_dev, tukey_alpha=0.25)
        expected = preprocess_subvolumes(subvolumes_host, mask_host, tukey_alpha=0.25)

        assert out_dev.dtype == cp.float32
        np.testing.assert_allclose(cp.asnumpy(out_dev), expected, atol=1e-5)

    def test_rejects_mixed_host_device_inputs(self):
        cp = pytest.importorskip("cupy")

        subvolumes_dev = cp.zeros((2, 4, 4, 4), dtype=cp.float32)
        mask_host = np.ones((2, 4, 4, 4), dtype=np.bool_)
        with pytest.raises(ValueError, match="same array module"):
            preprocess_subvolumes(subvolumes_dev, mask_host, 0.25)  # type: ignore[arg-type]
