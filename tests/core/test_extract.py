"""Tests for ``mamba_dvc.core.extract``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.extract import extract_subvolumes
from mamba_dvc.core.grid import build_grid

extract_subvolumes_checked = jaxtyped(typechecker=beartype)(extract_subvolumes)


def _naive_extract(
    volume: np.ndarray,
    starts: np.ndarray,
    window: tuple[int, int, int],
) -> np.ndarray:
    wz, wy, wx = window
    out = np.empty((starts.shape[0], wz, wy, wx), dtype=volume.dtype)
    for i, (z0, y0, x0) in enumerate(starts):
        out[i] = volume[z0 : z0 + wz, y0 : y0 + wy, x0 : x0 + wx]
    return out


class TestExtractSubvolumes:
    def test_matches_naive_on_random_volume(self):
        rng = np.random.default_rng(0)
        volume = rng.standard_normal((24, 28, 32), dtype=np.float32)
        starts = np.array(
            [[0, 0, 0], [3, 4, 5], [10, 12, 14], [16, 20, 24]],
            dtype=np.int64,
        )
        window = (8, 8, 8)

        got = extract_subvolumes(volume, starts, window)
        expected = _naive_extract(volume, starts, window)

        np.testing.assert_array_equal(got, expected)

    def test_output_shape_and_dtype(self):
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        starts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)

        out = extract_subvolumes(volume, starts, (4, 5, 6))

        assert out.shape == (2, 4, 5, 6)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_single_poi_at_origin(self):
        rng = np.random.default_rng(1)
        volume = rng.standard_normal((10, 10, 10), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)

        out = extract_subvolumes(volume, starts, (4, 4, 4))

        np.testing.assert_array_equal(out[0], volume[:4, :4, :4])

    def test_anisotropic_window(self):
        rng = np.random.default_rng(2)
        volume = rng.standard_normal((20, 30, 40), dtype=np.float32)
        starts = np.array([[2, 5, 10], [8, 15, 20]], dtype=np.int64)

        out = extract_subvolumes(volume, starts, (6, 10, 14))

        assert out.shape == (2, 6, 10, 14)
        np.testing.assert_array_equal(out[0], volume[2:8, 5:15, 10:24])
        np.testing.assert_array_equal(out[1], volume[8:14, 15:25, 20:34])

    def test_empty_batch_returns_empty(self):
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        starts = np.empty((0, 3), dtype=np.int64)

        out = extract_subvolumes(volume, starts, (4, 4, 4))

        assert out.shape == (0, 4, 4, 4)
        assert out.dtype == np.float32

    def test_writing_to_output_does_not_mutate_source(self):
        volume = np.ones((8, 8, 8), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)

        out = extract_subvolumes(volume, starts, (4, 4, 4))
        out[...] = 42.0

        assert float(volume.max()) == 1.0
        assert float(volume.min()) == 1.0

    def test_integrates_with_build_grid(self):
        rng = np.random.default_rng(3)
        volume_shape = (32, 40, 48)
        volume = rng.standard_normal(volume_shape, dtype=np.float32)

        grid = build_grid(volume_shape, window=16, overlap=0.5)
        out = extract_subvolumes(volume, grid.starts, grid.window)

        assert out.shape == (grid.positions.shape[0], 16, 16, 16)
        np.testing.assert_array_equal(out[0], volume[:16, :16, :16])

        last = grid.starts[-1]
        np.testing.assert_array_equal(
            out[-1],
            volume[last[0] : last[0] + 16, last[1] : last[1] + 16, last[2] : last[2] + 16],
        )

    def test_runtime_shape_check_via_jaxtyping(self):
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)

        out = extract_subvolumes_checked(volume, starts, (4, 4, 4))
        assert out.shape == (1, 4, 4, 4)

    @pytest.mark.parametrize(
        ("volume_shape", "starts", "window"),
        [
            # out-of-range start along z
            ((16, 16, 16), np.array([[13, 0, 0]], dtype=np.int64), (4, 4, 4)),
            # out-of-range start along y
            ((16, 16, 16), np.array([[0, 15, 0]], dtype=np.int64), (4, 4, 4)),
            # out-of-range start along x
            ((16, 16, 16), np.array([[0, 0, 14]], dtype=np.int64), (4, 4, 4)),
            # negative start
            ((16, 16, 16), np.array([[-1, 0, 0]], dtype=np.int64), (4, 4, 4)),
        ],
    )
    def test_out_of_bounds_raises(
        self,
        volume_shape: tuple[int, int, int],
        starts: np.ndarray,
        window: tuple[int, int, int],
    ):
        volume = np.zeros(volume_shape, dtype=np.float32)
        with pytest.raises(ValueError):
            extract_subvolumes(volume, starts, window)

    def test_window_larger_than_volume_raises(self):
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="exceeds volume shape"):
            extract_subvolumes(volume, starts, (16, 4, 4))

    def test_non_float32_volume_raises(self):
        volume = np.zeros((8, 8, 8), dtype=np.float64)
        starts = np.array([[0, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="float32"):
            extract_subvolumes(volume, starts, (4, 4, 4))  # type: ignore[arg-type]

    def test_non_3d_volume_raises(self):
        volume = np.zeros((8, 8), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="3D"):
            extract_subvolumes(volume, starts, (4, 4, 4))  # type: ignore[arg-type]

    def test_non_int64_starts_raises(self):
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int32)
        with pytest.raises(ValueError, match="int64"):
            extract_subvolumes(volume, starts, (4, 4, 4))  # type: ignore[arg-type]

    def test_wrong_starts_shape_raises(self):
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        starts = np.array([[0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match=r"shape \(batch, 3\)"):
            extract_subvolumes(volume, starts, (4, 4, 4))  # type: ignore[arg-type]

    @pytest.mark.parametrize("window", [(0, 4, 4), (-1, 4, 4), (4, 0, 4)])
    def test_invalid_window_raises(self, window: tuple[int, int, int]):
        volume = np.zeros((8, 8, 8), dtype=np.float32)
        starts = np.array([[0, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError):
            extract_subvolumes(volume, starts, window)


@pytest.mark.gpu
class TestExtractSubvolumesGPU:
    def test_matches_cpu_on_cupy_inputs(self):
        cp = pytest.importorskip("cupy")

        rng = np.random.default_rng(4)
        volume_host = rng.standard_normal((24, 28, 32), dtype=np.float32)
        starts_host = np.array(
            [[0, 0, 0], [3, 4, 5], [10, 12, 14], [16, 20, 24]],
            dtype=np.int64,
        )
        window = (8, 8, 8)

        volume_dev = cp.asarray(volume_host)
        starts_dev = cp.asarray(starts_host)

        out_dev = extract_subvolumes(volume_dev, starts_dev, window)
        out_host = cp.asnumpy(out_dev)
        expected = _naive_extract(volume_host, starts_host, window)

        assert out_dev.dtype == cp.float32
        np.testing.assert_array_equal(out_host, expected)

    def test_rejects_mixed_host_device_inputs(self):
        cp = pytest.importorskip("cupy")

        volume_dev = cp.zeros((8, 8, 8), dtype=cp.float32)
        starts_host = np.array([[0, 0, 0]], dtype=np.int64)
        with pytest.raises(ValueError, match="same array module"):
            extract_subvolumes(volume_dev, starts_host, (4, 4, 4))  # type: ignore[arg-type]
