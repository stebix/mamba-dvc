"""Tests for ``mamba_dvc.core.peakfit``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.ncc import correlate, peak_displacement
from mamba_dvc.core.peakfit import gaussian_subvoxel_fit
from mamba_dvc.core.window import preprocess_subvolumes
from mamba_dvc.validate.synthetic import make_pair, rigid_shift

gaussian_subvoxel_fit_checked = jaxtyped(typechecker=beartype)(gaussian_subvoxel_fit)


def _gaussian_peak(
    shape: tuple[int, int, int],
    peak_indices: tuple[int, int, int],
    fractional_offset: tuple[float, float, float],
    sigma: float = 1.5,
) -> np.ndarray:
    """Synthesize a single subvolume with a Gaussian peak at a known offset.

    The peak's true location is ``peak_indices + fractional_offset``;
    the integer peak detector sees the nearest grid cell as the max.
    Used to drive the subvoxel fit with a known ground truth.
    """
    wz, wy, wx = shape
    iz, iy, ix = peak_indices
    fz, fy, fx = fractional_offset

    z = np.arange(wz, dtype=np.float32)
    y = np.arange(wy, dtype=np.float32)
    x = np.arange(wx, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    rsq = (zz - (iz + fz)) ** 2 + (yy - (iy + fy)) ** 2 + (xx - (ix + fx)) ** 2
    return np.exp(-rsq / (2.0 * sigma * sigma)).astype(np.float32)


class TestGaussianSubvoxelFit:
    def test_zero_offset_for_centered_peak(self):
        # Peak exactly on a grid cell → fractional offset must be ~0.
        corr = _gaussian_peak((8, 8, 8), (3, 4, 2), (0.0, 0.0, 0.0))[None, ...]
        integer = np.asarray([[3, 4, 2]], dtype=np.int64)

        offsets = gaussian_subvoxel_fit(corr, integer)

        np.testing.assert_allclose(offsets[0], 0.0, atol=1e-3)

    @pytest.mark.parametrize(
        "fractional",
        [
            (0.3, 0.0, 0.0),
            (-0.3, 0.0, 0.0),
            (0.0, 0.45, 0.0),
            (0.0, 0.0, -0.4),
            (0.25, -0.15, 0.35),
        ],
    )
    def test_recovers_known_subvoxel_offset(self, fractional: tuple[float, float, float]):
        # Build a Gaussian peak with a known fractional offset; the
        # integer-peak detector picks the nearest grid cell, then the
        # subvoxel fit must recover the offset within float32 noise.
        wz, wy, wx = 12, 12, 12
        iz, iy, ix = 5, 6, 4
        corr = _gaussian_peak((wz, wy, wx), (iz, iy, ix), fractional)[None, ...]
        # The integer peak detected by argmax is at (iz, iy, ix); pass
        # that explicitly so the test does not depend on
        # peak_displacement's wrap convention.
        integer = np.asarray([[iz, iy, ix]], dtype=np.int64)

        offsets = gaussian_subvoxel_fit(corr, integer)

        np.testing.assert_allclose(
            offsets[0], np.asarray(fractional, dtype=np.float32), atol=5e-3
        )

    def test_offsets_clamped_to_half_voxel(self):
        # A pathological correlation map with a strong false neighbor
        # would push the fit past 0.5; the clamp keeps the integer
        # peak authoritative.
        rng = np.random.default_rng(0)
        corr = rng.random((1, 8, 8, 8), dtype=np.float32) * 0.1
        # Make c_minus huge along axis 0 so the unclamped fit would
        # overshoot.
        corr[0, 2, 4, 4] = 0.5
        corr[0, 3, 4, 4] = 1.0  # integer peak
        corr[0, 4, 4, 4] = 0.05

        integer = np.asarray([[3, 4, 4]], dtype=np.int64)
        offsets = gaussian_subvoxel_fit(corr, integer)

        # Whatever the fit produces along axis 0, it must lie in
        # [-0.5, 0.5]; the other two axes have noise → also bounded.
        assert -0.5 <= float(offsets[0, 0]) <= 0.5
        assert -0.5 <= float(offsets[0, 1]) <= 0.5
        assert -0.5 <= float(offsets[0, 2]) <= 0.5

    def test_handles_negative_correlation_values(self):
        # If a neighbor is negative (legit for NCC), the log path is
        # invalid; the fallback parabolic fit must not produce NaN.
        corr = np.zeros((1, 6, 6, 6), dtype=np.float32)
        corr[0, 2, 3, 3] = -0.1
        corr[0, 3, 3, 3] = 0.8
        corr[0, 4, 3, 3] = 0.2
        # Same lobe, repeated along the other axes for safety.
        corr[0, 3, 2, 3] = -0.1
        corr[0, 3, 4, 3] = 0.2
        corr[0, 3, 3, 2] = -0.1
        corr[0, 3, 3, 4] = 0.2

        integer = np.asarray([[3, 3, 3]], dtype=np.int64)
        offsets = gaussian_subvoxel_fit(corr, integer)

        assert np.all(np.isfinite(offsets))

    def test_cyclic_index_wrap(self):
        # Peak at the very first index (0); the i-1 neighbor must come
        # from index w-1 via cyclic wrap. The non-periodic Gaussian
        # used by the helper means the wraparound neighbor is *not* a
        # well-shaped peak side-lobe — so we only assert that the fit
        # does not crash and stays inside the clamp interval. POIs
        # whose peak hugs the cyclic edge are flagged downstream by
        # the outlier filter, not corrected here.
        wz = 8
        corr = _gaussian_peak((wz, 6, 6), (0, 3, 3), (0.2, 0.0, 0.0))[None, ...]
        integer = np.asarray([[0, 3, 3]], dtype=np.int64)

        offsets = gaussian_subvoxel_fit(corr, integer)

        assert np.all(np.isfinite(offsets))
        assert -0.5 <= float(offsets[0, 0]) <= 0.5

    def test_signed_integer_displacement_works(self):
        # peak_displacement returns signed integers; the fit must
        # interpret them modulo the axis length so the same correlation
        # volume produces the same result whether the integer peak is
        # passed as `w-1` (unsigned) or `-1` (signed).
        rng = np.random.default_rng(99)
        corr = rng.random((1, 8, 8, 8), dtype=np.float32)

        offsets_signed = gaussian_subvoxel_fit(corr, np.asarray([[-1, 4, 4]], dtype=np.int64))
        offsets_unsigned = gaussian_subvoxel_fit(corr, np.asarray([[7, 4, 4]], dtype=np.int64))

        np.testing.assert_array_equal(offsets_signed, offsets_unsigned)

    def test_runtime_shape_check_via_jaxtyping(self):
        corr = np.zeros((2, 6, 6, 6), dtype=np.float32)
        corr[0, 0, 0, 0] = 1.0
        corr[1, 0, 0, 0] = 1.0
        integer = np.zeros((2, 3), dtype=np.int64)
        offsets = gaussian_subvoxel_fit_checked(corr, integer)
        assert offsets.shape == (2, 3)

    def test_does_not_mutate_inputs(self):
        rng = np.random.default_rng(1)
        corr = rng.random((2, 6, 6, 6), dtype=np.float32)
        integer = np.zeros((2, 3), dtype=np.int64)
        corr_copy = corr.copy()
        integer_copy = integer.copy()

        _ = gaussian_subvoxel_fit(corr, integer)

        np.testing.assert_array_equal(corr, corr_copy)
        np.testing.assert_array_equal(integer, integer_copy)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="4D"):
            gaussian_subvoxel_fit(
                np.zeros((6, 6, 6), dtype=np.float32),  # type: ignore[arg-type]
                np.zeros((1, 3), dtype=np.int64),
            )

    def test_wrong_dtype_correlation_raises(self):
        with pytest.raises(ValueError, match="float32"):
            gaussian_subvoxel_fit(
                np.zeros((1, 6, 6, 6), dtype=np.float64),  # type: ignore[arg-type]
                np.zeros((1, 3), dtype=np.int64),
            )

    def test_wrong_dtype_integer_raises(self):
        with pytest.raises(ValueError, match="int64"):
            gaussian_subvoxel_fit(
                np.zeros((1, 6, 6, 6), dtype=np.float32),
                np.zeros((1, 3), dtype=np.int32),  # type: ignore[arg-type]
            )

    def test_batch_mismatch_raises(self):
        with pytest.raises(ValueError, match="batch size mismatch"):
            gaussian_subvoxel_fit(
                np.zeros((2, 6, 6, 6), dtype=np.float32),
                np.zeros((3, 3), dtype=np.int64),
            )


@pytest.mark.slow
class TestPeakfitOnSyntheticPair:
    def test_recovers_fractional_shift_from_real_pipeline(self):
        # End-to-end on a synthetic textured volume with a known
        # fractional rigid shift; this validates the full ncc → peakfit
        # chain at the accuracy target stated in plan §2 (≤ 0.1 voxel).
        shift = (0.3, -0.7, 1.4)
        pair = make_pair(shape=(64, 64, 64), field=rigid_shift(shift), seed=11)

        # Single subvolume centered on the volume (interior, no
        # boundary effects). Use a comfortable window so the
        # band-limited texture survives the Tukey taper.
        w = 48
        cz = (pair.reference.shape[0] - w) // 2
        cy = (pair.reference.shape[1] - w) // 2
        cx = (pair.reference.shape[2] - w) // 2
        ref_sv = pair.reference[cz : cz + w, cy : cy + w, cx : cx + w][None, ...]
        def_sv = pair.deformed[cz : cz + w, cy : cy + w, cx : cx + w][None, ...]

        ref_pp = preprocess_subvolumes(ref_sv, None, tukey_alpha=0.25)
        def_pp = preprocess_subvolumes(def_sv, None, tukey_alpha=0.25)

        corr = correlate(ref_pp, def_pp)
        integer, _ = peak_displacement(corr)
        fractional = gaussian_subvoxel_fit(corr, integer)

        recovered = integer.astype(np.float32) + fractional
        expected = np.asarray(shift, dtype=np.float32)
        error = np.abs(recovered[0] - expected)

        assert np.all(error < 0.1), (
            f"recovered={recovered[0]}, expected={expected}, error={error}"
        )


@pytest.mark.gpu
class TestPeakfitGPU:
    def test_matches_cpu_on_cupy_inputs(self):
        cp = pytest.importorskip("cupy")

        corr_host = _gaussian_peak((10, 10, 10), (4, 5, 3), (0.2, -0.3, 0.1))[None, ...]
        integer_host = np.asarray([[4, 5, 3]], dtype=np.int64)

        corr_dev = cp.asarray(corr_host)
        integer_dev = cp.asarray(integer_host)

        offsets_dev = gaussian_subvoxel_fit(corr_dev, integer_dev)
        offsets_host = gaussian_subvoxel_fit(corr_host, integer_host)

        np.testing.assert_allclose(cp.asnumpy(offsets_dev), offsets_host, atol=1e-5)

    def test_rejects_mixed_host_device_inputs(self):
        cp = pytest.importorskip("cupy")
        corr_dev = cp.zeros((1, 6, 6, 6), dtype=cp.float32)
        integer_host = np.zeros((1, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="same array module"):
            gaussian_subvoxel_fit(corr_dev, integer_host)  # type: ignore[arg-type]
