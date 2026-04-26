"""Tests for ``mamba_dvc.core.ncc``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.ncc import correlate, peak_displacement

correlate_checked = jaxtyped(typechecker=beartype)(correlate)
peak_displacement_checked = jaxtyped(typechecker=beartype)(peak_displacement)


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """Return ``x`` with its per-batch mean subtracted."""
    return (x - x.mean(axis=(1, 2, 3), keepdims=True)).astype(np.float32)


class TestCorrelate:
    def test_self_correlation_peak_at_origin(self):
        rng = np.random.default_rng(0)
        ref = _zero_mean(rng.standard_normal((2, 8, 8, 8), dtype=np.float32))

        corr = correlate(ref, ref)

        # Peak of a self-correlation is at zero lag → index (0, 0, 0).
        for b in range(2):
            flat_argmax = int(np.argmax(corr[b]))
            iz, rem = divmod(flat_argmax, 8 * 8)
            iy, ix = divmod(rem, 8)
            assert (iz, iy, ix) == (0, 0, 0), f"batch {b}: peak at ({iz}, {iy}, {ix})"

    def test_self_correlation_peak_value_is_unity(self):
        rng = np.random.default_rng(1)
        ref = _zero_mean(rng.standard_normal((3, 8, 8, 8), dtype=np.float32))

        corr = correlate(ref, ref)

        peaks = corr.reshape(3, -1).max(axis=1)
        np.testing.assert_allclose(peaks, 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "shift",
        [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (-2, 5, -3), (4, -1, 2)],
    )
    def test_recovers_integer_circshift(self, shift: tuple[int, int, int]):
        rng = np.random.default_rng(42)
        ref = _zero_mean(rng.standard_normal((1, 12, 12, 12), dtype=np.float32))

        # Convention: deformed(x) = reference(x - shift) → cyclic roll by `shift`.
        deformed = np.roll(ref, shift=shift, axis=(1, 2, 3))

        corr = correlate(ref, deformed)
        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray(shift, dtype=np.int64))

    def test_scaling_input_rescales_peak(self):
        rng = np.random.default_rng(2)
        ref = _zero_mean(rng.standard_normal((1, 8, 8, 8), dtype=np.float32))

        peak_unit = correlate(ref, ref).reshape(1, -1).max(axis=1)
        peak_scaled = correlate(ref, (3.0 * ref).astype(np.float32)).reshape(1, -1).max(axis=1)

        # Whole-window L2 normalization: scaling one input scales the
        # numerator by k and the denominator by k as well, so the peak
        # NCC value is invariant.
        np.testing.assert_allclose(peak_unit, peak_scaled, atol=1e-5)

    def test_zero_input_does_not_produce_nan(self):
        zeros = np.zeros((1, 6, 6, 6), dtype=np.float32)
        rng = np.random.default_rng(3)
        ref = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))

        corr = correlate(zeros, ref)

        assert np.all(np.isfinite(corr))
        np.testing.assert_allclose(corr, 0.0, atol=1e-5)

    def test_output_shape_dtype_contiguity(self):
        rng = np.random.default_rng(4)
        ref = _zero_mean(rng.standard_normal((2, 10, 10, 10), dtype=np.float32))

        corr = correlate(ref, ref)

        assert corr.shape == (2, 10, 10, 10)
        assert corr.dtype == np.float32
        assert corr.flags["C_CONTIGUOUS"]

    def test_anisotropic_window(self):
        # Even though v1 uses cubic windows, the kernel should not
        # accidentally rely on cubic shape — pyramid levels in v2 will
        # exercise this path.
        rng = np.random.default_rng(5)
        ref = _zero_mean(rng.standard_normal((1, 6, 8, 10), dtype=np.float32))
        deformed = np.roll(ref, shift=(2, -3, 4), axis=(1, 2, 3))

        corr = correlate(ref, deformed)
        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([2, -3, 4], dtype=np.int64))

    def test_runtime_shape_check_via_jaxtyping(self):
        rng = np.random.default_rng(6)
        ref = _zero_mean(rng.standard_normal((2, 6, 6, 6), dtype=np.float32))
        corr = correlate_checked(ref, ref)
        assert corr.shape == (2, 6, 6, 6)

    def test_does_not_mutate_inputs(self):
        rng = np.random.default_rng(7)
        ref = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))
        deformed = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))
        ref_copy = ref.copy()
        deformed_copy = deformed.copy()

        _ = correlate(ref, deformed)

        np.testing.assert_array_equal(ref, ref_copy)
        np.testing.assert_array_equal(deformed, deformed_copy)

    def test_shape_mismatch_raises(self):
        ref = np.zeros((2, 6, 6, 6), dtype=np.float32)
        deformed = np.zeros((2, 6, 6, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            correlate(ref, deformed)

    def test_wrong_ndim_raises(self):
        ref = np.zeros((6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="4D"):
            correlate(ref, ref)  # type: ignore[arg-type]

    def test_wrong_dtype_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            correlate(ref, ref)  # type: ignore[arg-type]

    def test_non_positive_eps_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="eps"):
            correlate(ref, ref, eps=0.0)


class TestPeakDisplacement:
    def test_origin_peak(self):
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 0, 0, 0] = 1.0

        disp, value = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([0, 0, 0], dtype=np.int64))
        assert float(value[0]) == pytest.approx(1.0)

    def test_negative_lag_wrap(self):
        # Index w-1 means lag = -1 along that axis after the wrap.
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 7, 0, 0] = 1.0

        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([-1, 0, 0], dtype=np.int64))

    def test_positive_lag(self):
        # Index 3 (with w=8, half=4) → positive lag 3.
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 0, 3, 0] = 1.0

        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([0, 3, 0], dtype=np.int64))

    def test_at_half_window_wraps_negative(self):
        # Index == half (4 with w=8): the wrap convention treats this
        # as a negative lag (w/2 → -w/2). Documented behavior.
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 4, 0, 0] = 1.0

        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([-4, 0, 0], dtype=np.int64))

    def test_batch_independence(self):
        corr = np.zeros((3, 8, 8, 8), dtype=np.float32)
        corr[0, 1, 0, 0] = 1.0
        corr[1, 0, 7, 0] = 1.0
        corr[2, 0, 0, 5] = 1.0

        disp, _ = peak_displacement(corr)

        expected = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -3]], dtype=np.int64)
        np.testing.assert_array_equal(disp, expected)

    def test_runtime_shape_check_via_jaxtyping(self):
        corr = np.zeros((2, 6, 6, 6), dtype=np.float32)
        corr[0, 0, 0, 0] = 1.0
        corr[1, 0, 0, 0] = 1.0
        disp, value = peak_displacement_checked(corr)
        assert disp.shape == (2, 3)
        assert value.shape == (2,)

    def test_wrong_ndim_raises(self):
        corr = np.zeros((6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="4D"):
            peak_displacement(corr)  # type: ignore[arg-type]

    def test_wrong_dtype_raises(self):
        corr = np.zeros((1, 6, 6, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            peak_displacement(corr)  # type: ignore[arg-type]


@pytest.mark.gpu
class TestNCCGPU:
    def test_correlate_matches_cpu_on_cupy_inputs(self):
        cp = pytest.importorskip("cupy")

        rng = np.random.default_rng(8)
        ref_host = _zero_mean(rng.standard_normal((2, 8, 8, 8), dtype=np.float32))
        deformed_host = np.roll(ref_host, shift=(1, -2, 3), axis=(1, 2, 3))

        ref_dev = cp.asarray(ref_host)
        deformed_dev = cp.asarray(deformed_host)

        corr_dev = correlate(ref_dev, deformed_dev)
        corr_host = correlate(ref_host, deformed_host)

        np.testing.assert_allclose(cp.asnumpy(corr_dev), corr_host, atol=1e-4)

    def test_peak_displacement_matches_cpu_on_cupy_inputs(self):
        cp = pytest.importorskip("cupy")

        corr_host = np.zeros((2, 8, 8, 8), dtype=np.float32)
        corr_host[0, 1, 0, 0] = 1.0
        corr_host[1, 0, 7, 0] = 1.0

        corr_dev = cp.asarray(corr_host)
        disp_dev, value_dev = peak_displacement(corr_dev)
        disp_host, value_host = peak_displacement(corr_host)

        np.testing.assert_array_equal(cp.asnumpy(disp_dev), disp_host)
        np.testing.assert_allclose(cp.asnumpy(value_dev), value_host, atol=1e-6)

    def test_rejects_mixed_host_device_inputs(self):
        cp = pytest.importorskip("cupy")
        ref_dev = cp.zeros((1, 6, 6, 6), dtype=cp.float32)
        deformed_host = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="same array module"):
            correlate(ref_dev, deformed_host)  # type: ignore[arg-type]
