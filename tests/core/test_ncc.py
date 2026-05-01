"""Tests for ``mamba_dvc.core.ncc``.

The two NCC kernels (cyclic and linear) live side-by-side; tests
parametrize over both modes whenever the property under test is shared.
The bias-toward-zero regression test (the one
``docs/insights/error-minimization.md`` §"Plan revisions" calls out as
"the cheapest possible test that would have caught the current bias on
day one") only asserts the strict accuracy bound for the linear+overlap
combination -- the cyclic kernel is documented to fail it.
"""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.ncc import (
    correlate,
    correlate_cyclic,
    correlate_linear,
    peak_displacement,
)
from mamba_dvc.validate.synthetic import make_texture

correlate_checked = jaxtyped(typechecker=beartype)(correlate)
peak_displacement_checked = jaxtyped(typechecker=beartype)(peak_displacement)


# --- Mode/normalization sweep --------------------------------------------------
# Every combination supported by the dispatcher. Cyclic only pairs with
# global; linear pairs with both. Used to parametrize the property tests
# below.
ALL_MODES = [
    pytest.param("cyclic", "global", id="cyclic-global"),
    pytest.param("linear", "global", id="linear-global"),
    pytest.param("linear", "overlap", id="linear-overlap"),
]


def _zero_mean(x: np.ndarray) -> np.ndarray:
    """Return ``x`` with its per-batch mean subtracted."""
    return (x - x.mean(axis=(1, 2, 3), keepdims=True)).astype(np.float32)


@pytest.mark.parametrize(("mode", "normalization"), ALL_MODES)
class TestCorrelateProperties:
    def test_self_correlation_peak_at_origin(self, mode, normalization):
        rng = np.random.default_rng(0)
        ref = _zero_mean(rng.standard_normal((2, 8, 8, 8), dtype=np.float32))

        corr = correlate(ref, ref, mode=mode, normalization=normalization)

        for b in range(2):
            flat_argmax = int(np.argmax(corr[b]))
            iz, rem = divmod(flat_argmax, 8 * 8)
            iy, ix = divmod(rem, 8)
            assert (iz, iy, ix) == (0, 0, 0), f"batch {b}: peak at ({iz}, {iy}, {ix})"

    def test_self_correlation_peak_value_is_unity(self, mode, normalization):
        rng = np.random.default_rng(1)
        ref = _zero_mean(rng.standard_normal((3, 8, 8, 8), dtype=np.float32))

        corr = correlate(ref, ref, mode=mode, normalization=normalization)

        peaks = corr.reshape(3, -1).max(axis=1)
        # The lag-0 overlap is the full window, so the overlap-aware
        # denominator collapses to the global L2; both modes must hit 1.
        np.testing.assert_allclose(peaks, 1.0, atol=1e-5)

    def test_scaling_input_rescales_peak(self, mode, normalization):
        rng = np.random.default_rng(2)
        ref = _zero_mean(rng.standard_normal((1, 8, 8, 8), dtype=np.float32))

        peak_unit = (
            correlate(ref, ref, mode=mode, normalization=normalization)
            .reshape(1, -1)
            .max(axis=1)
        )
        peak_scaled = (
            correlate(
                ref,
                (3.0 * ref).astype(np.float32),
                mode=mode,
                normalization=normalization,
            )
            .reshape(1, -1)
            .max(axis=1)
        )

        # NCC is scale-invariant in either kernel: numerator and
        # denominator both scale by k.
        np.testing.assert_allclose(peak_unit, peak_scaled, atol=1e-5)

    def test_zero_input_does_not_produce_nan(self, mode, normalization):
        zeros = np.zeros((1, 6, 6, 6), dtype=np.float32)
        rng = np.random.default_rng(3)
        ref = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))

        corr = correlate(zeros, ref, mode=mode, normalization=normalization)

        assert np.all(np.isfinite(corr))
        np.testing.assert_allclose(corr, 0.0, atol=1e-5)

    def test_output_shape_dtype_contiguity(self, mode, normalization):
        rng = np.random.default_rng(4)
        ref = _zero_mean(rng.standard_normal((2, 10, 10, 10), dtype=np.float32))

        corr = correlate(ref, ref, mode=mode, normalization=normalization)

        assert corr.shape == (2, 10, 10, 10)
        assert corr.dtype == np.float32
        assert corr.flags["C_CONTIGUOUS"]

    def test_anisotropic_window(self, mode, normalization):
        rng = np.random.default_rng(5)
        ref = _zero_mean(rng.standard_normal((1, 6, 8, 10), dtype=np.float32))
        # np.roll keeps the cyclic kernel's recovery exact at integer
        # shifts; the linear kernel is also exact for np.roll inputs
        # because the wrap-around content is the same content (rolled).
        deformed = np.roll(ref, shift=(2, -3, 4), axis=(1, 2, 3))

        corr = correlate(ref, deformed, mode=mode, normalization=normalization)
        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([2, -3, 4], dtype=np.int64))

    def test_recovers_integer_circshift(self, mode, normalization):
        # A cyclic shift is exact in both kernels: in the cyclic kernel
        # by construction; in the linear kernel because np.roll keeps
        # the same set of voxels in play (the wrap region pairs voxels
        # that were also paired in the no-shift case, so the linear
        # numerator at lag k matches a pure self-correlation).
        rng = np.random.default_rng(42)
        ref = _zero_mean(rng.standard_normal((1, 12, 12, 12), dtype=np.float32))
        for shift in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (-2, 5, -3)]:
            deformed = np.roll(ref, shift=shift, axis=(1, 2, 3))
            corr = correlate(ref, deformed, mode=mode, normalization=normalization)
            disp, _ = peak_displacement(corr)
            np.testing.assert_array_equal(
                disp[0],
                np.asarray(shift, dtype=np.int64),
                err_msg=f"mode={mode} norm={normalization} shift={shift}",
            )

    def test_does_not_mutate_inputs(self, mode, normalization):
        rng = np.random.default_rng(7)
        ref = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))
        deformed = _zero_mean(rng.standard_normal((1, 6, 6, 6), dtype=np.float32))
        ref_copy = ref.copy()
        deformed_copy = deformed.copy()

        _ = correlate(ref, deformed, mode=mode, normalization=normalization)

        np.testing.assert_array_equal(ref, ref_copy)
        np.testing.assert_array_equal(deformed, deformed_copy)


class TestDispatcherValidation:
    def test_unknown_mode_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="mode must be"):
            correlate(ref, ref, mode="bogus", normalization="global")  # type: ignore[arg-type]

    def test_unknown_normalization_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="normalization must be"):
            correlate(ref, ref, mode="linear", normalization="bogus")  # type: ignore[arg-type]

    def test_cyclic_with_overlap_normalization_raises(self):
        # Overlap-aware normalization is only meaningful in linear mode;
        # the dispatcher rejects the combination explicitly so callers
        # cannot silently get a global-normalized cyclic kernel under
        # an overlap-implying flag.
        ref = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match=r"cyclic.*global"):
            correlate(ref, ref, mode="cyclic", normalization="overlap")

    def test_shape_mismatch_raises(self):
        ref = np.zeros((2, 6, 6, 6), dtype=np.float32)
        deformed = np.zeros((2, 6, 6, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            correlate(ref, deformed, mode="linear", normalization="overlap")

    def test_wrong_ndim_raises(self):
        ref = np.zeros((6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="4D"):
            correlate(ref, ref, mode="linear", normalization="overlap")  # type: ignore[arg-type]

    def test_wrong_dtype_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            correlate(ref, ref, mode="linear", normalization="overlap")  # type: ignore[arg-type]

    def test_non_positive_eps_raises(self):
        ref = np.zeros((1, 6, 6, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="eps"):
            correlate(ref, ref, mode="linear", normalization="overlap", eps=0.0)

    def test_runtime_shape_check_via_jaxtyping(self):
        rng = np.random.default_rng(6)
        ref = _zero_mean(rng.standard_normal((2, 6, 6, 6), dtype=np.float32))
        corr = correlate_checked(ref, ref, mode="linear", normalization="overlap")
        assert corr.shape == (2, 6, 6, 6)


class TestLinearVsCyclicBias:
    """Regression test for the cyclic shrinkage bias.

    From ``docs/insights/error-minimization.md``: the cyclic kernel
    biases recovered displacements toward zero by a magnitude that
    grows with ``|u|/W``. The linear+overlap kernel does not.

    This test takes a clean band-limited noise volume, applies
    *non-cyclic* sub-window shifts via ``np.roll`` over a larger volume
    (so the wrap region carries content that is statistically
    indistinguishable from in-window content for the cyclic kernel; the
    linear kernel sees only the in-window overlap), runs the integer
    peak through both kernels, and asserts that the linear+overlap
    integer peak matches the truth exactly while the cyclic kernel may
    or may not -- but in either case the linear+overlap peak is
    closer.
    """

    @pytest.mark.parametrize(
        "shift",
        [(2, -3, 1), (1, 0, -2), (3, 0, -2), (-2, 4, 0)],
    )
    def test_linear_overlap_recovers_integer_shift_exactly(self, shift):
        # Build a textured volume large enough that we can take an
        # off-centre window and shift the deformed copy without the
        # window touching the volume edges. Window 32 with the texture
        # stats from the insights doc.
        big = make_texture((96, 96, 96), seed=1).astype(np.float32)
        # Centre the window at (48, 48, 48), sized 32 → starts at 32.
        w = 32
        z0 = y0 = x0 = 32
        ref = big[z0 : z0 + w, y0 : y0 + w, x0 : x0 + w]
        # Deformed: shifted slab from the same big volume. This is the
        # only construction that exposes the cyclic-vs-linear gap; an
        # np.roll of `ref` would produce identical results in both.
        dz, dy, dx = shift
        deformed = big[
            z0 - dz : z0 - dz + w,
            y0 - dy : y0 - dy + w,
            x0 - dx : x0 - dx + w,
        ]

        # Mean-subtract; no Tukey (matches Lewis NCC convention).
        ref_b = (ref - ref.mean()).astype(np.float32)[None]
        def_b = (deformed - deformed.mean()).astype(np.float32)[None]

        corr_lin = correlate_linear(ref_b, def_b, normalization="overlap")
        peak_lin, _ = peak_displacement(corr_lin)

        np.testing.assert_array_equal(peak_lin[0], np.asarray(shift, dtype=np.int64))

    @pytest.mark.parametrize(
        "shift",
        [(2, -3, 1), (3, 0, -2), (-3, 4, 1)],
    )
    def test_linear_overlap_beats_cyclic_on_integer_peak(self, shift):
        big = make_texture((96, 96, 96), seed=2).astype(np.float32)
        w = 32
        z0 = y0 = x0 = 32
        ref = big[z0 : z0 + w, y0 : y0 + w, x0 : x0 + w]
        dz, dy, dx = shift
        deformed = big[
            z0 - dz : z0 - dz + w,
            y0 - dy : y0 - dy + w,
            x0 - dx : x0 - dx + w,
        ]
        ref_b = (ref - ref.mean()).astype(np.float32)[None]
        def_b = (deformed - deformed.mean()).astype(np.float32)[None]

        corr_lin = correlate_linear(ref_b, def_b, normalization="overlap")
        corr_cyc = correlate_cyclic(ref_b, def_b)
        peak_lin, _ = peak_displacement(corr_lin)
        peak_cyc, _ = peak_displacement(corr_cyc)

        truth = np.asarray(shift, dtype=np.int64)
        err_lin = np.linalg.norm((peak_lin[0] - truth).astype(np.float64))
        err_cyc = np.linalg.norm((peak_cyc[0] - truth).astype(np.float64))
        # Linear is exact; cyclic may match or be off by 1 in some
        # axis. The point of the regression test is the strict equality
        # for linear; we don't over-constrain cyclic behaviour here.
        assert err_lin <= err_cyc, (
            f"linear err={err_lin} should be ≤ cyclic err={err_cyc} for shift={shift}"
        )


class TestPeakDisplacement:
    def test_origin_peak(self):
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 0, 0, 0] = 1.0

        disp, value = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([0, 0, 0], dtype=np.int64))
        assert float(value[0]) == pytest.approx(1.0)

    def test_negative_lag_wrap(self):
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 7, 0, 0] = 1.0

        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([-1, 0, 0], dtype=np.int64))

    def test_positive_lag(self):
        corr = np.zeros((1, 8, 8, 8), dtype=np.float32)
        corr[0, 0, 3, 0] = 1.0

        disp, _ = peak_displacement(corr)

        np.testing.assert_array_equal(disp[0], np.asarray([0, 3, 0], dtype=np.int64))

    def test_at_half_window_wraps_negative(self):
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
    @pytest.mark.parametrize(("mode", "normalization"), ALL_MODES)
    def test_correlate_matches_cpu_on_cupy_inputs(self, mode, normalization):
        cp = pytest.importorskip("cupy")

        rng = np.random.default_rng(8)
        ref_host = _zero_mean(rng.standard_normal((2, 8, 8, 8), dtype=np.float32))
        deformed_host = np.roll(ref_host, shift=(1, -2, 3), axis=(1, 2, 3))

        ref_dev = cp.asarray(ref_host)
        deformed_dev = cp.asarray(deformed_host)

        corr_dev = correlate(ref_dev, deformed_dev, mode=mode, normalization=normalization)
        corr_host = correlate(ref_host, deformed_host, mode=mode, normalization=normalization)

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
            correlate(ref_dev, deformed_host, mode="linear", normalization="overlap")  # type: ignore[arg-type]
