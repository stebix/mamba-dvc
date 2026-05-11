"""Tests for ``mamba_dvc.validate.synthetic``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.grid import build_grid
from mamba_dvc.validate.synthetic import (
    SyntheticPair,
    compose,
    make_pair,
    make_texture,
    rigid_shift,
    sample_on_grid,
    simple_shear,
    sinusoidal,
    uniform_dilation,
    warp,
)

make_texture_checked = jaxtyped(typechecker=beartype)(make_texture)
warp_checked = jaxtyped(typechecker=beartype)(warp)


def _coords_grid(shape: tuple[int, int, int]) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)


class TestMakeTexture:
    def test_is_standardized(self):
        tex = make_texture((16, 20, 24), sigma=1.5, seed=0)
        assert tex.shape == (16, 20, 24)
        assert tex.dtype == np.float32
        assert abs(float(tex.mean())) < 1e-3
        assert abs(float(tex.std()) - 1.0) < 1e-3

    def test_seed_is_deterministic(self):
        a = make_texture((12, 12, 12), sigma=1.5, seed=42)
        b = make_texture((12, 12, 12), sigma=1.5, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = make_texture((12, 12, 12), sigma=1.5, seed=0)
        b = make_texture((12, 12, 12), sigma=1.5, seed=1)
        assert not np.allclose(a, b)

    def test_shape_check_wrapper(self):
        tex = make_texture_checked((8, 8, 8), sigma=1.0, seed=0)
        assert tex.shape == (8, 8, 8)

    @pytest.mark.parametrize(
        ("shape", "sigma"),
        [
            ((8, 8), 1.0),
            ((8, 8, 0), 1.0),
            ((8, 8, 8), 0.0),
            ((8, 8, 8), -1.0),
        ],
    )
    def test_invalid_inputs_raise(self, shape: tuple[int, ...], sigma: float):
        with pytest.raises(ValueError):
            make_texture(shape, sigma=sigma)  # type: ignore[arg-type]


class TestFields:
    def test_rigid_shift_is_constant(self):
        field = rigid_shift((0.5, -1.5, 2.0))
        coords = np.array([[0, 0, 0], [5, 10, 15], [20, 30, 40]], dtype=np.float32)
        disp = field(coords)
        expected = np.tile([0.5, -1.5, 2.0], (3, 1)).astype(np.float32)
        np.testing.assert_allclose(disp, expected)

    def test_uniform_dilation_zero_at_center(self):
        center = (10.0, 12.0, 14.0)
        field = uniform_dilation(0.01, center)
        at_center = field(np.asarray([center], dtype=np.float32))
        np.testing.assert_allclose(at_center, 0.0, atol=1e-6)

    def test_uniform_dilation_linear_growth(self):
        center = (0.0, 0.0, 0.0)
        field = uniform_dilation(0.02, center)
        coords = np.array([[10, 20, 30], [50, 60, 70]], dtype=np.float32)
        disp = field(coords)
        np.testing.assert_allclose(disp, coords * 0.02, rtol=1e-5)

    def test_simple_shear_only_affects_shear_axis(self):
        field = simple_shear(0.05, shear_axis=0, gradient_axis=2)
        coords = np.array([[5, 7, 10]], dtype=np.float32)
        disp = field(coords)
        np.testing.assert_allclose(disp[0, 0], 0.5, rtol=1e-5)
        assert disp[0, 1] == 0.0
        assert disp[0, 2] == 0.0

    def test_simple_shear_rejects_same_axis(self):
        with pytest.raises(ValueError, match="must differ"):
            simple_shear(0.05, shear_axis=1, gradient_axis=1)

    @pytest.mark.parametrize(("s", "g"), [(-1, 0), (0, 3), (3, 0)])
    def test_simple_shear_axis_bounds(self, s: int, g: int):
        with pytest.raises(ValueError):
            simple_shear(0.05, shear_axis=s, gradient_axis=g)

    def test_sinusoidal_peak_and_phase(self):
        # Amplitude 2 along axis 0, wavelength 40 along axis 0.
        field = sinusoidal((2.0, 0.0, 0.0), (40.0, 40.0, 40.0))
        # At coord z=10 (= L/4), sin(2π/40 * 10) = sin(π/2) = 1 → disp = 2.
        coords = np.array([[10, 0, 0], [20, 0, 0], [0, 0, 0]], dtype=np.float32)
        disp = field(coords)
        np.testing.assert_allclose(disp[0, 0], 2.0, atol=1e-5)
        np.testing.assert_allclose(disp[1, 0], 0.0, atol=1e-5)  # sin(π) ≈ 0
        np.testing.assert_allclose(disp[2, 0], 0.0, atol=1e-5)

    def test_sinusoidal_rejects_bad_wavelength(self):
        with pytest.raises(ValueError):
            sinusoidal((1, 1, 1), (0.0, 10.0, 10.0))

    def test_compose_is_pointwise_sum(self):
        f1 = rigid_shift((1.0, 0.0, 0.0))
        f2 = rigid_shift((0.0, 2.0, -0.5))
        f3 = uniform_dilation(0.01, (0, 0, 0))
        combined = compose(f1, f2, f3)
        coords = np.array([[10, 20, 30], [0, 0, 0]], dtype=np.float32)
        expected = f1(coords) + f2(coords) + f3(coords)
        np.testing.assert_allclose(combined(coords), expected, rtol=1e-5)

    def test_compose_requires_at_least_one_field(self):
        with pytest.raises(ValueError):
            compose()


class TestWarp:
    def test_zero_field_is_identity(self):
        tex = make_texture((24, 24, 24), sigma=1.5, seed=0)
        zero = rigid_shift((0.0, 0.0, 0.0))
        warped = warp(tex, zero, order=3)
        # Cubic spline with zero displacement should reproduce the source
        # to within floating-point noise except possibly at boundaries;
        # compare the interior strictly.
        np.testing.assert_allclose(
            warped[4:-4, 4:-4, 4:-4],
            tex[4:-4, 4:-4, 4:-4],
            atol=1e-4,
        )

    def test_integer_rigid_shift_round_trip(self):
        shape = (48, 48, 48)
        tex = make_texture(shape, sigma=1.5, seed=7)
        shift = (3.0, -2.0, 5.0)
        warped = warp(tex, rigid_shift(shift), order=3)

        # Naive block-matching over integer lags on an interior window.
        # Expect the minimum SSD to be at exactly `shift`.
        win = 12
        z0, y0, x0 = 16, 18, 20
        ref_block = tex[z0 : z0 + win, y0 : y0 + win, x0 : x0 + win]

        radius = 3
        best = None
        best_ssd = np.inf
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    sz = z0 + int(shift[0]) + dz
                    sy = y0 + int(shift[1]) + dy
                    sx = x0 + int(shift[2]) + dx
                    def_block = warped[sz : sz + win, sy : sy + win, sx : sx + win]
                    ssd = float(((ref_block - def_block) ** 2).sum())
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best = (dz, dy, dx)
        assert best == (0, 0, 0)

    def test_warp_rejects_non_3d(self):
        with pytest.raises(ValueError):
            warp(np.zeros((8, 8), dtype=np.float32), rigid_shift((0, 0, 0)))

    def test_shape_check_wrapper(self):
        tex = make_texture((16, 16, 16), sigma=1.0, seed=0)
        warped = warp_checked(tex, rigid_shift((0.0, 0.0, 0.0)), order=3)
        assert warped.shape == tex.shape

    def test_pull_back_is_default(self):
        tex = make_texture((20, 20, 20), sigma=1.5, seed=3)
        field = rigid_shift((1.5, -0.5, 0.25))
        default = warp(tex, field, order=3)
        explicit = warp(tex, field, order=3, convention="pull_back")
        np.testing.assert_array_equal(default, explicit)

    def test_push_forward_equals_pull_back_of_negated_field(self):
        tex = make_texture((20, 20, 20), sigma=1.5, seed=5)
        shift = (1.0, -2.0, 0.5)
        pushed = warp(tex, rigid_shift(shift), order=3, convention="push_forward")
        # Sign-flip semantics: push_forward(u) ≡ pull_back(-u).
        flipped = (-shift[0], -shift[1], -shift[2])
        pulled = warp(tex, rigid_shift(flipped), order=3, convention="pull_back")
        np.testing.assert_allclose(pushed, pulled, atol=1e-6)

    def test_warp_rejects_unknown_convention(self):
        tex = make_texture((8, 8, 8), sigma=1.0, seed=0)
        with pytest.raises(ValueError, match="convention"):
            warp(tex, rigid_shift((0, 0, 0)), convention="bogus")  # type: ignore[arg-type]


class TestSampleOnGrid:
    def test_matches_direct_call(self):
        grid = build_grid((48, 48, 48), window=16, overlap=0.5)
        field = compose(
            rigid_shift((0.5, -1.0, 2.0)),
            uniform_dilation(0.01, (24, 24, 24)),
        )
        sampled = sample_on_grid(field, grid)
        direct = field(grid.positions)
        np.testing.assert_array_equal(sampled, direct)
        assert sampled.shape == (grid.positions.shape[0], 3)


class TestMakePair:
    def test_returns_pair_with_both_volumes(self):
        field = rigid_shift((0.5, 0.0, 0.0))
        pair = make_pair((24, 24, 24), field, texture_sigma=1.5, seed=0)
        assert isinstance(pair, SyntheticPair)
        assert pair.reference.shape == (24, 24, 24)
        assert pair.deformed.shape == (24, 24, 24)
        assert pair.reference.dtype == np.float32
        assert pair.deformed.dtype == np.float32
        assert pair.field is field
        assert pair.mask is None

    def test_mask_passthrough(self):
        mask = np.ones((16, 16, 16), dtype=np.bool_)
        pair = make_pair((16, 16, 16), rigid_shift((0, 0, 0)), mask=mask)
        assert pair.mask is mask

    def test_mask_shape_mismatch_raises(self):
        mask = np.ones((16, 16, 8), dtype=np.bool_)
        with pytest.raises(ValueError, match="mask shape"):
            make_pair((16, 16, 16), rigid_shift((0, 0, 0)), mask=mask)

    def test_deformed_equals_warp_of_reference(self):
        # Internal consistency: make_pair should be equivalent to
        # make_texture + warp with the same seed and field.
        shape = (20, 20, 20)
        field = rigid_shift((1.5, -0.5, 0.25))
        pair = make_pair(shape, field, seed=13, texture_sigma=1.5)
        tex = make_texture(shape, sigma=1.5, seed=13)
        deformed = warp(tex, field, order=3)
        np.testing.assert_array_equal(pair.reference, tex)
        np.testing.assert_array_equal(pair.deformed, deformed)
