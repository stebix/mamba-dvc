"""Tests for ``mamba_dvc.validate.phantoms``.

The phantom is intended for visualization showcases and DVC smoke
tests, so the assertions check structural properties (shape, dtype,
region intensity ordering, defect cavity, determinism) rather than
exact pixel values.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from mamba_dvc.validate.phantoms import (
    PhantomSpec,
    default_phantom,
    render_phantom,
)
from mamba_dvc.validate.synthetic import make_pair, rigid_shift

SMALL_SHAPE: tuple[int, int, int] = (32, 48, 48)
SHOWCASE_SHAPE: tuple[int, int, int] = (96, 128, 128)


def _coord_grids(shape):
    nz, ny, nx = shape
    zz = np.arange(nz).reshape(nz, 1, 1)
    yy = np.arange(ny).reshape(1, ny, 1)
    xx = np.arange(nx).reshape(1, 1, nx)
    return zz, yy, xx


class TestRenderPhantom:
    def test_shape_and_dtype(self):
        phantom = default_phantom(SMALL_SHAPE)
        assert phantom.shape == SMALL_SHAPE
        assert phantom.dtype == np.float32

    def test_is_finite_everywhere(self):
        phantom = default_phantom(SMALL_SHAPE)
        assert np.isfinite(phantom).all()

    def test_implant_region_is_brightest(self):
        # Default phantom: implant cylinder along x through volume center,
        # radius 4.0. Sample voxels well inside the implant tube.
        phantom = default_phantom(SHOWCASE_SHAPE)
        nz, ny, _nx = SHOWCASE_SHAPE
        cz, cy = nz / 2.0, ny / 2.0

        zz, yy, _xx = _coord_grids(SHOWCASE_SHAPE)
        # Cylinder along x: radial distance is over (z, y).
        radial2 = (zz - cz) ** 2 + (yy - cy) ** 2
        implant_core = np.broadcast_to(radial2 <= 4.0, SHOWCASE_SHAPE)
        # Background corner region (well outside the cortical shell).
        corner_region = np.zeros(SHOWCASE_SHAPE, dtype=bool)
        corner_region[:, :8, :8] = True

        assert phantom[implant_core].mean() > phantom[corner_region].mean()

    def test_defect_region_is_below_cortical(self):
        phantom = default_phantom(SHOWCASE_SHAPE)
        nz, ny, nx = SHOWCASE_SHAPE
        # Defect center per default_phantom: (cz + nz/8, cy - ny/8, cx + nx/8).
        dcz, dcy, dcx = nz / 2.0 + nz / 8.0, ny / 2.0 - ny / 8.0, nx / 2.0 + nx / 8.0
        zz, yy, xx = _coord_grids(SHOWCASE_SHAPE)
        # Sample a small core well inside the defect cavity.
        defect_core = ((zz - dcz) ** 2 + (yy - dcy) ** 2 + (xx - dcx) ** 2) <= 4.0
        defect_core = np.broadcast_to(defect_core, SHOWCASE_SHAPE)

        # Cortical shell sample: in the bone z-axis cylinder, near outer wall.
        cy_b, cx_b = ny / 2.0, nx / 2.0
        radial2 = (yy - cy_b) ** 2 + (xx - cx_b) ** 2
        cortical_band = np.broadcast_to(
            (radial2 >= 50.0**2) & (radial2 <= 56.0**2), SHOWCASE_SHAPE
        )

        assert defect_core.any()
        assert cortical_band.any()
        assert phantom[defect_core].mean() < phantom[cortical_band].mean()

    def test_non_degenerate_variance(self):
        phantom = default_phantom(SMALL_SHAPE)
        # Pure-background or constant scaffolds would give std ≈ 0; the
        # texture + multi-region composition should drive it well above.
        assert phantom.std() > 0.05

    def test_determinism_under_fixed_seed(self):
        a = default_phantom(SMALL_SHAPE, seed=42)
        b = default_phantom(SMALL_SHAPE, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_phantom(self):
        a = default_phantom(SMALL_SHAPE, seed=0)
        b = default_phantom(SMALL_SHAPE, seed=1)
        # Backgrounds and cortical rings will overlap, but the trabecular
        # struts and texture differ.
        assert not np.array_equal(a, b)

    def test_disabled_defect_leaves_more_bone(self):
        with_defect = default_phantom(SHOWCASE_SHAPE)
        spec = PhantomSpec(shape=SHOWCASE_SHAPE, defect_radius=0.0)
        without_defect = render_phantom(spec)
        # Carving a cavity removes mass: total intensity must drop.
        assert with_defect.sum() < without_defect.sum()

    def test_runtime_under_one_second_for_showcase_shape(self):
        t0 = time.perf_counter()
        default_phantom(SHOWCASE_SHAPE)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"default_phantom too slow: {elapsed:.3f}s"


class TestSpecValidation:
    def test_thickness_exceeding_outer_radius_raises(self):
        spec = PhantomSpec(
            shape=SMALL_SHAPE,
            cortical_outer_radius=5.0,
            cortical_thickness=10.0,
        )
        with pytest.raises(ValueError, match="cortical_thickness"):
            render_phantom(spec)

    def test_invalid_porosity_raises(self):
        spec = PhantomSpec(shape=SMALL_SHAPE, trabecular_porosity=1.5)
        with pytest.raises(ValueError, match="porosity"):
            render_phantom(spec)


class TestMakePairWithReference:
    def test_phantom_reference_is_used_as_is(self):
        phantom = default_phantom(SMALL_SHAPE, seed=3)
        pair = make_pair(SMALL_SHAPE, rigid_shift((0.0, 0.0, 0.0)), reference=phantom)
        np.testing.assert_array_equal(pair.reference, phantom)

    def test_zero_field_makes_deformed_close_to_reference(self):
        phantom = default_phantom(SMALL_SHAPE, seed=3)
        pair = make_pair(SMALL_SHAPE, rigid_shift((0.0, 0.0, 0.0)), reference=phantom)
        np.testing.assert_allclose(pair.deformed, phantom, atol=1e-5)

    def test_reference_shape_mismatch_raises(self):
        bad = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="reference shape"):
            make_pair(SMALL_SHAPE, rigid_shift((0.0, 0.0, 0.0)), reference=bad)

    def test_reference_dtype_mismatch_raises(self):
        bad = np.zeros(SMALL_SHAPE, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            make_pair(SMALL_SHAPE, rigid_shift((0.0, 0.0, 0.0)), reference=bad)
