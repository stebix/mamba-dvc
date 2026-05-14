"""Smoke tests for VTK-bridge converters in ``mamba_dvc.viz._conversion``.

Skipped when the ``viz`` extra is not installed (no PyVista on the
import path). These exercise *structure* — array names, shapes, dtypes,
spacing — not pixel output, so they stay deterministic across drivers.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytestmark = pytest.mark.viz

from mamba_dvc.types import (  # noqa: E402
    DisplacementField,
    GridSpec,
    POIStatus,
    VoxelSpacing,
)
from mamba_dvc.viz._conversion import (  # noqa: E402
    field_to_polydata,
    grid_to_structured,
    mask_to_image_data,
    volume_to_image_data,
)


def _tiny_volume(shape: tuple[int, int, int] = (4, 5, 6)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape, dtype=np.float32)


def _tiny_field(grid_shape: tuple[int, int, int] = (2, 3, 4)) -> DisplacementField:
    nz, ny, nx = grid_shape
    n = nz * ny * nx
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, size=(n, 3)).astype(np.float32)
    displacements = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    valid[::5] = False
    confidence = rng.uniform(0, 1, size=n).astype(np.float32)
    status = np.where(valid, POIStatus.OK, POIStatus.MASKED).astype(np.uint8)
    return DisplacementField(
        positions=positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=grid_shape,
        spacing=(2, 2, 2),
        window=(4, 4, 4),
    )


class TestVolumeToImageData:
    def test_dimensions_and_spacing_are_xyz_swapped(self):
        vol = _tiny_volume((4, 5, 6))
        grid = volume_to_image_data(vol, spacing=VoxelSpacing((10.0, 20.0, 30.0), "um"))
        # PyVista reports dimensions as (nx, ny, nz).
        assert tuple(grid.dimensions) == (6, 5, 4)
        assert tuple(grid.spacing) == pytest.approx((30.0, 20.0, 10.0))
        assert "intensity" in grid.point_data

    def test_intensity_roundtrip_preserves_value_at_known_voxel(self):
        vol = _tiny_volume((4, 5, 6))
        grid = volume_to_image_data(vol)
        # Reconstruct (z, y, x) by inverting the (x, y, z) Fortran ravel.
        nx, ny, nz = grid.dimensions
        flat = np.asarray(grid.point_data["intensity"])
        recon = flat.reshape((nx, ny, nz), order="F").transpose(2, 1, 0)
        np.testing.assert_array_equal(recon, vol)

    def test_rejects_non_float32(self):
        with pytest.raises(TypeError, match="float32"):
            volume_to_image_data(np.zeros((2, 2, 2), dtype=np.float64))

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="3D"):
            volume_to_image_data(np.zeros((4, 4), dtype=np.float32))


class TestMaskToImageData:
    def test_uint8_cast_and_dimensions(self):
        mask = np.ones((4, 5, 6), dtype=bool)
        grid = mask_to_image_data(mask)
        assert tuple(grid.dimensions) == (6, 5, 4)
        assert grid.point_data["mask"].dtype == np.uint8
        assert int(np.asarray(grid.point_data["mask"]).max()) == 1

    def test_rejects_non_bool(self):
        with pytest.raises(TypeError, match="bool"):
            mask_to_image_data(np.zeros((2, 2, 2), dtype=np.uint8))


class TestFieldToPolyData:
    def test_only_valid_drops_invalid_points(self):
        field = _tiny_field()
        poly = field_to_polydata(field, only_valid=True)
        assert poly.n_points == int(field.valid.sum())
        for name in ("displacement", "magnitude", "valid", "status", "confidence"):
            assert name in poly.point_data

    def test_keep_all_when_only_valid_false(self):
        field = _tiny_field()
        poly = field_to_polydata(field, only_valid=False)
        assert poly.n_points == field.positions.shape[0]

    def test_displacement_is_active_vectors(self):
        field = _tiny_field()
        poly = field_to_polydata(field)
        active = poly.active_vectors_name
        assert active == "displacement"

    def test_stride_decimates_after_valid_filter(self):
        field = _tiny_field()
        # Baseline: only valid POIs.
        baseline = field_to_polydata(field, only_valid=True)
        strided = field_to_polydata(field, only_valid=True, stride=2)
        # ``stride=2`` keeps every other POI from the valid subset.
        assert strided.n_points == (baseline.n_points + 1) // 2

    def test_stride_one_is_passthrough(self):
        field = _tiny_field()
        baseline = field_to_polydata(field, only_valid=True)
        strided = field_to_polydata(field, only_valid=True, stride=1)
        assert strided.n_points == baseline.n_points

    def test_stride_zero_raises(self):
        field = _tiny_field()
        with pytest.raises(ValueError, match="positive integer"):
            field_to_polydata(field, stride=0)

    def test_magnitude_is_l2_of_scaled_displacement(self):
        field = _tiny_field()
        spacing = VoxelSpacing((2.0, 1.0, 0.5), "um")
        poly = field_to_polydata(field, only_valid=False, spacing=spacing)
        scaled = field.displacements.copy()
        scaled[:, 0] *= 2.0
        scaled[:, 1] *= 1.0
        scaled[:, 2] *= 0.5
        expected = np.linalg.norm(scaled, axis=1).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(poly.point_data["magnitude"]), expected, rtol=1e-5
        )


class TestGridToStructured:
    def _grid(self) -> GridSpec:
        nz, ny, nx = 2, 3, 4
        zs = np.arange(nz)
        ys = np.arange(ny)
        xs = np.arange(nx)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        positions = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
        starts = positions.astype(np.int64)
        return GridSpec(
            positions=positions,
            starts=starts,
            grid_shape=(nz, ny, nx),
            spacing=(1, 1, 1),
            window=(2, 2, 2),
            volume_shape=(8, 8, 8),
        )

    def test_dimensions_match_grid_shape(self):
        grid = self._grid()
        structured = grid_to_structured(grid)
        nz, ny, nx = grid.grid_shape
        assert tuple(structured.dimensions) == (nx, ny, nz)
        assert structured.n_points == nz * ny * nx

    def test_scalar_attached(self):
        grid = self._grid()
        scalars = np.arange(grid.positions.shape[0], dtype=np.float32)
        structured = grid_to_structured(grid, scalars=scalars, scalar_name="confidence")
        assert "confidence" in structured.point_data
        np.testing.assert_array_equal(np.asarray(structured.point_data["confidence"]), scalars)

    def test_scalar_shape_mismatch_raises(self):
        grid = self._grid()
        with pytest.raises(ValueError, match="does not match"):
            grid_to_structured(grid, scalars=np.zeros(3, dtype=np.float32))
