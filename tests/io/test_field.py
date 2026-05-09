"""Tests for ``mamba_dvc.io.field``."""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.validate.synthetic import rigid_shift


def _sample_field_to_array(
    shape: tuple[int, int, int],
    field_callable,
) -> np.ndarray:
    """Densely sample an analytical field on every voxel into (Z,Y,X,3)."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    disp = field_callable(coords)
    return disp.reshape((*shape, 3)).astype(np.float32)


def _make_zarr_array(shape: tuple[int, ...], dtype: str = "float32") -> zarr.Array:
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    return root.create_array(name="flow", shape=shape, dtype=dtype, chunks=shape)


class TestGroundTruthFieldCallable:
    def test_rigid_shift_recovered_at_poi_centers(self) -> None:
        shape = (8, 12, 16)
        shift = (1.0, -0.5, 2.25)
        analytical = rigid_shift(shift)
        sampled = _sample_field_to_array(shape, analytical)

        gtf = GroundTruthField(sampled, convention="pull_back")
        # POI centers: a few interior coordinates.
        coords = np.array(
            [[2.0, 3.0, 4.0], [4.5, 6.5, 8.5], [6.0, 9.0, 12.0]],
            dtype=np.float32,
        )
        out = gtf(coords)
        expected = analytical(coords)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_off_grid_evaluation_uses_cubic_interpolation(self) -> None:
        # Linear ramp in dz only: u(z, y, x) = (z, 0, 0). Off-grid
        # cubic-spline samples in the interior should land within the
        # spline's reflect-boundary error budget (~1e-3 voxel for a
        # small volume) — well below DVC's 0.1-voxel target.
        shape = (16, 16, 16)
        zz, _, _ = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )
        sampled = np.zeros((*shape, 3), dtype=np.float32)
        sampled[..., 0] = zz

        gtf = GroundTruthField(sampled, convention="pull_back")
        coords = np.array([[7.5, 8.0, 8.5]], dtype=np.float32)
        out = gtf(coords)
        np.testing.assert_allclose(out, [[7.5, 0.0, 0.0]], atol=1e-2)

    def test_callable_protocol_conformance(self) -> None:
        # GroundTruthField should satisfy the DisplacementFunction protocol
        # well enough to be used with sample_on_grid.
        from mamba_dvc.types import GridSpec
        from mamba_dvc.validate.synthetic import sample_on_grid

        shape = (6, 6, 6)
        sampled = np.zeros((*shape, 3), dtype=np.float32)
        sampled[..., 1] = 0.75  # constant dy
        gtf = GroundTruthField(sampled, convention="pull_back")

        positions = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        grid = GridSpec(
            positions=positions,
            starts=np.zeros((1, 3), dtype=np.int64),
            grid_shape=(1, 1, 1),
            spacing=(1, 1, 1),
            window=(3, 3, 3),
            volume_shape=shape,
        )
        out = sample_on_grid(gtf, grid)
        np.testing.assert_allclose(out, [[0.0, 0.75, 0.0]], atol=1e-4)

    def test_invalid_array_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(Z, Y, X, 3\)"):
            GroundTruthField(np.zeros((4, 4, 4, 2), dtype=np.float32))


class TestAxisOrderRoundTrip:
    def test_3_zyx_and_zyx_3_storage_agree(self) -> None:
        shape = (5, 7, 9)
        sampled = _sample_field_to_array(shape, rigid_shift((0.5, -1.0, 1.5)))

        # Store as (Z, Y, X, 3).
        arr_zyx3 = _make_zarr_array((*shape, 3))
        arr_zyx3[:] = sampled

        # Store as (3, Z, Y, X).
        arr_3zyx = _make_zarr_array((3, *shape))
        arr_3zyx[:] = np.moveaxis(sampled, -1, 0)

        gtf_a = GroundTruthField.from_zarr(
            arr_zyx3, axis_order="zyx_3", convention="pull_back"
        )
        gtf_b = GroundTruthField.from_zarr(
            arr_3zyx, axis_order="3_zyx", convention="pull_back"
        )

        coords = np.array([[2.0, 3.0, 4.5], [4.0, 1.5, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(gtf_a(coords), gtf_b(coords), atol=1e-5)


class TestConventionFlip:
    def test_push_forward_equals_sign_flipped_pull_back(self) -> None:
        shape = (6, 6, 6)
        sampled = _sample_field_to_array(shape, rigid_shift((0.25, 0.5, -0.75)))

        gtf_pull = GroundTruthField(sampled, convention="pull_back")
        gtf_push = GroundTruthField(-sampled, convention="push_forward")

        coords = np.array([[2.0, 3.0, 4.0], [3.5, 1.5, 4.5]], dtype=np.float32)
        np.testing.assert_allclose(gtf_pull(coords), gtf_push(coords), atol=1e-5)


class TestFromZarrDryShape:
    def test_dry_shape_slices_spatial_axes(self) -> None:
        shape = (10, 12, 14)
        sampled = _sample_field_to_array(shape, rigid_shift((1.0, 0.0, 0.0)))
        arr = _make_zarr_array((3, *shape))
        arr[:] = np.moveaxis(sampled, -1, 0)

        gtf = GroundTruthField.from_zarr(
            arr,
            axis_order="3_zyx",
            convention="pull_back",
            dry_shape=(4, 6, 8),
        )
        assert gtf.shape == (4, 6, 8)
        # Constant shift should still be recovered after subblock crop.
        coords = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        out = gtf(coords)
        np.testing.assert_allclose(out, [[1.0, 0.0, 0.0]], atol=1e-4)

    def test_dry_shape_too_large_raises(self) -> None:
        arr = _make_zarr_array((3, 4, 4, 4))
        with pytest.raises(ValueError, match="exceeds source shape"):
            GroundTruthField.from_zarr(
                arr, axis_order="3_zyx", convention="pull_back", dry_shape=(5, 4, 4)
            )

    def test_non_4d_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        arr = root.create_array(
            name="flow", shape=(4, 4, 4), dtype="float32", chunks=(4, 4, 4)
        )
        with pytest.raises(ValueError, match="must be 4D"):
            GroundTruthField.from_zarr(arr, axis_order="3_zyx", convention="pull_back")

    def test_wrong_vector_axis_size_raises(self) -> None:
        arr = _make_zarr_array((4, 4, 4, 4))  # trailing axis is 4, not 3
        with pytest.raises(ValueError, match="vector axis must be size 3"):
            GroundTruthField.from_zarr(arr, axis_order="zyx_3", convention="pull_back")
