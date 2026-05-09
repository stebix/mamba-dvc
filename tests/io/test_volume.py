"""Tests for ``mamba_dvc.io.volume``: pure CPU, no disk I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from mamba_dvc.io.volume import load_volume, open_zarr_array


def _make_zarr_array(
    shape: tuple[int, ...],
    dtype: np.dtype | str,
    fill: int | float | bool = 0,
) -> zarr.Array:
    """Build an in-memory zarr array with deterministic contents."""
    store = zarr.storage.MemoryStore()
    root = zarr.create_group(store=store)
    arr = root.create_array(
        name="data",
        shape=shape,
        dtype=np.dtype(dtype),
        chunks=shape,
    )
    if isinstance(fill, bool):
        arr[:] = fill
    else:
        rng = np.random.default_rng(0)
        arr[:] = rng.standard_normal(size=shape).astype(np.dtype(dtype), copy=False)
    return arr


class TestLoadVolumeFloat32:
    def test_full_volume_float_input_is_passthrough(self) -> None:
        arr = _make_zarr_array((4, 6, 8), "float32")
        out = load_volume(arr, as_float32=True)
        assert out.shape == (4, 6, 8)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, np.asarray(arr[:]))

    def test_full_volume_uint16_is_coerced(self) -> None:
        arr = _make_zarr_array((3, 5, 7), "uint16")
        out = load_volume(arr, as_float32=True)
        assert out.dtype == np.float32
        assert out.shape == (3, 5, 7)
        assert out.flags["C_CONTIGUOUS"]


class TestLoadVolumeBool:
    def test_full_volume_bool_input_is_passthrough(self) -> None:
        arr = _make_zarr_array((4, 4, 4), "bool", fill=True)
        out = load_volume(arr, as_float32=False)
        assert out.dtype == np.bool_
        assert out.shape == (4, 4, 4)
        assert bool(out.all())

    def test_full_volume_uint8_is_coerced_to_bool(self) -> None:
        arr = _make_zarr_array((2, 3, 4), "uint8", fill=0)
        # Inject some non-zero entries so coercion has signal.
        arr[0, 0, 0] = 1
        arr[1, 2, 3] = 5
        out = load_volume(arr, as_float32=False)
        assert out.dtype == np.bool_
        assert out[0, 0, 0]
        assert out[1, 2, 3]
        assert not out[1, 1, 1]


class TestDryShape:
    def test_centered_subblock_extracted(self) -> None:
        arr = _make_zarr_array((10, 12, 14), "float32")
        out = load_volume(arr, dry_shape=(4, 6, 8), as_float32=True)
        assert out.shape == (4, 6, 8)
        # Centered slice: (10-4)//2 = 3, (12-6)//2 = 3, (14-8)//2 = 3
        np.testing.assert_array_equal(out, np.asarray(arr[3:7, 3:9, 3:11]))

    def test_centered_subblock_consistent_across_dtypes(self) -> None:
        # Same dry_shape on a bool mask centers identically.
        arr = _make_zarr_array((10, 12, 14), "bool", fill=False)
        arr[5, 6, 7] = True
        out = load_volume(arr, dry_shape=(4, 6, 8), as_float32=False)
        assert out.shape == (4, 6, 8)
        # The True voxel sits inside the centered subblock.
        assert out[5 - 3, 6 - 3, 7 - 3]

    def test_dry_shape_equal_to_full_shape(self) -> None:
        arr = _make_zarr_array((4, 4, 4), "float32")
        out = load_volume(arr, dry_shape=(4, 4, 4), as_float32=True)
        np.testing.assert_array_equal(out, np.asarray(arr[:]))

    def test_dry_shape_larger_than_volume_raises(self) -> None:
        arr = _make_zarr_array((4, 4, 4), "float32")
        with pytest.raises(ValueError, match="exceeds source shape"):
            load_volume(arr, dry_shape=(5, 4, 4))


class TestNonThreeDimensionalRaises:
    def test_2d_array_rejected(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        arr = root.create_array(name="data", shape=(8, 8), dtype="float32", chunks=(8, 8))
        with pytest.raises(ValueError, match="ndim=2"):
            load_volume(arr)

    def test_4d_array_rejected(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        arr = root.create_array(
            name="data", shape=(2, 4, 4, 4), dtype="float32", chunks=(2, 4, 4, 4)
        )
        with pytest.raises(ValueError, match="ndim=4"):
            load_volume(arr)


class TestOpenZarrArray:
    def test_open_returns_array_handle(self, tmp_path: Path) -> None:
        store_path = tmp_path / "scan.zarr"
        root = zarr.open_group(str(store_path), mode="w")
        a = root.create_array(name="vol", shape=(4, 4, 4), dtype="float32", chunks=(4, 4, 4))
        a[:] = np.arange(64, dtype=np.float32).reshape(4, 4, 4)

        handle = open_zarr_array(store_path, "vol")
        assert handle.shape == (4, 4, 4)
        assert handle.dtype == np.float32
        np.testing.assert_array_equal(handle[:], a[:])

    def test_missing_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_zarr_array(tmp_path / "nope.zarr", "vol")
