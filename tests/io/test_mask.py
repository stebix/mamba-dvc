"""Tests for ``mamba_dvc.io.mask``."""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from mamba_dvc.io.mask import load_mask


def _make_zarr_array(
    shape: tuple[int, ...],
    dtype: np.dtype | str,
    fill: int | float | bool = 0,
) -> zarr.Array:
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
        arr[:] = np.zeros(shape, dtype=np.dtype(dtype))
    return arr


class TestLoadMask:
    def test_bool_passthrough(self) -> None:
        arr = _make_zarr_array((4, 4, 4), "bool", fill=True)
        out = load_mask(arr)
        assert out.dtype == np.bool_
        assert out.shape == (4, 4, 4)
        assert bool(out.all())

    def test_uint8_coerced(self) -> None:
        arr = _make_zarr_array((3, 4, 5), "uint8", fill=0)
        arr[1, 2, 3] = 1
        out = load_mask(arr)
        assert out.dtype == np.bool_
        assert out[1, 2, 3]
        assert not out[0, 0, 0]

    def test_dry_shape_passthrough(self) -> None:
        arr = _make_zarr_array((10, 10, 10), "bool", fill=True)
        out = load_mask(arr, dry_shape=(4, 4, 4))
        assert out.shape == (4, 4, 4)
        assert bool(out.all())

    def test_expected_shape_match_passes(self) -> None:
        arr = _make_zarr_array((4, 6, 8), "bool", fill=True)
        out = load_mask(arr, expected_shape=(4, 6, 8))
        assert out.shape == (4, 6, 8)

    def test_expected_shape_mismatch_raises(self) -> None:
        arr = _make_zarr_array((4, 6, 8), "bool", fill=True)
        with pytest.raises(ValueError, match="does not match expected"):
            load_mask(arr, expected_shape=(4, 6, 9))

    def test_expected_shape_with_dry_shape(self) -> None:
        # When dry_shape is supplied, expected_shape is checked against
        # the post-slice shape, not the original.
        arr = _make_zarr_array((10, 10, 10), "bool", fill=True)
        out = load_mask(arr, dry_shape=(4, 4, 4), expected_shape=(4, 4, 4))
        assert out.shape == (4, 4, 4)
