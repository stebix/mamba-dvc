"""Host-side materializer for zarr-backed 3D volumes.

The single chokepoint for turning a ``zarr.Array`` into a contiguous NumPy
buffer ready for ``correlate()`` / ``correlate_multi_gpu()``. Subblock
slicing (``dry_shape``), dtype coercion, and contiguity all live here so
no other module re-implements them.

See ``docs/plans/zarr-interface.md`` §2 ("Materialization happens at one
chokepoint") for the rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

__all__ = ["center_slice", "load_volume", "open_zarr_array"]


def open_zarr_array(store_path: Path, internal_path: str) -> zarr.Array[Any]:
    """Open a zarr array at ``store_path[internal_path]``.

    Works for both v3 and v2 stores via the high-level ``zarr.open``
    entry point. Reads metadata only; no voxels are materialized.

    Parameters
    ----------
    store_path
        Path to the zarr store on disk.
    internal_path
        Path of the array inside the store (e.g. ``"base/volume0"``).

    Returns
    -------
    zarr.Array
        Lazy handle to the array. Slicing / ``[:]`` triggers I/O.

    Raises
    ------
    FileNotFoundError
        If ``store_path`` does not exist.
    TypeError
        If ``internal_path`` resolves to a group rather than an array.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"zarr store not found: {store_path}")
    root = zarr.open_group(str(store_path), mode="r")
    node = root[internal_path]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"{store_path}:{internal_path} is a group, not an array")
    return node


def center_slice(
    shape: tuple[int, ...], dry: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """Return centered ``(z, y, x)`` slice of size ``dry`` inside ``shape``.

    Raises
    ------
    ValueError
        If any entry of ``dry`` exceeds the matching entry of ``shape``.
    """
    out: list[slice] = []
    for full, want in zip(shape, dry, strict=True):
        if want > full:
            raise ValueError(f"dry_shape entry {want} exceeds source shape {full}")
        lo = (full - want) // 2
        out.append(slice(lo, lo + want))
    return tuple(out)  # type: ignore[return-value]


def load_volume(
    array: zarr.Array[Any],
    *,
    dry_shape: tuple[int, int, int] | None = None,
    as_float32: bool = True,
) -> np.ndarray[Any, Any]:
    """Materialize a 3D zarr array into a contiguous host buffer.

    Parameters
    ----------
    array
        Lazy zarr array; must be 3D.
    dry_shape
        Optional ``(z, y, x)`` size of a centered subblock. ``None``
        (default) reads the full array. Mirrors the
        ``--dry-run-shape`` flag of ``scripts/run_e2e_zarr.py``.
    as_float32
        ``True`` (default) coerces to ``np.float32``; ``False`` coerces
        to ``np.bool_`` (mask path). The bool path is deliberately
        narrow — there is no third option.

    Returns
    -------
    numpy.ndarray
        Contiguous ``(z, y, x)`` buffer in the requested dtype.

    Raises
    ------
    ValueError
        If ``array`` is not 3D, or if ``dry_shape`` exceeds the array's
        shape on any axis.
    """
    if array.ndim != 3:
        raise ValueError(f"expected 3D array, got ndim={array.ndim}")

    if dry_shape is not None:
        sl = center_slice(array.shape, dry_shape)
        data = np.asarray(array[sl])
    else:
        data = np.asarray(array[:])

    if as_float32:
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
    else:
        if data.dtype != np.bool_:
            data = data.astype(np.bool_, copy=False)

    return np.ascontiguousarray(data)
