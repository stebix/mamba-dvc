"""Host-side loader for boolean mask volumes.

A thin specialization of :func:`mamba_dvc.io.volume.load_volume` that
forces ``as_float32=False`` and optionally validates the loaded shape
against a caller-supplied reference shape (typically the reference
volume's). Lives next to ``volume.py`` so the reader and the perf
harness share one materialization path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr
from jaxtyping import Bool

from mamba_dvc.io.volume import load_volume

__all__ = ["load_mask"]


def load_mask(
    array: zarr.Array[Any],
    *,
    dry_shape: tuple[int, int, int] | None = None,
    expected_shape: tuple[int, int, int] | None = None,
) -> Bool[np.ndarray, "z y x"]:
    """Load a mask zarr array as a contiguous bool buffer.

    Parameters
    ----------
    array
        Lazy zarr array; must be 3D. Any non-bool dtype is coerced to
        ``np.bool_``.
    dry_shape
        Optional ``(z, y, x)`` size of a centered subblock. Must match
        the value used for the corresponding volumes so the mask
        aligns.
    expected_shape
        Optional ``(z, y, x)`` shape the loaded mask must match
        (typically the reference volume's post-``dry_shape`` shape).
        ``None`` skips the check.

    Returns
    -------
    numpy.ndarray
        Contiguous ``(z, y, x)`` bool buffer.

    Raises
    ------
    ValueError
        If ``array`` is not 3D, ``dry_shape`` exceeds the array's
        shape, or the loaded shape disagrees with ``expected_shape``.
    """
    data = load_volume(array, dry_shape=dry_shape, as_float32=False)
    if expected_shape is not None and data.shape != expected_shape:
        raise ValueError(f"mask shape {data.shape} does not match expected {expected_shape}")
    return data
