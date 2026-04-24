"""Batched subvolume extraction from a volume at given start indices.

This module owns step 2 of the DVC pipeline (plan
``docs/plans/overview.md`` §2): gather per-POI subvolumes out of a
reference, deformed, or mask volume into a batched tensor suitable for
the preprocessing stage in :mod:`mamba_dvc.core.window`.

Design choices (plan §2, §4, §6):

- Single verb, one volume at a time. Callers run it three times (ref,
  deformed, mask) and compose the results. Keeps the module boring,
  testable, and reusable for debugging views or future densification.
- Caller owns batching. ``starts`` is already the chunk the caller
  wants; the per-batch memory budget (§6: ~0.43 GB at batch 256 and
  ``W = 96``) is the caller's responsibility.
- Duck-typed across NumPy and CuPy via ``cupy.get_array_module``. The
  implementation is identical — both array libraries support the
  stride-trick rolling-window view and fancy indexing.

The gather is implemented as an ``as_strided`` rolling-window view over
the source volume, followed by fancy indexing on the outer three axes.
This materializes exactly one ``(batch, wz, wy, wx)`` output tensor and
never builds per-axis coordinate tensors — minimizing intermediate
allocations on GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float32, Int64

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = ["extract_subvolumes"]


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``.

    Falls back to NumPy when CuPy is not installed, so host-only
    machines still exercise the same code path as production GPUs.
    """
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def _normalize_window(window: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate and coerce a 3-tuple window into ``(wz, wy, wx)``."""
    if len(window) != 3:
        raise ValueError(f"window must have length 3, got {len(window)}")
    wz, wy, wx = (int(w) for w in window)
    if wz <= 0 or wy <= 0 or wx <= 0:
        raise ValueError(f"window entries must be positive, got {window}")
    return (wz, wy, wx)


def _rolling_window_view(
    volume: Float32[np.ndarray, "z y x"],
    window: tuple[int, int, int],
) -> np.ndarray:
    """Return a zero-copy rolling-window view over ``volume``.

    The resulting view has shape
    ``(z - wz + 1, y - wy + 1, x - wx + 1, wz, wy, wx)`` and shares
    storage with ``volume``. Fancy-indexing the outer three axes with a
    ``(batch, 3)`` array of start coordinates materializes a
    ``(batch, wz, wy, wx)`` gather in a single kernel, without building
    per-axis coordinate tensors.

    The outer and inner stride triples are identical because, in both
    cases, advancing one step traverses one voxel of the underlying
    buffer — it only means different things semantically (window start
    vs. position inside the window).
    """
    xp = _array_module(volume)
    sz, sy, sx = volume.strides
    wz, wy, wx = window
    vz, vy, vx = volume.shape
    shape = (vz - wz + 1, vy - wy + 1, vx - wx + 1, wz, wy, wx)
    strides = (sz, sy, sx, sz, sy, sx)
    return xp.lib.stride_tricks.as_strided(volume, shape=shape, strides=strides)


def extract_subvolumes(
    volume: Float32[np.ndarray, "z y x"],
    starts: Int64[np.ndarray, "batch 3"],
    window: tuple[int, int, int],
) -> Float32[np.ndarray, "batch w w w"]:
    """Gather subvolumes at the given integer start indices.

    Parameters
    ----------
    volume
        ``(z, y, x)`` source volume. May be a NumPy array (host) or a
        CuPy array (device); the output uses the same array module.
        Dtype must be ``float32``.
    starts
        ``(batch, 3)`` array of integer ``(z0, y0, x0)`` start indices,
        one per requested subvolume. Must be ``int64`` and live in the
        same array module as ``volume`` — no implicit host/device
        transfer happens here.
    window
        ``(wz, wy, wx)`` subvolume size in voxels. All entries must be
        positive and no larger than the corresponding axis of
        ``volume``.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        ``(batch, wz, wy, wx)`` float32 C-contiguous array of gathered
        subvolumes. Array module matches ``volume``.

    Raises
    ------
    ValueError
        If ``volume`` is not 3D float32, if ``starts`` has wrong shape
        or dtype, if ``window`` is malformed, or if any requested
        subvolume would escape the volume bounds.

    Notes
    -----
    Implemented as a rolling-window ``as_strided`` view + fancy index
    on the outer three axes. The view is read-only by contract (shared
    storage with ``volume``); callers must not write through it. The
    returned array is a fresh allocation and is safe to mutate.

    Bounds are validated eagerly on ``starts`` before indexing the
    strided view, because ``as_strided`` itself performs no bounds
    checking and would happily return garbage for out-of-range indices.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got ndim={volume.ndim}")
    if volume.dtype != np.float32:
        raise ValueError(f"volume must be float32, got {volume.dtype}")

    if starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError(f"starts must have shape (batch, 3), got {starts.shape}")
    if starts.dtype != np.int64:
        raise ValueError(f"starts must be int64, got {starts.dtype}")

    win = _normalize_window(window)
    vz, vy, vx = volume.shape
    wz, wy, wx = win
    if wz > vz or wy > vy or wx > vx:
        raise ValueError(
            f"window {win} exceeds volume shape {volume.shape} on at least one axis"
        )

    xp = _array_module(volume)
    if _array_module(starts) is not xp:
        raise ValueError(
            "starts and volume must live in the same array module "
            "(both NumPy or both CuPy); no implicit transfer is performed"
        )

    if starts.shape[0] == 0:
        return xp.empty((0, wz, wy, wx), dtype=xp.float32)

    # Eager bounds check — as_strided does no checking of its own.
    # 0-d reductions support __int__ in both NumPy and CuPy, so the pulls
    # stay clean without branching on the array module.
    min_z = int(starts[:, 0].min())
    min_y = int(starts[:, 1].min())
    min_x = int(starts[:, 2].min())
    max_z = int(starts[:, 0].max())
    max_y = int(starts[:, 1].max())
    max_x = int(starts[:, 2].max())
    if min_z < 0 or min_y < 0 or min_x < 0:
        raise ValueError(f"starts must be non-negative, got min=({min_z}, {min_y}, {min_x})")
    if max_z + wz > vz or max_y + wy > vy or max_x + wx > vx:
        raise ValueError(
            f"subvolume at start ({max_z}, {max_y}, {max_x}) with window {win} "
            f"exceeds volume shape {volume.shape}"
        )

    view = _rolling_window_view(volume, win)
    gathered = view[starts[:, 0], starts[:, 1], starts[:, 2]]
    return xp.ascontiguousarray(gathered, dtype=xp.float32)
