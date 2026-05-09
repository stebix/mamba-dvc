"""Shared validation, mask resolution, and the batched NCC inner loop.

Internal collaboration surface between
:mod:`mamba_dvc.pipeline.correlate` (single-device pure orchestrator)
and :mod:`mamba_dvc.gpu.dispatch` (multi-GPU sibling). Both modules
import the same helpers from here so there is one source of truth for:

- per-mode Tukey defaults (``TUKEY_DEFAULTS``),
- window normalization (``normalize_window``),
- mask resolution + validation (``resolve_masks``),
- the batched FFT NCC + peakfit + search-radius gate
  (``correlate_admitted_subset``).

The leading-underscore *module name* signals "internal to the
``mamba_dvc.pipeline`` package -- not part of the public API and not
covered by SemVer guarantees". Symbols themselves are not underscored,
so importers do not need to suppress ``reportPrivateUsage``.

The dispatch seam — what each multi-GPU worker calls on its share of
the admitted POIs — is :func:`correlate_admitted_subset`. It is
array-module-agnostic: when ``reference`` is a CuPy array (workers
replicate to device on attach), it runs entirely on device and returns
CuPy arrays. When ``reference`` is NumPy, it stays on host. All input
arrays must live in the same module — the dispatch parent and the
public ``correlate()`` enforce that invariant before calling in.

Sparse-output convention
------------------------
:func:`correlate_admitted_subset` returns three full-length arrays
covering the entire lattice (length ``prod(grid.grid_shape)``). Slots
outside the requested ``admitted_idx`` carry zero displacement, zero
confidence, and ``POIStatus.MASKED``. The dispatch parent allocates
once and scatters each shard's slots into the merged buffers; no
merging logic is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Bool, Float32, Int64, UInt8

from mamba_dvc.core.extract import extract_subvolumes
from mamba_dvc.core.ncc import NCCMode, NCCNormalization, peak_displacement
from mamba_dvc.core.ncc import correlate as correlate_ncc
from mamba_dvc.core.peakfit import gaussian_subvoxel_fit
from mamba_dvc.core.window import preprocess_subvolumes
from mamba_dvc.types import GridSpec, POIStatus

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = [
    "TUKEY_DEFAULTS",
    "NCCMode",
    "NCCNormalization",
    "correlate_admitted_subset",
    "normalize_window",
    "resolve_masks",
]

# Per-mode Tukey defaults. Linear NCC does not need spectral-leakage
# suppression from a window because zero-padding eliminates the cyclic
# wrap; the box itself plays the role of the window in the Lewis
# denominator. Cyclic NCC retains the conventional 0.25 taper from
# plan §2 to suppress the leakage that the wrap would otherwise inject.
TUKEY_DEFAULTS: dict[NCCMode, float] = {NCCMode.LINEAR: 0.0, NCCMode.CYCLIC: 0.25}


def _array_module(array: Any) -> Any:
    """Return the NumPy-compatible array module for ``array``."""
    if _cp is None:
        return np
    return _cp.get_array_module(array)


def normalize_window(window: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Coerce a scalar or 3-tuple window into ``(wz, wy, wx)``.

    Parameters
    ----------
    window
        Scalar (applies isotropically) or per-axis tuple.

    Returns
    -------
    tuple of int
        ``(wz, wy, wx)`` window extents.

    Raises
    ------
    ValueError
        If a non-scalar input does not have exactly three entries.
    """
    if isinstance(window, int):
        return (window, window, window)
    if len(window) != 3:
        raise ValueError(f"window tuple must have length 3, got {len(window)}")
    wz, wy, wx = (int(w) for w in window)
    return (wz, wy, wx)


def resolve_masks(
    mask: Bool[np.ndarray, "z y x"] | None,
    deformed_mask: Bool[np.ndarray, "z y x"] | None,
    volume_shape: tuple[int, int, int],
) -> tuple[Bool[np.ndarray, "z y x"], Bool[np.ndarray, "z y x"]]:
    """Resolve effective reference and deformed masks from optional inputs.

    Centralizes the validation + fallback rules so the single-device
    orchestrator and the multi-GPU dispatcher share a single source of
    truth. When ``mask`` is ``None`` the reference mask is an all-True
    array of ``volume_shape``; when ``deformed_mask`` is ``None`` it
    falls back to the reference mask.

    Parameters
    ----------
    mask, deformed_mask
        Optional ``(z, y, x)`` boolean validity masks.
    volume_shape
        Shape both masks are required to match.

    Returns
    -------
    tuple
        ``(effective_mask, effective_def_mask)``, both ``(z, y, x)``
        boolean arrays.

    Raises
    ------
    ValueError
        If a supplied mask has the wrong shape or dtype.
    """
    if mask is None:
        effective_mask = np.ones(volume_shape, dtype=np.bool_)
    else:
        if mask.shape != volume_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {volume_shape}"
            )
        if mask.dtype != np.bool_:
            raise ValueError(f"mask must have bool dtype, got {mask.dtype}")
        effective_mask = mask

    if deformed_mask is None:
        effective_def_mask = effective_mask
    else:
        if deformed_mask.shape != volume_shape:
            raise ValueError(
                f"deformed_mask shape {deformed_mask.shape} does not match "
                f"volume shape {volume_shape}"
            )
        if deformed_mask.dtype != np.bool_:
            raise ValueError(f"deformed_mask must have bool dtype, got {deformed_mask.dtype}")
        effective_def_mask = deformed_mask

    return effective_mask, effective_def_mask


def correlate_admitted_subset(
    reference: Float32[np.ndarray, "z y x"],
    deformed: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"],
    deformed_mask: Bool[np.ndarray, "z y x"],
    grid: GridSpec,
    admitted_idx: Int64[np.ndarray, "k"],
    *,
    search_radius: int,
    batch_size: int,
    eps: float,
    ncc_mode: NCCMode,
    ncc_normalization: NCCNormalization,
    tukey_alpha: float,
) -> tuple[
    Float32[np.ndarray, "points 3"],
    Float32[np.ndarray, "points"],
    UInt8[np.ndarray, "points"],
]:
    """Run NCC + peakfit + search-radius gate on a subset of admitted POIs.

    Sees the full reference / deformed / mask volumes; processes only
    the POIs whose flat-lattice index appears in ``admitted_idx``.
    Result arrays cover the *entire* lattice (length
    ``prod(grid.grid_shape)``) and are sparse: slots outside
    ``admitted_idx`` carry zero displacement, zero confidence, and
    ``POIStatus.MASKED``. This is the contract the multi-GPU dispatch
    layer relies on -- each worker's per-shard output is scattered
    back into the parent's full-length buffers without merging logic.

    The function is array-module-agnostic. When ``reference`` is a
    CuPy array (volumes replicated to GPU by the dispatch worker), the
    helper runs entirely on device and returns CuPy arrays. When all
    inputs are NumPy (the single-device public API path), it stays on
    host. All arrays passed in must live in the same module:
    ``reference``, ``deformed``, ``mask``, ``deformed_mask``,
    ``grid.starts``, and ``admitted_idx``.

    Parameters
    ----------
    reference, deformed
        ``(z, y, x)`` float32 volumes.
    mask, deformed_mask
        ``(z, y, x)`` boolean validity masks, already resolved by
        :func:`resolve_masks`. The deformed-frame mask is part of the
        v2-ready API; in v1 it is the same object as ``mask`` when the
        caller did not supply one.
    grid
        Lattice descriptor produced by
        :func:`mamba_dvc.core.grid.build_grid`. Its ``starts`` field
        must live in the same array module as the volumes.
    admitted_idx
        ``(k,)`` int64 flat-lattice indices for the POIs this call is
        responsible for. May be empty (the function is a no-op then,
        returning the all-MASKED defaults).
    search_radius
        Maximum permissible integer-lag displacement along any axis.
        POIs whose integer peak exceeds this radius are flagged
        ``OUT_OF_RANGE`` and have their displacement and confidence
        zeroed.
    batch_size
        Number of POIs processed per FFT batch.
    eps
        Safety floor passed through to
        :func:`mamba_dvc.core.ncc.correlate`.
    ncc_mode, ncc_normalization
        NCC kernel selection. The helper trusts the values it
        receives; the orchestrator and dispatcher validate them.
    tukey_alpha
        Already-resolved Tukey taper fraction (callers apply the
        per-mode default before calling).

    Returns
    -------
    displacements
        ``(points, 3)`` float32, fractional ``(dz, dy, dx)`` voxel
        displacements. Zero outside ``admitted_idx`` and at
        OUT_OF_RANGE slots.
    confidence
        ``(points,)`` float32 peak NCC values. Zero outside
        ``admitted_idx`` and at OUT_OF_RANGE slots.
    status
        ``(points,)`` uint8 :class:`POIStatus` value per POI. Slots
        outside ``admitted_idx`` are ``MASKED``; admitted slots are
        ``OK`` (or ``OUT_OF_RANGE`` when the integer peak escaped the
        search radius). The outlier test is *not* applied here -- it
        runs on the assembled full-lattice arrays in the caller, so
        cross-shard neighborhoods see consistent data.

    Notes
    -----
    Sparse-output convention: the dispatch parent allocates
    ``displacements``, ``confidence``, ``status`` of length
    ``points`` once, then scatters each shard's output by writing
    ``out[shard_admitted_idx] = shard_arr[shard_admitted_idx]``. Slots
    that no shard claims keep their initial ``MASKED`` / zero values,
    which matches the shape of "this POI was rejected by mask
    admission". The single-device path does the same scatter
    trivially: one shard whose ``admitted_idx`` covers every admitted
    POI.
    """
    xp = _array_module(reference)
    n_points = int(np.prod(grid.grid_shape))

    displacements = xp.zeros((n_points, 3), dtype=xp.float32)
    confidence = xp.zeros(n_points, dtype=xp.float32)
    status = xp.full(n_points, int(POIStatus.MASKED), dtype=xp.uint8)

    if int(admitted_idx.shape[0]) == 0:
        return displacements, confidence, status

    # Mark shard slots OK by default; OUT_OF_RANGE is overlaid below
    # for any POI whose integer peak escaped the search radius.
    status[admitted_idx] = int(POIStatus.OK)

    n_admitted = int(admitted_idx.shape[0])
    for batch_start in range(0, n_admitted, batch_size):
        batch_slice = admitted_idx[batch_start : batch_start + batch_size]
        chunk = grid.starts[batch_slice]

        ref_sv = extract_subvolumes(reference, chunk, grid.window)
        def_sv = extract_subvolumes(deformed, chunk, grid.window)
        mref_sv = extract_subvolumes(mask, chunk, grid.window)
        mdef_sv = extract_subvolumes(deformed_mask, chunk, grid.window)

        ref_pp = preprocess_subvolumes(ref_sv, mref_sv, tukey_alpha=tukey_alpha)
        def_pp = preprocess_subvolumes(def_sv, mdef_sv, tukey_alpha=tukey_alpha)

        corr = correlate_ncc(
            ref_pp,
            def_pp,
            mode=ncc_mode,
            normalization=ncc_normalization,
            eps=eps,
        )
        integer, peak = peak_displacement(corr)
        fractional = gaussian_subvoxel_fit(corr, integer)

        displacements[batch_slice] = integer.astype(xp.float32) + fractional
        confidence[batch_slice] = peak

    # search_radius enforcement on the shard's POIs only. integer-peak
    # magnitude is what determines whether the lag is unambiguous; the
    # fractional refinement only nudges by < 1 voxel and cannot push a
    # borderline POI from valid to invalid in a way that matters at our
    # accuracy budget.
    shard_disp = displacements[admitted_idx]
    shard_max = xp.abs(shard_disp).max(axis=1)
    out_of_range_mask = shard_max > xp.float32(search_radius)
    out_of_range_idx = admitted_idx[out_of_range_mask]
    status[out_of_range_idx] = int(POIStatus.OUT_OF_RANGE)
    displacements[out_of_range_idx] = xp.float32(0.0)
    confidence[out_of_range_idx] = xp.float32(0.0)

    return displacements, confidence, status
