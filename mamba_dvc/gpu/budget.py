"""Pre-flight VRAM oracle: pick a batch size from free memory and config.

Implements the functional core of the deferred ``_resolve_batch_size``
heuristic called out in :mod:`mamba_dvc.gpu.dispatch` and
``docs/plans/overview.md`` §6. Plan: ``docs/plans/batchsize-oracle-v1.md``.

The module answers a single question the call site of
:func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu` previously had to
guess: given the free VRAM on each GPU and the resolved DVC config
(window, NCC mode, normalization, mask presence), what is the largest
``batch_size`` that fits with a safety margin?

Three layers, each independently testable:

- :func:`kernel_footprint` -- per-POI transient bytes + per-call fixed
  bytes for one ``(window, mode, normalization)`` triple. Pure
  arithmetic over an analytic cost model (see §4 of the plan).
- :func:`resident_bytes` -- bytes the workers will pin on each device
  before any FFT work starts (replicated reference, deformed, mask,
  optional distinct deformed mask).
- :func:`estimate_max_batch` / :func:`recommend_batch_size` -- the
  arithmetic and the orchestrator entry point. The latter takes the
  minimum of free VRAM across ``device_ids`` and returns one cross-
  device batch size; per-device returns are a deferred v2 hook.

Cost model status
-----------------
The per-POI coefficients in :data:`_PER_POI_COEFFS` and
:data:`_FIXED_COEFFS` are first-pass analytic estimates from plan §4.
They are intentionally conservative (they sum the maximum-resident
working set rather than tracking lifetimes precisely). The plan calls
for a build-time calibration script to overwrite them with measured
values; until that script lands they are the working numbers, and
:data:`_HEADROOM_DEFAULT` carries the model error.

CuPy isolation
--------------
Only :func:`probe_free_vram` touches CuPy. Everything else is pure
arithmetic over plain Python ints, so the cost model is fully testable
on CPU-only hosts. The pattern mirrors :mod:`mamba_dvc.core.ncc` and
:mod:`mamba_dvc.gpu.dispatch`, where ``_cp`` is guarded at import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from mamba_dvc.core.ncc import NCCMode, NCCNormalization

try:
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    _cp = None  # type: ignore[assignment]

__all__ = [
    "BudgetInputs",
    "KernelFootprint",
    "estimate_max_batch",
    "is_cupy_available",
    "kernel_footprint",
    "probe_free_vram",
    "recommend_batch_size",
    "resident_bytes",
]


def is_cupy_available() -> bool:
    """Return ``True`` iff CuPy is importable on this host.

    Provided so callers can gate the ``"auto"`` batch-resolution path
    without catching :class:`RuntimeError` from :func:`probe_free_vram`
    (which is also raised on the unrelated underflow path). The single
    seam for the CuPy-presence check keeps the rest of the package off
    of the private ``_cp`` symbol.
    """
    return _cp is not None


# Per-POI coefficient ``c`` such that the FFT kernel's transient
# working set is approximately ``c * prod(window) * batch`` bytes.
# Numbers are analytic estimates from plan ``batchsize-oracle-v1.md`` §4
# (sum of every buffer kept alive simultaneously, including a CUFFT
# workspace fudge). They are intentionally on the high side so a
# pre-calibration deployment errs toward smaller batches rather than
# OOM. See the module docstring for the upgrade path.
# TODO when centred layout lands: the centred-correlation linear kernel
# flagged in ``docs/plans/overview.md`` §10 has a different padding
# pattern (no full ``2W`` doubling), so the LINEAR coefficients here will
# need recomputation -- otherwise ``recommend_batch_size`` will ship a
# stale model and either OOM or under-utilize.
_PER_POI_COEFFS: dict[tuple[NCCMode, NCCNormalization], int] = {
    (NCCMode.CYCLIC, NCCNormalization.GLOBAL): 36,
    (NCCMode.LINEAR, NCCNormalization.GLOBAL): 232,
    (NCCMode.LINEAR, NCCNormalization.OVERLAP): 480,
}


# Per-call fixed cost: indicator-box spectrum (overlap path only) plus
# the small constant that does not scale with batch. The plan absorbs
# the cuFFT plan cache (~hundreds of MB) into ``headroom_fraction``
# rather than modelling it precisely, because the cache is allocated
# lazily and survives across calls -- the headroom factor in
# :func:`estimate_max_batch` is the right place for it.
_FIXED_COEFFS: dict[tuple[NCCMode, NCCNormalization], int] = {
    (NCCMode.CYCLIC, NCCNormalization.GLOBAL): 0,
    (NCCMode.LINEAR, NCCNormalization.GLOBAL): 0,
    (NCCMode.LINEAR, NCCNormalization.OVERLAP): 8,  # one (2W)^3 complex64 box spectrum
}


_HEADROOM_DEFAULT = 0.15
"""Fraction of free VRAM held in reserve for plan cache + fragmentation.

Plan §7: 5% covers the cuFFT plan cache empirically; the extra 10%
buys a meaningful safety margin without halving the batch. Tune at the
call site if a deployment proves the model unusually tight or loose.
"""


@dataclass(frozen=True)
class KernelFootprint:
    """Memory cost of one FFT NCC kernel configuration.

    Parameters
    ----------
    per_poi_bytes
        Transient bytes per point-of-interest. Multiplied by
        ``batch_size`` to get the batch-scaling cost.
    fixed_bytes
        Batch-independent transient bytes (indicator-box spectrum, etc.).

    Notes
    -----
    The numbers are intentionally a *maximum-resident* sum -- they do
    not track buffer lifetimes precisely. CuPy and cuFFT may re-use
    storage within a single call, so the true peak is usually smaller
    than ``per_poi_bytes * batch + fixed_bytes``. Treating the model as
    an upper bound is the conservative choice for a batch oracle: a
    smaller-than-optimal batch is a throughput annoyance; an overshoot
    is a worker-killing OOM.
    """

    per_poi_bytes: int
    fixed_bytes: int

    def __post_init__(self) -> None:
        """Validate non-negativity."""
        if self.per_poi_bytes < 0:
            raise ValueError(f"per_poi_bytes must be non-negative, got {self.per_poi_bytes}")
        if self.fixed_bytes < 0:
            raise ValueError(f"fixed_bytes must be non-negative, got {self.fixed_bytes}")


@dataclass(frozen=True)
class BudgetInputs:
    """Resolved DVC config for the cost model.

    Parameters
    ----------
    volume_shape
        ``(z, y, x)`` shape of the reference volume. Used for the
        replicated-on-device residency calculation in
        :func:`resident_bytes`.
    window
        ``(wz, wy, wx)`` subvolume size. Drives the per-POI cost via
        ``prod(window)``.
    mode
        :class:`NCCMode` selection -- ``cyclic`` or ``linear``.
    normalization
        :class:`NCCNormalization` selection -- ``global`` or
        ``overlap``. The dispatch helper rejects the
        ``(cyclic, overlap)`` combination; the budget treats it as a
        validation error in :func:`kernel_footprint` for symmetry.
    has_mask
        ``True`` when the caller supplied a mask. The replicated
        residency cost includes mask bytes either way (dispatch
        synthesizes an all-True mask when the user passes ``None``),
        but exposing the flag here keeps the input shape parallel to
        the dispatch path and leaves a hook for a future "no mask
        upload" optimization.
    deformed_mask_distinct
        ``True`` when ``deformed_mask`` is a different array than
        ``mask`` and so requires its own resident copy on each device.
        The dispatch path publishes the same SHM under both names when
        ``deformed_mask`` falls back to ``mask``; the budget mirrors
        that to avoid double-counting the 1.57 GB mask.

    Raises
    ------
    ValueError
        If ``volume_shape`` or ``window`` are not 3-tuples of positive
        integers, or if any window axis exceeds the corresponding
        volume axis.
    """

    volume_shape: tuple[int, int, int]
    window: tuple[int, int, int]
    mode: NCCMode
    normalization: NCCNormalization
    has_mask: bool
    deformed_mask_distinct: bool

    def __post_init__(self) -> None:
        """Validate the geometry inputs."""
        if len(self.volume_shape) != 3:
            raise ValueError(f"volume_shape must have length 3, got {len(self.volume_shape)}")
        if any(s <= 0 for s in self.volume_shape):
            raise ValueError(f"volume_shape entries must be positive, got {self.volume_shape}")
        if len(self.window) != 3:
            raise ValueError(f"window must have length 3, got {len(self.window)}")
        if any(w <= 0 for w in self.window):
            raise ValueError(f"window entries must be positive, got {self.window}")
        if any(w > s for w, s in zip(self.window, self.volume_shape, strict=True)):
            raise ValueError(
                f"window {self.window} exceeds volume_shape {self.volume_shape} on at "
                f"least one axis"
            )
        if not self.has_mask and self.deformed_mask_distinct:
            raise ValueError(
                "deformed_mask_distinct=True requires has_mask=True; a distinct "
                "deformed mask only makes sense alongside a reference mask"
            )


def kernel_footprint(
    window: tuple[int, int, int],
    mode: NCCMode,
    normalization: NCCNormalization,
) -> KernelFootprint:
    """Return the memory footprint of one FFT NCC kernel configuration.

    Parameters
    ----------
    window
        ``(wz, wy, wx)`` subvolume size in voxels.
    mode
        Kernel selector: ``NCCMode.CYCLIC`` or ``NCCMode.LINEAR``.
    normalization
        Denominator selector: ``NCCNormalization.GLOBAL`` or
        ``NCCNormalization.OVERLAP``. The combination
        ``(CYCLIC, OVERLAP)`` is rejected for parity with
        :func:`mamba_dvc.core.ncc.correlate`.

    Returns
    -------
    KernelFootprint
        ``per_poi_bytes`` and ``fixed_bytes`` for the requested config.
        Both fields are positive; multiply ``per_poi_bytes`` by
        ``batch_size`` to get the batch-scaling transient cost.

    Raises
    ------
    ValueError
        For an unsupported ``(mode, normalization)`` combination or a
        malformed ``window``.

    Notes
    -----
    The current numbers are analytic estimates (plan §4); a build-time
    calibration script is planned to replace them with measured values
    on the deployment hardware. Until then the conservatism flagged in
    the module docstring applies.
    """
    if len(window) != 3:
        raise ValueError(f"window must have length 3, got {len(window)}")
    if any(w <= 0 for w in window):
        raise ValueError(f"window entries must be positive, got {window}")

    key = (mode, normalization)
    if key not in _PER_POI_COEFFS:
        raise ValueError(
            f"unsupported (mode, normalization) combination: {mode.value!r}, "
            f"{normalization.value!r}; valid combinations are "
            f"{[(m.value, n.value) for m, n in _PER_POI_COEFFS]}"
        )

    voxels = prod(window)
    return KernelFootprint(
        per_poi_bytes=_PER_POI_COEFFS[key] * voxels,
        fixed_bytes=_FIXED_COEFFS[key] * voxels,
    )


def resident_bytes(inputs: BudgetInputs) -> int:
    """Bytes a worker pins on its device before any FFT work starts.

    Accounts for the four host arrays each worker uploads via
    ``cp.asarray`` in :func:`mamba_dvc.gpu.dispatch._run_helper_on_device`:
    reference (float32), deformed (float32), mask (bool), and either
    a distinct deformed mask (bool) or the same buffer aliased back.

    Parameters
    ----------
    inputs
        Resolved config; only ``volume_shape``, ``has_mask``, and
        ``deformed_mask_distinct`` are read.

    Returns
    -------
    int
        Total bytes pinned per device. At
        ``volume_shape=(960, 1280, 1280)`` with mask and a falling-back
        deformed mask this is ≈ 14.15 GB.
    """
    voxels = prod(inputs.volume_shape)
    # ref + def, both float32.
    total = 2 * 4 * voxels
    # Dispatch synthesizes an all-True bool mask even when the user
    # passes None, so a mask buffer is always resident. The
    # ``has_mask`` flag is kept on BudgetInputs for parity with the
    # dispatch path and as a hook for a future skip-the-upload
    # optimization; today it does not change the answer.
    total += voxels  # bool mask
    if inputs.deformed_mask_distinct:
        total += voxels  # distinct bool deformed mask
    return total


def estimate_max_batch(
    free_bytes: int,
    inputs: BudgetInputs,
    *,
    headroom_fraction: float = _HEADROOM_DEFAULT,
    min_batch: int = 1,
    max_batch: int = 1024,
    multiple_of: int = 8,
) -> int:
    """Largest ``batch_size`` that fits a per-device VRAM ceiling.

    Solves for ``B`` in
    ``free_bytes * (1 - headroom_fraction)
        >= resident_bytes(inputs) + footprint.fixed_bytes
           + footprint.per_poi_bytes * B``.

    Then floors the result to a multiple of ``multiple_of`` (cuFFT
    prefers small-prime batch dimensions) and clamps to
    ``[min_batch, max_batch]``.

    Parameters
    ----------
    free_bytes
        Free VRAM on the target device, in bytes. The caller is
        responsible for probing this *before* uploads via
        :func:`probe_free_vram`.
    inputs
        Resolved DVC config. Drives both the residency cost and the
        per-POI footprint.
    headroom_fraction
        Fraction of ``free_bytes`` reserved for the cuFFT plan cache
        and allocator fragmentation. Plan §7 sets the default at 0.15.
        Must be in ``[0, 1)``.
    min_batch
        Hard floor below which the function raises rather than ships a
        batch that is too small to be useful. Plan §5: clamping a
        too-tight config to ``min_batch`` would OOM mid-shard and burn
        the spawn cost; raising lets the caller adjust before any work
        starts.
    max_batch
        Upper clamp. Defaults to 1024, well past anything cuFFT plans
        cleanly at production window sizes.
    multiple_of
        Round-down granularity. Set to 1 to disable the cuFFT-friendly
        rounding (e.g. for tests that need exact integer results).

    Returns
    -------
    int
        Recommended batch size, in ``[min_batch, max_batch]`` and a
        multiple of ``multiple_of``.

    Raises
    ------
    ValueError
        For out-of-range ``headroom_fraction``, non-positive
        ``min_batch`` / ``max_batch`` / ``multiple_of``, or
        ``min_batch > max_batch``.
    RuntimeError
        If the computed batch falls below ``min_batch``. The message
        includes ``free_bytes``, the residency cost, the per-POI cost,
        and the current ``headroom_fraction`` so the caller can see
        which knob to relax.
    """
    if not 0.0 <= headroom_fraction < 1.0:
        raise ValueError(f"headroom_fraction must be in [0, 1), got {headroom_fraction}")
    if min_batch <= 0:
        raise ValueError(f"min_batch must be positive, got {min_batch}")
    if max_batch <= 0:
        raise ValueError(f"max_batch must be positive, got {max_batch}")
    if min_batch > max_batch:
        raise ValueError(f"min_batch ({min_batch}) must not exceed max_batch ({max_batch})")
    if multiple_of <= 0:
        raise ValueError(f"multiple_of must be positive, got {multiple_of}")
    if free_bytes < 0:
        raise ValueError(f"free_bytes must be non-negative, got {free_bytes}")

    footprint = kernel_footprint(inputs.window, inputs.mode, inputs.normalization)
    resident = resident_bytes(inputs)

    # Reserve the headroom on the *full* free pool, not on the
    # post-residency remainder. Plan cache and fragmentation grow with
    # what we have, not with what we spend on FFT working set.
    available = int(free_bytes * (1.0 - headroom_fraction))
    spendable = available - resident - footprint.fixed_bytes

    if spendable <= 0 or footprint.per_poi_bytes == 0:
        # ``per_poi_bytes == 0`` is unreachable for any defined config
        # (every entry in _PER_POI_COEFFS is positive), but the guard
        # keeps a future zero-cost stub from dividing by zero.
        raw = 0 if spendable <= 0 else max_batch
    else:
        raw = spendable // footprint.per_poi_bytes

    # Round down to the cuFFT-friendly multiple, then clamp.
    floored = (raw // multiple_of) * multiple_of
    candidate = min(int(floored), max_batch)

    if candidate < min_batch:
        raise RuntimeError(
            f"VRAM budget too tight for min_batch={min_batch}: "
            f"free={free_bytes} B, resident={resident} B, "
            f"per_poi={footprint.per_poi_bytes} B, fixed={footprint.fixed_bytes} B, "
            f"headroom_fraction={headroom_fraction}; "
            f"would-be batch={candidate}. Reduce window, drop a device, or "
            f"lower headroom_fraction."
        )
    return candidate


def probe_free_vram(device_id: int) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for one CUDA device.

    Thin wrapper around ``cp.cuda.runtime.memGetInfo`` that pins the
    current device for the duration of the probe. Lives in this module
    so that everything else in :mod:`mamba_dvc.gpu.budget` stays pure
    Python -- the seam matches the ``_cp is None`` guard pattern used
    in :mod:`mamba_dvc.core.ncc` and :mod:`mamba_dvc.gpu.dispatch`.

    Parameters
    ----------
    device_id
        Non-negative CUDA device index.

    Returns
    -------
    tuple of int
        ``(free_bytes, total_bytes)``. Both are positive.

    Raises
    ------
    RuntimeError
        If CuPy is not importable on this host.
    ValueError
        If ``device_id`` is negative.
    """
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    if _cp is None:
        raise RuntimeError(
            "probe_free_vram requires CuPy; install it on this host or skip the "
            "auto-batch path and pass batch_size explicitly"
        )
    with _cp.cuda.Device(device_id):
        free, total = _cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def recommend_batch_size(
    inputs: BudgetInputs,
    device_ids: Sequence[int],
    *,
    headroom_fraction: float = _HEADROOM_DEFAULT,
    floor: int = 8,
    ceiling: int | None = None,
) -> int:
    """Probe each device, return one cross-device ``batch_size``.

    Takes the **minimum** of free VRAM across ``device_ids`` so that a
    heterogeneous fleet degrades to the slowest device. The dispatch
    helper consumes a single ``batch_size``, and a per-device dict
    return is the deferred v2 hook (plan §8).

    Parameters
    ----------
    inputs
        Resolved DVC config; passed through to
        :func:`estimate_max_batch`.
    device_ids
        Non-empty sequence of CUDA device indices. Must match the
        ``device_ids`` the caller will pass to
        :func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`.
    headroom_fraction
        Reserve fraction; see :func:`estimate_max_batch`.
    floor
        Hard floor; raises if the computed batch would be lower. Maps
        to ``min_batch`` on :func:`estimate_max_batch` and follows the
        same raise-vs-clamp semantics from plan §5.
    ceiling
        Optional upper clamp set by the caller. ``None`` (default)
        disables the user-side cap and falls back to the
        :func:`estimate_max_batch` ``max_batch`` default.

    Returns
    -------
    int
        Recommended ``batch_size`` for ``correlate_multi_gpu``.

    Raises
    ------
    ValueError
        If ``device_ids`` is empty.
    RuntimeError
        If CuPy is unavailable, or if the per-device budget is too
        tight for ``floor`` (re-raised from
        :func:`estimate_max_batch`).
    """
    if len(device_ids) == 0:
        raise ValueError("device_ids must be non-empty")

    free_per_device = [probe_free_vram(d)[0] for d in device_ids]
    min_free = min(free_per_device)

    kwargs: dict[str, int | float] = {
        "headroom_fraction": headroom_fraction,
        "min_batch": floor,
    }
    if ceiling is not None:
        kwargs["max_batch"] = ceiling
    return estimate_max_batch(min_free, inputs, **kwargs)  # pyright: ignore[reportArgumentType]
