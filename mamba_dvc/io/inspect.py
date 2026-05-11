"""Streaming summary statistics over a :class:`DvcDataset`.

Walks the reference, every mask, and every (image, flow) pair, emits
per-array statistics, and returns a frozen
:class:`DatasetStats` tree. Designed for triage in Jupyter before
kicking off a ``correlate()`` run, and as a quick post-load sanity
check on top of :mod:`mamba_dvc.io.verify`.

The inspector never holds a full volume in RAM: each array is
streamed as Z-slabs whose voxel count fits a configurable byte budget
(default 256 MB). Median is approximated via a two-pass histogram
(bounded memory) by default; ``median="exact"`` materializes the
volume for ``numpy.median`` and ``median="skip"`` records ``nan``.

See ``docs/plans/dataset-inspect.md`` for the full design contract.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import zarr

from mamba_dvc.io.dataset import DvcDataset
from mamba_dvc.io.field import FieldAxisOrder, FieldConvention, GroundTruthField
from mamba_dvc.io.volume import load_volume
from mamba_dvc.validate.synthetic import warp

__all__ = [
    "ConventionCheck",
    "DatasetStats",
    "DeformationStats",
    "DisplacementStats",
    "FlowLayoutError",
    "IntensityStats",
    "MaskStats",
    "ProgressEvent",
    "ProgressReporter",
    "inspect_dataset",
    "null_reporter",
    "print_reporter",
    "tqdm_reporter",
    "verify_flow_convention",
]


MedianMethod = Literal["exact", "histogram", "skipped"]
MedianMode = Literal["exact", "histogram", "skip"]
ProgressStage = Literal[
    "reference",
    "mask",
    "deformation_image",
    "deformation_flow",
    "done",
]


# Byte budget per streamed slab. Picked so a slab of a (960, 1280, 1280)
# float32 volume is ~52 MB (one Z-row) up to ~256 MB; comfortable for
# notebook kernels and small enough that NumPy ops stay cache-friendly.
_DEFAULT_SLAB_BUDGET_BYTES = 256 * 1024 * 1024


# ----------------------------------------------------------------- errors


class FlowLayoutError(ValueError):
    """A flow array's shape disagrees with its expected on-disk layout.

    Raised by :func:`inspect_dataset` before any voxels are read,
    when the size-3 vector axis is in the wrong position or the
    spatial axes do not match the dataset's reference shape.
    """


# ----------------------------------------------------------------- progress


@dataclass(frozen=True)
class ProgressEvent:
    """One inspector step boundary.

    Emitted exactly once per item (reference, each mask, each
    deformation image, each deformation flow) plus a final
    ``"done"`` event.

    Parameters
    ----------
    stage
        Which kind of array just finished (``"done"`` for the
        terminal event).
    name
        The item's identifying name -- ``"reference"`` for the
        reference, mask name for masks, deformation name for
        deformations, empty string for the ``"done"`` event.
    item_index
        0-based index into the resolved walk plan; equals
        ``item_total`` on the ``"done"`` event.
    item_total
        Total number of items in the walk (excluding the terminal
        ``"done"`` event).
    elapsed_seconds
        Wall-clock seconds since :func:`inspect_dataset` started.
    """

    stage: ProgressStage
    name: str
    item_index: int
    item_total: int
    elapsed_seconds: float


class ProgressReporter(Protocol):
    """Callable that consumes :class:`ProgressEvent` updates."""

    def __call__(self, event: ProgressEvent) -> None:
        """Handle one progress event."""
        ...


def null_reporter(event: ProgressEvent) -> None:
    """Drop every event. Default when ``progress=None``."""
    _ = event


def print_reporter(event: ProgressEvent) -> None:
    """Print one line per event. Useful for headless / CI runs."""
    if event.stage == "done":
        print(f"[inspect] done in {event.elapsed_seconds:.2f}s ({event.item_total} items)")
        return
    print(
        f"[inspect] {event.item_index + 1}/{event.item_total} "
        f"{event.stage}={event.name!r} t={event.elapsed_seconds:.2f}s"
    )


def tqdm_reporter() -> ProgressReporter:
    """Return a closure that drives a ``tqdm.auto`` progress bar.

    ``tqdm.auto`` selects the notebook widget under Jupyter and the
    plain bar otherwise. ``tqdm`` is imported lazily here so the
    inspector module has no hard dependency on it.

    Returns
    -------
    ProgressReporter
        Callable ready to be passed as ``progress=`` to
        :func:`inspect_dataset`.
    """
    from tqdm.auto import tqdm

    state: dict[str, Any] = {"bar": None, "last_index": -1}

    def _report(event: ProgressEvent) -> None:
        bar = state["bar"]
        if bar is None and event.stage != "done":
            bar = tqdm(total=event.item_total, unit="item")
            state["bar"] = bar
        if event.stage == "done":
            if bar is not None:
                bar.close()
                state["bar"] = None
            return
        if bar is not None:
            bar.set_description(f"{event.stage}:{event.name}")
            steps = event.item_index - state["last_index"]
            if steps > 0:
                bar.update(steps)
                state["last_index"] = event.item_index

    return _report


# ----------------------------------------------------------------- result


@dataclass(frozen=True)
class IntensityStats:
    """Streaming summary stats for a single intensity volume.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape of the source array.
    dtype
        String name of the source dtype (``"float32"``,
        ``"uint16"``, ...). Accumulators run in float64 regardless;
        this records what was on disk.
    count
        Total voxel count. Equal to ``prod(shape)``.
    mean
        Voxel mean.
    min, max
        Voxel min / max. Exact (single pass).
    median
        Median voxel value. Exact when ``median_method == "exact"``,
        histogram estimate when ``"histogram"``, ``nan`` when
        ``"skipped"``.
    median_method
        How :attr:`median` was computed.
    """

    shape: tuple[int, int, int]
    dtype: str
    count: int
    mean: float
    min: float
    max: float
    median: float
    median_method: MedianMethod


@dataclass(frozen=True)
class MaskStats:
    """Coverage stats for a single boolean mask.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape.
    foreground_count
        Number of ``True`` voxels.
    background_count
        Number of ``False`` voxels.
    foreground_fraction
        ``foreground_count / (foreground_count + background_count)``.
        ``nan`` when the mask has zero total voxels (cannot occur on
        a verified store but guarded for completeness).
    """

    shape: tuple[int, int, int]
    foreground_count: int
    background_count: int
    foreground_fraction: float


@dataclass(frozen=True)
class DisplacementStats:
    """Streaming summary stats for one ``(z, y, x, 3)`` flow array.

    Parameters
    ----------
    shape
        Spatial ``(z, y, x)`` voxel shape (vector axis suppressed).
    axis_order
        On-disk layout used while iterating: ``"3_zyx"`` reads the
        leading vector axis first, ``"zyx_3"`` reads it last. The
        inspector validates this matches the array shape before
        reading any voxels.
    mean_magnitude
        Mean of ``||u||`` over all voxels.
    max_magnitude
        Max of ``||u||`` over all voxels.
    median_magnitude
        Median of ``||u||`` (subject to the same exact / histogram /
        skipped contract as :class:`IntensityStats`).
    median_method
        How :attr:`median_magnitude` was computed.
    per_axis_mean
        ``(mean(dz), mean(dy), mean(dx))``. Reported using on-disk
        sign -- the inspector does not apply pull-back/push-forward
        flips, since :func:`inspect_dataset` reports what is on
        disk, not what ``correlate()`` would return.
    per_axis_max_abs
        ``(max|dz|, max|dy|, max|dx|)``.
    """

    shape: tuple[int, int, int]
    axis_order: FieldAxisOrder
    mean_magnitude: float
    max_magnitude: float
    median_magnitude: float
    median_method: MedianMethod
    per_axis_mean: tuple[float, float, float]
    per_axis_max_abs: tuple[float, float, float]


@dataclass(frozen=True)
class DeformationStats:
    """Per-entry stats: image always, flow only for synthetic entries.

    Parameters
    ----------
    name
        Deformation key inside ``DvcDataset.deformations``.
    kind
        ``"real"`` for entries under ``iterations/`` (no
        ground-truth flow), ``"synthetic"`` for entries under
        ``synthetic/``.
    image
        Stats for the deformed-image volume.
    flow
        Stats for the ground-truth flow array, or ``None`` when
        ``kind == "real"``.
    """

    name: str
    kind: Literal["real", "synthetic"]
    image: IntensityStats
    flow: DisplacementStats | None


@dataclass(frozen=True)
class ConventionCheck:
    """Empirical convention check for one synthetic entry.

    Warps the reference under the stored flow using each sign
    convention and reports which residual against the stored deformed
    image is smaller. The smaller residual identifies the convention
    the store was generated with; the ratio between residuals is a
    confidence indicator -- close to ``1.0`` means the displacement
    is too small to discriminate (e.g. a near-zero field).

    Parameters
    ----------
    name
        Synthetic entry name.
    declared
        Convention recorded in the manifest (falling back to the
        profile default). What the producer claims.
    empirical
        Convention with the lower residual. What the data shows.
    agrees
        ``declared == empirical``.
    residual_pull_back
        Mean-squared error between the stored deformed image and
        ``warp(reference, flow, convention="pull_back")``.
    residual_push_forward
        MSE for the ``"push_forward"`` interpretation.
    ratio
        ``max(residuals) / min(residuals)``. ``1.0`` means the test
        is inconclusive (typically a near-zero field); large values
        mean the smaller residual is dominantly smaller.
    dry_shape
        Centered ``(z, y, x)`` subblock the test was run on, after
        clamping per-axis to the dataset's volume shape.
    """

    name: str
    declared: FieldConvention
    empirical: FieldConvention
    agrees: bool
    residual_pull_back: float
    residual_push_forward: float
    ratio: float
    dry_shape: tuple[int, int, int]


@dataclass(frozen=True)
class DatasetStats:
    """Top-level inspector result over a :class:`DvcDataset`.

    Parameters
    ----------
    reference
        Stats for the reference volume.
    masks
        ``{mask_name: MaskStats}`` for every inspected mask.
    deformations
        ``{name: DeformationStats}`` for every inspected deformation.
    skipped
        ``{name: reason}`` forwarded from
        :attr:`DvcDataset.broken_entries`.
    elapsed_seconds
        Wall-clock seconds the inspector took, end to end.
    median_method
        The median strategy the call was configured with -- recorded
        on the result so downstream consumers know whether to trust
        :attr:`IntensityStats.median` for sub-bin precision.
    """

    reference: IntensityStats
    masks: Mapping[str, MaskStats]
    deformations: Mapping[str, DeformationStats]
    skipped: Mapping[str, str]
    elapsed_seconds: float
    median_method: MedianMethod


# ----------------------------------------------------------------- public


def inspect_dataset(
    dataset: DvcDataset,
    *,
    include: Sequence[str] | None = None,
    masks: Sequence[str] | None = None,
    median: MedianMode = "histogram",
    histogram_bins: int = 16384,
    slab_budget_bytes: int = _DEFAULT_SLAB_BUDGET_BYTES,
    progress: ProgressReporter | None = None,
) -> DatasetStats:
    """Stream summary statistics over every array a ``DvcDataset`` exposes.

    Parameters
    ----------
    dataset
        An opened :class:`DvcDataset`. Materialization happens
        chunk-by-chunk inside this call; the dataset is unchanged
        on return.
    include
        Optional subset of deformation names to inspect. ``None``
        (default) walks every healthy entry. Names not present in
        ``dataset.deformations`` raise :class:`KeyError`.
    masks
        Optional subset of mask names. ``None`` walks every mask.
    median
        ``"histogram"`` (default) computes a two-pass histogram-based
        median estimate; ``"exact"`` materializes each volume in
        full and calls :func:`numpy.median`; ``"skip"`` records
        ``nan`` for all median fields.
    histogram_bins
        Bin count for the histogram path. Default 16384 ≈ 64 KB of
        bin state, ~0.006% relative resolution across the volume's
        intensity range.
    slab_budget_bytes
        Approximate working-memory ceiling per streamed slab. Used
        to derive the slab size along Z; smaller is safer on RAM,
        larger amortizes Python overhead.
    progress
        Optional :class:`ProgressReporter`. ``None`` (default) is
        silent; pass :func:`tqdm_reporter` for a Jupyter-friendly
        progress bar.

    Returns
    -------
    DatasetStats
        Frozen tree of per-array statistics.

    Raises
    ------
    KeyError
        If ``include`` or ``masks`` names an entry the dataset does
        not expose.
    ValueError
        If ``histogram_bins < 2`` or ``slab_budget_bytes <= 0``.
    FlowLayoutError
        If a synthetic deformation's flow array shape disagrees
        with the dataset's resolved ``axis_order``.
    """
    if histogram_bins < 2:
        raise ValueError(f"histogram_bins must be >= 2, got {histogram_bins}")
    if slab_budget_bytes <= 0:
        raise ValueError(f"slab_budget_bytes must be positive, got {slab_budget_bytes}")

    reporter: ProgressReporter = progress if progress is not None else null_reporter
    method: MedianMethod = "skipped" if median == "skip" else median

    mask_names = _resolve_mask_names(dataset, masks)
    deform_names = _resolve_deformation_names(dataset, include)
    flow_axis_order = _resolve_flow_axis_order(dataset)

    # Validate every flow up-front, before reading any voxels: a malformed
    # store should fail fast, not after a 30-minute reference scan.
    for name in deform_names:
        entry = dataset.deformations[name]
        if entry.flow is not None:
            _validate_flow_axis_layout(
                entry.flow,
                name=name,
                axis_order=flow_axis_order,
                spatial_shape=dataset.volume_shape,
            )

    item_total = (
        1
        + len(mask_names)
        + sum(2 if dataset.deformations[name].flow is not None else 1 for name in deform_names)
    )

    started = time.perf_counter()
    item_index = 0

    reference = _intensity_stats(
        dataset.reference,
        median=median,
        histogram_bins=histogram_bins,
        slab_budget_bytes=slab_budget_bytes,
    )
    reporter(
        ProgressEvent(
            stage="reference",
            name="reference",
            item_index=item_index,
            item_total=item_total,
            elapsed_seconds=time.perf_counter() - started,
        )
    )
    item_index += 1

    mask_results: dict[str, MaskStats] = {}
    for mask_name in mask_names:
        mask_results[mask_name] = _mask_stats(
            dataset.masks[mask_name],
            slab_budget_bytes=slab_budget_bytes,
        )
        reporter(
            ProgressEvent(
                stage="mask",
                name=mask_name,
                item_index=item_index,
                item_total=item_total,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        item_index += 1

    deformation_results: dict[str, DeformationStats] = {}
    for name in deform_names:
        entry = dataset.deformations[name]
        image_stats = _intensity_stats(
            entry.image,
            median=median,
            histogram_bins=histogram_bins,
            slab_budget_bytes=slab_budget_bytes,
        )
        reporter(
            ProgressEvent(
                stage="deformation_image",
                name=name,
                item_index=item_index,
                item_total=item_total,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        item_index += 1

        flow_stats: DisplacementStats | None = None
        if entry.flow is not None:
            flow_stats = _displacement_stats(
                entry.flow,
                axis_order=flow_axis_order,
                spatial_shape=dataset.volume_shape,
                median=median,
                histogram_bins=histogram_bins,
                slab_budget_bytes=slab_budget_bytes,
            )
            reporter(
                ProgressEvent(
                    stage="deformation_flow",
                    name=name,
                    item_index=item_index,
                    item_total=item_total,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            item_index += 1

        deformation_results[name] = DeformationStats(
            name=name,
            kind=entry.kind,
            image=image_stats,
            flow=flow_stats,
        )

    skipped: dict[str, str] = {name: be.reason for name, be in dataset.broken_entries.items()}

    elapsed = time.perf_counter() - started
    reporter(
        ProgressEvent(
            stage="done",
            name="",
            item_index=item_total,
            item_total=item_total,
            elapsed_seconds=elapsed,
        )
    )

    return DatasetStats(
        reference=reference,
        masks=mask_results,
        deformations=deformation_results,
        skipped=skipped,
        elapsed_seconds=elapsed,
        median_method=method,
    )


def verify_flow_convention(
    dataset: DvcDataset,
    *,
    entries: Sequence[str] | None = None,
    dry_shape: tuple[int, int, int] = (32, 64, 64),
    order: int = 3,
) -> dict[str, ConventionCheck]:
    """Empirically detect the warp convention behind each synthetic entry.

    Warps the reference under the stored flow with both
    ``"pull_back"`` and ``"push_forward"`` conventions and picks
    whichever yields the lower mean-squared residual against the
    stored deformed image. The empirical winner is cross-checked
    against the convention declared in the manifest / profile.

    Runs on a small centered subblock by default so the test stays
    cheap; expand ``dry_shape`` if the confidence :attr:`ConventionCheck.ratio`
    sits near ``1.0`` and you suspect a near-zero displacement.

    Parameters
    ----------
    dataset
        Opened :class:`DvcDataset`.
    entries
        Synthetic entry names to check. ``None`` (default) walks
        every name from :meth:`DvcDataset.list_synthetic`. Unknown
        names raise :class:`KeyError`; real (non-synthetic) names
        raise :class:`ValueError`.
    dry_shape
        ``(z, y, x)`` centered subblock used for the residual test,
        in voxels. Each component is clamped to the dataset's volume
        shape. Smaller is faster; larger gives a stronger signal
        when the field is small.
    order
        Spline order forwarded to
        :func:`mamba_dvc.validate.synthetic.warp` and
        :class:`mamba_dvc.io.field.GroundTruthField`. Default ``3``
        matches the synthesis path.

    Returns
    -------
    dict[str, ConventionCheck]
        One :class:`ConventionCheck` per inspected entry.

    Raises
    ------
    KeyError
        If ``entries`` names a deformation absent from the dataset.
    ValueError
        If ``entries`` includes a real entry, or ``dry_shape`` is
        not a 3-tuple of positive ints.
    FlowLayoutError
        If a flow array's shape disagrees with the dataset's
        resolved ``axis_order``.
    """
    if len(dry_shape) != 3 or any(d <= 0 for d in dry_shape):
        raise ValueError(f"dry_shape must be a 3-tuple of positive ints, got {dry_shape}")

    synthetic_set = set(dataset.list_synthetic())
    if entries is None:
        targets = sorted(synthetic_set)
    else:
        targets = list(entries)
        unknown = [n for n in targets if n not in dataset.deformations]
        if unknown:
            raise KeyError(f"unknown deformation name(s): {unknown!r}")
        non_synthetic = [n for n in targets if n not in synthetic_set]
        if non_synthetic:
            raise ValueError(
                f"verify_flow_convention is synthetic-only; "
                f"got non-synthetic entries: {non_synthetic!r}"
            )

    axis_order = _resolve_flow_axis_order(dataset)
    declared = _resolve_flow_convention(dataset)
    effective_shape = (
        min(dry_shape[0], dataset.volume_shape[0]),
        min(dry_shape[1], dataset.volume_shape[1]),
        min(dry_shape[2], dataset.volume_shape[2]),
    )

    for name in targets:
        entry = dataset.deformations[name]
        assert entry.flow is not None
        _validate_flow_axis_layout(
            entry.flow,
            name=name,
            axis_order=axis_order,
            spatial_shape=dataset.volume_shape,
        )

    reference = load_volume(dataset.reference, dry_shape=effective_shape, as_float32=True)

    out: dict[str, ConventionCheck] = {}
    for name in targets:
        entry = dataset.deformations[name]
        assert entry.flow is not None
        deformed = load_volume(entry.image, dry_shape=effective_shape, as_float32=True)

        # convention="pull_back" is the identity branch of GroundTruthField's
        # sign normalization, so the callable returns on-disk values verbatim
        # regardless of what the producer actually used.
        raw_field = GroundTruthField.from_zarr(
            entry.flow,
            axis_order=axis_order,
            convention="pull_back",
            dry_shape=effective_shape,
            interpolation=order,
        )

        pulled = warp(reference, raw_field, order=order, convention="pull_back")
        pushed = warp(reference, raw_field, order=order, convention="push_forward")

        diff_pull = pulled.astype(np.float64, copy=False) - deformed.astype(
            np.float64, copy=False
        )
        diff_push = pushed.astype(np.float64, copy=False) - deformed.astype(
            np.float64, copy=False
        )
        r_pull = float(np.mean(diff_pull**2))
        r_push = float(np.mean(diff_push**2))

        if r_pull <= r_push:
            empirical: FieldConvention = "pull_back"
            smaller, larger = r_pull, r_push
        else:
            empirical = "push_forward"
            smaller, larger = r_push, r_pull
        if smaller == 0.0 and larger == 0.0:
            ratio = 1.0
        elif smaller == 0.0:
            ratio = float("inf")
        else:
            ratio = larger / smaller

        out[name] = ConventionCheck(
            name=name,
            declared=declared,
            empirical=empirical,
            agrees=(declared == empirical),
            residual_pull_back=r_pull,
            residual_push_forward=r_push,
            ratio=ratio,
            dry_shape=effective_shape,
        )

    return out


# ----------------------------------------------------------------- helpers


def _resolve_flow_axis_order(dataset: DvcDataset) -> FieldAxisOrder:
    """Replicate ``DvcDataset._flow_axis_order`` without piercing privacy.

    Mirrors the manifest-then-profile fallback the dataset uses
    internally; kept here so the inspector module is self-contained
    and the dataset's protected helper stays protected.
    """
    manifest = dataset.manifest
    if manifest is not None and manifest.synthetic.flow.axis_order is not None:
        return manifest.synthetic.flow.axis_order
    return dataset.profile.flow_defaults.axis_order


def _resolve_flow_convention(dataset: DvcDataset) -> FieldConvention:
    """Replicate ``DvcDataset._flow_convention`` without piercing privacy."""
    manifest = dataset.manifest
    if manifest is not None and manifest.synthetic.flow.convention is not None:
        return manifest.synthetic.flow.convention
    return dataset.profile.flow_defaults.convention


def _resolve_mask_names(dataset: DvcDataset, masks: Sequence[str] | None) -> list[str]:
    """Return the resolved, sorted mask-name walk list."""
    available = sorted(dataset.masks.keys())
    if masks is None:
        return available
    requested = list(masks)
    missing = [name for name in requested if name not in dataset.masks]
    if missing:
        raise KeyError(f"unknown mask name(s): {missing!r}")
    return sorted(set(requested))


def _resolve_deformation_names(
    dataset: DvcDataset, include: Sequence[str] | None
) -> list[str]:
    """Return the resolved, sorted deformation-name walk list.

    ``broken_entries`` are never inspected -- they're surfaced via
    :attr:`DatasetStats.skipped` instead, and asking the inspector to
    walk one is a usage error.
    """
    if include is None:
        return sorted(dataset.deformations.keys())
    requested = list(include)
    missing = [name for name in requested if name not in dataset.deformations]
    if missing:
        broken_hits = [name for name in missing if name in dataset.broken_entries]
        if broken_hits:
            raise KeyError(
                f"deformation(s) {broken_hits!r} are broken and cannot be "
                f"inspected; see DvcDataset.broken_entries"
            )
        raise KeyError(f"unknown deformation name(s): {missing!r}")
    return sorted(set(requested))


def _validate_flow_axis_layout(
    flow: zarr.Array[Any],
    *,
    name: str,
    axis_order: FieldAxisOrder,
    spatial_shape: tuple[int, int, int],
) -> None:
    """Verify a flow array matches its declared on-disk layout.

    Parameters
    ----------
    flow
        Lazy zarr array bound by ``DvcDataset``.
    name
        Deformation name, used in the error message.
    axis_order
        Resolved axis order from manifest + profile defaults.
        ``"3_zyx"`` ⇒ shape must be ``(3, Z, Y, X)``;
        ``"zyx_3"`` ⇒ shape must be ``(Z, Y, X, 3)``.
    spatial_shape
        ``(Z, Y, X)`` shape of the dataset's reference volume.

    Raises
    ------
    FlowLayoutError
        If ``flow`` is not 4D, the size-3 axis is in the wrong
        position, or the spatial axes do not equal ``spatial_shape``.
    """
    if flow.ndim != 4:
        raise FlowLayoutError(f"deformation {name!r}: flow must be 4D, got ndim={flow.ndim}")
    full = tuple(int(s) for s in flow.shape)
    if axis_order == "3_zyx":
        expected = (3, *spatial_shape)
        vector_axis = 0
    else:
        expected = (*spatial_shape, 3)
        vector_axis = 3
    if full == expected:
        return

    size3_positions = [i for i, s in enumerate(full) if s == 3]
    if len(size3_positions) == 0:
        hint = "no axis has size 3"
    elif size3_positions == [vector_axis]:
        hint = (
            f"vector axis is at the expected position {vector_axis} but "
            f"the spatial axes {tuple(s for i, s in enumerate(full) if i != vector_axis)} "
            f"do not match the reference shape {spatial_shape}"
        )
    elif len(size3_positions) == 1:
        actual = size3_positions[0]
        position_label = (
            "leading"
            if actual == 0
            else "trailing"
            if actual == flow.ndim - 1
            else f"position {actual}"
        )
        hint = (
            f"size-3 axis is at the {position_label} position {actual}; "
            f"axis_order={axis_order!r} expects it at position {vector_axis}"
        )
    else:
        hint = f"multiple axes have size 3 (positions {size3_positions}); layout is ambiguous"
    raise FlowLayoutError(
        f"deformation {name!r}: flow shape {full} does not match "
        f"expected {expected} for axis_order={axis_order!r} -- {hint}"
    )


def _slab_size(
    *,
    plane_voxels: int,
    bytes_per_voxel: int,
    z_total: int,
    budget_bytes: int,
) -> int:
    """Largest Z-slab size that fits ``budget_bytes``.

    Always returns at least 1; capped at ``z_total`` so a single slab
    can read the entire array when it fits in budget.
    """
    plane_bytes = plane_voxels * bytes_per_voxel
    if plane_bytes <= 0:
        return max(1, z_total)
    slab = max(1, budget_bytes // plane_bytes)
    return min(slab, z_total)


def _intensity_stats(
    array: zarr.Array[Any],
    *,
    median: MedianMode,
    histogram_bins: int,
    slab_budget_bytes: int,
) -> IntensityStats:
    """Stream-compute :class:`IntensityStats` for one 3D array."""
    if array.ndim != 3:
        raise ValueError(f"expected 3D intensity array, got ndim={array.ndim}")
    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    dtype = np.dtype(array.dtype)
    bytes_per_voxel = dtype.itemsize or 1
    plane_voxels = shape[1] * shape[2]
    slab_z = _slab_size(
        plane_voxels=plane_voxels,
        bytes_per_voxel=bytes_per_voxel,
        z_total=shape[0],
        budget_bytes=slab_budget_bytes,
    )

    if median == "exact":
        full = np.asarray(array[:])
        full_count = int(full.size)
        full_min = float(full.min())
        full_max = float(full.max())
        full_mean = float(full.astype(np.float64, copy=False).mean())
        full_median = float(np.median(full))
        return IntensityStats(
            shape=shape,
            dtype=dtype.name,
            count=full_count,
            mean=full_mean,
            min=full_min,
            max=full_max,
            median=full_median,
            median_method="exact",
        )

    total = 0
    summed = 0.0
    val_min = np.inf
    val_max = -np.inf
    for slab in _iter_z_slabs(array, slab_z=slab_z):
        slab64 = slab.astype(np.float64, copy=False)
        total += int(slab.size)
        summed += float(slab64.sum())
        slab_min = float(slab64.min())
        slab_max = float(slab64.max())
        if slab_min < val_min:
            val_min = slab_min
        if slab_max > val_max:
            val_max = slab_max

    mean = summed / total if total > 0 else float("nan")
    if median == "skip":
        median_value = float("nan")
        method: MedianMethod = "skipped"
    else:
        method = "histogram"
        median_value = _histogram_median(
            array,
            slab_z=slab_z,
            value_min=val_min,
            value_max=val_max,
            histogram_bins=histogram_bins,
        )

    return IntensityStats(
        shape=shape,
        dtype=dtype.name,
        count=total,
        mean=mean,
        min=val_min if np.isfinite(val_min) else float("nan"),
        max=val_max if np.isfinite(val_max) else float("nan"),
        median=median_value,
        median_method=method,
    )


def _mask_stats(
    array: zarr.Array[Any],
    *,
    slab_budget_bytes: int,
) -> MaskStats:
    """Stream-compute :class:`MaskStats` for one boolean 3D array."""
    if array.ndim != 3:
        raise ValueError(f"expected 3D mask array, got ndim={array.ndim}")
    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    plane_voxels = shape[1] * shape[2]
    slab_z = _slab_size(
        plane_voxels=plane_voxels,
        bytes_per_voxel=1,
        z_total=shape[0],
        budget_bytes=slab_budget_bytes,
    )

    foreground = 0
    total = 0
    for slab in _iter_z_slabs(array, slab_z=slab_z):
        slab_bool = slab.astype(np.bool_, copy=False)
        foreground += int(np.count_nonzero(slab_bool))
        total += int(slab_bool.size)

    background = total - foreground
    fraction = foreground / total if total > 0 else float("nan")
    return MaskStats(
        shape=shape,
        foreground_count=foreground,
        background_count=background,
        foreground_fraction=fraction,
    )


def _displacement_stats(
    flow: zarr.Array[Any],
    *,
    axis_order: FieldAxisOrder,
    spatial_shape: tuple[int, int, int],
    median: MedianMode,
    histogram_bins: int,
    slab_budget_bytes: int,
) -> DisplacementStats:
    """Stream-compute :class:`DisplacementStats` for one flow array.

    Iterates Z-slabs of the underlying zarr array and normalizes each
    slab to ``(slab_z, Y, X, 3)`` before computing magnitudes and
    per-axis stats. Magnitudes are accumulated in float32; per-axis
    sums in float64.
    """
    plane_voxels = spatial_shape[1] * spatial_shape[2]
    bytes_per_voxel = 12  # float32, 3 components per voxel
    slab_z = _slab_size(
        plane_voxels=plane_voxels,
        bytes_per_voxel=bytes_per_voxel,
        z_total=spatial_shape[0],
        budget_bytes=slab_budget_bytes,
    )

    total_voxels = 0
    mag_sum = 0.0
    mag_max = -np.inf
    per_axis_sum = np.zeros(3, dtype=np.float64)
    per_axis_max_abs = np.zeros(3, dtype=np.float64)

    # First pass: per-axis + magnitude min/max/mean accumulators.
    mag_min = np.inf
    for slab in _iter_flow_z_slabs(
        flow, axis_order=axis_order, spatial_shape=spatial_shape, slab_z=slab_z
    ):
        slab32 = slab.astype(np.float32, copy=False)
        # mag is float32; precision is fine at our 0.1-voxel target.
        mag = np.linalg.norm(slab32, axis=-1)
        total_voxels += int(mag.size)
        mag_sum += float(mag.astype(np.float64, copy=False).sum())
        smin = float(mag.min())
        smax = float(mag.max())
        if smin < mag_min:
            mag_min = smin
        if smax > mag_max:
            mag_max = smax
        per_axis_sum += slab32.sum(axis=(0, 1, 2), dtype=np.float64)
        slab_axis_max = np.abs(slab32).max(axis=(0, 1, 2)).astype(np.float64)
        per_axis_max_abs = np.maximum(per_axis_max_abs, slab_axis_max)

    mean_magnitude = mag_sum / total_voxels if total_voxels > 0 else float("nan")
    per_axis_mean_arr = (
        per_axis_sum / total_voxels
        if total_voxels > 0
        else np.full(3, np.nan, dtype=np.float64)
    )

    if median == "skip":
        median_magnitude = float("nan")
        method: MedianMethod = "skipped"
    elif median == "exact":
        # Exact path: materialize once, normalize layout, compute magnitudes.
        full = _normalize_flow_to_zyx3(np.asarray(flow[:]), axis_order=axis_order)
        full_mag = np.linalg.norm(full.astype(np.float32, copy=False), axis=-1)
        median_magnitude = float(np.median(full_mag))
        method = "exact"
    else:
        method = "histogram"
        median_magnitude = _histogram_median_flow(
            flow,
            axis_order=axis_order,
            spatial_shape=spatial_shape,
            slab_z=slab_z,
            value_min=mag_min,
            value_max=mag_max,
            histogram_bins=histogram_bins,
        )

    return DisplacementStats(
        shape=spatial_shape,
        axis_order=axis_order,
        mean_magnitude=mean_magnitude,
        max_magnitude=mag_max if np.isfinite(mag_max) else float("nan"),
        median_magnitude=median_magnitude,
        median_method=method,
        per_axis_mean=(
            float(per_axis_mean_arr[0]),
            float(per_axis_mean_arr[1]),
            float(per_axis_mean_arr[2]),
        ),
        per_axis_max_abs=(
            float(per_axis_max_abs[0]),
            float(per_axis_max_abs[1]),
            float(per_axis_max_abs[2]),
        ),
    )


def _iter_z_slabs(array: zarr.Array[Any], *, slab_z: int) -> Any:
    """Yield Z-slabs of ``array`` as contiguous NumPy buffers.

    Generator over slabs of shape ``(<= slab_z, *array.shape[1:])``.
    """
    z_total = int(array.shape[0])
    for z0 in range(0, z_total, slab_z):
        z1 = min(z0 + slab_z, z_total)
        yield np.asarray(array[z0:z1])


def _iter_flow_z_slabs(
    flow: zarr.Array[Any],
    *,
    axis_order: FieldAxisOrder,
    spatial_shape: tuple[int, int, int],
    slab_z: int,
) -> Any:
    """Yield Z-slabs of a flow array normalized to ``(slab_z, Y, X, 3)``.

    Handles both ``"3_zyx"`` and ``"zyx_3"`` on-disk layouts so the
    stats kernel does not branch.
    """
    z_total = spatial_shape[0]
    for z0 in range(0, z_total, slab_z):
        z1 = min(z0 + slab_z, z_total)
        if axis_order == "zyx_3":
            slab = np.asarray(flow[z0:z1])
        else:
            slab = np.moveaxis(np.asarray(flow[:, z0:z1]), 0, -1)
        yield slab


def _normalize_flow_to_zyx3(
    array: np.ndarray[Any, Any], *, axis_order: FieldAxisOrder
) -> np.ndarray[Any, Any]:
    """Return ``array`` reshaped to ``(Z, Y, X, 3)`` regardless of input layout."""
    if axis_order == "zyx_3":
        return array
    return np.moveaxis(array, 0, -1)


def _histogram_median(
    array: zarr.Array[Any],
    *,
    slab_z: int,
    value_min: float,
    value_max: float,
    histogram_bins: int,
) -> float:
    """Two-pass histogram-based median estimate for an intensity volume."""
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return float("nan")
    if value_max <= value_min:
        return float(value_min)

    edges = np.linspace(value_min, value_max, histogram_bins + 1, dtype=np.float64)
    counts = np.zeros(histogram_bins, dtype=np.int64)
    for slab in _iter_z_slabs(array, slab_z=slab_z):
        h, _ = np.histogram(slab, bins=edges)
        counts += h.astype(np.int64, copy=False)
    return _median_from_histogram(counts=counts, edges=edges)


def _histogram_median_flow(
    flow: zarr.Array[Any],
    *,
    axis_order: FieldAxisOrder,
    spatial_shape: tuple[int, int, int],
    slab_z: int,
    value_min: float,
    value_max: float,
    histogram_bins: int,
) -> float:
    """Histogram-based median estimate for ``||u||`` over a flow array."""
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return float("nan")
    if value_max <= value_min:
        return float(value_min)

    edges = np.linspace(value_min, value_max, histogram_bins + 1, dtype=np.float64)
    counts = np.zeros(histogram_bins, dtype=np.int64)
    for slab in _iter_flow_z_slabs(
        flow, axis_order=axis_order, spatial_shape=spatial_shape, slab_z=slab_z
    ):
        slab32 = slab.astype(np.float32, copy=False)
        mag = np.linalg.norm(slab32, axis=-1)
        h, _ = np.histogram(mag, bins=edges)
        counts += h.astype(np.int64, copy=False)
    return _median_from_histogram(counts=counts, edges=edges)


def _median_from_histogram(
    *,
    counts: np.ndarray[Any, Any],
    edges: np.ndarray[Any, Any],
) -> float:
    """Linearly-interpolated median from histogram counts.

    ``edges`` has length ``len(counts) + 1`` and is the array
    :func:`numpy.histogram` consumed; the median is the value where
    the cumulative count crosses ``total / 2``, interpolated
    inside its bin.
    """
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    half = total / 2.0
    cumsum = np.cumsum(counts.astype(np.int64))
    bin_idx = int(np.searchsorted(cumsum, half, side="left"))
    bin_idx = min(bin_idx, len(counts) - 1)
    bin_count = int(counts[bin_idx])
    if bin_count == 0:
        return float(edges[bin_idx])
    prev_cum = int(cumsum[bin_idx - 1]) if bin_idx > 0 else 0
    fraction = (half - prev_cum) / bin_count
    lo = float(edges[bin_idx])
    hi = float(edges[bin_idx + 1])
    return lo + fraction * (hi - lo)
