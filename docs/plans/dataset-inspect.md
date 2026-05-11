# Dataset inspection — plan & contract

A streaming profiler over a `DvcDataset`. Walks the reference, every
mask, and every (image, flow) pair, emits per-array summary statistics,
and returns a frozen tree of results. Designed for triage in Jupyter
before kicking off a `correlate()` run, and as a quick "did this store
load the way I expect" check after `DvcDataset.open(...)` succeeded.

Fulfills no slot in `overview.md` — this is new diagnostic surface
adjacent to `io/verify.py` (structural conformance) and
`validate/known_fields.py` (correlation accuracy). It exists because
loaded zarr stores are large and opaque; every notebook session
otherwise reinvents the same `np.mean / np.median` on lazy arrays
and trips over memory.

This document mirrors the structure of `zarr-interface.md`:
position → invariants → API → flow → data structures → memory →
test surface → deferrals.

## 1. Position in the codebase

```
                    DvcDataset.open(path)
                            │
                            ▼
            ┌─────────────────────────────────────────┐
            │ inspect_dataset(dataset, *, ...)        │
            │   - walks reference, masks, deformations│
            │   - streams zarr chunks (no full load)  │
            │   - histogram-based percentile path     │
            │   - emits ProgressEvent per item        │
            └────────────────┬────────────────────────┘
                             │
            ┌────────────────┼────────────────┬────────────────┐
            ▼                ▼                ▼                ▼
     IntensityStats     MaskStats     DisplacementStats    skipped:
     (reference)        (per mask)    (per flow)           {name: reason}
            │                │                │
            └────────────────┴────────────────┘
                             ▼
                       DatasetStats
                       (frozen tree)
```

`io/inspect.py` is **pure CPU, no compute beyond accumulation**: no
NCC, no FFT, no warping, no GPU. It treats the dataset as data and
produces summary statistics. All file-system / storage concerns stop
at the `DvcDataset` boundary already; this layer sits one level above
the materializers in `volume.py` / `mask.py` and one level above
`field.py::GroundTruthField`.

`io/verify.py` answers *"is this store loadable?"* (shapes, dtypes,
axis order). `inspect_dataset` answers *"what does the loaded data
look like?"* (intensities, mask coverage, displacement magnitudes).
Both are diagnostic; neither participates in the hot path.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Streaming over zarr chunks; never `array[:]` on a full volume | Reference + deformed are 6.3 GB float32; flow fields are 18.9 GB. Holding even one in RAM strands a notebook kernel. |
| Pure CPU, NumPy-only | The inspector is I/O-bound on disk reads, not compute. GPU would only complicate memory accounting. |
| Frozen-result discipline | `DatasetStats` and its components are `frozen=True`, mirroring `EvaluationPair`, `DisplacementField`, `VerificationReport`. Callers compose new objects, never mutate. |
| One `ProgressEvent` per item, not per chunk | A 30-deformation dataset emits ~30 events. Per-chunk reporting (~10 k events) spams Jupyter and adds no signal. |
| Median is approximate by default, exact is opt-in | A two-pass histogram is bounded-memory and good enough for triage. Exact `np.median` needs ~2× the volume size in workspace; offered as `median="exact"` for `dry_shape` subblocks. |
| Broken entries are forwarded, not raised | `DvcDataset.broken_entries` is the source of truth for what couldn't be bound; the inspector copies these into `DatasetStats.skipped` and continues. The store has already been verified by the time we get here. |
| Inspector is pure of side effects beyond progress emission | No file writes, no logging, no global state. Idempotent: calling twice on the same dataset returns identical stats (modulo `elapsed_seconds`). |
| Progress reporter is a `Protocol`, not a hard tqdm dep | `mamba_dvc/io/` should not import `tqdm` at module load; users opt into `tqdm_reporter()` in the notebook. Built-in `null_reporter` is the default. |

## 3. API

### Public function (`mamba_dvc/io/inspect.py`)

```python
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from mamba_dvc.io.dataset import DvcDataset


def inspect_dataset(
    dataset: DvcDataset,
    *,
    include: Sequence[str] | None = None,
    masks: Sequence[str] | None = None,
    median: Literal["exact", "histogram", "skip"] = "histogram",
    histogram_bins: int = 16384,
    progress: ProgressReporter | None = None,
) -> DatasetStats:
    """Stream summary statistics over every array a DvcDataset exposes.

    Parameters
    ----------
    dataset
        An opened ``DvcDataset``. Materialization happens chunk-by-chunk
        inside this call; the dataset is unchanged on return.
    include
        Optional subset of deformation names to inspect. ``None``
        (default) walks every healthy entry. Names not present in
        ``dataset.deformations`` raise ``KeyError``.
    masks
        Optional subset of mask names. ``None`` walks every mask.
    median
        ``"histogram"`` (default) computes a two-pass histogram-based
        median estimate; ``"exact"`` materializes each volume in full
        and calls ``numpy.median``; ``"skip"`` records ``nan`` for all
        median fields.
    histogram_bins
        Bin count for the histogram path. Default 16384 ≈ 64 KB of
        bin state, ~0.006% relative resolution across the volume's
        intensity range.
    progress
        Optional ``ProgressReporter``. ``None`` (default) is silent.
        Use ``tqdm_reporter()`` for a Jupyter-friendly progress bar.

    Returns
    -------
    DatasetStats
        Frozen tree of per-array statistics; see §5.

    Raises
    ------
    KeyError
        If ``include`` or ``masks`` names an entry the dataset does not
        expose.
    ValueError
        If ``histogram_bins < 2``.
    """
```

### Progress reporter

```python
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ProgressEvent:
    """One inspector step boundary.

    Emitted exactly once per item (reference, each mask, each
    deformation image, each deformation flow) plus a final ``"done"``
    event. ``item_index`` and ``item_total`` count items, not chunks.
    """

    stage: Literal[
        "reference",
        "mask",
        "deformation_image",
        "deformation_flow",
        "done",
    ]
    name: str                         # "reference" | mask name | deformation name
    item_index: int                   # 0-based; equals item_total at "done"
    item_total: int
    elapsed_seconds: float


class ProgressReporter(Protocol):
    """Protocol satisfied by any ``ProgressEvent``-consuming callable."""

    def __call__(self, event: ProgressEvent) -> None: ...


def null_reporter(event: ProgressEvent) -> None:
    """Drop every event. Default when ``progress=None``."""


def print_reporter(event: ProgressEvent) -> None:
    """Print one line per event. Useful for headless / CI runs."""


def tqdm_reporter() -> ProgressReporter:
    """Return a closure that drives a ``tqdm.auto`` progress bar.

    ``tqdm.auto`` selects the notebook widget under Jupyter and the
    plain bar otherwise. Imports ``tqdm`` lazily so the inspector
    module has no hard dependency on it.
    """
```

`tqdm_reporter()` is a factory rather than a function so the closure
can own the `tqdm` instance and call `.close()` on the `"done"` event.

## 4. Flow

### Walk order

For determinism the inspector walks items in this order:

1. `reference` — one pass.
2. Masks, sorted by name (subset of `dataset.list_masks()` filtered by `masks=`).
3. Deformations, sorted by name (subset of `dataset.list_all()` filtered by `include=`):
   - deformation image,
   - deformation flow if `kind == "synthetic"`.

`item_total` in `ProgressEvent` is computed up front from the resolved
inclusion sets so the reporter can render an accurate bar.

### Per-item algorithms

**Intensity volumes (reference + deformation images):**

- *Pass 1* — iterate zarr chunks; accumulate `count`, `sum`,
  `min`, `max` in float64. Single linear scan; ~6.3 GB of disk
  reads at ~5 GB/s NVMe ≈ 1–2 s per volume cold.
- *Pass 2 (only if `median="histogram"`)* — second iterate over the
  same chunks, building a `numpy.histogram` over `[min, max]` with
  `histogram_bins` bins. The median is the bin where the CDF crosses
  0.5, linearly interpolated within the bin.
- *`median="exact"` path* — materialize via `load_volume` and call
  `numpy.median`. Documented as memory-heavy; pair with `dry_shape`
  in v2 if needed (see §8).
- *`median="skip"` path* — record `nan` and `median_method="skipped"`.

**Masks:** single chunk-by-chunk pass; per chunk
`numpy.count_nonzero(chunk)` accumulates `foreground_count`. The
complement gives `background_count`; the ratio gives
`foreground_fraction`. No median.

**Displacement fields (`(Z, Y, X, 3)` after normalization by
`GroundTruthField`):** the on-disk layout may be `"3_zyx"` or
`"zyx_3"`. Rather than re-implement the axis-order branching that
`GroundTruthField.from_zarr` already handles, the inspector treats
the flow at the zarr level: it iterates Z-slabs of the underlying
zarr array (slab thickness chosen so each slab is ≤ 1 GB), and:

1. transposes axis order to `(z, y, x, 3)` per slab if needed,
2. accumulates per-axis `sum`, `max(abs(...))`,
3. computes magnitudes `mag = sqrt(dz² + dy² + dx²)` per slab,
4. updates `mean_magnitude` (sum + count), `max_magnitude`,
5. feeds magnitudes into a magnitude-histogram for the median.

The slab is large enough that vectorized ops dominate; small enough
that peak working memory stays bounded regardless of total field size.

The inspector does **not** apply the convention sign flip
(`pull_back` vs `push_forward`). Magnitudes are sign-invariant and
per-axis stats are unsigned (`max_per_axis_abs`); per-axis means use
the on-disk sign so `push_forward`-stored data reports a negative
mean. This is intentional and documented — the inspector reports
what's on disk, not what `correlate()` would return.

### Skipped entries

`dataset.broken_entries` is forwarded into `DatasetStats.skipped` as
`{name: reason}`. The inspector itself does not try to inspect a
broken entry; the structural problem was already caught by `verify`.

### Timing

`elapsed_seconds` is wall-clock from the first `progress(reference)`
event to the final `progress(done)` event, captured via
`time.perf_counter`.

## 5. Data structures

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IntensityStats:
    """Streaming summary stats for a single intensity volume.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape of the source array.
    dtype
        String name of the source dtype (``"float32"``, ``"uint16"``,
        etc.). The inspector reads in the source dtype and
        accumulates in float64; this records what was on disk.
    count
        Total voxel count. Equal to ``prod(shape)``; included so
        downstream tooling does not have to recompute it.
    mean
        Voxel mean.
    min, max
        Voxel min / max. Exact (single pass).
    median
        Median voxel value. Exact when ``median_method == "exact"``;
        histogram estimate when ``"histogram"``; ``nan`` when
        ``"skipped"``.
    median_method
        How ``median`` was computed.
    """

    shape: tuple[int, int, int]
    dtype: str
    count: int
    mean: float
    min: float
    max: float
    median: float
    median_method: Literal["exact", "histogram", "skipped"]


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
        ``nan`` only if ``shape`` has a zero axis (cannot occur on a
        verified store, but guarded for completeness).
    """

    shape: tuple[int, int, int]
    foreground_count: int
    background_count: int
    foreground_fraction: float


@dataclass(frozen=True)
class DisplacementStats:
    """Streaming summary stats for one ``(Z, Y, X, 3)`` flow array.

    Parameters
    ----------
    shape
        Spatial ``(z, y, x)`` voxel shape (vector axis suppressed).
    mean_magnitude, max_magnitude, median_magnitude
        Stats over ``||u||₂`` per voxel. ``median_magnitude`` follows
        the same exact / histogram / skipped contract as
        :class:`IntensityStats`.
    median_method
        How ``median_magnitude`` was computed.
    per_axis_mean
        ``(mean(dz), mean(dy), mean(dx))``. Reported using on-disk
        sign — see §4 for the convention note.
    per_axis_max_abs
        ``(max|dz|, max|dy|, max|dx|)``.
    """

    shape: tuple[int, int, int]
    mean_magnitude: float
    max_magnitude: float
    median_magnitude: float
    median_method: Literal["exact", "histogram", "skipped"]
    per_axis_mean: tuple[float, float, float]
    per_axis_max_abs: tuple[float, float, float]


@dataclass(frozen=True)
class DeformationStats:
    """Per-entry stats: image always, flow only for synthetic entries."""

    name: str
    kind: Literal["real", "synthetic"]
    image: IntensityStats
    flow: DisplacementStats | None


@dataclass(frozen=True)
class DatasetStats:
    """Top-level inspector result over a ``DvcDataset``.

    Returned by :func:`inspect_dataset`. ``frozen=True``; aggregations
    over multiple datasets build new objects.
    """

    reference: IntensityStats
    masks: Mapping[str, MaskStats]
    deformations: Mapping[str, DeformationStats]
    skipped: Mapping[str, str]            # forwarded broken_entries reasons
    elapsed_seconds: float
    median_method: Literal["exact", "histogram", "skipped"]
```

## 6. Memory model

Per-volume working set is bounded by chunk size, not volume size.

| Source | Per-item peak working memory (typical) |
|---|---|
| Intensity volume (chunked iteration) | one zarr chunk in float64 + 64 KB histogram = ≤ 100 MB |
| Mask (chunked iteration) | one chunk in bool + scalar accumulators ≈ chunk size |
| Flow field (Z-slab iteration) | ≤ 1 GB working slab + 64 KB magnitude histogram |
| Total inspector heap | ≤ 1 GB irrespective of volume count |

The `median="exact"` branch is the only path that holds a full
volume — by design, opt-in, and only meaningful for `dry_shape`
subblocks once that integrates (see §8).

## 7. Test surface

`tests/io/test_inspect.py`:

1. **Synthetic in-memory store** — build a tiny `zarr.MemoryStore`
   with a known reference (e.g. linear ramp 0..N-1, `(8, 8, 8)`),
   one mask with known foreground count, one synthetic deformation
   with a constant-shift flow.
   - Assert `IntensityStats.mean == (N - 1) / 2`, exact min/max.
   - Assert `MaskStats.foreground_count == known`.
   - Assert `DisplacementStats.mean_magnitude == ||shift||`,
     `per_axis_mean == shift`.
2. **Histogram median accuracy** — 256³ float32 textured volume
   (uniform random); assert
   `|hist_median − np.median(arr)| < (max - min) / histogram_bins`.
3. **Median modes** — assert `median="skip"` produces `nan` and
   `median_method="skipped"` everywhere; assert `median="exact"`
   matches `np.median` exactly on a small volume.
4. **Progress events** — collect events with a list-appending
   reporter; assert exact stage / name / index sequence and that
   the final event is `stage="done", item_index == item_total`.
5. **Skipped entries** — open a malformed store with
   `strict=False` so `dataset.broken_entries` is non-empty; assert
   the reasons appear verbatim in `DatasetStats.skipped` and the
   inspector does not raise.
6. **Inclusion filters** — `include=["fs004"]`, `masks=["screw"]`;
   assert only those entries appear and unknown names raise
   `KeyError`.
7. **Determinism** — call twice; assert all stats fields are equal
   (excluding `elapsed_seconds`).

Real-volume tests against the experimental zarr store are
`@pytest.mark.slow` and not in CI.

## 8. Deferrals (v2)

| v2 feature | v1 hook |
|---|---|
| `dry_shape` parameter on the inspector | The materializers already accept `dry_shape`; threading it through `inspect_dataset` is a one-flag change. Skipped in v1 because the histogram path makes full-volume cheap enough that subblock inspection is rarely needed. |
| GPU-accelerated stats | `core/` patterns make it easy to lift the accumulators into CuPy if disk-read overhead ever stops dominating. Today it does not. |
| Streamed JSON / Parquet emit | `DatasetStats` is a flat tree of dataclasses; `dataclasses.asdict` is enough for ad-hoc dumps. A persistent reporting format (filename, schema versioning) is out of scope until a downstream consumer asks for one. |
| Per-deformation summary against the reference | The plan covers each array independently. Cross-array stats (e.g. mean intensity drift between reference and each deformation, per-mask intensity distribution) belong in a higher layer that consumes `DatasetStats`. |
| Confidence histograms over an existing `DisplacementField` | Different surface — that's a `validate/` concern, not an `io/` concern. |
| `inspect_pair(EvaluationPair)` shortcut | Trivial wrapper if a notebook needs it. Not exported in v1. |

## 9. Notebook usage

```python
from pathlib import Path
from mamba_dvc.io.dataset import DvcDataset
from mamba_dvc.io.inspect import inspect_dataset, tqdm_reporter

ds = DvcDataset.open(Path("/data/scan_X.zarr"))

stats = inspect_dataset(
    ds,
    progress=tqdm_reporter(),         # tqdm.auto picks notebook widget
    median="histogram",                # default
)

stats.reference.mean, stats.reference.median
stats.masks["screw"].foreground_fraction
stats.deformations["fs004"].flow.max_magnitude

# fast triage — skip medians:
fast = inspect_dataset(ds, median="skip", progress=tqdm_reporter())

# subset for a focused notebook:
subset = inspect_dataset(
    ds,
    include=["fs004", "fs010"],
    masks=["screw"],
    progress=tqdm_reporter(),
)
```

For dataframe-style inspection downstream:

```python
import pandas as pd

rows = [
    {
        "name": name,
        "kind": s.kind,
        "image_mean": s.image.mean,
        "image_median": s.image.median,
        "flow_max_mag": s.flow.max_magnitude if s.flow else None,
    }
    for name, s in stats.deformations.items()
]
pd.DataFrame(rows)
```

That's a notebook-side composition; `mamba_dvc/io/` does not depend
on pandas.
