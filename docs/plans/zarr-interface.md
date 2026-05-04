# Zarr dataset interface — plan & contract

A typed reader layer on top of the bare `zarr.open(...)[path]` calls
that `scripts/run_e2e_zarr.py` uses today. Lifts a zarr store into a
`DvcDataset` object that exposes the **base image**, **mask(s)**, and
the named **deformed images** (real + synthetic with ground-truth
fields), and a small evaluation surface that compares
`DisplacementField` results against the GT fields.

Fulfills the slots `overview.md` §4 reserves for `io/volume.py`,
`io/mask.py`, `io/field.py`, and `validate/known_fields.py`. Nothing in
`core/`, `pipeline/`, or `gpu/` changes — the reader feeds NumPy
buffers, exactly what `correlate()` and `correlate_multi_gpu()`
already accept.

This document is structured to mirror `correlate.md`, `dispatch.md`,
and `timestep-pipeline.md`: position → invariants → API → flow → data
structures → evaluation → test surface → deferrals.

## 1. Position in the codebase

```
                    ┌──────────────────────────────────────────────┐
                    │ DvcDataset.open(path)                        │
                    │   - lists deformations (real | synthetic)    │
                    │   - exposes lazy zarr.Arrays                 │
                    │   - load_pair(name) → EvaluationPair         │
                    └────────────────┬─────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   io.volume.load_volume     io.mask.load_mask     io.field.GroundTruthField
   (host materializer)       (bool dtype coerce)   (callable, sampled GT)
                                     │
                                     ▼
                        EvaluationPair (ref, def, mask, gt)
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
              ▼                                             ▼
   pipeline.correlate / gpu.dispatch.correlate_multi_gpu    │
              │                                             │
              ▼                                             │
       DisplacementField   ──►  validate.known_fields.evaluate_pair
                                                            │
                                                            ▼
                                                       ErrorReport
```

`io/dataset.py` is **pure plumbing**: it produces NumPy buffers and
callables. All compute lives downstream. All file-system / storage
concerns stop at this boundary.

`validate/known_fields.py` is **pure CPU**: a NumPy-only comparison of
a `DisplacementField` against a `GroundTruthField` callable, sampled
at the POI centers.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Opening a store reads no voxels | Reference + deformed are 6.3 GB each; opening is interactive. Materialization is per-call and explicit. |
| Materialization happens at one chokepoint | `io/volume.py::load_volume` is the single host-side materializer. Subblock slicing (`dry_shape`), dtype coerce, and contiguity all live there — once. The current copy in `run_e2e_zarr.py` migrates here verbatim. |
| Synthetic and experimental GT share one type | `GroundTruthField` is callable with the same signature as `validate.synthetic.DisplacementFunction`. `validate.synthetic.sample_on_grid(field, grid)` works on it unchanged. Test assertions written against synthetic shifts run unmodified against experimental data. |
| Schema is data, not branching | `StoreLayout` is a small adapter so we can support more than one zarr layout (yours now, future collaborators) without rewriting the dataset. New layouts plug in by registering a layout, not editing `DvcDataset`. |
| The "real" deformation is a first-class entry | It has `image` but no `field`. `kind="real"` distinguishes it from synthetic entries; `evaluate_pair` raises a clear error if asked to score a real entry without supplying a field externally. |
| Reader emits no compute | No NCC, no warping, no FFT. Reader output is fungible with hand-loaded NumPy buffers. |
| Frozen-result discipline carries through | `EvaluationPair`, `ErrorReport`, `BoundaryStratifiedTable` are `frozen=True`. Series-style aggregation builds new objects. |
| One `EvaluationPair` corresponds to one `correlate()` call | This keeps the interface trivially composable with the timestep `correlate_series` driver — a sweep is a `for name in dataset.list_synthetic()` loop wrapping single-pair calls. |

## 3. API

### Dataset (`mamba_dvc/io/dataset.py`)

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from jaxtyping import Bool, Float32

from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.types import VoxelSpacing


DeformationKind = Literal["real", "synthetic"]


@dataclass(frozen=True)
class DeformationEntry:
    """One deformed-image instance inside a DvcDataset.

    ``field`` is ``None`` for the real experimental deformation
    (no synthetic ground truth available); present for
    synthetically-deformed entries.
    """

    name:  str
    image: zarr.Array                # lazy
    field: zarr.Array | None         # lazy; None ⇔ kind == "real"
    kind:  DeformationKind
    attrs: Mapping[str, Any]         # passthrough .zattrs


class DvcDataset:
    """A zarr-backed (reference, masks, deformations) bundle.

    Lazy by construction. Opening reads no voxels; materialization is
    explicit per call. Use ``load_pair(name)`` to assemble inputs for
    a single ``correlate()`` invocation.
    """

    reference:    zarr.Array
    masks:        Mapping[str, zarr.Array]    # at minimum the key "default"
    deformations: Mapping[str, DeformationEntry]
    spacing:      VoxelSpacing | None
    volume_shape: tuple[int, int, int]
    layout:       "StoreLayout"

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        layout: "StoreLayout | None" = None,
        mode: Literal["r", "r+"] = "r",
    ) -> "DvcDataset":
        """Open a zarr store and bind its named entries.

        ``layout`` defaults to :class:`DefaultLayout` (§5). Stores
        produced by other tools can register their own layout and pass
        it explicitly.
        """

    # -- listing --------------------------------------------------------
    def list_all(self)        -> list[str]: ...
    def list_synthetic(self)  -> list[str]: ...
    def list_real(self)       -> list[str]: ...

    # -- materializers --------------------------------------------------
    # All accept dry_shape for centered subblocks (mirrors
    # run_e2e_zarr.py --dry-run-shape today).
    def load_reference(
        self, *, dry_shape: tuple[int, int, int] | None = None
    ) -> Float32[np.ndarray, "z y x"]: ...

    def load_mask(
        self,
        name: str = "default",
        *,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> Bool[np.ndarray, "z y x"]: ...

    def load_pair(
        self,
        deformation: str,
        *,
        mask: str | None = "default",
        dry_shape: tuple[int, int, int] | None = None,
    ) -> "EvaluationPair":
        """Materialize ref + def + mask + GT for a single correlate() call.

        ``deformation`` must be a key returned by ``list_all()``.
        ``mask=None`` skips the mask. ``mask="default"`` (the default)
        uses the shared screw mask.
        """
```

### EvaluationPair (`mamba_dvc/io/dataset.py`)

```python
@dataclass(frozen=True)
class EvaluationPair:
    """Materialized inputs ready for one correlate() call.

    Carries everything a single-pair run needs plus the ground-truth
    callable. Pass straight to ``correlate()`` /
    ``correlate_multi_gpu()`` and pipe the result + ``self`` into
    ``validate.known_fields.evaluate_pair``.
    """

    reference: Float32[np.ndarray, "z y x"]
    deformed:  Float32[np.ndarray, "z y x"]
    mask:      Bool[np.ndarray, "z y x"] | None
    gt_field:  GroundTruthField | None         # None for real
    name:      str                             # the deformation key
    kind:      DeformationKind
    spacing:   VoxelSpacing | None
```

### Volume / mask helpers (`mamba_dvc/io/volume.py`, `mamba_dvc/io/mask.py`)

These are the pieces extracted from `scripts/run_e2e_zarr.py` so the
reader and the perf harness share one implementation:

```python
# io/volume.py
def open_zarr_array(store_path: Path, internal_path: str, *, mode: str = "r") -> zarr.Array: ...

def load_volume(
    array: zarr.Array,
    *,
    dry_shape: tuple[int, int, int] | None = None,
    as_float32: bool = True,
) -> np.ndarray:
    """Materialize a 3D zarr array into a contiguous host buffer.

    ``dry_shape`` selects a centered (z,y,x) subblock for quick smoke
    runs (mirrors today's ``--dry-run-shape``). ``as_float32=False``
    coerces to ``np.bool_`` for masks.
    """
```

```python
# io/mask.py
def load_mask(
    array: zarr.Array,
    *,
    dry_shape: tuple[int, int, int] | None = None,
) -> Bool[np.ndarray, "z y x"]:
    """Load a mask zarr array as contiguous bool, with shape validation."""
```

### Ground-truth field (`mamba_dvc/io/field.py`)

```python
from typing import Literal

import numpy as np
from jaxtyping import Float32
from scipy.ndimage import map_coordinates

from mamba_dvc.types import GridSpec


FieldConvention = Literal["pull_back", "push_forward"]
FieldAxisOrder  = Literal["zyx_3", "3_zyx"]   # or "xyz_3" — see Q2 in §11


class GroundTruthField:
    """Sampled (Z, Y, X, 3) deformation array exposed as a callable.

    Drop-in compatible with
    :class:`mamba_dvc.validate.synthetic.DisplacementFunction` so all
    synthetic-test machinery (composition, ``sample_on_grid``,
    pull-back warp) works on experimental data unchanged.

    Parameters
    ----------
    array
        Sampled displacement field. Internal storage is normalized to
        ``(Z, Y, X, 3)`` with last-axis order ``(dz, dy, dx)``;
        constructor adapters (``from_zyx_3``, ``from_3_zyx``, …)
        handle the on-disk variants.
    convention
        ``"pull_back"`` matches ``validate.synthetic.warp``:
        ``deformed(x) = reference(x − u(x))``. ``correlate()``
        recovers the same ``u``. ``"push_forward"`` is the opposite
        sign and is converted to pull-back at construction time so
        downstream code never branches on convention.
    interpolation
        Spline order for off-grid evaluation. Default 3 (cubic) — same
        order ``validate.synthetic.warp`` uses, so synthetic-vs-experimental
        comparisons share the same interpolation error budget.

    Notes
    -----
    Storage is normalized at construction. Comparisons against
    ``DisplacementField.displacements`` work without sign-flipping.
    """

    def __init__(
        self,
        array: Float32[np.ndarray, "z y x 3"],
        *,
        convention: FieldConvention = "pull_back",
        interpolation: int = 3,
    ) -> None: ...

    @classmethod
    def from_zarr(
        cls,
        zarr_array: zarr.Array,
        *,
        axis_order: FieldAxisOrder,
        convention: FieldConvention,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> "GroundTruthField": ...

    def __call__(
        self, coords: Float32[np.ndarray, "points 3"]
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate ``u`` at arbitrary voxel coordinates (z, y, x)."""

    def sampled_on(
        self, grid: GridSpec
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate at every POI center in ``grid``."""
```

The conformance-to-`DisplacementFunction` is the load-bearing trick.
Once `GroundTruthField` is callable, all of
`validate.synthetic.compose / sample_on_grid / warp` work on it,
and `validate/known_fields.py` is just a thin error-stats layer.

### Evaluation (`mamba_dvc/validate/known_fields.py`)

```python
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32

from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
from mamba_dvc.io.dataset import DvcDataset, EvaluationPair
from mamba_dvc.types import DisplacementField


@dataclass(frozen=True)
class BoundaryStratifiedTable:
    """MAE binned by distance from the mask boundary."""

    edges:  tuple[float, ...]                # right-closed bin edges
    counts: Float32[np.ndarray, "bins"]      # n POIs per bin
    mae:    Float32[np.ndarray, "bins"]
    rmse:   Float32[np.ndarray, "bins"]


@dataclass(frozen=True)
class ErrorReport:
    """Aggregate accuracy of one DisplacementField vs its GT."""

    name:                str
    n_total:             int
    n_valid:             int
    mae:                 float
    rmse:                float
    p95:                 float
    per_axis_mae:        tuple[float, float, float]    # (dz, dy, dx)
    by_distance:         BoundaryStratifiedTable | None
    confidence_vs_error: float                          # Spearman ρ
    estimated:           DisplacementField              # raw, for plots
    truth:               Float32[np.ndarray, "points 3"]


def evaluate_pair(
    pair: EvaluationPair,
    field: DisplacementField,
    *,
    distance_bins: Sequence[float] = (0, 5, 10, 20, 50),
) -> ErrorReport:
    """Score a single DVC result against the pair's GT field.

    Raises
    ------
    ValueError
        If ``pair.gt_field is None`` (real-deformation entry).
    """


def run_and_evaluate(
    dataset: DvcDataset,
    deformation: str,
    *,
    dispatcher: MultiGPUDispatcher | None = None,
    dry_shape: tuple[int, int, int] | None = None,
    **correlate_kwargs,
) -> ErrorReport:
    """Materialize → correlate → evaluate, in one call."""


def sweep(
    dataset: DvcDataset,
    *,
    names: Sequence[str] | None = None,    # default: list_synthetic()
    dispatcher: MultiGPUDispatcher | None = None,
    dry_shape: tuple[int, int, int] | None = None,
    **correlate_kwargs,
) -> list[ErrorReport]:
    """Iterate over every synthetic deformation; returns one row per name.

    With a ``MultiGPUDispatcher`` (timestep-pipeline §3) the spawn
    cost is paid once across the whole sweep. Without one, each
    pair pays the full per-call dispatch overhead.
    """
```

`evaluate_pair` is the primitive; `run_and_evaluate` is the
single-name convenience; `sweep` is the "tabulate everything"
mechanism the user asked for. None of them owns a worker pool —
callers pass a dispatcher in if they want to amortize spawn.

## 4. Flow

### Read flow

```
DvcDataset.open(path)
  └── layout.discover(root) → {reference_path, mask_paths,
                               deformation_entries: [(name, image_path,
                               field_path | None, kind, attrs)]}
  └── bind zarr.Array handles (no I/O)
  └── read .zattrs spacing → VoxelSpacing | None

DvcDataset.load_pair(name)
  ├── load_reference()        ─► io.volume.load_volume   (float32)
  ├── load_mask("default")    ─► io.mask.load_mask       (bool)
  ├── load_volume(entry.image)─► io.volume.load_volume   (float32)
  └── if entry.field is not None:
          GroundTruthField.from_zarr(entry.field, axis_order, convention)
      else:
          gt_field = None
  └── return EvaluationPair(...)
```

### Evaluate flow

```
evaluate_pair(pair, field):
  truth      = pair.gt_field.sampled_on(grid_from(field))     # (N, 3)
  err        = field.displacements - truth                    # (N, 3)
  mask_valid = field.valid                                     # (N,)
  mae        = mean(|err[valid]|)
  rmse       = sqrt(mean(err[valid]² ))
  p95        = percentile(|err[valid]|, 95)
  per_axis   = mean(|err[valid, axis]|, axis=0)
  by_dist    = bin_by_distance(field.positions, mask, edges) → MAE per bin
  conf_corr  = spearman(field.confidence[valid], |err|[valid])
  return ErrorReport(...)
```

`bin_by_distance` uses `scipy.ndimage.distance_transform_edt(~mask)`
once on the materialized mask. The transform is cheap relative to the
volume itself and runs on CPU; it is computed lazily inside
`evaluate_pair` so callers that don't ask for stratified stats don't
pay for it (passing `distance_bins=()` skips the bin construction).

## 5. Store layouts

The reader supports more than one schema via a `StoreLayout` adapter.
v1 ships one default plus the open-ended hook.

### `DefaultLayout` (proposed — pinned by §11 Q1)

```
scan.zarr/
  reference                       # (Z, Y, X) float32
  masks/
    default                       # (Z, Y, X) bool   (the shared screw mask)
    [<other>]                     # optional, e.g. per-deformation masks (v2)
  deformations/
    real/
      image                       # (Z, Y, X) float32
      .zattrs : {"kind": "real"}
    <synthetic_name>/
      image                       # (Z, Y, X) float32
      field                       # (Z, Y, X, 3) float32  (dz, dy, dx)
      .zattrs : {"kind": "synthetic",
                 "axis_order": "zyx_3",
                 "convention": "pull_back",
                 "units": "voxel"}
  .zattrs : {"spacing": [sz, sy, sx], "unit": "um"}    # optional
```

Discovery rule: any group under `deformations/` with an `image` array
becomes an entry. Presence of `field` decides `kind`:

```
kind = "synthetic" if entry has "field" else "real"
```

`.zattrs` on the synthetic group disambiguates the axis order /
convention / units. Missing attrs default to the values written
above; mismatches raise a clear error rather than silently coercing.

### `StoreLayout` protocol

```python
class StoreLayout(Protocol):
    def discover(self, root: zarr.Group) -> "LayoutBindings": ...

@dataclass(frozen=True)
class LayoutBindings:
    reference_path:         str
    mask_paths:             Mapping[str, str]            # name → path
    deformation_entries:    Sequence["DeformationBinding"]
    spacing_attr:           VoxelSpacing | None

@dataclass(frozen=True)
class DeformationBinding:
    name:        str
    image_path:  str
    field_path:  str | None
    kind:        DeformationKind
    axis_order:  FieldAxisOrder | None       # None ⇒ no field
    convention:  FieldConvention | None
    attrs:       Mapping[str, Any]
```

A future v2 layout (per-frame masks, time-series stores) plugs in by
implementing `StoreLayout.discover` — `DvcDataset` is unchanged.

## 6. Migration of `scripts/run_e2e_zarr.py`

Today the script owns:
- `_open_zarr_array(store_path, internal_path)`  → moves to `io/volume.py::open_zarr_array`
- `_center_slice(shape, dry)`                    → moves to `io/volume.py` (private helper)
- `load_volume(...)`                             → moves to `io/volume.py::load_volume`

Post-migration, the script becomes ~30 lines:

```python
ds   = DvcDataset.open(args.store, layout=DefaultLayout())   # or args-driven
pair = ds.load_pair(args.deformation, dry_shape=args.dry_run_shape)
field = correlate_multi_gpu(pair.reference, pair.deformed, mask=pair.mask, ...)
report = evaluate_pair(pair, field) if pair.gt_field else None
```

Back-compat for the old `--reference-path / --deformed-path / --mask-path`
flags is retained via a `RawPathsLayout` adapter that takes those three
strings and synthesizes a one-deformation `LayoutBindings`. CI smoke
runs continue to pass without rewriting their invocation.

## 7. Test surface

Pure-CPU, no GPU, no real data:

- `tests/io/test_dataset.py::TestOpenSyntheticStore` —
  build a tiny in-memory zarr store with `DefaultLayout` shape, open
  it, assert `list_all() / list_synthetic() / list_real()` return the
  expected names, and `volume_shape` is right.
- `tests/io/test_dataset.py::TestLoadPair` — `load_pair("synthetic_X")`
  yields a fully-populated `EvaluationPair`; `load_pair("real_Y")`
  yields `gt_field=None`.
- `tests/io/test_dataset.py::TestDryShape` — centered subblock matches
  `_center_slice` semantics.
- `tests/io/test_field.py::TestGroundTruthFieldCallable` —
  `GroundTruthField(make_pair(...).field_array)` and the analytical
  `rigid_shift((dz,dy,dx))` agree at POI centers within interpolation
  tolerance.
- `tests/io/test_field.py::TestAxisOrderRoundTrip` —
  storing the same field as `(Z,Y,X,3)` and `(3,Z,Y,X)` produces
  equal callables after `from_zarr`.
- `tests/io/test_field.py::TestConventionFlip` —
  push-forward storage produces the same callable as the matching
  pull-back storage with sign-flipped values.
- `tests/validate/test_known_fields.py::TestEvaluatePairZeroErr` —
  feeding the GT itself through as the "estimated" field yields
  `mae == 0`, `rmse == 0`, `p95 == 0`.
- `tests/validate/test_known_fields.py::TestRealEntryRaises` —
  `evaluate_pair` on an `EvaluationPair` with `gt_field is None`
  raises a clear error.
- `tests/validate/test_known_fields.py::TestBoundaryStratification` —
  with a synthetic mask + a known-bias field, MAE in the inner bin is
  smaller than the boundary bin.

GPU-marked, opt-in:

- `tests/integration/test_known_fields_smoke.py` — open a small real
  zarr fixture (committed binary or generated by a synthetic-deformation
  helper) and run `run_and_evaluate` end-to-end on `device_ids=[0]`.

The reader has no GPU dependence; only the integration test needs a
device.

## 8. Performance notes

- **Open is free.** `zarr.open` walks metadata only; opening the store
  for a shape probe is cheap enough to do interactively in a notebook.
- **`load_volume`** is the dominant cost (≈ 6.3 GB / NVMe bandwidth).
  The current `arr[:]` materialize is already close to NVMe-bound; a
  v2 path can swap to `tensorstore` for async / GDS reads when the
  store is on a network/SSD where parallel chunk reads help. The
  function signature does not change — only the internal read path.
- **Sweep amortization.** With `MultiGPUDispatcher`, mask + reference
  upload happen once per series, and only the deformed image transfers
  per pair. A 10-deformation sweep at v1 defaults goes from
  ~10 × 9 s = 90 s (spawn-per-call) to ~10 × 3 s + 6 s = 36 s.
- **GT field memory.** A `(960, 1280, 1280, 3)` float32 field is
  18.9 GB — too big to hold many simultaneously on the host.
  `GroundTruthField` materializes one field at a time, drops it on
  the next `load_pair`, and never replicates to GPU. Cubic
  `map_coordinates` at ~12 k POIs is sub-second on CPU.

## 9. Buildout order

Each step is its own commit, runs independently, and ships with the
test additions called out in §7.

1. `io/volume.py`: extract `open_zarr_array`, `load_volume`,
   `_center_slice` from `scripts/run_e2e_zarr.py`. Tests:
   `TestDryShape`, `TestLoadVolumeFloat32`, `TestLoadVolumeBool`.
2. `io/mask.py`: thin loader wrapping `load_volume(..., as_float32=False)`.
3. `io/field.py`: `GroundTruthField` class with axis/convention
   adapters. Tests: `TestGroundTruthFieldCallable`,
   `TestAxisOrderRoundTrip`, `TestConventionFlip`.
4. `io/dataset.py`: `DvcDataset`, `EvaluationPair`,
   `DeformationEntry`, `DefaultLayout`, `RawPathsLayout`. Tests:
   `TestOpenSyntheticStore`, `TestLoadPair`.
5. `validate/known_fields.py`: `ErrorReport`,
   `BoundaryStratifiedTable`, `evaluate_pair`. Tests:
   `TestEvaluatePairZeroErr`, `TestRealEntryRaises`,
   `TestBoundaryStratification`.
6. `validate/known_fields.py`: `run_and_evaluate`, `sweep` +
   dispatcher integration. Test:
   `TestSweepReusesDispatcher` (mock).
7. Rewrite `scripts/run_e2e_zarr.py` on top of `DvcDataset` + keep
   the `RawPathsLayout` flag set for back-compat.
8. Update `docs/plans/overview.md` §4 `io/` description to reference
   this doc; mark §9 tier-2 entry as "implemented in
   `validate/known_fields.py`, see `docs/plans/zarr-interface.md`".

## 10. Known non-features (by design, v1)

- **No write side.** This reader is read-only. Writing
  `DisplacementField` results back into the store is the timestep
  pipeline's `on_pair` writer (see `timestep-pipeline.md` §8), not
  this layer.
- **No per-frame deformed masks.** `DefaultLayout` allows the
  `masks/` group to hold more than `default`, but `load_pair` only
  selects one. v2 per-frame masks are an additive change (overview §10).
- **No tensorstore async.** Open path is synchronous `zarr.open` +
  `arr[:]`. The async reader is a swap-in for `load_volume` when
  needed.
- **No on-the-fly synthetic deformation.** A "deformation generator"
  that takes the base image + a `DisplacementFunction` and writes a
  new entry is a useful sibling tool but not part of the reader.
  Lives at the script level if/when needed.
- **No checkpoint of partial sweeps.** `sweep` runs to completion;
  failed entries propagate. A failure-tolerant variant lifts the
  `correlate_series` / `SeriesPairStatus` machinery and is left for
  v2.

## 11. Open questions (blocking — answer before §9 step 4)

These were posed in conversation; landing them here so the eventual
implementation has a single source of truth.

1. **Store layout.** Does the proposed `DefaultLayout` (§5) match
   your existing stores? Specifically:
   - Are paths `reference`, `masks/default`, `deformations/<name>/image`,
     `deformations/<name>/field` correct?
   - Single shared mask, or multiple per store?
   - Any pre-existing convention from upstream tooling we should adopt
     instead of inventing?
2. **Field axis order.** `(Z, Y, X, 3)` last-axis-is-vector (current
   default), or `(3, Z, Y, X)` channel-first, or `(X, Y, Z, 3)`?
3. **Field convention.** Pull-back (`deformed(x) = reference(x − u(x))`,
   matching `validate.synthetic.warp`) or push-forward?
4. **Field units.** Voxels, or physical units (µm/mm) needing
   spacing-conversion before comparing against `correlate()` output?
5. **Real vs synthetic marker.** `kind` attr in `.zattrs`, or
   "no `field` array present" sufficient?
6. **Dtype on disk.** Are deformed images already float32, or
   uint16/int16 needing rescale? Affects `load_volume(as_float32=True)`
   default behavior.

The strawman in §5 picks one answer for each so the surrounding
design is concrete; the answers replace those defaults verbatim once
the user confirms or redirects.

## 12. Open questions (non-blocking)

- Should `GroundTruthField` cache the materialized array across
  calls, or rematerialize on each `load_pair`? Default no
  (rematerialize) — sweeps process one entry at a time and the
  18.9 GB working set rules out caching anyway.
- Should `ErrorReport` carry the boundary-distance histogram itself,
  or only the binned MAE? Currently only the binned stats; a debug
  hook on `evaluate_pair` can return the raw distances for
  notebook plots.
- Does `sweep` deserve an `on_result` callback (mirrors
  `correlate_series`'s `on_pair`) for streaming reports to disk?
  Probably yes — defer until a real sweep run shows the report
  list is large enough to matter.
- Should `DvcDataset.load_pair` accept an explicit `mask_dry_shape`
  separate from `dry_shape` so the user can debug with a smaller
  volume but the full mask? Probably no — the dry-shape contract
  says everything is sliced consistently. If it's needed, raise it
  later.
