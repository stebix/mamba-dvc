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
   io.profiles.StoreProfile            io.manifest.StoreManifest
   (registered at import time)         (optional sidecar / .zattrs override)
              │                                             │
              └──────────────┬──────────────────────────────┘
                             ▼
                    io.manifest.verify(profile, manifest, root)
                             │
                             ▼
                    ┌──────────────────────────────────────────────┐
                    │ DvcDataset.open(path, manifest=None)         │
                    │   - resolves profile (registry / manifest)   │
                    │   - verifies store against profile+manifest  │
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
| Schema is data, not branching | Layout lives in two cooperating artifacts: a Python-side `StoreProfile` (structural defaults, registered by name) and an optional YAML/zattrs `StoreManifest` (per-store overrides + subsetting). Future collaborators register a new profile; `DvcDataset` is unchanged. |
| Profile is structural, not enumerative | The profile pins group names (`base / iterations / synthetic`), array names (`volume0 / volume1 / flow`), and invariants (dtypes, spatial-shape match, vector-axis position). It does **not** list per-entry instances — those are discovered from the store and optionally subset by the manifest. |
| Kind is structural, not attributed | `iterations/<name>` ⇒ real; `synthetic/<name>` ⇒ synthetic. No `kind` attr is read from per-entry `.zattrs`. The first carries `volume1` only; the second carries `volume1` + `flow`. |
| Verifier is on by default | `DvcDataset.open` runs `verify(profile, manifest, root)` and raises on any error (path missing, dtype mismatch, shape mismatch, missing flow on a synthetic entry). `strict=False` returns the report instead of raising for diagnostic notebooks. |
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
from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.io.profiles import StoreProfile
from mamba_dvc.types import VoxelSpacing


DeformationKind = Literal["real", "synthetic"]


@dataclass(frozen=True)
class DeformationEntry:
    """One deformed-image instance inside a DvcDataset.

    ``flow`` is ``None`` for entries under ``iterations/`` (real
    experimental deformations, no synthetic ground truth available);
    present for entries under ``synthetic/``.
    """

    name:  str
    image: zarr.Array                # lazy; either iterations/<name>/volume1
                                     # or synthetic/<name>/volume1
    flow:  zarr.Array | None         # lazy; None ⇔ kind == "real"
    kind:  DeformationKind           # derived from parent group


class DvcDataset:
    """A zarr-backed (reference, masks, deformations) bundle.

    Lazy by construction. Opening reads no voxels; materialization is
    explicit per call. Use ``load_pair(name)`` to assemble inputs for
    a single ``correlate()`` invocation.
    """

    reference:    zarr.Array
    masks:        Mapping[str, zarr.Array]
    deformations: Mapping[str, DeformationEntry]   # union of real + synthetic
    spacing:      VoxelSpacing | None
    volume_shape: tuple[int, int, int]
    profile:      StoreProfile
    manifest:     StoreManifest | None

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        manifest: StoreManifest | None = None,
        strict: bool = True,
        mode: Literal["r", "r+"] = "r",
    ) -> "DvcDataset":
        """Open a zarr store, resolve its profile, and verify.

        ``manifest=None`` triggers sidecar / .zattrs discovery via
        :meth:`StoreManifest.discover`. The resolved profile comes
        from the manifest's ``store_format`` field, or — when no
        manifest exists — from the registry (single registered
        profile is auto-selected; otherwise the call errors).

        ``strict=True`` (default) raises :class:`StoreVerificationError`
        on any verifier error. ``strict=False`` returns the dataset
        anyway and exposes the report on ``self.verification_report``
        for diagnostic use in notebooks.
        """

    # -- listing --------------------------------------------------------
    def list_all(self)        -> list[str]: ...   # union of real + synthetic
    def list_synthetic(self)  -> list[str]: ...
    def list_real(self)       -> list[str]: ...
    def list_masks(self)      -> list[str]: ...

    # -- materializers --------------------------------------------------
    # All accept dry_shape for centered subblocks (mirrors
    # run_e2e_zarr.py --dry-run-shape today).
    def load_reference(
        self, *, dry_shape: tuple[int, int, int] | None = None
    ) -> Float32[np.ndarray, "z y x"]: ...

    def load_mask(
        self,
        name: str | None = None,
        *,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> Bool[np.ndarray, "z y x"]:
        """Load a mask by name. ``None`` picks the profile's default."""

    def load_pair(
        self,
        deformation: str,
        *,
        mask: str | None = None,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> "EvaluationPair":
        """Materialize ref + def + mask + GT for a single correlate() call.

        ``deformation`` must be a key returned by ``list_all()``.
        ``mask=None`` (default) uses the profile's ``default_mask``
        (``"mask"`` for ``bone_screw_synchrotron_v1``); pass an
        explicit name (``"mask_fill"``) to override; pass the sentinel
        ``"none"`` to skip the mask entirely.
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
DvcDataset.open(path, manifest=None, strict=True)
  ├── manifest = manifest or StoreManifest.discover(path)        # YAML | .zattrs | None
  ├── profile  = get_profile(manifest.profile_name)              # or default_profile()
  ├── root     = zarr.open(path, mode="r")
  ├── report   = verify(profile, manifest, root)
  │   └── raise StoreVerificationError(report) if strict and not report.ok
  ├── discover entries:
  │     real_names      = sorted(child for child in root[profile.real_group])
  │     synthetic_names = sorted(child for child in root[profile.synthetic_group])
  │     mask_names      = sorted(c for c in root[profile.base_group]
  │                              if c != profile.reference_name)
  │   then apply manifest subset filters (entries: discover | [...])
  ├── bind zarr.Array handles (no I/O):
  │     reference  = root[f"{profile.base_group}/{profile.reference_name}"]
  │     masks[n]   = root[f"{profile.base_group}/{n}"]            for each name
  │     entry[n].image = root[f"{group}/{n}/{profile.deformed_name}"]
  │     entry[n].flow  = root[f"{profile.synthetic_group}/{n}/{profile.flow_name}"]
  │                      for synthetic only; None for real
  └── read manifest.spacing → VoxelSpacing | None

DvcDataset.load_pair(name)
  ├── load_reference()                      ─► io.volume.load_volume   (float32)
  ├── load_mask(mask or profile.default_mask)─► io.mask.load_mask      (bool)
  ├── load_volume(entry.image)              ─► io.volume.load_volume   (float32)
  └── if entry.flow is not None:
          flow_spec = manifest.synthetic.flow or profile.flow_defaults
          GroundTruthField.from_zarr(entry.flow,
                                     axis_order=flow_spec.axis_order,
                                     convention=flow_spec.convention,
                                     dry_shape=dry_shape)
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

## 5. Store schema — profile + manifest

Store layout is **data, not branching**. Two distinct artifacts cooperate:

- **Profile** (§5a): Python-side declaration of *structural* schema —
  group names, array names, default flow metadata, default mask, and
  invariants. Identified by string name. Registered at import time.
  One named profile is required.
- **Manifest** (§5b): optional sidecar (YAML next to the store, or
  root `.zattrs["dvc_store"]`) carrying *overrides* and *subsetting*
  on top of a profile. A store that conforms to its profile exactly
  needs no manifest at all.

The verifier (§5c) runs profile + manifest against the actual zarr
store on `DvcDataset.open` and emits a `VerificationReport`.

### 5a. `StoreProfile` and the profile registry

A profile is the *named identity of a layout convention* — the
collaborator group's "this is how we lay out our zarr stores"
agreement, encoded once. v1 ships exactly one:
`bone_screw_synchrotron_v1`, modeled on the rat-bone µCT layout
in use today.

```
5R_Ti_4w_000.zarr/                    # bone_screw_synchrotron_v1
  base/
    volume0                           # (Z, Y, X) float32   reference
    mask                              # (Z, Y, X) bool      shared screw mask
    mask_fill                         # (Z, Y, X) bool      filled-screw mask
    [<other masks>]                   # discoverable
  iterations/
    016/
      volume1                         # (Z, Y, X) float32   real deformed
    [<other entries>]                 # discoverable
  synthetic/
    fs004/
      volume1                         # (Z, Y, X) float32   synthetic deformed
      flow                            # (3, Z, Y, X) float32  (dz, dy, dx)
    [<other entries>]                 # discoverable
```

**Structural rules baked into the profile:**

- Top-level groups `base`, `iterations`, `synthetic` (names fixed).
- `base/volume0` is the reference. (Name fixed.)
- Each child of `base` other than `volume0` is a mask. Mask names are
  open; profile declares `default_mask: "mask"`.
- Each child of `iterations/` is a real-deformation entry; it must
  contain a `volume1` array.
- Each child of `synthetic/` is a synthetic-deformation entry; it
  must contain `volume1` and `flow`.
- `kind` is **structural**: parent group is `iterations` ⇒ real,
  parent group is `synthetic` ⇒ synthetic. Never read from a
  per-entry attribute.
- Flow defaults: `axis_order="3_zyx"`, `vector_order="dz_dy_dx"`,
  `convention="pull_back"` (assumed; see §11 Q3 — needs verification
  against your data; flip via `GroundTruthField(convention=...)`),
  `units="voxel"`.
- Volume / mask dtype invariants: `volume0` and `volume1` are
  `float32`; masks are `bool`; flow is `float32`.

```python
# mamba_dvc/io/profiles.py

@dataclass(frozen=True)
class StoreProfile:
    """Named structural schema for a zarr store."""

    name:           str
    base_group:     str                 # "base"
    reference_name: str                 # "volume0"
    default_mask:   str | None          # "mask"
    real_group:     str                 # "iterations"
    synthetic_group: str                # "synthetic"
    deformed_name:  str                 # "volume1"  (in iterations/* and synthetic/*)
    flow_name:      str                 # "flow"     (in synthetic/* only)
    flow_defaults:  "FlowSpec"
    dtype_invariants: "DtypeInvariants"


def register_profile(profile: StoreProfile) -> None: ...
def get_profile(name: str) -> StoreProfile: ...
def default_profile() -> StoreProfile:
    """The single registered profile when only one exists; raises otherwise."""


# Registered at import time:
BONE_SCREW_SYNCHROTRON_V1 = StoreProfile(
    name="bone_screw_synchrotron_v1",
    base_group="base",
    reference_name="volume0",
    default_mask="mask",
    real_group="iterations",
    synthetic_group="synthetic",
    deformed_name="volume1",
    flow_name="flow",
    flow_defaults=FlowSpec(
        axis_order="3_zyx",
        vector_order="dz_dy_dx",
        convention="pull_back",      # assumed; see §11 Q3
        units="voxel",
    ),
    dtype_invariants=DtypeInvariants(
        reference="float32",
        deformed="float32",
        mask="bool",
        flow="float32",
    ),
)
register_profile(BONE_SCREW_SYNCHROTRON_V1)
```

A second collaborator's layout adds itself with another
`register_profile(...)` call — no edits to `DvcDataset`, no change
to the manifest schema.

### 5b. `StoreManifest` (optional sidecar)

When omitted, `DvcDataset.open` resolves the profile from the registry
(single-profile world: pick the only one; multi-profile: error unless
the manifest names which to use) and reads the store with profile
defaults. The manifest only appears when the user needs to:

1. **Disambiguate** in a multi-profile world (`store_format: <name>`).
2. **Override** flow metadata (e.g. once §11 Q3 is verified and
   `convention` flips).
3. **Subset** entries (run a sweep over only some of `synthetic/`).
4. **Pin** an explicit mask choice as the default.

The sidecar is searched in two locations, in order:

- `<store>.yaml` next to the `.zarr/` directory (sibling sidecar).
- `<store>.zarr/.zattrs["dvc_store"]` (root group attribute, embedded).

`DvcDataset.open` errors if both exist with conflicting content.

```yaml
# 5R_Ti_4w_000.zarr.yaml — minimal manifest, all overrides optional.
dvc_store_version: 1
store_format: bone_screw_synchrotron_v1     # required when >1 profile registered

# Optional: store-wide spacing. Not in the profile because it varies
# per scan even within one collaborator's data.
spacing:
  values: [1.0, 1.0, 1.0]
  unit: voxel

# Optional overrides, all keyed by section.
base:
  default_mask: mask_fill                   # override profile default
  masks: discover                           # or explicit subset list

iterations:
  entries: discover                         # or ["016", "024"]

synthetic:
  entries: discover                         # or ["fs004", "fs104"]
  flow:
    convention: push_forward                # override profile default
```

```python
# mamba_dvc/io/manifest.py

@dataclass(frozen=True)
class StoreManifest:
    version:       int                                  # == 1 in v1
    profile_name:  str | None                           # None ⇒ pick from registry
    spacing:       VoxelSpacing | None
    base:          "BaseManifest"                       # default_mask override; mask subset
    iterations:    "EntriesManifest"                    # entries subset
    synthetic:     "SyntheticManifest"                  # entries subset; flow overrides

    @classmethod
    def discover(cls, store_path: Path) -> "StoreManifest | None":
        """Look for sidecar YAML or root .zattrs; return None if neither exists."""

    @classmethod
    def from_yaml(cls, path: Path) -> "StoreManifest": ...
    @classmethod
    def from_zattrs(cls, root: zarr.Group) -> "StoreManifest | None": ...

    def to_yaml(self, path: Path) -> None: ...
```

### 5c. Verifier

`verify(profile, manifest, root)` walks the merged (profile-defaults +
manifest-overrides) view and checks every claim against the actual
zarr store. Emits one `VerificationReport.errors` entry per violation;
`strict=True` on `DvcDataset.open` raises `StoreVerificationError` on
the report.

```python
@dataclass(frozen=True)
class VerificationReport:
    ok:       bool
    errors:   tuple[str, ...]
    warnings: tuple[str, ...]


def verify(
    profile: StoreProfile,
    manifest: StoreManifest | None,
    root: zarr.Group,
) -> VerificationReport: ...
```

| Check | Failure example |
|---|---|
| Top-level groups exist | `expected group 'base' at store root` |
| Reference exists with declared dtype | `base/volume0: expected float32, got uint16` |
| Reference is 3D | `base/volume0: expected ndim=3, got 4` |
| Each declared mask exists with bool dtype | `base/mask_fill: expected bool, got uint8` |
| Default mask resolves | `default_mask 'mask' not found under base/` |
| Every iterations entry has `volume1` | `iterations/016: missing 'volume1'` |
| Every synthetic entry has `volume1` and `flow` | `synthetic/fs004: missing 'flow'` |
| Volume spatial shapes match reference | `synthetic/fs104/volume1: shape (960,1280,1280) != reference (960,1280,1281)` |
| Mask spatial shapes match reference | `base/mask: shape mismatch` |
| Flow has correct vector axis | `synthetic/fs004/flow: expected leading axis size 3 (axis_order=3_zyx), got 4` |
| Flow spatial shape matches reference | `synthetic/fs004/flow: spatial shape != reference` |
| Manifest-listed entries exist on disk | `manifest references synthetic/fs999, not in store` |

## 6. Migration of `scripts/run_e2e_zarr.py`

Today the script owns:
- `_open_zarr_array(store_path, internal_path)`  → moves to `io/volume.py::open_zarr_array`
- `_center_slice(shape, dry)`                    → moves to `io/volume.py` (private helper)
- `load_volume(...)`                             → moves to `io/volume.py::load_volume`

Post-migration, the script swaps free-form `--reference-path / --deformed-path /
--mask-path` flags for an entry-name selector:

```python
ds   = DvcDataset.open(args.store)                          # profile auto-resolved
pair = ds.load_pair(
    args.deformation,                                       # e.g. "fs104" or "016"
    mask=args.mask,                                         # default: profile default_mask
    dry_shape=args.dry_run_shape,
)
field = correlate_multi_gpu(pair.reference, pair.deformed, mask=pair.mask, ...)
report = evaluate_pair(pair, field) if pair.gt_field else None
```

The new CLI shape: `--store <path> --deformation <name> [--mask <name>]
[--manifest <yaml>]`. The legacy `--reference-path / --deformed-path /
--mask-path` triplet is **removed**, not deprecated — it predates the
profile + manifest contract and there is no clean overlap. Existing
invocations need updating; the script's `--report` / `--out` /
`--device-ids` / window / batch / NCC flags survive unchanged.

### Discoverability — `mamba-dvc inspect`

Sibling to `run_e2e_zarr.py`: a no-side-effect CLI that opens a store
through `DvcDataset.open(strict=False)`, prints the verifier outcome,
and renders a `rich.tree.Tree` of every parsed slot. Lives at
`mamba_dvc/cli.py`; entry point is `[project.scripts] mamba-dvc =
"mamba_dvc.cli:app"`. The compute pipeline is untouched — the
inspector reads metadata and flags only.

```
mamba-dvc inspect <path> [--manifest YAML] [--profile NAME]
                         [--verbose] [--json] [--no-color]
```

| Flag | Effect |
|---|---|
| `--manifest <yaml>` | Skip sidecar / `.zattrs` discovery; load this manifest. |
| `--profile <name>` | Synthesize a `StoreManifest(profile_name=...)`, merging on top of any discovered manifest. Errors fast on an unregistered name. |
| `--verbose` | Append `chunks=…`, `codecs=…`, and on-disk size per array leaf. |
| `--json` | Emit a machine-readable inventory + verifier report instead of the rich tree. |

Exit codes: `0` = clean, `1` = verifier reported errors (tree still
rendered), `2` = could not open (missing path, bad manifest, unknown
profile). The `1` path is the load-bearing one for scripting — CI can
gate uploads on `mamba-dvc inspect --json` returning `ok=true`.

The header labels the manifest source explicitly: `none`,
`sidecar yaml (<file>)`, `root .zattrs`, `--manifest <name>`, or
`--profile <name> (over <other source>)`. This is the discoverability
piece — a notebook user staring at an unfamiliar store can run
`mamba-dvc inspect` and see exactly which schema artifact the loader
would resolve.

## 7. Test surface

Pure-CPU, no GPU, no real data:

- `tests/io/test_manifest.py::TestParseMinimal` — minimal sidecar YAML
  (only `dvc_store_version` + `store_format`) round-trips through
  `StoreManifest.from_yaml`.
- `tests/io/test_manifest.py::TestParseFullOverride` — every override
  field round-trips: spacing, default mask, entries subset, flow
  override.
- `tests/io/test_manifest.py::TestUnknownProfileRaises` —
  `store_format` naming an unregistered profile fails fast.
- `tests/io/test_profiles.py::TestRegistry` — `register_profile` adds,
  `get_profile` retrieves, `default_profile()` raises when >1 registered
  with no manifest.
- `tests/io/test_verifier.py::TestVerifyHappyPath` — build an in-memory
  zarr matching `bone_screw_synchrotron_v1`; `verify(...)` returns
  `ok=True` with no errors.
- `tests/io/test_verifier.py::TestVerifyShapeMismatch` — synthetic
  entry's `volume1` has wrong spatial shape; verifier reports a
  shape-mismatch error.
- `tests/io/test_verifier.py::TestVerifyMissingFlow` — synthetic entry
  missing `flow`; verifier flags it; iterations entries with no flow
  don't trigger the same check.
- `tests/io/test_verifier.py::TestVerifyDtypeMismatch` — `volume0`
  written as uint16; verifier reports a dtype-mismatch error.
- `tests/io/test_verifier.py::TestVerifyFlowAxisOrder` — flow stored
  as `(Z,Y,X,3)` while profile says `3_zyx`; verifier flags the
  vector-axis mismatch.
- `tests/io/test_dataset.py::TestOpenAgainstProfile` — build an
  in-memory store conforming to `bone_screw_synchrotron_v1` (no
  manifest); `DvcDataset.open(path)` succeeds and exposes the right
  entries.
- `tests/io/test_dataset.py::TestListings` — `list_real()`,
  `list_synthetic()`, `list_masks()` return the discovered entries.
- `tests/io/test_dataset.py::TestLoadPairSynthetic` —
  `load_pair("fs104")` yields a fully-populated `EvaluationPair` with
  `kind="synthetic"`, `gt_field` callable.
- `tests/io/test_dataset.py::TestLoadPairReal` —
  `load_pair("016")` yields `kind="real"`, `gt_field=None`.
- `tests/io/test_dataset.py::TestLoadPairMaskOverride` — passing
  `mask="mask_fill"` selects the alternate mask; default picks
  profile's `default_mask`.
- `tests/io/test_dataset.py::TestDryShape` — centered subblock semantics
  preserved across reference, deformed, mask, and flow.
- `tests/io/test_dataset.py::TestManifestSubset` — sidecar limits
  `synthetic.entries` to `["fs004"]`; only that entry appears in
  `list_synthetic()`.
- `tests/io/test_field.py::TestGroundTruthFieldCallable` —
  sampling a stored rigid-shift field at POI centers agrees with
  `validate.synthetic.rigid_shift` within interpolation tolerance.
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
4. `io/profiles.py` + `io/manifest.py`: `StoreProfile`, profile
   registry, `bone_screw_synchrotron_v1`, `StoreManifest` parser
   (YAML sidecar + `.zattrs` discovery), `verify` + `VerificationReport`.
   Adds **PyYAML** as an explicit dependency in `pyproject.toml`.
   Tests: `TestParseMinimal`, `TestParseFullOverride`,
   `TestUnknownProfileRaises`, `TestRegistry`,
   `TestVerifyHappyPath`, `TestVerifyShapeMismatch`,
   `TestVerifyMissingFlow`, `TestVerifyDtypeMismatch`,
   `TestVerifyFlowAxisOrder`.
5. `io/dataset.py`: `DvcDataset`, `EvaluationPair`,
   `DeformationEntry`. Consumes profile + optional manifest from
   step 4. Tests: `TestOpenAgainstProfile`, `TestListings`,
   `TestLoadPairSynthetic`, `TestLoadPairReal`,
   `TestLoadPairMaskOverride`, `TestDryShape`,
   `TestManifestSubset`.
6. `validate/known_fields.py`: `ErrorReport`,
   `BoundaryStratifiedTable`, `evaluate_pair`. Tests:
   `TestEvaluatePairZeroErr`, `TestRealEntryRaises`,
   `TestBoundaryStratification`.
7. `validate/known_fields.py`: `run_and_evaluate`, `sweep` +
   dispatcher integration. Test:
   `TestSweepReusesDispatcher` (mock).
8. Rewrite `scripts/run_e2e_zarr.py` on top of `DvcDataset` (legacy
   raw-path flags removed; new `--deformation` / `--mask` /
   `--manifest` interface).
9. Update `docs/plans/overview.md` §4 `io/` description to reference
   this doc; mark §9 tier-2 entry as "implemented in
   `validate/known_fields.py`, see `docs/plans/zarr-interface.md`".
10. `mamba_dvc/cli.py`: typer-based `mamba-dvc inspect` (rich tree +
    validity panel + `--verbose` / `--json` / `--profile`). Adds
    **typer** and **rich** as runtime dependencies and registers the
    `mamba-dvc` console script. Tests: `tests/cli/test_inspect.py`
    (happy path, manifest override, verifier failure, missing path,
    unknown profile, `--verbose`, `--json` × 2, `--profile`).

## 10. Known non-features (by design, v1)

- **No write side.** This reader is read-only. Writing
  `DisplacementField` results back into the store is the timestep
  pipeline's `on_pair` writer (see `timestep-pipeline.md` §8), not
  this layer.
- **No per-frame deformed masks.** Profile allows `base/` to hold
  more than one mask, but `load_pair` only selects one. v2 per-frame
  masks are an additive change (overview §10).
- **No `store init` / `store verify` CLI.** The verifier runs
  implicitly inside `DvcDataset.open(strict=True)`. A standalone
  CLI to seed sidecar manifests from an existing store is useful
  but deferred — once the profile lands, conforming stores need no
  manifest at all, so there is little to seed.
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

## 11. Resolved schema decisions and remaining open question

### Resolved (baked into `bone_screw_synchrotron_v1`)

| Decision | Value | Source |
|---|---|---|
| Top-level grouping | `base / iterations / synthetic` | user-confirmed layout |
| Reference name | `base/volume0` | structural |
| Mask names | open under `base/`; default `mask` | structural + profile |
| Real-vs-synthetic marker | parent group (no per-entry attr) | structural |
| Real-deformation array name | `iterations/<name>/volume1` | structural |
| Synthetic-deformation array names | `synthetic/<name>/{volume1, flow}` | structural |
| Flow axis order | `3_zyx` — `(3, Z, Y, X)` | user-verified |
| Flow vector order | `dz_dy_dx` | profile assumption |
| Flow units | `voxel` | user-confirmed; no spacing conversion |
| `volume0` / `volume1` dtype | `float32` | user-confirmed; no rescale |
| Mask dtype | `bool` | structural |
| Flow dtype | `float32` | structural |

### Still open (one item)

**Q3 — flow convention (pull-back vs push-forward).** Status: unknown.
The profile defaults to `pull_back` (matching
`validate.synthetic.warp` and the sign convention `correlate()`
returns) so the assertion `correlate(ref, def).displacements ≈
gt_field(positions)` works without sign-flipping. **Verification
plan:** the first end-to-end run on a synthetic entry where the GT
is a known small rigid shift will reveal the sign — if recovered
displacements come out as the negative of the stored flow, flip the
manifest's `synthetic.flow.convention` to `push_forward` and re-run.
`GroundTruthField` normalizes storage to pull-back at construction,
so downstream code never branches on convention. Until verified, the
profile carries `pull_back` plus a `# TODO(verify)` comment in
`io/profiles.py`.

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
