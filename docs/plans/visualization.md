# Visualization submodule — research & implementation plan

Scope: the `mamba_dvc/viz/` submodule for volume rendering of µCT volumes
and vector-field rendering of recovered / ground-truth displacement
fields. Used for interactive inspection in Jupyter, debugging the
correlator, and exporting figures.

This plan is decision-grade: backend chosen, layout fixed, hook points
to not-yet-landed types named explicitly. Implementation can begin
without further design rounds.

## 0. Anchors

Read first: `docs/plans/overview.md` (algorithm, architecture, types).
The visualizers are pure consumers of types defined in
`mamba_dvc/types.py` and `mamba_dvc/validate/synthetic.py`; no new
algorithmic state lives here.

Existing data types this plan targets:

- `Float32[np.ndarray, "z y x"]` reference / deformed volumes.
- `Bool[np.ndarray, "z y x"]` shared mask.
- `GridSpec` (`mamba_dvc/types.py`) — POI lattice descriptor.
- `DisplacementField` (`mamba_dvc/types.py`) — recovered field with
  `positions`, `displacements`, `valid`, `confidence`, `status`,
  `grid_shape`, `spacing`, `window`.
- `POIStatus` enum — `OK | MASKED | OUTLIER | LOW_CONF | OUT_OF_RANGE`.
- `SyntheticPair`, `DisplacementFunction` (`validate/synthetic.py`).

## 1. Backend decision: PyVista (+ matplotlib for 2D)

PyVista is the right tool for v1. VisPy buys speed we do not need at
the cost of writing the volume raycaster, glyph instancing, and
transfer-function UI by hand.

**Frontend priority — Jupyter first.** v1 targets notebook-based
inspection. `backend.plotter()` defaults to a `trame`/Jupyter
backend, exposes a `qt: bool = False` override for the native Qt
app, and falls back to `off_screen=True` headless rendering for
figure export and CI image-diff tests. The Qt path stays as a
measured-performance escape hatch: if the trame backend stutters on
the full `(960, 1280, 1280)` volume in a notebook (re-encode
overhead per frame interaction), the same renderer code reuses the
Qt window without API change. Concretely, that means **no
backend-specific imports at module level** in `volume.py`,
`field.py`, etc. — only `backend.py` knows which Plotter type it is
returning.

| Need | PyVista | VisPy |
|---|---|---|
| Volume rendering of `(960, 1280, 1280)` float32 | VTK GPU raycast, transfer functions free | Custom raycast shader |
| Vector glyphs at ~12 k POIs | `Glyph` filter, one call | Manual instanced rendering |
| Slicing widget, plane probe, contours, isosurface | Built in | Build from primitives |
| Jupyter integration | Mature `trame` / `html` backends | Works, less integrated |
| Static figures for the paper | `off_screen=True` + `screenshot()` | DIY framebuffer |
| Type idioms that match ours | `ImageData` ≡ volume; `PolyData`+`Glyph` ≡ POI grid | Generic scene graph |

VisPy is the right escape hatch only if a future time-series scrubber
needs to re-upload the volume to VTK at high frame rates and stutters.
Defer until measured.

For 2D slice browsing (`reference`, `deformed`, `mask`, optional
displacement overlay) use **matplotlib + ipywidgets**, not PyVista.
The `reactive-viz` extra in `pyproject.toml` already pulls
`ipympl`, `ipywidgets`, `jupyter` — reuse those. Reserve PyVista for
genuinely 3D views.

Dependencies to add (under a new `viz` extra in `pyproject.toml`):

```toml
[project.optional-dependencies]
viz = [
    "pyvista>=0.45",
    "trame>=3.0",            # jupyter backend
    "trame-vtk>=2.8",
    "trame-vuetify>=2.7",
    "matplotlib>=3.9",       # 2D slice browser
]
```

`viz` is optional so headless / GPU-only environments do not pull VTK.

## 2. Module layout

```
mamba_dvc/viz/
  __init__.py
  backend.py        # plotter context manager, jupyter backend selection,
                    # screenshot / offscreen helpers
  volume.py         # render_volume(volume, mask=None, ...)
                    # render_isosurface(volume, level, ...)
  field.py          # render_displacement(field, scale=...)
                    # render_field_function(fn, grid)
                    # render_streamlines(field, target_shape)   [v2 hook]
  diagnostics.py    # render_status(field)
                    # render_confidence(field)
                    # highlight_outliers(field)
  validate.py       # render_error_glyphs(report)
                    # render_error_histogram(report)            (matplotlib)
                    # render_error_vs_mask_distance(report)     (matplotlib)
  slices.py         # browse_slices(reference, deformed, mask, field=None)
                    # 2D ipywidgets-driven matplotlib browser
  export.py         # save_screenshot, save_animation
  _conversion.py    # numpy <-> vtkImageData; POIs+displacements -> PolyData
```

Dependency arrow is one-way: `viz` imports `types`, `validate`. Nothing
in `core/`, `pipeline/`, `gpu/`, `io/` may import `viz`.

## 3. API contracts

All renderers take or return a `pv.Plotter` so composition is trivial
and headless export uses the same code path as interactive use.

```python
# mamba_dvc/viz/backend.py

@contextmanager
def plotter(*, jupyter: bool | None = None,
            off_screen: bool = False) -> Iterator[pv.Plotter]: ...

def screenshot(plotter: pv.Plotter, path: Path | str, *,
               size: tuple[int, int] = (1920, 1080)) -> None: ...
```

```python
# mamba_dvc/viz/volume.py

def render_volume(
    plotter: pv.Plotter,
    volume: Float32[np.ndarray, "z y x"],
    *,
    mask: Bool[np.ndarray, "z y x"] | None = None,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    opacity: str | Sequence[float] = "sigmoid",
    cmap: str = "bone",
    clim: tuple[float, float] | None = None,
) -> pv.Plotter: ...

def render_isosurface(
    plotter: pv.Plotter,
    volume: Float32[np.ndarray, "z y x"],
    *,
    level: float,
    spacing: VoxelSpacing | tuple[float, float, float] | None = None,
    color: str = "lightgray",
    opacity: float = 1.0,
) -> pv.Plotter: ...
```

```python
# mamba_dvc/viz/field.py

def render_displacement(
    plotter: pv.Plotter,
    field: DisplacementField,
    *,
    scale: float = 1.0,
    glyph: Literal["arrow", "line"] = "arrow",
    color_by: Literal["magnitude", "component_z", "component_y",
                      "component_x", "confidence"] = "magnitude",
    cmap: str = "viridis",
    only_valid: bool = True,
) -> pv.Plotter: ...

def render_field_function(
    plotter: pv.Plotter,
    fn: DisplacementFunction,
    grid: GridSpec,
    *,
    scale: float = 1.0,
    color: str = "tab:orange",
) -> pv.Plotter: ...

def render_streamlines(  # v2: requires densify
    plotter: pv.Plotter,
    field: DisplacementField,
    target_shape: tuple[int, int, int],
    *,
    n_seeds: int = 200,
) -> pv.Plotter: ...
```

```python
# mamba_dvc/viz/diagnostics.py

DEFAULT_STATUS_COLORS: Mapping[POIStatus, str] = MappingProxyType({
    POIStatus.OK:           "#2ca02c",   # green
    POIStatus.MASKED:       "#7f7f7f",   # gray
    POIStatus.OUTLIER:      "#d62728",   # red
    POIStatus.LOW_CONF:     "#ff7f0e",   # orange
    POIStatus.OUT_OF_RANGE: "#e377c2",   # magenta
})

def render_status(
    plotter: pv.Plotter,
    field: DisplacementField,
    *,
    color_map: Mapping[POIStatus, str] | None = None,    # None -> DEFAULT_STATUS_COLORS
    point_size: float = 8.0,
) -> pv.Plotter: ...

def render_confidence(
    plotter: pv.Plotter,
    field: DisplacementField,
    *,
    cmap: str = "magma",
) -> pv.Plotter: ...

def highlight_outliers(
    plotter: pv.Plotter,
    field: DisplacementField,
    *,
    show_neighborhood: bool = True,
) -> pv.Plotter: ...
```

```python
# mamba_dvc/viz/validate.py

def render_error_glyphs(plotter: pv.Plotter,
                        report: ValidationReport, ...) -> pv.Plotter: ...
def render_error_histogram(report: ValidationReport, ...) -> Figure: ...
def render_error_vs_mask_distance(report: ValidationReport, ...) -> Figure: ...
```

```python
# mamba_dvc/viz/slices.py

def browse_slices(
    reference: Float32[np.ndarray, "z y x"],
    deformed: Float32[np.ndarray, "z y x"],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    field: DisplacementField | None = None,
    *,
    axis: Literal["z", "y", "x"] = "z",
) -> Any: ...    # ipywidgets HBox/VBox
```

Calling pattern:

```python
with viz.backend.plotter() as p:
    viz.volume.render_volume(p, reference, mask=mask)
    viz.field.render_displacement(p, field, color_by="magnitude")
    viz.diagnostics.highlight_outliers(p, field)
```

## 4. Type bridges

`_conversion.py` owns every numpy ↔ VTK conversion so the rest of the
module is pure-Python:

- `volume_to_image_data(volume, spacing) -> pv.ImageData` — wraps a
  C-contiguous float32 view zero-copy when possible. Honors
  `spacing` so anisotropic voxel sizes Just Work.
- `mask_to_image_data(mask, spacing) -> pv.ImageData` — uint8 cast,
  used as a transfer-function gate or a translucent overlay.
- `field_to_polydata(field, only_valid) -> pv.PolyData` — points =
  `positions`, vector array = `displacements`, scalar arrays =
  `valid`, `status`, `confidence`. One conversion feeds every
  glyph / status / confidence renderer.
- `grid_to_structured(grid, scalars=None) -> pv.StructuredGrid` —
  for scalar overlays on the regular POI lattice (e.g. confidence
  as a colored grid).
- `unify_spacing(*specs: VoxelSpacing) -> VoxelSpacing` — converts a
  set of input specs into one canonical unit (default `"mm"`) or
  raises `ValueError` on a non-convertible mix. Single chokepoint
  for unit math; renderers never multiply by 1e-3 themselves.

### 4.1 Voxel spacing & physical units

The user concern: hardcoding a unit ("um") into a variable name leaks
that assumption into every renderer and breaks down if datasets arrive
in mixed units. Resolution:

```python
# mamba_dvc/types.py  (new dataclass, lands before viz/ does)

PhysicalUnit = Literal["voxel", "nm", "um", "mm"]

@dataclass(frozen=True)
class VoxelSpacing:
    """Per-axis spacing with an explicit physical unit.

    Carried alongside volumes (and emitted by ``io/volume.py`` once it
    lands). The numeric ``values`` are unitless; ``unit`` names the
    physical interpretation. Default ``unit="voxel"`` keeps anything
    that does not care unit-agnostic.
    """

    values: tuple[float, float, float]    # (z, y, x), strictly positive
    unit: PhysicalUnit = "voxel"
```

Rules:

1. **Single-source-of-truth = IO layer.** `io/volume.py` (when it
   lands) emits a `VoxelSpacing` next to every volume. Visualizers
   never re-interpret raw numbers.
2. **Renderers accept `VoxelSpacing | tuple | None`.** A bare tuple
   is treated as `unit="voxel"` (back-compat for synthetic /
   test code). `None` means `"voxel"` with `(1, 1, 1)` values.
   `_conversion.py` normalizes at the boundary.
3. **Axis labels.** Renderers label axes with the unit string. The
   `Plotter` shows `"z [um]"` etc.; for unit-less data, just `"z"`.
4. **Mixed-unit figures.** When a single plot composes objects from
   different sources (e.g. recovered field from one dataset, GT from
   another), call `unify_spacing(*specs)` first. It converts to a
   canonical unit (`"mm"` by default) and raises if any input has
   `unit="voxel"` mixed with a physical unit — that combination is
   meaningless and should be loud, not silent.
5. **Per-axis differing units** (e.g. nm in xy, µm in z — rare in
   modern formats but seen in old TIFF stacks). Deferred. If it ever
   shows up, extend `unit` to `PhysicalUnit | tuple[PhysicalUnit,
   PhysicalUnit, PhysicalUnit]`. The dataclass shape lets us extend
   without breaking signatures.
6. **What the µCT pipeline emits today.** Synchrotron µCT voxels
   are isotropic in the data the project targets, so `VoxelSpacing`
   in v1 will almost always carry equal values. The dataclass earns
   its keep when we move beyond synthetic / single-dataset work.

`spacing` is plumbed through every public entry point. The default
of `None` lets pre-IO synthetic tests stay terse; the dataclass kicks
in once real data arrives.

## 5. Connections to landed pipeline / validation code

| Visualizer | Consumes | Producer (already landed) |
|---|---|---|
| `render_volume` | `Float32[ndarray, "z y x"]`, optional mask | tests, `validate/synthetic.make_pair` |
| `render_isosurface` | same | same |
| `render_displacement` | `DisplacementField` | `pipeline.correlate.correlate()` |
| `render_field_function` | `DisplacementFunction`, `GridSpec` | `validate/synthetic.{rigid_shift, uniform_dilation, simple_shear, sinusoidal, compose}` + `core.grid.build_grid` |
| `render_status` | `DisplacementField.status` (`POIStatus`) | `pipeline.correlate.correlate()` (sets `MASKED`, `OUT_OF_RANGE`, `OUTLIER`) |
| `render_confidence` | `DisplacementField.confidence` | `core.ncc.peak_displacement` via `correlate()` |
| `highlight_outliers` | `DisplacementField.status == OUTLIER` + `grid_shape`/`spacing` | `core.outlier.detect_outliers` via `correlate()` |
| `browse_slices` | volumes + optional `DisplacementField` | any pair |

Two non-trivial ergonomics:

1. `DisplacementField.displacements.reshape(*field.grid_shape, 3)`
   recovers the regular lattice for streamline / volumetric overlays
   without needing densification. The redundancy that `overview.md`
   §5 calls out for densification pays off here unchanged.
2. `POIStatus.OK == 0` means a default-zero status array renders as
   "all good" with no special casing in `render_status`. Keep the
   colormap explicit so failure modes are individually
   distinguishable; do not collapse all non-OK states into "bad".

## 6. Hook points to not-yet-landed types

Each item is an explicit seam. The viz module ships with the seam
named so v2 work is additive, not refactoring.

### 6.1 `densify(field, target_shape) -> Float32[ndarray, "z y x 3"]`
- Source: `overview.md` §10.
- Blocks: `field.render_streamlines`, any per-voxel field coloring,
  warp-residual heatmaps.
- Mitigation: ship `render_streamlines` as a `NotImplementedError`
  stub with the v2 hook in its docstring. Glyph renderers do not
  depend on it.

### 6.2 `ValidationReport` dataclass
- Not yet defined anywhere; `validate/known_fields.py` is also
  not landed. The plan in `overview.md` §9.2 names the stats
  but no carrier type exists.
- Proposal: add to `types.py` **before** writing any of `viz/validate.py`
  so both `validate.known_fields` and `viz.validate` target the same
  shape:

```python
@dataclass(frozen=True)
class ValidationReport:
    estimated:     DisplacementField
    truth:         Float32[np.ndarray, "points 3"]   # GT at POI centers
    error:         Float32[np.ndarray, "points 3"]   # estimated - truth
    mask_distance: Float32[np.ndarray, "points"] | None  # voxels to mask boundary
    summary:       Mapping[str, float]               # MAE, RMSE, p95, etc.
```

- Then `viz.validate.*` consumes only `ValidationReport`. The
  comparator can land later without forcing a viz refactor.

### 6.3 Time-series driver
- Source: `overview.md` §10.
- Blocks: animated playback of a sequence of `DisplacementField`s.
- Mitigation: keep every per-frame renderer stateless. The animator
  is then `render_field_animation(plotter, fields: Sequence[...])` —
  adds no per-frame API surface.

### 6.4 IO (`io/volume.py`, `io/mask.py`, `io/field.py`)
- All not yet landed.
- Mitigation: viz takes raw arrays today. When IO lands, add
  thin path-overload variants that call `io.read_*` then dispatch.
  Do **not** bake path handling into the array-first entry points.
- IO is also the producer of `VoxelSpacing` (see §4.1). Until IO
  lands, viz callers either pass `None` (treated as voxel units)
  or construct a `VoxelSpacing` by hand from dataset metadata.

### 6.5 Per-frame deformed mask (v2)
- Already in `correlate()`'s signature today.
- Volume rendering can already accept two masks; no API change
  needed when the v2 branch activates.

### 6.6 GPU residency
- `correlate()` returns host arrays. VTK does not consume CuPy.
- If `gpu/dispatch.py` ever returns device arrays, gather to host
  before rendering. Keep this rule in `_conversion.py` as a hard
  precondition (`asarray(np.asarray(...))` with a dtype check, not
  a silent transfer).

## 7. Non-negotiables

- **No new types in `viz/`.** Every visualizer consumes existing
  dataclasses (`GridSpec`, `DisplacementField`, `SyntheticPair`,
  `ValidationReport`, `VoxelSpacing`) or raw arrays.
- **Renderers return the `Plotter` / `Figure`**, never call `.show()`
  themselves. Composition and headless export both work.
- **`viz` is optional.** Pulled in via the `viz` extra; never imported
  by `core/`, `pipeline/`, `gpu/`, `io/`.
- **Pure-host arrays in.** Device arrays are a caller error.
- **Unit math lives in one place.** Only `_conversion.unify_spacing`
  multiplies by powers of ten; every other module passes
  `VoxelSpacing` through opaquely.
- **Status colors are configurable.** `DEFAULT_STATUS_COLORS` is the
  exposed default; every renderer that draws by status accepts a
  `color_map` override. No private color tables.

## 8. Validation & testing

- Headless rendering tests run under `pytest` using `off_screen=True`
  and image diffing against committed reference PNGs. Mark `slow`
  per `pyproject.toml`'s existing marker — these are not gpu-only.
- Conversion functions in `_conversion.py` are unit-tested without
  VTK (shape, dtype, scalar arrays present); rendering smoke tests
  exercise the VTK path once.
- Synthetic pairs from `validate/synthetic.py` drive end-to-end
  visualization tests: build a `SyntheticPair`, run `correlate()`,
  render volume + recovered field + ground-truth field on the same
  plotter, screenshot, diff. This doubles as an integration probe
  on the correlator output structure.

## 9. Implementation order

1. Add `viz` extra to `pyproject.toml`; confirm `pyvista`, `trame`,
   `matplotlib` resolve under Python 3.14.
2. **Define `VoxelSpacing` in `types.py`** (blocks renderers that
   accept spacing; tiny, lands first).
3. `_conversion.py` + unit tests (no VTK rendering yet — exercises
   `unify_spacing` and the spacing adapter).
4. `volume.py::render_volume` and `backend.py::plotter` — Jupyter
   backend default; minimum viable: render a `make_texture` cube in
   a notebook.
5. `field.py::render_displacement` + `render_field_function` —
   side-by-side recovered vs. analytical on a synthetic pair.
6. `diagnostics.py` (status, confidence, outliers) — exposes
   `DEFAULT_STATUS_COLORS`.
7. `slices.py` 2D browser — cheap because matplotlib only.
8. **Define `ValidationReport` in `types.py`** (blocks step 9).
9. `validate.py` — only after the report dataclass lands.
10. `export.py` (`save_screenshot`, `save_animation`).
11. Headless image-diff tests; CI gating decision (probably opt-in
    via `slow` marker, not blocking the default test run).
12. Qt backend smoke test in `backend.py` — only if (4) shows
    notebook-side stutter on full-resolution volumes.

`render_streamlines` ships as a stub at any point and lights up
when `densify` lands.

## 10. Decisions log

These were open questions in the previous draft; now resolved.

### 10.1 Frontend priority — Jupyter first; Qt as escape hatch
- v1 default: `trame`/Jupyter backend, configured in
  `backend.plotter()`. Notebook-driven exploration is the primary
  workflow (consistent with the existing `reactive-viz` extra).
- Headless `off_screen=True` reuses the same Plotter for figure
  export and image-diff tests.
- Qt path is reserved for the case where the trame backend stutters
  on the full `(960, 1280, 1280)` volume during interaction. Same
  renderer code; only `backend.py` changes. Decision to enable Qt is
  measurement-driven, not pre-emptive (step 12 in §9).

### 10.2 Voxel spacing & physical units — `VoxelSpacing` dataclass
- See §4.1 for the full design.
- Headline: a `VoxelSpacing(values, unit)` dataclass in `types.py`,
  emitted by `io/volume.py`, normalized at the `_conversion.py`
  boundary, with `unify_spacing(*specs)` as the only place unit
  arithmetic happens.
- `unit="voxel"` is the default for unit-agnostic / synthetic data;
  physical units (`"nm" | "um" | "mm"`) flow in from IO.
- Mixed-unit composition is a hard error rather than a silent
  conversion when the mix is non-convertible (e.g. voxel + um).
- Per-axis differing units deferred — extension is non-breaking.

### 10.3 `POIStatus` color conventions — locked defaults, configurable
- `viz/diagnostics.py` exports `DEFAULT_STATUS_COLORS` (see §3) as a
  read-only mapping. The defaults are:
  `OK=green`, `MASKED=gray`, `OUTLIER=red`, `LOW_CONF=orange`,
  `OUT_OF_RANGE=magenta`.
- Every renderer that draws by status takes a `color_map: Mapping[
  POIStatus, str] | None = None` parameter; `None` falls back to
  `DEFAULT_STATUS_COLORS`. Callers override per-figure as needed
  (e.g. for accessibility palettes or print-safe schemes).
- Project-wide consistency comes from the default; flexibility comes
  from the parameter. Both are non-negotiables in §7.
