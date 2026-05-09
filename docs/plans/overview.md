# mamba-dvc — Architecture Plan

Python-based FFT digital volume correlation for synchrotron µCT of rat bone with screw implant. Target volume shape `(960, 1280, 1280)` float32, run on a 4× NVIDIA A6000 workstation.

## 0. Project context

**Data**
- Modality: synchrotron µCT
- Specimen: rat bone with rigid screw implant
- Volume shape: `(960, 1280, 1280)` float32, ~6.29 GB per volume
- Inputs per sample: reference volume, deformed volume, binary screw mask (shared across samples), ground-truth displacement/flow field used to synthesize the deformed volume
- Expected displacement: ±10 voxels typical, ±20 voxels cap
- Accuracy target: ~0.1 voxel (Gaussian subvoxel fit is sufficient)

**Hardware**
- 4× NVIDIA A6000, 49 GB VRAM each (196 GB total GPU memory)
- 128 GB system RAM
- Windows host, bash-style shell

**Python / runtime**
- Python 3.14 (verified supported by CuPy v14 and PyTorch ≥ 2.10)
- Primary compute: CuPy 14 on CUDA 13.x wheels
- No Python downgrade needed

## 1. Scope

### v1 — build now
- Single-pair DVC: `(reference, deformed, mask) → sparse displacement field at POI grid`
- Single-pass FFT normalized cross-correlation (NCC), no iterative warp loop, no pyramid
- Gaussian 3D peak subvoxel refinement
- Mask-aware POI filtering, preprocessing, and outlier detection
- Multi-GPU POI sharding across 4 A6000s
- Validation against both synthetic analytical fields and the provided experimental ground-truth fields

### v2 — design extension points for, do not build now
- IC-GN subvoxel registration (targets ~0.01 voxel accuracy — Wang et al. 2016)
- Padfield masked FFT cross-correlation (exact masked NCC — Padfield 2010)
- Iterative deformation-warping loop (FIDVC) for larger displacements / strain regimes
- Coarse-to-fine multi-level pyramid
- Dense per-voxel field densification (B-spline / thin-plate from sparse POIs)
- Time-series driver (pair sequencing, prefetch, warm-start)

## 2. Algorithm (literature-grounded, v1)

The v1 algorithm is a stripped-down descendant of **FIDVC** (Bar-Kochba et al. 2014) and the FFT-CC component of Wang et al. (2016). The iterative warp loop and IC-GN stage — FIDVC's distinguishing contributions — are not needed at our displacement and accuracy targets.

### Why no iteration, no pyramid

- Max displacement ±20 voxels. Correlation SNR degrades as the overlapping sub-region `(W − |d|)³` shrinks. Rule of thumb: keep `|d| ≤ W/3`, so `W ≥ 60` suffices; we will pick `W = 96` for comfort. At `W = 96` the overlap fraction at max displacement is `(76/96)³ ≈ 0.50` — healthy.
- No aliasing risk because lag is well below `W/2`.
- Gaussian peak fit reliably reaches ~0.05 voxel on well-textured subvolumes — below the 0.1 voxel target.
- Therefore: a single forward pass of FFT NCC suffices. Iteration buys nothing.

### Pipeline (single pass)

1. **Grid construction** (`core/grid.py`) — emit POI centers on a regular lattice over the reference. Mask each candidate POI; mark invalid if the POI's subvolume has `< mask_threshold` (default 0.9) valid-voxel fraction.
2. **Subvolume extraction** — gather ref/deformed subvolumes for all valid POIs into batched GPU tensors.
3. **Preprocessing** (`core/window.py`) — per subvolume:
   - Mean-subtract over unmasked voxels only (avoid bias from zeros inside the screw region).
   - Apply 3D Tukey window (α ≈ 0.25) to suppress spectral leakage.
   - Zero out masked voxels after windowing.
4. **Batched FFT NCC** (`core/ncc.py`) — two kernels live side-by-side; the pipeline driver picks one. Default is **linear NCC**: zero-pad each subvolume to `2W` along every spatial axis, batched 3D `rfftn` of both stacks, conjugate multiply, `irfftn` at padded size, then crop the central `W` block of valid lags and repack into cyclic-FFT layout. Normalize per lag with the **Lewis (1995) overlap-aware denominator** `sqrt(S_ref(k) · S_def(k))` where `S(k)` is the sum of squared signal over the in-bounds intersection at lag `k` (computed via two extra FFT-based correlations of `ref²` and `def²` against an indicator box, sharing one box FFT per batch). The legacy **cyclic NCC** (no zero-pad, single whole-window L2 normalization, plus a Tukey α=0.25 window in preprocessing to suppress wrap-induced leakage) is retained for A/B comparison; the rule of thumb `|d| ≤ W/3` survives in both kernels but for SNR reasons in linear mode (no aliasing argument) and for combined SNR + bias reasons in cyclic mode. The cyclic kernel exhibits a shrinkage bias toward zero whose magnitude scales with `|u|/W`; the linear+overlap kernel does not. See `docs/insights/error-minimization.md` for the diagnostic chain that pinned the bias and the plan revisions it drove.
5. **Peak localization** — `argmax` on each correlation volume; lag is in cyclic-FFT layout (lag 0 at index 0; positive lags in `[0, W/2)`; negative lags in `[W/2, W)`) for both kernels, so this step is mode-agnostic.
6. **Gaussian subvoxel fit** (`core/peakfit.py`) — closed-form log-space parabolic fit on the 3×3×3 neighborhood of the integer peak, along each axis independently (separable 3D Gaussian). Returns fractional `(dz, dy, dx)` plus a `confidence` score (peak NCC value).
7. **Outlier rejection** (`core/outlier.py`) — normalized median test (Westerweel & Scarano) on a 3×3×3 neighborhood in the POI grid; skip masked-out neighbors so the boundary isn't flagged. Failed POIs set `valid=False` but stay in output.
8. **Emit `DisplacementField`** (`types.py`) — positions, displacements, valid flags, confidence, grid metadata.

No step warps, no step iterates. If correlation residuals reveal a region exceeding ±20 voxels, flag that as a v2 trigger — don't silently diverge.

### Window and grid sizing

| Parameter | Default | Rationale |
|---|---|---|
| `window` | 96 (cube) | `(W − d_max)/W = 0.79`; correlation SNR healthy |
| `overlap` | 0.5 (stride 48) | ~13 k POIs; sufficient spatial resolution |
| `search_radius` | W/2 = 48 | Circular NCC, unambiguous up to ±48 |
| `mask_threshold` | 0.9 | POI needs ≥ 90% valid voxels |
| `tukey_alpha` | 0.25 | Conventional for DIC/DVC FFT work |

Grid count at defaults: `floor((960 − 96)/48)+1 = 19`, `floor((1280 − 96)/48)+1 = 25` → **19 × 25 × 25 ≈ 11 875 POIs**.

An optional config switch for `window = 128` exists for tough regions (bigger subvolume, fewer POIs, same pipeline).

## 3. Mask handling (v1)

**One shared binary mask** (`1 = valid bone tissue`, `0 = screw / low-contrast / exclude`) covering all samples. API still accepts an optional second mask for the deformed volume to keep v2 options open (`deformed_mask=None` falls back to `mask`).

### Mask is consulted at four points

1. **POI admission** — `grid.filter_by_mask(mask, threshold)` drops subvolumes with too much invalid overlap. Emits a `(N,) bool` valid array so the output grid shape is preserved.
2. **Subvolume preprocessing** — statistics computed over valid voxels only; masked voxels zeroed post-window. This introduces a small bias for POIs near the boundary (mask fraction 0.9–1.0), acceptable at our accuracy target. The principled fix is Padfield's masked FFT (noted as v2).
3. **Outlier neighborhood** — normalized median test excludes masked neighbors so POIs adjacent to the screw aren't systematically flagged.
4. **Output annotation** — each POI carries its admission reason: `valid`, `masked`, `outlier`, `low_confidence`.

### Upgrade path (v2)
- Swap `core/ncc.py` implementation for Padfield (2010): 6 FFTs, exact masked NCC. Same API, no pipeline change.
- Accept deformed-frame masks (already in the API signature; just activate the branch).

## 4. Architecture

```
mamba_dvc/
  io/                # see docs/plans/zarr-interface.md for the full contract
    volume.py        # zarr → contiguous host buffer; centered dry_shape
    mask.py          # bool-coerced mask loader with shape validation
    field.py         # GroundTruthField: callable wrapper around stored flow
    profiles.py      # named StoreProfile registry (bone_screw_synchrotron_v1)
    manifest.py      # optional StoreManifest: per-scan overrides + subsetting
    verify.py        # structural verifier (dtypes, shapes, axis-order)
    dataset.py       # DvcDataset.open(...).load_pair(...) → EvaluationPair
  core/
    grid.py        # POI lattice, mask-aware admission
    window.py      # Tukey/Hann, masked mean-subtract, masked zeroing
    ncc.py         # batched FFT NCC (CuPy); normalization over valid support
    peakfit.py     # 3D separable Gaussian subvoxel fit (swappable interface)
    outlier.py     # normalized median test, mask-aware neighborhood
  pipeline/
    correlate.py   # main single-pair driver (pure function)
  gpu/
    dispatch.py    # multi-GPU POI sharding (one process per GPU)
    memory.py      # pinned host buffers, async H2D/D2H, CUDA stream setup
  types.py         # DisplacementField, GridSpec, MaskSpec dataclasses
  validate/
    synthetic.py   # analytical fields (rigid shift, dilation, shear, sinusoid)
    known_fields.py # comparator for experimental ground-truth pairs
  cli.py           # thin entry point
```

**Data flow (single pair, single GPU):**
`load ref, def, mask → build POI grid (masked) → extract subvolume batches → preprocess (windowed, masked-mean-subtracted) → batched rfftn → conjugate multiply → batched irfftn → peak + Gaussian fit → outlier test → DisplacementField`

**Pure-core rule:** `pipeline.correlate.correlate()` is a pure function with no hidden state. All sequencing, prefetch, warm-start (for eventual time series) lives in a driver layer that wraps the pure core. This means v1's single-pair code is v2's inner loop without refactoring.

## 5. Core API

Array shapes are encoded in the type system via [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping). Shape symbols (`Z`, `Y`, `X`, `N`) are consistent across the codebase and documented once in `types.py`. Runtime checking is opt-in via `jaxtyped`/`beartype` in tests; production code uses the annotations for static review by `pyright` and for documentation.

```python
# mamba_dvc/pipeline/correlate.py — single-device pure function

from jaxtyping import Bool, Float32

def correlate(
    reference: Float32[np.ndarray, "Z Y X"],
    deformed: Float32[np.ndarray, "Z Y X"],
    mask: Bool[np.ndarray, "Z Y X"] | None = None,
    deformed_mask: Bool[np.ndarray, "Z Y X"] | None = None,  # falls back to `mask`
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float | None = None,           # per-mode default (linear=0.0, cyclic=0.25)
    search_radius: int | None = None,           # default: min(window) // 2
    batch_size: int = 256,
    eps: float = 1e-12,
    ncc_mode: Literal["linear", "cyclic"] = "linear",
    ncc_normalization: Literal["overlap", "global"] = "overlap",
) -> DisplacementField: ...
```

```python
# mamba_dvc/gpu/dispatch.py — multi-GPU sibling

def correlate_multi_gpu(
    reference: Float32[np.ndarray, "Z Y X"],
    deformed: Float32[np.ndarray, "Z Y X"],
    mask: Bool[np.ndarray, "Z Y X"] | None = None,
    deformed_mask: Bool[np.ndarray, "Z Y X"] | None = None,
    *,
    device_ids: Sequence[int] | None = None,    # None → all visible GPUs
    # ...remaining args identical to correlate(); batch_size defaults to 64...
) -> DisplacementField: ...
```

`correlate()` is the pure single-device path required by the §4
pure-core rule. `correlate_multi_gpu()` lives in `gpu/dispatch.py` and
shells out to one CuPy worker per device, sharing the same algorithmic
core via the private helper `pipeline.correlate._correlate_admitted_subset`.
See `docs/plans/dispatch.md` for the dispatch contract and worker
lifecycle.

```python
# mamba_dvc/types.py

from jaxtyping import Bool, Float32, UInt8

@dataclass(frozen=True)
class DisplacementField:
    positions:     Float32[np.ndarray, "N 3"]  # POI centers in voxel coords (z, y, x)
    displacements: Float32[np.ndarray, "N 3"]  # voxel displacement (dz, dy, dx)
    valid:         Bool[np.ndarray, "N"]       # True = usable estimate
    confidence:    Float32[np.ndarray, "N"]    # peak NCC value
    status:        UInt8[np.ndarray, "N"]      # enum: OK | MASKED | OUTLIER | LOW_CONF
    grid_shape:    tuple[int, int, int]        # (nz, ny, nx) POIs along each axis
    spacing:       tuple[int, int, int]        # grid stride in voxels
    window:        tuple[int, int, int]        # subvolume size used
```

All public functions in `core/` and `pipeline/` use `jaxtyping` annotations on array inputs and outputs; scalar parameters stay as plain Python types. Internal helpers may skip annotations when the shape is trivial.

`grid_shape` + `spacing` + `positions` are redundant by design — they make v2 densification (reshape to regular grid → B-spline) a one-liner.

## 6. Memory & multi-GPU strategy

### Per-GPU budget (A6000, 49 GB)

The cyclic kernel is the cheap reference; the linear kernel pays ~8× spectral footprint per POI because every FFT runs at `2W` per axis.

**Cyclic NCC (legacy / A-side comparison)** at `W=96`, batch 256:

| Item | Size |
|---|---|
| Reference volume (float32) | 6.29 GB |
| Deformed volume (float32) | 6.29 GB |
| Mask (bool) | 1.57 GB |
| Subvolume batch — ref + def (float32, 96³, batch 256) | 2 × 216 MB = 0.43 GB |
| FFT forward (complex64, 96³, batch 256) | 2 × 418 MB = 0.84 GB |
| Product + inverse workspace | ~0.8 GB |
| cuFFT plan cache / misc | ~1 GB |
| **Total in flight** | **~17 GB** |

Comfortable fit. Batch 512 also viable (~20 GB).

**Linear NCC + overlap-aware (default)** at `W=96`: each `(2W)³` float32 buffer is 28 MB per POI, and the kernel materializes ref/def + their squared copies + four padded spectra during the overlap-aware path. At batch 256 the transient working set runs **>50 GB**, over budget on a single A6000. Drop `batch_size` to ~64 to fit (~14 GB transient, plus the resident volumes/mask). A future heuristic in `gpu/dispatch.py` will recommend a batch size from the resolved `(window, mode, normalization, free VRAM)` tuple programmatically; today the call site is responsible.

### Decomposition

- **Replicate** both volumes and the mask on each of the 4 GPUs.
- **Shard** POIs block-wise across GPUs (partition `positions` into 4 contiguous slabs along Z to preserve spatial locality for the outlier test).
- **One Python process per GPU**, spawned via `multiprocessing.spawn` (Windows-safe). Avoids GIL + simplifies CuPy context per process.
- **Final gather** via shared memmap or a `multiprocessing.Queue` of numpy arrays. Outlier test runs after gather on a single process (grid is small enough that the 3×3×3 neighborhood filter is trivially fast on CPU).
- **No inter-GPU comms during NCC.**

### Alternative considered

Spatial halo decomposition (each GPU owns a Z-slab of the volumes plus halo = search radius). Saves replication memory but adds bookkeeping. Not needed here — 17 GB/49 GB is not tight.

## 7. Performance estimate

Defaults: `W = 96`, ~12 k POIs, 4 GPUs → ~3 k POIs/GPU.

- At batch 256: ~12 batches/GPU.
- Per batch: 2 × `rfftn(256, 96, 96, 96)` + 1 × `irfftn(...)` on A6000 cuFFT ≈ tens of ms.
- Total compute per GPU: sub-second.
- Volume load (6.3 GB × 2) dominates cold start: ~15 s from NVMe, less if cached.
- **End-to-end: 15–30 s per pair including I/O; <5 s amortized in a loop.**

If we later move to `W = 128` with denser grid: still under a minute.

## 8. Library choices (locked for v1)

| Dep | Version | Role |
|---|---|---|
| `cupy-cuda13x` | `>=14.0` | Primary GPU array + batched cuFFT |
| `numpy` | `>=2.0` | Host arrays |
| `scipy` | `>=1.13` | CPU reference implementation for tests |
| `zarr` | `>=3.0` | Chunked volume I/O |
| `tensorstore` | `*` | Optional async reads / GDS |
| `tifffile` | `*` | µCT TIFF stack loader |
| `jaxtyping` | `>=0.2` | Shape-aware array annotations on public signatures |
| `beartype` | `>=0.19` | Runtime shape/type enforcement under tests (paired with `jaxtyped`) |
| `pytest` | `*` | Tests |
| `ruff` | `*` | Lint + format |
| `pyright` | `*` | Static type checking |
| `pre-commit` | `*` | Git hook runner (ruff format + check, pyright) |

**Not using:**
- PyTorch — not needed in v1 (no tricubic warp until v2)
- JAX — CuPy + cuFFT is a better fit for FFT-heavy workloads with explicit memory control

Will add to `pyproject.toml` as the first scaffolding commit.

## 9. Validation strategy

### Tier 1 — analytical unit tests (`validate/synthetic.py`)

Generate small reference volumes (e.g. 256³) with rich synthetic texture (filtered noise), apply a **known** displacement field, produce the deformed volume by warping (SciPy `map_coordinates`, cubic), run `correlate()`, assert error bounds.

Fields to cover:
- Rigid integer shift (e.g. `(3, −2, 5)`) — should recover exactly.
- Rigid fractional shift (e.g. `(0.7, −1.3, 2.5)`) — exercises Gaussian fit; error `< 0.1` voxel.
- Uniform dilation (strain 0.01) — tests spatially varying but smooth field.
- Simple shear — detects axis-mixing bugs.
- Sinusoidal field (low spatial frequency) — stress-tests grid density.

Each runs in seconds on one GPU; all are CI-eligible.

**Bias regression test.** `tests/core/test_ncc.py::TestLinearVsCyclicBias` asserts that the linear+overlap kernel recovers integer shifts exactly on textured volumes where the deformed window is a *non-rolled* slab from a larger field — the construction `docs/insights/error-minimization.md` pinned the cyclic shrinkage with. This is the cheapest possible test that would have caught the bias on day one; it must pass for any PR that touches `core/ncc.py`.

### Tier 2 — experimental ground truth (`validate/known_fields.py`)

Implemented; see `docs/plans/zarr-interface.md` for the reader + scoring layer
contract. Open a `DvcDataset`, call `load_pair("fs004")` to get an
`EvaluationPair`, run `correlate()`, then feed both into
`validate.known_fields.evaluate_pair(...)` which produces an `ErrorReport`
with:
- Per-POI error vector `estimated − ground_truth` (interpolate GT at POI centers).
- MAE, RMSE, 95th-percentile error overall.
- Same stats stratified by distance from the screw mask boundary (expect worse near the screw).
- Confidence-vs-error correlation — sanity check that `confidence` is a useful quality signal.

`run_and_evaluate` and `sweep` orchestrate the materialize → correlate → score
loop end-to-end. Not in CI (too big), run manually; drives the accept/reject
decision on the implementation.

## 10. Extensibility hooks (for v2; baked into v1 design)

| v2 feature | v1 hook that supports it |
|---|---|
| IC-GN subvoxel refinement | `peakfit.py` is called via `fit(corr_map, initial) → (displacement, confidence)`. IC-GN replaces the function; pipeline untouched. |
| Padfield masked FFT NCC | `ncc.py` exposes `correlate(ref_sv, def_sv, mask_sv) → corr_map`. Swap implementation. |
| Iterative warp loop | `pipeline.correlate.correlate()` is pure; wrap in an iterator that alternates it with a warper. |
| Multi-level pyramid | Pyramid driver calls `correlate()` with decreasing `window` and shifted POI grid; passes coarse field as `initial_guess`. |
| Dense field | `DisplacementField.grid_shape + spacing` makes reshape-to-lattice trivial; add `densify(field, target_shape) → ndarray` module. |
| Time series | Wrap `correlate()` in `series.py` with prefetch (load `t+1` while computing `t`) and optional warm-start (use previous field to set outlier priors). |
| Per-frame deformed masks | `deformed_mask` parameter already in API; activate the branch in `window.py` and `ncc.py`. |
| Centred correlation layout | `correlate_linear` currently re-packs to cyclic-FFT layout for drop-in compatibility with the cyclic kernel's downstream consumers. A v2 cleanup may switch to a centred layout (lag 0 at the volume centre) and update `peak_displacement` / `peakfit._gather_neighbors` to honour a `centered` flag — the natural layout for a linear correlator that has no wrap-around to encode. |

## 11. Code quality standards (locations)

| File | Content |
|---|---|
| `pyproject.toml` | `[tool.ruff]` (lint + format, `D` rules enabled, `pydocstyle.convention = "numpy"`), `[tool.pyright]` (strict mode on `mamba_dvc/`, `reportMissingTypeStubs = "warning"`), `[tool.pytest.ini_options]` |
| `.pre-commit-config.yaml` (repo root) | `ruff` (format + check with `--fix`), `pyright`, trailing-whitespace, end-of-file-fixer, large-file check, merge-conflict check |
| `docs/standards.md` | Human-readable conventions: naming, jaxtyping policy for public array signatures, NumPy docstring style, test expectations, commit message format |
| `CLAUDE.md` (repo root) | AI agent conventions: module layout, preferred libs, commands to run before claiming done (`ruff format`, `ruff check`, `pyright`, `pytest`) |

### Type and docstring conventions

- **Static checker:** `pyright` in strict mode on the `mamba_dvc/` package. Tests and `validate/` may relax to `basic` where `jaxtyping` string shapes would otherwise need many `# type: ignore` escapes.
- **Shape annotations:** every public function in `core/`, `pipeline/`, `io/`, and `gpu/` annotates array inputs/outputs with `jaxtyping`. Shape symbols (`Z`, `Y`, `X`, `N`, `W`) are shared across modules — document the vocabulary in `types.py`.
- **Runtime checks:** tests wrap target functions with `@jaxtyped(typechecker=beartype)` so shape mismatches fail loudly in CI without slowing production code.
- **Docstrings:** NumPy convention (`numpydoc`) for every public function, class, and module. Sections: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` as applicable. Ruff's `pydocstyle` (`D`) rules enforce style with `convention = "numpy"`.
- **Pre-commit:** `pre-commit install` is part of the dev setup. Hooks: `ruff-format`, `ruff check --fix`, `pyright`, plus standard hygiene hooks. CI runs `pre-commit run --all-files` as its lint gate.

Separate planning doc; not blocking the architecture. Draft can land alongside the first scaffolding commit.

## 12. Immediate next steps

1. Populate `pyproject.toml` with the v1 dependency set and tooling config.
2. Scaffold the module tree under `mamba_dvc/` (empty files + `__init__.py`).
3. Implement `validate/synthetic.py` first — it gives us a testbed before any real algorithm lands.
4. Implement `core/ncc.py` (batched FFT NCC on a single GPU) + `core/peakfit.py` (Gaussian fit) — smallest slice that produces a verifiable displacement on synthetic data.
5. Add `core/grid.py` + `core/window.py` + mask plumbing → single-GPU `pipeline.correlate.correlate()`.
6. Add `gpu/dispatch.py` for multi-GPU sharding.
7. Wire `validate/known_fields.py` against the experimental dataset.

## Sources

- Bar-Kochba E., Toyjanova J., Andrews E., Kim K., Franck C. (2014) *A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations*, Experimental Mechanics. <https://link.springer.com/article/10.1007/s11340-014-9874-2>
- FranckLab/FIDVC reference MATLAB implementation. <https://github.com/FranckLab/FIDVC>
- Wang T., Jiang Z., Kemao Q., Lin F., Soon S.H. (2016) *GPU Accelerated Digital Volume Correlation*, Experimental Mechanics. <https://link.springer.com/article/10.1007/s11340-015-0091-4>
- Wang T., Zhang L., Jiang Z., Qian K. (2017) *GPU Accelerated High Accuracy Digital Volume Correlation*. <https://link.springer.com/chapter/10.1007/978-3-319-51439-0_6>
- Yang et al. (2020) *3D SIFT aided path-independent digital volume correlation and its GPU acceleration*. <https://www.sciencedirect.com/science/article/abs/pii/S0143816620310368>
- Buljac A. et al. *Some recent advances in digital volume correlation* (review). <https://www.sciencedirect.com/science/article/abs/pii/S0143816619315982>
- Padfield D. (2010) *Masked Object Registration in the Fourier Domain*, IEEE Trans. Image Processing. (Reference implementation: `skimage.registration.phase_cross_correlation` with `reference_mask`)
- Westerweel J., Scarano F. (2005) *Universal outlier detection for PIV data*, Experiments in Fluids.
- *Phase correlation with sub-pixel accuracy: A comparative study in 1D and 2D*. <https://www.sciencedirect.com/science/article/abs/pii/S1077314215000685>
- Phase correlation (Wikipedia). <https://en.wikipedia.org/wiki/Phase_correlation>
- CuPy v14 release announcement (Python 3.14 support, Feb 2026). <https://medium.com/cupy-team/announcing-cupy-v14-e8515ec05fca>
- PyTorch 2.10 release blog (Python 3.14 support, Jan 2026). <https://pytorch.org/blog/pytorch-2-10-release-blog/>
