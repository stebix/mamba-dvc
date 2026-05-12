# Campaign performance — where the wall time goes, and how to get it back

Findings and recommendations from profiling the `results_v1` overnight
campaign (3 transduced stores × their synthetic + real deformation
entries × 4-point `mask_threshold` sweep = 68 variant runs, host
`RECO-4xA6000`, git `231319b`).

This is an analysis + prioritized recommendation document, not a
contract. The one item with a worked-out implementation plan is the
boundary-distance KDTree swap — see
[`kdtree-impl-plan.md`](kdtree-impl-plan.md).

## 1. Measured budget

From the per-run `wall_correlate_s` / `wall_evaluate_s` in the sidecar
JSONs, summed over all 68 runs:

| phase | total | per pair (typical) | parallelism today |
|---|---|---|---|
| correlate (FFT NCC) | 2.24 h | ~100 s (range 72–159) | 4× A6000, POIs sharded within a pair |
| evaluate (EDT + GT scoring) | 8.03 h | ~460 s on a scored synthetic pair | **single-process CPU** |
| **total compute** | **10.28 h** | | |
| observed wall clock | **~11.8 h** | (job timestamps 2026-05-11 23:53 → 2026-05-12 11:41 UTC) | effectively serial |

The campaign ran **essentially serially**: 10.3 h of compute + ~1.5 h of
store-open / zarr-read / npz-write overhead ≈ the 11.8 h wall. Multi-GPU
buys almost nothing as things stand, because the four A6000s are used
*within* a single pair's NCC, not across pairs, and the dominant cost —
the synthetic evaluation — is CPU and single-process.

## 2. The core observation

**Mask foreground fraction is 0.1–0.2, and the sparsity is already
exploited in the part that is *not* the bottleneck, and ignored in the
parts that are.**

- `pipeline.correlate` filters POIs through `filter_by_mask` *before*
  any NCC (`mamba_dvc/pipeline/correlate.py:215`). At `mask_threshold
  0.9` only ~3 % of the 11 875-point grid is admitted, so NCC runs on
  only ~few hundred 96³ windows. The ~100 s/pair is therefore mostly
  CPU-side load + the 6.3 GB integral-image build + ~30 GB pair
  materialization + H2D upload — *not* GPU compute. Sparsity already
  applied here. ✔
- The boundary-stratified error table calls
  `scipy.ndimage.distance_transform_edt(~mask)` over the **full
  960×1280×1280 ≈ 1.6e9-voxel volume**, single-threaded, and does so
  **once per variant** — 4× per synthetic pair — even though the mask,
  hence the EDT, is byte-identical across the `mask_threshold` sweep.
  That one line in `validate/known_fields.py` is ~7 of the ~8
  evaluate-hours. Sparsity completely ignored here. ✘
- Volume I/O uploads and FFTs the whole volume, including the 80–90 %
  background. Sparsity ignored here too. ✘

So the speedup the foreground fraction "should" buy is real — it just
has to be applied to the EDT and to I/O, which is where it currently
isn't.

## 3. Recommendations, prioritized

### R1 — Boundary distance: kill the full-volume EDT, compute once per pair

`validate/known_fields.py` `_stratify_by_distance` → replace
`distance_transform_edt(~mask)` with a **surface KDTree**: the nearest
background voxel to any foreground POI lies on the thin background
boundary shell `S = (~mask) & binary_dilation(mask, 3×3×3)` (~10⁶–10⁷
voxels with 10–20 % foreground), so `cKDTree(argwhere(S))` queried at
the ~hundreds of POI coords gives the EDT value *exactly* (with a
one-line `mask`-lookup for the rare POI whose center rounds onto
background, where the EDT is 0 by definition). Build the index **once
per materialized pair** (same scope as the existing `truth_cache` in
`run/batch.py`), reuse across the 4 variants.

Effect: per scored pair ~460 s → ~seconds; the evaluate phase goes
from ~8 h to ~minutes; campaign → ~3 h, then gated by I/O. Worked plan:
[`kdtree-impl-plan.md`](kdtree-impl-plan.md).

Alternative considered — GPU EDT (`cupyx.scipy.ndimage.distance_transform_edt`,
or `cucim`): also lands at seconds/pair and the 4 GPUs are idle during
evaluate, but it pulls in a ~25 GB device-side feature-transform
footprint, a CuPy-version coupling, and an upload of the 6.3 GB mask —
all avoided by the CPU KDTree. Pick one, not both; the KDTree is the
lower-risk choice for this bottleneck. GPU EDT only wins if the *dense*
distance field is wanted for something else (visualization, finer
stratification).

### R2 — Make the `mask_threshold` sweep share work instead of redoing the pipeline 4×

Each `mt*` variant currently re-runs *everything* — correlate,
GT-sampling, EDT, scoring — when the only thing that differs is which
POIs get labelled `MASKED`, and `filter_by_mask` is monotone in the
threshold (`admitted@0.9 ⊂ admitted@0.7 ⊂ … ⊂ admitted@0.3`). So:
correlate once on the loosest admitted set; compute the boundary index
and the GT sample once (the latter is already cached via `truth_cache`,
keyed by `_grid_key` — extend the same idea to correlation); then
derive all four variant summaries by re-applying the threshold filter +
**outlier test** + scoring on subsets. The outlier test result *does*
change with the admitted set (neighbourhoods differ), so re-run that
per threshold — but it's cheap (CPU, few hundred points). The 6.3 GB
`uint32` integral image rebuilt per variant in `core/grid.py` falls out
of this too. Roughly another ~4× on the whole campaign, on top of R1.

### R3 — Cut the per-pair data movement (~1.5 h overhead + a chunk of the 100 s correlate)

Per pair: ~12.6 GB read from `D:` (reference + deformed) → ~30 GB
materialized → ~50 GB of H2D traffic replicating to 4 GPUs — and the
*reference* volume is identical across all deformations of a store and
across the 4-variant sweep, yet `run/batch.py:367` does
`del pair; gc.collect()` and reloads it every time.

- Group jobs by store; keep the reference volume resident on the GPUs
  across that store's deformations.
- Confirm the pinned-host-buffer path in `gpu/` is actually wired into
  the H2D copies (it was designed for this — `overview.md` §… / the
  `gpu/memory` notes).
- **Crop to the foreground bounding box** before upload / FFT. Bone +
  screw is localized, so the bbox is much smaller than 960×1280×1280;
  shrinks bytes moved, the integral-image build, *and* the FFT working
  set. Store the offset to map POI coords back. (This is the second
  "exploit the 0.1–0.2 fraction" lever, after R1.)
- Overlapped zarr reads via `tensorstore` (already a dep) so the next
  pair loads while the current one correlates. **GPU Direct Storage**
  (cuFile) would read zarr straight into device memory and skip the
  host bounce, but it is Linux-only — not viable on this Windows box.

### R4 — Pipeline the phases / parallelize across pairs

Today the evaluate phase is single-process CPU while the 4 GPUs idle,
and correlate is GPU-bound while the CPUs idle. Run the next pair's
correlate concurrently with the current pair's evaluate (a two-stage
pipeline), or fan independent pairs' EDT/scoring across CPU cores.
Matters less once R1 lands (evaluate stops being expensive); worth
~2× on its own with the current EDT.

### R5 — Fix the degenerate `distance_bins` default (minor, correctness-adjacent)

At `mask_threshold 0.9` every admitted POI lands in the `[0, 5)` bin —
the foreground is shell-like, so there are no OK windows deeper than
~5 vox into it; the table only resolves anything at `mt0.3`. Either
rescale the edges to the per-mask EDT range, or only build the table for
the looser thresholds — which, if R1 is *not* done, also dodges the EDT
on the strict variants. Orthogonal to the rest.

## 4. Expected trajectory

| state | campaign wall | gated by |
|---|---|---|
| today | ~11.8 h | the per-variant full-volume EDT |
| + R1 (KDTree, computed once/pair) | ~3–4 h | I/O + correlate |
| + R2 (share correlate/GT across sweep) | ~1–1.5 h | I/O + ~14 correlations |
| + R3/R4 (I/O caching, bbox, phase overlap) | well under 1 h | residual I/O |

Recommended order: **R1 first** (self-contained, low risk, removes the
bottleneck), then R2, then R3/R4.

## 5. Out of scope here

Algorithmic accuracy (the `fs004` flow field is broken across all three
stores — ~11–16 vox MAE, worsens as the threshold loosens, signature of
a displacement amplitude exceeding `search_radius = window // 2`; that
is a data/GT-generation question, separate from runtime). The `fs104 /
204 / 304 / 402` fields are sub-voxel-to-~1.4-vox at `mt0.9` and the
sweep confirms `0.9` as the right operating point.
