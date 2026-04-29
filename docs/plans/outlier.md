# Outlier rejection — plan & mathematical strategy

Step 7 of the single-pass DVC pipeline (`docs/plans/overview.md` §2,
last bullet of "Pipeline (single pass)"). Consumes the assembled
displacement grid emitted by `pipeline.correlate.correlate()` after the
`search_radius` clamp, flags POIs whose displacement is statistically
inconsistent with their lattice neighbors, and returns a per-POI
outlier mask. The orchestrator promotes flagged POIs to
`POIStatus.OUTLIER` before constructing the final `DisplacementField`.

## 1. Slice scope

This is the first implementation slice for outlier detection. It
ships:

1. `mamba_dvc/core/outlier.py` — one pure verb `detect_outliers` that
   computes the normalized-median-test flag mask on a regular POI
   lattice.
2. `tests/core/test_outlier.py` — unit tests against the
   Westerweel & Scarano reference formula on hand-built lattices,
   plus mask-awareness, boundary, and degenerate-input cases.
3. Wiring in `pipeline/correlate.py` between the `search_radius` gate
   (current step 6) and the `DisplacementField` construction. Flagged
   POIs receive `status = OUTLIER`, zero displacement, and zero
   confidence — same handling as `MASKED` and `OUT_OF_RANGE`.
4. Integration test in `tests/integration/test_pipeline_smoke.py`
   asserting that a single synthetic-injected bad POI is recovered as
   `OUTLIER` while its neighbors remain `OK`.

No multi-GPU dispatch, no iterative replacement, no driver-level
thresholding — those land in v2.

## 2. Position in the pipeline

```
… ncc ──► peakfit ──► search_radius gate ──► outlier ──► DisplacementField
                                              (THIS STEP)
```

The outlier test runs **on the assembled grid**, after every batch has
returned and the pure-core gather is complete. Plan §6 states it
explicitly: "the outlier test runs CPU-side on the assembled grid."
That placement is non-negotiable: the normalized median test reads a
3×3×3 neighborhood in lattice space, which only exists once every POI
has a known displacement. Sharding it per-batch would force halo
exchange.

Because the test is host-side and the assembled grid is small
(~12 k POIs at v1 defaults, ~50 k at the densest reasonable setting),
NumPy is the right backend. CuPy support is not in scope.

## 3. Design invariants

| Invariant | Rationale |
|---|---|
| Pure function, single verb | Same shape as every other `core/` module. The orchestrator owns sequencing; this module owns the test. |
| Lattice-aware, not point-cloud-aware | We require regular grid input. Plan §5's `DisplacementField.grid_shape + spacing` makes the reshape free. A scattered-point fallback (kd-tree neighborhood) would be useful for v2 dense fields, not now. |
| Mask-aware neighborhood | Plan §3 mandates that masked POIs are excluded from neighbor pools so POIs adjacent to the screw are not systematically flagged. The `valid` input encodes admission *and* search-radius gating, so the test naturally inherits both prior filters. |
| Failed POIs stay in the lattice | The output grid shape is preserved end-to-end (correlate.md §2). The verb returns a flag mask of length `points`; the orchestrator decides what to do with it. |
| Per-component test, OR combination | A POI is an outlier if any of `(dz, dy, dx)` fails the test. Cross-component coupling (Mahalanobis) is out of scope; the literature target metric is per-component. |
| Insufficient-neighborhood POIs are not flagged | If fewer than `min_neighbors` valid neighbors exist (corner of the grid abutting a mask cutout), the test cannot reliably classify the point. Default behavior: leave it `OK`. The boundary should not be more strictly filtered than the interior. |
| Westerweel & Scarano (2005) parameters as defaults | `epsilon = 2.0`, `noise_floor = 0.1` voxels. These are the canonical PIV/DVC values; they are documented in the docstring with the literature reference, so future tuning is grounded. |

## 4. Mathematical strategy: normalized median test

### 4.1 Notation

- `D ∈ ℝ^{N_z × N_y × N_x × 3}` — displacement lattice, reshaped from
  `DisplacementField.displacements`. Components index `(dz, dy, dx)`.
- `V ∈ {0, 1}^{N_z × N_y × N_x}` — boolean validity at the time of the
  test; `True` iff status is currently `OK` (i.e. POI passed admission
  and was inside the search radius).
- For each lattice index `p = (z, y, x)`, the neighborhood is the
  3×3×3 cube minus the center: `𝒩(p) = { p + δ : δ ∈ {−1,0,1}³, δ ≠ 0 }`,
  intersected with the lattice and with `V`.

### 4.2 Westerweel & Scarano's normalized residual

For each component `c ∈ {z, y, x}`, define

```
u_med(p)  = median { D[q, c]               : q ∈ 𝒩(p), V[q] }
r_i(p)    =        | D[q_i, c] − u_med(p) |   for each q_i ∈ 𝒩(p), V[q_i]
r_med(p)  = median r_i(p)                                          (the residual MAD)
ρ_c(p)    = | D[p, c] − u_med(p) | / ( r_med(p) + ε_0 )            (normalized residual)
```

`ε_0 = noise_floor` is a constant in voxel units that prevents division
by zero in flat regions where the neighborhood is locally constant.
Westerweel & Scarano fix it at `0.1` voxel for PIV; the same value is
used in essentially all DIC/DVC follow-ups, so we match.

A POI is flagged as an outlier iff

```
max_c ρ_c(p) > ε                 (default ε = 2.0)
```

### 4.3 Why median + MAD instead of mean + std

The neighborhood contains 26 candidates. Up to ~50% contamination is
tolerable for the median; a mean-based test breaks down at ~10%. Real
DVC failure modes — a single POI with a mis-correlated peak in an
otherwise smooth region, or a small cluster of bad fits near a
high-strain feature — fit the regime where median/MAD wins and
mean/std fails. Westerweel & Scarano (2005) report this as the
universal recommendation for PIV; the same statistical argument
carries to DVC verbatim.

### 4.4 Mask awareness, in detail

A masked or out-of-range POI is never included in any other POI's
neighborhood — the median and MAD are computed only over `q ∈ 𝒩(p)`
with `V[q] = True`. Two consequences:

1. POIs near the screw or near the volume boundary do not see ghost
   neighbors with `displacement = 0`; their median is computed from
   genuine estimates only.
2. A POI with too few valid neighbors (default `< 4`) is left `OK`
   unconditionally. This is the right default: the test cannot make a
   confident statement, and biasing the boundary toward "outlier" is
   exactly the failure mode plan §3 set out to prevent.

`min_neighbors = 4` matches the lower end of recommendations in the
PIV literature for boundary-tolerant settings; with three or fewer
points, both median and MAD become very brittle.

### 4.5 Vectorized implementation outline

```
1. Pad D with NaN to shape (N_z+2, N_y+2, N_x+2, 3).
2. Pad V with False to the same lattice extent.
3. Use `numpy.lib.stride_tricks.sliding_window_view` over (3, 3, 3)
   to materialize a (N_z, N_y, N_x, 3, 3, 3, 3) view (last axis is
   the component dim; the three 3-axes are the neighborhood).
4. Set the center voxel of the 3×3×3 window to NaN (the test
   excludes self).
5. Mask invalid neighbors to NaN via the broadcast-padded V view.
6. `nanmedian` along the neighborhood axes per component → u_med.
7. `nanmedian` of the absolute residuals → r_med.
8. ρ_c = |D[p, c] − u_med(p)| / (r_med(p) + ε_0).
9. flag = (ρ_c.max(axis=-1) > ε) & (count_valid_neighbors >= min_neighbors)
```

Memory footprint: the strided view shares storage with the padded
lattice (zero copy). The `nanmedian` materializes one lattice-shaped
array per component — a few MB at v1 scales. CPU runtime is single-
digit milliseconds; not worth GPU offload.

## 5. Interaction with the broader pipeline

### 5.1 Why the orchestrator computes `valid` immediately before calling

The current `correlate.py` builds `valid = status == POIStatus.OK`
only at the end. The outlier verb needs that boolean *as input*, so
the wiring computes it once after the search-radius gate, calls
`detect_outliers`, then promotes flagged POIs:

```python
valid_pre = status == POIStatus.OK
flag = detect_outliers(grid, displacements, valid_pre, …)
status[flag] = POIStatus.OUTLIER
displacements[flag] = 0.0
confidence[flag] = 0.0
valid = status == POIStatus.OK         # final convenience view
```

Outlier flagging happens *after* `OUT_OF_RANGE` so an out-of-range POI
never participates in any neighborhood. This matches the precedence
already implicit in the status enum — earlier failures are recorded
with their original cause rather than being overwritten by `OUTLIER`.

### 5.2 Why we do not feed `confidence` into the test

A natural extension is to weight the median by `confidence`, or to
threshold low-confidence POIs as outliers. Both are deferred:

- The classical normalized median test does not use a confidence
  channel, and its statistical guarantees rely on the unweighted
  median. Adding a weight changes the breakdown point.
- A confidence-only filter is a separate concern (the `LOW_CONF`
  status code is already reserved for it). Folding it into the
  outlier test conflates two different rejection criteria.

Both are open questions for v2 once we have ground-truth comparisons
from `validate/known_fields.py` to calibrate against.

### 5.3 Why this is a host-side test even on multi-GPU runs

Plan §6: "Final gather via shared memmap or `multiprocessing.Queue`
of numpy arrays. Outlier test runs after gather on a single process."
The lattice is at most a few hundred thousand POIs; the dominant cost
is the FFT, not a CPU median filter. Promoting the test to GPU would
add launch overhead and inter-process complexity for no gain.

## 6. API

```python
# mamba_dvc/core/outlier.py

def detect_outliers(
    grid: GridSpec,
    displacements: Float32[ndarray, "points 3"],
    valid: Bool[ndarray, "points"],
    *,
    epsilon: float = 2.0,
    noise_floor: float = 0.1,
    min_neighbors: int = 4,
) -> Bool[ndarray, "points"]: ...
```

- `grid` provides `grid_shape`, used to reshape `displacements` and
  `valid` to the regular lattice. The orchestrator already has the
  `GridSpec`; passing the dataclass keeps the call site short.
- `displacements` and `valid` are flat `(points,)` arrays in the same
  POI ordering `build_grid` emits (C-order over `(nz, ny, nx)`).
- Returns a `(points,)` boolean: `True` iff the POI is flagged as an
  outlier. By contract, only POIs with `valid[p] = True` and a
  sufficiently populated neighborhood can be flagged — masked or
  out-of-range POIs always return `False` from this verb (the
  orchestrator already has them tagged).
- `epsilon`, `noise_floor`, `min_neighbors` carry the canonical
  Westerweel & Scarano defaults; tunable for future calibration
  studies.

## 7. Validation strategy (tests that must pass)

Mirrors `tests/core/test_grid.py` and `tests/core/test_ncc.py`.

- **Single bad POI in a smooth field.** Build a synthetic lattice
  with `D[p] = (0, 0, 0)` everywhere except one interior POI set to
  `(5, 0, 0)`. Assert exactly that POI is flagged.
- **Smooth gradient is not flagged.** Linear field
  `D[p] = α · p` (small α) → no POI flagged at default `epsilon`.
- **Boundary POI is not flagged when its neighbors are smooth.**
  Place a smooth field on a 5×5×5 lattice; assert corner POIs
  (which have 7 valid neighbors) are not flagged when their
  displacement matches the local trend.
- **Mask-aware neighbor exclusion.** Mark half of the lattice
  `valid = False`; assert that POIs at the validity boundary are
  not flagged purely because of the false-zero side.
- **Insufficient-neighborhood guard.** Construct a configuration
  where a POI has only three valid neighbors → assert the POI is
  *not* flagged regardless of its displacement.
- **Status pre-filter is honored.** A POI with `valid=False` is
  never returned `True` from this verb, even with absurd
  displacement.
- **Determinism and shape contract.** Output shape `(points,)`,
  dtype `bool`, equal across two calls with identical inputs.
- **Reference parity** — match a textbook scalar-loop
  implementation of Westerweel & Scarano (2005) Eq. 3 element-wise
  on a small randomized lattice.
- **Runtime shape check** via `@jaxtyped(typechecker=beartype)` on
  the test wrapper.
- **Integration test in `tests/integration/test_pipeline_smoke.py`.**
  Run `correlate()` on a synthetic uniform-shift pair, manually
  perturb one POI's displacement post-hoc by editing the deformed
  volume's local content (so NCC genuinely picks the wrong peak),
  rerun, and assert the affected POI lands in
  `status == POIStatus.OUTLIER`.

## 8. Future-proofing (v2 hooks)

| v2 need | How `outlier.py` stays compatible |
|---|---|
| Iterative warp loop | The outlier mask becomes a prior for the next iteration's POI gating. The verb does not change; the driver decides whether to feed flagged POIs back as warm-starts or drop them. |
| Multi-level pyramid | Coarser-level outlier flags can seed the next level's `valid` input. The verb is reused at each level unmodified. |
| Time-series driver | Per-frame outlier masks accumulate per pair. The verb is called once per pair; persistence lives in `series.py`. |
| Universal outlier replacement (Westerweel 2005 §4) | Wraps the verb: use the median computed inside `detect_outliers` to substitute flagged values. Trivial to add as `replace_outliers(field) → field`. Not in v1 because we do not have evidence yet that downstream consumers prefer interpolation over masking. |
| Confidence-weighted variant | The verb gains an optional `confidence` parameter. Defaults preserve current behavior. |
| Anisotropic / per-component thresholds | `epsilon` accepts a 3-tuple. Defaults preserve current behavior. |
| Scattered-point neighborhoods (post-densification) | A separate verb `detect_outliers_scattered(positions, displacements, …)` using a kd-tree. The lattice version stays the fast path. |

## 9. Known non-features (by design)

- **No replacement / interpolation of flagged POIs.** The verb
  reports; downstream decides. v2 may add `replace_outliers` once we
  have a use case driving it.
- **No iterative outlier passes.** A single pass is canonical in PIV
  and adequate for our error rates. Multi-pass converges quickly but
  hides bugs in the underlying NCC; v1 prefers transparency.
- **No anisotropic thresholds.** µCT voxels are isotropic; one
  `epsilon` is the right shape until proven otherwise.
- **No confidence channel input.** See §5.2; the canonical test does
  not use one and adding one couples two rejection criteria.
- **No CuPy backend.** The lattice is small; host NumPy is faster end
  to end after accounting for transfer.

## 10. Open questions (non-blocking)

- **Should the orchestrator expose `epsilon` / `noise_floor` /
  `min_neighbors` in `correlate()`'s signature?** Defer until we have
  a calibration result from `validate/known_fields.py`. Hard-coding
  the canonical defaults keeps the v1 API small.
- **Should we also count `LOW_CONF` POIs as invalid neighbors?**
  `LOW_CONF` is reserved but not populated in v1, so the question is
  moot for now. When it is populated, the same rule should apply
  (any non-OK status is excluded from neighborhoods).
- **Should the verb accept a custom neighborhood footprint?** A 5×5×5
  cube or an ellipsoid weighted by spacing might be appropriate for
  highly anisotropic grids. Defer until a v2 grid actually goes
  anisotropic.
