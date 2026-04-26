# Single-pair orchestrator — plan & contract

The v1 single-pass DVC driver. Wraps the six algorithmic verbs landed
under `mamba_dvc/core/` into one pure function and returns a
`DisplacementField` (plan `docs/plans/overview.md` §5). This is the
"pure core" that future v2 features (iterative warp, pyramid,
time-series, multi-GPU dispatch) layer on top of without touching the
inner math.

## 1. Position in the codebase

```
build_grid ─► extract ─► window ─► ncc ─► peakfit ─► (outlier, v2 step 7)
        │       │           │        │        │
        └──── pipeline.correlate.correlate() bundles all of the above ───┘
                                  │
                                  ▼
                          DisplacementField
```

The orchestrator does not live in `core/`. It is the thin glue layer
that:

1. Accepts host-side `(reference, deformed, mask)` arrays.
2. Builds the POI grid and runs mask admission.
3. Runs the algorithmic chain on a single device (CPU or one GPU) in
   batches.
4. Enforces the `search_radius` constraint as a status flag, not an
   exception.
5. Emits one `DisplacementField` carrying displacement, confidence,
   per-POI status, and grid metadata.

`pipeline/` exists for v2: a multi-GPU dispatch (`gpu/dispatch.py`) and
the time-series driver will both *call* `correlate()` per pair. It
must therefore stay pure — no module-level state, no logging side
effects, no multiprocessing-aware code.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Pure function | Plan §4 calls this out as non-negotiable. v2's iterative warp wraps `correlate()` in a loop; v2's time-series driver wraps it with prefetch. Both fail if the orchestrator owns hidden state. |
| Single device, single process | Multi-GPU dispatch lives one layer up (`gpu/dispatch.py`), and replicates volumes to each device before calling this function once per shard of POIs. |
| Failed POIs are flagged, not pruned | The output grid shape is preserved end-to-end. A POI that fails admission, search-radius, or (eventually) the outlier test stays in `positions` with `valid=False` and a status code. v2 densification expects a regular lattice. |
| `deformed_mask` parameter present from day 1 | Plan §3 requires it for v2's per-frame deformed masks. v1 falls back to `mask` when not supplied, but the signature does not change. |
| No `DisplacementField` mutation after construction | Returned dataclass is `frozen=True`. Callers building stratified reports (`validate.known_fields`) work over copies or new arrays. |
| Batching is internal | Caller passes one pair at a time; the orchestrator picks the batch size based on the POI count and a memory cap. v2 driver layers can fan out across pairs without re-batching internally. |

## 3. API

```python
# mamba_dvc/pipeline/correlate.py

def correlate(
    reference: Float32[ndarray, "z y x"],
    deformed: Float32[ndarray, "z y x"],
    mask: Bool[ndarray, "z y x"] | None = None,
    deformed_mask: Bool[ndarray, "z y x"] | None = None,
    *,
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float = 0.25,
    search_radius: int | None = None,
    batch_size: int = 256,
    eps: float = 1e-12,
) -> DisplacementField: ...
```

- `mask` defaults to all-True (no admission filtering).
- `deformed_mask` falls back to `mask` (v1).
- `search_radius` defaults to `min(window) // 2`. POIs whose integer
  peak displacement exceeds the radius along any axis are flagged
  `OUT_OF_RANGE` and have their displacement zeroed (the wrapped value
  would be misleading).
- `batch_size` partitions admitted POIs along the leading axis. It
  influences peak GPU memory but never the output (the algorithm is
  per-POI; batches do not communicate).
- `eps` matches `core.ncc.correlate`'s safety floor.

## 4. Algorithm flow

```
1. Validate and coerce inputs (shapes, dtypes, mask alignment).
2. grid = build_grid(reference.shape, window, overlap)
3. admitted = filter_by_mask(grid, mask, mask_threshold)
4. status[admitted == False] = MASKED
5. for chunk in batches(admitted_starts, batch_size):
       ref_sv = extract_subvolumes(reference, chunk, grid.window)
       def_sv = extract_subvolumes(deformed,  chunk, grid.window)
       mref_sv = extract_subvolumes(mask,           chunk, grid.window)  # bool
       mdef_sv = extract_subvolumes(deformed_mask,  chunk, grid.window)  # bool

       # Both ref and def use their own per-frame mask for preprocessing —
       # in v1 they happen to be the same array.
       ref_pp = preprocess_subvolumes(ref_sv, mref_sv, tukey_alpha)
       def_pp = preprocess_subvolumes(def_sv, mdef_sv, tukey_alpha)

       corr = correlate_ncc(ref_pp, def_pp, eps=eps)
       integer, peak = peak_displacement(corr)
       fractional = gaussian_subvoxel_fit(corr, integer)

       displacement[chunk_slice] = integer + fractional
       confidence[chunk_slice]   = peak

6. Apply search_radius:
       too_far = any axis with |integer| > search_radius
       status[too_far] = OUT_OF_RANGE
       displacement[too_far] = 0
7. Build DisplacementField and return.
```

`extract_subvolumes` is currently float32-only. To extract bool mask
subvolumes we either (a) cast the mask to float32 once before the
batching loop and call extract on it, or (b) extend extract to accept
bool. Option (a) is the smallest change and incurs one full-volume
boolean→float32 conversion (1.6 GB at 960×1280×1280); option (b) is
cleaner and adds one branch in `extract_subvolumes`. **Decision: (b),
extend `extract_subvolumes` to accept bool dtype.** It avoids the
allocation, keeps the orchestrator readable, and the change in
`extract.py` is local.

## 5. Status enum

```python
class POIStatus(IntEnum):
    OK             = 0
    MASKED         = 1   # rejected by filter_by_mask
    OUTLIER        = 2   # reserved for core/outlier.py (v1 step 7)
    LOW_CONF       = 3   # reserved for confidence-threshold rejection
    OUT_OF_RANGE   = 4   # |integer peak| > search_radius
```

`OK` is `0` so a fresh `np.zeros(N, dtype=uint8)` initializes every
POI as OK by default; admission and downstream filters mark deviations.

`valid` in `DisplacementField` is the convenience derived field
`status == OK`. Both are stored — callers benefit from the boolean
view, debugging benefits from the categorical enum.

## 6. DisplacementField

Already specced in plan §5 but not yet implemented. The shape symbols
align with our lowercase vocabulary (`points`, `z`, `y`, `x`):

```python
@dataclass(frozen=True)
class DisplacementField:
    positions:     Float32[ndarray, "points 3"]   # POI centers (z, y, x)
    displacements: Float32[ndarray, "points 3"]   # voxel displacement (dz, dy, dx)
    valid:         Bool[ndarray, "points"]        # True iff status == OK
    confidence:    Float32[ndarray, "points"]     # peak NCC value
    status:        UInt8[ndarray, "points"]       # POIStatus enum
    grid_shape:    tuple[int, int, int]           # (nz, ny, nx) POIs per axis
    spacing:       tuple[int, int, int]           # grid stride in voxels
    window:        tuple[int, int, int]           # subvolume size used
```

Reshaping to a regular lattice (for v2 densification or visualization)
is `field.displacements.reshape(*field.grid_shape, 3)`.

## 7. Validation strategy

Mirrors the test layout of the underlying core modules.

- **All-OK path.** Synthetic pair with no mask, fractional shift well
  inside `search_radius` → all `valid` flags True, displacements match
  ground truth within 0.1 voxel on interior POIs.
- **Mask admission.** Construct a mask that excludes a slab; assert
  POIs in that slab have `status == MASKED`, `valid == False`, and
  zero displacement.
- **Search-radius clamp.** Apply a synthetic shift larger than
  `search_radius`; assert flagged POIs have `status == OUT_OF_RANGE`
  and zero displacement.
- **Grid metadata round-trip.** `field.grid_shape`, `spacing`,
  `window` match the `GridSpec` used internally.
- **`deformed_mask=None` falls back to `mask`.** Behavior identical to
  passing the same array twice.
- **Batch invariance.** Same inputs at `batch_size=8` and
  `batch_size=64` produce bit-identical output (within float32 noise).
- **Smoke test rewired.** The integration test in
  `tests/integration/test_pipeline_smoke.py` now calls `correlate()`
  directly; the inline composition is deleted.

## 8. Future-proofing (v2 hooks)

| v2 need | How the orchestrator stays compatible |
|---|---|
| Multi-GPU dispatch | `gpu/dispatch.py` shards POIs along Z, replicates volumes per GPU, calls `correlate()` per shard, gathers and merges `DisplacementField`s. No change to this function. |
| Iterative warp loop | Outer loop calls `correlate(reference, warp(deformed, prev_field), …)` repeatedly until convergence. The pure-function contract is what makes this safe. |
| Coarse-to-fine pyramid | Same loop structure, with decreasing `window` and an `initial_guess` parameter added to `peakfit` (v2). |
| Time-series driver | `series.py` calls `correlate()` once per frame pair. Prefetch and warm-start live in the driver. |
| Padfield masked FFT | `core/ncc.correlate` gains an internal mask branch; orchestrator already passes mask subvolumes through. |
| Outlier test (`core/outlier.py`) | Slot in between step 6 (search-radius) and step 7 (DisplacementField construction). New status code `OUTLIER` is already reserved. |

## 9. Known non-features (by design, v1)

- **No multiprocessing here.** Multi-GPU dispatch is one layer up.
- **No logging or progress bars.** Drivers can wrap.
- **No async I/O.** Reference and deformed are already in memory; the
  orchestrator does not load from disk.
- **No automatic CPU/GPU choice.** Whatever module the input arrays
  belong to is what runs. The caller decides.
- **No outlier filter wired in yet.** The seat is reserved; the
  algorithmic step is its own commit.

## 10. Open questions (non-blocking)

- **Should `correlate()` accept a `GridSpec` directly?** Useful for v2
  pyramids that want to reuse a coarser grid as a warm-start. Defer
  until pyramid actually exists.
- **Should `confidence` for non-OK POIs be `nan` or `0`?** Current
  plan: `0`, because callers stratifying by confidence already gate on
  `valid`. nans propagate through plotting code in unhelpful ways.
- **Should `displacements` for non-OK POIs be `nan` or `0`?** Same
  argument, same answer: `0`. Documented in the docstring.
