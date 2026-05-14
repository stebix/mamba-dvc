# MultiGPUDispatcher buffer-aliasing bug

`MultiGPUDispatcher` (persistent multi-process path) returns
`DisplacementField` instances whose `displacements`, `confidence`, and
`status` arrays are all aliased to dispatcher-owned scratch. Each
`.correlate(...)` call overwrites those arrays in place, so every field
collected across a series ends up pointing at the *last* call's data.

Discovered while inspecting the §5–§8 error metrics in
`notebooks/auto-dispatch-temporal-showcase.ipynb`; isolated by the §9
debug ladder in the same notebook.

- **Affected commit**: `26d4247 refactor(gpu): extract validation + outlier + SHM helpers, hoist merged buffers`
- **Symptom surface**: `correlate_series(dispatcher=…)` and any direct
  use of `MultiGPUDispatcher` with `len(device_ids) > 1` that calls
  `.correlate(...)` more than once inside the `with` block.
- **Not affected**: per-call `correlate_multi_gpu(...)`, single-device
  in-process dispatcher path, host-only `correlate(...)`.

## Symptom

Running the showcase notebook through a 4-GPU dispatcher across 15
SEQUENTIAL + 15 REFERENCE_ANCHORED pairs at `(192, 192, 192)`,
`window=64`:

| Strategy | per-pair MAE (median) | p95 |
|---|---|---|
| `SEQUENTIAL` | 6.13 | 13.89 |
| `REFERENCE_ANCHORED` | 7.00 | 10.92 |
| `SEQUENTIAL` cumulative | 20.15 | 78.73 |

Plan target on textured noise is **sub-0.1 vx**. Reported numbers are
~60× over budget.

Two additional shapes that didn't fit any algorithmic failure mode:

1. **Inverted error vs. displacement.** Anchored MAE at `t=1`
   (`|u|≈1 vx`) was 6.13 vx; at `t=15` (`|u|≈14 vx`) it was 0.0548 vx.
   The largest displacement gave the *smallest* error. Apodization
   mismatch, search-radius pressure, and Tukey-taper bias all predict
   the opposite shape.
2. **Identical status histograms across every pair.** 51 OK / 74 OUTLIER
   for every single one of 30 pairs across both strategies. Real
   correlator output is never that uniform.

## Diagnostic chain

`notebooks/auto-dispatch-temporal-showcase.ipynb` §9. Five focused
bisections; results below are from the executed notebook.

### Step 1 — multi-GPU vs single-GPU dispatcher on the same pair

| Pair (GT magnitude) | Path | MAE | `<u_rec>` | `n_valid` |
|---|---|---|---|---|
| small `(frames[0], frames[1])` (≈1 vx) | multi-GPU `correlate_series[0]` | **6.13** | (+3.075, +4.947, +1.214) | 125/125 |
| small | single-GPU dispatcher `[0]` | **0.0081** | (+0.498, +0.797, +0.199) | 125/125 |
| large `(frames[0], frames[15])` (≈14 vx) | multi-GPU `correlate_series[-1]` | 0.0548 | (+7.536, +12.124, +2.975) | 51/125 |
| large | single-GPU dispatcher `[0]` | 0.0548 | (+7.536, +12.124, +2.975) | 51/125 |

Single-GPU recovers the small pair to 0.008 vx (well inside budget).
Multi-GPU "first-pair" data on the same input reports 6.13 vx error.
**Arithmetic check**:

```
(3.075, 4.947, 1.214) ≈ (51/125) × (7.536, 12.124, 2.975)
                     = (51/125) × (recovered_large_pair)
```

So the multi-GPU "first pair" is reading the **last pair's buffer**,
masked by the first pair's snapshot of `valid` (all 125 True at small
displacement, before any outliers were flagged).

### Step 2 — integer shift sanity

`np.roll((2, -3, 1))` through single-GPU dispatcher: MAE 0.033,
`<u_rec>` = (+2.012, -3.012, +1.011). Sign convention, search-radius
gate, and outlier test all fine in isolation.

### Step 3 — status histograms

```
REFERENCE_ANCHORED        OK    MASKED  OUTLIER  LOW_CONF  OUT_OF_RANGE
  t= 1..15                51         0       74         0             0
SEQUENTIAL                OK    MASKED  OUTLIER  LOW_CONF  OUT_OF_RANGE
  t= 1..15                51         0       74         0             0
```

30 pairs, identical histograms. This is the cleanest read: every field's
`status` array is the same ndarray object.

### Step 4 — search-radius sweep at small `|u|`

| `search_radius` | MAE | `<u_rec>` |
|---|---|---|
| 32 | 0.0081 | (+0.498, +0.797, +0.199) |
| 8 | 0.0081 | (+0.498, +0.797, +0.199) |
| 4 | 0.0081 | (+0.498, +0.797, +0.199) |
| 2 | 0.0081 | (+0.498, +0.797, +0.199) |

Single-pair dispatcher; identical to 4 decimal places across the sweep.
Peak-pickup ambiguity was a red herring.

### Step 5 — NCC kernel A/B at small `|u|`

| Mode | small-pair MAE | large-pair MAE |
|---|---|---|
| `linear` + `overlap` (current default) | 0.0081 | 0.0548 |
| `cyclic` + `global` (legacy) | 0.0261 | 0.1082 |

`linear-overlap` strictly dominates at both scales. The `|u|/W`
shrinkage fix that landed for the original showcase (see
`docs/insights/error-minimization.md`) is *not* regressed by anything
in the dispatcher pipeline.

Every probe in steps 2–5 is a **single `.correlate()` call per dispatcher
instance**. None of them reproduce the bug. Steps 1's multi-GPU `[0]`
result and the original cell-13 sweep both involve **multiple
`.correlate()` calls on the same dispatcher** and both reproduce the
bug. Bisection complete: the bug is in the persistent dispatcher's
per-pair return path.

## Root cause

Commit `26d4247` hoisted the merged-result buffers from per-call locals
into dispatcher attributes "to avoid reallocating ~3 arrays of length
n_points every call." Three locations interact:

`mamba_dvc/gpu/dispatch.py:1243`
```python
self._merged_disp = np.zeros((n_points, 3), dtype=np.float32)
self._merged_conf = np.zeros(n_points, dtype=np.float32)
self._merged_stat = np.full(n_points, POIStatus.MASKED, dtype=np.uint8)
```

`mamba_dvc/gpu/dispatch.py:1515`
```python
merged_disp = self._merged_disp
merged_conf = self._merged_conf
merged_stat = self._merged_stat
merged_disp.fill(0.0)
merged_conf.fill(0.0)
merged_stat.fill(POIStatus.MASKED)
...
return merged_disp, merged_conf, merged_stat
```

`mamba_dvc/gpu/dispatch.py:1465`
```python
return DisplacementField(
    ...,
    displacements=disp,        # ← aliased to self._merged_disp
    confidence=conf,           # ← aliased to self._merged_conf
    status=stat,               # ← aliased to self._merged_stat
    valid=valid,               # ← fresh array (status == OK)
    ...,
)
```

`DisplacementField` is `frozen=True`, but freezing the dataclass does
not freeze the ndarrays it references. Every pair scatters into the
same three buffers, so every field stored on the
`DisplacementSeries.fields` tuple points at the dispatcher's most
recent fill.

`valid` is the only field that isn't aliased: `_apply_outlier_rejection`
ends with `return status == POIStatus.OK`, which is a fresh ndarray each
call. That's why post-hoc inspection sees `valid.sum() == 125` (pair 0's
snapshot) but `status` reports 51 OK / 74 OUTLIER (latest pair's
state). The `DisplacementField.valid` docstring states "the two never
disagree" — today's code silently violates that invariant after the
first pair completes.

### Why neighbouring paths are safe

- `_dispatch_pair_inproc` returns `cp.asnumpy(disp_dev)` etc.; CuPy
  always copies to host on `asnumpy`, so each call yields fresh
  ndarrays.
- `_gather_shards` (used by the per-call `correlate_multi_gpu`)
  allocates `np.zeros((n_points, 3), …)` locally on every call.
- Worker processes (`_serve_worker`) `cp.asnumpy` their device buffers
  before `send`; pipe payloads are fresh host arrays.

So the bug surface is exactly one method (`_dispatch_pair_mp`) plus the
two sites that allocate and release the hoisted scratch.

## Impact

Anything that holds a `DisplacementField` across a subsequent
`.correlate(...)` call on the same dispatcher is reading silently
corrupted data. Concretely:

- `correlate_series(dispatcher=…)` — every entry in the returned
  `DisplacementSeries.fields` except the last is wrong.
- `DisplacementSeries.cumulative()` — composes wrong inputs.
- `evaluate_synthetic(..., device_ids=…)` — the sweep harness threads
  one dispatcher through every `(strategy, lag)` pair; every per-pair
  error number it has produced through the dispatcher path is wrong.
- Any future caller that snapshots a field for later analysis.

Per-call `correlate_multi_gpu(...)` and the host-only `correlate(...)`
are unaffected — they each allocate their own outputs.

## Fix plan

Tiny code change; the value is in the regression coverage.

### Step 1 — write the regression tests first

Four tests in `tests/gpu/test_dispatch.py`, all gated `@pytest.mark.gpu`.
Each opens a persistent dispatcher and exercises a multi-pair sequence.

1. **`test_field_buffers_do_not_alias_across_pairs`** — for every pair
   `(i, j)` with `i != j` and every aliasable array
   (`displacements`, `confidence`, `status`):

   ```python
   assert not np.shares_memory(fields[i].displacements, fields[j].displacements)
   ```

   Direct catch of the bug.

2. **`test_field_values_persist_after_subsequent_pairs`** — snapshot
   `fields[0].displacements`, run two more pairs, assert the snapshot
   still equals `fields[0].displacements`. User-visible symptom.

3. **`test_valid_matches_status_for_every_returned_field`** —
   `(field.valid == (field.status == POIStatus.OK)).all()` for every
   field. Pins the `DisplacementField.valid` docstring invariant.

4. **`test_persistent_dispatcher_matches_single_pair_dispatcher`** —
   value-level pin. Run one pair through a fresh single-pair dispatcher
   to get `f_single`. Then run the same pair through a persistent
   dispatcher, snapshot the result, run two more unrelated pairs, and
   assert the snapshot is still allclose (`atol=1e-6`) to `f_single`.
   Catches both alias bugs and any future drift between the two paths.

Fixture: 3-frame synthetic series at `(96, 96, 96)`, `window=32`.
Synthesis runs in <2 s; the dispatcher startup dominates. Total
runtime ≤ 30 s on one A6000.

### Step 2 — fix `MultiGPUDispatcher`

Four coordinated edits in `mamba_dvc/gpu/dispatch.py`. Net diff: ~17
lines deleted, ~7 lines added.

**`__init__`** — remove the lifecycle attributes:

```python
# delete
self._merged_disp: np.ndarray | None = None
self._merged_conf: np.ndarray | None = None
self._merged_stat: np.ndarray | None = None
```

**`__enter__`** — remove the hoisted allocations and the comment that
defended them (lines ~1239–1245):

```python
# delete
# Allocate the merged-result buffers once for the dispatcher's
# lifetime; ``_dispatch_pair_mp`` resets the slots each pair via
# ``.fill(...)`` instead of reallocating ~3 arrays of length
# n_points every call.
self._merged_disp = np.zeros((n_points, 3), dtype=np.float32)
self._merged_conf = np.zeros(n_points, dtype=np.float32)
self._merged_stat = np.full(n_points, POIStatus.MASKED, dtype=np.uint8)
```

**`_dispatch_pair_mp`** — allocate per call (mirrors `_gather_shards`):

```python
- # Reset the per-pair merged buffers (allocated once at __enter__).
- assert self._merged_disp is not None
- assert self._merged_conf is not None
- assert self._merged_stat is not None
- merged_disp = self._merged_disp
- merged_conf = self._merged_conf
- merged_stat = self._merged_stat
- merged_disp.fill(0.0)
- merged_conf.fill(0.0)
- merged_stat.fill(POIStatus.MASKED)
+ # Allocate per-pair so the returned DisplacementField owns its
+ # arrays exclusively. See
+ # test_field_buffers_do_not_alias_across_pairs.
+ merged_disp = np.zeros((self._n_points, 3), dtype=np.float32)
+ merged_conf = np.zeros(self._n_points, dtype=np.float32)
+ merged_stat = np.full(self._n_points, POIStatus.MASKED, dtype=np.uint8)
```

**`_cleanup_state`** — drop the now-stale resets:

```python
# delete
self._merged_disp = None
self._merged_conf = None
self._merged_stat = None
```

Performance impact: per-pair allocation is ~2 KB at `(192, 192, 192)`
and ~150 KB at production `(960, 1280, 1280)` — orders of magnitude
below the per-pair SHM publish (~28 MB and ~6 GB respectively) and
unmeasurable next to the CUDA work.

### Step 3 — verify against the notebook

Re-execute `notebooks/auto-dispatch-temporal-showcase.ipynb`. Expected
shapes after the fix:

- per-pair `SEQUENTIAL` MAE ≈ 0.01 vx (constant 1 vx steps; flat in `t`)
- per-pair `REFERENCE_ANCHORED` MAE growing slowly from ~0.01 (t=1) to
  ~0.06 vx (t=15) — the `|u|/W` residual that step 5 of the ladder
  measured directly
- cumulative `SEQUENTIAL` drift growing from ~0.01 to a few tenths of
  a vx over 15 steps (random walk on 0.01 vx noise)
- status histograms varying across pairs instead of identical

If the notebook reproduces those shapes, the fix is verified
end-to-end.

## Defensive additions worth folding in

### `DisplacementField.__post_init__` invariant check

`DisplacementField.valid` is documented as a convenience view of
`status == POIStatus.OK`. A two-line check at construction would have
caught this bug the moment the first field was returned:

```python
def __post_init__(self) -> None:
    if not bool(np.array_equal(self.valid, self.status == POIStatus.OK)):
        raise ValueError(
            "DisplacementField.valid must equal (status == POIStatus.OK)"
        )
```

Protects every future producer of `DisplacementField`, not just the
dispatcher. Cost: one ndarray comparison per construction (~µs at our
sizes). Recommended.

### Docstring update

One line on `MultiGPUDispatcher.correlate` stating the no-alias
contract: "The returned `DisplacementField` owns its `displacements`,
`confidence`, and `status` arrays; callers may hold the result across
subsequent `.correlate(...)` calls without copying." Documents the new
guarantee for future readers.

## Sequencing and commit shape

Two commits:

1. `fix(gpu): MultiGPUDispatcher returns independent buffers per pair`
   — dispatch.py changes plus the four new tests. Tests precede the
   implementation diff inside the commit so a reader sees the broken
   contract before the fix.
2. `feat(types): DisplacementField asserts valid/status consistency`
   — the `__post_init__` invariant. Separate commit because it touches
   a different module and protects more than just this caller.

Both pass `ruff format`, `ruff check`, `pyright`, `pytest -m "not gpu"`.
The new tests are GPU-only and run under `pytest -m gpu` on the
workstation.

## Risk assessment

- **Code risk**: very low. The hoisted-buffer pattern was used in
  exactly three sites in one class; the per-call allocation pattern is
  the standard everywhere else.
- **Performance risk**: zero in practice. Per-pair alloc dwarfed by the
  SHM publish that already happens per pair.
- **Test flakiness**: deterministic. `make_texture` uses a fixed seed;
  the alias assertions are structural rather than numerical.

## Tests that would have caught it before merge

Any of:

1. A `np.shares_memory` check between two `DisplacementField`s returned
   by the same dispatcher.
2. A `(field.valid == (field.status == POIStatus.OK)).all()` assertion
   on a multi-pair series.
3. A value-level test that snapshots `fields[0].displacements` before
   running later pairs and asserts equality after.

The existing `tests/gpu/test_dispatch.py` covers correctness of a
single pair through the multi-GPU path but does not exercise the
persistent dispatcher's *across-pair* invariant. Step 1 above closes
that gap.
