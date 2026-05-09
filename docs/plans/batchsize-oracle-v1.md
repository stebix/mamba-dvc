# Batch-size oracle — plan & contract (v1, functional core)

The pre-flight that picks `batch_size` for `correlate_multi_gpu()` from
free per-GPU VRAM and the resolved DVC config. Implements the deferred
`_resolve_batch_size` heuristic called out in
`docs/plans/overview.md` §6 and `docs/plans/dispatch.md` §10.

This document covers **only the functional core** — the cost model, the
probe, and the recommender. Wiring it into `gpu/dispatch.py` and
`pipeline/correlate.py` is a separate follow-up so the model can be
landed and tested in isolation.

## 1. Problem framing

The transient working set of one shard is dominated by the kernel's
per-POI footprint times `batch_size`. At `W = 96` the linear+overlap
kernel — the production default — jumps from comfortable at batch 64
(≈14 GB) to OOM at batch 256 (≈50 GB) on a 49 GB A6000. The free-VRAM
ceiling depends on:

- the resident base (replicated `reference + deformed + mask + deformed_mask`,
  ≈14 GB at full `(960, 1280, 1280)` shape with mask),
- the kernel mode (`cyclic` vs `linear`) and normalization (`global`
  vs `overlap`),
- the window size,
- whatever the driver had already pinned (cuFFT plan cache, prior
  allocations).

The user-facing question — "what `batch_size` should I pass?" — has a
closed-form answer for the dominant terms, plus a small constant for
plan cache and fragmentation that we calibrate once and absorb into a
headroom factor.

The current state of the world is that the call site picks `batch_size`
manually (default 64 in `dispatch`, 256 in `correlate`) and the user
iterates if they OOM. This plan removes the iteration.

## 2. Module placement

New file: `mamba_dvc/gpu/budget.py`. Pure module, no top-level CuPy
import (mirrors the `_cp` guard pattern in `core/ncc.py`,
`core/extract.py`, `gpu/dispatch.py`). Tests run on CPU-only hosts; the
only piece that touches CuPy is `probe_free_vram`, isolated at the
bottom of the module.

```
mamba_dvc/gpu/
  budget.py     ← new; cost model + recommender (functional core)
  dispatch.py   ← unchanged in this plan; wiring is a follow-up
  shm.py
```

Why a separate module:

- The cost model is pure arithmetic over `(window, mode, normalization,
  free_bytes)` — independent of the dispatch state machine. Keeping it
  standalone means we can test it without spawning workers.
- The recommender is callable from CLI tooling and `scripts/run_e2e_zarr.py`
  immediately, well before the dispatch wiring lands.
- It keeps the seam crisp: `budget.py` is the only module that calls
  into `cp.cuda.runtime.memGetInfo`, so the rest of the gpu/ tree stays
  algorithm-pure.

## 3. Public API (functional core)

Three layers, each independently testable.

```python
# mamba_dvc/gpu/budget.py

@dataclass(frozen=True)
class KernelFootprint:
    """Bytes-per-POI transient cost + per-call fixed cost for one kernel config."""
    per_poi_bytes: int        # scales with batch_size
    fixed_bytes: int          # cuFFT plan cache, indicator-box spectrum, etc.

@dataclass(frozen=True)
class BudgetInputs:
    volume_shape: tuple[int, int, int]
    window: tuple[int, int, int]
    mode: NCCMode
    normalization: NCCNormalization
    has_mask: bool                  # determines whether mask buffers are resident
    deformed_mask_distinct: bool    # avoids double-counting when def_mask falls back

def kernel_footprint(
    window: tuple[int, int, int],
    mode: NCCMode,
    normalization: NCCNormalization,
) -> KernelFootprint: ...

def resident_bytes(inputs: BudgetInputs) -> int: ...

def estimate_max_batch(
    free_bytes: int,
    inputs: BudgetInputs,
    *,
    headroom_fraction: float = 0.15,
    min_batch: int = 1,
    max_batch: int = 1024,
    multiple_of: int = 8,
) -> int: ...

def probe_free_vram(device_id: int) -> tuple[int, int]:
    """(free_bytes, total_bytes). Thin wrapper around
    cp.cuda.runtime.memGetInfo. Lives here so the pure cost model in
    this module is the only thing that calls into CuPy."""

def recommend_batch_size(
    inputs: BudgetInputs,
    device_ids: Sequence[int],
    *,
    headroom_fraction: float = 0.15,
    floor: int = 8,
    ceiling: int | None = None,
) -> int:
    """Probe each device, take the min of free bytes, return a single
    cross-device batch. The min rule means heterogeneous fleets degrade
    to the slowest device — the safe default. Per-device returns are a
    v2 hook (see §8)."""
```

Three reasons for this shape:

1. `kernel_footprint` is the only piece that has to be calibrated
   against measurement. Splitting it out means one place to change
   when the kernel changes.
2. `estimate_max_batch` takes `free_bytes` as a plain integer, so we
   can drive it from a probe, a unit test, or a synthetic
   "what if 4 GB?" sweep without touching CuPy.
3. `recommend_batch_size` is the call site's one-liner. Everything
   else is internal but exposed so tests can pin each layer.

## 4. Cost model

Two regimes; numbers below are analytic estimates for `W = 96`,
`batch = B`, float32 / complex64. The per-POI total is the constant
`KernelFootprint.per_poi_bytes`.

### 4.1 `mode=cyclic, normalization=global`

| Buffer | Bytes |
|---|---|
| `ref_pp`, `def_pp` `(B, W³)` | 2 · 4·W³ |
| `rfftn(ref)`, `rfftn(def)` `(B, W, W, W/2+1)` complex64 | 2 · 8 · W²·(W/2+1) ≈ 8·W³ |
| `cross_power` complex64 | 8·W³ |
| `irfftn` output float32 | 4·W³ |
| cuFFT workspace (empirically ~ one spectrum) | ~8·W³ |
| **per-POI total** | **≈ 36·W³ B bytes** |

### 4.2 `mode=linear, normalization=overlap` (production default)

| Buffer | Bytes |
|---|---|
| `ref_padded`, `def_padded` `(B, 2W, 2W, 2W)` float32 | 2 · 32·W³ |
| `ref_sq_padded`, `def_sq_padded` (alive through second FFT) | 2 · 32·W³ |
| 4 padded spectra complex64 `(B, 2W, 2W, W+1)` | 4 · ~64·W³ ≈ 256·W³ |
| `numerator_full`, `s_ref_sq_full`, `s_def_sq_full` | 3 · 32·W³ |
| `denom`, `corr` cropped float32 | 2 · 4·W³ |
| cuFFT workspace at `2W` (significantly larger than at `W`) | ~64·W³ |
| **per-POI total** | **≈ 480·W³ B bytes ≈ 13–14× cyclic** |

This is order-of-magnitude only — the real constants must come from a
one-time calibration.

### 4.3 Calibration (build-time, not runtime)

A small script `scripts/calibrate_budget.py` (delivered alongside this
module):

1. For each `(mode, normalization, W)` of interest, runs the kernel at
   `B ∈ {16, 32, 64}` on a single GPU.
2. Measures the `cp.cuda.runtime.memGetInfo()` delta around the kernel
   call.
3. Fits `bytes(B) = a·B + b` and records `(a, b)` per config.
4. Writes `mamba_dvc/gpu/_budget_constants.json` containing
   `{ "<mode>__<normalization>__W<wz>x<wy>x<wx>": {per_poi: a, fixed: b} }`.

`kernel_footprint` reads this file at module import and falls back to
the analytic estimate above (with a `RuntimeWarning`) if the requested
config is missing. JSON over a Python module so the calibration script
can rewrite it without a code review round trip.

Calibration re-runs only when the kernel changes, the hardware
changes, or a new `(mode, normalization, W)` combination is added.
Constants are checked in.

### 4.4 Resident base

`resident_bytes(inputs)` accounts for the four host arrays each worker
uploads via `cp.asarray`:

| Buffer | Bytes | Always resident? |
|---|---|---|
| `reference` float32 | `4·prod(volume_shape)` | yes |
| `deformed` float32 | `4·prod(volume_shape)` | yes |
| `mask` bool | `prod(volume_shape)` | yes (synthesized when `has_mask=False`) |
| `deformed_mask` bool | `prod(volume_shape)` | only when `deformed_mask_distinct=True` |

At `(960, 1280, 1280)` with mask and falling-back deformed_mask:
`2 · 6.29 + 1.57 = 14.15 GB`. Without distinct deformed_mask the cost
matches dispatch's actual upload pattern (it deliberately publishes
shared SHM under both names rather than copying).

## 5. Probe semantics

`probe_free_vram(device_id)` must run **before** any volume upload by
the caller. The recommender contract:

1. Caller probes on a "cold" device (no large allocations yet by us).
2. We subtract `resident_bytes(inputs)` from `free`, since those
   uploads are about to happen.
3. We multiply by `(1 - headroom_fraction)` to reserve for plan cache
   and fragmentation.
4. Divide by `kernel_footprint(...).per_poi_bytes`, floor to
   `multiple_of`, clamp to `[min_batch, max_batch]`.

Underflow case: if the result is `< min_batch`, raise `RuntimeError`
rather than silently shipping `batch=0` or clamping to `min_batch`.
Clamping a too-tight config makes a worker OOM mid-shard, killing the
process and burning the spawn cost; raising lets the caller pick a
smaller `W` or fewer devices before any work starts. The error
message includes the offending numbers (free, resident, per-POI,
headroom, requested config) so debugging is greppable.

Per-device heterogeneity: the recommender takes the **minimum** of
free bytes across `device_ids`. The dispatch layer shards POIs
roughly evenly along lattice-Z and uses a single `batch_size` in the
helper kwargs (`gpu/dispatch.py:_run_helper_on_device`), so a single
batch number is what the call site needs. Per-device dicts are a v2
hook — see §8.

## 6. Test surface

### 6.1 Pure tests (no GPU, run in CI)

`tests/gpu/test_budget.py`:

- `TestKernelFootprint` — pin per-POI bytes for `W ∈ {64, 96, 128}`
  on both kernels. Regression-guards the model when the kernel
  changes; if the cost model and the kernel disagree, this is the
  test that fails.
- `TestKernelFootprintMissingConstants` — when
  `_budget_constants.json` lacks the requested config, the analytic
  fallback fires with a `RuntimeWarning`; the returned numbers are
  positive and within an order of magnitude of the calibrated values
  for known configs.
- `TestResidentBytes` — `(960, 1280, 1280)` with/without mask,
  with/without distinct `deformed_mask`. Pins the 14 GB / 12.6 GB
  break.
- `TestEstimateMaxBatch` — given a synthetic `free_bytes`:
  - Returned batch is the largest multiple of `multiple_of` that fits.
  - Monotone non-decreasing in `free_bytes`.
  - Honours `min_batch` / `max_batch` clamps.
- `TestEstimateMaxBatchUnderflow` — `free_bytes < resident +
  per_poi · min_batch` raises `RuntimeError` with the offending
  numbers in the message.
- `TestRecommendBatchSizeMin` — monkeypatch `probe_free_vram` to
  return synthetic per-device tuples; verify the min-rule, ceiling
  clamp, and that empty `device_ids` raises `ValueError`.
- `TestRecommendBatchSizeNoCupy` — when CuPy is unavailable,
  `probe_free_vram` raises `RuntimeError` with the same wording the
  rest of the gpu/ tree uses (matches `dispatch._resolve_device_ids`).

### 6.2 GPU-flagged tests (`@pytest.mark.gpu`)

- `TestProbeFreeVramShape` — `(free, total)` are positive ints,
  `total` matches `cp.cuda.runtime.getDeviceProperties` for that
  device.
- `TestRecommendedBatchActuallyFits` (`@pytest.mark.slow`) — at the
  recommended batch, run one
  `pipeline._internal.correlate_admitted_subset` step on synthetic
  volumes and confirm no OOM. The end-to-end empirical sanity check
  on the cost model.

The full multi-GPU production check (4× A6000) is not in CI, same as
the rest of the dispatch path.

## 7. Tradeoffs and decisions

| Choice | Recommended | Alternative | Why |
|---|---|---|---|
| Analytic model + calibrated constants | yes | empirical bisection per call | Bisection costs seconds, runs every call, and the smallest probe can OOM if resident is already too big. Analytic + one-time calibration is robust and fast. |
| `headroom_fraction = 0.15` | yes | 0.05 (aggressive) / 0.30 (paranoid) | Plan cache on A6000 at production W is empirically ~5%; fragmentation pushes it up. 15% gives a meaningful safety margin without halving batch. |
| Min-of-devices → single batch | yes | per-device dict return | The dispatch helper takes one `batch_size`. A heterogeneous batch would need the helper signature to take a per-shard batch. v2 hook: `recommend_batch_size(..., per_device=True) -> dict`. |
| Probe before upload (cold) | yes | probe after upload | Probing after upload undercounts available headroom and fights with our own residency model. Integration must call the recommender before SHM publish + worker spawn. |
| Calibration as build-time | yes | runtime | Keeps `recommend_batch_size` deterministic and side-effect-free in production. Calibration re-runs only when the kernel or hardware changes. |
| Round to multiple of 8 | yes | exact integer | cuFFT plans on small-prime batch sizes; multiple-of-8 is the cheap performance win. |
| Reject (raise) on underflow | yes | clamp to `min_batch` and continue | A clamped run that OOMs mid-shard kills a worker and burns spawn cost; raising with a clear message lets the caller adjust before any work starts. |
| Constants in JSON, not a Python module | yes | a Python module | JSON is regenerable by the calibration script without a code review. Constants change with hardware, not with logic. |

## 8. Out of scope (deliberately deferred)

- **Wiring into `correlate_multi_gpu`.** Signature becomes
  `batch_size: int | Literal["auto"] = "auto"`, with `"auto"` calling
  `recommend_batch_size(...)`. Detailed in §10.1; single-device
  `correlate()` gets the same auto path.
- **Adaptive back-off on OOM mid-run.** v2; needs cooperation from
  the helper to retry a shrunk batch.
- **Per-shard batch sizes for heterogeneous fleets.** The
  `per_device=True` hook above; meaningful only on mixed-GPU hosts.
- **Pinned host buffers (`gpu/memory.py`).** Different concern; also
  a v2 item from the overview plan.
- **The calibration script itself.** Mentioned here as a dependency;
  a thin script with its own short doc lands alongside the module.

## 9. Build order

1. `mamba_dvc/gpu/budget.py` skeleton: `KernelFootprint`,
   `BudgetInputs`, the analytic-only `kernel_footprint`,
   `resident_bytes`, `estimate_max_batch`. No CuPy. All pure tests
   pass.
2. `probe_free_vram` and `recommend_batch_size`. Pure tests via
   monkeypatch. GPU-flagged tests run only where a device exists.
3. `scripts/calibrate_budget.py` and the first `_budget_constants.json`
   covering `(linear, overlap, W=96)` and `(cyclic, global, W=96)` —
   the only two configs in v1 use today.
4. Optional: extend `scripts/run_e2e_zarr.py` to accept
   `--batch-size auto` and call the recommender. Detailed in §10.2;
   gives a real-world smoke test for free.

## 10. Implementation checkpoints — call-site and CLI wiring

§1–§9 cover the functional core (the new module). This section
sketches the three downstream consumers in the order they should land.
Each is independent of the others by construction, so any subset is a
valid scope reduction.

| # | Checkpoint | Surface | Lines moved |
|---|---|---|---|
| 10.1 | `correlate_multi_gpu(batch_size="auto")` | API | foundation; everything else delegates here |
| 10.2 | `run_e2e_zarr.py --batch-size auto` | script | one-line follow-up to 10.1 |
| 10.3 | `mamba-dvc budget` subcommand | CLI | standalone; debug tool, not on the correlation path |

### 10.1 Checkpoint 1 — `correlate_multi_gpu(batch_size="auto")` (API)

The keystone change. Once this lands, every existing call site
(scripts, future CLI subcommands, notebooks) gets auto-batch by
default without further touches.

**Signature changes**

| Function | Before | After |
|---|---|---|
| `mamba_dvc.gpu.dispatch.correlate_multi_gpu` | `batch_size: int = 64` | `batch_size: int \| Literal["auto"] = "auto"` |
| `mamba_dvc.pipeline.correlate.correlate` | `batch_size: int = 256` | `batch_size: int \| Literal["auto"] = "auto"` |

**Resolution placement**

In `correlate_multi_gpu`, `"auto"` resolves *after* `_resolve_device_ids`
returns and *before* `published(...)` runs. The probe must be cold —
no SHM publish, no worker spawn, no `cp.asarray` upload yet. Steps:

1. Build a `BudgetInputs` from the resolved volume shape, normalized
   window, mode, normalization, `mask is not None`, and
   `mask is not deformed_mask` (matches the dispatch's SHM aliasing
   trick at `dispatch.py:537-540`).
2. Call `recommend_batch_size(inputs, resolved_device_ids)`.
3. Emit one structured stderr line documenting the choice
   (see §11.1 for the exact format).

`pipeline.correlate.correlate` resolves through the same path with a
one-element device list, so the inequality is consistent between the
single- and multi-device entry points.

**Numeric `batch_size` continues to work** and skips the recommender
entirely (no probe, no log). The existing tests in
`tests/gpu/test_dispatch.py` that pin specific sizes
(`batch_size=32`, `batch_size=64`) keep working unchanged — the literal
branch is the gate.

**Validation**

| Branch | Check |
|---|---|
| `int` | `batch_size > 0` (today's check) |
| `"auto"` | unconditionally accepted |
| anything else | `ValueError` listing the two valid forms |

**Error path**

`recommend_batch_size` already raises `RuntimeError` with the
offending numbers. Propagate as-is — the caller (script / CLI) is
responsible for clean rendering.

**Test surface**

- `tests/pipeline/test_correlate.py::TestAutoBatch` (no GPU): patch
  `recommend_batch_size` to return a known int, assert the helper is
  called with that batch and that the input `BudgetInputs` is
  populated correctly.
- One `@pytest.mark.gpu @pytest.mark.slow` test confirming
  `batch_size="auto"` on a real device produces a `DisplacementField`
  bit-equivalent to one produced with the same numeric batch.
- The existing `_assert_fields_match`-using tests keep their numeric
  batches; no change.

**Stale comment to clean up**

`gpu/dispatch.py:443-447` justifies the asymmetric `64` default with
"the production kernel ... exceeds an A6000's transient budget at the
orchestrator's `256` default". With `"auto"`, that paragraph rots —
replace with a one-liner pointing at `mamba_dvc.gpu.budget`.

### 10.2 Checkpoint 2 — `scripts/run_e2e_zarr.py --batch-size auto`

One-line change once Checkpoint 1 lands. The script's argparse for
`--batch-size` (currently `type=int`) accepts either an int or the
literal string `"auto"`. Default stays at the current `64` for one PR
cycle so existing invocations don't surprise; flip to `"auto"` once
a few production runs validate the cost model end-to-end (§11.2).

The script's existing `_vram_snapshot` (at `run_e2e_zarr.py:221-234`)
duplicates what `probe_free_vram` does. Optional follow-up: have the
snapshot delegate to `mamba_dvc.gpu.budget.probe_free_vram` so the two
callers can't drift in return shape. Report layout unchanged.

### 10.3 Checkpoint 3 — `mamba-dvc budget` subcommand (CLI)

Pure inspection — does *not* run correlation. The `inspect` analogue
for the budget oracle: "what would dispatch do on this machine right
now?". Lives next to `cli.inspect`; same Typer app, same rendering
idiom (rich panels + JSON option).

**Invocation**

```
mamba-dvc budget [--store PATH | --volume-shape Z,Y,X] \
                 [--window 96] [--mode linear] \
                 [--normalization overlap] [--devices 0,1,2,3] \
                 [--headroom 0.15] [--ceiling N] [--json]
```

`--store` reuses the manifest discovery path from `inspect` (sidecar
YAML, fall back to embedded `.zattrs`); `--volume-shape` is the
disk-free escape hatch for sizing decisions before any data exists.

**Output**

Rich panel + table mirroring `inspect`:

- Header panel: resolved `BudgetInputs` (volume shape, window, mode,
  normalization, mask flags).
- Per-device VRAM table: `(device, free, total, %free)`, with the
  limiting device highlighted.
- Recommendation panel: the chosen `batch_size`, plus a derivation
  line — `resident=14.2 GB, per_poi=480 MB, fixed=8 MB,
  headroom=15%, ceiling=∞` — and the satisfied inequality.

`--json` emits a machine-readable inventory parallel to
`inspect --json`.

**Exit codes**

| Code | Meaning |
|---|---|
| 0 | Recommendation rendered; the inequality holds. |
| 1 | Recommender raised `RuntimeError` (underflow). Render the message cleanly, no traceback. |
| 2 | Malformed input (bad path, unknown profile, bad `--volume-shape`). |

**Tests**

`tests/cli/test_budget.py` via `CliRunner` + monkeypatched
`probe_free_vram` (mirrors the `tests/cli/test_inspect.py` pattern).
One `@pytest.mark.gpu` test exercises a real probe; the rest are
pure.

**Refactor surface (do before adding the second consumer)**

`cli.py::_resolve_manifest` is the discovery logic that
`budget --store` also wants. Factor it into `mamba_dvc.cli._common`
(or similar) before adding the second consumer; the alternative is
two copies that drift over time.

## 11. Open questions

### 11.1 Should `correlate_multi_gpu` return the resolved batch size?

Today it returns `DisplacementField`. Checkpoint 10.1's stderr-log
contract gives the user the chosen batch in human-readable form, but
not in a programmatically inspectable shape — the e2e script's
report section can't show "auto batch=64 derived from ..." without
parsing its own stderr.

| Option | Cost | Notes |
|---|---|---|
| **(A) Don't change return type; emit a structured stderr line** *(recommended)* | 0 | Plain `print(..., file=sys.stderr)`; format `auto.batch_size=64 auto.free_bytes=... auto.resident_bytes=... auto.per_poi_bytes=... auto.headroom=0.15`. Greppable for tooling. |
| (B) Return `tuple[DisplacementField, ResolutionInfo]` | API churn at every call site | Easy for tooling; ugly for the pure-core rule. |
| (C) Out-parameter callback `on_resolution: Callable[..., None]` | Minor | Awkward; nobody else in the codebase does this. |

Recommendation: **(A)**. The pure-core rule already gates how much
state escapes from `correlate`; a resolution-info return value is
the kind of "tells the user what we did" surface that v2's
time-series driver will want centralized anyway.

### 11.2 When does `--batch-size auto` become the script default?

Checkpoint 10.1 makes `"auto"` the default at the API layer.
Checkpoint 10.2 keeps the script's default at the current `64` for
one cycle. The question is when to flip the script default.

- Pro flip: simpler UX, one source of truth, exercises the
  recommender on every run.
- Pro hold: explicit numeric default lets you reproduce historical
  runs without remembering what the recommender would have picked
  at the time.

Recommendation: hold for one PR cycle, flip once a 4×A6000 production
run has validated the cost model end-to-end at full
`(960, 1280, 1280)` shape with the production kernel.

## 12. Inconsistencies to clean up alongside the wiring

These are sins we'd prefer not to multiply. None block the
checkpoints, but each is cheaper to fix while the surrounding code is
already open.

- **Manifest discovery duplication.** `cli.py::_resolve_manifest`
  will have two consumers (`inspect` and `budget`); factor into a
  shared helper before Checkpoint 10.3 lands.
- **Two VRAM probe paths.** `scripts/run_e2e_zarr.py::_vram_snapshot`
  and `mamba_dvc.gpu.budget.probe_free_vram` do the same thing with
  different return shapes. Unify on the latter when Checkpoint 10.2
  lands.
- **Centred-correlation v2 hook.** `docs/plans/overview.md` §10
  flags a future centred-layout linear kernel. The cost model
  coefficients in `_PER_POI_COEFFS` would need recomputation if/when
  that lands; add a `# TODO when centred layout lands` comment to
  the constants block in `gpu/budget.py` so a future reader does not
  ship a stale model.
