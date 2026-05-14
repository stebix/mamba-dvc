# Run interface — batch displacement-field campaigns — plan & contract

A thin orchestration layer that turns a **config file** into a
**directory of persisted `DisplacementField` results** plus an
aggregation manifest, looping over `(store × deformation × parameter
variant)`. This is the step from the one-pair manual harness
(`scripts/run_e2e_zarr.py`) to "gather results" — many datasets, many
deformations, parameter sweeps (e.g. `mask_threshold ∈ {0.9, 0.7, 0.5,
0.3}`), resumable, with full provenance.

Fulfills the gap noted in `zarr-interface.md` §10: `sweep()` there is
single-store, synthetic-only, in-memory, no persistence, no resume — a
template, not the production driver.

Nothing in `core/`, `pipeline/`, `gpu/`, or `io/` changes
structurally. The driver is a consumer of the existing surface:
`DvcDataset.open` / `.list_*` / `.load_pair`, `correlate_multi_gpu`,
`evaluate_pair`. One small additive change to `evaluate_pair` (an
optional precomputed `truth=` argument) and one optional helper
(`DisplacementField.save_npz` / `load_npz`) are the only upstream
touches.

This document mirrors the structure of `zarr-interface.md` /
`dispatch.md`: position → invariants → API → flow → data structures →
output layout → test surface → deferrals.

## 1. Position in the codebase

```
   configs/results_v1.yaml  (campaign definition, checked in, diffable)
              │
              ▼
   ┌────────────────────────────────────────────────────────────┐
   │ cli/run.py  ─  `mamba-dvc run config.yaml [--dry-run …]`    │
   └───────────────────────────┬────────────────────────────────┘
                               ▼
   ┌────────────────────────────────────────────────────────────┐
   │ mamba_dvc/run/batch.py                                      │
   │   BatchSpec      ← parse + validate + expand variants       │
   │   plan_jobs()    ← (store, deformation, variant) × resume   │
   │   run_batch()    ← the materialize-once / iterate-many loop  │
   │   _run_one()     ← load_pair → correlate → (evaluate) → write│
   └───────────────────────────┬────────────────────────────────┘
              ┌────────────────┼─────────────────┐
              ▼                ▼                 ▼
   DvcDataset.open /     correlate_multi_gpu   evaluate_pair(…, truth=)
   .list_* / .load_pair  (existing, unchanged)  (additive: precomputed truth)
              │                │                 │
              ▼                ▼                 ▼
        EvaluationPair    DisplacementField    ErrorReport
              └────────────────┴─────────────────┘
                               ▼
   results/<campaign>/  ─  per-variant .npz + .json sidecar
                            manifest.jsonl + variants.json
```

`mamba_dvc/run/` is a **new top-level subpackage** for "drive the
pipeline over many inputs". It is *not* `validate/` (that is
synthetic-GT scoring; real deformations have no GT) and *not*
`pipeline/` (that is the pure single-pair core). The batch driver is
orchestration: I/O sequencing, persistence, resume, provenance — the
same role `correlate_series` plays for time series, but over the
`(dataset, deformation, params)` product instead of over timesteps.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| **The batch loop is serial over `(store, deformation, variant)`.** No process pool wrapping `correlate_multi_gpu`. | `correlate_multi_gpu` already spawns one process per GPU internally. Nesting a second pool oversubscribes the 4 GPUs and the host pinned-memory pool — exactly the `cudaErrorMemoryAllocation` failure seen in manual runs. |
| **`load_pair` runs once per `(store, deformation, load-tier params)`; compute-tier variants loop inside.** | `load_pair` is ~30 s + ~30 GB RSS; `gt_field` construction adds the cubic-B-spline prefilter. A 4-value `mask_threshold` sweep must not re-pay those four times. |
| **The GT truth array is computed once per `(pair, grid-spec)` and reused across compute-tier variants.** | `field.positions` is the full POI lattice — identical for every `mask_threshold` (only `valid`/`displacements`/`confidence`/`status` differ). `gt_field(positions)` is currently the second-largest cost in `evaluate_pair` (see §9 deferral); reusing it is free correctness. |
| **One result = one `(store, deformation, variant)` tuple.** Variant identity is a stable hash of the *resolved* parameter set. | Re-running the same config reuses filenames → resume works. Adding a sweep value adds exactly one new result and disturbs nothing. |
| **Every result carries its full resolved parameters + provenance in a sidecar.** A hash-named `.npz` is meaningless without it. | Reproducibility; lets `variants.json` and the sidecars reconstruct what `mt0.5-a1b2c3d4` *was*. |
| **A failed pair/variant is recorded and skipped, not fatal.** | A multi-hour campaign must survive one bad entry or a transient OOM. Mirrors `run_e2e_zarr.py`'s exception-to-report behaviour. |
| **The flow convention is never guessed by the batch driver.** It comes from the config (or, once wired, the store profile/entry). | The `pull_back` default in `run_e2e_zarr.py` / `GroundTruthField` silently inflates synthetic error ~50× for the canonical stores (see `zarr-flow-convention.md`). A results campaign that gets this wrong is worthless. |

## 3. Parameter tiers

Every parameter that the driver can vary belongs to exactly one tier,
which decides *which loop level* it sits at:

| Tier | Params | Changing it forces… |
|---|---|---|
| **store** | the zarr store path | a new `DvcDataset.open` |
| **entry** | deformation name | nothing (just a different `load_pair` arg) |
| **load** | `mask` (which mask name / `none`), `dry_shape`, `flow_convention` | re-`load_pair` (zarr read; rebuilds `gt_field`) |
| **grid** | `window`, `overlap` | re-`correlate_multi_gpu` *and* invalidates the cached GT truth (POI lattice changes) |
| **compute** | `mask_threshold`, `batch_size`, `ncc_mode`, `ncc_normalization`, `search_radius`, `tukey_alpha` | re-`correlate_multi_gpu` only; `positions` unchanged |

A `mask_threshold` sweep is pure **compute**-tier: it loops in the
innermost position, reusing the materialized pair and the cached truth.
`window`/`overlap` sweeps loop one level out (re-correlate, re-evaluate
truth, but no re-load). `mask` / `flow_convention` sweeps loop at the
**load** tier (full re-materialization).

## 4. Config schema

YAML (or TOML). One file = one campaign. Checked in alongside the
results so the campaign is reproducible and diffable.

```yaml
# configs/results_v1.yaml
campaign: results_v1                 # → results/<campaign>/

stores:
  - D:\jannik\synchrotron-data\transduced\103L_Mg5Gd_4w_000.zarr
  - D:\jannik\synchrotron-data\transduced\…
manifest: null                       # optional shared StoreManifest YAML
strict: true                         # false → open stores that fail verification anyway,
                                     #         processing only their healthy entries

select:                              # which deformations per store
  real: all                          # "all" | "none" | [explicit names]
  synthetic: all

defaults:                            # base params, every variant starts here
  window: 96
  overlap: 0.5
  mask_threshold: 0.5
  batch_size: auto
  ncc_mode: linear
  ncc_normalization: overlap
  flow_convention: push_forward       # synthetic GT; do NOT rely on the code default
  mask: null                          # null → profile/manifest default; "none" → no mask
  dry_shape: null                     # null → full volume; [z,y,x] → centered subblock

sweep:                                # cartesian product over named axes
  mask_threshold: [0.9, 0.7, 0.5, 0.3]
  # window: [64, 96]                  # uncommenting → 4 × 2 = 8 variants

variants:                             # explicit extra points, each merged onto `defaults`
  - { name: padfield-precheck, mask_threshold: 0.2, window: 128 }

devices: [0, 1, 2, 3]                 # forwarded to correlate_multi_gpu
distance_bins: []                     # evaluate_pair stratification; [] = skip the EDT
out_dir: ./results                    # results/<campaign>/ created underneath
```

**Expansion.** The driver computes `defaults × sweep` (full cartesian
product over the `sweep` axes) and appends each entry of `variants`
(merged onto `defaults`). The result is a flat list of *resolved*
parameter dicts. Duplicates (same resolved dict) are collapsed. Each
resolved dict is then split into its tier slices for the loop.

**Variant id.** `variant_id = f"{name}-{h}"` where `h` is the first 8
hex chars of `sha256` over the canonical (sorted-key) JSON of the
resolved parameter dict, and `name` is the explicit `variants[].name`
or, for `sweep`-generated variants, an auto-slug of the non-default
axes (`mt0.5`, `mt0.3-w64`, …). Stable across runs.

## 5. API

```python
# mamba_dvc/run/batch.py

@dataclass(frozen=True)
class Variant:
    """One fully-resolved parameter set + its stable identity."""
    variant_id: str
    name: str
    params: Mapping[str, Any]                 # full resolved dict
    load_params: Mapping[str, Any]            # mask, dry_shape, flow_convention
    grid_params: Mapping[str, Any]            # window, overlap
    compute_params: Mapping[str, Any]         # the rest, forwarded to correlate_multi_gpu

@dataclass(frozen=True)
class BatchSpec:
    """Parsed + validated campaign config."""
    campaign: str
    stores: tuple[Path, ...]
    manifest: Path | None
    strict: bool                              # → DvcDataset.open(strict=); False = skip bad sub-entries
    select_real: Literal["all", "none"] | tuple[str, ...]
    select_synthetic: Literal["all", "none"] | tuple[str, ...]
    variants: tuple[Variant, ...]
    devices: tuple[int, ...] | None
    distance_bins: tuple[float, ...]
    out_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> BatchSpec: ...

@dataclass(frozen=True)
class Job:
    store: Path
    deformation: str
    kind: Literal["real", "synthetic"]
    variant: Variant

@dataclass(frozen=True)
class JobResult:
    job: Job
    status: Literal["ok", "failed", "skipped"]
    npz_path: Path | None
    sidecar_path: Path | None
    error: str | None                         # traceback if failed
    summary: Mapping[str, Any]                 # n_total, n_ok, mae, …, timings

def plan_jobs(spec: BatchSpec, *, force: bool = False,
              only: Mapping[str, str] | None = None,
              stores: Sequence[Path] | None = None,
              open_dataset: DatasetOpener | None = None) -> list[Job]:
    """Enumerate (store, deformation, variant); drop already-done unless force."""

def run_batch(spec: BatchSpec, *, force: bool = False,
              only: Mapping[str, str] | None = None,
              stores: Sequence[Path] | None = None,
              correlate_fn: CorrelateFn | None = None,
              open_dataset: DatasetOpener | None = None,
              observer: BatchObserver | None = None) -> list[JobResult]:
    """Execute the plan with the materialize-once / iterate-many loop.

    The loop is print-free. Live progress is an opt-in `observer`
    (default `NullObserver` — silent); the CLI plugs in a Rich
    spinner/progress bar (or a plain per-job log when stdout is not a
    terminal). `--dry-run` lives in the CLI on top of `plan_jobs`, not
    as a `run_batch` flag.
    """
```

```python
# mamba_dvc/run/progress.py — the optional live-progress hook

class BatchObserver(Protocol):
    """Hooks run_batch calls at well-defined points; implementations may
    override any subset (subclass NullObserver for that)."""
    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None: ...
    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None: ...
    def on_job_start(self, job: Job) -> None: ...          # fired *before* the work
    def on_job_end(self, result: JobResult) -> None: ...
    def on_batch_end(self, results: list[JobResult]) -> None: ...

class NullObserver: ...   # all no-ops; the default — keeps the library print-free

# Contract: on_batch_start once (n_jobs is exact, counts store-open failures);
# then per materialized (deformation, mask, dry_shape) slice one
# on_pair_load_start, then an on_job_start/on_job_end pair per variant;
# then on_batch_end once. Every recorded JobResult — including load_pair and
# store-open failures — is bracketed by on_job_start/on_job_end.
```

```python
# cli/run.py  →  `mamba-dvc run`
#   mamba-dvc run CONFIG.yaml
#     [--dry-run]                 # print the plan (N stores × M deformations × K variants), exit
#     [--force]                   # recompute everything
#     [--force-variant ID ...]    # recompute specific variant ids
#     [--only KEY=VALUE ...]      # subset filter on resolved params, e.g. --only mask_threshold=0.5
#     [--devices 0,1]             # override config devices (e.g. to dodge pinned-mem OOM)
#     [--store PATH ...]          # restrict to a subset of the config's stores
```

```python
# additive change in validate/known_fields.py
def evaluate_pair(pair, field, *, distance_bins=(0,5,10,20,50),
                  truth: Float32[np.ndarray, "points 3"] | None = None) -> ErrorReport:
    """If `truth` is given, skip the internal `pair.gt_field(field.positions)`
    call and use it directly. The caller is responsible for it matching
    `field.positions` (same grid-spec)."""

# optional helper in types.py
class DisplacementField:
    def save_npz(self, path: Path) -> None: ...
    @classmethod
    def load_npz(cls, path: Path) -> "DisplacementField": ...
```

## 6. Execution flow

```text
spec = BatchSpec.from_yaml(config)
jobs = plan_jobs(spec, force, only)            # already-done variants filtered out

if dry_run:                                    # print plan, exit 0
    print_plan(jobs); return

results = []
for store in spec.stores:
    ds = DvcDataset.open(store, manifest=...)
    for deformation, kind in select(ds, spec):
        # group this entry's pending variants by load-tier params
        for load_key, load_params, group in group_by_load(pending(jobs, store, deformation)):
            try:
                pair = ds.load_pair(deformation, **load_params)     # ── expensive, ONCE
            except Exception:
                results += [failed(j, traceback) for j in group]; continue

            truth_cache: dict[GridKey, np.ndarray] = {}
            for variant in sorted(group, key=lambda v: v.grid_params):  # adjacency → cache hits
                try:
                    field = correlate_multi_gpu(
                        pair.reference, pair.deformed, mask=pair.mask,
                        device_ids=spec.devices, **variant.grid_params, **variant.compute_params,
                    )
                    report = None
                    if pair.gt_field is not None:
                        gk = grid_key(variant)                       # (window, overlap)
                        if gk not in truth_cache:
                            truth_cache[gk] = pair.gt_field(field.positions)   # ── GT eval, ONCE/grid
                        report = evaluate_pair(pair, field,
                                               distance_bins=spec.distance_bins,
                                               truth=truth_cache[gk])
                    npz, side = write_result(spec, store, deformation, kind, variant,
                                             field, report, provenance=...)
                    results.append(ok(job, npz, side, summarize(field, report)))
                except CUDARuntimeError as exc:                      # pinned-mem OOM, etc.
                    # retry once; then with spec.devices[:2]; then record failed
                    ...
                except Exception:
                    results.append(failed(job, traceback))
            del pair                                                 # free 30 GB before next entry

write_manifest(spec, results)                  # append to manifest.jsonl, refresh variants.json
return results
```

Notes:

- **Resume** is implemented in `plan_jobs`: a `(store, deformation,
  variant_id)` is "done" iff `results/<campaign>/<stem>/<deformation>/<variant_id>.npz`
  exists *and* its sidecar's `params_hash` matches the variant's hash.
  `manifest.jsonl` is the fast path (read it once); the on-disk check
  is the source of truth.
- **`del pair`** between entries is load-bearing — without it the next
  `load_pair` stacks another ~30 GB on RSS.
- **Prefetch.** `BatchSpec.prefetch` (default 1) makes the loop fetch the
  *next* `(deformation, mask, dry_shape)` group's pair in a single
  background thread while the current group's variants run — the load is
  the campaign's biggest single cost (~comparable to a group's whole
  correlate + evaluate), and the variant loop is overwhelmingly
  GIL-releasing (GPU dispatch blocked on workers, big NumPy reductions),
  so the overlap is near-perfect (~1.9x measured). Only one load runs at
  a time (one loader thread; more would thrash the disk and multiply
  RSS), and a step is skipped — synchronous load — when free host RAM
  (`psutil`) would not comfortably hold another ~30 GB pair. Results are
  bit-identical with prefetch on/off; a failed load is still a per-job
  `load_pair` failure, just observed one group later. `batch.load_pair`
  timing records then measure load *wall time*, not critical-path time
  (see `mamba_dvc.instrument`). Implemented in `_prefetched_load_groups`;
  the "non-pinned host staging" deferral (§9) is orthogonal and open.
- **Pinned-memory OOM** mitigation is a bounded retry ladder inside the
  variant loop: same params → `devices[:2]` → mark failed. The proper
  fix (non-pinned host staging in `dispatch`) is out of scope here;
  tracked separately.

## 7. Output layout

```
results/<campaign>/
  config.snapshot.yaml                 # the exact config used (copied at launch)
  manifest.jsonl                       # one JSON object per (store, deformation, variant) result
  events.jsonl                         # append-only event stream from `mamba-dvc run` (CLI only)
  variants.json                        # variant_id → { name, params }   (decode hash ids)
  <store_stem>/
    <deformation>/
      <variant_id>.npz                 # DisplacementField arrays (positions, displacements,
      <variant_id>.json                #   valid, confidence, status, grid_shape, spacing, window)
                                       # sidecar: { params (resolved), provenance, timings,
                                       #            vram_before/after, summary, error_report? }
```

`<store_stem>` is the zarr directory name minus `.zarr`
(`103L_Mg5Gd_4w_000`). The `.npz` schema is exactly what
`run_e2e_zarr.py --out` already writes — kept identical so existing
notebooks load both.

**`manifest.jsonl` row** (swept params flattened into columns so a
DataFrame groups cleanly):

```json
{"campaign":"results_v1","store":"103L_Mg5Gd_4w_000","deformation":"fs104",
 "kind":"synthetic","variant_id":"mt0.5-a1b2c3d4","name":"mt0.5",
 "window":96,"overlap":0.5,"mask_threshold":0.5,"batch_size":"auto",
 "ncc_mode":"linear","ncc_normalization":"overlap","flow_convention":"push_forward",
 "n_total":11875,"n_ok":1234,"n_masked":10543,"n_outlier":98,
 "mae":0.213,"rmse":1.406,"p95":0.810,"per_axis_mae":[0.067,0.316,0.257],
 "wall_load_s":29.3,"wall_correlate_s":73.6,"wall_evaluate_s":0.01,
 "status":"ok","npz":"103L_Mg5Gd_4w_000/fs104/mt0.5-a1b2c3d4.npz"}
```

So the `mask_threshold` sweep readout is one line:

```python
df = pd.read_json("results/results_v1/manifest.jsonl", lines=True)
df.query("deformation == 'fs104'").groupby("mask_threshold")[["n_ok", "mae", "rmse", "p95"]]
```

For `real` entries `mae/rmse/p95/per_axis_mae` and `error_report` are
absent (no GT); `n_ok`/`n_masked`/… and timings are still there.

**`events.jsonl`** is the CLI-only append-only event stream (see
`mamba_dvc/run/eventlog.py`). One JSON line per event with a `kind`
discriminator: `batch_start`, `pair_load_start`, `job_start`, `job_end`,
`batch_end`, plus `phase` lines for every `mamba_dvc.timing` record
(`batch.load_pair`, `dispatch.*`, `evaluate.*`, etc.) and `warning`
lines for every `warnings.warn` raised during the campaign. Every line
carries `ts` (UTC ISO timestamp), `campaign`, and `session_id` (uuid4
per `mamba-dvc run` invocation, so resumed files are groupable). Inside
a load group, `store` / `deformation` are inherited; inside a job,
`variant_id` / `kind_of_job` are inherited.

```python
df = pd.read_json("results/results_v1/events.jsonl", lines=True)
df.groupby("kind").size()                                   # counts
df.query("kind == 'phase'")["phase"].value_counts()          # phase histogram
df.query("kind == 'job_end' and status == 'failed'")[["variant_id", "phase"]]
```

The events stream is the CLI's responsibility — `run_batch` itself
stays print- and write-free in its lifecycle. `--no-events` opts out;
`--dry-run` never writes an events file. Concurrent `mamba-dvc run`
invocations against the same campaign dir are **undefined behaviour**
(append-mode atomicity is not guaranteed on Windows for lines larger
than PIPE_BUF).

## 8. Test surface

CPU, fast (no GPU):

- `test_batch_spec.py::test_expand_sweep` — `defaults × sweep + variants`
  expansion, dedup, tier-split, variant-id stability (same dict →
  same id; one changed value → exactly one new id).
- `test_batch_spec.py::test_select` — `real/synthetic = all|none|[names]`
  against a fake `DvcDataset`.
- `test_plan_jobs.py::test_resume` — pre-seed `manifest.jsonl` +
  matching sidecars; assert done variants are dropped, `--force`
  re-includes them, a hash mismatch re-includes the affected one.
- `test_run_batch.py::test_loop_reuse` — fake `DvcDataset` +
  fake `correlate_fn` (à la `CorrelateFn` in `known_fields`); assert
  `load_pair` called once per `(store, deformation, load-tier)`, the
  truth callable invoked once per `(pair, grid-spec)`, and one
  `.npz` + `.json` written per variant.
- `test_run_batch.py::test_failure_isolated` — one variant's
  `correlate_fn` raises; assert it's recorded `failed` and the rest of
  the batch completes.
- `test_evaluate_pair.py::test_precomputed_truth` — `evaluate_pair(...,
  truth=t)` equals `evaluate_pair(...)` when `t == pair.gt_field(field.positions)`,
  and the `gt_field` callable is *not* invoked when `truth` is given.

GPU-marked, opt-in:

- `tests/integration/test_run_batch_smoke.py` — a 2-variant
  `mask_threshold` mini-campaign on a small zarr fixture (or a
  `--dry-run-shape`-equivalent in the config), `devices=[0]`; assert
  the two `.npz` differ in `valid` count and the manifest has two `ok`
  rows.

## 9. Prerequisites & deferrals

**Should land before a real campaign** (not strictly blocking the
infrastructure, but blocking *useful output*):

1. **`evaluate_pair(..., truth=)`** — required to hoist the GT eval out
   of the variant loop (§6). Small, additive, tested above.
2. **`GroundTruthField.__call__` prefilter perf** — currently re-runs
   the cubic-B-spline prefilter over the full `(960,1280,1280)` flow
   array three times *per call* (~320 s observed). Fix: `spline_filter`
   each component once at construction, `prefilter=False` in `__call__`.
   Without this, every synthetic variant pays ~320 s of pure waste even
   with the truth-cache (the cache only saves *re-eval within a pair*,
   not the first eval). See `zarr-flow-convention.md`.
3. **Flow-convention source of truth** — make the synthetic deformation
   entry / store profile carry its convention so neither the script nor
   the batch config has to assert `push_forward` by hand. Until then,
   `defaults.flow_convention` in the config is the guard, and the
   driver should *warn loudly* if it's unset.
4. **Mask-threshold operating point** — run the `sweep` above on one
   `(store, fs104)` first, eyeball `n_ok` vs `mae`, pick the threshold
   for the production campaign config. The infra doesn't care; the
   results do (2.3 % POI coverage at 0.9 is not a campaign).

**Deferred (v2 / later):**

- **Pinned-host staging in `dispatch`** so 4 workers don't each pin
  ~13 GB of replicated volume — the real fix for the OOM the retry
  ladder only papers over.
- **Writing fields back into the zarr store** under a `results/` group
  (the time-series pipeline's `on_pair` writer, `times-series-buildout.md`)
  — premature; flat `.npz` + manifest is enough to "gather results".
- **Cross-store / cross-deformation aggregation reports** (the
  `ErrorReport`-rollup tables) — that's a notebook concern over
  `manifest.jsonl`, not the driver's job.
- **Parallelism across stores** with disjoint GPU subsets (run store A
  on GPUs 0–1, store B on 2–3) — possible, but the host RSS (2 × 30 GB
  materialized pairs) and pinned-memory pressure make it dangerous
  until the staging fix lands. Out of scope.
```
