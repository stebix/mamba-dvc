# mamba-dvc

FFT-based digital volume correlation (DVC) for synchrotron µCT of rat bone with
screw implant. Target volumes are `(960, 1280, 1280)` float32; compute runs on
4× NVIDIA A6000 via CuPy.

The architecture and algorithmic decisions are documented in
[`docs/plans/overview.md`](docs/plans/overview.md). The zarr reader contract —
profile / manifest schema, verifier rules, ground-truth field handling — lives
in [`docs/plans/zarr-interface.md`](docs/plans/zarr-interface.md).

## Install

Python 3.14 is required (CuPy 14 / CUDA 13.x). Dependencies are managed with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Inspecting a zarr store: `mamba-dvc inspect`

```bash
uv run mamba-dvc inspect /path/to/scan.zarr
```

Opens the store, resolves its `StoreProfile` (and any sidecar
`StoreManifest`), runs the verifier, and prints:

1. A header listing the resolved profile and the manifest source
   (sidecar YAML, root `.zattrs`, `--manifest` flag, `--profile` flag, or
   `none`).
2. A validity panel — green `OK` on a clean store, red `FAILED` with one
   bullet per verifier error otherwise.
3. A tree of every parsed slot: reference (`[reference]`), masks
   (`[default]` tag on the resolved default), real-deformation entries
   under `iterations/`, and synthetic entries under `synthetic/` with
   their flow arrays tagged by axis order and convention.

The store is opened with `strict=False`, so a non-conforming store still
renders the tree of whatever could be parsed alongside the error list.

### Common invocations

```bash
# Add chunks, codecs, and on-disk size per array leaf.
uv run mamba-dvc inspect /path/to/scan.zarr --verbose

# Machine-readable inventory + verifier report (skips rich rendering).
uv run mamba-dvc inspect /path/to/scan.zarr --json

# Use an explicit manifest YAML; bypass sidecar / .zattrs discovery.
uv run mamba-dvc inspect /path/to/scan.zarr --manifest /path/to/manifest.yaml

# Force a specific registered profile (synthesizes / merges a manifest).
uv run mamba-dvc inspect /path/to/scan.zarr --profile bone_screw_synchrotron_v1
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Verifier passed; store conforms to its profile + manifest. |
| `1` | Verifier reported errors. The tree of what was parsed is still rendered to stdout, and `--json` payloads carry `validity.ok = false`. |
| `2` | Could not open: missing path, malformed manifest YAML, or unknown profile name. The error message goes to stderr. |

The `1` path is the load-bearing one for scripting — CI can gate uploads on
`mamba-dvc inspect --json <path>` returning `validity.ok == true`.

## Running a campaign: `mamba-dvc run`

A **campaign** is one YAML config describing a batch of displacement-field
runs: a set of zarr stores × a subset of their real/synthetic deformations ×
a parameter sweep. Each cell of that product is a **variant**, persisted as
one `.npz` field plus a JSON sidecar under
`<out_dir>/<campaign>/<store>/<deformation>/<variant_id>.npz`. The driver
is resumable, prefetches across pairs, and survives per-cell failures —
designed for multi-hour sweeps over the full 4× A6000 fleet.

The architecture and invariants live in
[`docs/plans/run-interface.md`](docs/plans/run-interface.md). This section
covers the user-facing surface.

### Quick start

```bash
# Preview the plan (stores × deformations × variants) and exit.
uv run mamba-dvc run configs/results_v3.yaml --dry-run

# Execute the campaign.
uv run mamba-dvc run configs/results_v3.yaml

# Re-running the same config resumes: every variant already on disk
# (matched by params hash) is skipped.
```

A real campaign on the production volume takes hours; check the plan with
`--dry-run` first so you catch misnamed stores or an empty `select` before
spinning up the GPUs.

### Concepts

| Term | Meaning |
|---|---|
| **Campaign** | One YAML config; its `campaign:` name becomes the output sub-directory. |
| **Store** | One zarr volume (reference + masks + deformation entries). One open per `(store, flow_convention)` pair. |
| **Deformation entry** | A `real` or `synthetic` deformation under the store. Only `synthetic` entries are scored (real entries have no ground truth). |
| **Variant** | One fully-resolved parameter dict. Identity is `name-hash`, where `hash` is the first 8 hex chars of `sha256` over the canonical (sorted-key) JSON of the resolved params. Stable across runs. |
| **Job** | One `(store, deformation, variant)` triple. One job → one `.npz` + sidecar. |

### Config schema

Top-level keys (any other key is a hard error — typos like `store:` for
`stores:` don't silently no-op):

| Key | Type | Meaning |
|---|---|---|
| `campaign` | string (required) | Campaign name; results land under `<out_dir>/<campaign>/`. |
| `stores` | list of paths (required) | Zarr stores to iterate. Paths are taken as-is — relative to the process CWD, not the config file. |
| `manifest` | path or `null` | Optional shared `StoreManifest` YAML applied to every store. `null` falls back to per-store discovery (sidecar YAML / embedded `.zattrs`). |
| `strict` | bool (default `true`) | Forwarded to `DvcDataset.open`. `false` opens stores that fail verification and processes only their healthy entries — set this when a known-bad sub-entry would otherwise lose the whole store. |
| `select.real` / `select.synthetic` | `"all"` \| `"none"` \| list of names | Which deformation entries to run per store. Real entries are correlated and persisted but not scored. |
| `defaults` | mapping | Base parameter knobs. Every variant starts here. |
| `sweep` | mapping of knob → list | Cartesian product over the named axes; each combination becomes one variant. |
| `variants` | list of mappings | Explicit extra variants, each merged onto `defaults`. May override the sweep. |
| `devices` | list of GPU ids or `null` | Forwarded to `correlate_multi_gpu`. `null` selects every visible device. |
| `distance_bins` | list of floats | Boundary-stratification edges for `evaluate_pair`. Empty (the default) skips the expensive EDT. |
| `prefetch` | int (default `1`) | Load-group prefetch depth; `0` disables. See [Performance](#performance-notes). |
| `out_dir` | path (default `results`) | Root output dir; the campaign dir is `<out_dir>/<campaign>`. |

A canonical example (`configs/results_v3.yaml`):

```yaml
campaign: results_v3

stores:
  - C:\Users\jstebani\Desktop\dvc-data-storage\103L_Mg5Gd_4w_000.zarr
  - C:\Users\jstebani\Desktop\dvc-data-storage\5L_PEEK_4w_000.zarr
  - C:\Users\jstebani\Desktop\dvc-data-storage\5R_Ti_4w_000.zarr

manifest: null         # per-store discovery
strict: false          # one store has a truncated sub-entry; skip it, keep the rest

select:
  real: none           # no GT → skip in a scoring campaign
  synthetic: all

defaults:
  window: 96
  overlap: 0.8
  mask: mask_fill
  mask_threshold: 0.9
  flow_convention: push_forward   # canonical transduced stores; leaving null silently inflates error ~50x
  batch_size: auto
  ncc_mode: linear
  ncc_normalization: overlap
  search_radius: null            # null → window // 2
  tukey_alpha: null              # null → per-ncc-mode default
  dry_shape: null                # null → full volume; [z, y, x] → centered subblock

sweep:
  mask_threshold: [0.9, 0.7, 0.5]   # 3 variants per (store, deformation)

variants: []

devices: [0, 1, 2, 3]
distance_bins: []     # skip the slow boundary-distance EDT
prefetch: 1
out_dir: C:\Users\jstebani\Desktop\dvc-result-storage
```

### Parameter knobs and tiers

Every knob the driver can vary belongs to exactly one tier, which decides
*which loop level* it sits at. Understanding this is how a multi-knob sweep
stays fast.

| Tier | Knobs | Changing it forces |
|---|---|---|
| **load** | `mask`, `dry_shape`, `flow_convention` | Re-`load_pair` (zarr read; rebuilds GT flow callable) — the campaign's single biggest cost (~30 GB I/O on production volumes). |
| **grid** | `window`, `overlap` | Re-correlate AND invalidates the cached GT truth (the POI lattice moves). |
| **compute** | `mask_threshold`, `batch_size`, `ncc_mode`, `ncc_normalization`, `search_radius`, `tukey_alpha` | Re-correlate only; `field.positions` is unchanged, so the cached GT truth survives. |

The execution loop is **materialize-once / iterate-many**: one `load_pair`
per load-tier slice, then loops the cheaper grid- and compute-tier variants
inside. A four-value `mask_threshold` sweep pays the `load_pair` cost once
per `(store, deformation)`, not four times.

### Variant expansion

```
variants = (defaults × sweep) + variants_block
```

The driver computes the cartesian product over the `sweep` axes, then
appends each entry of `variants:` (merged onto `defaults`). Duplicates
(identical resolved params) collapse to the first occurrence. Each resolved
dict is then split into its three tier slices for the loop.

The variant's display name is either an explicit `variants[].name`, or an
auto-slug of the knobs that differ from the resolved defaults
(`mt0.5`, `mt0.3-w64`), or `base` when nothing differs. The full id is
`name-hash` (e.g. `mt0.5-a1b2c3d4`), used as the result filename stem.

### CLI flags

```
mamba-dvc run CONFIG.yaml
    [--dry-run]                  # print the plan, exit 0
    [--force]                    # recompute variants already on disk
    [--only KEY=VALUE]           # repeatable; filter variants by resolved knob value
    [--devices 0,1]              # override the config's devices (e.g. to dodge pinned-mem OOM)
    [--store PATH]               # repeatable; restrict to a subset of the config's stores
    [--prefetch N]               # override the config's prefetch depth
    [--quiet | -q]               # suppress the live progress line
    [--timing]                   # print a per-phase wall-time breakdown at the end
    [--no-color]                 # disable styled terminal output
    [--no-events]                # disable the events.jsonl event stream
```

`--only` keys are knob names; values are compared via the same slug rule
the auto-slug uses (`mask_threshold=0.5` matches the float `0.5`,
`dry_shape=64x64x64` matches the tuple `(64, 64, 64)`).

`--store` accepts either a full path (matched as a path) or the `.zarr`
directory basename — handy when you only want to re-run one store from a
multi-store config.

### Output layout

```
<out_dir>/<campaign>/
  config.snapshot.yaml         # the exact config used (copied at launch)
  manifest.jsonl               # one JSON object per (store, deformation, variant) result
  variants.json                # variant_id → { name, params }   (decodes the hash-suffixed ids)
  events.jsonl                 # append-only event stream (unless --no-events)
  <store_stem>/                # zarr directory name minus ".zarr"
    <deformation>/
      <variant_id>.npz         # DisplacementField arrays
      <variant_id>.json        # sidecar: resolved params, provenance, timings, summary, error report
```

The `.npz` schema is identical to what `scripts/run_e2e_zarr.py --out`
writes, so existing notebooks load both. Sidecars carry the resolved
params, library/git provenance, wall-clock timings, the POI status
breakdown, and (for synthetic entries) the full error report.

### Reading the manifest

`manifest.jsonl` is the campaign's flat data product: one JSON object per
job, with swept params flattened into top-level columns so a DataFrame
groups cleanly.

```json
{"campaign":"results_v3","store":"103L_Mg5Gd_4w_000","deformation":"fs104",
 "kind":"synthetic","variant_id":"mt0.5-a1b2c3d4","name":"mt0.5",
 "window":96,"overlap":0.8,"mask_threshold":0.5,"batch_size":"auto",
 "ncc_mode":"linear","ncc_normalization":"overlap","flow_convention":"push_forward",
 "n_total":11875,"n_ok":1234,"n_masked":10543,"n_outlier":98,
 "mae":0.213,"rmse":1.406,"p95":0.810,"per_axis_mae":[0.067,0.316,0.257],
 "wall_correlate_s":73.6,"wall_evaluate_s":0.01,
 "status":"ok","npz":"103L_Mg5Gd_4w_000/fs104/mt0.5-a1b2c3d4.npz"}
```

```python
import pandas as pd
df = pd.read_json("<out_dir>/<campaign>/manifest.jsonl", lines=True)

# Mask-threshold sweep on one deformation, one store
(df.query("store == '103L_Mg5Gd_4w_000' and deformation == 'fs104'")
   .groupby("mask_threshold")[["n_ok", "mae", "rmse", "p95"]])
```

For `real` entries, the GT-only fields (`mae` / `rmse` / `p95` /
`per_axis_mae` / `error_report`) are absent; the POI counts and timings
are still there. Failed jobs carry `status: "failed"` and a tail of the
traceback in `error_tail`.

### Loading a single field back

`mamba_dvc.run.campaign.load_campaign_field` filters the manifest and
materializes one `DisplacementField` plus its sidecar dict:

```python
from mamba_dvc.run.campaign import load_campaign_field

field, sidecar = load_campaign_field(
    "<out_dir>/<campaign>",
    store="103L_Mg5Gd_4w_000",
    deformation="fs104",
    variant_id="mt0.5-a1b2c3d4",
)
```

The filter raises `LookupError` on zero or multiple matches — the
canonical "plot one variant" use case wants exactly one match, and the
canonical "plot many variants" use case wants the caller to drive the
loop explicitly.

### Resume

Re-running the same config resumes: `plan_jobs` drops every variant whose
`.npz` and sidecar are already on disk and whose sidecar's `params_hash`
matches the variant's current hash. The on-disk check is the source of
truth — a stale `manifest.jsonl` is rebuilt from the surviving artifacts.

- `--force` recomputes everything.
- Changing any knob in the config produces a **new** `variant_id`, so the
  old result stays on disk and the new one is added alongside it. Hand-edit
  the campaign name (or delete the variant) if that's not what you want.
- **Concurrent invocations against the same campaign dir are undefined
  behaviour** — strictly serial re-runs only.

### Performance notes

- **Prefetch.** With `prefetch: 1` (the default) the next
  `(deformation, mask, dry_shape)` load group's `load_pair` runs in a
  background thread while the current group's variants execute. The
  variant loop is overwhelmingly GIL-releasing (GPU dispatch blocked on
  worker processes, big NumPy reductions), so the overlap is near-perfect
  (~1.9× measured) when a load costs about as much as its group of
  variants. Larger depths keep more loads *queued* but only one runs at a
  time (one loader thread), so `1` is the practical sweet spot.
- **RAM guard.** The prefetcher checks free host RAM (`psutil`) before
  queueing another pair (~30 GB on the production volume) and skips
  step-by-step when it wouldn't fit. Results are bit-identical with
  prefetch on or off — only *when* `load_pair` runs changes.
- **Pinned-memory OOM fallback.** If `correlate_multi_gpu` raises on an
  explicitly-pinned 4-GPU device set, the driver retries once on the first
  two devices before recording the job as failed. Without an explicit
  `devices:` it does not fall back — it won't enumerate the fleet itself.
- **`--timing`.** Adds a small GPU-sync overhead per phase and prints a
  per-phase wall-time breakdown table at the end. Per-pair NCC sub-phases
  appear only on the single-GPU path.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Every job succeeded (or `--dry-run`). |
| `1` | At least one job failed; the rest still ran and are persisted. |
| `2` | Malformed config or flags; nothing was executed. |

Scripting can branch on `1` versus `0`: a `1` exit still leaves a valid
`manifest.jsonl` with `status: "failed"` rows you can re-drive.

## Logging and observability

Every `mamba-dvc run` invocation emits three observability surfaces from the
same set of lifecycle hooks: a live terminal display, an append-only
`events.jsonl` stream alongside the results, and an optional end-of-run
per-phase timing breakdown. The library itself is print-free — these are
configured at the CLI boundary.

### Live display

The renderer is selected automatically by output target. The on-disk
artifacts (manifest, `.npz` fields, sidecars, `events.jsonl`) are identical
in every case — only the live rendering differs.

| Mode | Trigger | What you see |
|---|---|---|
| Rich | interactive terminal | Spinner + progress bar with the current `store/deformation/variant`, m/n complete, elapsed, ETA. Failed jobs print one red `FAILED` line above the live bar. |
| Plain | non-terminal stdout (pipe, CI log) | One append-only `done/total  status  locator  elapsed` line per finished job. No cursor control. |
| Silent | `--quiet` | No live output; only end-of-run panels. |

End-of-run panels (campaign header, plan, ok/failed results, per-failure
lines) print regardless of mode. With `--timing`, an extra per-phase
breakdown table is appended.

### The `events.jsonl` event stream

By default each campaign writes `<campaign_dir>/events.jsonl` — an
append-only JSON Lines stream unifying three sources through one structlog
pipeline:

1. **Lifecycle hooks** — `batch_start`, `pair_load_start`, `job_start`,
   `job_end`, `batch_end`.
2. **Phase timing records** — per-phase wall times from the
   `mamba_dvc.timing` logger, including worker-subprocess `ncc.*` records
   when the dispatcher's `emit_phase_records` flag is on.
3. **Captured warnings** — `warnings.warn` calls (unpinned
   `flow_convention`, unopenable stores, unmatched `--store`, …) emitted
   as `kind:"warning"` rows.

Disable with `--no-events`. Every row carries:

- An ISO-8601 UTC `ts`
- A `kind` discriminator (`batch_start`, `job_end`, `phase`, `warning`, …)
- A `session_id` (uuid4) bound for the process lifetime
- The `campaign` name
- Per-load `store` / `deformation` and per-job `variant_id` /
  `kind_of_job` fields, inherited automatically via contextvars

Slice it with `jq`:

```bash
# Per-job runtime distribution
jq -r 'select(.kind == "job_end") | [.store, .deformation, .variant_id, .wall_correlate_s] | @tsv' events.jsonl

# All warnings raised during the run
jq -c 'select(.kind == "warning")' events.jsonl
```

### Resume semantics

`events.jsonl` is opened in append mode. Strictly-serial re-invocations
against the same campaign directory are safe — each run gets a fresh
`session_id`, so downstream consumers separate resumed sessions with
`df.groupby("session_id")` (or by `batch_start` boundaries).
**Concurrent invocations against the same campaign dir are undefined
behaviour**: Windows offers no `O_APPEND` atomicity guarantee for line
writes exceeding `PIPE_BUF`.

### Programmatic use

The CLI plumbing is a thin wrapper around two reusable primitives in
`mamba_dvc.run.eventlog`:

- `EventSink(campaign_dir, campaign=...)` — batch-shaped wrapper used by
  `mamba-dvc run`. Yields a `StructlogObserver` to plug into `run_batch`.
- `SessionScope(out_dir, **fields)` — the campaign-agnostic primitive.
  Use directly from non-batch harnesses (e.g. the temporal series sweep,
  which binds `series=` / `strategy=` instead of `campaign=`). Pair with
  `SeriesPairLogger` / `DispatchLogger` to bracket each pair and each
  dispatcher invocation with structured boundary rows.

```python
from mamba_dvc.run.eventlog import SessionScope, SeriesPairLogger, DispatchLogger

with SessionScope(out_dir, series="rat-103L", strategy="incremental"):
    with MultiGPUDispatcher(..., dispatch_observer=DispatchLogger()) as disp:
        correlate_series(frames, dispatcher=disp, pair_observer=SeriesPairLogger())
```

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest -m "not gpu"   # skip GPU-only tests on a CPU-only machine
uv run pre-commit run --all-files
```

See [`CLAUDE.md`](CLAUDE.md) for the full contributor guide.
