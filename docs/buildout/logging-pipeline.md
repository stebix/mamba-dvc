# Logging pipeline buildout

This document is the buildout plan for unifying the run-campaign model's
observability onto a single on-disk event stream and adding lightweight
memory-info logging on top of it.

The plan is split into two sequential phases:

- **Phase 1** — the `events.jsonl` substrate (lifecycle + phases +
  warnings) under a shared `structlog` pipeline. Lands first and must be
  green on `ruff` / `pyright` / `pytest` before Phase 2 starts.
- **Phase 2** — lightweight memory-info logging (system memory, process
  RSS, optional CuPy pool stats, on-error allocation trace) reusing the
  Phase 1 sink and schema. Sequenced so the Phase 1 churn lands clean
  before Phase 2 adds more emit-sites in `batch.py`.

The **Critical files** and **Verification** sections list both phases
distinctly.

## Context (Phase 1)

The run-campaign model today exposes three independent observability
streams with **no shared persistence layer**:

1. `BatchObserver` (`mamba_dvc/run/progress.py`) — in-process Protocol,
   drives the live Rich UI.
2. `mamba_dvc.timing` logger (`mamba_dvc/instrument.py`) — stdlib
   logger, parent-side phase records.
3. `warnings.warn` — footgun signals (unpinned flow convention at
   `batch.py:861`, unopenable stores at `batch.py:202`, unmatched
   `--store` at `batch.py:795`).

After a campaign ends the only durable evidence on disk is
`manifest.jsonl` (one row per job) and the per-job `<variant_id>.json`
sidecar. There is no time-ordered event log, no warnings transcript, no
phase-record archive. Headless monitoring (tail the log from another
shell) and post-hoc investigations (what happened in what order) both
have to be reconstructed from disk snapshots instead of replayed from a
single event source.

This plan adds a single append-only `campaign_dir/events.jsonl`
capturing **all three streams** through a shared `structlog` pipeline.
`structlog` earns its place as the **output substrate** (timestamping,
JSON rendering, contextvars-based field binding); the three
emit-channels keep their current call-site shapes because each is
correct for its concern. Worker-subprocess timing capture is
**deferred** — the file format won't have to change when that lands.

### Why structlog (and why partial unification)

The three channels are not redundant — they serve different *emit-time*
concerns:

- `BatchObserver` is a UI contract (pull-based, O(1) hooks, must not raise).
- `mamba_dvc.timing` is a high-frequency push channel cheaply gated by
  `isEnabledFor(DEBUG)`.
- `warnings.warn` is treated specially by the stdlib (`filterwarnings`,
  `-W`, pytest capture).

What's *missing* is a shared persistence path. `structlog` supplies
exactly that: one processor chain ending in `JSONRenderer` + file
handler, a stdlib bridge via `structlog.stdlib.ProcessorFormatter`, and
contextvars that auto-attach `(campaign, store, deformation,
variant_id)` to every event in scope. No call-site rewrites of
`timed()` / `log_phase()` / `warnings.warn`.

What is *not* unified:

- The Rich renderer stays observer-driven (logging-handler-as-UI would
  couple schema to UI responsiveness).
- `timed()` / `log_phase()` keep their stdlib-logging emit sites.
  `structlog` enters only at the configuration boundary in the CLI.

## Approach (Phase 1)

### 1. Dependency

Add `structlog>=24.0` to `pyproject.toml:7-21` `[project] dependencies`
(next to `rich`). Pure Python, ~120 KB, no native bits, ships type
stubs.

### 2. New module: `mamba_dvc/run/eventlog.py`

Public surface:

- **`EventSink`** — context manager owning the file handle, structlog
  config, and stdlib bridge.
  - `__enter__`:
    - Open `events.jsonl` in append mode. Append is safe within a single
      process and across strictly-serial re-invocations of `mamba-dvc run`
      against the same campaign dir; **concurrent invocations against
      the same campaign dir are undefined behaviour** (no O_APPEND
      atomicity guarantee on Windows, event lines exceed PIPE_BUF). The
      module docstring states this loudly.
    - Configure structlog with processor chain: `merge_contextvars`,
      `TimeStamper(utc=True, fmt="iso")`, `EventRenamer(to="kind")`,
      `JSONRenderer(serializer=json.dumps, default=_jsonable)`.
    - Build a stdlib `logging.FileHandler(file)` with
      `structlog.stdlib.ProcessorFormatter` using the same processors
      plus a `foreign_pre_chain` for non-structlog records (so
      `mamba_dvc.timing` records and `py.warnings` records are formatted
      identically to structlog-native events).
    - Attach handler to `mamba_dvc.timing` and `py.warnings` loggers,
      **and raise `mamba_dvc.timing` to DEBUG for the sink's lifetime**
      (remembering prior level for restore on `__exit__`). Phase records
      then always land in `events.jsonl` while events are on; the
      `--timing` CLI flag remains the gate for the end-of-run
      `accumulating()` breakdown panel only.
    - `logging.captureWarnings(True)`; remember prior state.
    - `structlog.contextvars.bind_contextvars(campaign=<campaign>,
      session_id=<uuid4>)`. The `session_id` makes resumed files
      groupable: each EventSink lifetime gets a fresh uuid, so
      `df.groupby('session_id')` cleanly separates appended sessions.
  - `__exit__`: detach handler, restore `mamba_dvc.timing` level,
    restore `captureWarnings`, `clear_contextvars`, close file.
  - Yields `(StructlogObserver, file_handler)` so the CLI can stack
    `accumulating()` alongside without re-opening anything.

- **`StructlogObserver(BatchObserver)`** — translates each hook into a
  structlog event:
  - `on_batch_start`: `log.info("batch_start", n_jobs=..., n_variants=...)`.
  - `on_pair_load_start`: `bind_contextvars(store=..., deformation=...)`;
    emit `pair_load_start` with `n_variants`.
  - `on_job_start`: `bind_contextvars(variant_id=...,
    kind_of_job=job.kind)` (renamed from `kind` to avoid clashing with
    the event-discriminator `kind` field); emit `job_start`.
  - `on_job_end`: emit `job_end` with `status`, `phase` (from
    `summary["phase"]` on failure), `wall_correlate_s`,
    `wall_evaluate_s`, `error_tail` (last 2 KB, same convention as
    `manifest.jsonl`); `unbind_contextvars("variant_id", "kind_of_job")`.
  - `on_batch_end`: emit `batch_end` with `n_ok`, `n_failed`,
    `duration_s`; `clear_contextvars()`.

- **`Tee(*observers: BatchObserver) -> BatchObserver`** — fan-out
  composite. Each of the five hooks dispatches to every member in order.
  Lives in `eventlog.py` because its sole reason to exist is composing
  the renderer with the sink observer.

  **Exception policy**: per-member calls are wrapped in a
  `try/except Exception` block. Failures are logged at WARNING on a
  dedicated `mamba_dvc.eventlog` logger (one record per failure,
  carrying the failing observer class name and `traceback.format_exc()`)
  and execution proceeds to the next member. This is load-bearing: the
  Rich renderer can throw if the terminal disconnects mid-run, and the
  file sink can throw on a full disk — neither failure mode may break
  the other consumer, and neither may abort `run_batch`. The existing
  `BatchObserver` contract ("hooks must not raise") still applies to
  *implementors*; `Tee` enforces it defensively at the composition
  boundary.

### 3. CLI wiring: `mamba_dvc/cli/_run.py`

- Add `--no-events` flag (default `False`). `--timing` semantics are
  **narrowed**: it now only controls whether the end-of-run
  `accumulating()` breakdown panel prints. Phase records always reach
  `events.jsonl` while events are on, because `EventSink` raises
  `mamba_dvc.timing` to DEBUG for its lifetime. The narrowing is the
  intuitive behaviour ("one unified log") and removes a footgun where
  users would enable events expecting all timing info to land.
- In `_execute_campaign` (`_run.py:229-259`): wrap the existing observer
  selection in an outer `EventSink` context (unless `--no-events`). The
  yielded `StructlogObserver` is teed with the chosen rendering observer
  (`_RichProgressObserver` / `_PlainLogObserver` / `NullObserver`).
  Library callers that invoke `run_batch` directly stay silent — no
  implicit file writes.
- `--dry-run` path (`_run.py:183-190`) does **not** create the events
  file; that path emits no events.

### 4. Library export: `mamba_dvc/run/__init__.py`

Add `EventSink`, `StructlogObserver`, `Tee` to `__all__` and the import
block (`__init__.py:26-39`) so programmatic callers can opt in.

### 5. Docs

- `docs/plans/run-interface.md` §7 ("Output layout",
  `run-interface.md:346-388`): add `events.jsonl` to the directory
  layout and an example `pd.read_json("events.jsonl",
  lines=True).query("kind == 'phase'")` aggregation.
- `mamba_dvc/run/eventlog.py` module docstring: state the unification
  stance (output substrate, not new emit channel), document the
  deferred worker-subprocess gap with a link back to
  `instrument.py:42-52`, and document the `--timing` interaction.

### Event schema (kind-discriminated JSONL)

```
{"ts":"2026-05-13T15:42:01.123456+00:00","kind":"batch_start","campaign":"results_v2","n_jobs":42,"n_variants":7}
{"ts":"...","kind":"pair_load_start","campaign":"...","store":"103L_Mg5Gd_4w_000","deformation":"fs104","n_variants":3}
{"ts":"...","kind":"job_start","campaign":"...","store":"...","deformation":"fs104","variant_id":"mt0.5-a1b2c3d4","kind_of_job":"synthetic"}
{"ts":"...","kind":"phase","campaign":"...","store":"...","deformation":"fs104","variant_id":"...","phase":"ncc.batch","seconds":0.041,"n_points":1234}
{"ts":"...","kind":"warning","campaign":"...","category":"UserWarning","message":"flow_convention is unset ...","filename":"...","lineno":...}
{"ts":"...","kind":"job_end","campaign":"...","store":"...","deformation":"fs104","variant_id":"...","status":"ok","wall_correlate_s":73.6,"wall_evaluate_s":0.01}
{"ts":"...","kind":"batch_end","campaign":"results_v2","n_ok":40,"n_failed":2,"duration_s":1843.2}
```

Context binding auto-attaches `campaign` / `session_id` / `store` /
`deformation` / `variant_id` in their respective scopes; no per-call
boilerplate. `kind_of_job` carries the `Job.kind` (`"real"` /
`"synthetic"` / `"?"`) without colliding with the event discriminator —
this rename is forced by `EventRenamer(to="kind")`. If `kind` collisions
recur for other fields during implementation, switch the discriminator
to the structlog-default `event` and drop the renamer; the consumer-side
cost is trivial.

`session_id` is a uuid4 bound for the EventSink's lifetime. Resumed
files contain multiple sessions appended in order; downstream
consumers use `df.groupby("session_id")` (or `kind == "batch_start"` as
the cheaper start-of-session marker) to separate them.

### Threading & concurrency

- Observer hooks fire on the consumer thread (`batch.py:392-432`
  existing contract). Contextvars bound by `StructlogObserver` (e.g.
  `store`, `deformation`, `variant_id`) live in the consumer thread's
  context only.
- Phase records from the prefetch loader thread (`batch.load_pair`,
  `batch.py:607`) flow through stdlib `logging` — already thread-safe
  via per-handler lock — but **`contextvars` are not propagated to
  worker threads automatically by `ThreadPoolExecutor`** (bpo-34014 was
  closed as out-of-date; the feature does not exist in any CPython
  release). Naïvely emitting from the loader thread would produce
  records with no `store`/`deformation` fields, since
  `merge_contextvars` reads the *emitting* thread's bindings at format
  time.

  **Fix:** in `_prefetched_load_groups` (`batch.py:616-694`), wrap the
  loader callable at submit time with `contextvars.copy_context().run`
  so the consumer's bindings are inherited as a true scope:

  ```python
  ctx = contextvars.copy_context()
  fut = executor.submit(ctx.run, _load_one, load_key, jobs)
  ```

  This makes the loader thread see exactly the contextvars the consumer
  had bound at submit time. The deformation is still passed as an
  explicit argument to emit sites that need a deterministic value
  regardless of scope (e.g. `pair_bytes` computation in Phase 2 that may
  run before any inner bind), but the *primary* context-attachment path
  is `copy_context`.
- One shared file handle is safe because `logging.FileHandler.emit()`
  holds the handler lock around the whole `format` + `write` + `flush`
  sequence. structlog's `ProcessorFormatter` adds no cross-thread
  requirements.

### Multi-process worker gap (deferred)

`gpu.dispatch` workers still don't propagate their `ncc.*` records. The
file format is forward-compatible: a future `multiprocessing.Queue` +
`QueueListener` bridge would land those records as more `kind:"phase"`
lines without schema change. Documented in `eventlog.py` module
docstring and cross-referenced from `instrument.py:42-52`.

## Critical files

**Phase 1:**

- `pyproject.toml:7-21` — add `structlog>=24.0` to `[project]
  dependencies`.
- **New** `mamba_dvc/run/eventlog.py` — `EventSink`, `StructlogObserver`,
  `Tee`, processor chain, stdlib bridge.
- `mamba_dvc/run/__init__.py:26-39` — export the new symbols.
- `mamba_dvc/cli/_run.py:137-160` — add `--no-events` flag.
- `mamba_dvc/cli/_run.py:229-259` (`_execute_campaign`) — wrap observer
  selection in `EventSink`.
- `docs/plans/run-interface.md:346-388` — document `events.jsonl`.
- **New** `tests/run/test_eventlog.py`:
  - `test_lifecycle_events_round_trip` — tiny campaign through
    `run_batch` (reusing `_FakeDataset` / `_fake_correlate` from
    `tests/run/test_batch.py:40-77`), assert every event kind appears
    with expected fields and contextvars-inherited keys.
  - `test_phase_records_routed` — stack `accumulating()` with
    `EventSink`, emit a `timed()` block, assert `kind:"phase"` line
    lands with all `mdvc_*` fields surfaced.
  - `test_warnings_captured` — `warnings.warn` inside the sink lands as
    `kind:"warning"`. **Scope this test (and every other EventSink
    test) to its narrowest possible block** — no autouse fixture, no
    session-scoped sink. `logging.captureWarnings(True)` is
    process-global mutable state and pytest's warning capture
    (`pytest.warns`, `recwarn`, `filterwarnings` marks) does not
    interoperate cleanly with records redirected through it
    (pytest #2449, still open). A bleeding sink will silently
    suppress unrelated tests' `pytest.warns` matches.
  - `test_context_inheritance` — bind store/deformation, emit, assert
    fields present; unbind, emit, assert absent.
  - `test_thread_safe_phase_emit` — bind `store`/`deformation` in the
    *parent* thread, submit a `timed()`-emitting callable to a
    `ThreadPoolExecutor` wrapped in `copy_context().run`, assert (a) all
    lines well-formed and (b) the loader-thread lines carry the parent's
    `store`/`deformation` contextvars (mirrors the production
    `_prefetched_load_groups` path; uses `threading` already imported at
    `tests/run/test_batch.py:6`).
  - `test_tee_fans_out` — two recording observers under one `Tee`,
    assert identical hook sequences.
- `tests/run/test_batch.py` — add `test_run_batch_with_structlog_observer`
  confirming `run_batch(observer=StructlogObserver(...))` produces a
  complete event sequence end-to-end (no behaviour change to
  `run_batch`).

**Phase 2** (only after Phase 1 lands green):

- `pyproject.toml:39-46` — promote `psutil` from `[dependency-groups]
  dev` to `[project] dependencies`, or leave dev-only and rely on the
  existing graceful-degradation pattern from `batch.py:566-572`.
  Recommendation: promote — Phase 2 makes memory reporting a
  first-class user feature, not a dev-only nicety.
- `mamba_dvc/instrument.py:75` — add `log_memory()` emitter alongside
  the existing `log_phase()` (analogous shape: structured `extra`
  fields on a new `mamba_dvc.memory` logger, gated by
  `isEnabledFor(DEBUG)`).
- `mamba_dvc/run/eventlog.py` — `EventSink.__enter__` additionally
  attaches the file handler to `mamba_dvc.memory`. Add an
  `_allocation_snapshot()` helper gathering psutil + CuPy pool stats;
  called by `log_memory()`.
- `mamba_dvc/run/batch.py:593-613` (`_load_or_capture`) — emit
  `log_memory("pre_load", deformation=...)` immediately before
  `ds.load_pair(...)` and `log_memory("post_load", deformation=...,
  pair_bytes=actual_pair_bytes)` immediately after a successful load.
  On load failure: `log_memory("error", trigger_phase="load_pair")` is
  folded into the existing `except` block.
- `mamba_dvc/run/batch.py:446-507` (`_run_variant`) — both `except
  Exception` blocks (`batch.py:462`, `batch.py:498`) call
  `log_memory("error", trigger_phase=...)` before returning the
  `_failed(...)` JobResult.
- `mamba_dvc/run/batch.py:265-443` (`run_batch`) — the store-open
  failure path (`batch.py:346-364`) calls `log_memory("error",
  trigger_phase="open_store")` before recording the placeholder.
- `tests/test_instrument.py` — extend with
  `test_log_memory_emits_structured_record`,
  `test_log_memory_silent_when_disabled`,
  `test_log_memory_graceful_without_psutil` (monkeypatch psutil to
  `ImportError`).
- `tests/run/test_eventlog.py` — add `test_memory_snapshot_pre_post_load`
  and `test_memory_on_error` (force a fake correlate raise, assert the
  trailing `kind:"memory", trigger:"error"` line carries the expected
  `trigger_phase`).
- `docs/plans/run-interface.md` — document the `kind:"memory"` schema
  and trigger taxonomy alongside Phase 1's event types.

## Reuse / existing utilities

- `_to_jsonable` (`mamba_dvc/run/batch.py:1046-1057`) — same numpy/NaN
  coercion logic needed by the JSONRenderer's `default` callback. Lift
  into a small `mamba_dvc/run/_jsonable.py` helper. There is no
  circular-import risk that forces this; the rationale is **cohesion**
  — one canonical place for JSONable coercion, used by both the
  manifest writer in `batch.py` and the JSONRenderer in `eventlog.py`,
  so future numpy/NaN edge cases get fixed in one spot.
- `PhaseAccumulator` (`mamba_dvc/instrument.py:243-294`) — unchanged;
  continues to consume `mamba_dvc.timing` records in parallel with the
  new file handler. Both attach to the same logger; both are
  independent consumers.
- `accumulating()` context manager (`instrument.py:297-322`) —
  unchanged; remains the `--timing` mechanism. The new file handler is
  attached independently (handler-level addition, no logger-level
  conflict).
- `BatchObserver` Protocol (`mamba_dvc/run/progress.py:52-73`) —
  unchanged. `StructlogObserver` and `Tee` both satisfy it.

## Verification

### Phase 1 gates (must be green before Phase 2 starts)

1. **Format & lint**: `uv run ruff format .` and `uv run ruff check
   --fix .` clean.
2. **Type check**: `uv run pyright` clean (structlog ships type stubs;
   pyright strict per `pyproject.toml:106` should be satisfied).
3. **Unit tests**: `uv run pytest tests/run/test_eventlog.py
   tests/run/test_batch.py tests/test_instrument.py` green. The
   instrument tests must remain green — proof that the file handler
   does not interfere with `PhaseAccumulator`.
4. **CLI dry-run smoke**: `uv run mamba-dvc run configs/results_v2.yaml
   --dry-run` — no `events.jsonl` written.
5. **CLI live, smallest possible**: `uv run mamba-dvc run
   configs/results_v2.yaml --only mask_threshold=0.9 --store
   103L_Mg5Gd_4w_000 --devices 0` → `<campaign_dir>/events.jsonl`
   exists; `python -c "import pandas as pd; df =
   pd.read_json('<campaign_dir>/events.jsonl', lines=True);
   print(df.groupby('kind').size())"` shows expected counts (1
   batch_start, ≥1 pair_load_start, N job_start/job_end pairs, 1
   batch_end, ≥0 warnings). Phase records appear in this run **without**
   `--timing` (EventSink raises `mamba_dvc.timing` to DEBUG for its
   lifetime); `df.query("kind == 'phase'")` is non-empty.
5b. **Loader-thread context attachment**: from the same file,
   `df.query("kind == 'phase' and phase == 'batch.load_pair'")` rows
   carry non-null `store` and `deformation` fields. This proves the
   `copy_context().run` wrap in `_prefetched_load_groups` propagates
   the consumer-thread bindings into the loader thread; a regression
   here would silently strip the fields.
6. **`--timing` panel only**: rerun with `--timing` added; the existing
   end-of-run breakdown panel prints in addition to the events file.
   `df.query("kind == 'phase'")["phase"].value_counts()` matches the
   step-5 file (the `--timing` flag no longer changes what reaches the
   events file — only the panel behaviour). The `ncc.*` records remain
   absent on the multi-GPU path; the worker-capture gap is deferred.
7. **`--no-events`**: rerun with `--no-events`; no `events.jsonl`
   written; behaviour and output otherwise identical.
8. **Resume**: rerun the same command after step 5; assert
   `events.jsonl` grew (new `batch_start` … `batch_end` block appended,
   prior block intact). This proves resume across processes appends
   rather than truncates.

### Phase 2 gates

All Phase 1 gates re-run green, plus:

9. **Memory unit tests**: `uv run pytest tests/test_instrument.py
   tests/run/test_eventlog.py` green, including the new memory tests.
10. **Memory snapshot smoke**: rerun the smallest-possible CLI
    invocation; `df.query("kind == 'memory'").groupby("trigger").size()`
    shows ≥1 `pre_load`, ≥1 `post_load`, ≥0 `error`.
11. **Memory fields populated**: `df.query("kind ==
    'memory'").iloc[0]` shows non-null `vm_total_gb`,
    `vm_available_gb`, `process_rss_gb`. When CuPy is importable, `gpu`
    is a non-empty list of per-device dicts.
12. **Forced-failure trace**: intentionally trigger one failure (e.g.
    by passing an invalid `--devices` or a missing flow file) and
    confirm the manifest row's `phase` field has a matching
    `kind:"memory", trigger:"error", trigger_phase:<same>` event
    immediately preceding the `job_end` failure record. This is the
    "trace back current allocations on error" requirement.
13. **Graceful degradation**: `pip uninstall psutil` in a throwaway env
    (or monkeypatch in a test); `events.jsonl` still gets
    `kind:"memory"` lines, only with the available fields populated and
    missing ones absent. No crash, no traceback in `events.jsonl`.

## Explicitly out of scope (separate follow-ups)

- **`run_e2e_zarr.py` instrumentation.** Phase 1 wires
  `mamba_dvc/cli/_run.py` only. `run_e2e_zarr` keeps its existing
  stdlib-logging story; if it acquires structured events later, it can
  reuse the Phase 1 `EventSink` substrate without schema change.
- **Worker-subprocess timing capture** (`gpu.dispatch` spawned workers
  → `multiprocessing.Queue` → parent `QueueListener` → same structlog
  pipeline). Closes the `ncc.*` blind spot; file format unchanged.
- **Periodic memory sampling** (e.g. every 30 s during long load_pair
  calls). Out of scope for Phase 2 — lifecycle triggers should suffice;
  add a `--memory-interval N` flag later if a real-world campaign
  reveals a blind spot.
- **Aggregation utilities** — a `mamba-dvc events summarize`
  subcommand or a notebook for the `events.jsonl`. JSONL is already
  pandas-friendly; defer until there's a real consumer.
- **Real-time tailing helpers** — JSONL already supports `tail -f` and
  `jq` natively.

---

## Phase 2 details: lightweight memory-info logging

### Goal

Capture, on the **same `events.jsonl` substrate**, three classes of
memory information:

1. **System memory status** of the host (total / available / used /
   percent) at lifecycle transitions.
2. **Resident memory of loaded data** — process RSS, plus the
   byte-size of the currently materialized `EvaluationPair` after each
   `load_pair`.
3. **Allocation trace on error** — a snapshot emitted alongside every
   failure path so a `job_end` failure can be correlated with the
   memory state at the moment things went wrong.

### Approach

Mirror the existing `mamba_dvc.timing` pattern from
`mamba_dvc/instrument.py` exactly:

- New logger `mamba_dvc.memory`. New emitter `log_memory(trigger: str,
  **fields) -> None` analogous to `log_phase` (`instrument.py:109-136`).
  Zero-cost when the logger is not enabled for DEBUG — production
  callers pay exactly one `isEnabledFor` check per call.
- `log_memory` internally gathers a snapshot via
  `_allocation_snapshot()`:
  - **System**: `psutil.virtual_memory()` → `vm_total_gb`,
    `vm_available_gb`, `vm_used_gb`, `vm_percent` (rounded to 2 dp;
    GB-rounded fields stay human-scannable, raw bytes available in
    `vm_total_bytes` etc. for the curious).
  - **Process**: `psutil.Process(os.getpid()).memory_info()` →
    `process_rss_gb`, `process_vms_gb`.
  - **GPU (optional)**: when `cupy` is importable, iterate visible
    devices with explicit context selection — `with cupy.cuda.Device(i):`
    is required because `cupy.cuda.runtime.memGetInfo()` reads the
    *current* device only. Per device, read
    `cupy.cuda.MemoryPool.used_bytes()` + `total_bytes()` +
    `cupy.cuda.runtime.memGetInfo()` → `gpu: [{"id":i,
    "pool_used_gb":..., "pool_total_gb":..., "free_gb":...,
    "total_gb":...}, ...]`. Skipped silently when CuPy is missing or no
    device context is active (mirrors the existing `_sync()` defensive
    pattern at `instrument.py:97-106`). **This sees the parent
    process's pool only**; `gpu.dispatch` worker subprocess pools are
    invisible, the same blind spot as the worker-timing gap — call out
    in the module docstring alongside the timing gap.
  - **Caller fields**: any `**fields` (e.g. `deformation`,
    `pair_bytes`, `trigger_phase`) attached as structured record
    attributes.
- Every snapshot piece is independently guarded — if psutil is absent
  the system/process fields are omitted but the record still emits with
  the caller fields and GPU section. If CuPy is absent, GPU section is
  omitted. Never raises.
- `EventSink.__enter__` additionally attaches the file handler to
  `mamba_dvc.memory` and sets it to DEBUG for the sink's lifetime. The
  `mamba_dvc.timing` raise-to-DEBUG continues to be gated by `--timing`
  (its records are diagnostic and add GPU-sync overhead);
  `mamba_dvc.memory` records carry no GPU-sync cost, so always-on
  inside the sink is fine.

### Emission sites (in `mamba_dvc/run/batch.py`)

All sites are inside the existing structlog contextvars scope, so
`(campaign, store, deformation, variant_id)` flow automatically.

| Trigger      | Call site                                                                                                                           | Emitted from                              | Fields beyond defaults                                                                                                                |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `"start"`    | `EventSink.__enter__`                                                                                                               | consumer thread                           | —                                                                                                                                     |
| `"pre_load"` | `_load_or_capture` immediately before `ds.load_pair(...)` (`batch.py:606`)                                                          | loader thread (when prefetch on) or consumer thread | `deformation`                                                                                                                         |
| `"post_load"`| same, immediately after successful `ds.load_pair(...)`                                                                              | same                                      | `deformation`, `pair_bytes` (sum of `nbytes` for the host-resident reference + deformed arrays after `load_pair` returns; device-side allocations are answered by the `gpu` block instead) |
| `"error"`    | each `except Exception` in `_run_variant` (`batch.py:462`, `batch.py:498`) and in `_load_or_capture` (`batch.py:611`) and in the store-open failure path (`batch.py:353-363`) | consumer or loader thread                 | `trigger_phase` (one of `"open_store"`, `"load_pair"`, `"correlate"`, `"evaluate_or_write"`), `exc_type`                              |
| `"end"`      | `EventSink.__exit__`                                                                                                                | consumer thread                           | —                                                                                                                                     |

The error trigger is the load-bearing addition for the "trace back
current allocations" requirement: every failure path emits the memory
snapshot **before** returning the `_failed(...)` JobResult, so the
JSONL has `kind:"memory", trigger:"error"` immediately followed by
`kind:"job_end", status:"failed"` with the same `variant_id`
contextvars. Downstream: `df.query("kind == 'memory' and trigger ==
'error'")` is the allocation-at-failure table.

### Schema example

```json
{"ts":"2026-05-13T15:42:01.123456+00:00","kind":"memory","trigger":"pre_load",
 "campaign":"results_v2","store":"103L_Mg5Gd_4w_000","deformation":"fs104",
 "vm_total_gb":255.0,"vm_available_gb":210.5,"vm_used_gb":44.5,"vm_percent":17.4,
 "vm_total_bytes":273804165120,"vm_available_bytes":226075180544,
 "process_rss_gb":12.3,"process_vms_gb":28.9,
 "gpu":[{"id":0,"pool_used_gb":4.2,"pool_total_gb":12.0,"free_gb":43.8,"total_gb":48.0},
        {"id":1,"pool_used_gb":4.2,"pool_total_gb":12.0,"free_gb":43.8,"total_gb":48.0}]}

{"ts":"...","kind":"memory","trigger":"error","trigger_phase":"correlate",
 "campaign":"results_v2","store":"...","deformation":"fs104","variant_id":"mt0.5-a1b2c3d4",
 "exc_type":"CUDARuntimeError",
 "vm_total_gb":255.0,"vm_available_gb":3.1,"vm_used_gb":251.9,"vm_percent":98.8,
 "process_rss_gb":189.4, ...}
```

### Threading note

Snapshots emitted from the loader thread (`_load_or_capture`'s
`pre_load` / `post_load`) inherit the consumer's `contextvars` via the
`copy_context().run` submit-time wrap installed for Phase 1 (see Phase 1
"Threading & concurrency"); `ThreadPoolExecutor` does **not** propagate
contextvars on its own. The deformation is also passed explicitly to
`log_memory(...)` so the field is set even if the snapshot fires before
any inner contextvar bind. The stdlib `logging` lock guarantees the
file lines do not interleave.

### What we deliberately do *not* capture in Phase 2

- The prefetch queue's `in_flight` state from `_prefetched_load_groups`
  (`batch.py:616-694`). Plumbing the queue handle out to
  `_run_variant`'s error path costs more than the diagnostic value —
  the process RSS at the moment of failure plus the
  immediately-preceding `pre_load` / `post_load` events together already
  tell you what's resident. Revisit only if a real incident proves this
  insufficient.
- Per-allocation Python-object accounting (`tracemalloc`, `pympler`).
  Heavyweight, intrusive; out of scope. The CuPy pool stats are the
  only fine-grained allocation view that costs essentially nothing.
