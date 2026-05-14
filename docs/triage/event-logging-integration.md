# Event logging integration: multi-GPU + temporal pipelines

Follow-up to the post-merge integration of `temporal-buildout` with the
trunk-side `events.jsonl` substrate (commits `0c66195 feat(run):
events.jsonl substrate via structlog` and `1186794 feat(cli): wire
events.jsonl into 'mamba-dvc run'`). The substrate today is shaped
around `run_batch` and the `BatchObserver` protocol; the temporal driver
(`mamba_dvc/pipeline/series.py`) and the persistent `MultiGPUDispatcher`
(`mamba_dvc/gpu/dispatch.py`) do not yet sit inside it. This document is
the plan for closing those gaps without breaking the pure-core rule or
re-coupling `correlate_series` to a logger.

Companion doc: `docs/triage/multi-gpu-aliasing-bug.md`. The
buffer-aliasing bug it describes was discovered *because* the temporal
harness produced uniformly bad numbers across every pair; observability
into the dispatcher's per-pair return path is what made that bisection
possible. The work below is in the same spirit — make the buildout
debuggable by default.

## Where the streams already converge

`EventSink` (from `mamba_dvc/run/eventlog.py`) unifies three streams:

1. **`BatchObserver` hooks** via `StructlogObserver` —
   `batch_start`/`pair_load_start`/`job_start`/`job_end`/`batch_end`
   lines.
2. **Phase records** from `mamba_dvc.instrument.timed(...)` /
   `log_phase(...)` — anything the `mamba_dvc.timing` stdlib logger
   emits, promoted via `_promote_mdvc_fields` to `kind:"phase"` JSON
   with the `mdvc_*` prefix stripped.
3. **`warnings.warn(...)`** — `warnings.showwarning` is swapped for
   `EventSink._show_warning` so each warning lands as `kind:"warning"`
   with `category`/`message`/`filename`/`lineno`.

Contextvars in scope: `campaign` + `session_id` (sink lifetime),
`store` + `deformation` (per load group), `variant_id` + `kind_of_job`
(per job). The `mamba_dvc.eventlog` module docstring explicitly defers
one piece: **`gpu.dispatch` worker subprocesses do not propagate their
`ncc.*` records.** That deferral is one of the items below.

## What the multi-GPU pipeline emits today

Parent-process phases (`mamba_dvc/gpu/dispatch.py`):

| Phase                              | Site                            |
|------------------------------------|---------------------------------|
| `dispatch.build_grid`              | `correlate_multi_gpu:660`       |
| `dispatch.resolve_masks`           | `correlate_multi_gpu:664`       |
| `dispatch.filter_by_mask`          | `correlate_multi_gpu:668`       |
| `dispatch.materialize_contiguous`  | `correlate_multi_gpu:678`       |
| `dispatch.<join>` (workers)        | `correlate_multi_gpu:715`       |
| `dispatch.outlier`                 | `correlate_multi_gpu:748`       |
| `dispatch.h2d` / `helper` / `d2h`  | in-process path, lines 329/340/353 |

Worker phases (`_serve_worker:793` and below): emit `ncc.*` records to
the worker's local logging tree, which is **not** wired to the parent
file handler. They are silently dropped on multi-device runs.

`MultiGPUDispatcher.correlate(...)` itself does not emit a per-pair
event. The only signal that a pair completed is a returned
`DisplacementField`; the only signal that a pair failed is a raised
`RuntimeError`. For a 30-pair temporal sweep against a 4-GPU
dispatcher, that means `events.jsonl` would contain zero
dispatcher-level rows between the outermost campaign markers.

## What the temporal pipeline emits today

`correlate_series` (`mamba_dvc/pipeline/series.py:55`) is pure by
design. It emits:

- One `RuntimeWarning` per failed pair (line 227, intercepted by
  `EventSink._show_warning` if a sink is active — works today).
- Nothing on the success path. The `on_pair` callback exists for the UI
  but is not used by any current caller.

The phase records inside `correlate(...)` and the dispatcher fire as
normal. Strategy/lag/`t_ref`/`t_def`/series identity are **not** bound
as contextvars, so a phase record from inside a temporal sweep is
indistinguishable from a phase record from a one-shot
`correlate(...)`. Post-hoc grouping of `events.jsonl` by pair or
strategy is therefore impossible.

The strategy-sweep harness `evaluate_synthetic(...)` and
`scripts/eval_temporal_strategies.py` do not open an `EventSink`.

## Goal

After this work, a run like

```python
with SessionScope(out_dir, series="rat-103L-fs104"):
    series = correlate_series(frames, mask=mask, dispatcher=disp, ...)
```

produces an `events.jsonl` whose rows are:

```
{kind:"series_start", session_id, series, strategy, n_frames}
{kind:"pair_start",   t_ref, t_def, strategy}
{kind:"phase",        phase:"dispatch.build_grid", seconds, t_ref, t_def, ...}
{kind:"phase",        phase:"ncc.fft_ref", seconds, device_id, t_ref, t_def, ...}   ← new
{kind:"pair_end",     t_ref, t_def, status:"OK", n_valid, mae?}
...
{kind:"warning",      category:"RuntimeWarning", message:"correlate_series: pair (...) failed ..."}
{kind:"series_end",   n_ok, n_failed, duration_s}
```

— same JSON shape as the batch sink, just with the temporal axis bound
into context. Phase records correlate to the pair that triggered them.
Worker-subprocess phases land alongside parent ones with the originating
`device_id` attached.

## Phase 1 — extract a campaign-agnostic file-handle scope

`EventSink` is wedded to the `(campaign_dir, campaign)` shape (and to
returning a `StructlogObserver` on `__enter__`). The temporal sweep does
not have a campaign; it has a *series*, optionally lifted to a strategy
× lag grid.

Refactor: pull the reusable substrate out of `EventSink` into a
`SessionScope` context manager. Concrete diff shape:

```python
# mamba_dvc/run/eventlog.py
class SessionScope:
    """Owns the events.jsonl handle, structlog config, warning swap.

    Bindings supplied via ``**fields`` are bound as contextvars for the
    scope's lifetime (plus a fresh ``session_id`` uuid4). No observer is
    returned -- callers that want one build it themselves.
    """
    def __init__(self, out_dir: Path, **fields: Any) -> None: ...
    def __enter__(self) -> "SessionScope": ...
    def __exit__(self, *_exc: object) -> None: ...

class EventSink(SessionScope):
    """Batch-shaped sink. Returns the observer on __enter__."""
    def __init__(self, campaign_dir: Path, *, campaign: str) -> None:
        super().__init__(campaign_dir, campaign=campaign)
    def __enter__(self) -> StructlogObserver:
        super().__enter__()
        return StructlogObserver()
```

Net diff: ~40 lines moved, no behavior change for the existing CLI
path. Tests for `EventSink` keep passing unchanged. New tests for
`SessionScope` cover: file is created, contextvars are bound and
unbound, warnings route through, the prior `warnings.showwarning` is
restored on exit.

Risk: low. This is a pure extract-superclass with no signature change on
`EventSink`. The existing `tests/run/test_eventlog.py` is the
regression net.

## Phase 2 — pair-level events in `correlate_series`

Goal: structured `pair_start`/`pair_end` events, with `t_ref`/`t_def`
bound as contextvars **inside the per-pair block** so any phase record
fired from `correlate(...)` or the dispatcher inherits them.

`correlate_series` must remain pure. Resolution: an optional
`pair_observer` parameter typed as a small protocol; default `None`
means today's behavior verbatim.

```python
# mamba_dvc/pipeline/series.py
class SeriesPairObserver(Protocol):
    def on_pair_start(self, *, t_ref: int, t_def: int) -> None: ...
    def on_pair_end(
        self, *, t_ref: int, t_def: int, status: SeriesPairStatus,
        field: DisplacementField,
    ) -> None: ...

def correlate_series(
    frames, mask=None, *,
    pair_observer: SeriesPairObserver | None = None,
    ...,
) -> DisplacementSeries:
    ...
```

`_run_pair` becomes:

```python
def _run_pair(reference, deformed, t_ref, t_def):
    if pair_observer is not None:
        pair_observer.on_pair_start(t_ref=t_ref, t_def=t_def)
    try:
        ...
        return field, SeriesPairStatus.OK
    except Exception as exc:
        warnings.warn(...)
        return _failed_field(), SeriesPairStatus.FAILED
    finally:
        if pair_observer is not None:
            pair_observer.on_pair_end(
                t_ref=t_ref, t_def=t_def, status=status, field=field,
            )
```

The actual structlog-backed implementation lives in `run/eventlog.py`
as a `SeriesPairLogger` that calls `bind_contextvars(t_ref=..., t_def=...)`
on start and `unbind_contextvars(...)` on end. The function in
`pipeline/series.py` doesn't import structlog or stdlib `logging` — the
protocol decouples them.

This is the same separation `BatchObserver` already enforces: the
`pipeline/` and `run/` modules talk through a protocol, structlog lives
only in `run/eventlog.py`.

Risk: low. `pair_observer=None` is the existing path bit-identically.

## Phase 3 — worker-subprocess `ncc.*` records

This is the deferral called out in `mamba_dvc/run/eventlog.py`'s module
docstring ("Worker-subprocess gap (deferred)"). It's the highest-value
piece for diagnosing campaign perf, and the highest-risk piece in this
plan.

Design: `multiprocessing.Queue` + `logging.handlers.QueueListener`.

1. **Parent side.** `MultiGPUDispatcher.__enter__` creates one
   `multiprocessing.Queue` and one `QueueListener` whose handlers
   forward records to the `mamba_dvc.timing` logger (i.e. the *same*
   stdlib logger that `EventSink` already attaches the file handler to).
   The listener thread starts inside the `with` block, stops on
   `__exit__`.

2. **Worker side.** `_serve_worker(...)` initialization installs a
   `QueueHandler` on the worker's `mamba_dvc.timing` logger and sets
   its level to `DEBUG`. The handler enqueues records, the listener
   dequeues them and re-emits them on the parent's logger so
   `EventSink`'s file handler picks them up.

3. **`device_id` provenance.** Each worker injects `device_id` as a
   logging *extra* on every record before it goes through the queue,
   via a `logging.Filter` installed alongside the `QueueHandler`. After
   `_promote_mdvc_fields`, the JSON line carries
   `device_id` at the top level so post-hoc grouping by GPU is
   one-liner pandas.

```python
# mamba_dvc/gpu/dispatch.py (sketch, inside _serve_worker)
class _DeviceIdFilter(logging.Filter):
    def __init__(self, device_id: int) -> None:
        super().__init__()
        self._device_id = device_id
    def filter(self, record: logging.LogRecord) -> bool:
        record.mdvc_device_id = self._device_id
        return True

queue_handler = logging.handlers.QueueHandler(log_queue)
queue_handler.addFilter(_DeviceIdFilter(device_id))
timing_logger = logging.getLogger("mamba_dvc.timing")
timing_logger.addHandler(queue_handler)
timing_logger.setLevel(logging.DEBUG)
```

Pickle constraint: `QueueHandler` already converts `LogRecord` payloads
to a pickle-safe form, so the existing `mdvc_*` extras (strings, ints,
floats) pass through cleanly. Anything non-picklable on the record
would surface immediately; today's call sites pass only primitives, so
nothing changes on the worker side.

Lifecycle constraint: the listener must drain before `__exit__`
returns. `QueueListener.stop()` blocks until the queue is empty; place
it before the existing `_cleanup_state` call.

Pair-context constraint: the `t_ref` / `t_def` contextvars bound by
Phase 2 live in the parent. They do **not** propagate across the
process boundary. Two options:

- **(a)** Forward the current pair's `(t_ref, t_def)` into the worker
  on the per-pair request payload (already a `multiprocessing.Pipe`
  send in `_dispatch_pair_mp`). The worker re-binds them on its own
  contextvars stack for the duration of the pair, so when its records
  hit the queue they carry the right context.
- **(b)** Let the listener tag enqueued records with the parent's
  *current* contextvars (`merge_contextvars` runs in the parent thread
  on re-emit). This works because the parent blocks on the worker join
  for the entire pair, so the parent's contextvars are exactly the
  right pair's contextvars at the moment the listener forwards.

(b) is the cleaner separation — workers stay context-unaware — and
exploits the synchronous join we already do. Recommend (b).

Risk: medium. Process-boundary logging has known pitfalls (queue full,
listener thread crash, pickle errors on rich payloads). Mitigations:

- Bound the queue at a generous size (e.g. 1024) with `block=False`
  enqueue and a per-handler dropped-record counter; expose that
  counter as a `kind:"warning"` line on `__exit__`.
- Wrap the listener thread in the same defensive-isolation pattern as
  `Tee` so a single bad record can't tear down the campaign.
- Gate the worker-side handler behind a dispatcher-level flag
  (`emit_phase_records: bool = False` on `MultiGPUDispatcher`) so the
  default behavior of the dispatcher is unchanged until the eventlog
  consumer is ready.

## Phase 4 — wire `EventSink` / `SessionScope` into the harnesses

Two callers want the integration:

1. **`mamba_dvc/validate/synthetic.py::evaluate_synthetic`** — open a
   `SessionScope(out_dir, series=series_id, strategy=strategy.name, lag=lag)`
   around the outer sweep loop, supply a `SeriesPairLogger` to
   `correlate_series`.
2. **`scripts/eval_temporal_strategies.py`** — same pattern, plus a
   `kind:"sweep_start"` / `kind:"sweep_end"` bracket around the whole
   strategy × lag grid.

The CLI `mamba-dvc run` keeps `EventSink` unchanged. The CLI does not
need to know about the temporal pipeline yet.

Risk: low. Both call sites are top-level scripts; the wiring is
~10 lines each.

## Test plan

`tests/run/test_eventlog.py` already exercises `EventSink` end-to-end.
Mirror that structure:

1. **`test_session_scope_opens_and_closes`** — `SessionScope` lifetime
   creates the file, swaps warnings, binds session_id, restores
   everything on exit. Pin the file path and contextvar restoration.

2. **`test_correlate_series_emits_pair_events`** — drive a 3-frame
   synthetic series under `SessionScope` + `SeriesPairLogger`, parse
   `events.jsonl`, assert the sequence
   `series_start → (pair_start, pair_end)*N → series_end` with
   matching `t_ref`/`t_def` on each pair, status `OK` for every pair.

3. **`test_correlate_series_failure_emits_warning_and_pair_end`** —
   inject a `dispatcher` that raises on the second pair. Assert one
   `kind:"warning"` row with `category:"RuntimeWarning"`, one
   `kind:"pair_end"` row with `status:"FAILED"` for that pair, and
   subsequent pairs continue.

4. **`test_phase_records_carry_pair_context`** — under a `SessionScope`
   with a `SeriesPairLogger`, run one pair through the in-process
   dispatcher path (no multiprocessing). Assert every `kind:"phase"`
   row inside the pair window carries the bound `t_ref` / `t_def`
   contextvars.

5. **`test_worker_phase_records_propagate`** (`@pytest.mark.gpu`,
   skip if `<2` CUDA devices) — run one pair through a 2-GPU
   `MultiGPUDispatcher` with `emit_phase_records=True` inside a
   `SessionScope`. Assert at least one `kind:"phase"` row with
   `phase` starting `ncc.` and a top-level `device_id` field is in
   the file.

6. **`test_worker_phase_dropped_records_surface_as_warning`** — queue
   overflow simulation: monkeypatch the queue capacity to 1, drive >1
   record. Assert one `kind:"warning"` row on `__exit__` reporting the
   dropped count. Pins the defensive-isolation contract from Phase 3.

Tests 1–4 are CPU-only and run under `pytest -m "not gpu"`. Test 5 is
GPU + multi-device gated. Test 6 is CPU-only because it exercises the
queue-handler machinery without any GPU dispatch.

## Sequencing and commit shape

Four commits, in order, each green on `ruff format`, `ruff check`,
`pyright`, `pytest -m "not gpu"`:

1. `refactor(run): extract SessionScope from EventSink` — Phase 1,
   tests for the extracted scope.
2. `feat(pipeline): SeriesPairObserver hook in correlate_series` —
   Phase 2 protocol + threading through `_run_pair`. The structlog
   implementation `SeriesPairLogger` lands in `run/eventlog.py` in
   the same commit because the protocol is uninteresting without it.
3. `feat(gpu): worker phase-record forwarding via QueueListener` —
   Phase 3, behind the `emit_phase_records` flag so the default
   behavior is unchanged. Includes the dropped-record warning and the
   parent-side `device_id` propagation path.
4. `feat(validate,scripts): open SessionScope in synthetic sweeps` —
   Phase 4 wiring. `evaluate_synthetic` and
   `scripts/eval_temporal_strategies.py` adopt the new substrate.

Each commit moves one concern. A revert at any level rolls back to a
working state of the layer below.

## Risk assessment

- **Pure-core rule.** Phase 2 keeps `correlate_series` pure modulo the
  opt-in observer protocol. The protocol matches the precedent of
  `BatchObserver` in `run_batch`; no new coupling.
- **Performance.** The phase-records pipeline is already a no-op when
  the `mamba_dvc.timing` logger is at default (`WARNING`) — the
  `SessionScope` raises it to `DEBUG` only for its own lifetime, same
  as `EventSink` already does. Per-pair `bind_contextvars` /
  `unbind_contextvars` is a dict insert per call; sub-µs.
- **Multi-process logging.** Phase 3 is the real risk surface. The
  defensive mitigations (bounded queue, defensive listener, gated
  flag) keep failures non-fatal. The dispatcher's behavior with the
  flag off is bit-identical to today's behavior.
- **Schema stability.** Every event introduced here re-uses existing
  shapes from `EventSink`: `kind:"phase"` via `_promote_mdvc_fields`,
  `kind:"warning"` via the existing emitter, `kind:"pair_start"` /
  `kind:"pair_end"` as new structlog events with no special
  processing. Consumers parsing `events.jsonl` today read these as
  ordinary JSON lines with no schema migration.

## What an `events.jsonl` line carries after this work

Per-pair phase line, after Phase 3:

```json
{
  "ts": "2026-05-14T09:31:02.117Z",
  "session_id": "8c4a...",
  "series": "rat-103L-fs104",
  "strategy": "SEQUENTIAL",
  "t_ref": 4,
  "t_def": 5,
  "device_id": 2,
  "phase": "ncc.fft_ref",
  "seconds": 0.0184,
  "n_points": 125,
  "kind": "phase"
}
```

A four-line awk query gives per-pair, per-device, per-phase wall
totals across an entire 30-pair × 4-GPU campaign — exactly the
breakdown the temporal buildout currently has no way to produce.
