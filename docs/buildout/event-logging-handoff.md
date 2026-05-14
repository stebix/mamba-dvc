# Event-logging buildout — handoff

Snapshot for picking up the eventlog wiring in a fresh conversation. The
full design plan is `docs/triage/event-logging-integration.md`; this
doc is the **operational entry point** — where things stand on the
branch and the public surfaces you can use today.

- **Branch.** `temporal-buildout` (worktree:
  `C:\Users\jstebani\Desktop\worktrees__mamba-dvc\temporal-buildout`)
- **Merge base with trunk.** `9d3c560` (post-merge of the
  `events.jsonl` substrate from trunk into the temporal/multi-GPU
  buildout).
- **Status.** All four phases shipped — L0 (SessionScope extract),
  L1 (`SeriesPairObserver`), L2 (`DispatchObserver`), L3 (worker
  phase-record bridge), L4 (harness wiring). The temporal sweep
  produces a fully sliceable `events.jsonl` covering parent **and**
  worker phases out of the box.

---

## What shipped — L0 through L4

### L0 — `SessionScope` extracted from `EventSink`
Commit `846b923 refactor(run): extract SessionScope from EventSink`.

`mamba_dvc.run.eventlog.SessionScope` is the campaign-agnostic
substrate that owns the `events.jsonl` file handle, the structlog
config, the `warnings.showwarning` swap, and the level management.
Takes an `out_dir` plus arbitrary `**fields` as contextvar bindings on
top of a fresh `session_id` (uuid4). `session_id` is reserved — passing
it raises `TypeError`. `EventSink` is a composition wrapper around
`SessionScope`; its external API is unchanged.

### L1 — `SeriesPairObserver` Protocol + `SeriesPairLogger`
Commit `63b1094 feat(pipeline,run): SeriesPairObserver hook + SeriesPairLogger`.

- `mamba_dvc.types.SeriesPairObserver` — Protocol with
  `on_pair_start(*, t_ref, t_def)` and
  `on_pair_end(*, t_ref, t_def, status, field)`. Lives in `types.py`
  so consumer modules import only the Protocol, never structlog.
- `mamba_dvc.pipeline.series.correlate_series` — opt-in
  `pair_observer: SeriesPairObserver | None = None`. Default
  behavior unchanged. FAILED pairs still produce `pair_end` with
  `status=FAILED` and a zero-filled field; the per-pair
  `RuntimeWarning` is still emitted.
- `mamba_dvc.run.eventlog.SeriesPairLogger` — structlog-backed
  implementation. Binds `t_ref` / `t_def` as contextvars on
  `on_pair_start` and unbinds on `on_pair_end` so every phase record
  fired inside the pair window (parent-side `dispatch.*`,
  `evaluate.*`, plus L3 worker-side `ncc.*`) inherits the temporal
  coordinates.

### L2 — `DispatchObserver` Protocol + `DispatchLogger`
Commits `7c88a17 feat(types,gpu): DispatchObserver hook on MultiGPUDispatcher`
and `9fa80aa feat(run): DispatchLogger + structured pair events in eventlog`.

- `mamba_dvc.types.DispatchObserver` — Protocol with
  `on_pair_start(*, volume_shape)` and
  `on_pair_end(*, status_counts, n_valid)`.
- `MultiGPUDispatcher.__init__` accepts an optional
  `dispatch_observer` kwarg. Hooks fire at the top of
  `correlate()` and after the outlier test, once per `.correlate(...)`
  call. Composes cleanly with L1 — under both, dispatch rows
  inherit the L1-bound `t_ref` / `t_def` via `merge_contextvars`.
- `mamba_dvc.run.eventlog.DispatchLogger` — structlog-backed
  consumer. Emits `kind:"dispatch_pair_start"` (with `volume_shape`)
  and `kind:"dispatch_pair_end"` (with `status_counts` keyed by
  `POIStatus.name`, and `n_valid`).

### L3 — Worker-subprocess phase-record bridge
Commit `a7e08b0 feat(gpu,run): worker phase-record bridge via QueueListener`.

- New `emit_phase_records: bool = False` kwarg on
  `MultiGPUDispatcher.__init__`. Default keeps multi-process behavior
  bit-identical to pre-L3.
- When `True` and the dispatcher routes to multi-process
  (`len(device_ids) > 1`):
  - `__enter__` creates a `mp.Queue(maxsize=1024)` and starts a
    `logging.handlers.QueueListener` that forwards every dequeued
    record to the parent's `mamba_dvc.timing` logger (so
    SessionScope's file handler picks them up alongside parent-side
    `dispatch.*` rows).
  - Each worker installs a non-blocking
    `_NonBlockingQueueHandler` on its `mamba_dvc.timing` logger with
    a `_WorkerLogFilter` that stamps `mdvc_device_id` plus the
    per-pair context dict forwarded on each pair request.
  - Parent reads `structlog.contextvars.get_contextvars()` at
    pair-dispatch time and ships the snapshot to workers; workers
    re-stamp every key as `mdvc_<key>` extras. `_promote_mdvc_fields`
    on the parent strips the prefix uniformly. This is what makes
    `device_id` / `t_ref` / `t_def` / `session_id` show up inline on
    every worker-originating `kind:"phase"` row.
- Queue overflow: non-blocking enqueue, per-worker drop counter,
  surfaced as one `RuntimeWarning` per worker on `__exit__` (rendered
  by SessionScope as a `kind:"warning"` line).
- Worker protocol changed: pair message is now a 4-tuple
  `("pair", deformed_handle, ref_handle, pair_context | None)`;
  workers send a final `("dropped", count)` on the result pipe at
  shutdown when logging is on (the parent's recv on that message is
  the synchronisation point that orders all worker records ahead of
  the listener's drain).
- Single-device in-process path needs no L3 wiring — phase records
  there already flow through the parent's logging tree natively.

### L4 — `SessionScope` in synthetic sweeps
Commit `90f4cba feat(validate,scripts): open SessionScope in synthetic sweeps`.

- `mamba_dvc.validate.series_error.evaluate_synthetic` gains opt-in
  `pair_observer` / `dispatch_observer` kwargs. The harness binds
  `strategy` / `lag` as contextvars around each iteration when an
  observer is supplied; `dispatch_observer` is forwarded to the
  `MultiGPUDispatcher` when `device_ids` is set.
- `scripts/eval_temporal_strategies.py` opens a `SessionScope`
  around the whole sweep when not `--ephemeral`, with a `series`
  binding derived from shape/seed/timesteps. Brackets the sweep
  with `kind:"sweep_start"` / `kind:"sweep_end"` rows carrying the
  configuration + duration.

---

## How to use what's wired today

The temporal sweep produces a sliceable `events.jsonl` end-to-end,
including worker `ncc.*` rows, with the L3 flag set:

```python
from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
from mamba_dvc.pipeline.series import correlate_series
from mamba_dvc.run.eventlog import (
    DispatchLogger, SeriesPairLogger, SessionScope,
)

with SessionScope(
    out_dir, series="rat-103L-fs104", strategy="SEQUENTIAL"
) as scope:
    with MultiGPUDispatcher(
        volume_shape=ref.shape,
        mask=mask,
        device_ids=[0, 1, 2, 3],
        dispatch_observer=DispatchLogger(),
        emit_phase_records=True,  # L3 worker bridge
    ) as disp:
        series = correlate_series(
            frames,
            mask=mask,
            dispatcher=disp,
            pair_observer=SeriesPairLogger(),
        )
print("events at:", scope.events_path)
```

What lands in `events.jsonl` today (every row carries `session_id` +
any `**fields` bound on `SessionScope`):

| Row                                        | Source                            | Coverage                                       |
| ------------------------------------------ | --------------------------------- | ---------------------------------------------- |
| `kind:"pair_start"` / `kind:"pair_end"`    | `SeriesPairLogger`                | one per pair, with `t_ref` / `t_def`           |
| `kind:"dispatch_pair_start"` / `..._end`   | `DispatchLogger`                  | one per pair the dispatcher runs               |
| `kind:"phase"` with `phase:"dispatch.*"`   | parent `instrument.timed` blocks  | every pair                                     |
| `kind:"phase"` with `phase:"ncc.*"`        | workers (L3 bridge)               | every pair, carries `device_id` + pair context |
| `kind:"warning"`                           | per-pair `RuntimeWarning`         | failures + L3 queue-overflow drops             |

Single-device runs (`device_ids=[d]` or `device_ids=[0,0]` on a dev
host) need no `emit_phase_records=True` — phase records flow through
the parent's logging tree natively, and you still get parent-side
`dispatch.*` records. The flag is meaningful only when the dispatcher
actually spawns workers.

---

## References

- **Full design plan.** `docs/triage/event-logging-integration.md` —
  the four-phase plan, code sketches, and test matrix.
- **Related triage.** `docs/triage/multi-gpu-aliasing-bug.md` — the
  bug whose investigation motivated making the dispatch cycle
  debuggable in the first place.
- **Trunk-side substrate docs.** `docs/buildout/logging-pipeline.md`
  — the original design of the `events.jsonl` substrate.
- **Key commits on this branch.**
  - `9d3c560` — merge: trunk events.jsonl substrate
  - `846b923` — L0 (SessionScope extract)
  - `63b1094` — L1 (SeriesPairObserver + SeriesPairLogger)
  - `7c88a17` / `9fa80aa` — L2 (DispatchObserver Protocol + Logger)
  - `90f4cba` — L4 (SessionScope in synthetic sweeps)
  - `a7e08b0` — L3 (worker phase-record bridge via QueueListener)

---

## Verification commands

```powershell
# Non-GPU regression (~4 min)
uv run pytest -m "not gpu"

# Multi-pair persistence regression suite (the bug-fix tests)
uv run pytest tests/gpu/test_dispatch.py::TestMultiGPUDispatcherMultiProcessPersistence

# L3 end-to-end propagation (multi-process path, uses device_ids=[0,0]
# so it runs on a single-GPU dev host).
uv run pytest tests/gpu/test_dispatch.py::TestMultiGPUDispatcherEventLogging

# Format + lint + typecheck gate
uv run ruff format . ; uv run ruff check . ; uv run pyright
```

The pre-commit hooks mirror the lint/typecheck gate. `nbstripout` is
required in the main repo's venv (not auto-installed by `uv sync`);
install once with:

```powershell
uv pip install --python "C:/Users/jstebani/Desktop/mamba-dvc/.venv/Scripts/python.exe" nbstripout
```
