# Event-logging buildout — handoff

Snapshot for picking up the eventlog wiring in a fresh conversation. The
full design plan is `docs/triage/event-logging-integration.md`; this doc
is the **operational entry point** — where things stand on the branch,
the public surfaces you can use today, and what's still TODO.

- **Branch.** `temporal-buildout` (worktree:
  `C:\Users\jstebani\Desktop\worktrees__mamba-dvc\temporal-buildout`)
- **Merge base with trunk.** `9d3c560` (post-merge of the events.jsonl
  substrate from trunk into the temporal/multi-GPU buildout).
- **Status.** Foundations (L0, L1) shipped. Dispatcher-side wiring
  (L2) and worker-subprocess bridge (L3) are TODO. Harness wiring (L4)
  is unblocked but unshipped.

---

## What's done — L0 + L1

### L0 — `SessionScope` extracted from `EventSink`
Commit `846b923 refactor(run): extract SessionScope from EventSink`.

`mamba_dvc.run.eventlog.SessionScope` is the campaign-agnostic substrate
that owns the `events.jsonl` file handle, the structlog config, the
`warnings.showwarning` swap, and the level management. Takes an
`out_dir` plus arbitrary `**fields` as contextvar bindings on top of a
fresh `session_id` (uuid4). `session_id` is reserved — passing it
raises `TypeError`.

`EventSink` is now a **composition** wrapper around `SessionScope`, not
a subclass. The `__enter__` return-type diverges (`StructlogObserver`
vs `SessionScope`) and inheritance would break Liskov; composition
keeps the types honest. External API of `EventSink` is unchanged.

**File:** `mamba_dvc/run/eventlog.py`
**Tests:** `tests/run/test_eventlog.py::TestSessionScope` (5 tests)

### L1 — `SeriesPairObserver` Protocol + `SeriesPairLogger`
Commit `63b1094 feat(pipeline,run): SeriesPairObserver hook + SeriesPairLogger`.

- `mamba_dvc.types.SeriesPairObserver` — Protocol with
  `on_pair_start(*, t_ref, t_def)` and
  `on_pair_end(*, t_ref, t_def, status, field)`. Lives in `types.py`
  so consumer modules import only the Protocol, never structlog.
- `mamba_dvc.pipeline.series.correlate_series` — accepts an optional
  `pair_observer: SeriesPairObserver | None = None`. Default behavior
  is unchanged. Observer hooks fire in a documented order with the
  existing `on_pair` callback:

  ```
  pair_observer.on_pair_start
      → dispatch (correlate / dispatcher.correlate)
  pair_observer.on_pair_end
      → on_pair
  ```

  Per-pair failure isolation extends to the observer: a FAILED pair
  still produces `pair_end` with `status=FAILED` and a zero-filled
  field. The existing `RuntimeWarning` is still emitted.

- `mamba_dvc.run.eventlog.SeriesPairLogger` — structlog-backed
  implementation. Binds `t_ref` / `t_def` as contextvars on
  `on_pair_start` and unbinds on `on_pair_end`, so every phase record
  fired inside the pair window (`dispatch.*`, `ncc.*`, `evaluate.*`)
  inherits the temporal coordinates. Emits explicit `kind:"pair_start"`
  / `kind:"pair_end"` lines; `pair_end` carries `status.name` and
  `n_valid`.

**Files:** `mamba_dvc/types.py`, `mamba_dvc/pipeline/series.py`,
`mamba_dvc/run/eventlog.py`
**Tests:**
- `tests/pipeline/test_series.py::TestPairObserverHook` (3 tests —
  protocol contract, no structlog).
- `tests/run/test_eventlog.py::TestSeriesPairLogger` (5 tests — full
  parse-back against a real `SessionScope`).

### Test summary
- 728 non-GPU tests pass (~4 min). 13 new tests this session.
- 4 multi-pair persistence regression tests on the GPU (the bug-fix
  suite from `87096ce`) pass post-merge.
- `pytest -m gpu` full suite (30 tests) **not yet re-run** after L1;
  only the persistence subset has been verified on the workstation.

---

## How to use what's wired today

The temporal sweep can already produce a sliceable `events.jsonl`
without any further code changes on the dispatcher side. Minimal
pattern:

```python
from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
from mamba_dvc.pipeline.series import correlate_series
from mamba_dvc.run.eventlog import SeriesPairLogger, SessionScope

with SessionScope(
    out_dir, series="rat-103L-fs104", strategy="SEQUENTIAL"
) as scope:
    with MultiGPUDispatcher(
        volume_shape=ref.shape, mask=mask, device_ids=[0, 1, 2, 3], ...
    ) as disp:
        series = correlate_series(
            frames,
            mask=mask,
            dispatcher=disp,
            pair_observer=SeriesPairLogger(),
        )
print("events at:", scope.events_path)
```

What you get in `events.jsonl` *today*:

| Row | Source | Coverage |
|---|---|---|
| `kind:"pair_start"` / `kind:"pair_end"` | `SeriesPairLogger` | one per pair |
| `kind:"phase"` with `phase:"dispatch.*"` | `instrument.timed` in dispatcher parent | every pair, with `t_ref` / `t_def` bound |
| `kind:"phase"` with `phase:"ncc.*"` | dispatcher workers | **missing on multi-GPU runs** (L3 deferral) |
| `kind:"warning"` | per-pair `RuntimeWarning` from `correlate_series` | only on failures |

So per-pair *parent-process* visibility is sliceable now. Worker-side
`ncc.*` records remain dropped on multi-device runs until L3 lands.

---

## TODOs

Ordered by the sequencing in
`docs/triage/event-logging-integration.md`.

### L2 — Per-pair start/end events on `MultiGPUDispatcher`

Adds explicit pair-boundary events emitted by the dispatcher itself,
useful for ad-hoc dispatcher use (not just from inside
`correlate_series`).

- **Files.** `mamba_dvc/types.py` (`DispatchObserver` Protocol),
  `mamba_dvc/gpu/dispatch.py` (constructor + `correlate` hook),
  `mamba_dvc/run/eventlog.py` (`DispatchLogger` structlog impl),
  `tests/gpu/test_dispatch.py`.
- **Protocol surface (planned).**
  ```python
  class DispatchObserver(Protocol):
      def on_pair_start(self, *, volume_shape: tuple[int, int, int]) -> None: ...
      def on_pair_end(
          self, *, status_counts: dict[POIStatus, int], n_valid: int
      ) -> None: ...
  ```
- **Commits (planned, in order).**
  1. `feat(types,gpu): DispatchObserver hook on MultiGPUDispatcher`
     (Protocol + dispatcher hook with no consumer — reviewable alone).
  2. `feat(run): DispatchLogger + structured pair events in eventlog`
     (structlog consumer; lands after dispatcher contract is reviewed).
- **Composes with L1.** When both L1 and L2 are active, dispatcher
  `pair_start` / `pair_end` rows automatically carry the
  L1-bound `t_ref` / `t_def` contextvars.

### L3 — Worker-subprocess phase-record bridge

The deferral the `eventlog.py` module docstring explicitly calls out.
Highest-risk piece of the wiring (process-boundary logging).

- **Mechanism.** `multiprocessing.Queue` + `logging.handlers.QueueListener`.
  Workers install a `QueueHandler` on `mamba_dvc.timing` plus a
  `_DeviceIdFilter` that injects `mdvc_device_id` on every record.
  Parent's `QueueListener` thread re-emits records on its own timing
  logger where `SessionScope`'s file handler picks them up. After
  `_promote_mdvc_fields`, each JSON line carries top-level `device_id`.
- **Gating.** New `emit_phase_records: bool = False` on
  `MultiGPUDispatcher.__init__`. Default keeps current behavior
  bit-identical.
- **Pair-context propagation.** *Don't* propagate contextvars across
  the process boundary. The parent is synchronously blocked on the
  worker join during a pair; its contextvars at re-emit time are the
  correct pair's contextvars. Listener thread inherits them.
- **Backpressure.** Bounded queue (1024), non-blocking enqueue,
  drop counter surfaced as `kind:"warning"` on `__exit__`.
- **Tests.** `test_worker_phase_records_propagate` (GPU, skip-if-<2
  devices); `test_worker_phase_dropped_records_surface_as_warning`
  (CPU-only, monkeypatch queue maxsize=1).
- **Risk.** Pickle constraints on `LogRecord` payloads, listener
  thread crash isolation, queue full handling. The triage doc covers
  the mitigations.

### L4 — Wire `SessionScope` into the harnesses

Application-layer wiring. Two callers want it:

1. **`mamba_dvc/validate/synthetic.py::evaluate_synthetic`** — open
   a `SessionScope` around the outer sweep loop with bindings
   `series=...`, supply a `SeriesPairLogger` to `correlate_series`.
2. **`scripts/eval_temporal_strategies.py`** — same pattern plus a
   `kind:"sweep_start"` / `kind:"sweep_end"` bracket around the whole
   strategy × lag grid.

Unblocked by L0+L1 today — could land before L2/L3 if you want the
temporal sweep to start producing `events.jsonl` immediately. L2/L3
add detail to the rows already produced; they don't change the call
site.

---

## Open design decisions to revisit

1. **L1 contract: `pair_observer` / `on_pair` interleaving.** The
   docstring on `correlate_series.pair_observer` documents the order
   `on_pair_start → dispatch → on_pair_end → on_pair`. If L2 changes
   that order (e.g. `pair_observer.on_pair_start` could move *inside*
   `dispatcher.correlate` rather than around it), it's a contract
   change worth surfacing then. Today both sit at the same level in
   `_run_pair`.
2. **L2 protocol-vs-direct.** Triage doc recommends the Protocol
   approach (`DispatchObserver` in `types.py`, structlog impl in
   `run/eventlog.py`) for consistency with L1. The cheaper alternative
   is a direct structlog call inside `dispatch.py` itself; that
   couples `gpu/dispatch.py` to `run/eventlog.py`. Recommend keeping
   the Protocol pattern.
3. **L3 contextvar strategy.** Two options were sketched: (a) forward
   `(t_ref, t_def)` into the worker on the per-pair request payload
   and re-bind worker-side; (b) let the parent listener thread tag
   records with the parent's *current* contextvars at re-emit time.
   (b) is cleaner — workers stay context-unaware, exploits the
   synchronous worker join. Recommendation: (b).

---

## References

- **Full design plan.** `docs/triage/event-logging-integration.md` —
  the four-phase plan, including the code sketches and the test
  matrix for L3.
- **Related triage.** `docs/triage/multi-gpu-aliasing-bug.md` — the
  bug whose investigation motivated making the dispatch cycle
  debuggable in the first place.
- **Trunk-side substrate docs.** `docs/buildout/logging-pipeline.md`
  (from trunk's merge) — the original design of the `events.jsonl`
  substrate.
- **Key commits on this branch.**
  - `9d3c560` — merge: trunk events.jsonl substrate
  - `846b923` — L0 (SessionScope extract)
  - `63b1094` — L1 (SeriesPairObserver + SeriesPairLogger)

---

## Verification commands

```powershell
# Non-GPU regression (~4 min)
uv run pytest -m "not gpu"

# Multi-pair persistence regression suite (the bug-fix tests)
uv run pytest tests/gpu/test_dispatch.py::TestMultiGPUDispatcherMultiProcessPersistence

# Full GPU suite (~30 tests, requires CUDA)
uv run pytest -m gpu

# Format + lint + typecheck gate
uv run ruff format . ; uv run ruff check . ; uv run pyright
```

The pre-commit hooks mirror the lint/typecheck gate. `nbstripout`
is required in the main repo's venv (not auto-installed by
`uv sync`); install once with:

```powershell
uv pip install --python "C:/Users/jstebani/Desktop/mamba-dvc/.venv/Scripts/python.exe" nbstripout
```
