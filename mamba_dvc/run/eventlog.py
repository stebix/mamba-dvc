"""Single on-disk event stream for the run-campaign model.

The run-campaign model has three observability streams that each emit
correctly for their own concern but share no persistence layer:

1. :class:`~mamba_dvc.run.progress.BatchObserver` -- pull-based UI hook.
2. The ``mamba_dvc.timing`` stdlib logger -- per-phase wall times
   (:mod:`mamba_dvc.instrument`).
3. :func:`warnings.warn` -- footgun signals raised by the planner /
   loader (unpinned ``flow_convention``, unopenable stores, unmatched
   ``--store``).

This module adds one append-only ``events.jsonl`` capturing **all three
streams** through a shared ``structlog`` pipeline. structlog earns its
place as the *output substrate* (timestamping, JSON rendering,
contextvars-based field binding); the three emit-channels keep their
current call-site shapes because each is correct for its concern.

What is unified
---------------
- :class:`SessionScope` configures the structlog processor chain and
  attaches a stdlib :class:`logging.FileHandler` to
  ``mamba_dvc.eventlog`` (where :class:`StructlogObserver` writes) and
  ``mamba_dvc.timing`` (the phase records). Warnings reach the same
  file through a :func:`warnings.showwarning` override that calls the
  ``mamba_dvc.eventlog`` structlog logger directly with structured
  fields -- no :func:`logging.captureWarnings` bridge, no
  :func:`warnings.formatwarning` parsing. All three paths render
  identical JSON lines on the same file handle.
- :class:`SessionScope` raises the ``mamba_dvc.timing`` logger to
  ``DEBUG`` for its lifetime (and restores the prior level on
  ``__exit__``) so phase records always reach ``events.jsonl`` while
  events are on. The ``--timing`` CLI flag remains the gate for the
  end-of-run :func:`~mamba_dvc.instrument.accumulating` breakdown panel
  only.
- :class:`SessionScope` binds a fresh ``session_id`` (uuid4) plus any
  caller-supplied ``**fields`` as contextvars for its lifetime;
  :class:`EventSink` extends it with ``campaign``; :class:`StructlogObserver`
  binds ``store`` / ``deformation`` per load group and ``variant_id`` /
  ``kind_of_job`` per job. Every event in scope inherits them with no
  per-call boilerplate.

Two scope flavours
------------------
:class:`SessionScope` is the campaign-agnostic primitive — open it
directly with ``SessionScope(out_dir, series=..., strategy=...)`` from
non-batch callers (the temporal sweep harness is the v2 consumer).
:class:`EventSink` is the batch-shaped wrapper the CLI uses; it
composes a :class:`SessionScope` internally and its ``__enter__``
returns a pre-built :class:`StructlogObserver` ready to hand to
:func:`mamba_dvc.run.run_batch`.

What is *not* unified
---------------------
- The Rich renderer stays observer-driven (logging-handler-as-UI would
  couple schema to UI responsiveness).
- ``timed()`` / ``log_phase()`` keep their stdlib-logging emit sites.
  structlog enters only at the configuration boundary in the CLI.

Resume semantics
----------------
``events.jsonl`` is opened in append mode. Append is safe within a
single process and across **strictly-serial** ``mamba-dvc run``
re-invocations against the same campaign dir. **Concurrent invocations
against the same campaign dir are undefined behaviour**: Windows offers
no ``O_APPEND`` atomicity guarantee, and event lines exceed PIPE_BUF.
Each :class:`SessionScope` lifetime gets a fresh ``session_id`` uuid;
downstream consumers separate resumed sessions with
``df.groupby("session_id")`` (or by ``batch_start`` boundaries).

Threading
---------
The prefetch loader runs ``load_pair`` in a background thread
(:func:`mamba_dvc.run.batch._prefetched_load_groups`). ``ThreadPoolExecutor``
does **not** auto-propagate :mod:`contextvars` to worker threads (bpo-34014
was closed as out-of-date); the consumer-thread bindings are inherited
by the loader thread because the submit call wraps the callable in
``contextvars.copy_context().run``. Without that wrap, phase records
emitted from the loader thread would land without ``store`` /
``deformation`` / ``session_id`` fields, since
:func:`structlog.contextvars.merge_contextvars` runs at format time in
the emitting thread.

Worker-subprocess gap (deferred)
--------------------------------
``gpu.dispatch`` workers still don't propagate their ``ncc.*`` records
to the parent's logging configuration (see
:mod:`mamba_dvc.instrument`). The file format is forward-compatible: a
future ``multiprocessing.Queue`` + ``QueueListener`` bridge would land
those records as more ``kind:"phase"`` lines without schema change.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import warnings
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars
from structlog.processors import EventRenamer, JSONRenderer, TimeStamper
from structlog.stdlib import LoggerFactory, ProcessorFormatter

from mamba_dvc.run._jsonable import to_jsonable
from mamba_dvc.run.progress import BatchObserver

if TYPE_CHECKING:
    from collections.abc import Callable

    from mamba_dvc.run.batch import Job, JobResult
    from mamba_dvc.types import DisplacementField, POIStatus, SeriesPairStatus


__all__ = [
    "DispatchLogger",
    "EventSink",
    "SeriesPairLogger",
    "SessionScope",
    "StructlogObserver",
    "Tee",
]


_EVENTLOG_LOGGER_NAME: Final[str] = "mamba_dvc.eventlog"
_TIMING_LOGGER_NAME: Final[str] = "mamba_dvc.timing"

# Loggers the file handler attaches to. Adding ``mamba_dvc.memory`` here
# in Phase 2 routes memory snapshots to the same file with no schema
# change. Warnings reach the file via the structlog observer's own
# logger (see :meth:`SessionScope._show_warning`), so ``py.warnings`` is
# deliberately *not* on this list.
_FILE_LOGGERS: Final[tuple[str, ...]] = (
    _EVENTLOG_LOGGER_NAME,
    _TIMING_LOGGER_NAME,
)


# ----------------------------------------------------------------- processors


def _promote_mdvc_fields(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Promote ``mdvc_*`` LogRecord extras to top-level event-dict fields.

    The ``mamba_dvc.timing`` records carry their structured payload as
    ``mdvc_<name>`` attributes on the :class:`~logging.LogRecord` (see
    :mod:`mamba_dvc.instrument`). When :class:`ProcessorFormatter`
    ingests a foreign record into structlog's event dict, those
    attributes are available via ``_record``. Promote them with the
    prefix stripped and set ``event = "phase"`` so the discriminator
    renames it to ``kind: "phase"``.
    """
    record = event_dict.get("_record")
    if not isinstance(record, logging.LogRecord):
        return event_dict
    if record.name != _TIMING_LOGGER_NAME:
        return event_dict
    event_dict["event"] = "phase"
    for attr, value in record.__dict__.items():
        if attr.startswith("mdvc_"):
            event_dict[attr.removeprefix("mdvc_")] = value
    return event_dict


def _json_default(obj: object) -> object:
    """Coerce structlog event-dict values JSON's encoder cannot handle natively."""
    return to_jsonable(obj)


# ----------------------------------------------------------------- observer


class StructlogObserver:
    """A :class:`BatchObserver` that emits each hook as a structlog event.

    Hook → event ``kind``:

    ============================  ==================
    Hook                          Emitted ``kind``
    ============================  ==================
    ``on_batch_start``            ``batch_start``
    ``on_pair_load_start``        ``pair_load_start``
    ``on_job_start``              ``job_start``
    ``on_job_end``                ``job_end``
    ``on_batch_end``              ``batch_end``
    ============================  ==================

    Contextvars are bound in scope: ``store`` / ``deformation`` on
    ``on_pair_load_start`` (overwritten each new load group),
    ``variant_id`` / ``kind_of_job`` on ``on_job_start`` (unbound on
    ``on_job_end``). The owning :class:`EventSink` binds ``campaign``
    and ``session_id`` for its whole lifetime.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger(_EVENTLOG_LOGGER_NAME)
        self._batch_start_perf: float | None = None

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Emit ``kind:"batch_start"`` and remember the wall-clock start."""
        self._batch_start_perf = time.perf_counter()
        self._log.info("batch_start", n_jobs=n_jobs, n_variants=n_variants)

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Bind ``store`` / ``deformation`` contextvars; emit ``kind:"pair_load_start"``."""
        bind_contextvars(store=store.name, deformation=deformation)
        self._log.info("pair_load_start", n_variants=n_variants)

    def on_job_start(self, job: Job) -> None:
        """Bind ``variant_id`` / ``kind_of_job``; emit ``kind:"job_start"``."""
        bind_contextvars(variant_id=job.variant.variant_id, kind_of_job=job.kind)
        self._log.info("job_start")

    def on_job_end(self, result: JobResult) -> None:
        """Emit ``kind:"job_end"`` with status + timings; unbind per-job vars."""
        fields: dict[str, Any] = {"status": result.status}
        summary = result.summary
        for key in ("phase", "wall_correlate_s", "wall_evaluate_s"):
            if key in summary:
                fields[key] = summary[key]
        if result.error:
            # Same 2 KB convention as the manifest writer.
            fields["error_tail"] = result.error[-2000:]
        self._log.info("job_end", **fields)
        unbind_contextvars("variant_id", "kind_of_job")

    def on_batch_end(self, results: list[JobResult]) -> None:
        """Emit ``kind:"batch_end"`` with the ok/failed split and total duration."""
        n_ok = sum(1 for r in results if r.status == "ok")
        n_failed = sum(1 for r in results if r.status == "failed")
        duration_s: float | None = None
        if self._batch_start_perf is not None:
            duration_s = round(time.perf_counter() - self._batch_start_perf, 3)
        self._log.info("batch_end", n_ok=n_ok, n_failed=n_failed, duration_s=duration_s)


# ----------------------------------------------------------------- series observer


class SeriesPairLogger:
    """Structlog-backed :class:`mamba_dvc.types.SeriesPairObserver`.

    Binds ``t_ref`` / ``t_def`` contextvars on :meth:`on_pair_start` and
    unbinds them on :meth:`on_pair_end`, so every phase record emitted
    inside the pair window (``dispatch.*``, ``ncc.*``, ``evaluate.*``)
    inherits the temporal coordinates with no per-call boilerplate.
    Also emits explicit ``kind:"pair_start"`` and ``kind:"pair_end"``
    boundary lines so an analyst slicing ``events.jsonl`` can join
    phases to the originating pair without ad-hoc bracketing.

    Hook → event ``kind``:

    ============================  ==================
    Hook                          Emitted ``kind``
    ============================  ==================
    ``on_pair_start``             ``pair_start``
    ``on_pair_end``               ``pair_end``
    ============================  ==================

    The ``pair_end`` row carries ``status`` (the
    :class:`SeriesPairStatus` member name) and ``n_valid`` (the count of
    OK POIs on the returned field), so the file alone is sufficient to
    plot the success rate over a series without re-opening the field
    archives.

    Usage::

        with SessionScope(out_dir, series="rat-103L"):
            series = correlate_series(
                frames,
                dispatcher=disp,
                pair_observer=SeriesPairLogger(),
            )
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger(_EVENTLOG_LOGGER_NAME)

    def on_pair_start(self, *, t_ref: int, t_def: int) -> None:
        """Bind ``t_ref`` / ``t_def`` contextvars; emit ``kind:"pair_start"``."""
        bind_contextvars(t_ref=t_ref, t_def=t_def)
        self._log.info("pair_start")

    def on_pair_end(
        self,
        *,
        t_ref: int,
        t_def: int,
        status: SeriesPairStatus,
        field: DisplacementField,
    ) -> None:
        """Emit ``kind:"pair_end"`` with status + n_valid; unbind pair vars."""
        # t_ref / t_def reach the emitted row through the bound contextvars
        # set in on_pair_start. They are kwargs on the Protocol so callers
        # can drive the logger without relying on bind state, but the
        # outgoing row inherits them via merge_contextvars rather than an
        # explicit duplicate field.
        del t_ref, t_def
        self._log.info(
            "pair_end",
            status=status.name,
            n_valid=int(field.valid.sum()),
        )
        unbind_contextvars("t_ref", "t_def")


# ----------------------------------------------------------------- dispatch observer


class DispatchLogger:
    """Structlog-backed :class:`mamba_dvc.types.DispatchObserver`.

    Brackets every pair the
    :class:`mamba_dvc.gpu.dispatch.MultiGPUDispatcher` actually runs
    with explicit ``kind:"dispatch_pair_start"`` /
    ``kind:"dispatch_pair_end"`` lines so an analyst slicing
    ``events.jsonl`` sees the dispatch boundary even when the
    parent-process ``dispatch.*`` phase records are absent (e.g. a run
    with the ``mamba_dvc.timing`` logger at default ``WARNING``).

    Hook → event ``kind``:

    ============================  ==================
    Hook                          Emitted ``kind``
    ============================  ==================
    ``on_pair_start``             ``dispatch_pair_start``
    ``on_pair_end``               ``dispatch_pair_end``
    ============================  ==================

    The ``dispatch_pair_start`` row carries ``volume_shape`` (so the
    file is self-describing without joining against a separate
    dispatcher-construction event); ``dispatch_pair_end`` carries
    ``status_counts`` (a ``{POIStatus.name: count}`` dict, with
    zero-count statuses omitted) and ``n_valid``.

    Composition with :class:`SeriesPairLogger`
    ------------------------------------------
    When both loggers are active under the same
    :class:`SessionScope` and the dispatcher is driven from
    :func:`mamba_dvc.pipeline.correlate_series`, every dispatch row
    inherits the L1-bound ``t_ref`` / ``t_def`` contextvars
    automatically — the dispatcher's hooks fire while
    :class:`SeriesPairLogger` is mid-pair. Slicing
    ``events.jsonl`` by ``(t_ref, t_def)`` then groups
    ``pair_start`` / ``dispatch_pair_start`` / ``dispatch_pair_end`` /
    ``pair_end`` together with no ad-hoc bracket inference.

    Usage::

        with SessionScope(out_dir, series="rat-103L"):
            with MultiGPUDispatcher(
                volume_shape=ref.shape,
                mask=mask,
                device_ids=[0, 1, 2, 3],
                dispatch_observer=DispatchLogger(),
            ) as disp:
                series = correlate_series(
                    frames,
                    dispatcher=disp,
                    pair_observer=SeriesPairLogger(),
                )
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger(_EVENTLOG_LOGGER_NAME)

    def on_pair_start(self, *, volume_shape: tuple[int, int, int]) -> None:
        """Emit ``kind:"dispatch_pair_start"`` with the bound volume shape."""
        self._log.info("dispatch_pair_start", volume_shape=list(volume_shape))

    def on_pair_end(
        self,
        *,
        status_counts: dict[POIStatus, int],
        n_valid: int,
    ) -> None:
        """Emit ``kind:"dispatch_pair_end"`` with status_counts + n_valid.

        Keys in ``status_counts`` are stringified to the
        :class:`POIStatus` member name (``"OK"``, ``"MASKED"``, …)
        rather than the integer value, so a downstream JSONL reader
        does not have to rehydrate the enum to interpret the row.
        """
        self._log.info(
            "dispatch_pair_end",
            status_counts={k.name: int(v) for k, v in status_counts.items()},
            n_valid=int(n_valid),
        )


# ----------------------------------------------------------------- tee


class Tee:
    """Fan-out :class:`BatchObserver` composite with per-member exception isolation.

    Each hook dispatches to every member observer in order. A member
    that raises is caught: the failure is logged on
    ``mamba_dvc.eventlog`` at ``WARNING`` (so it lands in
    ``events.jsonl`` as a ``kind:"warning"`` line) and execution
    proceeds to the next member. This is load-bearing: the Rich
    renderer can throw if the terminal disconnects mid-run, and the
    file sink can throw on a full disk -- neither failure mode may
    break the other consumer, and neither may abort ``run_batch``. The
    :class:`BatchObserver` Protocol still requires implementors not to
    raise; :class:`Tee` is a defensive boundary, not an excuse.
    """

    def __init__(self, *observers: BatchObserver) -> None:
        self._observers: tuple[BatchObserver, ...] = observers
        self._log = logging.getLogger(_EVENTLOG_LOGGER_NAME)

    def _safe(
        self,
        hook: str,
        observer: BatchObserver,
        call: Callable[[], None],
    ) -> None:
        try:
            call()
        except Exception:
            self._log.warning(
                "tee observer %s.%s raised", type(observer).__name__, hook, exc_info=True
            )

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Dispatch ``on_batch_start`` to every member, isolating failures."""
        for obs in self._observers:
            self._safe(
                "on_batch_start",
                obs,
                lambda o=obs: o.on_batch_start(n_jobs=n_jobs, n_variants=n_variants),
            )

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Dispatch ``on_pair_load_start`` to every member, isolating failures."""
        for obs in self._observers:
            self._safe(
                "on_pair_load_start",
                obs,
                lambda o=obs: o.on_pair_load_start(store, deformation, n_variants=n_variants),
            )

    def on_job_start(self, job: Job) -> None:
        """Dispatch ``on_job_start`` to every member, isolating failures."""
        for obs in self._observers:
            self._safe("on_job_start", obs, lambda o=obs: o.on_job_start(job))

    def on_job_end(self, result: JobResult) -> None:
        """Dispatch ``on_job_end`` to every member, isolating failures."""
        for obs in self._observers:
            self._safe("on_job_end", obs, lambda o=obs: o.on_job_end(result))

    def on_batch_end(self, results: list[JobResult]) -> None:
        """Dispatch ``on_batch_end`` to every member, isolating failures."""
        for obs in self._observers:
            self._safe("on_batch_end", obs, lambda o=obs: o.on_batch_end(results))


# ----------------------------------------------------------------- sink


def _build_formatter() -> ProcessorFormatter:
    """Build the shared :class:`ProcessorFormatter` used on the file handler.

    Native structlog records and foreign stdlib records (``mamba_dvc.timing``,
    ``py.warnings``) both run through the same final processor chain --
    ``remove_processors_meta`` → :class:`EventRenamer` → :class:`JSONRenderer`
    -- so the resulting JSON shape is identical regardless of source.
    Foreign records additionally pass through ``foreign_pre_chain``, which
    timestamps them, merges contextvars, and applies the per-source
    promotion processors (``mdvc_*`` for timing records, parse for
    warnings).
    """
    foreign_pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        TimeStamper(fmt="iso", utc=True, key="ts"),
        _promote_mdvc_fields,
    ]
    final_processors: list[Any] = [
        ProcessorFormatter.remove_processors_meta,
        EventRenamer(to="kind"),
        JSONRenderer(serializer=json.dumps, default=_json_default),
    ]
    return ProcessorFormatter(foreign_pre_chain=foreign_pre_chain, processors=final_processors)


def _configure_structlog() -> None:
    """Install the global structlog config that routes native events to stdlib.

    Native ``structlog.get_logger(...).info(...)`` calls produce an
    event dict that the final processor (:meth:`ProcessorFormatter.wrap_for_formatter`)
    wraps as a :class:`logging.LogRecord` and hands to the
    stdlib logger of the same name. The stdlib :class:`FileHandler`
    then applies :func:`_build_formatter`, so native and foreign records
    converge on the same JSON shape.

    Called from :meth:`SessionScope.__enter__`. Idempotent: structlog's
    config is process-global and re-configuring it with the same
    processors is harmless.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            TimeStamper(fmt="iso", utc=True, key="ts"),
            ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class SessionScope:
    """Context manager owning the file handle, structlog config, and stdlib bridge.

    The campaign-agnostic substrate underneath :class:`EventSink`. Use
    directly when a caller wants ``events.jsonl`` semantics but does not
    fit the batch-campaign shape (e.g. the temporal sweep harness, which
    binds ``series`` / ``strategy`` rather than ``campaign``).

    Parameters
    ----------
    out_dir
        Directory under which ``events.jsonl`` is opened (append mode).
        Created on ``__enter__`` if it does not exist.
    **fields
        Arbitrary contextvar bindings applied for the scope's lifetime
        on top of a fresh ``session_id`` (uuid4). Every event emitted
        while the scope is open inherits them via
        :func:`structlog.contextvars.merge_contextvars`. Keys that
        collide with ``session_id`` raise ``TypeError`` at construction
        time -- ``session_id`` is reserved.

    Notes
    -----
    See module docstring for the unification stance, resume semantics,
    and threading caveats. Warnings are captured by replacing
    :func:`warnings.showwarning` for the scope's lifetime -- structured
    fields go in, structured fields come out, no
    :func:`warnings.formatwarning` parsing.

    Re-entry within the same process is supported (each
    ``with SessionScope(...) as s:`` block is self-contained); nested
    scopes are not -- nesting would double-bind ``session_id`` and break
    the resume-semantics contract documented at module level.
    """

    def __init__(self, out_dir: Path, **fields: Any) -> None:
        if "session_id" in fields:
            raise TypeError("'session_id' is reserved and bound automatically")
        self._out_dir = out_dir
        self._fields = fields
        self._handler: logging.FileHandler | None = None
        # Stash prior logger levels so __exit__ can put them back rather
        # than wiping them. Loggers under root inherit WARNING by default;
        # we raise them to allow the records SessionScope wants to capture
        # (INFO for the observer, DEBUG for phase timing).
        self._prior_levels: dict[str, int] = {}
        self._prior_showwarning: Any = None
        self._session_id: str = uuid.uuid4().hex

    @property
    def events_path(self) -> Path:
        """The ``events.jsonl`` path the scope writes to."""
        return self._out_dir / "events.jsonl"

    def __enter__(self) -> SessionScope:
        """Open the file, attach handlers, bind contextvars; return self."""
        _configure_structlog()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.events_path, mode="a", encoding="utf-8")
        handler.setFormatter(_build_formatter())
        handler.setLevel(logging.DEBUG)
        for name in _FILE_LOGGERS:
            logging.getLogger(name).addHandler(handler)

        # Raise loggers as needed: INFO for the observer's own emit site,
        # DEBUG for phase records (matches what accumulating() does).
        for name, level in (
            (_EVENTLOG_LOGGER_NAME, logging.INFO),
            (_TIMING_LOGGER_NAME, logging.DEBUG),
        ):
            logger = logging.getLogger(name)
            self._prior_levels[name] = logger.level
            logger.setLevel(level)

        # Swap warnings.showwarning for our structured emitter. No
        # logging.captureWarnings(True) probe, no formatwarning parsing.
        # We restore the *exact* prior callable on __exit__, so any other
        # consumer that had its own bridge (including
        # logging.captureWarnings) keeps working before and after.
        self._prior_showwarning = warnings.showwarning
        warnings.showwarning = self._show_warning  # type: ignore[assignment]

        bind_contextvars(session_id=self._session_id, **self._fields)
        self._handler = handler
        return self

    def __exit__(self, *_exc: object) -> None:
        """Detach handlers, restore prior state, close the file."""
        clear_contextvars()
        if self._prior_showwarning is not None:
            warnings.showwarning = self._prior_showwarning
            self._prior_showwarning = None
        for name, level in self._prior_levels.items():
            logging.getLogger(name).setLevel(level)
        self._prior_levels.clear()
        handler = self._handler
        if handler is not None:
            for name in _FILE_LOGGERS:
                logging.getLogger(name).removeHandler(handler)
            handler.close()
            self._handler = None

    def _show_warning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: ModuleType | None = None,
        line: str | None = None,
    ) -> None:
        """Emit one ``kind:"warning"`` event for a :func:`warnings.warn` call.

        Signature matches :func:`warnings.showwarning` exactly, so
        :func:`warnings.warn` invokes this with structured args -- no
        :func:`~warnings.formatwarning` round-trip, no regex parsing.
        ``file`` and ``line`` are accepted for signature compatibility
        and ignored: ``events.jsonl`` is the destination (no writing
        to ``file``), and ``line`` duplicates what the source location
        already conveys.
        """
        structlog.get_logger(_EVENTLOG_LOGGER_NAME).info(
            "warning",
            category=getattr(category, "__name__", str(category)),
            message=str(message),
            filename=filename,
            lineno=int(lineno),
        )


class EventSink:
    """Batch-shaped wrapper around :class:`SessionScope`.

    The CLI's ``mamba-dvc run`` opens this around :func:`run_batch` so
    the observer hooks plug into the existing :class:`BatchObserver`
    consumer chain. Composes a :class:`SessionScope` internally rather
    than subclassing it: ``__enter__`` returns a :class:`StructlogObserver`
    where the underlying scope returns ``self``, and that diverging
    return type is a contract break under LSP. Composition keeps the
    type relationship honest.

    Parameters
    ----------
    campaign_dir
        Directory under which ``events.jsonl`` is opened (append mode).
    campaign
        Campaign name; bound as the ``campaign`` contextvar for the
        sink's lifetime.

    Yields
    ------
    StructlogObserver
        Pre-built observer the CLI tees with its chosen renderer.
    """

    def __init__(self, campaign_dir: Path, *, campaign: str) -> None:
        self._scope = SessionScope(campaign_dir, campaign=campaign)

    @property
    def events_path(self) -> Path:
        """The ``events.jsonl`` path the sink writes to."""
        return self._scope.events_path

    def __enter__(self) -> StructlogObserver:
        """Open the underlying scope and return a fresh observer."""
        self._scope.__enter__()
        return StructlogObserver()

    def __exit__(self, *exc: object) -> None:
        """Close the underlying scope."""
        self._scope.__exit__(*exc)
