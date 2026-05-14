"""Phase timing for the DVC pipeline (lightweight, opt-in instrumentation).

The pipeline (:mod:`mamba_dvc.pipeline.correlate`,
:mod:`mamba_dvc.gpu.dispatch`, :mod:`mamba_dvc.validate.known_fields`,
:mod:`mamba_dvc.run.batch`) times its internal phases and emits one
structured :mod:`logging` record per phase on the ``mamba_dvc.timing``
logger at ``DEBUG``. Python's logging is silent below ``WARNING`` by
default, so a production caller pays only one
``logger.isEnabledFor(DEBUG)`` check per phase and sees nothing -- the
instrumentation is genuinely free when off.

Three ways to consume the records
---------------------------------
1. *Nothing* (the default) -- silent.
2. A handler on ``mamba_dvc.timing`` (or any ancestor) at ``DEBUG`` --
   one human-readable line per phase per invocation. Verbose; meant for
   one example, not a campaign.
3. A :class:`PhaseAccumulator` (a :class:`logging.Handler`) -- aggregates
   ``(count, total, min, max)`` per phase across every invocation while it
   is attached, then renders one table. This is what ``mamba-dvc run
   --timing`` uses so a 68-pair campaign prints *one* breakdown instead of
   68 noisy per-pair dumps. Use the :func:`accumulating` context manager
   to attach one and raise the logger level for its lifetime.

Each record carries the timing on ``LogRecord`` attributes
(``mdvc_phase`` -> :class:`str`, ``mdvc_seconds`` -> :class:`float`, plus
``mdvc_<name>`` for any extra fields the call site passed) so consumers
read structured data rather than parsing the message text; the message
string is a fallback for plain handlers.

GPU phases
----------
:func:`timed` and :class:`PhaseTimer` accept ``sync=True``: when CuPy is
importable and a device context is active, the wrapper calls
``cupy.cuda.get_current_stream().synchronize()`` before reading the clock
(on both entry and exit) so the measurement reflects kernel *execution*,
not just launch. The synchronisation only happens when the logger is
enabled, so it is a diagnostic-mode cost (and a deliberate one -- it is
the price of accurate per-phase numbers), not a production one.

Multi-process note
------------------
:mod:`mamba_dvc.gpu.dispatch` runs the per-batch NCC loop in ``spawn``ed
worker processes for ``len(device_ids) > 1``. A worker does not inherit
the parent's logging configuration by default, so the ``ncc.*`` and the
``dispatch.h2d`` / ``dispatch.helper`` / ``dispatch.d2h`` records are
dropped on multi-device runs unless the dispatcher was opened with
``emit_phase_records=True`` (added in L3 of the event-logging
buildout). With that flag set, each worker installs a non-blocking
:class:`logging.handlers.QueueHandler` on its ``mamba_dvc.timing``
logger and the records reach the parent's
:class:`~mamba_dvc.run.eventlog.SessionScope` file handler tagged with
``mdvc_device_id``. The coarse ``dispatch.*`` phases that run on the
parent (grid build, mask admission, the contiguous-array
materialisation, the worker join, the outlier test) are always visible.
For a detailed single-pair breakdown without the bridge, run the
single-device path (:func:`mamba_dvc.pipeline.correlate.correlate`, or
``correlate_multi_gpu(device_ids=[d])``).

Prefetch note
-------------
:mod:`mamba_dvc.run.batch` may run ``load_pair`` in a background loader
thread (``BatchSpec.prefetch >= 1``). The ``batch.load_pair`` record is
still emitted (``logging`` is thread-safe, so the accumulator folds it in
correctly), and its total is the real wall time spent loading — but that
time overlaps the consumer's ``dispatch.*`` / ``evaluate.*`` work, so it
is not all on the campaign's critical path. Read ``batch.load_pair`` as
"how much I/O happened", not "how much I/O cost the wall clock"; the sum
of the breakdown's phases can exceed the observed campaign wall time by
roughly the overlapped load time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

__all__ = ["PhaseAccumulator", "PhaseTimer", "accumulating", "log_phase", "timed"]

logger = logging.getLogger("mamba_dvc.timing")

# LogRecord attribute names carrying the structured payload. Prefixed so
# they cannot collide with logging's own reserved attributes (which would
# make ``logger.debug(..., extra=...)`` raise).
_PHASE_ATTR = "mdvc_phase"
_SECONDS_ATTR = "mdvc_seconds"
_FIELD_PREFIX = "mdvc_"

try:  # pragma: no cover - import guard
    import cupy as _cp  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - CPU-only hosts
    _cp = None  # type: ignore[assignment]


def _enabled() -> bool:
    """Return whether the ``mamba_dvc.timing`` logger would emit a DEBUG record."""
    return logger.isEnabledFor(logging.DEBUG)


def _sync() -> None:
    """Drain the current CuPy stream so a following clock read is accurate.

    No-op when CuPy is unavailable or no device context is active. Never
    raises -- instrumentation must not be able to crash the pipeline.
    """
    if _cp is None:
        return
    with suppress(Exception):  # pragma: no cover - defensive
        _cp.cuda.get_current_stream().synchronize()


def log_phase(phase: str, seconds: float, **fields: object) -> None:
    """Emit one ``DEBUG`` timing record on ``mamba_dvc.timing`` for a measured span.

    Use directly when a duration is already in hand (e.g. one computed
    across a ``try``/``except`` that :func:`timed` cannot wrap); otherwise
    prefer the :func:`timed` context manager.

    Parameters
    ----------
    phase
        Dotted phase name, e.g. ``"dispatch.total"``.
    seconds
        Elapsed wall time for the phase.
    **fields
        Extra context attached to the record as ``mdvc_<name>``
        attributes (and appended to the message), e.g. ``n_admitted=...``.

    Notes
    -----
    A no-op when the logger is not enabled for ``DEBUG``.
    """
    if not _enabled():
        return
    extra: dict[str, object] = {_PHASE_ATTR: phase, _SECONDS_ATTR: float(seconds)}
    for key, value in fields.items():
        extra[_FIELD_PREFIX + key] = value
    suffix = (" " + " ".join(f"{k}={v}" for k, v in fields.items())) if fields else ""
    logger.debug("phase=%s dt=%.4fs%s", phase, seconds, suffix, extra=extra)


@contextmanager
def timed(phase: str, *, sync: bool = False, **fields: object) -> Generator[None]:
    """Time the wrapped block; emit one ``DEBUG`` record on ``mamba_dvc.timing``.

    Parameters
    ----------
    phase
        Dotted phase name, e.g. ``"dispatch.filter_by_mask"``.
    sync
        If true, synchronise the current CuPy stream before reading the
        clock (entry and exit) so GPU work is fully accounted; ignored
        when CuPy is unavailable. See the module docstring.
    **fields
        Extra context attached to the record, e.g. ``n_points=...``.

    Notes
    -----
    Zero-cost when the ``mamba_dvc.timing`` logger is not enabled for
    ``DEBUG``: the block runs with no timing and no ``sync``.
    """
    if not _enabled():
        yield
        return
    if sync:
        _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            _sync()
        log_phase(phase, time.perf_counter() - t0, **fields)


class PhaseTimer:
    """Accumulate sub-phase wall time across an inner loop; flush once.

    Use when a phase contains a loop whose iterations should *not* each
    get their own record -- e.g. the batched NCC loop. Every ``with
    timer(name): ...`` block adds to ``name``'s running total; one call to
    :meth:`flush` emits one record per distinct name (in first-seen
    order), tagged with whatever ``**fields`` are passed (typically
    ``n_batches=...``). Like :func:`timed`, the whole object is inert when
    the logger is disabled (enablement is captured once at construction).

    Parameters
    ----------
    sync
        Forwarded to every ``with`` block: synchronise the current CuPy
        stream around each timed section (diagnostic-mode only).
    """

    def __init__(self, *, sync: bool = False) -> None:
        self._sync = sync
        self._enabled = _enabled()
        self._totals: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Generator[None]:
        """Time one section, adding the elapsed wall time to ``name``'s total."""
        if not self._enabled:
            yield
            return
        if self._sync:
            _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                _sync()
            self._totals[name] = self._totals.get(name, 0.0) + (time.perf_counter() - t0)

    def flush(self, **fields: object) -> None:
        """Emit one record per accumulated section, then clear the accumulator."""
        if not self._enabled:
            return
        for name, seconds in self._totals.items():
            log_phase(name, seconds, **fields)
        self._totals.clear()


@dataclass
class _PhaseStat:
    """Running ``(count, total, min, max)`` for one phase name."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0

    def update(self, seconds: float) -> None:
        """Fold one observed duration into the running stats."""
        self.count += 1
        self.total += seconds
        self.min = min(self.min, seconds)
        self.max = max(self.max, seconds)

    @property
    def mean(self) -> float:
        """Mean duration, or ``0.0`` when nothing has been recorded yet."""
        return self.total / self.count if self.count else 0.0


class PhaseAccumulator(logging.Handler):
    """A :class:`logging.Handler` that aggregates phase records into one table.

    Attach to the ``mamba_dvc.timing`` logger -- the :func:`accumulating`
    context manager does this and also raises the logger to ``DEBUG`` so
    :func:`timed` blocks actually fire. Every record produced by
    :func:`timed` / :func:`log_phase` / :class:`PhaseTimer` updates a
    running ``(count, total, min, max)`` for its phase name; :meth:`render`
    formats the lot as a fixed-width text table sorted by total time
    descending. Records lacking the timing attributes (an unrelated line
    that propagated here) are ignored.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.stats: dict[str, _PhaseStat] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Fold one timing record into the running per-phase stats."""
        phase = getattr(record, _PHASE_ATTR, None)
        seconds = getattr(record, _SECONDS_ATTR, None)
        if not isinstance(phase, str) or not isinstance(seconds, (int, float)):
            return
        self.stats.setdefault(phase, _PhaseStat()).update(float(seconds))

    def render(self, *, title: str | None = None) -> str:
        """Return a fixed-width table of the accumulated phases (sorted by total).

        An empty accumulator renders a single explanatory line -- e.g.
        when timing was never enabled, or every NCC batch ran in a worker
        subprocess whose records did not propagate.
        """
        if not self.stats:
            body = (
                "(no phases recorded — timing logger never enabled, or all phases "
                "ran in worker subprocesses)"
            )
            return f"{title}\n{body}" if title else body
        rows = sorted(self.stats.items(), key=lambda kv: kv[1].total, reverse=True)
        name_w = max(len("phase"), *(len(name) for name, _ in rows))
        header = (
            f"{'phase':<{name_w}}  {'count':>6}  {'total/s':>11}  "
            f"{'mean/s':>10}  {'min/s':>10}  {'max/s':>10}"
        )
        lines = [header, "-" * len(header)]
        for name, st in rows:
            lines.append(
                f"{name:<{name_w}}  {st.count:>6d}  {st.total:>11.3f}  "
                f"{st.mean:>10.4f}  {st.min:>10.4f}  {st.max:>10.4f}"
            )
        table = "\n".join(lines)
        return f"{title}\n{table}" if title else table


@contextmanager
def accumulating() -> Generator[PhaseAccumulator]:
    """Attach a fresh :class:`PhaseAccumulator` to ``mamba_dvc.timing`` for the block.

    Raises the ``mamba_dvc.timing`` logger to ``DEBUG`` (so :func:`timed`
    blocks fire), attaches a new accumulator, yields it, and on exit
    detaches the handler and restores the previous logger level. Nesting
    is supported -- each call gets an independent accumulator.

    Notes
    -----
    Enabling timing turns on the diagnostic-mode ``sync=True`` GPU
    synchronisations in the NCC loop, which prevent kernel overlap across
    sub-phases and so add a modest overhead -- the cost of getting
    accurate per-phase numbers.
    """
    handler = PhaseAccumulator()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()
