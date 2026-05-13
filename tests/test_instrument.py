"""Tests for the phase-timing instrumentation (``mamba_dvc.instrument``)."""

from __future__ import annotations

import logging

import pytest
from mamba_dvc import instrument
from mamba_dvc.instrument import PhaseAccumulator, PhaseTimer, accumulating, log_phase, timed

_TIMING_LOGGER = "mamba_dvc.timing"


def _make_record(phase: object, seconds: object) -> logging.LogRecord:
    record = logging.LogRecord(_TIMING_LOGGER, logging.DEBUG, __file__, 1, "msg", (), None)
    if phase is not None:
        record.mdvc_phase = phase  # type: ignore[attr-defined]
    if seconds is not None:
        record.mdvc_seconds = seconds  # type: ignore[attr-defined]
    return record


# --------------------------------------------------------------------- timed


def test_timed_silent_when_logger_disabled() -> None:
    """A bare ``timed`` block emits nothing at the default logger level."""
    sink = PhaseAccumulator()
    logging.getLogger(_TIMING_LOGGER).addHandler(sink)
    try:
        with timed("phase.a"):
            pass
    finally:
        logging.getLogger(_TIMING_LOGGER).removeHandler(sink)
    assert sink.stats == {}


def test_timed_emits_one_record_when_enabled() -> None:
    with accumulating() as acc, timed("phase.b", n=3):
        pass
    assert "phase.b" in acc.stats
    assert acc.stats["phase.b"].count == 1


def test_timed_sync_flag_is_harmless_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sync=True`` must not raise when CuPy is unavailable."""
    monkeypatch.setattr(instrument, "_cp", None)
    with accumulating() as acc, timed("phase.sync", sync=True):
        pass
    assert "phase.sync" in acc.stats


# ----------------------------------------------------------------- PhaseTimer


def test_phase_timer_accumulates_across_iterations() -> None:
    with accumulating() as acc:
        pt = PhaseTimer()
        for _ in range(3):
            with pt("loop.step"):
                pass
        with pt("loop.other"):
            pass
        pt.flush(n_batches=3)
    # Three ``with pt("loop.step")`` blocks collapse into one record.
    assert acc.stats["loop.step"].count == 1
    assert acc.stats["loop.other"].count == 1


def test_phase_timer_flush_clears_state() -> None:
    with accumulating() as acc:
        pt = PhaseTimer()
        with pt("x"):
            pass
        pt.flush()
        pt.flush()  # nothing left to emit
    assert acc.stats["x"].count == 1


def test_phase_timer_inert_when_disabled() -> None:
    """Constructed while disabled → no records even if the logger is later enabled."""
    pt = PhaseTimer()
    with pt("never"):
        pass
    with accumulating() as acc:
        pt.flush()
    assert acc.stats == {}


# ------------------------------------------------------------ PhaseAccumulator


def test_accumulator_aggregates_count_total_min_max() -> None:
    acc = PhaseAccumulator()
    acc.emit(_make_record("p", 1.5))
    acc.emit(_make_record("p", 2.5))
    acc.emit(_make_record("q", 0.25))
    assert acc.stats["p"].count == 2
    assert acc.stats["p"].total == pytest.approx(4.0)
    assert acc.stats["p"].min == pytest.approx(1.5)
    assert acc.stats["p"].max == pytest.approx(2.5)
    assert acc.stats["p"].mean == pytest.approx(2.0)
    assert acc.stats["q"].count == 1


def test_accumulator_ignores_records_without_timing_payload() -> None:
    acc = PhaseAccumulator()
    acc.emit(_make_record(None, None))
    acc.emit(_make_record("only-phase", None))
    acc.emit(_make_record(None, 1.0))
    acc.emit(_make_record(123, 1.0))  # phase not a str
    assert acc.stats == {}


def test_render_includes_phase_names_sorted_by_total() -> None:
    acc = PhaseAccumulator()
    acc.emit(_make_record("cheap", 0.1))
    acc.emit(_make_record("expensive", 9.0))
    table = acc.render(title="breakdown")
    assert "breakdown" in table
    assert "cheap" in table and "expensive" in table
    assert table.index("expensive") < table.index("cheap")


def test_render_empty_accumulator() -> None:
    assert "no phases" in PhaseAccumulator().render()


# ------------------------------------------------------------- accumulating()


def test_accumulating_restores_logger_state() -> None:
    logger = logging.getLogger(_TIMING_LOGGER)
    level_before = logger.level
    handlers_before = list(logger.handlers)
    with accumulating() as acc:
        assert logger.level == logging.DEBUG
        assert acc in logger.handlers
        log_phase("inside", 1.0)
    assert logger.level == level_before
    assert logger.handlers == handlers_before
    assert acc.stats["inside"].count == 1


def test_accumulating_nests_independently() -> None:
    with accumulating() as outer:
        with timed("a"):
            pass
        with accumulating() as inner, timed("b"):
            pass
        with timed("c"):
            pass
    # ``inner`` only saw the records emitted during its block.
    assert set(inner.stats) == {"b"}
    # ``outer`` saw everything (including those that also reached ``inner``).
    assert {"a", "b", "c"} <= set(outer.stats)
