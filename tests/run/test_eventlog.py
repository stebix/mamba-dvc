"""Tests for ``mamba_dvc.run.eventlog`` — the events.jsonl substrate."""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from mamba_dvc.instrument import accumulating, timed
from mamba_dvc.run.eventlog import (
    DispatchLogger,
    EventSink,
    SeriesPairLogger,
    SessionScope,
    StructlogObserver,
    Tee,
)
from mamba_dvc.types import (
    DisplacementField,
    POIStatus,
    SeriesPairStatus,
)

from tests.run.test_batch import (
    _fake_correlate,
    _FakeDataset,
    _opener_for,
    _spec,
)

# --------------------------------------------------------------- helpers


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Load every JSONL line in ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class _RecordingObserver:
    """Records each hook call so tests can compare invocation order/args."""

    def __init__(self, name: str = "rec") -> None:
        self.name = name
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        self.calls.append(("on_batch_start", (), {"n_jobs": n_jobs, "n_variants": n_variants}))

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        self.calls.append(
            ("on_pair_load_start", (store, deformation), {"n_variants": n_variants})
        )

    def on_job_start(self, job: object) -> None:
        self.calls.append(("on_job_start", (job,), {}))

    def on_job_end(self, result: object) -> None:
        self.calls.append(("on_job_end", (result,), {}))

    def on_batch_end(self, results: list[object]) -> None:
        self.calls.append(("on_batch_end", (results,), {}))


# --------------------------------------------------------------- lifecycle


class TestLifecycle:
    def test_lifecycle_events_round_trip(self, tmp_path: Path) -> None:
        # Tiny campaign through run_batch: one store, one synthetic
        # deformation, one variant. Expect every event kind to appear.
        from mamba_dvc.run import run_batch

        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"])
        with EventSink(spec.campaign_dir, campaign=spec.campaign) as obs:
            run_batch(
                spec,
                correlate_fn=_fake_correlate,
                open_dataset=_opener_for({store: ds}),
                observer=obs,
            )

        events = _read_events(spec.campaign_dir / "events.jsonl")
        kinds = [e["kind"] for e in events]
        assert kinds.count("batch_start") == 1
        assert kinds.count("pair_load_start") >= 1
        assert kinds.count("job_start") == kinds.count("job_end") >= 1
        assert kinds.count("batch_end") == 1

        # Every event carries campaign + session_id from EventSink.
        for ev in events:
            assert ev["campaign"] == spec.campaign
            assert "session_id" in ev
            assert "ts" in ev  # timestamper

        # job_start / job_end carry the per-job contextvars set by
        # StructlogObserver.
        job_starts = [e for e in events if e["kind"] == "job_start"]
        assert all("variant_id" in e and "kind_of_job" in e for e in job_starts)
        assert all(e["kind_of_job"] in {"real", "synthetic"} for e in job_starts)

        # job_end carries status + timings.
        job_ends = [e for e in events if e["kind"] == "job_end"]
        for ev in job_ends:
            assert ev["status"] in {"ok", "failed"}
            assert "wall_correlate_s" in ev

        # batch_end has the ok/failed split.
        be = next(e for e in events if e["kind"] == "batch_end")
        assert be["n_ok"] + be["n_failed"] == len(job_ends)

    def test_events_path_property(self, tmp_path: Path) -> None:
        sink = EventSink(tmp_path / "camp", campaign="c")
        assert sink.events_path == tmp_path / "camp" / "events.jsonl"


# --------------------------------------------------------------- phase records


class TestPhaseRouting:
    def test_phase_records_routed_through_sink(self, tmp_path: Path) -> None:
        # EventSink raises mamba_dvc.timing to DEBUG; phase records land
        # in events.jsonl without --timing.
        sink = EventSink(tmp_path, campaign="c")
        with sink, timed("ncc.batch", n_points=4):
            pass
        events = _read_events(sink.events_path)
        phase = next(e for e in events if e["kind"] == "phase")
        assert phase["phase"] == "ncc.batch"
        assert phase["seconds"] > 0
        assert phase["n_points"] == 4
        assert phase["campaign"] == "c"

    def test_phase_records_routed_alongside_accumulating(self, tmp_path: Path) -> None:
        # accumulating() is the --timing CLI mechanism. Stacking it with
        # EventSink must produce both the file line and the in-memory
        # accumulation (independent handlers on the same logger).
        sink = EventSink(tmp_path, campaign="c")
        with sink, accumulating() as acc, timed("dispatch.total"):
            pass
        events = _read_events(sink.events_path)
        assert any(e["kind"] == "phase" and e["phase"] == "dispatch.total" for e in events)
        assert "dispatch.total" in acc.stats


# --------------------------------------------------------------- warnings


class TestWarnings:
    def test_warning_lands_as_kind_warning(self, tmp_path: Path) -> None:
        # warnings.warn -> our showwarning override -> structlog. No
        # captureWarnings bridge, no formatwarning parsing.
        sink = EventSink(tmp_path, campaign="c")
        with sink:
            warnings.warn("flow_convention is unset", UserWarning, stacklevel=1)
        events = _read_events(sink.events_path)
        warn = next(e for e in events if e["kind"] == "warning")
        assert warn["category"] == "UserWarning"
        assert warn["message"] == "flow_convention is unset"
        assert isinstance(warn["lineno"], int)
        assert isinstance(warn["filename"], str)

    def test_showwarning_restored_on_exit(self, tmp_path: Path) -> None:
        # __exit__ must put the prior warnings.showwarning back so other
        # consumers (or the default text-to-stderr behavior) survive.
        prior = warnings.showwarning
        with EventSink(tmp_path, campaign="c"):
            assert warnings.showwarning is not prior
        assert warnings.showwarning is prior

    def test_warning_carries_campaign_and_session(self, tmp_path: Path) -> None:
        # Sanity that the override emits inside the contextvars scope.
        sink = EventSink(tmp_path, campaign="c")
        with sink:
            warnings.warn("hi", UserWarning, stacklevel=1)
        warn = next(e for e in _read_events(sink.events_path) if e["kind"] == "warning")
        assert warn["campaign"] == "c"
        assert "session_id" in warn


# --------------------------------------------------------------- contextvars


class TestContextInheritance:
    def test_explicit_contextvars_appear_and_clear(self, tmp_path: Path) -> None:
        sink = EventSink(tmp_path, campaign="c")
        log = structlog.get_logger("mamba_dvc.eventlog")
        with sink:
            structlog.contextvars.bind_contextvars(store="s1", deformation="d1")
            log.info("custom_a")
            structlog.contextvars.unbind_contextvars("store", "deformation")
            log.info("custom_b")
        events = _read_events(sink.events_path)
        ev_a = next(e for e in events if e["kind"] == "custom_a")
        ev_b = next(e for e in events if e["kind"] == "custom_b")
        assert ev_a["store"] == "s1"
        assert ev_a["deformation"] == "d1"
        assert "store" not in ev_b
        assert "deformation" not in ev_b

    def test_sink_exit_clears_contextvars(self, tmp_path: Path) -> None:
        # After __exit__ no campaign / session_id should leak into events
        # emitted on a fresh sink against a different file.
        sink_a = EventSink(tmp_path / "a", campaign="A")
        with sink_a:
            structlog.get_logger("mamba_dvc.eventlog").info("inside_a")
        sink_b = EventSink(tmp_path / "b", campaign="B")
        with sink_b:
            structlog.get_logger("mamba_dvc.eventlog").info("inside_b")
        ev_b = _read_events(sink_b.events_path)
        assert all(e["campaign"] == "B" for e in ev_b)


# --------------------------------------------------------------- threading


class TestThreadSafety:
    def test_contextvars_propagate_via_copy_context(self, tmp_path: Path) -> None:
        # The critical regression test: emit a phase record from a
        # worker thread, *with* copy_context().run wrap, and assert the
        # store/deformation bound in the parent appear on the line.
        # Without the wrap, those fields would be absent — that's the
        # bug the original plan's threading note implied did not exist.
        sink = EventSink(tmp_path, campaign="c")
        with sink:
            structlog.contextvars.bind_contextvars(store="parent_s", deformation="parent_d")
            with ThreadPoolExecutor(max_workers=1) as pool:
                ctx = contextvars.copy_context()

                def _emit() -> None:
                    with timed("batch.load_pair", deformation="parent_d"):
                        pass

                pool.submit(ctx.run, _emit).result()
        events = _read_events(sink.events_path)
        phase = next(e for e in events if e["kind"] == "phase")
        assert phase["store"] == "parent_s"
        assert phase["deformation"] == "parent_d"

    def test_load_pair_record_carries_store_with_prefetch(self, tmp_path: Path) -> None:
        # Regression: in the real run_batch sequence,
        # StructlogObserver.on_pair_load_start binds the ``store`` /
        # ``deformation`` contextvars *after* the loader future has
        # already been submitted via copy_context() -- so the snapshot
        # the loader thread inherits cannot include them. ``deformation``
        # arrives anyway because _load_or_capture passes it explicitly
        # to ``timed("batch.load_pair", ...)``; ``store`` does not unless
        # it is plumbed the same way. Drive run_batch end-to-end with
        # prefetch=1 (so the loader thread is in play) and assert the
        # ``batch.load_pair`` row carries the store name.
        from mamba_dvc.run import run_batch

        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], prefetch=1)
        with EventSink(spec.campaign_dir, campaign=spec.campaign) as obs:
            run_batch(
                spec,
                correlate_fn=_fake_correlate,
                open_dataset=_opener_for({store: ds}),
                observer=obs,
            )
        events = _read_events(spec.campaign_dir / "events.jsonl")
        load_pair = [
            e for e in events if e["kind"] == "phase" and e["phase"] == "batch.load_pair"
        ]
        assert load_pair, "expected at least one batch.load_pair phase record"
        for ev in load_pair:
            assert ev.get("store") == "scanA.zarr", (
                f"batch.load_pair must carry store; got {ev!r}"
            )
            assert ev.get("deformation") == "fs1"

    def test_lines_well_formed_under_concurrent_emit(self, tmp_path: Path) -> None:
        # Concurrent emits from multiple threads must produce one
        # well-formed JSON line per record (FileHandler holds an RLock
        # around format+write+flush).
        sink = EventSink(tmp_path, campaign="c")
        with sink:

            def _spam(label: str) -> None:
                for i in range(20):
                    with timed("batch.load_pair", deformation=f"{label}-{i}"):
                        pass

            threads = [
                threading.Thread(target=_spam, args=(name,)) for name in ("a", "b", "c")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        events = _read_events(sink.events_path)
        # 3 threads * 20 emits = 60 phase lines, all parseable.
        phase = [e for e in events if e["kind"] == "phase"]
        assert len(phase) == 60


# --------------------------------------------------------------- tee


class TestTee:
    def test_fans_out_to_every_member(self) -> None:
        a = _RecordingObserver("a")
        b = _RecordingObserver("b")
        tee = Tee(a, b)
        tee.on_batch_start(n_jobs=2, n_variants=1)
        tee.on_pair_load_start(Path("s"), "d", n_variants=1)
        tee.on_job_start(object())
        tee.on_job_end(object())
        tee.on_batch_end([])
        # Both observers see identical hook sequences.
        assert [c[0] for c in a.calls] == [c[0] for c in b.calls]
        assert len(a.calls) == 5

    def test_raising_member_does_not_block_others(self) -> None:
        class _Bad:
            def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
                raise RuntimeError("boom")

            def on_pair_load_start(self, *_: object, **__: object) -> None:
                pass

            def on_job_start(self, *_: object, **__: object) -> None:
                pass

            def on_job_end(self, *_: object, **__: object) -> None:
                pass

            def on_batch_end(self, *_: object, **__: object) -> None:
                pass

        good = _RecordingObserver("good")
        tee = Tee(_Bad(), good)
        # Should not raise.
        tee.on_batch_start(n_jobs=1, n_variants=1)
        assert good.calls and good.calls[0][0] == "on_batch_start"

    def test_failure_logged_to_eventlog(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _Bad:
            def on_batch_end(self, results: list[object]) -> None:
                raise ValueError("kaboom")

            def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
                pass

            def on_pair_load_start(self, *_: object, **__: object) -> None:
                pass

            def on_job_start(self, *_: object, **__: object) -> None:
                pass

            def on_job_end(self, *_: object, **__: object) -> None:
                pass

        tee = Tee(_Bad())
        caplog.set_level(logging.WARNING, logger="mamba_dvc.eventlog")
        tee.on_batch_end([])
        warnings_emitted = [r for r in caplog.records if "tee observer" in r.getMessage()]
        assert warnings_emitted, "expected a tee fanout WARNING"
        assert "_Bad" in warnings_emitted[0].getMessage()
        assert "on_batch_end" in warnings_emitted[0].getMessage()


# --------------------------------------------------------------- session scope


class TestSessionScope:
    """Cover the campaign-agnostic base scope.

    Most of the file-handle / warning-swap / structlog-config substrate
    is exercised by every :class:`EventSink` test above (EventSink is a
    SessionScope). These tests target the surface that's new:
    arbitrary ``**fields`` binding, the ``session_id`` reservation, and
    standalone (non-batch) usage.
    """

    def test_round_trip_with_arbitrary_fields(self, tmp_path: Path) -> None:
        # SessionScope is the temporal-sweep entry point; pass series /
        # strategy as bindings and confirm they appear on emitted events
        # without any 'campaign' binding leaking in.
        scope = SessionScope(tmp_path, series="rat-103L-fs104", strategy="SEQUENTIAL")
        with scope, timed("ncc.fft_ref", n_points=64):
            pass
        events = _read_events(scope.events_path)
        phase = next(e for e in events if e["kind"] == "phase")
        assert phase["phase"] == "ncc.fft_ref"
        assert phase["series"] == "rat-103L-fs104"
        assert phase["strategy"] == "SEQUENTIAL"
        assert "campaign" not in phase
        assert "session_id" in phase

    def test_session_id_is_reserved(self, tmp_path: Path) -> None:
        # session_id is bound automatically; collisions are user error,
        # not silent overwrites.
        with pytest.raises(TypeError, match="reserved"):
            SessionScope(tmp_path, session_id="manual")

    def test_session_id_is_fresh_per_scope(self, tmp_path: Path) -> None:
        # Two scopes open serially against different dirs get distinct
        # session_ids — the contract resume consumers rely on for the
        # df.groupby("session_id") workflow.
        with SessionScope(tmp_path / "a") as s_a:
            structlog.get_logger("mamba_dvc.eventlog").info("ping")
            sid_a = s_a._session_id
        with SessionScope(tmp_path / "b") as s_b:
            structlog.get_logger("mamba_dvc.eventlog").info("ping")
            sid_b = s_b._session_id
        assert sid_a != sid_b
        ev_a = _read_events(s_a.events_path)
        ev_b = _read_events(s_b.events_path)
        assert all(e["session_id"] == sid_a for e in ev_a)
        assert all(e["session_id"] == sid_b for e in ev_b)

    def test_warnings_routed_without_campaign(self, tmp_path: Path) -> None:
        # The warning-swap path must work on a bare SessionScope (no
        # 'campaign' field) — this is the exact configuration the
        # temporal sweep will use.
        scope = SessionScope(tmp_path, series="rat-103L-fs104")
        with scope:
            warnings.warn("flow_convention is unset", UserWarning, stacklevel=1)
        warn = next(e for e in _read_events(scope.events_path) if e["kind"] == "warning")
        assert warn["category"] == "UserWarning"
        assert warn["series"] == "rat-103L-fs104"
        assert "campaign" not in warn

    def test_event_sink_still_binds_campaign(self, tmp_path: Path) -> None:
        # EventSink is now a SessionScope subclass; confirm the campaign
        # binding still goes through unchanged after the refactor.
        sink = EventSink(tmp_path, campaign="cmp-A")
        with sink as obs:
            assert isinstance(obs, StructlogObserver)
            structlog.get_logger("mamba_dvc.eventlog").info("ping")
        ev = next(e for e in _read_events(sink.events_path) if e["kind"] == "ping")
        assert ev["campaign"] == "cmp-A"
        assert "session_id" in ev


# --------------------------------------------------------------- series pair logger


def _make_displacement_field(*, n_ok: int, n_total: int) -> DisplacementField:
    """Build a minimal :class:`DisplacementField` with a known OK count."""
    status = np.full(n_total, POIStatus.MASKED, dtype=np.uint8)
    status[:n_ok] = POIStatus.OK
    return DisplacementField(
        positions=np.zeros((n_total, 3), dtype=np.float32),
        displacements=np.zeros((n_total, 3), dtype=np.float32),
        valid=(status == POIStatus.OK),
        confidence=np.zeros(n_total, dtype=np.float32),
        status=status,
        grid_shape=(n_total, 1, 1),
        spacing=(1, 1, 1),
        window=(8, 8, 8),
    )


class TestSeriesPairLogger:
    """Cover the structlog-backed :class:`SeriesPairObserver`."""

    def test_pair_start_and_pair_end_round_trip(self, tmp_path: Path) -> None:
        # The canonical observable: one pair_start line, one pair_end
        # line, both carrying t_ref / t_def via the contextvar binding.
        log = SeriesPairLogger()
        field = _make_displacement_field(n_ok=7, n_total=10)
        with SessionScope(tmp_path, series="rat-103L"):
            log.on_pair_start(t_ref=0, t_def=1)
            log.on_pair_end(t_ref=0, t_def=1, status=SeriesPairStatus.OK, field=field)
        events = _read_events(tmp_path / "events.jsonl")
        kinds = [e["kind"] for e in events]
        assert kinds == ["pair_start", "pair_end"]
        for ev in events:
            assert ev["t_ref"] == 0
            assert ev["t_def"] == 1
            assert ev["series"] == "rat-103L"

    def test_pair_end_carries_status_name_and_n_valid(self, tmp_path: Path) -> None:
        # pair_end is the row a downstream analyst joins to per-pair
        # success counts; status is the SeriesPairStatus member name,
        # n_valid is field.valid.sum().
        log = SeriesPairLogger()
        field = _make_displacement_field(n_ok=42, n_total=125)
        with SessionScope(tmp_path):
            log.on_pair_start(t_ref=2, t_def=3)
            log.on_pair_end(t_ref=2, t_def=3, status=SeriesPairStatus.OK, field=field)
        end = next(
            e for e in _read_events(tmp_path / "events.jsonl") if e["kind"] == "pair_end"
        )
        assert end["status"] == "OK"
        assert end["n_valid"] == 42

    def test_failed_pair_end_emits_status_failed(self, tmp_path: Path) -> None:
        # FAILED is the other SeriesPairStatus member; verify the
        # status string is the enum name, not the int value.
        log = SeriesPairLogger()
        field = _make_displacement_field(n_ok=0, n_total=10)
        with SessionScope(tmp_path):
            log.on_pair_start(t_ref=5, t_def=6)
            log.on_pair_end(t_ref=5, t_def=6, status=SeriesPairStatus.FAILED, field=field)
        end = next(
            e for e in _read_events(tmp_path / "events.jsonl") if e["kind"] == "pair_end"
        )
        assert end["status"] == "FAILED"
        assert end["n_valid"] == 0

    def test_phase_records_inside_pair_inherit_t_ref_t_def(self, tmp_path: Path) -> None:
        # The whole point of the contextvar binding: any phase record
        # fired between on_pair_start and on_pair_end carries t_ref /
        # t_def, so events.jsonl is sliceable per-pair without ad-hoc
        # bracket inference. This is the regression test for the L1
        # contract documented in docs/triage/event-logging-integration.md.
        log = SeriesPairLogger()
        field = _make_displacement_field(n_ok=5, n_total=10)
        with SessionScope(tmp_path):
            log.on_pair_start(t_ref=4, t_def=5)
            with timed("dispatch.build_grid"):
                pass
            with timed("ncc.fft_ref"):
                pass
            log.on_pair_end(t_ref=4, t_def=5, status=SeriesPairStatus.OK, field=field)
        phases = [e for e in _read_events(tmp_path / "events.jsonl") if e["kind"] == "phase"]
        assert len(phases) == 2
        for p in phases:
            assert p["t_ref"] == 4
            assert p["t_def"] == 5

    def test_t_ref_t_def_unbound_after_pair_end(self, tmp_path: Path) -> None:
        # Pair vars must not leak past on_pair_end -- otherwise a phase
        # record emitted between pairs (e.g. from a teardown step in
        # the dispatcher) would attribute itself to the previous pair.
        log = SeriesPairLogger()
        field = _make_displacement_field(n_ok=1, n_total=2)
        with SessionScope(tmp_path):
            log.on_pair_start(t_ref=10, t_def=11)
            log.on_pair_end(t_ref=10, t_def=11, status=SeriesPairStatus.OK, field=field)
            with timed("between_pairs"):
                pass
        between = next(
            e
            for e in _read_events(tmp_path / "events.jsonl")
            if e["kind"] == "phase" and e["phase"] == "between_pairs"
        )
        assert "t_ref" not in between
        assert "t_def" not in between


# --------------------------------------------------------------- dispatch logger


class TestDispatchLogger:
    """Cover the structlog-backed :class:`DispatchObserver`."""

    def test_pair_start_and_pair_end_round_trip(self, tmp_path: Path) -> None:
        # Canonical observable: one dispatch_pair_start line, one
        # dispatch_pair_end line, both inheriting the SessionScope
        # bindings (no contextvar binding inside the logger itself).
        log = DispatchLogger()
        with SessionScope(tmp_path, series="rat-103L"):
            log.on_pair_start(volume_shape=(48, 48, 48))
            log.on_pair_end(
                status_counts={POIStatus.OK: 5, POIStatus.MASKED: 3},
                n_valid=5,
            )
        events = _read_events(tmp_path / "events.jsonl")
        kinds = [e["kind"] for e in events]
        assert kinds == ["dispatch_pair_start", "dispatch_pair_end"]
        for ev in events:
            assert ev["series"] == "rat-103L"
            assert "session_id" in ev

    def test_pair_start_carries_volume_shape(self, tmp_path: Path) -> None:
        # volume_shape is the only payload on the start row; rendered as
        # a JSON array (length 3) so a downstream reader can index it
        # without a tuple round-trip.
        log = DispatchLogger()
        with SessionScope(tmp_path):
            log.on_pair_start(volume_shape=(960, 1280, 1280))
            log.on_pair_end(status_counts={POIStatus.OK: 1}, n_valid=1)
        start = next(
            e
            for e in _read_events(tmp_path / "events.jsonl")
            if e["kind"] == "dispatch_pair_start"
        )
        assert start["volume_shape"] == [960, 1280, 1280]

    def test_pair_end_status_counts_use_member_names(self, tmp_path: Path) -> None:
        # status_counts keys must be POIStatus.name strings, not the
        # int values — avoids a rehydrate step in the JSONL reader and
        # keeps the row human-grep-able.
        log = DispatchLogger()
        with SessionScope(tmp_path):
            log.on_pair_start(volume_shape=(8, 8, 8))
            log.on_pair_end(
                status_counts={
                    POIStatus.OK: 7,
                    POIStatus.MASKED: 2,
                    POIStatus.OUTLIER: 1,
                },
                n_valid=7,
            )
        end = next(
            e
            for e in _read_events(tmp_path / "events.jsonl")
            if e["kind"] == "dispatch_pair_end"
        )
        assert end["status_counts"] == {"OK": 7, "MASKED": 2, "OUTLIER": 1}
        assert end["n_valid"] == 7

    def test_empty_status_counts_renders_as_empty_dict(self, tmp_path: Path) -> None:
        # The dispatcher's _status_histogram omits zero-count members;
        # an all-empty pair yields {} and DispatchLogger must pass that
        # through verbatim, not coerce to None or drop the key.
        log = DispatchLogger()
        with SessionScope(tmp_path):
            log.on_pair_start(volume_shape=(4, 4, 4))
            log.on_pair_end(status_counts={}, n_valid=0)
        end = next(
            e
            for e in _read_events(tmp_path / "events.jsonl")
            if e["kind"] == "dispatch_pair_end"
        )
        assert end["status_counts"] == {}
        assert end["n_valid"] == 0

    def test_composes_with_series_pair_logger_contextvars(self, tmp_path: Path) -> None:
        # Composition contract: when a SeriesPairLogger has bound
        # t_ref / t_def for a pair and the dispatcher fires its hooks
        # inside that window, both dispatch_pair_* rows must inherit
        # the temporal coordinates via merge_contextvars. This is the
        # whole point of the L1+L2 stacking — slice events.jsonl by
        # (t_ref, t_def) and group every row from the same pair.
        series_log = SeriesPairLogger()
        dispatch_log = DispatchLogger()
        field = _make_displacement_field(n_ok=4, n_total=4)
        with SessionScope(tmp_path, series="rat-103L"):
            series_log.on_pair_start(t_ref=2, t_def=3)
            dispatch_log.on_pair_start(volume_shape=(16, 16, 16))
            dispatch_log.on_pair_end(status_counts={POIStatus.OK: 4}, n_valid=4)
            series_log.on_pair_end(t_ref=2, t_def=3, status=SeriesPairStatus.OK, field=field)
        events = _read_events(tmp_path / "events.jsonl")
        kinds = [e["kind"] for e in events]
        assert kinds == [
            "pair_start",
            "dispatch_pair_start",
            "dispatch_pair_end",
            "pair_end",
        ]
        for ev in events:
            assert ev["t_ref"] == 2
            assert ev["t_def"] == 3
            assert ev["series"] == "rat-103L"
