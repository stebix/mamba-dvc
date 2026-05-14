"""Tests for ``mamba_dvc.validate.series_error.evaluate_synthetic``.

These are end-to-end behavioral tests for the rapid-iteration eval
interface: a reference image + a temporal field function go in,
(strategy, lag) error tables come out. The synthetic motion is a small
rigid shift per unit time so the underlying NCC operates near its
sweet spot and per-pair errors collapse to a small floor that the
tests can assert against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mamba_dvc.run.eventlog import SeriesPairLogger, SessionScope
from mamba_dvc.types import PairingStrategy, SeriesPairStatus
from mamba_dvc.validate.series_error import (
    CumulativeDriftTable,
    PerPairErrorTable,
    SyntheticEvalReport,
    evaluate_synthetic,
)
from mamba_dvc.validate.synthetic import linear_motion, make_texture, rigid_shift

# Per-frame motion in voxels. Small enough that DVC stays well inside the
# search radius across a few-frame series and large enough that recovered
# displacements are unambiguously non-zero.
_VELOCITY = (0.0, 1.0, 0.0)


@pytest.fixture
def small_reference() -> np.ndarray:
    return make_texture((48, 48, 48), sigma=1.5, seed=11)


def _correlator_kwargs() -> dict[str, object]:
    return {
        "window": 24,
        "overlap": 0.5,
        "search_radius": 8,
        "batch_size": 32,
    }


class TestReportStructure:
    def test_default_strategies_produce_per_pair_tables_for_each_lag(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference,
            u,
            timesteps=[0, 1, 2, 3],
            lags=(1, 2),
            **_correlator_kwargs(),
        )

        assert isinstance(report, SyntheticEvalReport)
        assert report.timesteps == (0, 1, 2, 3)
        assert report.lags == (1, 2)
        for lag in (1, 2):
            for strategy in (
                PairingStrategy.REFERENCE_ANCHORED,
                PairingStrategy.SEQUENTIAL,
            ):
                assert (strategy, lag) in report.per_pair
                assert isinstance(report.per_pair[(strategy, lag)], PerPairErrorTable)

    def test_cumulative_table_only_for_sequential(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference, u, timesteps=[0, 1, 2, 3], **_correlator_kwargs()
        )
        assert (PairingStrategy.SEQUENTIAL, 1) in report.cumulative
        assert (PairingStrategy.REFERENCE_ANCHORED, 1) not in report.cumulative
        assert isinstance(
            report.cumulative[(PairingStrategy.SEQUENTIAL, 1)], CumulativeDriftTable
        )


class TestPerPairAccuracy:
    def test_constant_shift_recovered_with_small_per_pair_error(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference, u, timesteps=[0, 1, 2, 3], **_correlator_kwargs()
        )

        for strategy in (
            PairingStrategy.REFERENCE_ANCHORED,
            PairingStrategy.SEQUENTIAL,
        ):
            table = report.per_pair[(strategy, 1)]
            valid_mae = table.mae[~np.isnan(table.mae)]
            assert valid_mae.size > 0
            # The exact MAE depends on subvoxel-fit accuracy, but for a constant
            # integer shift on band-limited texture it should comfortably land
            # well below half a voxel.
            assert float(valid_mae.max()) < 0.3


class TestPairIndexBookkeeping:
    def test_sequential_lag_one_pairs_are_consecutive(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference, u, timesteps=[0, 1, 2, 3], **_correlator_kwargs()
        )
        table = report.per_pair[(PairingStrategy.SEQUENTIAL, 1)]
        np.testing.assert_array_equal(table.t_ref, np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(table.t_def, np.array([1, 2, 3], dtype=np.int64))

    def test_anchored_lag_two_pairs_anchor_to_first_at_double_stride(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference,
            u,
            timesteps=[0, 1, 2, 3, 4],
            lags=(2,),
            strategies=(PairingStrategy.REFERENCE_ANCHORED,),
            **_correlator_kwargs(),
        )
        table = report.per_pair[(PairingStrategy.REFERENCE_ANCHORED, 2)]
        np.testing.assert_array_equal(table.t_ref, np.array([0, 0], dtype=np.int64))
        np.testing.assert_array_equal(table.t_def, np.array([2, 4], dtype=np.int64))


class TestCumulativeDrift:
    def test_cumulative_t_axis_matches_pair_t_def(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference, u, timesteps=[0, 1, 2, 3], **_correlator_kwargs()
        )
        per_pair = report.per_pair[(PairingStrategy.SEQUENTIAL, 1)]
        cumulative = report.cumulative[(PairingStrategy.SEQUENTIAL, 1)]
        # On a clean run with no failures, cumulative covers every pair.
        np.testing.assert_array_equal(cumulative.t, per_pair.t_def)

    def test_drift_stays_small_for_constant_shift(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        report = evaluate_synthetic(
            small_reference, u, timesteps=[0, 1, 2, 3], **_correlator_kwargs()
        )
        cumulative = report.cumulative[(PairingStrategy.SEQUENTIAL, 1)]
        valid_mae = cumulative.mae[~np.isnan(cumulative.mae)]
        assert valid_mae.size > 0
        # Per-step error of < 0.3 vx implies cumulative drift bounded by
        # n_step * per_step. With 3 steps that is well under 1 vx.
        assert float(valid_mae.max()) < 1.0


class TestInputValidation:
    def test_updated_reference_strategy_rejected(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        with pytest.raises(ValueError, match="UPDATED_REFERENCE"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1, 2],
                strategies=(PairingStrategy.UPDATED_REFERENCE,),
                **_correlator_kwargs(),
            )

    def test_non_positive_lag_rejected(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        with pytest.raises(ValueError, match="positive"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1, 2],
                lags=(0,),
                **_correlator_kwargs(),
            )

    def test_lag_yielding_one_frame_warns_and_skips(self, small_reference):
        u = linear_motion(rigid_shift(_VELOCITY))
        with pytest.warns(RuntimeWarning, match="no pairs"):
            report = evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1],
                lags=(5,),
                **_correlator_kwargs(),
            )
        assert (PairingStrategy.SEQUENTIAL, 5) not in report.per_pair


def _read_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestEventLogging:
    """L4 wiring: pair_observer / dispatch_observer thread through the sweep."""

    def test_pair_observer_lands_pair_events_with_strategy_and_lag(
        self, tmp_path: Path, small_reference: np.ndarray
    ):
        # The whole point of L4: open a SessionScope, pass a
        # SeriesPairLogger to evaluate_synthetic, and the resulting
        # events.jsonl carries a pair_start / pair_end pair per
        # (strategy, lag, t_ref, t_def). The strategy / lag
        # contextvars bound by the harness must appear on every row
        # so a downstream reader can groupby them.
        u = linear_motion(rigid_shift(_VELOCITY))
        with SessionScope(tmp_path, series="syn-test"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1, 2],
                lags=(1,),
                strategies=(
                    PairingStrategy.REFERENCE_ANCHORED,
                    PairingStrategy.SEQUENTIAL,
                ),
                pair_observer=SeriesPairLogger(),
                **_correlator_kwargs(),
            )
        events = _read_events(tmp_path / "events.jsonl")
        pair_starts = [e for e in events if e["kind"] == "pair_start"]
        pair_ends = [e for e in events if e["kind"] == "pair_end"]
        # 2 strategies x 2 pairs each (timesteps [0,1,2] gives 2 pairs).
        assert len(pair_starts) == 4
        assert len(pair_ends) == 4
        for ev in pair_starts + pair_ends:
            assert ev["strategy"] in {"REFERENCE_ANCHORED", "SEQUENTIAL"}
            assert ev["lag"] == 1
            assert ev["series"] == "syn-test"
            assert "session_id" in ev
            assert "t_ref" in ev
            assert "t_def" in ev
        # Every pair_end must report status; sweep is on a clean
        # synthetic so all should be OK.
        for end in pair_ends:
            assert end["status"] == SeriesPairStatus.OK.name

    def test_strategy_lag_unbound_after_iteration(
        self, tmp_path: Path, small_reference: np.ndarray
    ):
        # Loop-bound contextvars must not leak past evaluate_synthetic;
        # otherwise a later event emitted in the same SessionScope
        # would wrongly inherit the last (strategy, lag) values.
        u = linear_motion(rigid_shift(_VELOCITY))
        with SessionScope(tmp_path, series="syn-test"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1],
                pair_observer=SeriesPairLogger(),
                **_correlator_kwargs(),
            )
            # Emit a custom event after the call. It must not carry
            # strategy / lag — the harness unbound them on the way out.
            import structlog

            structlog.get_logger("mamba_dvc.eventlog").info("post_sweep_marker")
        marker = next(
            e
            for e in _read_events(tmp_path / "events.jsonl")
            if e["kind"] == "post_sweep_marker"
        )
        assert "strategy" not in marker
        assert "lag" not in marker

    def test_pair_observer_none_emits_no_events(
        self, tmp_path: Path, small_reference: np.ndarray
    ):
        # Backward-compat: omitting pair_observer (the pre-L4 call
        # shape) leaves events.jsonl empty even inside a SessionScope.
        u = linear_motion(rigid_shift(_VELOCITY))
        with SessionScope(tmp_path, series="syn-test"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1, 2],
                **_correlator_kwargs(),
            )
        events = _read_events(tmp_path / "events.jsonl")
        # No pair_observer => no pair_start / pair_end rows. Phase
        # records (kind:"phase") may still appear because the timing
        # logger is at DEBUG inside the SessionScope, but the
        # observer-driven kinds must be absent.
        assert not any(e["kind"] == "pair_start" for e in events)
        assert not any(e["kind"] == "pair_end" for e in events)

    def test_dispatch_observer_without_devices_warns(self, small_reference: np.ndarray):
        # The host-only path (device_ids=None) does not open a
        # MultiGPUDispatcher, so a supplied dispatch_observer would
        # silently never fire. The harness raises a RuntimeWarning to
        # surface the misconfiguration instead.
        from mamba_dvc.run.eventlog import DispatchLogger

        u = linear_motion(rigid_shift(_VELOCITY))
        with pytest.warns(RuntimeWarning, match="dispatch_observer"):
            evaluate_synthetic(
                small_reference,
                u,
                timesteps=[0, 1, 2],
                dispatch_observer=DispatchLogger(),
                **_correlator_kwargs(),
            )
