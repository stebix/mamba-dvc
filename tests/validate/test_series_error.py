"""Tests for ``mamba_dvc.validate.series_error.evaluate_synthetic``.

These are end-to-end behavioral tests for the rapid-iteration eval
interface: a reference image + a temporal field function go in,
(strategy, lag) error tables come out. The synthetic motion is a small
rigid shift per unit time so the underlying NCC operates near its
sweet spot and per-pair errors collapse to a small floor that the
tests can assert against.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.types import PairingStrategy
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
