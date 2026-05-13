"""Tests for ``mamba_dvc.pipeline.series.correlate_series``.

The driver itself is pure plumbing on top of :func:`correlate`. These
tests exercise the strategy → pair-provider mapping, the per-pair
failure isolation contract, the ``on_pair`` side-channel, and the
input-validation branches. Each fixture uses a small textured volume
plus a constant inter-frame shift so the underlying correlator
returns near-exact results — the assertions focus on driver behavior
rather than NCC accuracy (which lives in
``tests/pipeline/test_correlate.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.pipeline.series import correlate_series
from mamba_dvc.types import (
    DisplacementField,
    DisplacementSeries,
    PairingStrategy,
    SeriesPairStatus,
)
from mamba_dvc.validate.synthetic import make_texture, rigid_shift, warp


def _make_frames(
    shape: tuple[int, int, int],
    shifts_per_frame: list[tuple[float, float, float]],
    *,
    seed: int = 7,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(reference, frames)`` warped by cumulative ``shifts_per_frame``."""
    ref = make_texture(shape, sigma=1.5, seed=seed)
    frames = []
    cumulative = np.zeros(3, dtype=np.float64)
    for delta in shifts_per_frame:
        cumulative += np.asarray(delta, dtype=np.float64)
        frames.append(warp(ref, rigid_shift(tuple(cumulative))))
    return ref, frames


def _correlator_kwargs() -> dict[str, object]:
    """Shared kwargs that keep tests fast on CPU-only hosts."""
    return {
        "window": 24,
        "overlap": 0.5,
        "search_radius": 6,
        "batch_size": 32,
    }


class TestStrategyPairProviders:
    def test_reference_anchored_pairs_first_with_each_subsequent(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        iterator = enumerate([ref, *frames])

        series = correlate_series(
            iterator,
            strategy=PairingStrategy.REFERENCE_ANCHORED,
            **_correlator_kwargs(),
        )
        np.testing.assert_array_equal(
            series.pair_indices, np.array([[0, 1], [0, 2]], dtype=np.int64)
        )

    def test_sequential_pairs_each_with_previous(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        iterator = enumerate([ref, *frames])

        series = correlate_series(
            iterator,
            strategy=PairingStrategy.SEQUENTIAL,
            **_correlator_kwargs(),
        )
        np.testing.assert_array_equal(
            series.pair_indices, np.array([[0, 1], [1, 2]], dtype=np.int64)
        )


class TestDriverReturnsDisplacementSeries:
    def test_three_frame_sequential_run_recovers_constant_shift(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        iterator = enumerate([ref, *frames])

        series = correlate_series(
            iterator, strategy=PairingStrategy.SEQUENTIAL, **_correlator_kwargs()
        )

        assert isinstance(series, DisplacementSeries)
        assert len(series.fields) == 3
        assert series.strategy is PairingStrategy.SEQUENTIAL
        for i, field in enumerate(series.fields):
            assert int(series.pair_status[i]) == int(SeriesPairStatus.OK)
            mean_displacement = field.displacements[field.valid].mean(axis=0)
            np.testing.assert_allclose(mean_displacement, np.array([0.0, 1.0, 0.0]), atol=0.1)


class TestOnPairCallback:
    def test_on_pair_fires_once_per_pair_in_order(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        iterator = enumerate([ref, *frames])

        seen: list[tuple[int, type]] = []

        def callback(t: int, field: DisplacementField) -> None:
            seen.append((t, type(field)))

        correlate_series(
            iterator,
            strategy=PairingStrategy.SEQUENTIAL,
            on_pair=callback,
            **_correlator_kwargs(),
        )

        assert seen == [(1, DisplacementField), (2, DisplacementField)]


class TestFailureIsolation:
    def test_per_pair_failure_marks_status_and_keeps_going(self, monkeypatch):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        iterator = enumerate([ref, *frames])

        # Original correlate; we wrap it to raise on the second call only.
        import mamba_dvc.pipeline.series as series_module

        original = series_module.correlate
        call_count = {"n": 0}

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("synthetic CUDA OOM")
            return original(*args, **kwargs)

        monkeypatch.setattr(series_module, "correlate", flaky)

        with pytest.warns(RuntimeWarning, match="failed"):
            series = correlate_series(
                iterator, strategy=PairingStrategy.SEQUENTIAL, **_correlator_kwargs()
            )

        assert len(series.fields) == 3
        assert int(series.pair_status[0]) == int(SeriesPairStatus.OK)
        assert int(series.pair_status[1]) == int(SeriesPairStatus.FAILED)
        assert int(series.pair_status[2]) == int(SeriesPairStatus.OK)
        # The FAILED slot carries a zero-filled field with every POI MASKED.
        assert not np.any(series.fields[1].valid)
        np.testing.assert_array_equal(
            series.fields[1].displacements, np.zeros_like(series.fields[1].displacements)
        )


class TestInputValidation:
    def test_empty_iterator_raises(self):
        with pytest.raises(ValueError, match="empty"):
            correlate_series(iter([]), strategy=PairingStrategy.SEQUENTIAL)

    def test_updated_reference_raises_not_implemented(self):
        shape = (48, 48, 48)
        ref = make_texture(shape)
        with pytest.raises(NotImplementedError, match="UPDATED_REFERENCE"):
            correlate_series(
                iter([(0, ref)]),
                strategy=PairingStrategy.UPDATED_REFERENCE,
            )

    def test_non_float32_first_frame_rejected(self):
        ref_f64 = make_texture((48, 48, 48)).astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            correlate_series(
                iter([(0, ref_f64)]),  # type: ignore[arg-type]
                strategy=PairingStrategy.SEQUENTIAL,
            )

    def test_single_frame_iterator_yields_empty_series(self):
        ref = make_texture((48, 48, 48))
        series = correlate_series(
            iter([(0, ref)]), strategy=PairingStrategy.SEQUENTIAL, **_correlator_kwargs()
        )
        assert len(series.fields) == 0
        assert series.pair_indices.shape == (0, 2)
        assert series.pair_status.shape == (0,)
