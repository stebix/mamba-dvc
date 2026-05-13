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


class _StubDispatcher:
    """In-Python stand-in for :class:`MultiGPUDispatcher`.

    Records every ``(reference, deformed)`` call so tests can assert
    the driver chose the right pairing semantics. Returns a trivial
    zero-displacement field with all POIs ``OK`` -- the assertions are
    on the call pattern, not the per-pair accuracy.
    """

    def __init__(self, grid_shape: tuple[int, int, int], *, has_anchored: bool) -> None:
        from mamba_dvc.core.grid import build_grid

        self._grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        self.has_anchored_reference = has_anchored
        self.calls: list[tuple[np.ndarray | None, np.ndarray]] = []

    def correlate(
        self,
        reference: np.ndarray | None,
        deformed: np.ndarray,
    ) -> DisplacementField:
        self.calls.append((reference, deformed))
        n_points = int(np.prod(self._grid.grid_shape))
        return DisplacementField(
            positions=self._grid.positions,
            displacements=np.zeros((n_points, 3), dtype=np.float32),
            valid=np.ones(n_points, dtype=np.bool_),
            confidence=np.ones(n_points, dtype=np.float32),
            status=np.zeros(n_points, dtype=np.uint8),
            grid_shape=self._grid.grid_shape,
            spacing=self._grid.spacing,
            window=self._grid.window,
        )


class TestDispatcherBranch:
    """The dispatcher kwarg routes every pair through ``dispatcher.correlate``.

    Uses an in-Python stub so this can run on CPU-only hosts. The real
    multi-GPU plumbing is exercised by ``tests/gpu/test_dispatch.py``.
    """

    def test_sequential_passes_prev_and_next_explicitly(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        stub = _StubDispatcher(shape, has_anchored=False)

        correlate_series(
            enumerate([ref, *frames]),
            strategy=PairingStrategy.SEQUENTIAL,
            dispatcher=stub,  # type: ignore[arg-type]
            **_correlator_kwargs(),
        )

        # SEQUENTIAL with no anchored reference: every call carries
        # the previous frame as the explicit reference.
        assert len(stub.calls) == 2
        for prev, _deformed in stub.calls:
            assert prev is not None
        # Pair 1: (ref, frames[0]); pair 2: (frames[0], frames[1]).
        np.testing.assert_array_equal(stub.calls[0][0], ref)
        np.testing.assert_array_equal(stub.calls[0][1], frames[0])
        np.testing.assert_array_equal(stub.calls[1][0], frames[0])
        np.testing.assert_array_equal(stub.calls[1][1], frames[1])

    def test_anchored_with_resident_reference_passes_none(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        stub = _StubDispatcher(shape, has_anchored=True)

        correlate_series(
            enumerate([ref, *frames]),
            strategy=PairingStrategy.REFERENCE_ANCHORED,
            dispatcher=stub,  # type: ignore[arg-type]
            **_correlator_kwargs(),
        )

        # REFERENCE_ANCHORED with dispatcher.has_anchored_reference=True:
        # the driver skips re-uploading the reference -- every call
        # receives reference=None.
        assert len(stub.calls) == 2
        for prev, _deformed in stub.calls:
            assert prev is None
        np.testing.assert_array_equal(stub.calls[0][1], frames[0])
        np.testing.assert_array_equal(stub.calls[1][1], frames[1])

    def test_anchored_without_resident_reference_passes_seed(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        stub = _StubDispatcher(shape, has_anchored=False)

        correlate_series(
            enumerate([ref, *frames]),
            strategy=PairingStrategy.REFERENCE_ANCHORED,
            dispatcher=stub,  # type: ignore[arg-type]
            **_correlator_kwargs(),
        )

        # REFERENCE_ANCHORED without resident anchor: the driver passes
        # the seed frame explicitly on every call.
        assert len(stub.calls) == 2
        for prev, _deformed in stub.calls:
            np.testing.assert_array_equal(prev, ref)

    def test_dispatcher_failure_records_failed_status(self):
        shape = (48, 48, 48)
        ref, frames = _make_frames(shape, [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])

        class _FlakyDispatcher(_StubDispatcher):
            def correlate(self, reference, deformed):  # type: ignore[override]
                self.calls.append((reference, deformed))
                if len(self.calls) == 1:
                    raise RuntimeError("synthetic dispatcher failure")
                return super().correlate(reference, deformed)

        stub = _FlakyDispatcher(shape, has_anchored=False)

        with pytest.warns(RuntimeWarning, match="failed"):
            series = correlate_series(
                enumerate([ref, *frames]),
                strategy=PairingStrategy.SEQUENTIAL,
                dispatcher=stub,  # type: ignore[arg-type]
                **_correlator_kwargs(),
            )

        assert int(series.pair_status[0]) == int(SeriesPairStatus.FAILED)
        assert int(series.pair_status[1]) == int(SeriesPairStatus.OK)


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
