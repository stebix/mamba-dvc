"""Tests for the series additions to ``mamba_dvc.types``.

Covers :class:`PairingStrategy`, :class:`SeriesPairStatus`, the
:class:`DisplacementSeries` dataclass, and its ``cumulative()`` method.
``cumulative()`` is exercised against the
:func:`compose_displacement_fields` helper it lazily imports — failure
isolation, anchored short-circuit, and SEQUENTIAL chaining of known
integer shifts.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.core.grid import build_grid
from mamba_dvc.types import (
    DisplacementField,
    DisplacementSeries,
    GridSpec,
    PairingStrategy,
    POIStatus,
    SeriesPairStatus,
)


def _constant_field(grid: GridSpec, shift: tuple[float, float, float]) -> DisplacementField:
    """Build a :class:`DisplacementField` whose every POI carries the same shift."""
    n_points = int(np.prod(grid.grid_shape))
    shift_arr = np.asarray(shift, dtype=np.float32)
    displacements = np.broadcast_to(shift_arr, (n_points, 3)).astype(np.float32, copy=True)
    valid = np.ones(n_points, dtype=np.bool_)
    confidence = np.ones(n_points, dtype=np.float32)
    status = np.full(n_points, np.uint8(POIStatus.OK), dtype=np.uint8)
    return DisplacementField(
        positions=grid.positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=grid.grid_shape,
        spacing=grid.spacing,
        window=grid.window,
    )


class TestPairingStrategyEnum:
    def test_string_values_are_stable(self):
        assert PairingStrategy.REFERENCE_ANCHORED.value == "reference_anchored"
        assert PairingStrategy.SEQUENTIAL.value == "sequential"
        assert PairingStrategy.UPDATED_REFERENCE.value == "updated_reference"

    def test_str_enum_round_trip(self):
        assert PairingStrategy("sequential") is PairingStrategy.SEQUENTIAL


class TestSeriesPairStatusEnum:
    def test_int_values(self):
        assert int(SeriesPairStatus.OK) == 0
        assert int(SeriesPairStatus.FAILED) == 1


class TestDisplacementSeriesConstruction:
    def test_frozen_dataclass_rejects_assignment(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        series = DisplacementSeries(
            fields=(),
            pair_indices=np.empty((0, 2), dtype=np.int64),
            pair_status=np.empty(0, dtype=np.uint8),
            grid=grid,
            strategy=PairingStrategy.SEQUENTIAL,
        )
        with pytest.raises(AttributeError):
            series.strategy = PairingStrategy.REFERENCE_ANCHORED  # pyright: ignore[reportAttributeAccessIssue]


class TestDisplacementSeriesCumulativeAnchored:
    def test_anchored_returns_fields_as_tuple(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        fields = tuple(_constant_field(grid, (0.0, float(i + 1), 0.0)) for i in range(3))
        series = DisplacementSeries(
            fields=fields,
            pair_indices=np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int64),
            pair_status=np.zeros(3, dtype=np.uint8),
            grid=grid,
            strategy=PairingStrategy.REFERENCE_ANCHORED,
        )
        cumulative = series.cumulative()
        assert len(cumulative) == 3
        for original, returned in zip(fields, cumulative, strict=True):
            assert returned is original


class TestDisplacementSeriesCumulativeSequential:
    def test_sequential_chains_constant_shifts_linearly(self):
        # Cumulative composition pushes corner POIs off the lattice on each
        # step — that's expected behavior, not a bug. Assert on the valid
        # subset only; the failing-pair / boundary-gating cases live in
        # ``tests/core/test_field_ops.py``.
        grid = build_grid((96, 96, 96), window=32, overlap=0.5)
        shift = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        fields = tuple(_constant_field(grid, tuple(shift)) for _ in range(3))
        series = DisplacementSeries(
            fields=fields,
            pair_indices=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            pair_status=np.zeros(3, dtype=np.uint8),
            grid=grid,
            strategy=PairingStrategy.SEQUENTIAL,
        )
        cumulative = series.cumulative()
        assert len(cumulative) == 3
        expected_shifts = [shift, 2.0 * shift, 3.0 * shift]
        for step, (expected, cumfield) in enumerate(
            zip(expected_shifts, cumulative, strict=True)
        ):
            valid = cumfield.valid
            assert valid.any(), f"step {step} has no valid POIs"
            np.testing.assert_allclose(
                cumfield.displacements[valid],
                np.broadcast_to(expected, (int(valid.sum()), 3)),
                atol=1e-5,
                err_msg=f"step {step} cumulative displacement mismatch",
            )

    def test_sequential_truncates_at_first_failed_pair(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        ok_field = _constant_field(grid, (0.0, 1.0, 0.0))
        series = DisplacementSeries(
            fields=(ok_field, ok_field, ok_field),
            pair_indices=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            pair_status=np.asarray(
                [SeriesPairStatus.OK, SeriesPairStatus.FAILED, SeriesPairStatus.OK],
                dtype=np.uint8,
            ),
            grid=grid,
            strategy=PairingStrategy.SEQUENTIAL,
        )
        cumulative = series.cumulative()
        assert len(cumulative) == 1
        np.testing.assert_allclose(
            cumulative[0].displacements, ok_field.displacements, atol=1e-5
        )

    def test_sequential_empty_when_first_pair_failed(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        ok_field = _constant_field(grid, (0.0, 1.0, 0.0))
        series = DisplacementSeries(
            fields=(ok_field, ok_field),
            pair_indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
            pair_status=np.asarray(
                [SeriesPairStatus.FAILED, SeriesPairStatus.OK], dtype=np.uint8
            ),
            grid=grid,
            strategy=PairingStrategy.SEQUENTIAL,
        )
        assert series.cumulative() == ()


class TestDisplacementSeriesCumulativeUpdatedReference:
    def test_updated_reference_raises(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        series = DisplacementSeries(
            fields=(),
            pair_indices=np.empty((0, 2), dtype=np.int64),
            pair_status=np.empty(0, dtype=np.uint8),
            grid=grid,
            strategy=PairingStrategy.UPDATED_REFERENCE,
        )
        with pytest.raises(NotImplementedError, match="UPDATED_REFERENCE"):
            series.cumulative()
