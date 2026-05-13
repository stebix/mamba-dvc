"""Tests for ``mamba_dvc.core.field_ops``.

The composition primitive is the load-bearing helper behind
:meth:`DisplacementSeries.cumulative` for the SEQUENTIAL pairing
strategy. Tests cover identity behavior under zero increments, exact
linear chaining of constant shifts, validity propagation through
invalid POIs, lattice-extent gating for queries that fall off the
grid, and shape-mismatch rejection at the API boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.core.field_ops import compose_displacement_fields, sample_field_at
from mamba_dvc.core.grid import build_grid
from mamba_dvc.types import DisplacementField, GridSpec, POIStatus


def _make_field(
    grid: GridSpec,
    displacements: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    confidence: float = 1.0,
) -> DisplacementField:
    n_points = int(np.prod(grid.grid_shape))
    if valid is None:
        valid = np.ones(n_points, dtype=np.bool_)
    status = np.where(valid, np.uint8(POIStatus.OK), np.uint8(POIStatus.MASKED)).astype(
        np.uint8
    )
    conf_arr = np.where(valid, np.float32(confidence), np.float32(0.0)).astype(np.float32)
    return DisplacementField(
        positions=grid.positions,
        displacements=displacements.astype(np.float32, copy=False),
        valid=valid,
        confidence=conf_arr,
        status=status,
        grid_shape=grid.grid_shape,
        spacing=grid.spacing,
        window=grid.window,
    )


def _zero_displacements(grid: GridSpec) -> np.ndarray:
    n_points = int(np.prod(grid.grid_shape))
    return np.zeros((n_points, 3), dtype=np.float32)


def _constant_displacements(grid: GridSpec, shift: tuple[float, float, float]) -> np.ndarray:
    n_points = int(np.prod(grid.grid_shape))
    return np.broadcast_to(np.asarray(shift, dtype=np.float32), (n_points, 3)).astype(
        np.float32, copy=True
    )


class TestComposeIdentity:
    def test_zero_increment_returns_total(self):
        # A non-zero u_total pushes query points off the lattice at the corner
        # POIs (semi-Lagrangian sampling has nowhere to look). Assert on the
        # valid subset — out-of-bounds gating is the documented behavior and
        # is exercised separately in TestComposeBoundsGating.
        grid = build_grid((96, 96, 96), window=32, overlap=0.5)
        total = _make_field(grid, _constant_displacements(grid, (0.0, 0.7, 0.3)))
        increment = _make_field(grid, _zero_displacements(grid))

        result = compose_displacement_fields(total, increment)
        valid = result.valid
        assert valid.any()
        np.testing.assert_allclose(
            result.displacements[valid], total.displacements[valid], atol=1e-6
        )

    def test_zero_total_returns_increment(self):
        # u_total = 0 makes the query points equal the lattice positions, so
        # every POI lands exactly on the source samples and stays valid.
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        total = _make_field(grid, _zero_displacements(grid))
        increment = _make_field(grid, _constant_displacements(grid, (0.4, 0.0, -0.2)))

        result = compose_displacement_fields(total, increment)
        assert result.valid.all()
        np.testing.assert_allclose(result.displacements, increment.displacements, atol=1e-6)


class TestComposeKnownShifts:
    def test_constant_shifts_compose_linearly(self):
        grid = build_grid((96, 96, 96), window=32, overlap=0.5)
        shift_a = (0.0, 1.0, 0.0)
        shift_b = (0.0, 0.0, 1.0)
        total = _make_field(grid, _constant_displacements(grid, shift_a))
        increment = _make_field(grid, _constant_displacements(grid, shift_b))

        result = compose_displacement_fields(total, increment)
        expected = _constant_displacements(
            grid, (shift_a[0] + shift_b[0], shift_a[1] + shift_b[1], shift_a[2] + shift_b[2])
        )
        valid = result.valid
        assert valid.any()
        np.testing.assert_allclose(result.displacements[valid], expected[valid], atol=1e-5)

    def test_cubic_interpolation_runs_and_matches_linear_on_constant_field(self):
        grid = build_grid((96, 96, 96), window=32, overlap=0.5)
        total = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.0)))
        increment = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.0)))

        linear = compose_displacement_fields(total, increment, interpolation="linear")
        cubic = compose_displacement_fields(total, increment, interpolation="cubic")
        common = linear.valid & cubic.valid
        assert common.any()
        np.testing.assert_allclose(
            linear.displacements[common], cubic.displacements[common], atol=1e-4
        )


class TestComposeValidityPropagation:
    def test_invalid_in_total_propagates_to_masked(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        valid_total = np.ones(n_points, dtype=np.bool_)
        valid_total[5] = False
        total = _make_field(
            grid, _constant_displacements(grid, (0.0, 0.5, 0.0)), valid=valid_total
        )
        increment = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.0)))

        result = compose_displacement_fields(total, increment)
        assert not result.valid[5]
        np.testing.assert_array_equal(result.displacements[5], np.zeros(3, dtype=np.float32))

    def test_invalid_in_increment_propagates_to_masked(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        total = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.0)))
        valid_inc = np.ones(n_points, dtype=np.bool_)
        valid_inc[3] = False
        increment = _make_field(
            grid, _constant_displacements(grid, (0.0, 0.5, 0.0)), valid=valid_inc
        )

        result = compose_displacement_fields(total, increment)
        assert not result.valid[3]


class TestComposeBoundsGating:
    def test_query_outside_lattice_extent_marked_masked(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        # An aggressive uniform shift that pushes every POI off the lattice extent
        # along the y-axis. The lattice spans roughly [16, 48] in y for a 64-cube
        # with window=32 — a shift of 200 is well outside.
        total = _make_field(grid, _constant_displacements(grid, (0.0, 200.0, 0.0)))
        increment = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.0)))

        result = compose_displacement_fields(total, increment)
        assert not np.any(result.valid)


class TestComposeShapeValidation:
    def test_grid_shape_mismatch_raises(self):
        # Both volume_shapes produce different lattice POI counts (3³ vs 5³)
        # so the grid_shape branch is the first thing the validator hits.
        grid_a = build_grid((48, 48, 48), window=24, overlap=0.5)
        grid_b = build_grid((96, 96, 96), window=32, overlap=0.5)
        total = _make_field(grid_a, _zero_displacements(grid_a))
        increment = _make_field(grid_b, _zero_displacements(grid_b))
        with pytest.raises(ValueError, match="grid_shape mismatch"):
            compose_displacement_fields(total, increment)

    def test_unknown_interpolation_rejected(self):
        grid = build_grid((48, 48, 48), window=24, overlap=0.5)
        total = _make_field(grid, _zero_displacements(grid))
        increment = _make_field(grid, _zero_displacements(grid))
        with pytest.raises(ValueError, match="interpolation"):
            compose_displacement_fields(total, increment, interpolation="quartic")  # type: ignore[arg-type]


class TestSampleFieldAt:
    def test_sampling_at_lattice_points_recovers_field(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        field = _make_field(grid, _constant_displacements(grid, (0.0, 0.5, 0.25)))
        sampled = sample_field_at(field, grid.positions)
        np.testing.assert_allclose(sampled, field.displacements, atol=1e-5)

    def test_out_of_bounds_uses_fill_value(self):
        grid = build_grid((64, 64, 64), window=32, overlap=0.5)
        field = _make_field(grid, _constant_displacements(grid, (0.0, 1.0, 0.0)))
        coords = np.asarray([[-100.0, -100.0, -100.0]], dtype=np.float32)
        sampled = sample_field_at(field, coords, fill_value=42.0)
        np.testing.assert_allclose(sampled, 42.0)
