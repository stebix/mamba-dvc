"""Tests for the time-series extensions to ``mamba_dvc.validate.synthetic``.

Covers :class:`TemporalDisplacementFunction` dispatch (curried vs.
pointwise), :func:`temporal_from_curried` and :func:`linear_motion`
convenience constructors, and :func:`make_series` frame synthesis +
input validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.validate.synthetic import (
    SyntheticSeries,
    linear_motion,
    make_series,
    make_texture,
    normalize_temporal_form,
    rigid_shift,
    temporal_from_curried,
)


class TestTemporalDispatch:
    def test_pointwise_form_returned_unchanged(self):
        def pointwise(coords, t):
            return rigid_shift((0.0, float(t), 0.0))(coords)

        normalized = normalize_temporal_form(pointwise)
        # The pointwise form is passed through.
        assert normalized is pointwise

    def test_curried_form_is_wrapped_into_pointwise(self):
        def curried(t):
            return rigid_shift((0.0, float(t), 0.0))

        normalized = normalize_temporal_form(curried)
        coords = np.zeros((5, 3), dtype=np.float32)
        out = normalized(coords, 3.0)
        np.testing.assert_allclose(out, np.tile([0.0, 3.0, 0.0], (5, 1)))

    def test_zero_arg_callable_rejected(self):
        with pytest.raises(ValueError, match="positional"):
            normalize_temporal_form(lambda: None)

    def test_three_arg_callable_rejected(self):
        with pytest.raises(ValueError, match="positional"):
            normalize_temporal_form(lambda a, b, c: a)

    def test_var_positional_rejected(self):
        with pytest.raises(ValueError, match=r"\*args"):
            normalize_temporal_form(lambda *args: args)


class TestTemporalFromCurried:
    def test_evaluates_at_arbitrary_t(self):
        u = temporal_from_curried(lambda t: rigid_shift((0.0, 2.0 * t, 0.0)))
        coords = np.zeros((4, 3), dtype=np.float32)
        np.testing.assert_allclose(u(coords, 1.5), np.tile([0.0, 3.0, 0.0], (4, 1)))


class TestLinearMotion:
    def test_scales_velocity_linearly_with_t(self):
        velocity = rigid_shift((0.0, 1.0, 0.5))
        u = linear_motion(velocity)
        coords = np.zeros((3, 3), dtype=np.float32)
        np.testing.assert_allclose(u(coords, 0.0), np.zeros((3, 3)), atol=1e-6)
        np.testing.assert_allclose(u(coords, 2.0), np.tile([0.0, 2.0, 1.0], (3, 1)))


class TestMakeSeriesShape:
    def test_returns_one_frame_per_timestep(self):
        shape = (48, 48, 48)
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        series = make_series(shape, u, timesteps=[0, 1, 2, 3])
        assert isinstance(series, SyntheticSeries)
        assert len(series.frames) == 4
        assert series.timesteps == (0, 1, 2, 3)
        for frame in series.frames:
            assert frame.shape == shape
            assert frame.dtype == np.float32

    def test_curried_form_accepted_via_dispatch(self):
        shape = (48, 48, 48)
        u_curried = lambda t: rigid_shift((0.0, float(t), 0.0))  # noqa: E731
        series = make_series(shape, u_curried, timesteps=[0, 1, 2])
        assert len(series.frames) == 3

    def test_returns_normalized_pointwise_field_in_field_at(self):
        shape = (48, 48, 48)
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        series = make_series(shape, u, timesteps=[0, 1, 2])
        coords = np.zeros((1, 3), dtype=np.float32)
        sampled = series.field_at(coords, 1.5)
        np.testing.assert_allclose(sampled, np.array([[0.0, 1.5, 0.0]]), atol=1e-6)


class TestMakeSeriesValidation:
    def test_empty_timesteps_rejected(self):
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        with pytest.raises(ValueError, match="non-empty"):
            make_series((48, 48, 48), u, timesteps=[])

    def test_non_monotonic_timesteps_rejected(self):
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        with pytest.raises(ValueError, match="strictly increasing"):
            make_series((48, 48, 48), u, timesteps=[0, 2, 1])

    def test_duplicate_timesteps_rejected(self):
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        with pytest.raises(ValueError, match="duplicates"):
            make_series((48, 48, 48), u, timesteps=[0, 1, 1])

    def test_mismatched_reference_shape_rejected(self):
        ref = make_texture((48, 48, 48))
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        with pytest.raises(ValueError, match="shape"):
            make_series((64, 64, 64), u, timesteps=[0, 1], reference=ref)


class TestMakeSeriesContent:
    def test_first_frame_matches_reference_for_zero_anchored_field(self):
        shape = (48, 48, 48)
        ref = make_texture(shape, seed=5)
        u = linear_motion(rigid_shift((0.0, 1.0, 0.0)))
        # u(., 0) == 0 → first frame equals reference up to prefilter noise.
        series = make_series(shape, u, timesteps=[0, 1, 2], reference=ref)
        diff = float(np.abs(series.frames[0] - ref).mean())
        # Cubic-spline prefilter at zero shift is not strictly identity but
        # the residual is tiny relative to the unit-variance texture.
        assert diff < 5e-3

    def test_higher_warp_order_does_not_change_shape(self):
        shape = (48, 48, 48)
        u = linear_motion(rigid_shift((0.0, 0.5, 0.0)))
        series_cubic = make_series(shape, u, timesteps=[0, 1], order=3)
        series_linear = make_series(shape, u, timesteps=[0, 1], order=1)
        assert series_cubic.frames[1].shape == series_linear.frames[1].shape
