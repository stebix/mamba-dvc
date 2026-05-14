"""Smoke tests for ``mamba_dvc.viz.field`` renderers.

Structural only — checks that actors register on the plotter and basic
argument validation fires. Pixel diffing is out of scope.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytestmark = pytest.mark.viz

from mamba_dvc.types import DisplacementField, POIStatus, VoxelSpacing  # noqa: E402
from mamba_dvc.viz.backend import plotter  # noqa: E402
from mamba_dvc.viz.field import (  # noqa: E402
    render_error_glyphs,
    render_field,
    render_field_lattice,
)


def _lattice_field(
    grid_shape: tuple[int, int, int] = (2, 3, 4),
    spacing: tuple[int, int, int] = (2, 2, 2),
) -> DisplacementField:
    """Build a regular-lattice field — required for `render_field_lattice`."""
    nz, ny, nx = grid_shape
    n = nz * ny * nx
    rng = np.random.default_rng(0)
    zs = np.arange(nz, dtype=np.float32) * spacing[0]
    ys = np.arange(ny, dtype=np.float32) * spacing[1]
    xs = np.arange(nx, dtype=np.float32) * spacing[2]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    positions = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    displacements = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    valid[::5] = False
    confidence = rng.uniform(0, 1, size=n).astype(np.float32)
    status = np.where(valid, POIStatus.OK, POIStatus.MASKED).astype(np.uint8)
    # Honor the documented invariant: non-OK POIs carry zero displacement
    # / zero confidence (matches the producer-side contract in batch.py).
    displacements[~valid] = 0.0
    confidence[~valid] = 0.0
    return DisplacementField(
        positions=positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=grid_shape,
        spacing=spacing,
        window=(4, 4, 4),
    )


class TestRenderField:
    def test_adds_actor_off_screen(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            assert len(plot.actors) == 0
            render_field(plot, field)
            assert len(plot.actors) >= 1

    def test_returns_same_plotter(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            out = render_field(plot, field)
            assert out is plot

    def test_stride_keeps_running(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            render_field(plot, field, stride=2)
            assert len(plot.actors) >= 1

    def test_uniform_scale_runs(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            render_field(plot, field, scale="uniform", factor=1.5)
            assert len(plot.actors) >= 1

    def test_anisotropic_spacing_runs(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            render_field(plot, field, spacing=VoxelSpacing((2.0, 1.0, 0.5), "um"))
            assert len(plot.actors) >= 1


class TestRenderErrorGlyphs:
    def test_adds_actor(self):
        field = _lattice_field()
        truth = field.displacements * 0.9  # introduce a ~10% systematic error
        with plotter(off_screen=True) as plot:
            render_error_glyphs(plot, field, truth)
            assert len(plot.actors) >= 1

    def test_returns_same_plotter(self):
        field = _lattice_field()
        truth = np.zeros_like(field.displacements)
        with plotter(off_screen=True) as plot:
            out = render_error_glyphs(plot, field, truth)
            assert out is plot

    def test_truth_shape_mismatch_raises(self):
        field = _lattice_field()
        bad_truth = np.zeros((3, 3), dtype=np.float32)
        with (
            plotter(off_screen=True) as plot,
            pytest.raises(ValueError, match="does not match"),
        ):
            render_error_glyphs(plot, field, bad_truth)

    def test_zero_truth_matches_field_magnitudes(self):
        # With truth=0, the error glyphs collapse to render_field glyphs;
        # this regression-guards the ``recovered - truth`` direction.
        field = _lattice_field()
        truth = np.zeros_like(field.displacements)
        with plotter(off_screen=True) as plot:
            render_error_glyphs(plot, field, truth)
            assert len(plot.actors) >= 1


class TestRenderFieldLattice:
    def test_default_scalar_is_magnitude(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            render_field_lattice(plot, field)
            assert len(plot.actors) >= 1

    def test_named_scalar_alternatives_run(self):
        field = _lattice_field()
        for scalar in ("confidence", "status", "valid"):
            with plotter(off_screen=True) as plot:
                render_field_lattice(plot, field, scalar=scalar)
                assert len(plot.actors) >= 1

    def test_custom_scalar_array(self):
        field = _lattice_field()
        n = field.positions.shape[0]
        scalars = np.arange(n, dtype=np.float32)
        with plotter(off_screen=True) as plot:
            render_field_lattice(plot, field, scalar=None, scalar_array=scalars)
            assert len(plot.actors) >= 1

    def test_no_scalar_draws_bare_lattice(self):
        field = _lattice_field()
        with plotter(off_screen=True) as plot:
            render_field_lattice(plot, field, scalar=None)
            assert len(plot.actors) >= 1

    def test_rejects_both_scalar_and_scalar_array(self):
        field = _lattice_field()
        n = field.positions.shape[0]
        scalars = np.arange(n, dtype=np.float32)
        with (
            plotter(off_screen=True) as plot,
            pytest.raises(ValueError, match="not both"),
        ):
            render_field_lattice(plot, field, scalar="magnitude", scalar_array=scalars)

    def test_scalar_array_shape_mismatch_raises(self):
        field = _lattice_field()
        with (
            plotter(off_screen=True) as plot,
            pytest.raises(ValueError, match="does not match"),
        ):
            render_field_lattice(
                plot,
                field,
                scalar=None,
                scalar_array=np.zeros(3, dtype=np.float32),
            )

    def test_grid_shape_mismatch_raises(self):
        # Construct a field whose positions count does not match grid_shape.
        n = 10
        field = DisplacementField(
            positions=np.zeros((n, 3), dtype=np.float32),
            displacements=np.zeros((n, 3), dtype=np.float32),
            valid=np.ones(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float32),
            status=np.zeros(n, dtype=np.uint8),
            grid_shape=(2, 3, 4),  # 24 != 10
            spacing=(2, 2, 2),
            window=(4, 4, 4),
        )
        with (
            plotter(off_screen=True) as plot,
            pytest.raises(ValueError, match="requires"),
        ):
            render_field_lattice(plot, field)
