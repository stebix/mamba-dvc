"""Smoke tests for ``mamba_dvc.viz.volume`` renderers.

Structural only — checks that an actor is registered with the plotter
and basic argument validation fires. No pixel diffing.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytestmark = pytest.mark.viz

from mamba_dvc.types import VoxelSpacing  # noqa: E402
from mamba_dvc.viz.backend import plotter  # noqa: E402
from mamba_dvc.viz.volume import render_isosurface, render_volume  # noqa: E402


def _vol(shape: tuple[int, int, int] = (8, 10, 12)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape, dtype=np.float32)


class TestRenderVolume:
    def test_adds_actor_off_screen(self):
        vol = _vol()
        with plotter(off_screen=True) as plot:
            assert len(plot.actors) == 0
            render_volume(plot, vol, spacing=VoxelSpacing((1.0, 1.0, 1.0), "um"))
            assert len(plot.actors) >= 1

    def test_returns_same_plotter(self):
        vol = _vol()
        with plotter(off_screen=True) as plot:
            out = render_volume(plot, vol)
            assert out is plot

    def test_mask_shape_mismatch_raises(self):
        vol = _vol((4, 4, 4))
        bad_mask = np.ones((4, 4, 5), dtype=bool)
        with (
            plotter(off_screen=True) as plot,
            pytest.raises(ValueError, match="does not match"),
        ):
            render_volume(plot, vol, mask=bad_mask)

    def test_mask_does_not_mutate_input(self):
        vol = _vol((4, 4, 4))
        snapshot = vol.copy()
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        with plotter(off_screen=True) as plot:
            render_volume(plot, vol, mask=mask)
        np.testing.assert_array_equal(vol, snapshot)


class TestRenderIsosurface:
    def test_adds_mesh_actor(self):
        vol = _vol()
        with plotter(off_screen=True) as plot:
            render_isosurface(plot, vol, level=0.0)
            assert len(plot.actors) >= 1
