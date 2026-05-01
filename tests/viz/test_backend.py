"""Smoke tests for ``mamba_dvc.viz.backend``."""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")
pytestmark = pytest.mark.viz

from mamba_dvc.viz.backend import plotter  # noqa: E402


class TestPlotterContext:
    def test_off_screen_yields_plotter(self):
        with plotter(off_screen=True) as plot:
            assert isinstance(plot, pv.Plotter)

    def test_qt_and_jupyter_mutually_exclusive(self):
        with (
            pytest.raises(ValueError, match="mutually exclusive"),
            plotter(jupyter=True, qt=True),
        ):
            pass

    def test_explicit_jupyter_false_off_screen(self):
        with plotter(jupyter=False, off_screen=True) as plot:
            assert plot.off_screen is True
