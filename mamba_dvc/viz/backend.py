"""Plotter context manager and screenshot helpers.

The only module in :mod:`mamba_dvc.viz` allowed to know which backend
is in use. Renderers (:mod:`volume`, :mod:`field`, ...) accept a
generic :class:`pyvista.Plotter` produced here.

Default behavior — Jupyter first
--------------------------------
``plotter()`` returns a Plotter configured for the ``trame`` Jupyter
backend. Set ``qt=True`` for the native desktop window (escape hatch
when notebook interaction stutters on full-resolution volumes), or
``off_screen=True`` for headless figure export and image-diff tests.

The context manager closes the Plotter on exit so notebook cells do
not accumulate VTK render windows.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyvista as pv


__all__ = ["plotter", "screenshot"]


@contextmanager
def plotter(
    *,
    jupyter: bool | None = None,
    qt: bool = False,
    off_screen: bool = False,
    window_size: tuple[int, int] = (1024, 768),
    shape: tuple[int, int] = (1, 1),
) -> Iterator[pv.Plotter]:
    """Yield a :class:`pyvista.Plotter` with backend pre-selected.

    Parameters
    ----------
    jupyter
        Force the trame Jupyter backend (``True``) or disable it
        (``False``). Default ``None`` autodetects: trame inside a
        notebook, native otherwise. Ignored when ``off_screen`` is set.
    qt
        Force the native Qt window. Mutually exclusive with
        ``jupyter=True``; raises :class:`ValueError` if both are set.
    off_screen
        Render headless (no window). Used for figure export and
        smoke / image-diff tests; takes precedence over ``jupyter``
        and ``qt``.
    window_size
        Initial render-window size, ``(width, height)`` in pixels.
    shape
        Subplot grid as ``(rows, cols)``. Defaults to ``(1, 1)``;
        use e.g. ``(1, 2)`` for side-by-side views and switch with
        ``plotter.subplot(row, col)``.

    Yields
    ------
    pyvista.Plotter
        Fresh Plotter; closed on context exit.

    Raises
    ------
    ValueError
        If ``jupyter=True`` and ``qt=True`` are both requested.
    """
    import pyvista as pv

    if jupyter is True and qt:
        raise ValueError("jupyter=True and qt=True are mutually exclusive")

    if off_screen or qt or jupyter is False:
        backend: str | None = None
    else:
        backend = "trame"

    plot = pv.Plotter(off_screen=off_screen, window_size=window_size, shape=shape)
    if backend is not None:
        # ``notebook`` flag tells PyVista to render via the configured
        # Jupyter backend on ``.show()``.
        plot.notebook = True
    try:
        yield plot
    finally:
        plot.close()


def screenshot(
    plot: pv.Plotter,
    path: Path | str,
    *,
    size: tuple[int, int] = (1920, 1080),
    transparent_background: bool = False,
) -> Path:
    """Render ``plot`` headless and save a PNG to ``path``.

    Parameters
    ----------
    plot
        Plotter to capture; works for both interactive and
        ``off_screen=True`` plotters. The window is resized for the
        screenshot but restored afterwards.
    path
        Destination PNG path. Parent directory must exist.
    size
        Output image size, ``(width, height)`` in pixels.
    transparent_background
        Pass through to PyVista's ``transparent_background`` flag.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """
    out = Path(path)
    plot.window_size = list(size)
    plot.screenshot(
        filename=str(out),
        transparent_background=transparent_background,
    )
    return out
