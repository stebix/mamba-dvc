"""Visualization submodule (`mamba_dvc.viz`).

Pure consumer of :mod:`mamba_dvc.types` and :mod:`mamba_dvc.validate`.
Renderers return :class:`pyvista.Plotter` / :class:`matplotlib.figure.Figure`
objects so composition and headless export share the same code path.

The submodule is optional: install via the ``viz`` extra in
``pyproject.toml``. Nothing in :mod:`mamba_dvc.core`,
:mod:`mamba_dvc.pipeline`, :mod:`mamba_dvc.gpu`, or :mod:`mamba_dvc.io`
may import from here.
"""
