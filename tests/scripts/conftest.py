"""Shared loading for ``tests/scripts``.

The ``scripts/`` directory is not a package -- it ships standalone
entry points -- so test code cannot just ``import run_e2e_zarr``.
This conftest loads the module once at collection time via
:mod:`importlib.util` and exposes it to tests under the
``run_e2e_zarr`` fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> ModuleType:
    """Load a top-level script as an unimportable module by file path.

    The module is registered in ``sys.modules`` before execution so any
    ``@dataclass`` declarations inside resolve correctly (the dataclass
    machinery introspects ``sys.modules[cls.__module__].__dict__`` to
    resolve forward references).
    """
    path = _SCRIPTS_DIR / f"{name}.py"
    qualname = f"_test_scripts_{name}"
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not build loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def run_e2e_zarr() -> ModuleType:
    """Loaded ``scripts/run_e2e_zarr.py`` module."""
    return _load("run_e2e_zarr")
