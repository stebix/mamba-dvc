"""End-to-end smoke test for the zarr-interface layer.

Builds an ephemeral on-disk zarr store conforming to
``bone_screw_synchrotron_v1`` (synthesized from
:mod:`mamba_dvc.validate.synthetic`), opens it via
:meth:`DvcDataset.open`, and runs
:func:`mamba_dvc.validate.known_fields.run_and_evaluate` against a single
synthetic deformation on ``device_ids=[0]``.

Marked ``gpu``; skipped on machines without CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

cupy = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


def _ensure_io_conftest_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    """Build a small bone-screw-synchrotron-v1 store with one synthetic entry."""
    _ensure_io_conftest_on_path()
    from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
    from mamba_dvc.validate.synthetic import rigid_shift

    from tests.io.conftest import _populate_store, _SyntheticEntry  # type: ignore

    store_path = tmp_path / "smoke.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    _populate_store(
        root,
        profile=BONE_SCREW_SYNCHROTRON_V1,
        shape=(96, 128, 128),
        masks=("mask",),
        real_entries=("016",),
        synthetic_entries=(_SyntheticEntry("fs004", rigid_shift((0.5, 0.0, 0.0))),),
        seed=0,
    )
    return store_path


def test_run_and_evaluate_single_gpu(synthetic_store: Path) -> None:
    """One synthetic entry, one GPU; MAE should sit well under one voxel."""
    from mamba_dvc.io.dataset import DvcDataset
    from mamba_dvc.validate.known_fields import run_and_evaluate

    ds = DvcDataset.open(synthetic_store)
    report = run_and_evaluate(
        ds,
        "fs004",
        device_ids=[0],
        window=32,
        overlap=0.5,
        batch_size=8,
    )
    assert report.name == "fs004"
    assert report.n_valid > 0
    # 0.5-voxel rigid shift with a reasonable POI grid should land
    # comfortably under 1.0 voxel MAE.
    assert np.isfinite(report.mae)
    assert report.mae < 1.0
