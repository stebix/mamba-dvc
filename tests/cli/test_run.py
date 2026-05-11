"""Tests for the ``mamba-dvc run`` subcommand (campaign driver CLI)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml
from mamba_dvc.cli import app
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from mamba_dvc.types import DisplacementField
from typer.testing import CliRunner

DiskStoreFactory = Callable[..., Path]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_config(path: Path, store: Path, out_dir: Path, **extra) -> Path:
    cfg: dict = {
        "campaign": "camp",
        "stores": [str(store)],
        "out_dir": str(out_dir),
        "select": {"real": "none", "synthetic": "all"},
        "defaults": {"flow_convention": "push_forward"},
    }
    cfg.update(extra)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _fake_correlate(reference, deformed, mask=None, *, device_ids=None, **kwargs):
    n = 3
    return DisplacementField(
        positions=np.full((n, 3), 2.0, dtype=np.float32),
        displacements=np.zeros((n, 3), dtype=np.float32),
        valid=np.ones(n, dtype=np.bool_),
        confidence=np.ones(n, dtype=np.float32),
        status=np.zeros(n, dtype=np.uint8),
        grid_shape=(n, 1, 1),
        spacing=(1, 1, 1),
        window=(96, 96, 96),
    )


class TestDryRun:
    def test_dry_run_lists_variants_and_jobs(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        store = make_disk_store(profile=BONE_SCREW_SYNCHROTRON_V1, shape=(8, 16, 16))
        cfg = _write_config(
            tmp_path / "c.yaml", store, tmp_path / "out", sweep={"mask_threshold": [0.7, 0.5]}
        )
        result = runner.invoke(app, ["run", str(cfg), "--dry-run", "--no-color"])
        assert result.exit_code == 0, result.output
        assert "2 variant(s)" in result.output
        assert "mt0.7-" in result.output and "mt0.5-" in result.output
        assert "pending job(s)" in result.output
        # No campaign dir created by a dry run.
        assert not (tmp_path / "out" / "camp").exists()

    def test_dry_run_only_filter_narrows(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        store = make_disk_store(profile=BONE_SCREW_SYNCHROTRON_V1, shape=(8, 16, 16))
        cfg = _write_config(
            tmp_path / "c.yaml",
            store,
            tmp_path / "out",
            sweep={"mask_threshold": [0.9, 0.7, 0.5]},
        )
        result = runner.invoke(
            app, ["run", str(cfg), "--dry-run", "--no-color", "--only", "mask_threshold=0.5"]
        )
        assert result.exit_code == 0, result.output
        assert "1 pending job(s)" in result.output

    def test_dry_run_unknown_only_knob_exits_2(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        store = make_disk_store(profile=BONE_SCREW_SYNCHROTRON_V1, shape=(8, 16, 16))
        cfg = _write_config(tmp_path / "c.yaml", store, tmp_path / "out")
        result = runner.invoke(app, ["run", str(cfg), "--dry-run", "--only", "nope=1"])
        assert result.exit_code == 2


class TestConfigErrors:
    def test_missing_config_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
        assert result.exit_code == 2

    def test_malformed_config_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("stores: [a.zarr]\n", encoding="utf-8")  # missing 'campaign'
        result = runner.invoke(app, ["run", str(bad)])
        assert result.exit_code == 2
        assert "campaign" in result.output


class TestExecute:
    def test_run_executes_and_writes_results(
        self,
        runner: CliRunner,
        make_disk_store: DiskStoreFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("mamba_dvc.run.batch.correlate_multi_gpu", _fake_correlate)
        store = make_disk_store(profile=BONE_SCREW_SYNCHROTRON_V1, shape=(8, 16, 16))
        out_dir = tmp_path / "out"
        cfg = _write_config(
            tmp_path / "c.yaml", store, out_dir, sweep={"mask_threshold": [0.7, 0.5]}
        )
        result = runner.invoke(app, ["run", str(cfg), "--no-color"])
        assert result.exit_code == 0, result.output
        campaign_dir = out_dir / "camp"
        assert (campaign_dir / "manifest.jsonl").exists()
        assert (campaign_dir / "variants.json").exists()
        assert (campaign_dir / "config.snapshot.yaml").exists()
        npzs = list(campaign_dir.rglob("*.npz"))
        assert len(npzs) == 2  # one synthetic entry x two threshold variants
        # Re-running resumes: nothing left to do.
        again = runner.invoke(app, ["run", str(cfg), "--no-color"])
        assert again.exit_code == 0
        assert "nothing pending" in again.output

    def test_run_reports_failure_exit_1(
        self,
        runner: CliRunner,
        make_disk_store: DiskStoreFactory,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr("mamba_dvc.run.batch.correlate_multi_gpu", boom)
        store = make_disk_store(profile=BONE_SCREW_SYNCHROTRON_V1, shape=(8, 16, 16))
        cfg = _write_config(tmp_path / "c.yaml", store, tmp_path / "out")
        result = runner.invoke(app, ["run", str(cfg), "--no-color"])
        assert result.exit_code == 1
        assert "FAILED" in result.output
        # The manifest still records the failed row.
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "camp" / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert any(r["status"] == "failed" for r in rows)
