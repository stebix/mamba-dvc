"""Tests for ``mamba_dvc.cli`` ``budget`` subcommand.

The subcommand is exercised with a monkeypatched ``probe_free_vram``
so the suite runs on CPU-only hosts. A ``@pytest.mark.gpu`` smoke
test would be a natural follow-up but is not required for v1.

Plan: ``docs/plans/batchsize-oracle-v1.md`` §10.3.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from mamba_dvc.cli import _budget as budget_mod
from mamba_dvc.cli import app
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from typer.testing import CliRunner

PROFILE = BONE_SCREW_SYNCHROTRON_V1

DiskStoreFactory = Callable[..., Path]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Monkeypatch ``probe_free_vram`` and ``is_cupy_available``.

    Returns a setter the test can call with a ``per_device`` mapping
    of ``device_id -> (free_bytes, total_bytes)`` plus an optional
    ``visible_count`` for the no-``--devices`` enumeration path.
    """

    def _set(
        *,
        per_device: dict[int, tuple[int, int]] | None = None,
        visible_count: int | None = None,
        cupy_available: bool = True,
    ) -> None:
        per_device = per_device or {0: (40 * 1024**3, 48 * 1024**3)}

        def _probe(device_id: int) -> tuple[int, int]:
            if device_id not in per_device:
                raise RuntimeError(f"unknown device {device_id} in fake probe")
            return per_device[device_id]

        monkeypatch.setattr(budget_mod, "probe_free_vram", _probe)
        monkeypatch.setattr(budget_mod, "is_cupy_available", lambda: cupy_available)

        if visible_count is not None:

            def _enumerate() -> list[int]:
                if visible_count <= 0:
                    raise RuntimeError("no CUDA devices visible to CuPy")
                return list(range(visible_count))

            monkeypatch.setattr(budget_mod, "_enumerate_devices", _enumerate)

    return _set


# ---------------------------------------------------------------- arg parsers


class TestParseVolumeShape:
    def test_three_positive_ints(self) -> None:
        assert budget_mod._parse_volume_shape("960,1280,1280") == (960, 1280, 1280)

    def test_strips_whitespace(self) -> None:
        assert budget_mod._parse_volume_shape(" 1, 2, 3 ") == (1, 2, 3)

    @pytest.mark.parametrize("bad", ["1,2", "1,2,3,4", "abc,1,1", "1,1,0", "-1,1,1"])
    def test_rejects_bad(self, bad: str) -> None:
        from typer import BadParameter

        with pytest.raises(BadParameter):
            budget_mod._parse_volume_shape(bad)


class TestParseWindow:
    def test_single_int_is_cube(self) -> None:
        assert budget_mod._parse_window("96") == (96, 96, 96)

    def test_triple_returns_tuple(self) -> None:
        assert budget_mod._parse_window("32,64,96") == (32, 64, 96)

    @pytest.mark.parametrize("bad", ["0", "-8", "1,2", "abc", "1,2,3,4", "1,1,0"])
    def test_rejects_bad(self, bad: str) -> None:
        from typer import BadParameter

        with pytest.raises(BadParameter):
            budget_mod._parse_window(bad)


class TestParseDevices:
    def test_single(self) -> None:
        assert budget_mod._parse_devices("0") == [0]

    def test_multiple(self) -> None:
        assert budget_mod._parse_devices("0,1,2,3") == [0, 1, 2, 3]

    def test_strips_whitespace(self) -> None:
        assert budget_mod._parse_devices(" 0 , 1 , 2 ") == [0, 1, 2]

    @pytest.mark.parametrize("bad", ["", ",", "abc", "0,-1", "0,1,abc"])
    def test_rejects_bad(self, bad: str) -> None:
        from typer import BadParameter

        with pytest.raises(BadParameter):
            budget_mod._parse_devices(bad)


# ----------------------------------------------------------------- happy path


class TestHappyPath:
    def test_volume_shape_renders_recommendation(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
                "--devices",
                "0",
            ],
        )
        assert result.exit_code == 0, result.stderr
        # Header panel shows the inputs we passed.
        assert "(960, 1280, 1280)" in result.stdout
        assert "linear" in result.stdout
        assert "overlap" in result.stdout
        # VRAM table rendered with the limiting marker.
        assert "<- limit" in result.stdout
        # Recommendation panel rendered with a positive batch.
        assert "batch_size = " in result.stdout
        assert "derivation" in result.stdout
        assert "inequality" in result.stdout

    def test_devices_default_to_enumeration(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        # Two-GPU fleet; verify both rows show up in the table when
        # --devices is not given.
        fake_probe(
            per_device={
                0: (40 * 1024**3, 48 * 1024**3),
                1: (35 * 1024**3, 48 * 1024**3),
            },
            visible_count=2,
        )
        result = runner.invoke(app, ["budget", "--volume-shape", "512,512,512"])
        assert result.exit_code == 0, result.stderr
        # Both device ids appear in the rendered table.
        assert " 0 " in result.stdout
        assert " 1 " in result.stdout

    def test_min_rule_uses_smallest_free_across_devices(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        # device 0 has plenty, device 1 has half. The recommended batch
        # must be sized against device 1.
        fake_probe(
            per_device={
                0: (40 * 1024**3, 48 * 1024**3),
                1: (10 * 1024**3, 48 * 1024**3),
            }
        )
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "512,512,512",
                "--devices",
                "0,1",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        # Limiting device is 1 (the smaller free pool).
        assert payload["probe"]["limiting_device"] == 1
        assert payload["probe"]["min_free_bytes"] == 10 * 1024**3


# ------------------------------------------------------------------- --store


class TestStoreFlag:
    def test_resolves_volume_shape_from_store(
        self,
        runner: CliRunner,
        make_disk_store: DiskStoreFactory,
        fake_probe: Callable[..., None],
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--store",
                str(path),
                # Fixture volume is (8, 16, 16); shrink the window to fit.
                "--window",
                "8",
                "--devices",
                "0",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        # Default fixture builds an (8, 16, 16) volume; budget picks it up.
        assert payload["inputs"]["volume_shape"] == [8, 16, 16]
        # Default fixture has a 'mask' array, so has_mask=True.
        assert payload["inputs"]["has_mask"] is True


# --------------------------------------------------------------- input errors


class TestMutualExclusion:
    def test_neither_store_nor_volume_shape(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["budget"])
        assert result.exit_code == 2
        assert "exactly one of --store or --volume-shape" in result.stderr

    def test_both_store_and_volume_shape(
        self,
        runner: CliRunner,
        make_disk_store: DiskStoreFactory,
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(
            app,
            [
                "budget",
                "--store",
                str(path),
                "--volume-shape",
                "8,16,16",
            ],
        )
        assert result.exit_code == 2
        assert "exactly one of --store or --volume-shape" in result.stderr


class TestBadInputs:
    def test_bad_volume_shape(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["budget", "--volume-shape", "not-a-shape"])
        assert result.exit_code == 2

    def test_bad_window(self, runner: CliRunner, fake_probe: Callable[..., None]) -> None:
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
                "--window",
                "abc",
                "--devices",
                "0",
            ],
        )
        assert result.exit_code == 2

    def test_window_exceeds_volume(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        # BudgetInputs validates that window <= volume on each axis.
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "32,32,32",
                "--window",
                "96",
                "--devices",
                "0",
            ],
        )
        assert result.exit_code == 2
        assert "exceeds volume_shape" in result.stderr


# ------------------------------------------------------------------- runtime


class TestNoCupy:
    def test_missing_cupy_exits_one(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        fake_probe(cupy_available=False)
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
            ],
        )
        assert result.exit_code == 1
        assert "CuPy" in result.stderr


class TestUnderflow:
    def test_too_tight_budget_exits_one(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        # Tiny free pool -- the recommender will raise RuntimeError.
        fake_probe(per_device={0: (1 * 1024**2, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
                "--devices",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "VRAM budget too tight" in result.stderr


class TestProbeFailure:
    def test_probe_runtime_error_exits_one(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(budget_mod, "is_cupy_available", lambda: True)

        def _boom(_device_id: int) -> tuple[int, int]:
            raise RuntimeError("device gone")

        monkeypatch.setattr(budget_mod, "probe_free_vram", _boom)

        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
                "--devices",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "device gone" in result.stderr


# -------------------------------------------------------------------- --json


class TestJson:
    def test_json_includes_inputs_probe_recommendation(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "960,1280,1280",
                "--devices",
                "0",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        # Top-level shape: three sections.
        assert set(payload.keys()) == {"inputs", "probe", "recommendation"}
        # Inputs are echoed back faithfully.
        assert payload["inputs"]["volume_shape"] == [960, 1280, 1280]
        assert payload["inputs"]["mode"] == "linear"
        assert payload["inputs"]["normalization"] == "overlap"
        # Probe block includes the limiting device + the per-device list.
        assert payload["probe"]["limiting_device"] == 0
        assert len(payload["probe"]["devices"]) == 1
        # Recommendation block has a positive batch + the cost numbers.
        rec = payload["recommendation"]
        assert rec["batch_size"] >= 8
        assert rec["resident_bytes"] > 0
        assert rec["per_poi_bytes"] > 0
        assert rec["headroom_fraction"] == 0.15


# ---------------------------------------------------------------- --ceiling


class TestCeiling:
    def test_ceiling_clamps_recommendation(
        self, runner: CliRunner, fake_probe: Callable[..., None]
    ) -> None:
        # Plenty of free VRAM, but ceiling=16 should clamp.
        fake_probe(per_device={0: (40 * 1024**3, 48 * 1024**3)})
        result = runner.invoke(
            app,
            [
                "budget",
                "--volume-shape",
                "256,256,256",
                "--devices",
                "0",
                "--ceiling",
                "16",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["recommendation"]["batch_size"] == 16
        assert payload["recommendation"]["ceiling"] == 16
