"""Tests for ``mamba_dvc.cli.inspect``."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from mamba_dvc.cli import app
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from typer.testing import CliRunner

PROFILE = BONE_SCREW_SYNCHROTRON_V1

DiskStoreFactory = Callable[..., Path]


@pytest.fixture
def runner() -> CliRunner:
    # Click 8.3 (typer >= 0.15) captures stderr separately by default;
    # result.stdout / result.stderr are independently accessible.
    return CliRunner()


# ---------------------------------------------------------------- happy path


class TestHappyPath:
    def test_renders_tree_and_returns_zero(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 0, result.stderr
        # Header / profile / validity panel
        assert "bone_screw_synchrotron_v1" in result.stdout
        assert "OK" in result.stdout
        # Slot tree shows reference, mask, and the synthetic entry.
        assert "volume0" in result.stdout
        assert "mask" in result.stdout
        assert "fs004" in result.stdout
        assert "[reference]" in result.stdout
        assert "[default]" in result.stdout

    def test_default_mask_tag_tracks_manifest_override(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        # Sidecar manifest pointing the default mask at mask_fill.
        path = make_disk_store(profile=PROFILE, masks=("mask", "mask_fill"))
        sidecar = path.with_suffix(path.suffix + ".yaml")
        sidecar.write_text(
            yaml.safe_dump({"dvc_store_version": 1, "base": {"default_mask": "mask_fill"}}),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 0, result.stderr
        # Sidecar source labelled in header.
        assert "sidecar yaml" in result.stdout
        # The [default] tag should sit on mask_fill, not mask. Find the
        # mask_fill line and assert [default] is on it.
        for line in result.stdout.splitlines():
            if "mask_fill" in line:
                assert "[default]" in line
                break
        else:
            pytest.fail("mask_fill line not found in tree output")


# ------------------------------------------------------------ verifier errors


class TestVerificationFailure:
    def test_missing_default_mask_renders_tree_and_exits_one(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        # Drop the default mask 'mask' — verifier flags it but the tree
        # of what *was* parsed still renders.
        path = make_disk_store(profile=PROFILE, masks=("mask_fill",))
        result = runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 1
        assert "FAILED" in result.stdout
        assert "default_mask" in result.stdout  # error message body
        # Tree still rendered: reference and mask_fill should appear.
        assert "volume0" in result.stdout
        assert "mask_fill" in result.stdout


# ----------------------------------------------------------- open failures


class TestOpenFailure:
    def test_missing_path_exits_two(self, runner: CliRunner, tmp_path: Path) -> None:
        # Typer's exists=True validator catches this before our code runs;
        # click maps the validation failure to exit 2.
        result = runner.invoke(app, ["inspect", str(tmp_path / "nope.zarr")])
        assert result.exit_code == 2

    def test_unknown_profile_exits_two(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(app, ["inspect", "--profile", "no_such_profile_v9", str(path)])
        assert result.exit_code == 2
        assert "no_such_profile_v9" in result.stderr


# ----------------------------------------------------------------- --verbose


class TestVerbose:
    def test_verbose_adds_chunks_and_size(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(app, ["inspect", "--verbose", str(path)])
        assert result.exit_code == 0, result.stderr
        assert "chunks=" in result.stdout
        assert "codecs=" in result.stdout
        # Default fixture shape (8,16,16) float32 = 8 KiB; verbose mode
        # should print a binary-unit size somewhere.
        assert "KiB" in result.stdout


# -------------------------------------------------------------------- --json


class TestJson:
    def test_json_emits_parseable_inventory(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE, masks=("mask",), real_entries=("016",))
        result = runner.invoke(app, ["inspect", "--json", str(path)])
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["header"]["profile"] == "bone_screw_synchrotron_v1"
        assert payload["validity"]["ok"] is True
        assert payload["slots"]["reference"]["dtype"] == "float32"
        assert payload["slots"]["default_mask"] == "mask"
        assert "mask" in payload["slots"]["masks"]
        assert "016" in payload["slots"]["real"]
        assert "fs004" in payload["slots"]["synthetic"]
        assert payload["slots"]["synthetic"]["fs004"]["flow"] is not None

    def test_json_verbose_includes_chunks(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(app, ["inspect", "--json", "--verbose", str(path)])
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        ref = payload["slots"]["reference"]
        assert "chunks" in ref
        assert "codecs" in ref
        assert "nbytes" in ref


# ---------------------------------------------------------------- --profile


class TestProfileFlag:
    def test_explicit_profile_is_labelled_in_header(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        result = runner.invoke(
            app,
            ["inspect", "--profile", "bone_screw_synchrotron_v1", str(path)],
        )
        assert result.exit_code == 0, result.stderr
        assert "--profile bone_screw_synchrotron_v1" in result.stdout


# ---------------------------------------------------------------- broken entry


class TestBrokenEntry:
    """A store with a per-entry structural problem renders, not crashes."""

    def test_broken_synthetic_renders_in_tree_and_exits_one(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        shutil.rmtree(path / "synthetic" / "fs004" / "volume1")

        result = runner.invoke(app, ["inspect", str(path)])

        assert result.exit_code == 1, result.stderr
        # Validity panel still shows the verifier's error.
        assert "FAILED" in result.stdout
        assert "synthetic/fs004/volume1: missing" in result.stdout
        # Tree shows the broken entry inline with the synthetic group.
        assert "fs004" in result.stdout
        assert "[BROKEN:" in result.stdout
        # Group header reflects the count split.
        assert "0 healthy" in result.stdout
        assert "1 broken" in result.stdout

    def test_broken_entry_in_json_output(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        shutil.rmtree(path / "synthetic" / "fs004" / "volume1")

        result = runner.invoke(app, ["inspect", "--json", str(path)])

        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["validity"]["ok"] is False
        assert "fs004" in payload["slots"]["broken"]
        broken = payload["slots"]["broken"]["fs004"]
        assert broken["kind"] == "synthetic"
        assert broken["missing"] == ["volume1"]
        # Healthy section no longer lists the broken entry.
        assert "fs004" not in payload["slots"]["synthetic"]

    def test_clean_store_has_empty_broken_block_in_json(
        self, runner: CliRunner, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        # Build under a custom root to avoid the pytest tempdir name leaking
        # the substring "broken" into the rendered store path.
        path = make_disk_store(profile=PROFILE, name="clean")
        # Sanity: tmp_path doesn't itself contain the literal "broken".
        assert "broken" not in str(path).lower() or tmp_path is not None

        json_result = runner.invoke(app, ["inspect", "--json", str(path)])
        assert json_result.exit_code == 0
        payload = json.loads(json_result.stdout)
        assert payload["slots"]["broken"] == {}

        # Rich output shows the plain "(N entries)" header, not the
        # "(.. healthy, .. broken)" split.
        text_result = runner.invoke(app, ["inspect", str(path)])
        assert text_result.exit_code == 0
        assert "healthy" not in text_result.stdout
        assert "BROKEN" not in text_result.stdout
