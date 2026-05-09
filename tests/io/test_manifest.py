"""Tests for ``mamba_dvc.io.manifest``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from mamba_dvc.io.manifest import (
    BaseManifest,
    EntriesManifest,
    FlowOverride,
    StoreManifest,
    SyntheticManifest,
)
from mamba_dvc.types import VoxelSpacing


def _write_yaml(path: Path, doc: dict) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return path


class TestParseMinimal:
    def test_only_version_and_format(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "scan.zarr.yaml",
            {
                "dvc_store_version": 1,
                "store_format": "bone_screw_synchrotron_v1",
            },
        )
        m = StoreManifest.from_yaml(path)
        assert m.version == 1
        assert m.profile_name == "bone_screw_synchrotron_v1"
        assert m.spacing is None
        assert m.base == BaseManifest()
        assert m.iterations == EntriesManifest()
        assert m.synthetic == SyntheticManifest()

    def test_empty_yaml_is_default_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        m = StoreManifest.from_yaml(path)
        assert m == StoreManifest()


class TestParseFullOverride:
    def test_every_section_round_trips(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "scan.zarr.yaml",
            {
                "dvc_store_version": 1,
                "store_format": "bone_screw_synchrotron_v1",
                "spacing": {"values": [1.0, 0.5, 0.5], "unit": "um"},
                "base": {"default_mask": "mask_fill", "masks": ["mask", "mask_fill"]},
                "iterations": {"entries": ["016", "024"]},
                "synthetic": {
                    "entries": ["fs004"],
                    "flow": {
                        "axis_order": "zyx_3",
                        "vector_order": "dz_dy_dx",
                        "convention": "push_forward",
                        "units": "voxel",
                    },
                },
            },
        )
        m = StoreManifest.from_yaml(path)
        assert m.profile_name == "bone_screw_synchrotron_v1"
        assert m.spacing == VoxelSpacing(values=(1.0, 0.5, 0.5), unit="um")
        assert m.base.default_mask == "mask_fill"
        assert m.base.masks == ["mask", "mask_fill"]
        assert m.iterations.entries == ["016", "024"]
        assert m.synthetic.entries == ["fs004"]
        assert m.synthetic.flow == FlowOverride(
            axis_order="zyx_3",
            vector_order="dz_dy_dx",
            convention="push_forward",
            units="voxel",
        )

    def test_round_trip_through_yaml_writer(self, tmp_path: Path) -> None:
        original = StoreManifest(
            profile_name="bone_screw_synchrotron_v1",
            spacing=VoxelSpacing(values=(2.0, 1.0, 1.0), unit="nm"),
            base=BaseManifest(default_mask="mask_fill", masks=["mask"]),
            iterations=EntriesManifest(entries=["016"]),
            synthetic=SyntheticManifest(
                entries=["fs004", "fs104"],
                flow=FlowOverride(convention="push_forward"),
            ),
        )
        out = tmp_path / "round-trip.yaml"
        original.to_yaml(out)
        reread = StoreManifest.from_yaml(out)
        assert reread == original


class TestParseRejects:
    def test_unknown_version_rejected(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path / "x.yaml", {"dvc_store_version": 99})
        with pytest.raises(ValueError, match="unsupported dvc_store_version"):
            StoreManifest.from_yaml(path)

    def test_invalid_axis_order_rejected(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "x.yaml",
            {
                "dvc_store_version": 1,
                "synthetic": {"flow": {"axis_order": "xyz_3"}},
            },
        )
        with pytest.raises(ValueError, match="expected one of"):
            StoreManifest.from_yaml(path)

    def test_invalid_spacing_unit_rejected(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path / "x.yaml",
            {
                "dvc_store_version": 1,
                "spacing": {"values": [1.0, 1.0, 1.0], "unit": "foo"},
            },
        )
        with pytest.raises(ValueError, match="unknown spacing unit"):
            StoreManifest.from_yaml(path)


class TestDiscover:
    def test_returns_none_when_no_sources(self, tmp_path: Path) -> None:
        store_path = tmp_path / "scan.zarr"
        # Neither sidecar nor zarr exists.
        assert StoreManifest.discover(store_path) is None

    def test_reads_sidecar_when_zarr_missing(self, tmp_path: Path) -> None:
        store_path = tmp_path / "scan.zarr"
        sidecar = store_path.with_suffix(".zarr.yaml")
        _write_yaml(
            sidecar,
            {"dvc_store_version": 1, "store_format": "bone_screw_synchrotron_v1"},
        )
        m = StoreManifest.discover(store_path)
        assert m is not None
        assert m.profile_name == "bone_screw_synchrotron_v1"
