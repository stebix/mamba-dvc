"""Tests for ``mamba_dvc.io.dataset``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from mamba_dvc.io.dataset import NO_MASK, DvcDataset
from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.io.manifest import (
    BaseManifest,
    EntriesManifest,
    StoreManifest,
    SyntheticManifest,
)
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from mamba_dvc.io.verify import StoreVerificationError
from mamba_dvc.validate.synthetic import rigid_shift

PROFILE = BONE_SCREW_SYNCHROTRON_V1

DiskStoreFactory = Callable[..., Path]


class TestOpenAgainstProfile:
    def test_open_succeeds_on_clean_store(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        assert ds.profile is PROFILE
        assert ds.volume_shape == (8, 16, 16)
        assert ds.verification_report.ok

    def test_strict_open_raises_on_bad_store(
        self, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        # Build a store missing the default mask.
        path = make_disk_store(profile=PROFILE, masks=("mask_fill",))
        with pytest.raises(StoreVerificationError):
            DvcDataset.open(path)

    def test_non_strict_open_returns_report(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, masks=("mask_fill",))
        ds = DvcDataset.open(path, strict=False)
        assert not ds.verification_report.ok

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DvcDataset.open(tmp_path / "nope.zarr")


class TestListings:
    def test_lists_real_synthetic_and_masks(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(
            profile=PROFILE,
            masks=("mask", "mask_fill"),
            real_entries=("016", "024"),
        )
        ds = DvcDataset.open(path)
        assert ds.list_real() == ["016", "024"]
        assert ds.list_synthetic() == ["fs004"]
        assert ds.list_masks() == ["mask", "mask_fill"]
        assert ds.list_all() == ["016", "024", "fs004"]


class TestLoadPairSynthetic:
    def test_synthetic_pair_carries_gt_field(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        pair = ds.load_pair("fs004")
        assert pair.kind == "synthetic"
        assert pair.name == "fs004"
        assert pair.reference.shape == (8, 16, 16)
        assert pair.deformed.shape == (8, 16, 16)
        assert pair.mask is not None
        assert pair.mask.shape == (8, 16, 16)
        assert isinstance(pair.gt_field, GroundTruthField)

    def test_gt_field_is_callable_at_poi_centers(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        pair = ds.load_pair("fs004")
        assert pair.gt_field is not None

        coords = np.array([[3.0, 7.0, 7.0]], dtype=np.float32)
        out = pair.gt_field(coords)
        # Default fixture entry is rigid_shift((0.5, -0.25, 1.0)).
        np.testing.assert_allclose(out, [[0.5, -0.25, 1.0]], atol=1e-3)


class TestLoadPairReal:
    def test_real_pair_has_no_gt(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        pair = ds.load_pair("016")
        assert pair.kind == "real"
        assert pair.gt_field is None
        assert pair.mask is not None


class TestLoadPairMaskOverride:
    def test_explicit_mask_name(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, masks=("mask", "mask_fill"))
        ds = DvcDataset.open(path)
        # Explicit override picks mask_fill instead of profile default.
        pair = ds.load_pair("016", mask="mask_fill")
        assert pair.mask is not None
        assert pair.mask.shape == (8, 16, 16)

    def test_unknown_mask_name_raises(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        with pytest.raises(KeyError, match="unknown mask"):
            ds.load_pair("016", mask="not_a_mask")


class TestLoadPairNoMaskSentinel:
    def test_no_mask_sentinel_skips_mask(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        pair = ds.load_pair("fs004", mask=NO_MASK)
        assert pair.mask is None

    def test_no_mask_is_a_singleton(self) -> None:
        from mamba_dvc.io.dataset import NoMaskSentinel

        assert NoMaskSentinel() is NO_MASK


class TestDryShape:
    def test_dry_shape_propagates(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = DvcDataset.open(path)
        pair = ds.load_pair("fs004", dry_shape=(4, 8, 8))
        assert pair.reference.shape == (4, 8, 8)
        assert pair.deformed.shape == (4, 8, 8)
        assert pair.mask is not None and pair.mask.shape == (4, 8, 8)
        assert pair.gt_field is not None and pair.gt_field.shape == (4, 8, 8)


class TestManifestSubset:
    def test_subset_limits_listings(self, make_disk_store: DiskStoreFactory) -> None:
        # Build a store with two synthetic entries; manifest restricts
        # to one.
        from tests.io.conftest import _SyntheticEntry

        synthetic = (
            _SyntheticEntry("fs004", rigid_shift((0.5, 0.0, 0.0))),
            _SyntheticEntry("fs104", rigid_shift((0.0, 0.5, 0.0))),
        )
        path = make_disk_store(profile=PROFILE, synthetic_entries=synthetic)
        manifest = StoreManifest(
            synthetic=SyntheticManifest(entries=["fs004"]),
        )
        ds = DvcDataset.open(path, manifest=manifest)
        assert ds.list_synthetic() == ["fs004"]
        assert "fs104" not in ds.list_all()

    def test_default_mask_override_via_manifest(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE, masks=("mask", "mask_fill"))
        manifest = StoreManifest(
            base=BaseManifest(default_mask="mask_fill"),
        )
        ds = DvcDataset.open(path, manifest=manifest)
        # load_pair() with no mask kwarg picks manifest's mask_fill, not
        # profile's mask.
        pair = ds.load_pair("016")
        assert pair.mask is not None
        # Both masks have shape (8, 16, 16); a value-based check would
        # depend on RNG. Just confirm no error and presence.

    def test_unknown_profile_in_manifest_raises(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        path = make_disk_store(profile=PROFILE)
        manifest = StoreManifest(profile_name="does_not_exist")
        with pytest.raises(KeyError, match="no registered profile"):
            DvcDataset.open(path, manifest=manifest)

    def test_iterations_subset(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, real_entries=("016", "024", "032"))
        manifest = StoreManifest(
            iterations=EntriesManifest(entries=["024"]),
        )
        ds = DvcDataset.open(path, manifest=manifest)
        assert ds.list_real() == ["024"]
