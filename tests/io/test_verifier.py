"""Tests for ``mamba_dvc.io.verify``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import zarr
from mamba_dvc.io.manifest import EntriesManifest, StoreManifest
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from mamba_dvc.io.verify import verify

PROFILE = BONE_SCREW_SYNCHROTRON_V1


class TestVerifyHappyPath:
    def test_clean_store_passes(self, make_store: Callable[..., zarr.Group]) -> None:
        root = make_store(profile=PROFILE)
        report = verify(PROFILE, manifest=None, root=root)
        assert report.ok, report.errors
        assert report.errors == ()


class TestVerifyShapeMismatch:
    def test_synthetic_volume_wrong_shape(self, make_store: Callable[..., zarr.Group]) -> None:
        root = make_store(profile=PROFILE)
        # Replace fs004/volume1 with a wrong-shape array.
        del root[f"{PROFILE.synthetic_group}/fs004/{PROFILE.deformed_name}"]
        bad_shape = (8, 16, 17)
        new_arr = root[f"{PROFILE.synthetic_group}/fs004"].create_array(
            name=PROFILE.deformed_name,
            shape=bad_shape,
            dtype="float32",
            chunks=bad_shape,
        )
        new_arr[:] = np.zeros(bad_shape, dtype=np.float32)

        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any("shape" in e and "fs004" in e for e in report.errors)


class TestVerifyMissingFlow:
    def test_synthetic_missing_flow_flagged(
        self, make_store: Callable[..., zarr.Group]
    ) -> None:
        root = make_store(profile=PROFILE)
        del root[f"{PROFILE.synthetic_group}/fs004/{PROFILE.flow_name}"]
        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any("flow" in e and "fs004" in e and "missing" in e for e in report.errors)

    def test_iterations_missing_flow_is_ok(
        self, make_store: Callable[..., zarr.Group]
    ) -> None:
        # Real-deformation entries don't carry a flow array; that's
        # not an error.
        root = make_store(profile=PROFILE)
        report = verify(PROFILE, manifest=None, root=root)
        assert report.ok, report.errors


class TestVerifyDtypeMismatch:
    def test_volume0_wrong_dtype(self, make_store: Callable[..., zarr.Group]) -> None:
        root = make_store(profile=PROFILE)
        del root[f"{PROFILE.base_group}/{PROFILE.reference_name}"]
        bad = root[PROFILE.base_group].create_array(
            name=PROFILE.reference_name,
            shape=(8, 16, 16),
            dtype="uint16",
            chunks=(8, 16, 16),
        )
        bad[:] = np.zeros((8, 16, 16), dtype=np.uint16)
        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any("expected float32" in e for e in report.errors)


class TestVerifyFlowAxisOrder:
    def test_zyx3_storage_when_profile_says_3zyx_flagged(
        self, make_store: Callable[..., zarr.Group]
    ) -> None:
        root = make_store(profile=PROFILE)
        # Replace fs004/flow with the alternate axis order.
        flow_path = f"{PROFILE.synthetic_group}/fs004/{PROFILE.flow_name}"
        del root[flow_path]
        new_arr = root[f"{PROFILE.synthetic_group}/fs004"].create_array(
            name=PROFILE.flow_name,
            shape=(8, 16, 16, 3),
            dtype="float32",
            chunks=(8, 16, 16, 3),
        )
        new_arr[:] = np.zeros((8, 16, 16, 3), dtype=np.float32)

        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any(
            "axis_order=3_zyx" in e or "leading axis size 3" in e for e in report.errors
        )


class TestVerifyMissingDefaultMask:
    def test_default_mask_absent_flagged(self, make_store: Callable[..., zarr.Group]) -> None:
        # Build a store whose base/ has only mask_fill — the default
        # 'mask' name is gone.
        root = make_store(profile=PROFILE, masks=("mask_fill",))
        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any("default_mask 'mask'" in e for e in report.errors)


class TestVerifyManifestSubset:
    def test_manifest_listed_entry_not_in_store_flagged(
        self, make_store: Callable[..., zarr.Group]
    ) -> None:
        root = make_store(profile=PROFILE)
        manifest = StoreManifest(
            iterations=EntriesManifest(entries=["nope999"]),
        )
        report = verify(PROFILE, manifest=manifest, root=root)
        assert not report.ok
        assert any("manifest references" in e and "nope999" in e for e in report.errors)

    def test_manifest_subset_of_existing_entries_passes(
        self, make_store: Callable[..., zarr.Group]
    ) -> None:
        root = make_store(profile=PROFILE, real_entries=("016", "024"))
        manifest = StoreManifest(iterations=EntriesManifest(entries=["016"]))
        report = verify(PROFILE, manifest=manifest, root=root)
        assert report.ok, report.errors


class TestVerifyTopLevelGroups:
    def test_missing_synthetic_group_flagged(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        # Build base/ + iterations/ but skip synthetic/.
        base = root.create_group(PROFILE.base_group)
        ref = base.create_array(
            name=PROFILE.reference_name,
            shape=(8, 8, 8),
            dtype="float32",
            chunks=(8, 8, 8),
        )
        ref[:] = np.zeros((8, 8, 8), dtype=np.float32)
        root.create_group(PROFILE.real_group)

        report = verify(PROFILE, manifest=None, root=root)
        assert not report.ok
        assert any("synthetic" in e for e in report.errors)
