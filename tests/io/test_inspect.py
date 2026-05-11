"""Tests for ``mamba_dvc.io.inspect``."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import zarr
from mamba_dvc.io.dataset import DvcDataset
from mamba_dvc.io.inspect import (
    DatasetStats,
    FlowLayoutError,
    ProgressEvent,
    inspect_dataset,
    verify_flow_convention,
)
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from mamba_dvc.validate.synthetic import rigid_shift, warp

PROFILE = BONE_SCREW_SYNCHROTRON_V1

DiskStoreFactory = Callable[..., Path]


class _EventCollector:
    """Append-only progress reporter for assertion-friendly tests."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def __call__(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _open(path: Path) -> DvcDataset:
    return DvcDataset.open(path)


class TestStreamingStatsAreConsistent:
    def test_reference_stats_match_numpy(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        truth = np.asarray(ds.reference[:]).astype(np.float64)

        stats = inspect_dataset(ds, median="exact")

        assert stats.reference.shape == ds.volume_shape
        assert stats.reference.count == truth.size
        assert math.isclose(stats.reference.mean, float(truth.mean()), rel_tol=1e-6)
        assert math.isclose(stats.reference.min, float(truth.min()), rel_tol=1e-6)
        assert math.isclose(stats.reference.max, float(truth.max()), rel_tol=1e-6)
        assert math.isclose(stats.reference.median, float(np.median(truth)), rel_tol=1e-6)
        assert stats.reference.median_method == "exact"

    def test_streaming_min_max_mean_match_full_load(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        # Force multi-slab streaming via a tiny budget; the slab-loop
        # min/max/mean must still equal the single-shot reductions.
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        truth = np.asarray(ds.reference[:]).astype(np.float64)

        stats = inspect_dataset(ds, median="skip", slab_budget_bytes=1024)

        assert math.isclose(stats.reference.mean, float(truth.mean()), rel_tol=1e-6)
        assert stats.reference.min == pytest.approx(float(truth.min()))
        assert stats.reference.max == pytest.approx(float(truth.max()))
        assert math.isnan(stats.reference.median)
        assert stats.reference.median_method == "skipped"


class TestHistogramMedianAccuracy:
    def test_histogram_median_within_bin_resolution(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        # 256 bins is small enough that the test detects an off-by-one
        # in the cumulative-sum logic without depending on exact float
        # equality.
        bins = 256
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        truth = np.asarray(ds.reference[:])
        exact = float(np.median(truth))
        spread = float(truth.max() - truth.min())

        stats = inspect_dataset(ds, median="histogram", histogram_bins=bins)

        tolerance = spread / bins
        assert abs(stats.reference.median - exact) <= tolerance
        assert stats.reference.median_method == "histogram"


class TestMaskStats:
    def test_foreground_count_matches_truth(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        truth = np.asarray(ds.masks["mask"][:]).astype(np.bool_)
        expected_fg = int(np.count_nonzero(truth))
        expected_bg = truth.size - expected_fg

        stats = inspect_dataset(ds, median="skip")

        ms = stats.masks["mask"]
        assert ms.foreground_count == expected_fg
        assert ms.background_count == expected_bg
        assert ms.foreground_fraction == pytest.approx(expected_fg / truth.size)
        assert ms.shape == ds.volume_shape


class TestDisplacementStats:
    def test_per_axis_mean_recovers_rigid_shift(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        # The default fixture is rigid_shift((0.5, -0.25, 1.0)) so every
        # voxel of the synthetic flow should carry the same vector and
        # the per-axis mean / max-abs are determined exactly.
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        stats = inspect_dataset(ds, median="exact")

        flow = stats.deformations["fs004"].flow
        assert flow is not None
        assert flow.axis_order == PROFILE.flow_defaults.axis_order
        np.testing.assert_allclose(flow.per_axis_mean, (0.5, -0.25, 1.0), atol=1e-6)
        np.testing.assert_allclose(flow.per_axis_max_abs, (0.5, 0.25, 1.0), atol=1e-6)
        expected_magnitude = math.sqrt(0.5**2 + 0.25**2 + 1.0**2)
        assert flow.mean_magnitude == pytest.approx(expected_magnitude, rel=1e-6)
        assert flow.max_magnitude == pytest.approx(expected_magnitude, rel=1e-6)
        assert flow.median_magnitude == pytest.approx(expected_magnitude, rel=1e-6)


class TestFlowLayoutValidation:
    def test_well_formed_flow_does_not_raise(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        # Should not raise.
        inspect_dataset(ds, median="skip")

    def test_swapped_size3_axis_position_raises(self, tmp_path: Path) -> None:
        # Build a store whose synthetic flow has shape (Z, Y, X, 3) but
        # whose resolved axis_order is the profile default "3_zyx" --
        # the inspector must catch the layout mismatch before reading
        # any voxels.
        path = _build_store_with_bad_flow_layout(
            tmp_path,
            shape=(8, 16, 16),
            bad_layout="zyx_3",  # actual on disk
        )
        ds = DvcDataset.open(path, strict=False)
        with pytest.raises(FlowLayoutError) as excinfo:
            inspect_dataset(ds, median="skip")
        msg = str(excinfo.value)
        assert "fs_bad" in msg
        assert "axis_order='3_zyx'" in msg

    def test_wrong_spatial_shape_raises(self, tmp_path: Path) -> None:
        path = _build_store_with_bad_flow_layout(
            tmp_path,
            shape=(8, 16, 16),
            bad_layout="3_zyx_wrong_spatial",
        )
        ds = DvcDataset.open(path, strict=False)
        with pytest.raises(FlowLayoutError) as excinfo:
            inspect_dataset(ds, median="skip")
        assert "spatial axes" in str(excinfo.value)


class TestProgressEvents:
    def test_event_sequence_matches_walk_plan(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, real_entries=("016",))
        ds = _open(path)
        collector = _EventCollector()

        inspect_dataset(ds, median="skip", progress=collector)

        stages = [e.stage for e in collector.events]
        names = [e.name for e in collector.events]
        # reference + 1 mask + (real image) + (synthetic image + flow) + done
        assert stages == [
            "reference",
            "mask",
            "deformation_image",
            "deformation_image",
            "deformation_flow",
            "done",
        ]
        assert names == ["reference", "mask", "016", "fs004", "fs004", ""]

        item_total = collector.events[0].item_total
        assert item_total == 5
        assert collector.events[-1].item_index == item_total
        # Indices are strictly increasing on non-done events.
        non_done = [e for e in collector.events if e.stage != "done"]
        assert [e.item_index for e in non_done] == list(range(len(non_done)))


class TestSkippedEntries:
    def test_broken_entries_forwarded(
        self, make_disk_store: DiskStoreFactory, tmp_path: Path
    ) -> None:
        # Build a healthy store, then surgically remove the synthetic
        # entry's volume1 so the dataset surfaces a BrokenEntry.
        path = make_disk_store(profile=PROFILE)
        root = zarr.open_group(str(path), mode="a")
        del root[f"{PROFILE.synthetic_group}/fs004/{PROFILE.deformed_name}"]

        ds = DvcDataset.open(path, strict=False)
        assert "fs004" in ds.broken_entries

        stats = inspect_dataset(ds, median="skip")
        assert "fs004" in stats.skipped
        assert "fs004" not in stats.deformations


class TestInclusionFilters:
    def test_include_subset(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, real_entries=("016", "024"))
        ds = _open(path)

        stats = inspect_dataset(ds, include=["016"], median="skip")

        assert list(stats.deformations.keys()) == ["016"]

    def test_unknown_include_raises(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        with pytest.raises(KeyError):
            inspect_dataset(ds, include=["does-not-exist"], median="skip")

    def test_unknown_mask_raises(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        with pytest.raises(KeyError):
            inspect_dataset(ds, masks=["does-not-exist"], median="skip")

    def test_mask_subset(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE, masks=("mask", "mask_fill"))
        ds = _open(path)

        stats = inspect_dataset(ds, masks=["mask"], median="skip")
        assert list(stats.masks.keys()) == ["mask"]


class TestDeterminism:
    def test_two_runs_match_modulo_elapsed(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        a = inspect_dataset(ds, median="histogram")
        b = inspect_dataset(ds, median="histogram")

        _assert_stats_equal_modulo_elapsed(a, b)


class TestArgumentValidation:
    def test_too_few_bins_raises(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        with pytest.raises(ValueError, match="histogram_bins"):
            inspect_dataset(ds, histogram_bins=1)

    def test_nonpositive_budget_raises(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        with pytest.raises(ValueError, match="slab_budget_bytes"):
            inspect_dataset(ds, slab_budget_bytes=0)


# --------------------------------------------------------------- helpers


def _assert_stats_equal_modulo_elapsed(a: DatasetStats, b: DatasetStats) -> None:
    assert a.reference == b.reference
    assert dict(a.masks) == dict(b.masks)
    assert dict(a.deformations) == dict(b.deformations)
    assert dict(a.skipped) == dict(b.skipped)
    assert a.median_method == b.median_method


def _build_store_with_bad_flow_layout(
    tmp_path: Path,
    *,
    shape: tuple[int, int, int],
    bad_layout: str,
) -> Path:
    """Build a minimal v1-profile store whose synthetic flow violates layout.

    ``bad_layout`` selects how the flow is stored relative to the
    profile's expected ``"3_zyx"``:

    - ``"zyx_3"``: shape (Z, Y, X, 3) -- size-3 axis at the wrong end.
    - ``"3_zyx_wrong_spatial"``: shape (3, Z+1, Y, X) -- right axis,
      wrong spatial shape.
    """
    store_path = tmp_path / "bad_flow.zarr"
    root = zarr.open_group(str(store_path), mode="w")

    base = root.create_group(PROFILE.base_group)
    rng = np.random.default_rng(0)
    ref_arr = base.create_array(
        name=PROFILE.reference_name, shape=shape, dtype="float32", chunks=shape
    )
    ref_arr[:] = rng.random(shape, dtype=np.float32)
    mask_arr = base.create_array(name="mask", shape=shape, dtype="bool", chunks=shape)
    mask_arr[:] = np.ones(shape, dtype=np.bool_)

    root.create_group(PROFILE.real_group)
    syn = root.create_group(PROFILE.synthetic_group)
    entry = syn.create_group("fs_bad")

    # Healthy deformed image so the entry is structurally bindable.
    def_arr = entry.create_array(
        name=PROFILE.deformed_name, shape=shape, dtype="float32", chunks=shape
    )
    def_arr[:] = rng.random(shape, dtype=np.float32)

    if bad_layout == "zyx_3":
        flow_arr = entry.create_array(
            name=PROFILE.flow_name,
            shape=(*shape, 3),
            dtype="float32",
            chunks=(*shape, 3),
        )
        flow_arr[:] = np.zeros((*shape, 3), dtype=np.float32)
    elif bad_layout == "3_zyx_wrong_spatial":
        flow_arr = entry.create_array(
            name=PROFILE.flow_name,
            shape=(3, shape[0] + 1, shape[1], shape[2]),
            dtype="float32",
            chunks=(3, shape[0] + 1, shape[1], shape[2]),
        )
        flow_arr[:] = np.zeros((3, shape[0] + 1, shape[1], shape[2]), dtype=np.float32)
    else:
        raise AssertionError(f"unknown bad_layout: {bad_layout}")

    return store_path


def _overwrite_synthetic_deformed(
    path: Path,
    *,
    entry_name: str,
    shift: tuple[float, float, float],
    convention: str,
) -> None:
    """Re-warp the reference under ``shift`` with ``convention`` and overwrite.

    Used to construct a store whose stored flow encodes one convention
    while the deformed image was produced under another -- the input
    ``verify_flow_convention`` needs to discriminate against.
    """
    root = zarr.open_group(str(path), mode="r+")
    base = root[PROFILE.base_group]
    syn = root[PROFILE.synthetic_group]
    assert isinstance(base, zarr.Group) and isinstance(syn, zarr.Group)
    entry = syn[entry_name]
    assert isinstance(entry, zarr.Group)
    ref_node = base[PROFILE.reference_name]
    def_node = entry[PROFILE.deformed_name]
    assert isinstance(ref_node, zarr.Array) and isinstance(def_node, zarr.Array)

    reference = np.asarray(ref_node[:], dtype=np.float32)
    rewarp = warp(
        reference,
        rigid_shift(shift),
        order=3,
        convention=convention,  # type: ignore[arg-type]
    )
    def_node[:] = rewarp


class TestVerifyFlowConvention:
    def test_default_fixture_is_pull_back(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        checks = verify_flow_convention(ds)

        assert set(checks) == set(ds.list_synthetic())
        check = checks["fs004"]
        assert check.empirical == "pull_back"
        assert check.declared == "pull_back"
        assert check.agrees is True
        assert check.residual_pull_back < check.residual_push_forward
        # rigid_shift((0.5, -0.25, 1.0)) on band-limited texture gives a
        # decisive ratio; keep the bound conservative.
        assert check.ratio > 2.0

    def test_push_forward_synthesis_is_detected(
        self, make_disk_store: DiskStoreFactory
    ) -> None:
        shift = (0.5, -0.25, 1.0)
        path = make_disk_store(profile=PROFILE)
        # Stored flow is unchanged (matches `shift`); only the deformed
        # image is re-rendered under the opposite convention.
        _overwrite_synthetic_deformed(
            path, entry_name="fs004", shift=shift, convention="push_forward"
        )
        ds = _open(path)

        check = verify_flow_convention(ds)["fs004"]

        assert check.empirical == "push_forward"
        # Declared convention in the v1 profile is pull_back; the store
        # disagrees with metadata -- agrees must be False.
        assert check.declared == "pull_back"
        assert check.agrees is False
        assert check.residual_push_forward < check.residual_pull_back
        assert check.ratio > 2.0

    def test_unknown_entry_raises_key_error(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        with pytest.raises(KeyError, match="unknown deformation"):
            verify_flow_convention(ds, entries=["nope"])

    def test_real_entry_raises_value_error(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)
        real_name = ds.list_real()[0]

        with pytest.raises(ValueError, match="synthetic-only"):
            verify_flow_convention(ds, entries=[real_name])

    def test_bad_dry_shape_raises_value_error(self, make_disk_store: DiskStoreFactory) -> None:
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        with pytest.raises(ValueError, match="dry_shape"):
            verify_flow_convention(ds, dry_shape=(0, 64, 64))
        with pytest.raises(ValueError, match="dry_shape"):
            verify_flow_convention(ds, dry_shape=(32, 64))  # type: ignore[arg-type]

    def test_dry_shape_clamps_to_volume_shape(self, make_disk_store: DiskStoreFactory) -> None:
        # Default dry_shape (32, 64, 64) exceeds the (8, 16, 16) fixture
        # along every axis; expect per-axis clamping to volume_shape.
        path = make_disk_store(profile=PROFILE)
        ds = _open(path)

        check = verify_flow_convention(ds, dry_shape=(999, 999, 999))["fs004"]

        assert check.dry_shape == ds.volume_shape
