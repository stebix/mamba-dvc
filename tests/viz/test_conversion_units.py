"""Unit tests for unit math in ``mamba_dvc.viz._conversion``.

Pure-numpy: no VTK/PyVista required, so these run on every developer
machine regardless of the ``viz`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.types import VoxelSpacing
from mamba_dvc.viz._conversion import downsample_for_render, normalize_spacing, unify_spacing


class TestNormalizeSpacing:
    def test_none_yields_unit_voxel_ones(self):
        sp = normalize_spacing(None)
        assert sp.values == (1.0, 1.0, 1.0)
        assert sp.unit == "voxel"

    def test_passthrough_voxel_spacing(self):
        original = VoxelSpacing((2.0, 1.0, 1.0), "um")
        assert normalize_spacing(original) is original

    def test_tuple_treated_as_voxel(self):
        sp = normalize_spacing((1.5, 0.5, 0.5))
        assert sp.values == (1.5, 0.5, 0.5)
        assert sp.unit == "voxel"

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length 3"):
            normalize_spacing((1.0, 1.0))  # type: ignore[arg-type]


class TestUnifySpacing:
    def test_single_um_to_mm_scales_by_1000(self):
        sp = unify_spacing(VoxelSpacing((1000.0, 1000.0, 1000.0), "um"))
        assert sp.unit == "mm"
        assert sp.values == pytest.approx((1.0, 1.0, 1.0))

    def test_multiple_consistent_inputs_pass(self):
        a = VoxelSpacing((1.0, 1.0, 1.0), "mm")
        b = VoxelSpacing((1000.0, 1000.0, 1000.0), "um")
        c = VoxelSpacing((1_000_000.0, 1_000_000.0, 1_000_000.0), "nm")
        out = unify_spacing(a, b, c)
        assert out.unit == "mm"
        assert out.values == pytest.approx((1.0, 1.0, 1.0))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            unify_spacing()

    def test_rejects_voxel_target(self):
        with pytest.raises(ValueError, match="physical unit"):
            unify_spacing(VoxelSpacing((1.0, 1.0, 1.0), "mm"), target="voxel")

    def test_rejects_voxel_only_inputs(self):
        with pytest.raises(ValueError, match="voxel-unit"):
            unify_spacing(VoxelSpacing((1.0, 1.0, 1.0), "voxel"))

    def test_rejects_mixed_voxel_and_physical(self):
        with pytest.raises(ValueError, match="mixed"):
            unify_spacing(
                VoxelSpacing((1.0, 1.0, 1.0), "voxel"),
                VoxelSpacing((1.0, 1.0, 1.0), "mm"),
            )

    def test_rejects_disagreeing_physical_inputs(self):
        with pytest.raises(ValueError, match="disagree"):
            unify_spacing(
                VoxelSpacing((1.0, 1.0, 1.0), "mm"),
                VoxelSpacing((2.0, 1.0, 1.0), "mm"),
            )

    def test_anisotropic_um_converted_per_axis(self):
        sp = unify_spacing(VoxelSpacing((500.0, 250.0, 250.0), "um"))
        assert sp.unit == "mm"
        assert sp.values == pytest.approx((0.5, 0.25, 0.25))


class TestDownsampleForRender:
    def test_factor_one_is_passthrough(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        vol_ds, mask_ds, sp_ds = downsample_for_render(vol, factor=1)
        assert vol_ds is vol
        assert mask_ds is None
        assert sp_ds.values == (1.0, 1.0, 1.0)
        assert sp_ds.unit == "voxel"

    def test_strided_volume_has_expected_shape(self):
        vol = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        vol_ds, _, _ = downsample_for_render(vol, factor=2)
        assert vol_ds.shape == (2, 2, 2)

    def test_first_voxel_is_preserved(self):
        # Strided slicing keeps the (0, 0, 0) corner at (0, 0, 0): this is the
        # alignment guarantee the function exists to deliver.
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        vol[0, 0, 0] = 7.0
        vol[2, 2, 2] = 3.0
        vol_ds, _, _ = downsample_for_render(vol, factor=2)
        assert vol_ds[0, 0, 0] == 7.0
        assert vol_ds[1, 1, 1] == 3.0

    def test_spacing_values_are_scaled_per_axis(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        spacing = VoxelSpacing((1.5, 2.5, 0.5), "um")
        _, _, sp_ds = downsample_for_render(vol, spacing=spacing, factor=2)
        assert sp_ds.values == pytest.approx((3.0, 5.0, 1.0))
        assert sp_ds.unit == "um"

    def test_mask_is_downsampled_alongside(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[0, 0, 0] = True
        mask[2, 2, 2] = True
        _, mask_ds, _ = downsample_for_render(vol, mask=mask, factor=2)
        assert mask_ds is not None
        assert mask_ds.shape == (2, 2, 2)
        assert bool(mask_ds[0, 0, 0]) is True
        assert bool(mask_ds[1, 1, 1]) is True

    def test_mask_shape_mismatch_raises(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        bad_mask = np.zeros((4, 4, 5), dtype=bool)
        with pytest.raises(ValueError, match="does not match"):
            downsample_for_render(vol, mask=bad_mask, factor=2)

    def test_factor_zero_raises(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="positive integer"):
            downsample_for_render(vol, factor=0)

    def test_factor_negative_raises(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="positive integer"):
            downsample_for_render(vol, factor=-1)

    def test_non_3d_volume_raises(self):
        with pytest.raises(ValueError, match="3D"):
            downsample_for_render(np.zeros((4, 4), dtype=np.float32), factor=2)
