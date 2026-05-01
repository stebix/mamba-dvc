"""Unit tests for unit math in ``mamba_dvc.viz._conversion``.

Pure-numpy: no VTK/PyVista required, so these run on every developer
machine regardless of the ``viz`` extra.
"""

from __future__ import annotations

import pytest
from mamba_dvc.types import VoxelSpacing
from mamba_dvc.viz._conversion import normalize_spacing, unify_spacing


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
