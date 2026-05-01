"""Unit tests for VoxelSpacing — no VTK required."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from mamba_dvc.types import VoxelSpacing


class TestVoxelSpacing:
    def test_default_unit_is_voxel(self):
        sp = VoxelSpacing((1.0, 1.0, 1.0))
        assert sp.unit == "voxel"

    def test_holds_anisotropic_values(self):
        sp = VoxelSpacing((2.0, 0.5, 0.5), "um")
        assert sp.values == (2.0, 0.5, 0.5)
        assert sp.unit == "um"

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="length 3"):
            VoxelSpacing((1.0, 1.0))  # type: ignore[arg-type]

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="positive"):
            VoxelSpacing((1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="positive"):
            VoxelSpacing((1.0, -1.0, 1.0))

    def test_is_frozen_and_hashable(self):
        sp = VoxelSpacing((1.0, 1.0, 1.0), "mm")
        with pytest.raises(FrozenInstanceError):
            sp.unit = "um"  # type: ignore[misc]
        # Hashable -> usable as dict key.
        d = {sp: 1}
        assert d[sp] == 1
