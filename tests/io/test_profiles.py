"""Tests for the profile registry."""

from __future__ import annotations

import pytest
from mamba_dvc.io import profiles
from mamba_dvc.io.profiles import (
    BONE_SCREW_SYNCHROTRON_V1,
    DtypeInvariants,
    FlowSpec,
    StoreProfile,
    default_profile,
    get_profile,
    list_profiles,
    register_profile,
)


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot/restore the module-level registry around each test."""
    saved = profiles._PROFILES.copy()
    yield
    profiles._PROFILES.clear()
    profiles._PROFILES.update(saved)


def _alt_profile(name: str = "alt_v1") -> StoreProfile:
    return StoreProfile(
        name=name,
        base_group="base",
        reference_name="volume0",
        default_mask=None,
        real_group="iterations",
        synthetic_group="synthetic",
        deformed_name="volume1",
        flow_name="flow",
        flow_defaults=FlowSpec(
            axis_order="zyx_3",
            vector_order="dz_dy_dx",
            convention="pull_back",
            units="voxel",
        ),
        dtype_invariants=DtypeInvariants(
            reference="float32", deformed="float32", mask="bool", flow="float32"
        ),
    )


class TestRegistry:
    def test_built_in_profile_is_registered(self) -> None:
        assert "bone_screw_synchrotron_v1" in list_profiles()
        assert get_profile("bone_screw_synchrotron_v1") is BONE_SCREW_SYNCHROTRON_V1

    def test_register_then_retrieve(self) -> None:
        alt = _alt_profile()
        register_profile(alt)
        assert get_profile("alt_v1") is alt

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_profile(BONE_SCREW_SYNCHROTRON_V1)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="no registered profile"):
            get_profile("does_not_exist")


class TestDefaultProfile:
    def test_returns_only_registered(self) -> None:
        assert default_profile() is BONE_SCREW_SYNCHROTRON_V1

    def test_multiple_registered_raises(self) -> None:
        register_profile(_alt_profile("second"))
        with pytest.raises(LookupError, match="multiple profiles"):
            default_profile()

    def test_no_profiles_registered_raises(self) -> None:
        profiles._PROFILES.clear()
        with pytest.raises(LookupError, match="no profiles registered"):
            default_profile()
