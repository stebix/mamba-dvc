"""Named structural schemas (``StoreProfile``) for zarr stores.

A profile encodes one collaborator's "this is how we lay out our zarr
stores" agreement: group names, array names, default mask, default
flow metadata, and dtype invariants. v1 ships a single registered
profile (``bone_screw_synchrotron_v1``); future collaborators add
their own with :func:`register_profile`.

See ``docs/plans/zarr-interface.md`` §5a for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mamba_dvc.io.field import FieldAxisOrder, FieldConvention

__all__ = [
    "BONE_SCREW_SYNCHROTRON_V1",
    "DtypeInvariants",
    "FlowSpec",
    "StoreProfile",
    "default_profile",
    "get_profile",
    "list_profiles",
    "register_profile",
]


VectorOrder = Literal["dz_dy_dx", "dx_dy_dz"]
FieldUnits = Literal["voxel", "nm", "um", "mm"]


@dataclass(frozen=True)
class FlowSpec:
    """Default on-disk metadata for synthetic flow arrays."""

    axis_order: FieldAxisOrder
    vector_order: VectorOrder
    convention: FieldConvention
    units: FieldUnits


@dataclass(frozen=True)
class DtypeInvariants:
    """Required NumPy dtype names for each well-known array role."""

    reference: str
    deformed: str
    mask: str
    flow: str


@dataclass(frozen=True)
class StoreProfile:
    """Named structural schema for a zarr store.

    Parameters
    ----------
    name
        Unique profile identifier; used by :class:`StoreManifest` to
        select a profile by name.
    base_group
        Name of the top-level group holding the reference and masks.
    reference_name
        Array name of the reference inside :attr:`base_group`.
    default_mask
        Default mask name to use when the caller does not specify one.
        ``None`` if the layout has no mask.
    real_group
        Name of the top-level group holding real-deformation entries
        (no synthetic ground truth).
    synthetic_group
        Name of the top-level group holding synthetic-deformation
        entries (with ground-truth flow).
    deformed_name
        Array name of the deformed image inside each entry under
        :attr:`real_group` and :attr:`synthetic_group`.
    flow_name
        Array name of the ground-truth flow inside each entry under
        :attr:`synthetic_group`.
    flow_defaults
        On-disk flow metadata defaults, overridable by a manifest.
    dtype_invariants
        Required dtype for each well-known array role.
    """

    name: str
    base_group: str
    reference_name: str
    default_mask: str | None
    real_group: str
    synthetic_group: str
    deformed_name: str
    flow_name: str
    flow_defaults: FlowSpec
    dtype_invariants: DtypeInvariants


_PROFILES: dict[str, StoreProfile] = {}


def register_profile(profile: StoreProfile) -> None:
    """Register ``profile`` under its name.

    Raises
    ------
    ValueError
        If a profile with the same name is already registered.
    """
    if profile.name in _PROFILES:
        raise ValueError(f"profile already registered: {profile.name!r}")
    _PROFILES[profile.name] = profile


def get_profile(name: str) -> StoreProfile:
    """Look up a registered profile by name.

    Raises
    ------
    KeyError
        If no profile is registered under ``name``.
    """
    if name not in _PROFILES:
        raise KeyError(f"no registered profile named {name!r}")
    return _PROFILES[name]


def list_profiles() -> list[str]:
    """Return the names of all registered profiles in registration order."""
    return list(_PROFILES.keys())


def default_profile() -> StoreProfile:
    """Return the sole registered profile.

    Raises
    ------
    LookupError
        If zero or more than one profile is registered. In the
        multi-profile case, callers must disambiguate via the manifest's
        ``store_format`` field.
    """
    if len(_PROFILES) == 0:
        raise LookupError("no profiles registered")
    if len(_PROFILES) > 1:
        names = ", ".join(sorted(_PROFILES.keys()))
        raise LookupError(
            f"multiple profiles registered ({names}); "
            f"select one via StoreManifest.profile_name"
        )
    return next(iter(_PROFILES.values()))


# bone_screw_synchrotron_v1: rat-bone µCT layout with a screw implant.
# Sign convention is unverified; first synthetic-pair end-to-end run
# will reveal whether the stored flow matches `correlate()`'s output
# without sign-flipping. If not, override with `convention: push_forward`
# in the per-store manifest.
# TODO(verify): confirm pull-back vs push-forward against a known shift.
BONE_SCREW_SYNCHROTRON_V1 = StoreProfile(
    name="bone_screw_synchrotron_v1",
    base_group="base",
    reference_name="volume0",
    default_mask="mask",
    real_group="iterations",
    synthetic_group="synthetic",
    deformed_name="volume1",
    flow_name="flow",
    flow_defaults=FlowSpec(
        axis_order="3_zyx",
        vector_order="dz_dy_dx",
        convention="pull_back",
        units="voxel",
    ),
    dtype_invariants=DtypeInvariants(
        reference="float32",
        deformed="float32",
        mask="bool",
        flow="float32",
    ),
)
register_profile(BONE_SCREW_SYNCHROTRON_V1)
