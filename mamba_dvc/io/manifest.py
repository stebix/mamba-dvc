"""Optional sidecar manifest that overrides a :class:`StoreProfile`.

A manifest never re-defines structural schema (that is the profile's
job). It carries:

1. An optional ``profile_name`` to pick among multiple registered
   profiles.
2. Per-scan ``spacing`` (which the profile cannot encode because it
   varies between scans).
3. Per-section overrides: an alternate default mask, an explicit
   subset of mask / iteration / synthetic entries, and a flow override
   for whichever convention or axis order the on-disk data actually
   uses.

Discovery: :meth:`StoreManifest.discover` checks for a sibling YAML
file (``<store>.yaml``) and the embedded ``root.attrs["dvc_store"]``
key, in that order. If both are present and disagree, an error is
raised on :meth:`DvcDataset.open`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
import zarr

from mamba_dvc.io.field import FieldAxisOrder, FieldConvention
from mamba_dvc.io.profiles import VectorOrder
from mamba_dvc.types import VoxelSpacing

__all__ = [
    "BaseManifest",
    "EntriesManifest",
    "FlowOverride",
    "StoreManifest",
    "SyntheticManifest",
]


# ``"discover"`` is the literal sentinel used in YAML to mean "list every
# child of this group". A list of strings names an explicit subset.
EntriesSpec = Literal["discover"] | list[str]


@dataclass(frozen=True)
class FlowOverride:
    """Per-store overrides for synthetic flow metadata.

    Any ``None`` field falls back to the profile's
    ``flow_defaults`` value.
    """

    axis_order: FieldAxisOrder | None = None
    vector_order: VectorOrder | None = None
    convention: FieldConvention | None = None
    units: str | None = None


@dataclass(frozen=True)
class BaseManifest:
    """Overrides applied to the profile's ``base_group``."""

    default_mask: str | None = None
    masks: EntriesSpec = "discover"


@dataclass(frozen=True)
class EntriesManifest:
    """Subset selector for ``iterations/`` entries."""

    entries: EntriesSpec = "discover"


@dataclass(frozen=True)
class SyntheticManifest:
    """Subset selector + flow overrides for ``synthetic/`` entries."""

    entries: EntriesSpec = "discover"
    flow: FlowOverride = field(default_factory=FlowOverride)


@dataclass(frozen=True)
class StoreManifest:
    """Per-store overrides on top of a :class:`StoreProfile`.

    All fields are optional; an empty manifest (``StoreManifest()``)
    is equivalent to "use the profile defaults exactly".
    """

    version: int = 1
    profile_name: str | None = None
    spacing: VoxelSpacing | None = None
    base: BaseManifest = field(default_factory=BaseManifest)
    iterations: EntriesManifest = field(default_factory=EntriesManifest)
    synthetic: SyntheticManifest = field(default_factory=SyntheticManifest)

    @classmethod
    def discover(cls, store_path: Path) -> StoreManifest | None:
        """Return the manifest for ``store_path`` if one exists.

        Looks for ``<store_path>.yaml`` (sibling sidecar) and
        ``root.attrs["dvc_store"]`` (embedded). When both are present,
        their parsed forms must be equal — otherwise raises
        ``ValueError``. Returns ``None`` if neither source exists.
        """
        sidecar_path = store_path.with_suffix(store_path.suffix + ".yaml")
        from_sidecar: StoreManifest | None = None
        from_attrs: StoreManifest | None = None

        if sidecar_path.exists():
            from_sidecar = cls.from_yaml(sidecar_path)

        if store_path.exists():
            root: zarr.Group | None = None
            try:
                root = zarr.open_group(str(store_path), mode="r")
            except FileNotFoundError, KeyError, ValueError:
                root = None
            if root is not None:
                from_attrs = cls.from_zattrs(root)

        if from_sidecar is not None and from_attrs is not None:
            if from_sidecar != from_attrs:
                raise ValueError(
                    f"manifest mismatch: {sidecar_path} disagrees with "
                    f"root.attrs['dvc_store'] in {store_path}"
                )
            return from_sidecar
        return from_sidecar or from_attrs

    @classmethod
    def from_yaml(cls, path: Path) -> StoreManifest:
        """Parse a manifest YAML file."""
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"manifest at {path} must be a mapping")
        return _parse_mapping(raw)

    @classmethod
    def from_zattrs(cls, root: zarr.Group) -> StoreManifest | None:
        """Read embedded manifest from ``root.attrs['dvc_store']``.

        Returns ``None`` if the key is absent.
        """
        raw = root.attrs.get("dvc_store")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"root.attrs['dvc_store'] must be a mapping, got {type(raw).__name__}"
            )
        return _parse_mapping(raw)

    def to_yaml(self, path: Path) -> None:
        """Write this manifest to a YAML file."""
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(_to_mapping(self), fh, sort_keys=False)


# ----------------------------------------------------------------- parsing


_REQUIRED_VERSION = 1


def _parse_mapping(raw: Mapping[str, Any]) -> StoreManifest:
    """Build a :class:`StoreManifest` from a YAML/attrs mapping."""
    version = raw.get("dvc_store_version", _REQUIRED_VERSION)
    if version != _REQUIRED_VERSION:
        raise ValueError(
            f"unsupported dvc_store_version: {version} "
            f"(this build understands {_REQUIRED_VERSION})"
        )

    profile_name = raw.get("store_format")
    if profile_name is not None and not isinstance(profile_name, str):
        raise ValueError("store_format must be a string")

    spacing = _parse_spacing(raw.get("spacing"))
    base = _parse_base(raw.get("base", {}))
    iterations = _parse_entries(raw.get("iterations", {}))
    synthetic = _parse_synthetic(raw.get("synthetic", {}))

    return StoreManifest(
        version=int(version),
        profile_name=profile_name,
        spacing=spacing,
        base=base,
        iterations=iterations,
        synthetic=synthetic,
    )


def _parse_spacing(raw: Any) -> VoxelSpacing | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("spacing must be a mapping with 'values' and 'unit'")
    values = raw.get("values")
    unit = raw.get("unit", "voxel")
    if not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError("spacing.values must be a 3-element sequence")
    if not isinstance(unit, str):
        raise ValueError("spacing.unit must be a string")
    if unit not in {"voxel", "nm", "um", "mm"}:
        raise ValueError(f"unknown spacing unit: {unit}")
    return VoxelSpacing(
        values=(float(values[0]), float(values[1]), float(values[2])),
        unit=unit,  # type: ignore[arg-type]
    )


def _parse_entries_spec(raw: Any, field_name: str) -> EntriesSpec:
    if raw is None or raw == "discover":
        return "discover"
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return list(raw)
    raise ValueError(f"{field_name} must be 'discover' or a list of strings")


def _parse_base(raw: Any) -> BaseManifest:
    if not isinstance(raw, Mapping):
        raise ValueError("base must be a mapping")
    default_mask = raw.get("default_mask")
    if default_mask is not None and not isinstance(default_mask, str):
        raise ValueError("base.default_mask must be a string")
    return BaseManifest(
        default_mask=default_mask,
        masks=_parse_entries_spec(raw.get("masks"), "base.masks"),
    )


def _parse_entries(raw: Any) -> EntriesManifest:
    if not isinstance(raw, Mapping):
        raise ValueError("iterations must be a mapping")
    return EntriesManifest(
        entries=_parse_entries_spec(raw.get("entries"), "iterations.entries"),
    )


def _parse_synthetic(raw: Any) -> SyntheticManifest:
    if not isinstance(raw, Mapping):
        raise ValueError("synthetic must be a mapping")
    return SyntheticManifest(
        entries=_parse_entries_spec(raw.get("entries"), "synthetic.entries"),
        flow=_parse_flow_override(raw.get("flow", {})),
    )


def _parse_flow_override(raw: Any) -> FlowOverride:
    if not isinstance(raw, Mapping):
        raise ValueError("synthetic.flow must be a mapping")
    return FlowOverride(
        axis_order=_optional_literal(raw.get("axis_order"), {"3_zyx", "zyx_3"}),
        vector_order=_optional_literal(raw.get("vector_order"), {"dz_dy_dx", "dx_dy_dz"}),
        convention=_optional_literal(raw.get("convention"), {"pull_back", "push_forward"}),
        units=raw.get("units"),
    )


def _optional_literal(value: Any, allowed: set[str]) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"expected one of {sorted(allowed)}, got {value!r}")
    return value


# ----------------------------------------------------------------- serialize


def _to_mapping(manifest: StoreManifest) -> dict[str, Any]:
    """Return the YAML-friendly mapping form of ``manifest``."""
    out: dict[str, Any] = {"dvc_store_version": manifest.version}
    if manifest.profile_name is not None:
        out["store_format"] = manifest.profile_name
    if manifest.spacing is not None:
        out["spacing"] = {
            "values": list(manifest.spacing.values),
            "unit": manifest.spacing.unit,
        }
    base_raw: dict[str, Any] = {}
    if manifest.base.default_mask is not None:
        base_raw["default_mask"] = manifest.base.default_mask
    if manifest.base.masks != "discover":
        base_raw["masks"] = list(manifest.base.masks)
    if base_raw:
        out["base"] = base_raw
    if manifest.iterations.entries != "discover":
        out["iterations"] = {"entries": list(manifest.iterations.entries)}
    syn: dict[str, Any] = {}
    if manifest.synthetic.entries != "discover":
        syn["entries"] = list(manifest.synthetic.entries)
    flow_raw: dict[str, Any] = {}
    flow = manifest.synthetic.flow
    if flow.axis_order is not None:
        flow_raw["axis_order"] = flow.axis_order
    if flow.vector_order is not None:
        flow_raw["vector_order"] = flow.vector_order
    if flow.convention is not None:
        flow_raw["convention"] = flow.convention
    if flow.units is not None:
        flow_raw["units"] = flow.units
    if flow_raw:
        syn["flow"] = flow_raw
    if syn:
        out["synthetic"] = syn
    return out
