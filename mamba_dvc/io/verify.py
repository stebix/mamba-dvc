"""Profile + manifest verifier for zarr stores.

Walks the merged (profile-defaults + manifest-overrides) view against
the actual zarr store and emits one :class:`VerificationReport.errors`
entry per violation. ``DvcDataset.open(strict=True)`` raises
:class:`StoreVerificationError` on a non-empty error tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.io.profiles import StoreProfile

__all__ = [
    "StoreVerificationError",
    "VerificationReport",
    "verify",
]


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of one :func:`verify` call.

    Parameters
    ----------
    ok
        ``True`` iff ``errors`` is empty. ``warnings`` does not affect
        ``ok``.
    errors
        One human-readable string per detected violation.
    warnings
        Non-fatal observations (e.g. unexpected extra entries that the
        verifier merely surfaces).
    """

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


class StoreVerificationError(ValueError):
    """Raised by ``DvcDataset.open(strict=True)`` on a failing report."""

    def __init__(self, report: VerificationReport) -> None:
        joined = "\n  - ".join(report.errors)
        super().__init__(f"store verification failed:\n  - {joined}")
        self.report = report


def verify(
    profile: StoreProfile,
    manifest: StoreManifest | None,
    root: zarr.Group,
) -> VerificationReport:
    """Validate ``root`` against ``profile`` (with optional ``manifest``).

    The verifier is structural: it checks group/array existence, dtypes,
    spatial-shape consistency relative to the reference, and the flow
    array's vector-axis size and position. It does **not** read any
    voxels.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1) Top-level groups.
    base = _expect_group(root, profile.base_group, errors)
    real = _expect_group(root, profile.real_group, errors)
    synthetic = _expect_group(root, profile.synthetic_group, errors)
    if base is None or real is None or synthetic is None:
        return VerificationReport(False, tuple(errors), tuple(warnings))

    # 2) Reference array.
    reference_path = f"{profile.base_group}/{profile.reference_name}"
    reference = _expect_array(base, profile.reference_name, reference_path, errors)
    if reference is None:
        return VerificationReport(False, tuple(errors), tuple(warnings))

    _check_dtype(reference, reference_path, profile.dtype_invariants.reference, errors)
    _check_ndim(reference, reference_path, 3, errors)

    reference_shape: tuple[int, int, int] | None = None
    if reference.ndim == 3:
        reference_shape = (
            int(reference.shape[0]),
            int(reference.shape[1]),
            int(reference.shape[2]),
        )

    # 3) Masks.
    base_children = sorted(_iter_array_names(base))
    discovered_masks = [n for n in base_children if n != profile.reference_name]
    selected_masks = _resolve_entries(
        manifest_entries=(manifest.base.masks if manifest is not None else "discover"),
        discovered=discovered_masks,
        section_label="base.masks",
        errors=errors,
    )
    for mask_name in selected_masks:
        mask_path = f"{profile.base_group}/{mask_name}"
        mask = _expect_array(base, mask_name, mask_path, errors)
        if mask is None:
            continue
        _check_dtype(mask, mask_path, profile.dtype_invariants.mask, errors)
        _check_spatial_shape(mask, mask_path, reference_shape, errors)

    default_mask_name: str | None = (
        manifest.base.default_mask
        if manifest is not None and manifest.base.default_mask is not None
        else profile.default_mask
    )
    # The default may exist on disk even if a manifest subset excludes
    # it from the selected list, so check disk presence directly.
    if (
        default_mask_name is not None
        and default_mask_name not in selected_masks
        and default_mask_name not in discovered_masks
    ):
        errors.append(
            f"default_mask {default_mask_name!r} not found under {profile.base_group}/"
        )

    # 4) Real-deformation entries.
    real_discovered = sorted(_iter_subgroup_names(real))
    real_selected = _resolve_entries(
        manifest_entries=(manifest.iterations.entries if manifest is not None else "discover"),
        discovered=real_discovered,
        section_label=f"{profile.real_group}.entries",
        errors=errors,
    )
    for entry_name in real_selected:
        entry_path = f"{profile.real_group}/{entry_name}"
        entry = _expect_group(root, entry_path, errors)
        if entry is None:
            continue
        deformed_path = f"{entry_path}/{profile.deformed_name}"
        deformed = _expect_array(entry, profile.deformed_name, deformed_path, errors)
        if deformed is None:
            continue
        _check_dtype(deformed, deformed_path, profile.dtype_invariants.deformed, errors)
        _check_spatial_shape(deformed, deformed_path, reference_shape, errors)

    # 5) Synthetic-deformation entries.
    synthetic_discovered = sorted(_iter_subgroup_names(synthetic))
    synthetic_selected = _resolve_entries(
        manifest_entries=(manifest.synthetic.entries if manifest is not None else "discover"),
        discovered=synthetic_discovered,
        section_label=f"{profile.synthetic_group}.entries",
        errors=errors,
    )

    flow_axis_order = (
        manifest.synthetic.flow.axis_order
        if manifest is not None and manifest.synthetic.flow.axis_order is not None
        else profile.flow_defaults.axis_order
    )

    for entry_name in synthetic_selected:
        entry_path = f"{profile.synthetic_group}/{entry_name}"
        entry = _expect_group(root, entry_path, errors)
        if entry is None:
            continue
        deformed_path = f"{entry_path}/{profile.deformed_name}"
        deformed = _expect_array(entry, profile.deformed_name, deformed_path, errors)
        if deformed is not None:
            _check_dtype(deformed, deformed_path, profile.dtype_invariants.deformed, errors)
            _check_spatial_shape(deformed, deformed_path, reference_shape, errors)

        flow_path = f"{entry_path}/{profile.flow_name}"
        flow = _expect_array(entry, profile.flow_name, flow_path, errors)
        if flow is None:
            continue
        _check_dtype(flow, flow_path, profile.dtype_invariants.flow, errors)
        _check_flow_axis_order(flow, flow_path, flow_axis_order, errors)
        _check_flow_spatial_shape(flow, flow_path, flow_axis_order, reference_shape, errors)

    return VerificationReport(
        ok=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------- group helpers


def _expect_group(parent: zarr.Group, name: str, errors: list[str]) -> zarr.Group | None:
    try:
        node = parent[name]
    except KeyError:
        errors.append(f"expected group {name!r} at {_describe(parent)}")
        return None
    if not isinstance(node, zarr.Group):
        errors.append(f"{name!r} is not a group (got {type(node).__name__})")
        return None
    return node


def _expect_array(
    parent: zarr.Group, name: str, full_path: str, errors: list[str]
) -> zarr.Array[Any] | None:
    try:
        node = parent[name]
    except KeyError:
        errors.append(f"{full_path}: missing")
        return None
    if not isinstance(node, zarr.Array):
        errors.append(f"{full_path}: expected array, got {type(node).__name__}")
        return None
    return node


def _iter_array_names(group: zarr.Group) -> list[str]:
    return [name for name, node in group.members() if isinstance(node, zarr.Array)]


def _iter_subgroup_names(group: zarr.Group) -> list[str]:
    return [name for name, node in group.members() if isinstance(node, zarr.Group)]


def _describe(group: zarr.Group) -> str:
    """Return a short string identifier for a zarr group."""
    return f"group {group.path or '<root>'!r}"


# ------------------------------------------------------------- check helpers


def _check_dtype(array: zarr.Array[Any], path: str, expected: str, errors: list[str]) -> None:
    expected_dtype = np.dtype(expected)
    actual_dtype = np.dtype(array.dtype)
    if actual_dtype != expected_dtype:
        errors.append(f"{path}: expected {expected_dtype.name}, got {actual_dtype.name}")


def _check_ndim(array: zarr.Array[Any], path: str, expected: int, errors: list[str]) -> None:
    if array.ndim != expected:
        errors.append(f"{path}: expected ndim={expected}, got {array.ndim}")


def _check_spatial_shape(
    array: zarr.Array[Any],
    path: str,
    reference_shape: tuple[int, int, int] | None,
    errors: list[str],
) -> None:
    if reference_shape is None or array.ndim != 3:
        if array.ndim != 3:
            errors.append(f"{path}: expected ndim=3, got {array.ndim}")
        return
    actual = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if actual != reference_shape:
        errors.append(f"{path}: shape {actual} != reference {reference_shape}")


def _check_flow_axis_order(
    flow: zarr.Array[Any], path: str, axis_order: str, errors: list[str]
) -> None:
    if flow.ndim != 4:
        errors.append(f"{path}: expected ndim=4 for flow, got {flow.ndim}")
        return
    if axis_order == "3_zyx":
        vector_axis = int(flow.shape[0])
        location = "leading"
    else:
        vector_axis = int(flow.shape[3])
        location = "trailing"
    if vector_axis != 3:
        errors.append(
            f"{path}: expected {location} axis size 3 (axis_order={axis_order}), "
            f"got {vector_axis}"
        )


def _check_flow_spatial_shape(
    flow: zarr.Array[Any],
    path: str,
    axis_order: str,
    reference_shape: tuple[int, int, int] | None,
    errors: list[str],
) -> None:
    if reference_shape is None or flow.ndim != 4:
        return
    if axis_order == "3_zyx":
        spatial = (int(flow.shape[1]), int(flow.shape[2]), int(flow.shape[3]))
    else:
        spatial = (int(flow.shape[0]), int(flow.shape[1]), int(flow.shape[2]))
    if spatial != reference_shape:
        errors.append(f"{path}: spatial shape {spatial} != reference {reference_shape}")


# --------------------------------------------------------- entries selection


def _resolve_entries(
    manifest_entries: Any,
    discovered: list[str],
    section_label: str,
    errors: list[str],
) -> list[str]:
    """Return the final entry list, surfacing missing manifest names as errors."""
    if manifest_entries == "discover":
        return discovered
    requested = list(manifest_entries)
    missing = [name for name in requested if name not in discovered]
    for name in missing:
        errors.append(f"{section_label}: manifest references {name!r}, not in store")
    return [name for name in requested if name in discovered]
