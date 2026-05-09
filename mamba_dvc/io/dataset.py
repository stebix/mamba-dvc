"""Zarr-backed dataset wrapper: ``DvcDataset`` + ``EvaluationPair``.

Lifts a zarr store into a typed object that exposes the reference,
masks, and named deformations (real + synthetic). Materialization is
explicit per call: opening the store reads no voxels.

See ``docs/plans/zarr-interface.md`` §3 for the API contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import zarr
from jaxtyping import Bool, Float32

from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.io.mask import load_mask
from mamba_dvc.io.profiles import (
    StoreProfile,
    default_profile,
    get_profile,
)
from mamba_dvc.io.verify import (
    StoreVerificationError,
    VerificationReport,
    verify,
)
from mamba_dvc.io.volume import load_volume
from mamba_dvc.types import VoxelSpacing

__all__ = [
    "NO_MASK",
    "DeformationEntry",
    "DvcDataset",
    "EvaluationPair",
    "NoMaskSentinel",
]


DeformationKind = Literal["real", "synthetic"]


class NoMaskSentinel:
    """Type of the :data:`NO_MASK` singleton.

    Distinct from ``None`` (which means "use profile default") and
    distinct from ``str`` (which names a mask). Pass ``NO_MASK`` to
    :meth:`DvcDataset.load_pair` to skip the mask entirely.
    """

    _instance: ClassVar[NoMaskSentinel | None] = None

    def __new__(cls) -> NoMaskSentinel:
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return the canonical name of the sentinel."""
        return "NO_MASK"


NO_MASK = NoMaskSentinel()


MaskSelector = str | NoMaskSentinel | None


@dataclass(frozen=True)
class DeformationEntry:
    """One deformed-image instance inside a :class:`DvcDataset`.

    ``flow`` is ``None`` for entries under ``iterations/`` (real
    experimental deformations) and present for entries under
    ``synthetic/``. ``kind`` is derived from the parent group, never
    from per-entry attributes.
    """

    name: str
    image: zarr.Array[Any]
    flow: zarr.Array[Any] | None
    kind: DeformationKind


@dataclass(frozen=True)
class EvaluationPair:
    """Materialized inputs ready for one ``correlate()`` call.

    Pass straight to ``correlate()`` / ``correlate_multi_gpu()`` and
    pipe the result + ``self`` into
    ``mamba_dvc.validate.known_fields.evaluate_pair``.
    """

    reference: Float32[np.ndarray, "z y x"]
    deformed: Float32[np.ndarray, "z y x"]
    mask: Bool[np.ndarray, "z y x"] | None
    gt_field: GroundTruthField | None
    name: str
    kind: DeformationKind
    spacing: VoxelSpacing | None


class DvcDataset:
    """A zarr-backed (reference, masks, deformations) bundle.

    Lazy by construction. Opening reads no voxels; materialization is
    explicit per call. Use :meth:`load_pair` to assemble inputs for a
    single ``correlate()`` invocation.
    """

    _root: zarr.Group
    profile: StoreProfile
    manifest: StoreManifest | None
    reference: zarr.Array[Any]
    masks: Mapping[str, zarr.Array[Any]]
    deformations: Mapping[str, DeformationEntry]
    spacing: VoxelSpacing | None
    volume_shape: tuple[int, int, int]
    verification_report: VerificationReport

    def __init__(
        self,
        *,
        root: zarr.Group,
        profile: StoreProfile,
        manifest: StoreManifest | None,
        reference: zarr.Array[Any],
        masks: Mapping[str, zarr.Array[Any]],
        deformations: Mapping[str, DeformationEntry],
        spacing: VoxelSpacing | None,
        volume_shape: tuple[int, int, int],
        verification_report: VerificationReport,
    ) -> None:
        self._root = root
        self.profile = profile
        self.manifest = manifest
        self.reference = reference
        self.masks = masks
        self.deformations = deformations
        self.spacing = spacing
        self.volume_shape = volume_shape
        self.verification_report = verification_report

    # ----------------------------------------------------------- factory

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        manifest: StoreManifest | None = None,
        strict: bool = True,
    ) -> DvcDataset:
        """Open a zarr store, resolve its profile, and verify.

        Parameters
        ----------
        path
            Path to the zarr store on disk.
        manifest
            Optional in-memory manifest. ``None`` triggers
            :meth:`StoreManifest.discover` (sidecar YAML + zarr root
            attrs); pass an explicit instance to bypass discovery.
        strict
            ``True`` (default) raises
            :class:`StoreVerificationError` on any verifier error.
            ``False`` returns the dataset anyway and exposes the
            report on :attr:`verification_report` for diagnostic use.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        StoreVerificationError
            If ``strict=True`` and the store does not conform to the
            resolved profile + manifest.
        """
        if not path.exists():
            raise FileNotFoundError(f"zarr store not found: {path}")

        if manifest is None:
            manifest = StoreManifest.discover(path)

        profile = _resolve_profile(manifest)

        root = zarr.open_group(str(path), mode="r")
        report = verify(profile, manifest, root)
        if strict and not report.ok:
            raise StoreVerificationError(report)

        reference = _expect_array(root, f"{profile.base_group}/{profile.reference_name}")
        volume_shape = (
            int(reference.shape[0]),
            int(reference.shape[1]),
            int(reference.shape[2]),
        )

        masks = _resolve_masks(root, profile, manifest)
        deformations = _resolve_deformations(root, profile, manifest)
        spacing = manifest.spacing if manifest is not None else None

        return cls(
            root=root,
            profile=profile,
            manifest=manifest,
            reference=reference,
            masks=masks,
            deformations=deformations,
            spacing=spacing,
            volume_shape=volume_shape,
            verification_report=report,
        )

    # ----------------------------------------------------------- listing

    def list_all(self) -> list[str]:
        """Names of every deformation (real + synthetic), sorted."""
        return sorted(self.deformations.keys())

    def list_real(self) -> list[str]:
        """Names of real-deformation entries, sorted."""
        return sorted(
            name for name, entry in self.deformations.items() if entry.kind == "real"
        )

    def list_synthetic(self) -> list[str]:
        """Names of synthetic-deformation entries, sorted."""
        return sorted(
            name for name, entry in self.deformations.items() if entry.kind == "synthetic"
        )

    def list_masks(self) -> list[str]:
        """Names of available mask arrays, sorted."""
        return sorted(self.masks.keys())

    # ------------------------------------------------------ materializers

    def load_reference(
        self, *, dry_shape: tuple[int, int, int] | None = None
    ) -> Float32[np.ndarray, "z y x"]:
        """Materialize the reference volume as float32."""
        return load_volume(self.reference, dry_shape=dry_shape, as_float32=True)

    def load_mask(
        self,
        name: str | None = None,
        *,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> Bool[np.ndarray, "z y x"]:
        """Load a mask by name; ``None`` picks the profile's default.

        Raises
        ------
        KeyError
            If ``name`` is not a known mask.
        ValueError
            If ``name`` is ``None`` and the profile has no default
            mask.
        """
        resolved_name = self._resolve_mask_name(name)
        if resolved_name is None:
            raise ValueError("no mask name supplied and profile has no default_mask")
        if resolved_name not in self.masks:
            raise KeyError(f"unknown mask: {resolved_name!r}")
        return load_mask(self.masks[resolved_name], dry_shape=dry_shape)

    def load_pair(
        self,
        deformation: str,
        *,
        mask: MaskSelector = None,
        dry_shape: tuple[int, int, int] | None = None,
    ) -> EvaluationPair:
        """Materialize ref + def + mask + GT for one ``correlate()`` call.

        Parameters
        ----------
        deformation
            Name of an entry from :meth:`list_all` (real or
            synthetic).
        mask
            ``None`` (default) picks the profile's default mask; a
            string names a specific mask; :data:`NO_MASK` skips the
            mask entirely.
        dry_shape
            Optional centered-subblock size, applied identically to
            reference, deformed, mask, and GT field.

        Raises
        ------
        KeyError
            If ``deformation`` is unknown.
        """
        if deformation not in self.deformations:
            raise KeyError(f"unknown deformation: {deformation!r}")
        entry = self.deformations[deformation]

        reference = self.load_reference(dry_shape=dry_shape)

        mask_array: Bool[np.ndarray, "z y x"] | None
        if isinstance(mask, NoMaskSentinel):
            mask_array = None
        else:
            resolved_mask = self._resolve_mask_name(mask)
            if resolved_mask is None:
                mask_array = None
            else:
                if resolved_mask not in self.masks:
                    raise KeyError(f"unknown mask: {resolved_mask!r}")
                mask_array = load_mask(
                    self.masks[resolved_mask],
                    dry_shape=dry_shape,
                    expected_shape=tuple(reference.shape),  # type: ignore[arg-type]
                )

        deformed = load_volume(entry.image, dry_shape=dry_shape, as_float32=True)

        gt_field: GroundTruthField | None = None
        if entry.flow is not None:
            flow_axis_order = self._flow_axis_order()
            flow_convention = self._flow_convention()
            gt_field = GroundTruthField.from_zarr(
                entry.flow,
                axis_order=flow_axis_order,
                convention=flow_convention,
                dry_shape=dry_shape,
            )

        return EvaluationPair(
            reference=reference,
            deformed=deformed,
            mask=mask_array,
            gt_field=gt_field,
            name=entry.name,
            kind=entry.kind,
            spacing=self.spacing,
        )

    # --------------------------------------------------------- internals

    def _resolve_mask_name(self, name: str | None) -> str | None:
        if name is not None:
            return name
        if self.manifest is not None and self.manifest.base.default_mask is not None:
            return self.manifest.base.default_mask
        return self.profile.default_mask

    def _flow_axis_order(self) -> Any:
        if self.manifest is not None and self.manifest.synthetic.flow.axis_order is not None:
            return self.manifest.synthetic.flow.axis_order
        return self.profile.flow_defaults.axis_order

    def _flow_convention(self) -> Any:
        if self.manifest is not None and self.manifest.synthetic.flow.convention is not None:
            return self.manifest.synthetic.flow.convention
        return self.profile.flow_defaults.convention


# --------------------------------------------------------------- helpers


def _resolve_profile(manifest: StoreManifest | None) -> StoreProfile:
    if manifest is not None and manifest.profile_name is not None:
        return get_profile(manifest.profile_name)
    return default_profile()


def _expect_array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    node = root[path]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"{path}: expected array, got {type(node).__name__}")
    return node


def _expect_group(root: zarr.Group, path: str) -> zarr.Group:
    node = root[path]
    if not isinstance(node, zarr.Group):
        raise TypeError(f"{path}: expected group, got {type(node).__name__}")
    return node


def _resolve_masks(
    root: zarr.Group, profile: StoreProfile, manifest: StoreManifest | None
) -> dict[str, zarr.Array[Any]]:
    base = _expect_group(root, profile.base_group)
    discovered = sorted(
        n
        for n, node in base.members()
        if isinstance(node, zarr.Array) and n != profile.reference_name
    )
    if manifest is not None and manifest.base.masks != "discover":
        selected = [n for n in manifest.base.masks if n in discovered]
    else:
        selected = discovered

    out: dict[str, zarr.Array[Any]] = {}
    for name in selected:
        node = base[name]
        if isinstance(node, zarr.Array):
            out[name] = node
    return out


def _resolve_deformations(
    root: zarr.Group, profile: StoreProfile, manifest: StoreManifest | None
) -> dict[str, DeformationEntry]:
    out: dict[str, DeformationEntry] = {}
    real = _expect_group(root, profile.real_group)
    real_discovered = sorted(n for n, node in real.members() if isinstance(node, zarr.Group))
    real_selected = _select_entries(
        manifest_entries=(manifest.iterations.entries if manifest is not None else "discover"),
        discovered=real_discovered,
    )
    for name in real_selected:
        entry_group = _expect_group(real, name)
        image = _expect_array(entry_group, profile.deformed_name)
        out[name] = DeformationEntry(name=name, image=image, flow=None, kind="real")

    synthetic = _expect_group(root, profile.synthetic_group)
    synthetic_discovered = sorted(
        n for n, node in synthetic.members() if isinstance(node, zarr.Group)
    )
    synthetic_selected = _select_entries(
        manifest_entries=(manifest.synthetic.entries if manifest is not None else "discover"),
        discovered=synthetic_discovered,
    )
    for name in synthetic_selected:
        entry_group = _expect_group(synthetic, name)
        image = _expect_array(entry_group, profile.deformed_name)
        flow = _expect_array(entry_group, profile.flow_name)
        out[name] = DeformationEntry(name=name, image=image, flow=flow, kind="synthetic")

    return out


def _select_entries(manifest_entries: Any, discovered: list[str]) -> list[str]:
    if manifest_entries == "discover":
        return discovered
    return [n for n in manifest_entries if n in discovered]
