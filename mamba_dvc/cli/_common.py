"""Shared helpers used by more than one ``mamba-dvc`` subcommand.

Today this is the manifest-discovery layer that ``inspect`` and
``budget`` both need to map a ``--store`` path to a
:class:`mamba_dvc.io.manifest.StoreManifest`. Future subcommands that
take a store path are expected to reuse the same probe order so the
behaviour stays consistent across the CLI.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import zarr

from mamba_dvc.io.manifest import StoreManifest
from mamba_dvc.io.profiles import get_profile

__all__ = ["resolve_manifest"]


def resolve_manifest(
    store_path: Path,
    *,
    manifest_path: Path | None,
    profile_override: str | None,
) -> tuple[StoreManifest | None, str]:
    """Build the manifest to pass to :meth:`DvcDataset.open`.

    Probes the same sources as :meth:`StoreManifest.discover` but
    returns a human-readable label naming which one was used (sidecar
    YAML, embedded ``.zattrs``, explicit ``--manifest`` flag, or
    ``"none"``). When ``--profile`` is set, an empty manifest is
    synthesized (or the discovered one updated) to carry the override.

    Parameters
    ----------
    store_path
        Resolved path to the ``.zarr`` store directory.
    manifest_path
        Explicit ``--manifest`` argument; takes precedence over both
        sidecar YAML and embedded ``.zattrs`` discovery.
    profile_override
        Explicit ``--profile`` argument; merged onto the discovered
        manifest (or used to synthesize an empty one).

    Returns
    -------
    tuple
        ``(manifest, source_label)``. ``manifest`` is ``None`` only when
        no source produced one and ``profile_override`` was not set.
        ``source_label`` is a human-readable string for header rendering.

    Raises
    ------
    KeyError
        If ``profile_override`` does not match a registered profile.
    ValueError
        If a discovered manifest fails to parse.
    """
    discovered: StoreManifest | None = None
    base_source = "none"

    if manifest_path is not None:
        discovered = StoreManifest.from_yaml(manifest_path)
        base_source = f"--manifest {manifest_path.name}"
    else:
        sidecar_path = store_path.with_suffix(store_path.suffix + ".yaml")
        if sidecar_path.exists():
            discovered = StoreManifest.from_yaml(sidecar_path)
            base_source = f"sidecar yaml ({sidecar_path.name})"
        else:
            from_attrs = _read_attrs_manifest(store_path)
            if from_attrs is not None:
                discovered = from_attrs
                base_source = "root .zattrs"

    if profile_override is not None:
        # Surfaces unknown-profile errors before DvcDataset.open runs.
        get_profile(profile_override)
        if discovered is None:
            merged = StoreManifest(profile_name=profile_override)
        else:
            merged = replace(discovered, profile_name=profile_override)
        if base_source == "none":
            return merged, f"--profile {profile_override}"
        return merged, f"--profile {profile_override} (over {base_source})"

    return discovered, base_source


def _read_attrs_manifest(store_path: Path) -> StoreManifest | None:
    """Best-effort read of an embedded ``root.attrs['dvc_store']``."""
    try:
        root = zarr.open_group(str(store_path), mode="r")
    except (FileNotFoundError, KeyError, ValueError):  # fmt: skip
        return None
    return StoreManifest.from_zattrs(root)
