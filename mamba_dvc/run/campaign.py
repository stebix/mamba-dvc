"""Read-side helpers for campaign output directories.

Campaign runs persist results as ``<store>/<deformation>/<variant_id>.npz``
plus a matching ``.json`` sidecar, with one row per job in
``manifest.jsonl``. This module provides the inverse of that write
pipeline: filter the manifest by ``(variant_id, store, deformation)``
and materialize the resulting :class:`DisplacementField` plus its
sidecar metadata.

The loader deliberately refuses ambiguity. If zero or more than one
manifest row matches the requested filter, it raises with the list of
alternatives instead of silently picking the first — the canonical
"plot one variant" use case wants exactly one match, and the canonical
"plot many variants" use case wants the caller to drive the loop
explicitly with concrete variant ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mamba_dvc.types import DisplacementField

__all__ = ["load_campaign_field"]


_MANIFEST_NAME = "manifest.jsonl"
_OK_STATUS = "ok"


def load_campaign_field(
    root: Path | str,
    *,
    variant_id: str | None = None,
    store: str | None = None,
    deformation: str | None = None,
) -> tuple[DisplacementField, dict[str, Any]]:
    """Load one persisted :class:`DisplacementField` from a campaign directory.

    Parameters
    ----------
    root
        Campaign directory. Must contain ``manifest.jsonl`` written by
        :func:`mamba_dvc.run.batch.run_batch`.
    variant_id, store, deformation
        Optional filters on the corresponding manifest columns. Each
        provided filter must match exactly. Filters compose with AND —
        the row's ``status`` must additionally be ``"ok"`` (failed jobs
        have no ``.npz`` and are therefore not loadable).

    Returns
    -------
    tuple
        ``(field, sidecar)``. ``field`` is the
        :class:`DisplacementField` reconstructed via
        :meth:`DisplacementField.load_npz`. ``sidecar`` is the parsed
        JSON sidecar dict, carrying the run's resolved variant params,
        summary metrics, and provenance — useful for annotating plots
        without re-reading the manifest.

    Raises
    ------
    FileNotFoundError
        If ``root`` or ``root/manifest.jsonl`` does not exist, or if the
        manifest references an ``.npz`` / ``.json`` file that is missing
        on disk.
    LookupError
        If the filter matches zero rows, or more than one — both states
        require the caller to disambiguate before the function can
        return a single field. The message lists matching variant ids.

    Notes
    -----
    The manifest stores ``npz`` paths relative to ``root`` and built
    with forward slashes. On Windows, :class:`pathlib.Path` accepts
    either separator, so the lookup is platform-agnostic.

    The sidecar path is derived from the manifest's ``npz`` path by
    swapping the extension; the writer side
    (:func:`mamba_dvc.run.batch._write_sidecar`) maintains that pairing
    invariant.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"campaign root does not exist: {root_path}")

    manifest_path = root_path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    matches: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest_path}:{lineno}: malformed manifest row ({exc})"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{manifest_path}:{lineno}: manifest row is not a JSON object"
                )
            if row.get("status") != _OK_STATUS:
                continue
            if variant_id is not None and row.get("variant_id") != variant_id:
                continue
            if store is not None and row.get("store") != store:
                continue
            if deformation is not None and row.get("deformation") != deformation:
                continue
            matches.append(row)

    if not matches:
        raise LookupError(
            f"no manifest rows in {manifest_path} match filter "
            f"(variant_id={variant_id!r}, store={store!r}, deformation={deformation!r}) "
            "with status='ok'"
        )
    if len(matches) > 1:
        sample = ", ".join(
            f"({m.get('store')!r}, {m.get('deformation')!r}, {m.get('variant_id')!r})"
            for m in matches[:5]
        )
        suffix = "" if len(matches) <= 5 else f", ... (+{len(matches) - 5} more)"
        raise LookupError(
            f"filter (variant_id={variant_id!r}, store={store!r}, "
            f"deformation={deformation!r}) matches {len(matches)} rows: "
            f"{sample}{suffix} — disambiguate by passing more filters"
        )

    row = matches[0]
    rel_npz = row.get("npz")
    if not isinstance(rel_npz, str):
        raise ValueError(f"matched manifest row has no 'npz' path: {row.get('variant_id')!r}")

    npz_path = root_path / rel_npz
    if not npz_path.exists():
        raise FileNotFoundError(f"manifest row references missing .npz: {npz_path}")
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"manifest row references missing sidecar: {sidecar_path}")

    field = DisplacementField.load_npz(npz_path)
    with sidecar_path.open("r", encoding="utf-8") as fh:
        sidecar = json.load(fh)
    if not isinstance(sidecar, dict):
        raise ValueError(f"sidecar is not a JSON object: {sidecar_path}")

    return field, sidecar
