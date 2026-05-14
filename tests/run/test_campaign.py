"""Tests for ``mamba_dvc.run.campaign.load_campaign_field``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mamba_dvc.run.campaign import load_campaign_field
from mamba_dvc.types import DisplacementField, POIStatus


def _build_field(seed: int = 0) -> DisplacementField:
    rng = np.random.default_rng(seed)
    n = 24
    positions = rng.uniform(0, 50, size=(n, 3)).astype(np.float32)
    displacements = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    confidence = rng.uniform(0, 1, size=n).astype(np.float32)
    status = np.where(valid, POIStatus.OK, POIStatus.MASKED).astype(np.uint8)
    return DisplacementField(
        positions=positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=(2, 3, 4),
        spacing=(2, 2, 2),
        window=(4, 4, 4),
    )


def _write_row(
    root: Path,
    *,
    store: str,
    deformation: str,
    variant_id: str,
    status: str = "ok",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Materialize one ``(store, deformation, variant_id)`` campaign entry."""
    rel_npz = f"{store}/{deformation}/{variant_id}.npz"
    npz_path = root / rel_npz
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if status == "ok":
        field = _build_field(seed=hash(variant_id) % (2**32))
        field.save_npz(npz_path)
        sidecar = {
            "campaign": "test",
            "store": store,
            "deformation": deformation,
            "variant_id": variant_id,
            "params_hash": "abc123",
            "summary": {"n_ok": int(field.valid.sum())},
        }
        if extra:
            sidecar.update(extra)
        with npz_path.with_suffix(".json").open("w", encoding="utf-8") as fh:
            json.dump(sidecar, fh)

    return {
        "campaign": "test",
        "store": store,
        "deformation": deformation,
        "kind": "scored",
        "variant_id": variant_id,
        "name": variant_id,
        "params_hash": "abc123",
        "status": status,
        "npz": rel_npz if status == "ok" else None,
    }


def _make_campaign(root: Path, rows: list[dict[str, object]]) -> Path:
    """Write ``manifest.jsonl`` with one row per entry; return ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return root


# --------------------------------------------------------------- happy paths


class TestLoadCampaignField:
    def test_single_match_returns_field_and_sidecar(self, tmp_path: Path):
        rows = [
            _write_row(tmp_path, store="S1", deformation="D1", variant_id="v0"),
        ]
        _make_campaign(tmp_path, rows)
        field, sidecar = load_campaign_field(tmp_path, variant_id="v0")
        assert isinstance(field, DisplacementField)
        assert sidecar["variant_id"] == "v0"
        assert sidecar["store"] == "S1"

    def test_filter_by_store(self, tmp_path: Path):
        rows = [
            _write_row(tmp_path, store="S1", deformation="D1", variant_id="shared"),
            _write_row(tmp_path, store="S2", deformation="D1", variant_id="shared"),
        ]
        _make_campaign(tmp_path, rows)
        _, sidecar = load_campaign_field(tmp_path, variant_id="shared", store="S2")
        assert sidecar["store"] == "S2"

    def test_filter_by_deformation(self, tmp_path: Path):
        rows = [
            _write_row(tmp_path, store="S1", deformation="D1", variant_id="shared"),
            _write_row(tmp_path, store="S1", deformation="D2", variant_id="shared"),
        ]
        _make_campaign(tmp_path, rows)
        _, sidecar = load_campaign_field(tmp_path, variant_id="shared", deformation="D2")
        assert sidecar["deformation"] == "D2"

    def test_filter_by_all_three(self, tmp_path: Path):
        rows = [
            _write_row(tmp_path, store="S1", deformation="D1", variant_id="v0"),
            _write_row(tmp_path, store="S1", deformation="D2", variant_id="v0"),
            _write_row(tmp_path, store="S2", deformation="D1", variant_id="v0"),
        ]
        _make_campaign(tmp_path, rows)
        _, sidecar = load_campaign_field(
            tmp_path, variant_id="v0", store="S2", deformation="D1"
        )
        assert sidecar["store"] == "S2"
        assert sidecar["deformation"] == "D1"

    def test_failed_rows_are_skipped(self, tmp_path: Path):
        # A failed row for v0 in S1 plus an ok row for v0 in S2: filter by
        # variant_id alone resolves to S2 because the failed row is excluded.
        rows = [
            _write_row(
                tmp_path, store="S1", deformation="D1", variant_id="v0", status="failed"
            ),
            _write_row(tmp_path, store="S2", deformation="D1", variant_id="v0"),
        ]
        _make_campaign(tmp_path, rows)
        _, sidecar = load_campaign_field(tmp_path, variant_id="v0")
        assert sidecar["store"] == "S2"


# --------------------------------------------------------------- error paths


class TestLoadCampaignFieldErrors:
    def test_missing_root(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="campaign root"):
            load_campaign_field(tmp_path / "nope", variant_id="v0")

    def test_missing_manifest(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_campaign_field(tmp_path, variant_id="v0")

    def test_zero_matches_raises_lookup_error(self, tmp_path: Path):
        rows = [_write_row(tmp_path, store="S1", deformation="D1", variant_id="v0")]
        _make_campaign(tmp_path, rows)
        with pytest.raises(LookupError, match="no manifest rows"):
            load_campaign_field(tmp_path, variant_id="v_missing")

    def test_multiple_matches_lists_alternatives(self, tmp_path: Path):
        rows = [
            _write_row(tmp_path, store="S1", deformation="D1", variant_id="v0"),
            _write_row(tmp_path, store="S2", deformation="D1", variant_id="v0"),
        ]
        _make_campaign(tmp_path, rows)
        with pytest.raises(LookupError, match="matches 2 rows") as excinfo:
            load_campaign_field(tmp_path, variant_id="v0")
        # The error message should list the ambiguous stores so the caller
        # knows what extra filter to add.
        assert "S1" in str(excinfo.value)
        assert "S2" in str(excinfo.value)

    def test_missing_npz_on_disk_raises(self, tmp_path: Path):
        row = _write_row(tmp_path, store="S1", deformation="D1", variant_id="v0")
        # Manifest points at npz, but delete the file before loading.
        (tmp_path / row["npz"]).unlink()
        _make_campaign(tmp_path, [row])
        with pytest.raises(FileNotFoundError, match=r"missing \.npz"):
            load_campaign_field(tmp_path, variant_id="v0")

    def test_missing_sidecar_on_disk_raises(self, tmp_path: Path):
        row = _write_row(tmp_path, store="S1", deformation="D1", variant_id="v0")
        sidecar_path = (tmp_path / row["npz"]).with_suffix(".json")
        sidecar_path.unlink()
        _make_campaign(tmp_path, [row])
        with pytest.raises(FileNotFoundError, match="missing sidecar"):
            load_campaign_field(tmp_path, variant_id="v0")

    def test_malformed_manifest_line_raises(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        tmp_path.mkdir(parents=True, exist_ok=True)
        manifest.write_text("not-json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed manifest row"):
            load_campaign_field(tmp_path, variant_id="v0")

    def test_blank_manifest_lines_are_ignored(self, tmp_path: Path):
        rows = [_write_row(tmp_path, store="S1", deformation="D1", variant_id="v0")]
        manifest = tmp_path / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(json.dumps(rows[0]) + "\n")
            fh.write("\n")
        field, _ = load_campaign_field(tmp_path, variant_id="v0")
        assert isinstance(field, DisplacementField)
