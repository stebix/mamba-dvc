"""Tests for ``DisplacementField.save_npz`` / ``load_npz`` round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mamba_dvc.types import DisplacementField, POIStatus


def _make_field(n: int = 7) -> DisplacementField:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 50.0, size=(n, 3)).astype(np.float32)
    displacements = rng.standard_normal((n, 3)).astype(np.float32)
    status = np.zeros(n, dtype=np.uint8)
    status[1] = POIStatus.MASKED
    status[3] = POIStatus.OUTLIER
    valid = status == POIStatus.OK
    confidence = np.where(valid, rng.uniform(0.5, 1.0, size=n), 0.0).astype(np.float32)
    displacements[~valid] = 0.0
    return DisplacementField(
        positions=positions,
        displacements=displacements,
        valid=valid,
        confidence=confidence,
        status=status,
        grid_shape=(7, 1, 1),
        spacing=(48, 48, 48),
        window=(96, 96, 96),
    )


class TestSaveLoadNpzRoundTrip:
    def test_round_trip_preserves_arrays_and_scalars(self, tmp_path: Path) -> None:
        field = _make_field()
        out = tmp_path / "field.npz"
        field.save_npz(out)
        assert out.exists()

        loaded = DisplacementField.load_npz(out)
        np.testing.assert_array_equal(loaded.positions, field.positions)
        np.testing.assert_array_equal(loaded.displacements, field.displacements)
        np.testing.assert_array_equal(loaded.valid, field.valid)
        np.testing.assert_array_equal(loaded.confidence, field.confidence)
        np.testing.assert_array_equal(loaded.status, field.status)
        assert loaded.grid_shape == field.grid_shape
        assert loaded.spacing == field.spacing
        assert loaded.window == field.window

    def test_load_coerces_dtypes(self, tmp_path: Path) -> None:
        field = _make_field()
        out = tmp_path / "field.npz"
        field.save_npz(out)
        loaded = DisplacementField.load_npz(out)
        assert loaded.positions.dtype == np.float32
        assert loaded.displacements.dtype == np.float32
        assert loaded.valid.dtype == np.bool_
        assert loaded.confidence.dtype == np.float32
        assert loaded.status.dtype == np.uint8
        assert all(isinstance(v, int) for v in loaded.grid_shape)

    def test_savez_appends_suffix(self, tmp_path: Path) -> None:
        field = _make_field()
        stem = tmp_path / "field"
        field.save_npz(stem)
        assert (tmp_path / "field.npz").exists()

    def test_missing_entry_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "partial.npz"
        np.savez(out, positions=np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(KeyError):
            DisplacementField.load_npz(out)
