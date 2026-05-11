"""Tests for ``mamba_dvc.validate.known_fields``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from mamba_dvc.io.dataset import DvcDataset, EvaluationPair
from mamba_dvc.io.field import GroundTruthField
from mamba_dvc.io.profiles import BONE_SCREW_SYNCHROTRON_V1
from mamba_dvc.types import DisplacementField, POIStatus
from mamba_dvc.validate.known_fields import evaluate_pair, run_and_evaluate, sweep
from mamba_dvc.validate.synthetic import rigid_shift


def _make_gt_field(
    shape: tuple[int, int, int], shift: tuple[float, float, float]
) -> GroundTruthField:
    """Build a constant-shift GroundTruthField sampled on a grid."""
    field_fn = rigid_shift(shift)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    sampled = field_fn(coords).reshape((*shape, 3)).astype(np.float32)
    return GroundTruthField(sampled, convention="pull_back")


def _make_field(
    positions: np.ndarray,
    displacements: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> DisplacementField:
    n = positions.shape[0]
    if confidence is None:
        confidence = np.ones(n, dtype=np.float32)
    if valid is None:
        valid = np.ones(n, dtype=np.bool_)
    status = np.where(valid, POIStatus.OK, POIStatus.OUTLIER).astype(np.uint8)
    return DisplacementField(
        positions=positions.astype(np.float32),
        displacements=displacements.astype(np.float32),
        valid=valid.astype(np.bool_),
        confidence=confidence.astype(np.float32),
        status=status,
        grid_shape=(n, 1, 1),
        spacing=(1, 1, 1),
        window=(3, 3, 3),
    )


def _eval_pair(
    *,
    gt_shape: tuple[int, int, int],
    shift: tuple[float, float, float],
    mask: np.ndarray | None = None,
    name: str = "fs004",
) -> EvaluationPair:
    return EvaluationPair(
        reference=np.zeros(gt_shape, dtype=np.float32),
        deformed=np.zeros(gt_shape, dtype=np.float32),
        mask=mask,
        gt_field=_make_gt_field(gt_shape, shift),
        name=name,
        kind="synthetic",
        spacing=None,
    )


class TestEvaluatePairZeroErr:
    def test_perfect_recovery(self) -> None:
        shape = (16, 16, 16)
        shift = (0.5, -0.25, 1.0)
        positions = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]], dtype=np.float32)
        # Estimated == truth → zero error everywhere.
        displacements = np.tile(np.array(shift, dtype=np.float32), (positions.shape[0], 1))
        mask = np.ones(shape, dtype=np.bool_)
        pair = _eval_pair(gt_shape=shape, shift=shift, mask=mask)
        field = _make_field(positions, displacements)

        report = evaluate_pair(pair, field)
        assert report.mae == pytest.approx(0.0, abs=1e-4)
        assert report.rmse == pytest.approx(0.0, abs=1e-4)
        assert report.p95 == pytest.approx(0.0, abs=1e-4)
        assert report.per_axis_mae == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)
        assert report.n_valid == 2
        assert report.n_total == 2


class TestRealEntryRaises:
    def test_no_gt_raises(self) -> None:
        pair = EvaluationPair(
            reference=np.zeros((4, 4, 4), dtype=np.float32),
            deformed=np.zeros((4, 4, 4), dtype=np.float32),
            mask=None,
            gt_field=None,
            name="016",
            kind="real",
            spacing=None,
        )
        positions = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        displacements = np.zeros((1, 3), dtype=np.float32)
        field = _make_field(positions, displacements)
        with pytest.raises(ValueError, match="no ground truth"):
            evaluate_pair(pair, field)


class TestBoundaryStratification:
    def test_inner_bin_smaller_than_boundary_with_known_bias(self) -> None:
        # Mask: a centered cube of True surrounded by False voxels.
        # POIs near the boundary carry larger systematic error than POIs
        # in the bulk — by_distance should reflect this.
        shape = (32, 32, 32)
        mask = np.zeros(shape, dtype=np.bool_)
        mask[8:24, 8:24, 8:24] = True

        # POI lattice on a regular 4-stride grid.
        idx = np.arange(2, 30, 4)
        zz, yy, xx = np.meshgrid(idx, idx, idx, indexing="ij")
        positions = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
        n = positions.shape[0]

        # Truth is a constant shift (0, 0, 0) for simplicity.
        truth_shift = (0.0, 0.0, 0.0)
        # Estimated displacements = truth + bias proportional to (1 / (1 + d)).
        # Translate "near boundary => high error" with a deterministic bias
        # along z. Use the inner-mask boundary distance directly via the
        # same EDT the implementation uses.
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(~mask)
        rounded = np.clip(
            np.round(positions).astype(np.int64),
            a_min=0,
            a_max=np.array(shape, dtype=np.int64) - 1,
        )
        poi_dist = np.asarray(dist[rounded[:, 0], rounded[:, 1], rounded[:, 2]])
        bias = (5.0 / (1.0 + poi_dist)).astype(np.float32)
        displacements = np.zeros((n, 3), dtype=np.float32)
        displacements[:, 0] = bias  # error grows toward the boundary

        pair = _eval_pair(gt_shape=shape, shift=truth_shift, mask=mask)
        field = _make_field(positions, displacements)

        report = evaluate_pair(pair, field, distance_bins=(0, 2, 5, 10))
        assert report.by_distance is not None
        table = report.by_distance
        assert len(table.edges) == 4
        assert table.counts.shape == (4,)

        # Sanity: bin 0 (boundary band, d <= 2) should carry larger MAE
        # than bin 3 (deepest band, d > 10), where the bias has decayed.
        # All four bins must be populated for the comparison.
        assert all(c > 0 for c in table.counts)
        boundary_mae = float(table.mae[0])
        bulk_mae = float(table.mae[-1])
        assert boundary_mae > bulk_mae


class TestMaskMissingDegrades:
    def test_warns_and_skips_table(self) -> None:
        shape = (8, 8, 8)
        positions = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
        displacements = np.zeros((1, 3), dtype=np.float32)
        pair = _eval_pair(gt_shape=shape, shift=(0.0, 0.0, 0.0), mask=None)
        field = _make_field(positions, displacements)

        with pytest.warns(UserWarning, match="pair.mask is None"):
            report = evaluate_pair(pair, field)
        assert report.by_distance is None

    def test_empty_distance_bins_skips_silently(self) -> None:
        shape = (8, 8, 8)
        positions = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
        displacements = np.zeros((1, 3), dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        pair = _eval_pair(gt_shape=shape, shift=(0.0, 0.0, 0.0), mask=mask)
        field = _make_field(positions, displacements)

        report = evaluate_pair(pair, field, distance_bins=())
        assert report.by_distance is None


class TestInvalidPOIsExcluded:
    def test_invalid_pois_drop_out_of_aggregates(self) -> None:
        # Two POIs: one valid (perfect), one invalid (huge error).
        # The invalid one should be excluded so MAE = 0.
        shape = (16, 16, 16)
        positions = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]], dtype=np.float32)
        truth_shift = (1.0, 0.0, 0.0)
        displacements = np.array([[1.0, 0.0, 0.0], [99.0, 99.0, 99.0]], dtype=np.float32)
        valid = np.array([True, False], dtype=np.bool_)
        mask = np.ones(shape, dtype=np.bool_)
        pair = _eval_pair(gt_shape=shape, shift=truth_shift, mask=mask)
        field = _make_field(positions, displacements, valid=valid)

        report = evaluate_pair(pair, field)
        assert report.n_valid == 1
        assert report.n_total == 2
        assert report.mae == pytest.approx(0.0, abs=1e-4)


class TestEvaluatePairPrecomputedTruth:
    def test_precomputed_truth_matches_internal_eval(self) -> None:
        shape = (16, 16, 16)
        shift = (0.5, -0.25, 1.0)
        positions = np.array([[5.0, 5.0, 5.0], [10.0, 9.0, 8.0]], dtype=np.float32)
        displacements = np.array([[0.4, -0.2, 1.1], [0.6, -0.3, 0.9]], dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        pair = _eval_pair(gt_shape=shape, shift=shift, mask=mask)
        field = _make_field(positions, displacements)

        precomputed = pair.gt_field(field.positions)
        a = evaluate_pair(pair, field)
        b = evaluate_pair(pair, field, truth=precomputed)
        assert b.mae == pytest.approx(a.mae)
        assert b.rmse == pytest.approx(a.rmse)
        assert b.p95 == pytest.approx(a.p95)
        assert b.per_axis_mae == pytest.approx(a.per_axis_mae)

    def test_precomputed_truth_bypasses_gt_field(self) -> None:
        shape = (8, 8, 8)
        positions = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
        displacements = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)

        calls: list[int] = []

        class _CountingField:
            def __call__(self, coords: np.ndarray) -> np.ndarray:
                calls.append(len(coords))
                return np.zeros_like(coords)

        pair = EvaluationPair(
            reference=np.zeros(shape, dtype=np.float32),
            deformed=np.zeros(shape, dtype=np.float32),
            mask=mask,
            gt_field=_CountingField(),  # type: ignore[arg-type]
            name="fs004",
            kind="synthetic",
            spacing=None,
        )
        field = _make_field(positions, displacements)
        report = evaluate_pair(
            pair, field, truth=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        )
        assert report.mae == pytest.approx(0.0, abs=1e-6)
        assert calls == []  # gt_field was never invoked

    def test_truth_allows_scoring_entry_without_gt_field(self) -> None:
        shape = (8, 8, 8)
        positions = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
        displacements = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        pair = EvaluationPair(
            reference=np.zeros(shape, dtype=np.float32),
            deformed=np.zeros(shape, dtype=np.float32),
            mask=mask,
            gt_field=None,
            name="016",
            kind="real",
            spacing=None,
        )
        field = _make_field(positions, displacements)
        report = evaluate_pair(
            pair, field, truth=np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        )
        assert report.mae == pytest.approx(0.0, abs=1e-6)

    def test_truth_wrong_shape_raises(self) -> None:
        shape = (8, 8, 8)
        positions = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)
        displacements = np.zeros((2, 3), dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        pair = _eval_pair(gt_shape=shape, shift=(0.0, 0.0, 0.0), mask=mask)
        field = _make_field(positions, displacements)
        with pytest.raises(ValueError, match="does not match"):
            evaluate_pair(pair, field, truth=np.zeros((1, 3), dtype=np.float32))


# --------------------------------------------------------------------- driver


# Reuse the make_disk_store fixture from tests/io/conftest.py via path import.
# pytest doesn't auto-share conftest across sibling test directories, so we
# provide a local builder.

PROFILE = BONE_SCREW_SYNCHROTRON_V1


@pytest.fixture
def disk_store(tmp_path: Path) -> Path:
    """Build a tiny on-disk zarr store conforming to the bone profile."""
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import zarr

    from tests.io.conftest import _populate_store, _SyntheticEntry  # type: ignore

    store_path = tmp_path / "scan.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    _populate_store(
        root,
        profile=PROFILE,
        shape=(8, 16, 16),
        masks=("mask",),
        real_entries=("016",),
        synthetic_entries=(
            _SyntheticEntry("fs004", rigid_shift((0.5, 0.0, 0.0))),
            _SyntheticEntry("fs104", rigid_shift((0.0, 0.5, 0.0))),
        ),
        seed=0,
    )
    return store_path


def _identity_correlate_fn(
    truth_shift: tuple[float, float, float],
) -> Callable:
    """Return a fake correlate_fn that emits the (perfect) truth field."""

    def _fn(reference, deformed, mask):
        # POI grid: a single test point at the volume center.
        center = (np.array(reference.shape, dtype=np.float32) / 2.0).astype(np.float32)
        positions = center.reshape(1, 3)
        displacements = np.array([truth_shift], dtype=np.float32)
        return DisplacementField(
            positions=positions,
            displacements=displacements,
            valid=np.array([True], dtype=np.bool_),
            confidence=np.array([1.0], dtype=np.float32),
            status=np.array([POIStatus.OK], dtype=np.uint8),
            grid_shape=(1, 1, 1),
            spacing=(1, 1, 1),
            window=(3, 3, 3),
        )

    return _fn


class TestRunAndEvaluate:
    def test_perfect_correlator_yields_zero_error(self, disk_store: Path) -> None:
        ds = DvcDataset.open(disk_store)
        report = run_and_evaluate(
            ds, "fs004", correlate_fn=_identity_correlate_fn((0.5, 0.0, 0.0))
        )
        assert report.name == "fs004"
        assert report.mae == pytest.approx(0.0, abs=1e-4)
        assert report.n_valid == 1


class TestSweep:
    def test_sweep_returns_one_report_per_synthetic(self, disk_store: Path) -> None:
        ds = DvcDataset.open(disk_store)
        # The fake correlate_fn always returns a "0.5 dz" displacement.
        # Only fs004 (truth = (0.5, 0, 0)) gets MAE = 0; fs104 will not.
        reports = sweep(
            ds,
            correlate_fn=_identity_correlate_fn((0.5, 0.0, 0.0)),
        )
        assert [r.name for r in reports] == ["fs004", "fs104"]
        assert reports[0].mae == pytest.approx(0.0, abs=1e-4)
        assert reports[1].mae > 0.1

    def test_sweep_reuses_correlate_fn(self, disk_store: Path) -> None:
        # Confirm the sweep calls the same fn instance once per name.
        calls: list[str] = []

        def counting_fn(reference, deformed, mask):
            calls.append("call")
            return _identity_correlate_fn((0.5, 0.0, 0.0))(reference, deformed, mask)

        ds = DvcDataset.open(disk_store)
        sweep(ds, correlate_fn=counting_fn)
        assert len(calls) == len(ds.list_synthetic())

    def test_sweep_explicit_names_subset(self, disk_store: Path) -> None:
        ds = DvcDataset.open(disk_store)
        reports = sweep(
            ds,
            names=["fs004"],
            correlate_fn=_identity_correlate_fn((0.5, 0.0, 0.0)),
        )
        assert len(reports) == 1
        assert reports[0].name == "fs004"
