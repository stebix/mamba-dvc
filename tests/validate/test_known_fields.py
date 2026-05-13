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
from mamba_dvc.validate.known_fields import (
    BoundaryDistanceIndex,
    evaluate_pair,
    run_and_evaluate,
    sweep,
)
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


def _edt_oracle(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The pre-KDTree computation: ``distance_transform_edt(~mask)`` sampled at POIs."""
    from scipy.ndimage import distance_transform_edt

    dist = np.asarray(distance_transform_edt(~mask))
    rounded = np.clip(
        np.round(positions).astype(np.int64),
        a_min=0,
        a_max=np.asarray(mask.shape, dtype=np.int64) - 1,
    )
    return dist[rounded[:, 0], rounded[:, 1], rounded[:, 2]].astype(np.float32)


def _ball(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2


class TestBoundaryDistanceIndex:
    def _masks(self) -> list[tuple[str, np.ndarray]]:
        slab = np.zeros((24, 20, 28), dtype=bool)
        slab[5:14, 3:9, 10:25] = True
        hollow = _ball((28, 28, 28), (14.0, 14.0, 14.0), 11.0) & ~_ball(
            (28, 28, 28), (14.0, 14.0, 14.0), 6.0
        )
        return [
            ("solid-ball", _ball((32, 32, 32), (16.0, 15.0, 17.0), 10.0)),
            ("off-center-slab", slab),
            ("hollow-shell", hollow),
        ]

    def test_reproduces_edt_for_random_points(self) -> None:
        rng = np.random.default_rng(0)
        for name, mask in self._masks():
            shape = np.asarray(mask.shape)
            # Mix of in-bounds foreground, in-bounds background, and
            # out-of-range coords (negative and >= shape) so the clip,
            # the on-tissue -> 0 branch, and the KDTree branch all run.
            positions = rng.uniform(-3.0, shape + 3.0, size=(300, 3)).astype(np.float32)
            index = BoundaryDistanceIndex.from_mask(mask)
            got = index.query(positions)
            expected = _edt_oracle(mask, positions)
            np.testing.assert_allclose(got, expected, atol=1e-4, err_msg=name)

    def test_foreground_center_pois_are_zero(self) -> None:
        mask = _ball((24, 24, 24), (12.0, 12.0, 12.0), 8.0)
        fg = np.argwhere(mask).astype(np.float32)
        index = BoundaryDistanceIndex.from_mask(mask)
        got = index.query(fg)
        assert np.all(got == 0.0)
        # ... and the EDT agrees there.
        np.testing.assert_allclose(_edt_oracle(mask, fg), 0.0, atol=1e-6)

    def test_out_of_range_pois_match_clipped_edt(self) -> None:
        mask = _ball((20, 22, 18), (9.0, 11.0, 8.0), 6.0)
        positions = np.array(
            [[-5.0, -5.0, -5.0], [100.0, 100.0, 100.0], [-1.0, 10.0, 30.0]],
            dtype=np.float32,
        )
        index = BoundaryDistanceIndex.from_mask(mask)
        np.testing.assert_allclose(
            index.query(positions), _edt_oracle(mask, positions), atol=1e-4
        )

    def test_all_true_mask_is_all_zero(self) -> None:
        mask = np.ones((8, 8, 8), dtype=bool)
        index = BoundaryDistanceIndex.from_mask(mask)
        assert index.tree is None
        positions = np.array([[2.0, 3.0, 4.0], [-1.0, 50.0, 0.0]], dtype=np.float32)
        assert np.all(index.query(positions) == 0.0)
        # distance_transform_edt(~all-True) == distance_transform_edt(all-False) == 0
        np.testing.assert_allclose(_edt_oracle(mask, positions), 0.0, atol=1e-6)

    def test_all_false_mask_is_all_inf(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=bool)
        index = BoundaryDistanceIndex.from_mask(mask)
        assert index.tree is None
        positions = np.array([[2.0, 3.0, 4.0], [7.0, 7.0, 7.0]], dtype=np.float32)
        assert np.all(np.isinf(index.query(positions)))

    def test_index_reuse_has_no_hidden_state(self) -> None:
        mask = _ball((28, 28, 28), (14.0, 13.0, 15.0), 9.0)
        rng = np.random.default_rng(1)
        p1 = rng.uniform(0.0, 28.0, size=(40, 3)).astype(np.float32)
        p2 = rng.uniform(0.0, 28.0, size=(17, 3)).astype(np.float32)
        index = BoundaryDistanceIndex.from_mask(mask)
        q1_first = index.query(p1)
        _ = index.query(p2)
        q1_again = index.query(p1)
        np.testing.assert_array_equal(q1_first, q1_again)
        np.testing.assert_allclose(q1_first, _edt_oracle(mask, p1), atol=1e-4)

    def test_rejects_bad_mask(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            BoundaryDistanceIndex.from_mask(np.ones((4, 4), dtype=bool))
        with pytest.raises(ValueError, match="bool"):
            BoundaryDistanceIndex.from_mask(np.ones((4, 4, 4), dtype=np.float32))


class TestEvaluatePairBoundaryIndex:
    def _setup(self):
        # Same construction as TestBoundaryStratification: a centered True
        # cube, a regular POI lattice, a bias that grows toward the mask.
        shape = (32, 32, 32)
        mask = np.zeros(shape, dtype=bool)
        mask[8:24, 8:24, 8:24] = True
        idx = np.arange(2, 30, 4)
        zz, yy, xx = np.meshgrid(idx, idx, idx, indexing="ij")
        positions = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
        poi_dist = _edt_oracle(mask, positions)
        displacements = np.zeros((positions.shape[0], 3), dtype=np.float32)
        displacements[:, 0] = 5.0 / (1.0 + poi_dist)
        pair = _eval_pair(gt_shape=shape, shift=(0.0, 0.0, 0.0), mask=mask)
        field = _make_field(positions, displacements)
        return pair, field

    def test_precomputed_index_matches_on_the_fly(self) -> None:
        pair, field = self._setup()
        bins = (0, 2, 5, 10)
        a = evaluate_pair(pair, field, distance_bins=bins)
        index = BoundaryDistanceIndex.from_mask(pair.mask)
        b = evaluate_pair(pair, field, distance_bins=bins, boundary_index=index)
        assert a.by_distance is not None and b.by_distance is not None
        assert a.by_distance.edges == b.by_distance.edges
        np.testing.assert_array_equal(a.by_distance.counts, b.by_distance.counts)
        np.testing.assert_allclose(a.by_distance.mae, b.by_distance.mae, equal_nan=True)
        np.testing.assert_allclose(a.by_distance.rmse, b.by_distance.rmse, equal_nan=True)

    def test_shape_mismatch_raises(self) -> None:
        pair, field = self._setup()
        wrong = BoundaryDistanceIndex.from_mask(np.ones((4, 4, 4), dtype=bool))
        with pytest.raises(ValueError, match="does not match"):
            evaluate_pair(pair, field, distance_bins=(0, 2, 5, 10), boundary_index=wrong)

    def test_index_ignored_when_table_skipped(self) -> None:
        pair, field = self._setup()
        wrong = BoundaryDistanceIndex.from_mask(np.ones((4, 4, 4), dtype=bool))
        # Empty distance_bins -> no table -> the (mismatched) index is never touched.
        report = evaluate_pair(pair, field, distance_bins=(), boundary_index=wrong)
        assert report.by_distance is None


class TestBoundaryDistanceIndexSpeedup:
    @pytest.mark.slow
    def test_kdtree_path_far_outpaces_full_volume_edt(self) -> None:
        import time

        from scipy.ndimage import distance_transform_edt

        # A solid ball (~16% foreground) in a volume big enough that the
        # full-volume EDT is the obvious cost; small enough to stay a
        # "slow" unit test rather than a benchmark.
        side = 192
        shape = (side, side, side)
        mask = _ball(shape, (side / 2.0, side / 2.0, side / 2.0), side * 0.34)
        rng = np.random.default_rng(0)
        fg = np.argwhere(mask)
        bg = np.argwhere(~mask)
        positions = np.concatenate(
            [
                fg[rng.choice(len(fg), size=250, replace=False)],
                bg[rng.choice(len(bg), size=250, replace=False)],
            ]
        ).astype(np.float32)

        t0 = time.perf_counter()
        dist_vol = np.asarray(distance_transform_edt(~mask))
        rounded = np.clip(np.round(positions).astype(np.int64), 0, np.asarray(shape) - 1)
        expected = dist_vol[rounded[:, 0], rounded[:, 1], rounded[:, 2]].astype(np.float32)
        t_edt = time.perf_counter() - t0

        t0 = time.perf_counter()
        index = BoundaryDistanceIndex.from_mask(mask)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        got = index.query(positions)
        t_query = time.perf_counter() - t0

        # Numerically identical to the thing it replaces.
        np.testing.assert_allclose(got, expected, atol=1e-4)
        # Re-querying the cached index across a parameter sweep is ~free.
        assert t_query < 0.1, f"per-variant query took {t_query:.3f}s"
        # Building + querying once is well under the full-volume EDT — a
        # coarse sanity bound (the real ratio here is ~10-30x), not a
        # benchmark. Skip the assertion on the odd machine where the EDT
        # is somehow trivially fast.
        if t_edt > 0.2:
            assert t_build + t_query < t_edt / 4, (
                f"kdtree path {t_build + t_query:.3f}s vs edt {t_edt:.3f}s"
            )


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
