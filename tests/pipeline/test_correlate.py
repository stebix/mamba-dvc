"""Tests for ``mamba_dvc.pipeline.correlate``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
from mamba_dvc.core.grid import build_grid
from mamba_dvc.core.outlier import detect_outliers
from mamba_dvc.pipeline._internal import TUKEY_DEFAULTS, correlate_admitted_subset
from mamba_dvc.pipeline.correlate import correlate
from mamba_dvc.types import DisplacementField, POIStatus
from mamba_dvc.validate.synthetic import make_pair, rigid_shift, sample_on_grid

correlate_checked = jaxtyped(typechecker=beartype)(correlate)


# Pipeline-level NCC mode sweep. Each entry expands into the kwargs the
# pipeline accepts; the linear+overlap pair is the default and the
# cyclic+global pair is the legacy code path retained for A/B work.
NCC_MODES = [
    pytest.param(
        {"ncc_mode": "linear", "ncc_normalization": "overlap"},
        id="linear-overlap",
    ),
    pytest.param(
        {"ncc_mode": "linear", "ncc_normalization": "global"},
        id="linear-global",
    ),
    pytest.param(
        {"ncc_mode": "cyclic", "ncc_normalization": "global"},
        id="cyclic-global",
    ),
]


class TestSignatureAndStructure:
    def test_returns_displacementfield_with_expected_shapes(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = ref.copy()

        field = correlate(ref, deformed, window=24, overlap=0.5, search_radius=8)

        assert isinstance(field, DisplacementField)
        n = int(np.prod(field.grid_shape))
        assert field.positions.shape == (n, 3)
        assert field.displacements.shape == (n, 3)
        assert field.valid.shape == (n,)
        assert field.confidence.shape == (n,)
        assert field.status.shape == (n,)
        assert field.window == (24, 24, 24)

    def test_grid_metadata_reshape_round_trip(self):
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)

        field = correlate(ref, ref, window=24, overlap=0.5, search_radius=8)

        # Reshape into the regular lattice and back.
        lattice = field.displacements.reshape(*field.grid_shape, 3)
        flat = lattice.reshape(-1, 3)
        np.testing.assert_array_equal(flat, field.displacements)

    def test_runtime_shape_check_via_jaxtyping(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        field = correlate_checked(ref, ref, window=16, search_radius=4)
        assert isinstance(field, DisplacementField)


class TestZeroShift:
    @pytest.mark.parametrize("ncc_kwargs", NCC_MODES)
    def test_recovers_zero_displacement_everywhere(self, ncc_kwargs):
        rng = np.random.default_rng(2)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)

        field = correlate(ref, ref, window=24, overlap=0.5, search_radius=8, **ncc_kwargs)

        assert field.valid.all()
        assert (field.status == POIStatus.OK).all()
        np.testing.assert_allclose(field.displacements, 0.0, atol=0.05)


class TestMaskAdmission:
    def test_masked_slab_is_flagged(self):
        # Construct a mask that excludes a slab of voxels along z; POIs
        # whose subvolume overlaps the slab too much must be flagged
        # MASKED.
        shape = (64, 64, 64)
        rng = np.random.default_rng(3)
        ref = rng.standard_normal(shape, dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        mask[:16, :, :] = False  # exclude the first 16 z-slices

        field = correlate(
            ref, ref, mask, window=32, overlap=0.5, mask_threshold=0.9, search_radius=8
        )

        # POIs starting at z=0 cover [0:32], 16/32=50% invalid -> MASKED.
        # POIs starting at z=16 cover [16:48], 0/32 invalid -> OK.
        z_starts = (field.positions[:, 0] - (field.window[0] - 1) / 2.0).astype(np.int64)
        is_top = z_starts == 0
        is_middle = z_starts == 16

        assert (field.status[is_top] == POIStatus.MASKED).all()
        np.testing.assert_array_equal(field.displacements[is_top], 0.0)
        np.testing.assert_array_equal(field.confidence[is_top], 0.0)
        assert (field.status[is_middle] == POIStatus.OK).all()

    def test_no_mask_keeps_every_poi_admitted(self):
        rng = np.random.default_rng(4)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)

        field = correlate(ref, ref, mask=None, window=24, overlap=0.5, search_radius=8)

        assert (field.status != POIStatus.MASKED).all()


class TestSearchRadius:
    def test_out_of_range_displacement_is_flagged(self):
        # Roll the deformed by 6 voxels along z; with search_radius=3
        # every POI sees an out-of-range integer peak and gets flagged.
        rng = np.random.default_rng(5)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=6, axis=0)

        field = correlate(ref, deformed, window=24, overlap=0.5, search_radius=3)

        admitted = field.status != POIStatus.MASKED
        flagged = field.status[admitted] == POIStatus.OUT_OF_RANGE
        assert flagged.all(), (
            f"expected all admitted POIs flagged OUT_OF_RANGE; "
            f"status histogram = {np.bincount(field.status, minlength=5)}"
        )
        np.testing.assert_array_equal(
            field.displacements[field.status == POIStatus.OUT_OF_RANGE], 0.0
        )

    def test_within_range_passes(self):
        rng = np.random.default_rng(6)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=2, axis=0)

        field = correlate(ref, deformed, window=24, overlap=0.5, search_radius=8)

        # Some POIs near the edge see boundary contamination, but the
        # majority should pass.
        assert (field.status == POIStatus.OK).sum() >= 0.8 * field.status.size


class TestDeformedMaskFallback:
    def test_omitted_deformed_mask_matches_passing_same_array(self):
        rng = np.random.default_rng(7)
        shape = (48, 48, 48)
        ref = rng.standard_normal(shape, dtype=np.float32)
        mask = np.ones(shape, dtype=np.bool_)
        mask[:8, :, :] = False

        field_default = correlate(ref, ref, mask, window=24, overlap=0.5, search_radius=8)
        field_explicit = correlate(
            ref, ref, mask, deformed_mask=mask, window=24, overlap=0.5, search_radius=8
        )

        np.testing.assert_array_equal(field_default.status, field_explicit.status)
        np.testing.assert_allclose(
            field_default.displacements, field_explicit.displacements, atol=1e-6
        )
        np.testing.assert_allclose(
            field_default.confidence, field_explicit.confidence, atol=1e-6
        )


class TestBatchInvariance:
    def test_small_and_large_batches_match(self):
        rng = np.random.default_rng(8)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=2, axis=0)

        field_small = correlate(
            ref, deformed, window=24, overlap=0.5, search_radius=8, batch_size=4
        )
        field_large = correlate(
            ref, deformed, window=24, overlap=0.5, search_radius=8, batch_size=64
        )

        np.testing.assert_array_equal(field_small.status, field_large.status)
        np.testing.assert_allclose(
            field_small.displacements, field_large.displacements, atol=1e-5
        )
        np.testing.assert_allclose(field_small.confidence, field_large.confidence, atol=1e-5)


class TestInputValidation:
    def test_shape_mismatch_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        deformed = np.zeros((32, 32, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            correlate(ref, deformed, window=16)

    def test_wrong_dtype_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float64)
        deformed = np.zeros((32, 32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            correlate(ref, deformed, window=16)  # type: ignore[arg-type]

    def test_wrong_ndim_raises(self):
        ref = np.zeros((32, 32), dtype=np.float32)
        deformed = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="3D"):
            correlate(ref, deformed, window=16)  # type: ignore[arg-type]

    def test_mask_shape_mismatch_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        mask = np.ones((32, 32, 16), dtype=np.bool_)
        with pytest.raises(ValueError, match="mask shape"):
            correlate(ref, ref, mask, window=16)

    def test_non_bool_mask_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        mask = np.ones((32, 32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool"):
            correlate(ref, ref, mask, window=16)  # type: ignore[arg-type]

    def test_non_positive_batch_size_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="batch_size"):
            correlate(ref, ref, window=16, batch_size=0)

    def test_non_positive_search_radius_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="search_radius"):
            correlate(ref, ref, window=16, search_radius=0)

    def test_unknown_ncc_mode_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="ncc_mode"):
            correlate(ref, ref, window=16, ncc_mode="bogus")  # type: ignore[arg-type]

    def test_unknown_ncc_normalization_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="ncc_normalization"):
            correlate(ref, ref, window=16, ncc_normalization="bogus")  # type: ignore[arg-type]


class TestNCCModeKnobs:
    def test_per_mode_tukey_default_resolves(self):
        # tukey_alpha=None should pick 0.0 in linear mode and 0.25 in
        # cyclic. The visible signature is identical; we verify the
        # resolution by comparing against an explicit alpha that
        # matches the per-mode default.
        rng = np.random.default_rng(11)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=(1, -1, 2), axis=(0, 1, 2))

        f_lin_auto = correlate(
            ref,
            deformed,
            window=24,
            overlap=0.5,
            search_radius=8,
            ncc_mode="linear",
            ncc_normalization="overlap",
        )
        f_lin_explicit = correlate(
            ref,
            deformed,
            window=24,
            overlap=0.5,
            search_radius=8,
            tukey_alpha=0.0,
            ncc_mode="linear",
            ncc_normalization="overlap",
        )
        np.testing.assert_allclose(
            f_lin_auto.displacements, f_lin_explicit.displacements, atol=1e-6
        )

        f_cyc_auto = correlate(
            ref,
            deformed,
            window=24,
            overlap=0.5,
            search_radius=8,
            ncc_mode="cyclic",
            ncc_normalization="global",
        )
        f_cyc_explicit = correlate(
            ref,
            deformed,
            window=24,
            overlap=0.5,
            search_radius=8,
            tukey_alpha=0.25,
            ncc_mode="cyclic",
            ncc_normalization="global",
        )
        np.testing.assert_allclose(
            f_cyc_auto.displacements, f_cyc_explicit.displacements, atol=1e-6
        )

    @pytest.mark.slow
    def test_linear_overlap_beats_cyclic_on_fractional_shift(self):
        # Showcase regime from docs/insights/error-minimization.md:
        # window 32 on a textured volume with a non-trivial shift. The
        # cyclic+global combo carries the |u|/W shrinkage bias; the
        # linear+overlap default should land closer to ground truth.
        shift = (1.5, -2.3, 0.7)
        shape: tuple[int, int, int] = (96, 128, 128)
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=23)

        common = dict(window=32, overlap=0.5, search_radius=8)
        f_linear = correlate(
            pair.reference,
            pair.deformed,
            ncc_mode="linear",
            ncc_normalization="overlap",
            **common,
        )
        f_cyclic = correlate(
            pair.reference,
            pair.deformed,
            ncc_mode="cyclic",
            ncc_normalization="global",
            **common,
        )

        # Compare on the interior valid POIs only (boundary POIs see
        # truncation effects that dominate the kernel's intrinsic
        # bias).
        from mamba_dvc.core.grid import build_grid

        grid = build_grid(shape, window=(32, 32, 32), overlap=0.5)
        gt = sample_on_grid(pair.field, grid)
        starts = f_linear.positions - (np.asarray(f_linear.window, dtype=np.float32) - 1) / 2.0
        wz, wy, wx = f_linear.window
        margin = 8
        interior = (
            (starts[:, 0] >= margin)
            & (starts[:, 0] + wz <= shape[0] - margin)
            & (starts[:, 1] >= margin)
            & (starts[:, 1] + wy <= shape[1] - margin)
            & (starts[:, 2] >= margin)
            & (starts[:, 2] + wx <= shape[2] - margin)
        )
        usable = interior & f_linear.valid & f_cyclic.valid

        mae_linear = np.abs(f_linear.displacements[usable] - gt[usable]).mean(axis=0)
        mae_cyclic = np.abs(f_cyclic.displacements[usable] - gt[usable]).mean(axis=0)

        # Strict regression on the new default; the loose "less than"
        # check on cyclic captures the bias without depending on its
        # exact magnitude.
        assert mae_linear.max() < 0.1, f"linear+overlap MAE {mae_linear} above 0.1 budget"
        assert mae_linear.max() < mae_cyclic.max(), (
            f"linear+overlap MAE {mae_linear} should be lower than cyclic MAE "
            f"{mae_cyclic} on the showcase regime"
        )


@pytest.mark.slow
class TestEndToEndAccuracy:
    @pytest.mark.parametrize(
        "shift",
        [
            (0.3, -0.7, 1.4),
            (-1.2, 1.8, -0.4),
        ],
    )
    def test_orchestrator_recovers_fractional_shift(self, shift):
        # End-to-end: the orchestrator should match the inline
        # composition the integration smoke test used to assemble.
        shape: tuple[int, int, int] = (160, 160, 160)
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=17)

        field = correlate(
            pair.reference,
            pair.deformed,
            window=64,
            overlap=0.5,
            search_radius=16,
        )

        ground_truth = sample_on_grid(pair.field, _grid_from_field(field))

        # Interior filter mirrors the smoke test.
        starts = field.positions - (np.asarray(field.window, dtype=np.float32) - 1) / 2.0
        wz, wy, wx = field.window
        max_shift = int(np.ceil(max(abs(s) for s in shift)))
        margin = max_shift + 4
        interior = (
            (starts[:, 0] >= margin)
            & (starts[:, 0] + wz <= shape[0] - margin)
            & (starts[:, 1] >= margin)
            & (starts[:, 1] + wy <= shape[1] - margin)
            & (starts[:, 2] >= margin)
            & (starts[:, 2] + wx <= shape[2] - margin)
        )

        usable = interior & field.valid
        assert usable.sum() >= 8
        error = np.abs(field.displacements[usable] - ground_truth[usable])
        assert np.all(error < 0.1), f"max error per axis: {error.max(axis=0)}"


def _grid_from_field(field: DisplacementField):
    """Reconstruct the GridSpec used internally for ground-truth sampling."""
    # The field's positions correspond to the same grid build_grid
    # would produce given the same volume shape, window, and overlap.
    # We back out the volume shape from positions + window; for the
    # tests above we know it.
    nz, ny, nx = field.grid_shape
    sz, sy, sx = field.spacing
    wz, wy, wx = field.window
    volume_shape = (
        int((nz - 1) * sz + wz),
        int((ny - 1) * sy + wy),
        int((nx - 1) * sx + wx),
    )
    overlap = 1.0 - sz / wz
    return build_grid(volume_shape, window=field.window, overlap=overlap)


# Default helper kwargs used across the dispatch-seam tests below. The
# linear+overlap mode is the pipeline default; tukey_alpha=0.0 is what
# the orchestrator resolves at the linear branch.
_HELPER_KWARGS = dict(
    batch_size=64,
    eps=1e-12,
    ncc_mode="linear",
    ncc_normalization="overlap",
    tukey_alpha=TUKEY_DEFAULTS["linear"],
)


class TestCorrelateAdmittedSubset:
    """Direct tests of the dispatch-seam helper.

    The helper is what each multi-GPU worker calls on its share of the
    admitted POIs (plan §6). These tests pin its contract: full-length
    sparse output with ``MASKED`` outside ``admitted_idx``, OK /
    OUT_OF_RANGE inside, and disjoint shards that scatter back into the
    same arrays a single full-admitted call produces.
    """

    @staticmethod
    def _smooth_pair(
        shape: tuple[int, int, int] = (48, 48, 48),
        shift: tuple[float, float, float] = (0.4, -0.6, 0.9),
        seed: int = 17,
    ):
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=seed)
        return pair.reference, pair.deformed

    def test_helper_plus_outlier_overlay_equals_correlate(self):
        # Spec: correlate() == correlate_admitted_subset(full_admitted)
        # followed by the outlier overlay step. Run both and assert
        # bit-identical (within float32 noise) at every slot.
        ref, deformed = self._smooth_pair()
        window = 24
        search_radius = 8

        field = correlate(
            ref, deformed, window=window, overlap=0.5, search_radius=search_radius
        )

        grid = build_grid(ref.shape, window=window, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        admitted_idx = np.arange(n_points, dtype=np.int64)
        eff_mask = np.ones(ref.shape, dtype=np.bool_)

        h_disp, h_conf, h_stat = correlate_admitted_subset(
            ref,
            deformed,
            eff_mask,
            eff_mask,
            grid,
            admitted_idx,
            search_radius=search_radius,
            **_HELPER_KWARGS,
        )

        # Apply the same outlier overlay correlate runs after the helper.
        valid_pre = h_stat == POIStatus.OK
        outlier_flag = detect_outliers(grid, h_disp, valid_pre)
        h_disp[outlier_flag] = 0.0
        h_conf[outlier_flag] = 0.0
        h_stat[outlier_flag] = POIStatus.OUTLIER

        np.testing.assert_array_equal(h_stat, field.status)
        np.testing.assert_allclose(h_disp, field.displacements, atol=1e-6)
        np.testing.assert_allclose(h_conf, field.confidence, atol=1e-6)

    def test_helper_plus_outlier_overlay_equals_correlate_with_mask(self):
        # Same equivalence but with mask admission active. We mirror
        # the orchestrator's filter_by_mask -> flatnonzero step to
        # build the helper's admitted_idx.
        from mamba_dvc.core.grid import filter_by_mask

        rng = np.random.default_rng(31)
        shape = (48, 48, 48)
        ref = rng.standard_normal(shape, dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)
        mask = np.ones(shape, dtype=np.bool_)
        mask[:8, :, :] = False  # exclude the first 8 z-slices
        window = 24
        search_radius = 8

        field = correlate(
            ref,
            deformed,
            mask,
            window=window,
            overlap=0.5,
            mask_threshold=0.9,
            search_radius=search_radius,
        )

        grid = build_grid(shape, window=window, overlap=0.5)
        admitted = filter_by_mask(grid, mask, threshold=0.9)
        admitted_idx = np.flatnonzero(admitted).astype(np.int64)

        h_disp, h_conf, h_stat = correlate_admitted_subset(
            ref,
            deformed,
            mask,
            mask,
            grid,
            admitted_idx,
            search_radius=search_radius,
            **_HELPER_KWARGS,
        )

        valid_pre = h_stat == POIStatus.OK
        outlier_flag = detect_outliers(grid, h_disp, valid_pre)
        h_disp[outlier_flag] = 0.0
        h_conf[outlier_flag] = 0.0
        h_stat[outlier_flag] = POIStatus.OUTLIER

        np.testing.assert_array_equal(h_stat, field.status)
        np.testing.assert_allclose(h_disp, field.displacements, atol=1e-6)
        np.testing.assert_allclose(h_conf, field.confidence, atol=1e-6)

    def test_disjoint_shards_scatter_to_full(self):
        # Plan §6 sharding: partition admitted_idx along lattice-Z into
        # contiguous slabs, run helper per shard, scatter each shard's
        # slots into the parent buffers, expect the result to match a
        # single full-admitted call. This is the invariant that makes
        # multi-GPU dispatch correct.
        ref, deformed = self._smooth_pair()
        window = 24
        search_radius = 8

        grid = build_grid(ref.shape, window=window, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        eff_mask = np.ones(ref.shape, dtype=np.bool_)

        all_admitted = np.arange(n_points, dtype=np.int64)
        ref_disp, ref_conf, ref_stat = correlate_admitted_subset(
            ref,
            deformed,
            eff_mask,
            eff_mask,
            grid,
            all_admitted,
            search_radius=search_radius,
            **_HELPER_KWARGS,
        )

        # Build three disjoint shards along lattice-Z (mirrors the
        # contiguous-slab partition the dispatch layer will produce).
        nz = grid.grid_shape[0]
        z_chunks = np.array_split(np.arange(nz, dtype=np.int64), 3)
        shards: list[np.ndarray] = []
        for chunk in z_chunks:
            mask_lat = np.zeros(grid.grid_shape, dtype=np.bool_)
            mask_lat[chunk[0] : chunk[-1] + 1] = True
            shards.append(np.flatnonzero(mask_lat).astype(np.int64))

        # Sanity: shards are disjoint and cover every POI.
        union = np.concatenate(shards)
        assert union.size == n_points
        np.testing.assert_array_equal(np.sort(union), all_admitted)

        # Parent allocates full-length buffers (default-init MASKED /
        # zero) and scatters each shard's slots in.
        merged_disp = np.zeros((n_points, 3), dtype=np.float32)
        merged_conf = np.zeros(n_points, dtype=np.float32)
        merged_stat = np.full(n_points, POIStatus.MASKED, dtype=np.uint8)

        for shard_idx in shards:
            s_disp, s_conf, s_stat = correlate_admitted_subset(
                ref,
                deformed,
                eff_mask,
                eff_mask,
                grid,
                shard_idx,
                search_radius=search_radius,
                **_HELPER_KWARGS,
            )
            merged_disp[shard_idx] = s_disp[shard_idx]
            merged_conf[shard_idx] = s_conf[shard_idx]
            merged_stat[shard_idx] = s_stat[shard_idx]

        np.testing.assert_array_equal(merged_stat, ref_stat)
        np.testing.assert_allclose(merged_disp, ref_disp, atol=1e-6)
        np.testing.assert_allclose(merged_conf, ref_conf, atol=1e-6)

    def test_non_shard_slots_are_masked(self):
        # Sparse-output contract: indices outside admitted_idx come back
        # MASKED with zero displacement / confidence. This is what lets
        # the dispatch parent rely on per-shard scatter without merging
        # logic.
        ref, deformed = self._smooth_pair()
        window = 24
        grid = build_grid(ref.shape, window=window, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        eff_mask = np.ones(ref.shape, dtype=np.bool_)

        admitted_idx = np.arange(n_points // 2, dtype=np.int64)

        h_disp, h_conf, h_stat = correlate_admitted_subset(
            ref,
            deformed,
            eff_mask,
            eff_mask,
            grid,
            admitted_idx,
            search_radius=8,
            **_HELPER_KWARGS,
        )

        non_shard = np.ones(n_points, dtype=np.bool_)
        non_shard[admitted_idx] = False
        assert (h_stat[non_shard] == POIStatus.MASKED).all()
        np.testing.assert_array_equal(h_disp[non_shard], 0.0)
        np.testing.assert_array_equal(h_conf[non_shard], 0.0)
        # Shard slots: every status is either OK or OUT_OF_RANGE. The
        # helper does not run the outlier test, so OUTLIER must not
        # appear.
        shard_stat = h_stat[admitted_idx]
        ok_or_oor = (shard_stat == POIStatus.OK) | (shard_stat == POIStatus.OUT_OF_RANGE)
        assert ok_or_oor.all()

    def test_empty_admitted_idx_returns_all_masked(self):
        # Degenerate but real: a Z-slab with no admitted POIs (e.g.
        # entirely covered by the screw mask) should produce a clean
        # all-MASKED, all-zero result without indexing into empty
        # batches.
        ref, deformed = self._smooth_pair()
        window = 24
        grid = build_grid(ref.shape, window=window, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        eff_mask = np.ones(ref.shape, dtype=np.bool_)

        empty = np.empty(0, dtype=np.int64)

        h_disp, h_conf, h_stat = correlate_admitted_subset(
            ref,
            deformed,
            eff_mask,
            eff_mask,
            grid,
            empty,
            search_radius=8,
            **_HELPER_KWARGS,
        )

        assert h_disp.shape == (n_points, 3)
        assert h_conf.shape == (n_points,)
        assert h_stat.shape == (n_points,)
        assert (h_stat == POIStatus.MASKED).all()
        np.testing.assert_array_equal(h_disp, 0.0)
        np.testing.assert_array_equal(h_conf, 0.0)

    def test_out_of_range_zeroes_and_flags_only_admitted(self):
        # Search-radius gate is a per-POI check inside the helper. When
        # the integer peak escapes the radius, the slot is flagged
        # OUT_OF_RANGE with zero disp/conf -- but only at admitted
        # slots; non-admitted slots stay MASKED.
        rng = np.random.default_rng(99)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=6, axis=0)
        window = 24
        grid = build_grid(ref.shape, window=window, overlap=0.5)
        n_points = int(np.prod(grid.grid_shape))
        eff_mask = np.ones(ref.shape, dtype=np.bool_)

        admitted_idx = np.arange(n_points // 2, dtype=np.int64)

        h_disp, h_conf, h_stat = correlate_admitted_subset(
            ref,
            deformed,
            eff_mask,
            eff_mask,
            grid,
            admitted_idx,
            search_radius=3,
            **_HELPER_KWARGS,
        )

        assert (h_stat[admitted_idx] == POIStatus.OUT_OF_RANGE).all()
        np.testing.assert_array_equal(h_disp[admitted_idx], 0.0)
        np.testing.assert_array_equal(h_conf[admitted_idx], 0.0)

        non_shard = np.ones(n_points, dtype=np.bool_)
        non_shard[admitted_idx] = False
        assert (h_stat[non_shard] == POIStatus.MASKED).all()
