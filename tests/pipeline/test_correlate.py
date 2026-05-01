"""Tests for ``mamba_dvc.pipeline.correlate``."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import jaxtyped
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
    from mamba_dvc.core.grid import build_grid

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
