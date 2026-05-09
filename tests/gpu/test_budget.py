"""Tests for ``mamba_dvc.gpu.budget``.

The cost model is pure arithmetic and runs on any host. The probe and
the end-to-end "recommended batch actually fits" tests are
``@pytest.mark.gpu`` because they need a real CUDA device.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from mamba_dvc.core.ncc import NCCMode, NCCNormalization
from mamba_dvc.gpu import budget
from mamba_dvc.gpu.budget import (
    BudgetInputs,
    KernelFootprint,
    estimate_max_batch,
    kernel_footprint,
    probe_free_vram,
    recommend_batch_size,
    resident_bytes,
)

_GIB = 1024**3


def _make_inputs(
    *,
    volume_shape: tuple[int, int, int] = (960, 1280, 1280),
    window: tuple[int, int, int] = (96, 96, 96),
    mode: NCCMode = NCCMode.LINEAR,
    normalization: NCCNormalization = NCCNormalization.OVERLAP,
    has_mask: bool = True,
    deformed_mask_distinct: bool = False,
) -> BudgetInputs:
    return BudgetInputs(
        volume_shape=volume_shape,
        window=window,
        mode=mode,
        normalization=normalization,
        has_mask=has_mask,
        deformed_mask_distinct=deformed_mask_distinct,
    )


class TestKernelFootprint:
    @pytest.mark.parametrize(
        ("mode", "normalization"),
        [
            (NCCMode.CYCLIC, NCCNormalization.GLOBAL),
            (NCCMode.LINEAR, NCCNormalization.GLOBAL),
            (NCCMode.LINEAR, NCCNormalization.OVERLAP),
        ],
    )
    def test_supported_combinations_return_positive_footprint(
        self, mode: NCCMode, normalization: NCCNormalization
    ):
        fp = kernel_footprint((96, 96, 96), mode, normalization)
        assert fp.per_poi_bytes > 0
        assert fp.fixed_bytes >= 0

    def test_per_poi_scales_linearly_with_window_voxels(self):
        # Doubling every window axis should multiply per_poi_bytes by 8.
        small = kernel_footprint((32, 32, 32), NCCMode.LINEAR, NCCNormalization.OVERLAP)
        big = kernel_footprint((64, 64, 64), NCCMode.LINEAR, NCCNormalization.OVERLAP)
        assert big.per_poi_bytes == 8 * small.per_poi_bytes

    def test_per_poi_uses_product_of_window(self):
        # Anisotropic windows: cost is prod(window), not max axis.
        cube = kernel_footprint((48, 48, 48), NCCMode.CYCLIC, NCCNormalization.GLOBAL)
        slab = kernel_footprint((96, 48, 24), NCCMode.CYCLIC, NCCNormalization.GLOBAL)
        # prod(96,48,24) == prod(48,48,48) == 110592.
        assert cube.per_poi_bytes == slab.per_poi_bytes

    def test_overlap_costlier_than_global_costlier_than_cyclic(self):
        # The model must rank the kernels in the order the plan
        # claims, regardless of the absolute coefficients. This is the
        # ordering invariant the dispatcher's batch-size default
        # already encodes (256 for cyclic, 64 for linear+overlap).
        win = (96, 96, 96)
        cyc = kernel_footprint(win, NCCMode.CYCLIC, NCCNormalization.GLOBAL)
        lin_g = kernel_footprint(win, NCCMode.LINEAR, NCCNormalization.GLOBAL)
        lin_o = kernel_footprint(win, NCCMode.LINEAR, NCCNormalization.OVERLAP)
        assert cyc.per_poi_bytes < lin_g.per_poi_bytes < lin_o.per_poi_bytes

    def test_cyclic_overlap_combination_rejected(self):
        # Mirrors the dispatch / core/ncc validation: cyclic kernel
        # has no per-lag overlap denominator, so the combo is invalid.
        with pytest.raises(ValueError, match="unsupported"):
            kernel_footprint((96, 96, 96), NCCMode.CYCLIC, NCCNormalization.OVERLAP)

    def test_malformed_window_raises(self):
        with pytest.raises(ValueError, match="length 3"):
            kernel_footprint((96, 96), NCCMode.LINEAR, NCCNormalization.OVERLAP)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            kernel_footprint((96, 0, 96), NCCMode.LINEAR, NCCNormalization.OVERLAP)
        with pytest.raises(ValueError, match="positive"):
            kernel_footprint((96, -1, 96), NCCMode.LINEAR, NCCNormalization.OVERLAP)


class TestKernelFootprintNegativeRejected:
    """KernelFootprint dataclass validation is its own backstop."""

    def test_negative_per_poi_raises(self):
        with pytest.raises(ValueError, match="per_poi_bytes"):
            KernelFootprint(per_poi_bytes=-1, fixed_bytes=0)

    def test_negative_fixed_raises(self):
        with pytest.raises(ValueError, match="fixed_bytes"):
            KernelFootprint(per_poi_bytes=1, fixed_bytes=-1)


class TestResidentBytes:
    def test_full_production_shape_with_mask_no_distinct(self):
        # (960, 1280, 1280): ref + def + mask, no distinct def_mask.
        # 2 * 4 * V + V where V = 960*1280*1280 = 1_572_864_000.
        # = 8V + V = 9V = 14_155_776_000 B ≈ 13.18 GiB.
        inputs = _make_inputs(deformed_mask_distinct=False)
        bytes_total = resident_bytes(inputs)
        v = 960 * 1280 * 1280
        assert bytes_total == 2 * 4 * v + v

    def test_full_production_shape_with_distinct_deformed_mask(self):
        # Same as above plus a second bool mask: 2*4*V + 2*V = 10V.
        inputs = _make_inputs(deformed_mask_distinct=True)
        v = 960 * 1280 * 1280
        assert resident_bytes(inputs) == 2 * 4 * v + 2 * v

    def test_has_mask_false_still_charges_for_synthesized_mask(self):
        # Dispatch synthesizes an all-True mask when the user passes
        # None, so the resident cost is the same. The has_mask flag is
        # a hook for a future skip-the-upload optimization, not a
        # current-day savings.
        with_mask = resident_bytes(_make_inputs(has_mask=True))
        without_mask = resident_bytes(
            _make_inputs(has_mask=False, deformed_mask_distinct=False)
        )
        assert with_mask == without_mask

    def test_small_volume(self):
        inputs = _make_inputs(
            volume_shape=(32, 32, 32),
            window=(16, 16, 16),
            deformed_mask_distinct=False,
        )
        v = 32 * 32 * 32
        assert resident_bytes(inputs) == 2 * 4 * v + v


class TestBudgetInputsValidation:
    def test_volume_shape_must_be_length_3(self):
        with pytest.raises(ValueError, match="length 3"):
            BudgetInputs(
                volume_shape=(960, 1280),  # type: ignore[arg-type]
                window=(96, 96, 96),
                mode=NCCMode.LINEAR,
                normalization=NCCNormalization.OVERLAP,
                has_mask=True,
                deformed_mask_distinct=False,
            )

    def test_volume_shape_positivity(self):
        with pytest.raises(ValueError, match="positive"):
            BudgetInputs(
                volume_shape=(0, 1280, 1280),
                window=(96, 96, 96),
                mode=NCCMode.LINEAR,
                normalization=NCCNormalization.OVERLAP,
                has_mask=True,
                deformed_mask_distinct=False,
            )

    def test_window_positivity(self):
        with pytest.raises(ValueError, match="positive"):
            _make_inputs(window=(96, -1, 96))

    def test_window_within_volume(self):
        with pytest.raises(ValueError, match="exceeds volume_shape"):
            _make_inputs(volume_shape=(64, 64, 64), window=(96, 96, 96))

    def test_distinct_deformed_mask_requires_has_mask(self):
        with pytest.raises(ValueError, match="has_mask=True"):
            _make_inputs(has_mask=False, deformed_mask_distinct=True)


class TestEstimateMaxBatch:
    def test_returns_multiple_of_eight(self):
        inputs = _make_inputs()
        # Pick a comfortable VRAM ceiling well above the 14 GB resident.
        batch = estimate_max_batch(40 * _GIB, inputs)
        assert batch % 8 == 0

    def test_monotone_in_free_bytes(self):
        inputs = _make_inputs()
        # Step the free pool up; the recommended batch must not shrink.
        last = -1
        for free in (16 * _GIB, 24 * _GIB, 32 * _GIB, 40 * _GIB, 48 * _GIB):
            try:
                b = estimate_max_batch(free, inputs)
            except RuntimeError:
                # Tight pools may refuse altogether; that is monotone-
                # consistent with later, larger pools succeeding.
                continue
            assert b >= last
            last = b

    def test_clamps_to_max_batch(self):
        inputs = _make_inputs(window=(32, 32, 32))
        # Massive free pool, small window: the raw answer would
        # exceed any sane ceiling. max_batch must clamp.
        batch = estimate_max_batch(200 * _GIB, inputs, max_batch=128)
        assert batch == 128

    def test_underflow_raises_with_diagnostic_message(self):
        # 16 GB free with the production config: resident alone is
        # ~13 GB and headroom eats more than the rest. Should raise.
        inputs = _make_inputs()
        with pytest.raises(RuntimeError) as excinfo:
            estimate_max_batch(16 * _GIB, inputs, min_batch=8)
        msg = str(excinfo.value)
        # Diagnostic content the plan calls for.
        assert "free=" in msg
        assert "resident=" in msg
        assert "per_poi=" in msg
        assert "headroom_fraction=" in msg

    def test_zero_free_underflows(self):
        inputs = _make_inputs()
        with pytest.raises(RuntimeError):
            estimate_max_batch(0, inputs)

    def test_multiple_of_one_disables_rounding(self):
        # Useful for tests that need exact integer division results.
        inputs = _make_inputs(window=(32, 32, 32))
        b1 = estimate_max_batch(40 * _GIB, inputs, multiple_of=1)
        b8 = estimate_max_batch(40 * _GIB, inputs, multiple_of=8)
        assert b1 >= b8
        assert b8 % 8 == 0

    def test_validates_headroom_fraction(self):
        inputs = _make_inputs()
        with pytest.raises(ValueError, match="headroom_fraction"):
            estimate_max_batch(48 * _GIB, inputs, headroom_fraction=-0.01)
        with pytest.raises(ValueError, match="headroom_fraction"):
            estimate_max_batch(48 * _GIB, inputs, headroom_fraction=1.0)

    def test_validates_min_max_batch(self):
        inputs = _make_inputs()
        with pytest.raises(ValueError, match="min_batch"):
            estimate_max_batch(48 * _GIB, inputs, min_batch=0)
        with pytest.raises(ValueError, match="max_batch"):
            estimate_max_batch(48 * _GIB, inputs, max_batch=0)
        with pytest.raises(ValueError, match="must not exceed"):
            estimate_max_batch(48 * _GIB, inputs, min_batch=64, max_batch=32)

    def test_validates_multiple_of(self):
        inputs = _make_inputs()
        with pytest.raises(ValueError, match="multiple_of"):
            estimate_max_batch(48 * _GIB, inputs, multiple_of=0)

    def test_negative_free_bytes_raises(self):
        inputs = _make_inputs()
        with pytest.raises(ValueError, match="free_bytes"):
            estimate_max_batch(-1, inputs)

    def test_recommended_batch_actually_fits_the_inequality(self):
        # The contract: resident + fixed + per_poi * B <= free * (1 - h).
        inputs = _make_inputs()
        free = 48 * _GIB
        h = 0.15
        b = estimate_max_batch(free, inputs, headroom_fraction=h)
        fp = kernel_footprint(inputs.window, inputs.mode, inputs.normalization)
        budgeted = resident_bytes(inputs) + fp.fixed_bytes + fp.per_poi_bytes * b
        assert budgeted <= int(free * (1.0 - h))

    def test_production_a6000_recommends_a_workable_batch(self):
        # Sanity: at 48 GB free (one A6000) with the production config,
        # the recommender should pick something in the same ballpark
        # as the dispatcher's hand-picked default of 64. The exact
        # number is calibration-dependent; we only pin order of
        # magnitude here so calibration tweaks don't break the test.
        inputs = _make_inputs()
        b = estimate_max_batch(48 * _GIB, inputs)
        assert 8 <= b <= 256


class TestRecommendBatchSize:
    def test_uses_minimum_free_across_devices(self, monkeypatch: pytest.MonkeyPatch):
        # Two synthetic devices: one tight, one comfortable. The
        # tight one must dominate.
        free_map = {0: 20 * _GIB, 1: 48 * _GIB}

        def fake_probe(d: int) -> tuple[int, int]:
            return free_map[d], 49 * _GIB

        monkeypatch.setattr(budget, "probe_free_vram", fake_probe)
        inputs = _make_inputs()

        cross = recommend_batch_size(inputs, [0, 1])
        only_min = estimate_max_batch(free_map[0], inputs, min_batch=8)
        assert cross == only_min

    def test_ceiling_clamps_below_estimate(self, monkeypatch: pytest.MonkeyPatch):
        free_map = {0: 48 * _GIB}
        monkeypatch.setattr(budget, "probe_free_vram", lambda d: (free_map[d], 49 * _GIB))
        inputs = _make_inputs(window=(32, 32, 32))
        # Without ceiling we'd get a much larger batch on a small
        # window; ceiling=64 must dominate.
        assert recommend_batch_size(inputs, [0], ceiling=64) == 64

    def test_empty_device_ids_raises(self):
        with pytest.raises(ValueError, match="device_ids"):
            recommend_batch_size(_make_inputs(), [])

    def test_underflow_propagates_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
        # 8 GB free per device on the production config will not fit
        # even at min_batch=8. recommend should re-raise the
        # estimate's RuntimeError unchanged.
        monkeypatch.setattr(budget, "probe_free_vram", lambda d: (8 * _GIB, 49 * _GIB))
        with pytest.raises(RuntimeError, match="VRAM budget too tight"):
            recommend_batch_size(_make_inputs(), [0, 1, 2, 3])

    def test_no_cupy_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Force the "CuPy unavailable" code path even when CuPy is
        # installed in the environment. Aligns the diagnostic with the
        # rest of the gpu/ tree (dispatch._resolve_device_ids has the
        # same shape).
        monkeypatch.setattr(budget, "_cp", None)
        with pytest.raises(RuntimeError, match="requires CuPy"):
            probe_free_vram(0)


@pytest.mark.gpu
class TestProbeFreeVramShape:
    def test_returns_positive_pair(self):
        free, total = probe_free_vram(0)
        assert free > 0
        assert total > 0
        assert free <= total

    def test_negative_device_id_raises(self):
        with pytest.raises(ValueError, match="device_id"):
            probe_free_vram(-1)


@pytest.mark.gpu
@pytest.mark.slow
class TestRecommendedBatchActuallyFits:
    """End-to-end empirical sanity check on the cost model.

    Runs one batched FFT NCC step at the recommended batch on a small
    synthetic volume; if the cost model is significantly off, the
    helper either OOMs (model too aggressive) or trivially fits with
    the tiniest batch (model too conservative -- fine, but flag-worthy
    in review). This test only catches the "OOM" failure mode.
    """

    def test_recommended_batch_runs_without_oom(self):
        # Skip cleanly if CuPy is missing (the @gpu marker filters this
        # out on CI but local invocations may still hit the body).
        cp = pytest.importorskip("cupy")

        from mamba_dvc.core.grid import build_grid
        from mamba_dvc.pipeline._internal import (
            TUKEY_DEFAULTS,
            correlate_admitted_subset,
        )

        # Modest volume so the test stays under the slow-marker
        # threshold; the budget model still does its arithmetic on
        # the requested shape.
        shape = (96, 96, 96)
        rng = np.random.default_rng(7)
        ref = rng.standard_normal(shape, dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)
        mask = np.ones(shape, dtype=np.bool_)

        with cp.cuda.Device(0):
            ref_d = cp.asarray(ref)
            def_d = cp.asarray(deformed)
            mask_d = cp.asarray(mask)

            grid = build_grid(shape, window=24, overlap=0.5)
            grid_d = type(grid)(
                positions=cp.asarray(grid.positions),
                starts=cp.asarray(grid.starts),
                grid_shape=grid.grid_shape,
                spacing=grid.spacing,
                window=grid.window,
                volume_shape=grid.volume_shape,
            )
            n_points = int(math.prod(grid.grid_shape))
            admitted_idx = cp.arange(n_points, dtype=cp.int64)

            inputs = _make_inputs(
                volume_shape=shape,
                window=grid.window,
                mode=NCCMode.LINEAR,
                normalization=NCCNormalization.OVERLAP,
                has_mask=True,
                deformed_mask_distinct=False,
            )
            batch = recommend_batch_size(inputs, [0])

            disp, conf, status = correlate_admitted_subset(
                ref_d,
                def_d,
                mask_d,
                mask_d,
                grid_d,
                admitted_idx,
                search_radius=8,
                batch_size=batch,
                eps=1e-12,
                ncc_mode=NCCMode.LINEAR,
                ncc_normalization=NCCNormalization.OVERLAP,
                tukey_alpha=TUKEY_DEFAULTS[NCCMode.LINEAR],
            )
            # Surface a clean failure if the helper produced nothing
            # usable -- but the actual assertion is "no exception was
            # raised above", which is what the OOM check needs.
            assert disp.shape == (n_points, 3)
            assert conf.shape == (n_points,)
            assert status.shape == (n_points,)
