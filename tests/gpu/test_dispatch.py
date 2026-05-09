"""Tests for ``mamba_dvc.gpu.dispatch``.

The pure sharding helpers can be exercised on any machine. The
end-to-end correlate-multi-gpu tests are marked ``gpu`` (and ``slow``)
because they spawn worker processes that materialize CuPy arrays on
device. They assert bit-identical results between the single-device
:func:`mamba_dvc.pipeline.correlate.correlate` and the multi-process
dispatch path -- this is the regression target for the dispatch
seam (helper return contract + per-shard scatter).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from mamba_dvc.core.grid import build_grid
from mamba_dvc.gpu.dispatch import _shard_admitted_indices, correlate_multi_gpu
from mamba_dvc.pipeline.correlate import correlate
from mamba_dvc.types import POIStatus
from mamba_dvc.validate.synthetic import make_pair, rigid_shift


class TestShardAdmittedIndices:
    """Pure-logic tests for the lattice-Z partition. No GPU needed."""

    def test_partition_covers_all_admitted_disjointly(self):
        # Build a 6x4x4 lattice with no mask -- every POI is admitted.
        grid = build_grid((96, 64, 64), window=16, overlap=0.0)
        n_points = int(np.prod(grid.grid_shape))
        admitted_idx = np.arange(n_points, dtype=np.int64)

        for n_shards in (1, 2, 3, 4):
            shards = _shard_admitted_indices(admitted_idx, grid, n_shards)
            assert len(shards) == n_shards
            union = np.concatenate(shards)
            np.testing.assert_array_equal(np.sort(union), admitted_idx)

    def test_contiguous_z_slabs(self):
        # Each shard's POIs decode to a contiguous range of lattice-Z
        # rows. This is the spatial-locality property plan §6 calls out.
        grid = build_grid((192, 64, 64), window=16, overlap=0.0)
        n_points = int(np.prod(grid.grid_shape))
        admitted_idx = np.arange(n_points, dtype=np.int64)
        ny, nx = grid.grid_shape[1], grid.grid_shape[2]

        shards = _shard_admitted_indices(admitted_idx, grid, 4)
        z_ranges: list[tuple[int, int]] = []
        for shard in shards:
            if shard.size == 0:
                continue
            z_lat = shard // (ny * nx)
            z_ranges.append((int(z_lat.min()), int(z_lat.max())))
        # Ranges must be ascending and non-overlapping.
        for (_lo_a, hi_a), (lo_b, _hi_b) in pairwise(z_ranges):
            assert hi_a < lo_b, f"slabs overlap or out of order: {z_ranges}"

    def test_handles_sparse_admission(self):
        # Mask out the middle Z-slab so the central shard ends up empty
        # while the outer two carry POIs.
        grid = build_grid((96, 32, 32), window=16, overlap=0.0)
        nz, ny, nx = grid.grid_shape
        all_idx = np.arange(int(np.prod(grid.grid_shape)), dtype=np.int64)
        z_lat_all = all_idx // (ny * nx)
        # Drop the middle third of Z entirely.
        keep = (z_lat_all < nz // 3) | (z_lat_all >= 2 * nz // 3)
        admitted_idx = all_idx[keep]

        shards = _shard_admitted_indices(admitted_idx, grid, 3)
        assert len(shards) == 3
        sizes = [int(s.size) for s in shards]
        assert sizes[0] > 0, "first slab should have POIs"
        assert sizes[2] > 0, "last slab should have POIs"
        # Middle slab may be empty depending on rounding -- assert
        # it's a proper subset of admitted, not that it's empty.
        union = np.concatenate(shards)
        np.testing.assert_array_equal(np.sort(union), admitted_idx)

    def test_rejects_non_positive_shards(self):
        grid = build_grid((48, 32, 32), window=16, overlap=0.0)
        admitted = np.arange(int(np.prod(grid.grid_shape)), dtype=np.int64)
        with pytest.raises(ValueError, match="n_shards"):
            _shard_admitted_indices(admitted, grid, 0)

    def test_empty_admitted_returns_empty_shards(self):
        grid = build_grid((48, 32, 32), window=16, overlap=0.0)
        empty = np.empty(0, dtype=np.int64)
        shards = _shard_admitted_indices(empty, grid, 3)
        assert len(shards) == 3
        for shard in shards:
            assert shard.size == 0


def _assert_fields_match(a, b, *, atol_disp: float = 1e-5, atol_conf: float = 1e-5) -> None:
    """Tight tolerance: parent vs. helper-on-GPU should be bit-identical
    modulo float32 reduction order. We allow a small atol to account
    for FFT operator scheduling differences between the host NumPy
    fallback and CuPy's cuFFT.
    """
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_allclose(a.displacements, b.displacements, atol=atol_disp)
    np.testing.assert_allclose(a.confidence, b.confidence, atol=atol_conf)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.grid_shape == b.grid_shape
    assert a.spacing == b.spacing
    assert a.window == b.window


@pytest.mark.gpu
class TestCorrelateMultiGpu:
    """End-to-end dispatch tests; require CuPy + at least one GPU."""

    @staticmethod
    def _smooth_pair(
        shape: tuple[int, int, int] = (48, 48, 48),
        shift: tuple[float, float, float] = (0.4, -0.6, 0.9),
        seed: int = 17,
    ):
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=seed)
        return pair.reference, pair.deformed

    @pytest.mark.slow
    def test_single_gpu_inprocess_matches_correlate(self):
        # device_ids=[0] takes the in-process happy path -- no spawn,
        # just upload-helper-D2H. It still routes through the same
        # _correlate_admitted_subset core that correlate() exercises on
        # host, so the result must match within float32 noise.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        host_field = correlate(ref, deformed, **kwargs)
        gpu_field = correlate_multi_gpu(ref, deformed, device_ids=[0], **kwargs)

        _assert_fields_match(host_field, gpu_field)

    @pytest.mark.slow
    def test_multi_process_dispatch_matches_correlate(self):
        # device_ids=[0, 0] spawns two workers pinned to GPU 0. This
        # exercises every part of the dispatch plumbing -- shm publish,
        # spawn, attach, helper run, D2H, pipe send, parent gather and
        # scatter -- on a single-GPU dev host, without requiring a
        # 4xA6000 box. With (single-pair, batch=64), both workers
        # together fit comfortably in one device's VRAM.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=32)

        host_field = correlate(ref, deformed, **kwargs)
        gpu_field = correlate_multi_gpu(ref, deformed, device_ids=[0, 0], **kwargs)

        _assert_fields_match(host_field, gpu_field)

    @pytest.mark.slow
    def test_auto_batch_matches_explicit_batch(self):
        # Plan §10.1 contract: ``batch_size="auto"`` produces a field
        # bit-equivalent (within float32 noise) to one produced with an
        # explicit numeric batch. Batch invariance is the underlying
        # property -- the pipeline.correlate.TestBatchInvariance test
        # pins it on the host side; this test pins it through the
        # auto-batch resolution path on a real device.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8)

        explicit = correlate_multi_gpu(ref, deformed, device_ids=[0], batch_size=32, **kwargs)
        auto = correlate_multi_gpu(ref, deformed, device_ids=[0], batch_size="auto", **kwargs)

        _assert_fields_match(explicit, auto)

    @pytest.mark.slow
    def test_multi_process_with_mask_admission(self):
        # Mask admission stratifies POIs into MASKED vs. processed. The
        # parent computes admission once and shards only the admitted
        # set; this test pins that the multi-process gather correctly
        # carries MASKED slots through (they live outside any shard's
        # admitted_idx and stay MASKED in the merged buffers).
        rng = np.random.default_rng(31)
        shape = (48, 48, 48)
        ref = rng.standard_normal(shape, dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)
        mask = np.ones(shape, dtype=np.bool_)
        mask[:8, :, :] = False  # exclude the first 8 z-slices

        kwargs = dict(
            window=24,
            overlap=0.5,
            mask_threshold=0.9,
            search_radius=8,
            batch_size=32,
        )

        host_field = correlate(ref, deformed, mask, **kwargs)
        gpu_field = correlate_multi_gpu(ref, deformed, mask, device_ids=[0, 0], **kwargs)

        _assert_fields_match(host_field, gpu_field)
        # Sanity: at least some POIs were rejected by admission.
        assert (host_field.status == POIStatus.MASKED).sum() > 0


@pytest.mark.gpu
class TestCorrelateMultiGpuValidation:
    def test_empty_device_ids_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="device_ids"):
            correlate_multi_gpu(ref, ref, device_ids=[], window=16)

    def test_shape_mismatch_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        deformed = np.zeros((32, 32, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            correlate_multi_gpu(ref, deformed, device_ids=[0], window=16)

    def test_wrong_dtype_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float64)
        deformed = np.zeros((32, 32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            correlate_multi_gpu(ref, deformed, device_ids=[0], window=16)  # type: ignore[arg-type]


class TestAutoBatchValidation:
    """Pure validation of the new ``batch_size`` literal contract.

    These run before ``_resolve_device_ids`` -- so no GPU is required
    and no patching is needed.
    """

    def test_zero_int_batch_size_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="positive int or 'auto'"):
            correlate_multi_gpu(ref, ref, window=16, batch_size=0)

    def test_unknown_string_batch_size_raises(self):
        ref = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="positive int or 'auto'"):
            correlate_multi_gpu(ref, ref, window=16, batch_size="big")  # type: ignore[arg-type]


class TestAutoBatchResolution:
    """``batch_size="auto"`` routes through the budget oracle.

    The dispatch helpers (``_run_single_process`` /
    ``_run_multi_process``) and the device-id resolver are patched out
    so the test does not require CUDA. The cold-probe ordering, the
    structured stderr log, and the BudgetInputs construction are
    exercised end-to-end on a CPU host.
    """

    @staticmethod
    def _stub_run(*args: Any, **_: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Single- and multi-process run helpers share the same prefix:
        # (reference, deformed, eff_mask, eff_def_mask, grid,
        # admitted_idx, device_id_or_ids, helper_kwargs). We only need
        # the grid (args[4]) to size the synthetic return.
        grid = args[4]
        n = int(np.prod(grid.grid_shape))
        return (
            np.zeros((n, 3), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.full(n, POIStatus.OK, dtype=np.uint8),
        )

    def _patch_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        free_per_device: dict[int, int],
        captured: dict[str, Any],
    ) -> None:
        from mamba_dvc.gpu import dispatch as dispatch_mod

        device_ids = list(free_per_device.keys())

        def fake_probe(d: int) -> tuple[int, int]:
            return free_per_device[d], 49 * (1024**3)

        def capturing_run(
            *args: Any, **kwargs: Any
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            captured["helper_kwargs"] = args[-1]
            return self._stub_run(*args, **kwargs)

        monkeypatch.setattr(dispatch_mod, "_resolve_device_ids", lambda _x: device_ids)
        monkeypatch.setattr(dispatch_mod, "probe_free_vram", fake_probe)
        monkeypatch.setattr(dispatch_mod, "_run_single_process", capturing_run)
        monkeypatch.setattr(dispatch_mod, "_run_multi_process", capturing_run)

    def test_auto_emits_structured_stderr_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        captured: dict[str, Any] = {}
        self._patch_dispatch(
            monkeypatch,
            free_per_device={0: 32 * (1024**3), 1: 28 * (1024**3)},
            captured=captured,
        )

        rng = np.random.default_rng(42)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)

        correlate_multi_gpu(ref, deformed, window=24, overlap=0.5, search_radius=8)

        err = capsys.readouterr().err
        # All seven structured fields must appear -- changing any of
        # them is a tooling break, not a refactor.
        for token in (
            "auto.batch_size=",
            "auto.free_bytes_min=",
            "auto.free_bytes_per_device=",
            "auto.resident_bytes=",
            "auto.per_poi_bytes=",
            "auto.fixed_bytes=",
            "auto.headroom=",
        ):
            assert token in err, f"missing token in stderr log: {token!r}\n{err}"
        # Per-device map carries both ids.
        assert "0:" in err and "1:" in err

    def test_auto_passes_resolved_int_to_helper(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}
        self._patch_dispatch(
            monkeypatch,
            free_per_device={0: 40 * (1024**3)},
            captured=captured,
        )

        rng = np.random.default_rng(7)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)

        correlate_multi_gpu(ref, deformed, window=24, overlap=0.5, search_radius=8)

        # The helper sees a resolved int, never the literal "auto".
        helper_batch = captured["helper_kwargs"]["batch_size"]
        assert isinstance(helper_batch, int)
        assert helper_batch > 0

    def test_auto_uses_minimum_free_across_devices(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        # Tight device must dominate. Compare the resolved batch from a
        # ``[0:32G, 1:8G]`` fleet against a ``[1:8G]`` solo run; they
        # must match because the cross-device rule is min-of-free.
        rng = np.random.default_rng(13)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)

        captured_dual: dict[str, Any] = {}
        self._patch_dispatch(
            monkeypatch,
            free_per_device={0: 32 * (1024**3), 1: 8 * (1024**3)},
            captured=captured_dual,
        )
        correlate_multi_gpu(ref, deformed, window=24, overlap=0.5, search_radius=8)
        dual_batch = captured_dual["helper_kwargs"]["batch_size"]
        # Drain the stderr emitted during the first run before the
        # second so capsys does not conflate the two log lines.
        capsys.readouterr()

        captured_solo: dict[str, Any] = {}
        self._patch_dispatch(
            monkeypatch,
            free_per_device={1: 8 * (1024**3)},
            captured=captured_solo,
        )
        correlate_multi_gpu(ref, deformed, window=24, overlap=0.5, search_radius=8)
        solo_batch = captured_solo["helper_kwargs"]["batch_size"]

        assert dual_batch == solo_batch

    def test_explicit_int_skips_resolver_and_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        from mamba_dvc.gpu import dispatch as dispatch_mod

        captured: dict[str, Any] = {}

        def capturing_run(
            *args: Any, **kwargs: Any
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            captured["helper_kwargs"] = args[-1]
            return self._stub_run(*args, **kwargs)

        monkeypatch.setattr(dispatch_mod, "_resolve_device_ids", lambda _x: [0])
        monkeypatch.setattr(dispatch_mod, "_run_single_process", capturing_run)

        # Probe must NOT be called for an explicit int batch.
        def fail_probe(_d: int) -> tuple[int, int]:
            raise AssertionError("probe_free_vram must not be called for int batch_size")

        monkeypatch.setattr(dispatch_mod, "probe_free_vram", fail_probe)

        rng = np.random.default_rng(91)
        ref = rng.standard_normal((48, 48, 48), dtype=np.float32)
        deformed = np.roll(ref, shift=1, axis=0)

        correlate_multi_gpu(
            ref, deformed, window=24, overlap=0.5, search_radius=8, batch_size=64
        )

        assert captured["helper_kwargs"]["batch_size"] == 64
        # No structured log line either.
        assert "auto." not in capsys.readouterr().err
