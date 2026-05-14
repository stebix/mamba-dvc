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

import json
import logging
import multiprocessing as mp
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mamba_dvc.core.grid import build_grid
from mamba_dvc.gpu.dispatch import (
    MultiGPUDispatcher,
    _NonBlockingQueueHandler,
    _shard_admitted_indices,
    _status_histogram,
    _WorkerLogFilter,
    correlate_multi_gpu,
)
from mamba_dvc.pipeline.correlate import correlate
from mamba_dvc.run.eventlog import SessionScope
from mamba_dvc.types import POIStatus
from mamba_dvc.validate.synthetic import make_pair, rigid_shift


class _RecordingDispatchObserver:
    """Observer that records every hook call for assertion in tests.

    No structlog import — exercises the
    :class:`mamba_dvc.types.DispatchObserver` protocol shape directly,
    so CPU-only and GPU tests can both reach for the same recorder.
    """

    def __init__(self) -> None:
        self.starts: list[tuple[int, int, int]] = []
        self.ends: list[tuple[dict[POIStatus, int], int]] = []

    def on_pair_start(self, *, volume_shape: tuple[int, int, int]) -> None:
        self.starts.append(volume_shape)

    def on_pair_end(self, *, status_counts: dict[POIStatus, int], n_valid: int) -> None:
        self.ends.append((dict(status_counts), n_valid))


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


class TestStatusHistogram:
    """Pure-helper coverage for :func:`_status_histogram`. No GPU needed."""

    def test_only_observed_statuses_appear(self):
        # Mixed status array. Absent statuses (LOW_CONF, OUT_OF_RANGE)
        # must not appear as zero entries — the Protocol contract says
        # zero-count members are omitted.
        status = np.array(
            [
                POIStatus.OK,
                POIStatus.OK,
                POIStatus.OK,
                POIStatus.MASKED,
                POIStatus.OUTLIER,
            ],
            dtype=np.uint8,
        )
        counts = _status_histogram(status)
        assert counts == {
            POIStatus.OK: 3,
            POIStatus.MASKED: 1,
            POIStatus.OUTLIER: 1,
        }

    def test_empty_status_array_returns_empty_dict(self):
        # Empty (zero-POI) input returns an empty dict, not a dict of
        # zero-valued entries.
        status = np.empty(0, dtype=np.uint8)
        assert _status_histogram(status) == {}

    def test_all_ok_only_returns_ok_entry(self):
        status = np.full(10, POIStatus.OK, dtype=np.uint8)
        assert _status_histogram(status) == {POIStatus.OK: 10}

    def test_keys_are_poistatus_members(self):
        # Keys must be ``POIStatus`` members so consumers can call
        # ``.name`` for human-readable labels — DispatchLogger relies on
        # this in the structlog-rendered ``status_counts`` field.
        status = np.array([POIStatus.OK, POIStatus.MASKED], dtype=np.uint8)
        counts = _status_histogram(status)
        for k in counts:
            assert isinstance(k, POIStatus)


def _make_phase_record(name: str = "mamba_dvc.timing") -> logging.LogRecord:
    """Build a minimal LogRecord that mimics what ``log_phase`` emits.

    The worker-side filter / handler stamp extras on top of a record;
    we don't need a full ``logging.makeLogRecord`` round-trip for unit
    coverage of the stamping/overflow contracts.
    """
    return logging.LogRecord(
        name=name,
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg="phase=test.phase dt=0.0010s",
        args=None,
        exc_info=None,
    )


class TestWorkerLogFilter:
    """Pure-logic coverage for the L3 worker-side log filter. No GPU needed."""

    def test_stamps_device_id_on_every_record(self):
        # The device_id is bound at filter construction (pinned per
        # worker process) and must surface on every record so the
        # listener's events.jsonl line carries the originating GPU.
        f = _WorkerLogFilter(device_id=2)
        record = _make_phase_record()
        assert f.filter(record) is True
        assert record.mdvc_device_id == 2

    def test_set_pair_context_stamps_every_key(self):
        # Pair-context dict flows from the parent on each pair
        # request; the filter copies each entry to ``mdvc_<key>`` so
        # ``_promote_mdvc_fields`` on the parent side strips the
        # prefix into a top-level events.jsonl field.
        f = _WorkerLogFilter(device_id=0)
        f.set_pair_context({"t_ref": 3, "t_def": 5, "session_id": "abc"})
        record = _make_phase_record()
        assert f.filter(record) is True
        assert record.mdvc_t_ref == 3
        assert record.mdvc_t_def == 5
        assert record.mdvc_session_id == "abc"

    def test_set_pair_context_to_none_clears_previous(self):
        # End-of-pair cleanup: workers call set_pair_context(None) in
        # the per-pair finally block so the *next* pair's records
        # don't inherit the previous pair's coords.
        f = _WorkerLogFilter(device_id=0)
        f.set_pair_context({"t_ref": 7})
        f.set_pair_context(None)
        record = _make_phase_record()
        f.filter(record)
        assert not hasattr(record, "mdvc_t_ref")

    def test_does_not_overwrite_existing_mdvc_attrs(self):
        # ``log_phase`` already puts ``mdvc_phase`` / ``mdvc_seconds`` on
        # the record. The filter must not clobber them when a pair
        # context happens to share a key.
        f = _WorkerLogFilter(device_id=0)
        f.set_pair_context({"phase": "should_not_overwrite"})
        record = _make_phase_record()
        record.mdvc_phase = "ncc.fft_ref"
        f.filter(record)
        assert record.mdvc_phase == "ncc.fft_ref"


class TestNonBlockingQueueHandler:
    """Worker-side overflow contract for the L3 queue bridge. No GPU needed."""

    def test_drops_on_full_and_counts(self):
        # Maxsize=1: first put fills it; second put would block under
        # the stdlib QueueHandler default, but this subclass swallows
        # queue.Full and increments the counter so the worker compute
        # path is never throttled.
        q: Any = mp.get_context("spawn").Queue(maxsize=1)
        handler = _NonBlockingQueueHandler(q)
        for _ in range(3):
            handler.emit(_make_phase_record())
        # Two records were rejected; one made it into the queue.
        assert handler.drop_count == 2
        # The single buffered record is still readable; emit() did not
        # corrupt the queue.
        msg = q.get_nowait()
        assert isinstance(msg, logging.LogRecord)
        # Drain the queue and join the feeder so pytest's reaper
        # doesn't trip on a still-running daemon thread.
        q.close()
        q.join_thread()

    def test_no_drops_when_under_capacity(self):
        q: Any = mp.get_context("spawn").Queue(maxsize=8)
        handler = _NonBlockingQueueHandler(q)
        for _ in range(4):
            handler.emit(_make_phase_record())
        assert handler.drop_count == 0
        q.close()
        q.join_thread()


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


class TestMultiGPUDispatcherValidation:
    """Constructor + property tests that need no GPU.

    The :class:`MultiGPUDispatcher` constructor is deliberately
    side-effect free: it validates shapes/dtypes/kwargs and stores
    config, but does not import CuPy or touch a device. The work that
    needs a GPU happens in :meth:`__enter__`. These tests pin that
    contract.
    """

    def test_requires_some_way_to_infer_shape(self):
        with pytest.raises(ValueError, match="volume_shape"):
            MultiGPUDispatcher(device_ids=[0])

    def test_volume_shape_inferred_from_mask(self):
        mask = np.ones((32, 24, 24), dtype=np.bool_)
        d = MultiGPUDispatcher(device_ids=[0], mask=mask, window=16)
        assert d.volume_shape == (32, 24, 24)
        assert not d.has_anchored_reference

    def test_volume_shape_inferred_from_anchored_reference(self):
        ref = np.zeros((24, 24, 24), dtype=np.float32)
        d = MultiGPUDispatcher(device_ids=[0], anchored_reference=ref, window=16)
        assert d.volume_shape == (24, 24, 24)
        assert d.has_anchored_reference

    def test_volume_shape_must_match_supplied_mask(self):
        mask = np.ones((32, 32, 32), dtype=np.bool_)
        with pytest.raises(ValueError, match="mask shape"):
            MultiGPUDispatcher(device_ids=[0], mask=mask, volume_shape=(16, 16, 16), window=8)

    def test_mask_dtype_must_be_bool(self):
        mask = np.ones((32, 32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool"):
            MultiGPUDispatcher(device_ids=[0], mask=mask, window=16)  # type: ignore[arg-type]

    def test_anchored_reference_dtype_must_be_float32(self):
        ref = np.zeros((32, 32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            MultiGPUDispatcher(
                device_ids=[0],
                anchored_reference=ref,
                window=16,  # type: ignore[arg-type]
            )

    def test_zero_int_batch_size_rejected(self):
        with pytest.raises(ValueError, match="positive int or 'auto'"):
            MultiGPUDispatcher(
                device_ids=[0],
                volume_shape=(32, 32, 32),
                window=16,
                batch_size=0,
            )

    def test_unknown_string_batch_size_rejected(self):
        with pytest.raises(ValueError, match="positive int or 'auto'"):
            MultiGPUDispatcher(
                device_ids=[0],
                volume_shape=(32, 32, 32),
                window=16,
                batch_size="big",  # type: ignore[arg-type]
            )

    def test_correlate_outside_context_raises(self):
        d = MultiGPUDispatcher(device_ids=[0], volume_shape=(16, 16, 16), window=8)
        ref = np.zeros((16, 16, 16), dtype=np.float32)
        with pytest.raises(RuntimeError, match="with block"):
            d.correlate(ref, ref)

    def test_post_open_properties_unavailable_before_enter(self):
        # ``device_ids`` and ``is_multiprocess`` derive from CuPy's
        # device-visibility view, which is only consulted at __enter__.
        # Accessing them pre-open is a programmer error -- pin that
        # contract so callers don't get a silent empty-tuple sentinel.
        d = MultiGPUDispatcher(device_ids=[0], volume_shape=(16, 16, 16), window=8)
        with pytest.raises(RuntimeError, match="with block"):
            _ = d.is_multiprocess
        with pytest.raises(RuntimeError, match="with block"):
            _ = d.device_ids

    def test_constructor_accepts_dispatch_observer(self):
        # CPU-only contract pin: the dispatch_observer kwarg is
        # optional and storing it does not require CuPy (no
        # __enter__). The observer is stored verbatim — consumers can
        # keep their reference and read from it after the with block.
        observer = _RecordingDispatchObserver()
        d = MultiGPUDispatcher(
            device_ids=[0],
            volume_shape=(16, 16, 16),
            window=8,
            dispatch_observer=observer,
        )
        assert d._dispatch_observer is observer

    def test_dispatch_observer_defaults_to_none(self):
        # Backward compatibility pin: omitting dispatch_observer must
        # leave the dispatcher side-effect free, matching pre-L2
        # behavior bit-for-bit.
        d = MultiGPUDispatcher(device_ids=[0], volume_shape=(16, 16, 16), window=8)
        assert d._dispatch_observer is None

    def test_emit_phase_records_defaults_to_false(self):
        # L3 contract: bit-identical pre-L3 multi-process behavior
        # when the flag is omitted. The dispatcher must not allocate
        # the queue / start a listener thread until explicitly opted
        # in.
        d = MultiGPUDispatcher(device_ids=[0], volume_shape=(16, 16, 16), window=8)
        assert d._emit_phase_records is False
        assert d._log_queue is None
        assert d._log_listener is None

    def test_emit_phase_records_stored_when_set(self):
        d = MultiGPUDispatcher(
            device_ids=[0],
            volume_shape=(16, 16, 16),
            window=8,
            emit_phase_records=True,
        )
        assert d._emit_phase_records is True
        # Queue/listener stay None until __enter__ (and only when the
        # opened dispatcher routes to multi-process); the constructor
        # does not allocate IPC primitives.
        assert d._log_queue is None
        assert d._log_listener is None


@pytest.mark.gpu
class TestMultiGPUDispatcherInProcess:
    """End-to-end tests for the single-device in-process dispatcher path.

    Routes through the same ``correlate_admitted_subset`` helper as
    :func:`correlate_multi_gpu`, so the results must match the
    one-shot shim within float32 noise on identical inputs.
    """

    @staticmethod
    def _smooth_pair(
        shape: tuple[int, int, int] = (48, 48, 48),
        shift: tuple[float, float, float] = (0.4, -0.6, 0.9),
        seed: int = 17,
    ):
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=seed)
        return pair.reference, pair.deformed

    @pytest.mark.slow
    def test_single_device_open_close_skips_spawn(self):
        # Buildout doc S10 ``TestEnterExitNoSpawn``: opening a
        # dispatcher with one device must not create a ``mp.Process``.
        # We patch the spawn context's ``Process`` factory to fail loudly
        # if it gets called.
        from mamba_dvc.gpu import dispatch as dispatch_mod

        ref, _deformed = self._smooth_pair()
        ctx = dispatch_mod.mp.get_context("spawn")

        class _NoSpawn:
            def Process(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
                raise AssertionError("single-device MultiGPUDispatcher must not spawn workers")

            def Pipe(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
                return ctx.Pipe(*args, **kwargs)

        # mp.get_context is called inside _enter_multiprocess only.
        # Patching at module level guarantees the multi-process path
        # would be detected even if the routing decision regressed.
        with pytest.MonkeyPatch.context() as mp_ctx:
            mp_ctx.setattr(dispatch_mod.mp, "get_context", lambda _name: _NoSpawn())
            with MultiGPUDispatcher(
                device_ids=[0],
                anchored_reference=ref,
                window=24,
                overlap=0.5,
                search_radius=8,
            ) as d:
                assert d.is_multiprocess is False
                assert d.device_ids == (0,)

    @pytest.mark.slow
    def test_single_device_matches_correlate_multi_gpu(self):
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        shim_field = correlate_multi_gpu(ref, deformed, device_ids=[0], **kwargs)

        with MultiGPUDispatcher(
            device_ids=[0],
            volume_shape=ref.shape,
            **kwargs,
        ) as d:
            disp_field = d.correlate(ref, deformed)

        _assert_fields_match(shim_field, disp_field)

    @pytest.mark.slow
    def test_anchored_reference_path_matches_explicit_reference(self):
        # Passing ``reference=None`` against the resident anchored ref
        # must yield bit-identical (within float32 noise) results to
        # passing the same reference explicitly.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with MultiGPUDispatcher(
            device_ids=[0],
            anchored_reference=ref,
            **kwargs,
        ) as d:
            anchored_field = d.correlate(None, deformed)
            explicit_field = d.correlate(ref, deformed)

        _assert_fields_match(anchored_field, explicit_field)

    @pytest.mark.slow
    def test_repeated_pairs_keep_workers_alive(self):
        # Same dispatcher, three pairs. Each .correlate() call must
        # succeed and the device_ids tuple must not change between
        # calls (the workers / in-process context are reused).
        ref, _ = self._smooth_pair(shift=(0.0, 0.0, 0.0))
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with MultiGPUDispatcher(
            device_ids=[0],
            anchored_reference=ref,
            **kwargs,
        ) as d:
            ids_before = d.device_ids
            for shift in ((0.3, 0.0, 0.0), (-0.4, 0.5, 0.0), (0.0, -0.6, 0.7)):
                _, deformed = self._smooth_pair(shift=shift)
                field = d.correlate(None, deformed)
                assert field.displacements.shape[1] == 3
            assert d.device_ids == ids_before

    @pytest.mark.slow
    def test_dispatch_observer_fires_around_each_pair(self):
        # End-to-end: with a DispatchObserver supplied at construction
        # the dispatcher must fire on_pair_start before the dispatch
        # work and on_pair_end after the outlier test, once per
        # ``.correlate()`` call. Three pairs → three start/end pairs.
        ref, _ = self._smooth_pair(shift=(0.0, 0.0, 0.0))
        observer = _RecordingDispatchObserver()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with MultiGPUDispatcher(
            device_ids=[0],
            anchored_reference=ref,
            dispatch_observer=observer,
            **kwargs,
        ) as d:
            for shift in ((0.3, 0.0, 0.0), (-0.4, 0.5, 0.0), (0.0, -0.6, 0.7)):
                _, deformed = self._smooth_pair(shift=shift)
                d.correlate(None, deformed)
            assert d.volume_shape == ref.shape

        assert len(observer.starts) == 3
        assert len(observer.ends) == 3
        for vshape in observer.starts:
            assert vshape == ref.shape

    @pytest.mark.slow
    def test_dispatch_observer_status_counts_match_returned_field(self):
        # n_valid handed to the observer and the per-status histogram
        # must agree with the returned DisplacementField — they are
        # views of the same merged status array post-outlier-rejection.
        ref, deformed = self._smooth_pair()
        observer = _RecordingDispatchObserver()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with MultiGPUDispatcher(
            device_ids=[0],
            volume_shape=ref.shape,
            dispatch_observer=observer,
            **kwargs,
        ) as d:
            field = d.correlate(ref, deformed)

        assert len(observer.ends) == 1
        status_counts, n_valid = observer.ends[0]
        assert n_valid == int(field.valid.sum())
        # Every reported count must agree with a recount on field.status.
        for member, count in status_counts.items():
            assert count == int(np.count_nonzero(field.status == member))
        # And every status that occurred on the field must show up in
        # the histogram (the omit-zero-counts contract).
        for member in POIStatus:
            occurred = int(np.count_nonzero(field.status == member))
            if occurred > 0:
                assert status_counts.get(member) == occurred
            else:
                assert member not in status_counts


@pytest.mark.gpu
class TestMultiGPUDispatcherMultiProcessPersistence:
    """Persistent multi-process dispatcher returns independent buffers per pair.

    Regression coverage for the buffer-aliasing bug captured in
    ``docs/triage/multi-gpu-aliasing-bug.md``: ``_dispatch_pair_mp``
    used to reuse dispatcher-scoped scratch, so every
    :class:`~mamba_dvc.types.DisplacementField` stored on
    :class:`~mamba_dvc.types.DisplacementSeries.fields` aliased the
    last call's data. ``device_ids=[0, 0]`` forces the multi-process
    path on a single-GPU dev host.
    """

    @staticmethod
    def _series(
        shape: tuple[int, int, int] = (96, 96, 96),
        shifts: tuple[tuple[float, float, float], ...] = (
            (0.5, -0.4, 0.3),
            (1.0, -0.8, 0.6),
            (1.5, -1.2, 0.9),
        ),
        seed: int = 17,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Shared reference + N deformed volumes from a fixed-seed texture."""
        ref = make_pair(shape=shape, field=rigid_shift((0.0, 0.0, 0.0)), seed=seed).reference
        deformeds = tuple(
            make_pair(shape=shape, field=rigid_shift(shift), seed=seed).deformed
            for shift in shifts
        )
        return ref, deformeds

    @staticmethod
    def _open_persistent(ref: np.ndarray) -> MultiGPUDispatcher:
        return MultiGPUDispatcher(
            device_ids=[0, 0],
            anchored_reference=ref,
            window=32,
            overlap=0.5,
            search_radius=8,
            batch_size=32,
        )

    @pytest.mark.slow
    def test_field_buffers_do_not_alias_across_pairs(self):
        # Direct catch: every pair of fields must own disjoint
        # displacements / confidence / status arrays.
        ref, deformeds = self._series()
        with self._open_persistent(ref) as d:
            fields = tuple(d.correlate(None, dfm) for dfm in deformeds)

        for i in range(len(fields)):
            for j in range(len(fields)):
                if i == j:
                    continue
                assert not np.shares_memory(
                    fields[i].displacements, fields[j].displacements
                ), f"displacements alias between fields[{i}] and fields[{j}]"
                assert not np.shares_memory(fields[i].confidence, fields[j].confidence), (
                    f"confidence aliases between fields[{i}] and fields[{j}]"
                )
                assert not np.shares_memory(fields[i].status, fields[j].status), (
                    f"status aliases between fields[{i}] and fields[{j}]"
                )

    @pytest.mark.slow
    def test_field_values_persist_after_subsequent_pairs(self):
        # User-visible symptom: a field handed out by the dispatcher
        # must not change when the next pair runs.
        ref, deformeds = self._series()
        with self._open_persistent(ref) as d:
            first = d.correlate(None, deformeds[0])
            snapshot_disp = first.displacements.copy()
            snapshot_conf = first.confidence.copy()
            snapshot_stat = first.status.copy()
            snapshot_valid = first.valid.copy()
            _ = d.correlate(None, deformeds[1])
            _ = d.correlate(None, deformeds[2])

        np.testing.assert_array_equal(first.displacements, snapshot_disp)
        np.testing.assert_array_equal(first.confidence, snapshot_conf)
        np.testing.assert_array_equal(first.status, snapshot_stat)
        np.testing.assert_array_equal(first.valid, snapshot_valid)

    @pytest.mark.slow
    def test_valid_matches_status_for_every_returned_field(self):
        # ``DisplacementField.valid`` is documented as a convenience
        # view of ``status == POIStatus.OK``. The alias bug silently
        # violated it after the first pair. Pin the invariant.
        ref, deformeds = self._series()
        with self._open_persistent(ref) as d:
            fields = tuple(d.correlate(None, dfm) for dfm in deformeds)

        for idx, field in enumerate(fields):
            np.testing.assert_array_equal(
                field.valid,
                field.status == POIStatus.OK,
                err_msg=f"valid/status disagree on fields[{idx}]",
            )

    @pytest.mark.slow
    def test_persistent_dispatcher_matches_single_pair_dispatcher(self):
        # Value-level pin: a fresh single-pair dispatcher run is the
        # ground truth. The same pair on a persistent dispatcher must
        # still match it after two more unrelated pairs run.
        ref, deformeds = self._series()

        with self._open_persistent(ref) as d_single:
            f_single = d_single.correlate(None, deformeds[0])
        f_single_disp = f_single.displacements.copy()
        f_single_conf = f_single.confidence.copy()
        f_single_stat = f_single.status.copy()

        with self._open_persistent(ref) as d_persist:
            f_first = d_persist.correlate(None, deformeds[0])
            disp_snap = f_first.displacements.copy()
            conf_snap = f_first.confidence.copy()
            stat_snap = f_first.status.copy()
            _ = d_persist.correlate(None, deformeds[1])
            _ = d_persist.correlate(None, deformeds[2])

        np.testing.assert_allclose(f_first.displacements, f_single_disp, atol=1e-6)
        np.testing.assert_allclose(f_first.confidence, f_single_conf, atol=1e-6)
        np.testing.assert_array_equal(f_first.status, f_single_stat)
        np.testing.assert_allclose(disp_snap, f_single_disp, atol=1e-6)
        np.testing.assert_allclose(conf_snap, f_single_conf, atol=1e-6)
        np.testing.assert_array_equal(stat_snap, f_single_stat)


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Parse an ``events.jsonl`` into a list of dicts. Mirrors test_eventlog.py."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.gpu
class TestMultiGPUDispatcherEventLogging:
    """L3 worker-phase-record bridge: worker ``ncc.*`` records reach events.jsonl.

    Uses ``device_ids=[0, 0]`` to force the multi-process path on a
    single-GPU host (same trick the persistence regression suite
    uses). Each worker installs its own QueueHandler so the bridge
    machinery is exercised end-to-end even on dev hosts with one
    physical GPU.
    """

    @staticmethod
    def _smooth_pair(
        shape: tuple[int, int, int] = (48, 48, 48),
        shift: tuple[float, float, float] = (0.5, -0.4, 0.3),
        seed: int = 17,
    ):
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=seed)
        return pair.reference, pair.deformed

    @pytest.mark.slow
    def test_worker_phase_records_propagate(self, tmp_path: Path):
        # End-to-end: open a SessionScope, run one pair through a
        # 2-worker multi-process dispatcher with emit_phase_records=True,
        # and assert at least one kind:"phase" row with phase starting
        # "ncc." and a top-level device_id field landed in events.jsonl.
        # Pre-L3 this would have been impossible — worker records were
        # dropped on the floor under spawn.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with (
            SessionScope(tmp_path, series="test"),
            MultiGPUDispatcher(
                device_ids=[0, 0],
                anchored_reference=ref,
                emit_phase_records=True,
                **kwargs,
            ) as d,
        ):
            d.correlate(None, deformed)

        events = _read_events(tmp_path / "events.jsonl")
        ncc_phases = [
            e
            for e in events
            if e.get("kind") == "phase" and str(e.get("phase", "")).startswith("ncc.")
        ]
        assert ncc_phases, "no worker ncc.* phase rows reached events.jsonl"
        for row in ncc_phases:
            assert "device_id" in row, f"phase row missing device_id: {row}"
            assert isinstance(row["device_id"], int)

    @pytest.mark.slow
    def test_worker_phase_records_carry_pair_context(self, tmp_path: Path):
        # When SeriesPairLogger (or any caller) has bound t_ref/t_def
        # at pair-dispatch time, the dispatcher forwards them on the
        # pair-request payload and the worker filter stamps them as
        # mdvc_t_ref / mdvc_t_def on every record — so the events.jsonl
        # phase row is joinable to the originating pair on (t_ref, t_def)
        # with no ad-hoc bracketing.
        from structlog.contextvars import bind_contextvars, unbind_contextvars

        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with (
            SessionScope(tmp_path, series="test"),
            MultiGPUDispatcher(
                device_ids=[0, 0],
                anchored_reference=ref,
                emit_phase_records=True,
                **kwargs,
            ) as d,
        ):
            bind_contextvars(t_ref=7, t_def=11)
            try:
                d.correlate(None, deformed)
            finally:
                unbind_contextvars("t_ref", "t_def")

        events = _read_events(tmp_path / "events.jsonl")
        ncc_phases = [
            e
            for e in events
            if e.get("kind") == "phase" and str(e.get("phase", "")).startswith("ncc.")
        ]
        assert ncc_phases
        for row in ncc_phases:
            assert row.get("t_ref") == 7
            assert row.get("t_def") == 11

    @pytest.mark.slow
    def test_default_flag_off_omits_worker_records(self, tmp_path: Path):
        # Pre-L3 contract pin: with emit_phase_records=False (the
        # default), the events.jsonl must NOT carry ncc.* rows — the
        # bit-identical-by-default promise.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with (
            SessionScope(tmp_path, series="test"),
            MultiGPUDispatcher(
                device_ids=[0, 0],
                anchored_reference=ref,
                **kwargs,
            ) as d,
        ):
            d.correlate(None, deformed)

        events = _read_events(tmp_path / "events.jsonl")
        ncc_phases = [
            e
            for e in events
            if e.get("kind") == "phase" and str(e.get("phase", "")).startswith("ncc.")
        ]
        assert not ncc_phases, f"unexpected worker rows leaked with flag off: {ncc_phases}"

    @pytest.mark.slow
    def test_queue_overflow_surfaces_warning_on_exit(self, tmp_path: Path):
        # Force overflow by monkeypatching _PHASE_QUEUE_MAXSIZE to 1
        # before __enter__. Workers will drop the vast majority of
        # their phase records; the parent must surface one
        # RuntimeWarning per worker with a non-zero drop count,
        # rendered as a kind:"warning" row by SessionScope.
        ref, deformed = self._smooth_pair()
        kwargs = dict(window=24, overlap=0.5, search_radius=8, batch_size=64)

        with pytest.MonkeyPatch.context() as mp_ctx:
            from mamba_dvc.gpu import dispatch as dispatch_mod

            mp_ctx.setattr(dispatch_mod, "_PHASE_QUEUE_MAXSIZE", 1)
            with (
                SessionScope(tmp_path, series="test"),
                MultiGPUDispatcher(
                    device_ids=[0, 0],
                    anchored_reference=ref,
                    emit_phase_records=True,
                    **kwargs,
                ) as d,
            ):
                d.correlate(None, deformed)

        events = _read_events(tmp_path / "events.jsonl")
        warnings_rows = [
            e
            for e in events
            if e.get("kind") == "warning"
            and "dropped" in str(e.get("message", ""))
            and "phase records" in str(e.get("message", ""))
        ]
        assert warnings_rows, (
            f'queue overflow did not surface as kind:"warning" on exit; '
            f"events: {[e.get('kind') for e in events]}"
        )
