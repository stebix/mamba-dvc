"""Tests for ``mamba_dvc.gpu.shm`` shared-memory plumbing."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import pytest
from mamba_dvc.gpu.shm import SharedArrayHandle, attach, publish, published


class TestPublishAttachRoundtrip:
    def test_float32_volume_roundtrips(self):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((4, 8, 16), dtype=np.float32)

        shm, handle = publish(arr)
        try:
            shm_consumer, view = attach(handle)
            try:
                np.testing.assert_array_equal(view, arr)
                assert view.dtype == arr.dtype
                assert view.shape == arr.shape
                # Storage is shared, not copied: writing through the
                # publisher's mapping is visible through the consumer
                # view immediately.
                pub_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                pub_view[0, 0, 0] = 42.0
                assert view[0, 0, 0] == np.float32(42.0)
            finally:
                shm_consumer.close()
        finally:
            shm.close()
            shm.unlink()

    def test_bool_mask_roundtrips(self):
        # Mask is the third volume each worker attaches; bool dtype
        # exercises the dtype-string roundtrip.
        arr = np.zeros((4, 8, 8), dtype=np.bool_)
        arr[1:3, 2:6, 0::2] = True

        shm, handle = publish(arr)
        try:
            assert handle.dtype_str == arr.dtype.str
            shm_consumer, view = attach(handle)
            try:
                np.testing.assert_array_equal(view, arr)
                assert view.dtype == np.bool_
            finally:
                shm_consumer.close()
        finally:
            shm.close()
            shm.unlink()


class TestPublishedContextManager:
    def test_yields_handle_and_releases_on_exit(self):
        arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)

        with published(arr) as handle:
            assert isinstance(handle, SharedArrayHandle)
            shm_consumer, view = attach(handle)
            try:
                np.testing.assert_array_equal(view, arr)
            finally:
                shm_consumer.close()
            block_name = handle.name

        # After exit, the publisher's block is closed and unlinked.
        # Re-attaching by name should fail; on Windows the tracker may
        # have already cleared the name (FileNotFoundError) and on
        # POSIX the unlink invalidates further opens.
        with pytest.raises((FileNotFoundError, OSError)):
            shared_memory.SharedMemory(name=block_name)

    def test_releases_on_inner_exception(self):
        arr = np.zeros((2, 2, 2), dtype=np.float32)

        with pytest.raises(RuntimeError, match="boom"), published(arr) as handle:
            _ = handle.name
            raise RuntimeError("boom")
        # Reaching this line means the context manager unwound cleanly
        # despite the inner raise; the unlink path ran in the finally.


class TestPublishValidation:
    def test_non_contiguous_raises(self):
        # A non-C-contiguous view (here a transposed slice) cannot be
        # wrapped over a flat shared buffer without copying first.
        arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        non_contig = arr.transpose(2, 0, 1)
        assert not non_contig.flags["C_CONTIGUOUS"]

        with pytest.raises(ValueError, match="C-contiguous"):
            publish(non_contig)


def _child_attach_check(handle: SharedArrayHandle, sentinel: float, q: mp.Queue) -> None:
    """Worker entry: attach by handle and report the value at index 0.

    Top-level so it pickles cleanly under ``multiprocessing.spawn``.
    """
    try:
        shm, view = attach(handle)
        try:
            value = float(view.flat[0])
            q.put(("ok", value, abs(value - sentinel) < 1e-6))
        finally:
            shm.close()
    except Exception as exc:  # pragma: no cover - failure path
        q.put(("err", repr(exc), False))


class TestCrossProcessAttach:
    @pytest.mark.slow
    def test_spawned_child_can_attach_by_name(self):
        # End-to-end: parent publishes, child process spawned via
        # `spawn` (Windows-default; we force it explicitly so this also
        # exercises the same path on Linux dev hosts) attaches by name
        # and reads the same content.
        ctx = mp.get_context("spawn")
        rng = np.random.default_rng(1)
        arr = rng.standard_normal((8, 8, 8), dtype=np.float32)
        sentinel = float(arr.flat[0])

        q: mp.Queue = ctx.Queue()
        with published(arr) as handle:
            proc = ctx.Process(target=_child_attach_check, args=(handle, sentinel, q))
            proc.start()
            proc.join(timeout=30)
            assert proc.exitcode == 0, f"child exited with {proc.exitcode}"

        status, value, matches = q.get(timeout=5)
        assert status == "ok", f"child reported error: {value}"
        assert matches, f"child read {value}, expected {sentinel}"
