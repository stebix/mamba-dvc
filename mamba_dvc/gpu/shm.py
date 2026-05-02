r"""Shared-memory plumbing for multi-process volume sharing.

The dispatch layer (:mod:`mamba_dvc.gpu.dispatch`) ships large 3D
volumes (reference, deformed, mask, deformed mask) from the parent to
per-GPU workers without pickling. Each volume is published into a
:class:`multiprocessing.shared_memory.SharedMemory` block; the parent
sends a small picklable :class:`SharedArrayHandle` over the spawn
boundary; each worker attaches by name, wraps the buffer as a NumPy
view, then ``cupy.asarray`` uploads it to its GPU.

Lifecycle (Windows-safe under ``multiprocessing.spawn``):

- The publisher must stay alive until every consumer has finished
  reading. The :func:`published` context manager owns the lifetime in
  the parent: ``shm.close(); shm.unlink()`` runs on context exit.
- Each consumer opens its own handle via :func:`attach`, reads, and
  ``close()``\\ s its handle before exit. Workers must NOT call
  ``unlink()`` -- that is the publisher's responsibility.
- ``unlink()`` semantics differ across platforms (it is POSIX-only in
  spirit), so this module wraps it defensively. The same code runs on
  Windows production hosts and Linux dev workstations.

Why shared memory and not pickle/queue
--------------------------------------
A single ``(960, 1280, 1280)`` float32 volume is 6.3 GB. Pickling that
through a :class:`multiprocessing.Pipe` for each of four workers is a
non-starter (24+ GB of transient memory and dozens of seconds of
serialization). Shared memory is one allocation in the parent and four
zero-copy attaches in the workers. When :mod:`mamba_dvc.io` lands and
volumes arrive as zarr stores, dispatch can switch to per-worker
independent opens; this module remains useful as the in-memory IPC
fallback.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

__all__ = ["SharedArrayHandle", "attach", "publish", "published"]


@dataclass(frozen=True)
class SharedArrayHandle:
    """Picklable descriptor for a shared-memory backed array.

    The dispatch parent ships one of these per volume to each worker
    via :class:`multiprocessing.Process` arguments. Workers reconstruct
    a NumPy view via :func:`attach`.

    Attributes
    ----------
    name
        Auto-generated SharedMemory block name (e.g. ``wnsm_abc123`` on
        Windows, ``/psm_…`` on POSIX). Workers pass this to
        :class:`multiprocessing.shared_memory.SharedMemory` to attach.
    shape
        Volume shape, e.g. ``(960, 1280, 1280)``.
    dtype_str
        NumPy dtype byte-order + kind string (``"<f4"``, ``"|b1"``, …).
        Stored as a string rather than a :class:`numpy.dtype` because
        stringly-typed reconstruction is robust against version skew
        between the parent and the worker process import order.
    """

    name: str
    shape: tuple[int, ...]
    dtype_str: str


def publish(
    arr: np.ndarray,
) -> tuple[shared_memory.SharedMemory, SharedArrayHandle]:
    """Copy ``arr`` into a fresh shared-memory block.

    Parameters
    ----------
    arr
        NumPy array to publish. Must be C-contiguous; non-contiguous
        layouts cannot be wrapped over a flat shared buffer without
        copying through stride logic the worker side does not have.

    Returns
    -------
    tuple
        ``(shm, handle)``. ``shm`` is the publisher's
        :class:`SharedMemory` object — the caller must keep it alive
        and ``close+unlink`` on teardown (see :func:`published`).
        ``handle`` is the picklable descriptor to ship to workers.

    Raises
    ------
    ValueError
        If ``arr`` is not C-contiguous.

    Notes
    -----
    The copy into the shared region is a single ``memcpy``. Tests on a
    6 GB float32 volume on the target hardware peg PCIe-class memory
    bandwidth (a few seconds), so this is not a hot-path concern.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "publish requires a C-contiguous array; use np.ascontiguousarray before publishing"
        )
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    handle = SharedArrayHandle(
        name=shm.name,
        shape=tuple(int(s) for s in arr.shape),
        dtype_str=arr.dtype.str,
    )
    return shm, handle


def attach(
    handle: SharedArrayHandle,
) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """Open an existing shared-memory block and return a NumPy view.

    Parameters
    ----------
    handle
        Descriptor produced by :func:`publish` in the parent process.

    Returns
    -------
    tuple
        ``(shm, view)``. The caller must keep ``shm`` alive for as long
        as it reads through ``view``, then call ``shm.close()`` (and
        NOT ``unlink`` — only the publisher unlinks).

    Notes
    -----
    The view shares storage with the publisher's array and any other
    attached views; writes through the view ARE observable to other
    consumers. The dispatch contract is read-only on workers, so this
    is fine in practice but worth knowing if a future call site wants
    in-place preprocessing.
    """
    shm = shared_memory.SharedMemory(name=handle.name)
    view = np.ndarray(handle.shape, dtype=np.dtype(handle.dtype_str), buffer=shm.buf)
    return shm, view


@contextmanager
def published(arr: np.ndarray) -> Generator[SharedArrayHandle]:
    r"""Context-managed publish: yield a handle; close+unlink on exit.

    Use this in the dispatch parent so the SharedMemory block is
    guaranteed to be released even if a worker raises::

        with published(reference) as h_ref, published(deformed) as h_def:
            run_workers(h_ref, h_def, ...)

    Parameters
    ----------
    arr
        Array to publish for the lifetime of the ``with`` block.

    Yields
    ------
    SharedArrayHandle
        Picklable descriptor for workers.

    Notes
    -----
    The handle is invalid after the context exits — workers must finish
    attaching (and ideally finish reading) before the ``with`` block
    returns. Multi-process dispatch achieves this by ``join``\ ing
    every worker before falling through.
    """
    shm, handle = publish(arr)
    try:
        yield handle
    finally:
        shm.close()
        # `unlink` is POSIX-only in spirit; on Windows the resource
        # tracker may have already removed the name by the time we
        # get here. Either way, the block is gone. FileNotFoundError
        # is a subclass of OSError, so this catches both.
        with suppress(OSError):
            shm.unlink()
