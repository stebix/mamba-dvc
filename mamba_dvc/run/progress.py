"""Observer protocol for streaming :func:`~mamba_dvc.run.batch.run_batch` progress.

The execution loop is print-free by design (see :mod:`mamba_dvc.run.batch`):
all human-facing output lives in the CLI. To still surface live progress
without coupling the library to a renderer, ``run_batch`` accepts an
optional :class:`BatchObserver` and calls its hooks at well-defined points.
:class:`NullObserver` (the default) does nothing — programmatic callers are
unaffected — while :mod:`mamba_dvc.cli._run` plugs in a Rich progress bar
(or a plain per-job log when stdout is not a terminal).

Event contract
--------------
``run_batch`` guarantees the following call order for one invocation:

#. :meth:`~BatchObserver.on_batch_start` exactly once, after the full plan
   is known, so ``n_jobs`` is the exact number of
   :meth:`~BatchObserver.on_job_start` / :meth:`~BatchObserver.on_job_end`
   pairs that will follow. ``n_jobs`` counts store-open failures too — each
   surfaces as one placeholder job.
#. For each freshly materialized ``(deformation, mask, dry_shape)`` slice:
   :meth:`~BatchObserver.on_pair_load_start` once, then an
   :meth:`~BatchObserver.on_job_start` / :meth:`~BatchObserver.on_job_end`
   pair for every variant in that load group.
#. :meth:`~BatchObserver.on_batch_end` exactly once, with the same list
   ``run_batch`` returns.

**Invariant.** Every :class:`~mamba_dvc.run.batch.JobResult` the driver
records — including ``load_pair`` failures and store-open failures — is
bracketed by an :meth:`~BatchObserver.on_job_start` /
:meth:`~BatchObserver.on_job_end` pair, with ``on_job_start`` fired
*before* the work that produces the result (so a renderer can show the
in-flight job). An observer that implements only those two hooks therefore
sees every outcome; :meth:`~BatchObserver.on_pair_load_start` is a hint for
richer renderers and is *not* emitted for a store that fails to open.

Hooks must not raise: an observer exception propagates out of ``run_batch``
and aborts the run mid-campaign.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from mamba_dvc.run.batch import Job, JobResult

__all__ = ["BatchObserver", "NullObserver"]


class BatchObserver(Protocol):
    """Hooks :func:`~mamba_dvc.run.batch.run_batch` calls to stream progress.

    Implementations may override any subset of the methods; subclassing
    :class:`NullObserver` is the easy way to do that. See the module
    docstring for the call-order contract.
    """

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Announce the campaign size before any work begins."""

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Announce a (potentially slow) ``load_pair`` feeding ``n_variants`` variants."""

    def on_job_start(self, job: Job) -> None:
        """Announce that ``job`` is about to run (fired before the work)."""

    def on_job_end(self, result: JobResult) -> None:
        """Report ``result`` (``"ok"`` / ``"failed"``) for the finished job."""

    def on_batch_end(self, results: list[JobResult]) -> None:
        """Announce the campaign finished, with every recorded result."""


class NullObserver:
    """A :class:`BatchObserver` that ignores every event (the default)."""

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Ignore the batch-start event."""

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Ignore the pair-load-start event."""

    def on_job_start(self, job: Job) -> None:
        """Ignore the job-start event."""

    def on_job_end(self, result: JobResult) -> None:
        """Ignore the job-end event."""

    def on_batch_end(self, results: list[JobResult]) -> None:
        """Ignore the batch-end event."""
