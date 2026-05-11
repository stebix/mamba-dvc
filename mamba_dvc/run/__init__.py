"""Batch displacement-field campaigns: drive the pipeline over many inputs.

This subpackage turns a campaign **config file** into a directory of
persisted :class:`~mamba_dvc.types.DisplacementField` results plus an
aggregation manifest, looping over the
``(store x deformation x parameter variant)`` product. It is the step
from the one-pair manual harness (``scripts/run_e2e_zarr.py``) to
"gather results" — resumable, with full provenance.

See ``docs/plans/run-interface.md`` for the design contract.

- :mod:`mamba_dvc.run.config` — :class:`BatchSpec` / :class:`Variant`:
  parse + validate the config, expand the sweep into resolved variants,
  split each into its load / grid / compute tier slices.
- :mod:`mamba_dvc.run.batch` — :class:`Job` / :class:`JobResult` /
  :func:`plan_jobs` / :func:`run_batch`: the materialize-once /
  iterate-many execution loop, resume, and result persistence.
"""

from __future__ import annotations

from mamba_dvc.run.batch import Job, JobResult, plan_jobs, run_batch
from mamba_dvc.run.config import BatchSpec, Variant

__all__ = [
    "BatchSpec",
    "Job",
    "JobResult",
    "Variant",
    "plan_jobs",
    "run_batch",
]
