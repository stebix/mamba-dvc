"""Canonical numpy/NaN coercion for JSON serialization.

One place to teach about numpy edge cases (scalars that aren't ``int`` /
``float``, ``ndarray`` instead of ``list``, ``NaN`` that ``json.dump``
rejects with ``allow_nan=False``). Used by:

- :mod:`mamba_dvc.run.batch` — the manifest writer and per-job JSON
  sidecar.
- :mod:`mamba_dvc.run.eventlog` — the structlog ``JSONRenderer``'s
  ``default`` callback, so numpy values bound by call sites or
  contextvars round-trip cleanly into ``events.jsonl``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

__all__ = ["to_jsonable"]


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays, tuples, and NaNs for JSON.

    ``NaN`` collapses to ``None`` so the result is valid JSON without
    ``allow_nan=True``. ``ndarray`` becomes a (nested) list to keep the
    output human-readable. Tuples become lists for the same reason.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in cast("Mapping[Any, Any]", obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in cast("Sequence[Any]", obj)]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
