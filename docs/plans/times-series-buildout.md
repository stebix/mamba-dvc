# Time-series synthetic harness — plan & contract

A closed-form, time-parameterized extension of the single-pair synthetic
harness in `mamba_dvc/validate/synthetic.py`. Generates `(reference,
frame_0, ..., frame_{n-1})` bundles with closed-form displacement-field
ground truth at every time step, and feeds the result through the
existing scoring layer in `mamba_dvc/validate/known_fields.py`. The
goal is **error-accumulation studies** under DVC matching across time
steps — both per-frame error growth (ref-anchored matching) and drift
accumulation (sequential matching with composition).

Implements `docs/plans/overview.md` §10's deferred *time series* hook
without disturbing the v1 single-pair contract.

## 1. Position in the codebase

```
mamba_dvc/validate/
  synthetic.py          # existing — single-pair, single-time
  known_fields.py       # existing — evaluate_pair / run_and_evaluate / sweep
  series.py             # NEW — TimeDisplacementFunction, TimeSeriesGT,
                        #       native field constructors, make_series, pairs
  series_evaluate.py    # NEW — evaluate_series, compose_predictions,
                        #       TimeSeriesReport (per-frame + drift curves)
```

```
                                 ┌─────────────────────────┐
   user                          │ make_series(            │  validate/series.py
   (test or CLI driver)          │   shape, field, times)  │
                                 └─────────┬───────────────┘
                                           │ TimeSeriesGT  (ref + lazy frames + closed-form u(x,t))
                                           ▼
                                 ┌─────────────────────────┐
                                 │ evaluate_series(        │  validate/series_evaluate.py
                                 │   series, correlate_fn, │
                                 │   mode, compose_drift)  │
                                 └─────────┬───────────────┘
                                           │ iterates pairs (a, b, gt_field, a_mask)
                                           ▼
                                 ┌─────────────────────────┐
                                 │ correlate_fn(a, b,      │  user-supplied (typically
                                 │              a_mask)    │  correlate_multi_gpu wrapper)
                                 └─────────┬───────────────┘
                                           │ DisplacementField
                                           ▼
                                 ┌─────────────────────────┐
                                 │ evaluate_pair(          │  validate/known_fields.py
                                 │   evaluation_pair, …)   │  (existing — reused)
                                 └─────────┬───────────────┘
                                           │ ErrorReport
                                           ▼
                                 accumulate into
                                 TimeSeriesReport (rmse_curve,
                                 p95_curve, bias_curve,
                                 drift_rmse_curve)
```

## 2. Scope

### v1 — build now

- Closed-form `TimeDisplacementFunction` Protocol with `(coords, t)` →
  `(points, 3)` displacement evaluation.
- Four native time-field constructors: rigid shift, uniform dilation,
  simple shear, sinusoidal — each with **linear-in-t** magnitude.
- `t_compose` for additive superposition, mirroring `synthetic.compose`.
- `make_series(...) → TimeSeriesGT` with eager and lazy materialization
  (LRU(4) for full-resolution series).
- `PairingMode = {REF_ANCHORED, SEQUENTIAL}`; both modes emit exactly
  `n` pairs.
- `evaluate_series(...)` reuses `validate.known_fields.evaluate_pair`
  per pair and aggregates into per-frame `ErrorReport`s plus convenience
  curves.
- `compose_drift=True` (sequential only): chain predicted increments
  into a running ref-frame estimate and compare against the cumulative
  GT — quantifies drift accumulation.
- One-line type widening of `EvaluationPair.gt_field` to accept either
  `GroundTruthField` or `DisplacementFunction`. No behavior change.

### v2 — design extension points for, do not build now

- Per-frame deformed masks (slot reserved on `TimeSeriesGT` and
  `TimeSeriesPair`).
- Warm-start time-series driver (correlate_fn accepting an
  `initial_guess`).
- Saturating / non-monotone magnitude schedules.
- Random smooth-field family with deterministic spatial structure and
  time-scaled magnitude.
- On-disk (zarr-backed) frame spool for series too large for RAM even
  under LRU(4).

## 3. Conceptual model

The natural primitive is **not** a list of frames but a closed-form
time-parameterized displacement field `u(x, t)` defined in the same
convention as `validate/synthetic.py`:

- Pull-back warp: `def_t(x) = ref(x − u(x, t))`.
- `u` evaluated at the **output** integer grid; this is the existing
  Eulerian-at-output-grid convention `synthetic.warp` already follows
  (see `synthetic.py` module docstring).
- Identity at `t = 0` is expected by convention; not enforced.

Given the closed form, every other quantity is derivable without
storing samples:

| Quantity | Closed form |
|---|---|
| frame *i* | `warp(reference, λc: u(c, t_i), order=k)` |
| cumulative GT (ref → frame *i*) | `λc: u(c, t_i)` |
| incremental GT (frame *i−1* → frame *i*) | `λc: u(c, t_i) − u(c, t_{i−1})` |

Inheriting the existing convention means the time-series harness reuses
the same warp kernel and the same evaluator-evaluation-grid semantics
as the single-pair harness; assertions transfer without sign flips or
frame-conversion bookkeeping.

## 4. Protocol + dataclasses

```python
# mamba_dvc/validate/series.py

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float32

from mamba_dvc.validate.synthetic import DisplacementFunction


@runtime_checkable
class TimeDisplacementFunction(Protocol):
    """Closed-form, time-parameterized displacement field ``u(x, t)``.

    Same coordinate convention as
    :class:`mamba_dvc.validate.synthetic.DisplacementFunction`: input
    ``(points, 3)`` voxel coords in ``(z, y, x)`` order, output
    ``(points, 3)`` displacement in the same order. Identity at
    ``t = 0`` is expected by convention.
    """

    def __call__(
        self,
        coords: Float32[np.ndarray, "points 3"],
        t: float,
    ) -> Float32[np.ndarray, "points 3"]:
        """Evaluate ``u(coords, t)``."""


class PairingMode(StrEnum):
    """How frames in a series are paired for evaluation."""

    REF_ANCHORED = "ref_anchored"
    """``(reference, frame_i)`` for ``i in 0..n-1`` — used for absolute
    error growth."""

    SEQUENTIAL = "sequential"
    """``(reference, frame_0)`` for ``i == 0`` and
    ``(frame_{i-1}, frame_i)`` for ``i in 1..n-1`` — used for drift
    studies. Both modes emit exactly ``n`` pairs so curves are
    index-aligned for plotting."""


@dataclass(frozen=True)
class TimeSeriesPair:
    """One frame pair drawn from a series, with its per-pair GT field.

    Cheap to construct: ``a`` and ``b`` are views into frames owned by
    the parent :class:`TimeSeriesGT`; ``gt_field`` is a closure over the
    series' :class:`TimeDisplacementFunction`. Iterating a series does
    not duplicate volumes.
    """

    a: Float32[np.ndarray, "z y x"]
    b: Float32[np.ndarray, "z y x"]
    gt_field: DisplacementFunction
    a_mask: Bool[np.ndarray, "z y x"] | None
    a_index: int | None
    """Frame index of ``a`` in the parent series. ``None`` when ``a``
    is the series reference."""
    b_index: int
    """Frame index of ``b`` in the parent series, ``0..n-1``."""
    t_a: float
    t_b: float
    mode: PairingMode

    def to_evaluation_pair(self, name: str) -> "EvaluationPair":
        """Adapt to :class:`mamba_dvc.io.dataset.EvaluationPair`.

        Reuses the existing scoring path:
        :func:`mamba_dvc.validate.known_fields.evaluate_pair` consumes an
        ``EvaluationPair`` and only ever calls ``gt_field(positions)``,
        so the closure satisfies its contract once the type seam is
        widened (see §7).
        """


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Provenance + behavior knobs for a :class:`TimeSeriesGT`."""

    times: Float32[np.ndarray, "n"]
    """Monotone non-decreasing frame timestamps. ``times[0] == 0``
    means ``frame_0 == reference`` (identity warp)."""
    materialize: Literal["eager", "lazy"]
    cache_capacity: int | None
    """LRU slot count when ``materialize == "lazy"``. ``None`` means
    unbounded (only safe for small shapes)."""
    warp_order: int
    """Spline order for :func:`mamba_dvc.validate.synthetic.warp`.
    Default ``3`` (cubic)."""
    warp_mode: str
    """Boundary mode forwarded to ``map_coordinates``. Default
    ``"reflect"``."""
    seed: int
    """RNG seed used for the texture (or for any seeded field family).
    Carried through for reproducibility."""
    field_repr: str
    """Human-readable repr of the time-field, e.g. ``"t_rigid_shift(velocity=(…))"``.
    Used as the default ``name`` for per-frame ``ErrorReport``s."""


@dataclass(frozen=True)
class TimeSeriesGT:
    """``reference`` + ``n`` deformed frames + closed-form ``u(x, t)``.

    Frames may be lazy: ``_frames`` carries ``None`` slots that
    :meth:`frame` materializes on demand and caches under an LRU policy
    keyed by frame index.
    """

    reference: Float32[np.ndarray, "z y x"]
    field: TimeDisplacementFunction
    config: TimeSeriesConfig
    mask: Bool[np.ndarray, "z y x"] | None
    """Reference-frame mask (shared across frames in v1). Per-frame
    deformed masks are the v2 hook (see §10)."""

    _frames: tuple[Float32[np.ndarray, "z y x"] | None, ...]
    """Eager: every slot populated. Lazy: every slot ``None``; the LRU
    cache lives in a private mutable attribute managed by
    :meth:`frame`."""

    def __len__(self) -> int:
        """Number of deformed frames, equal to ``len(times)``."""

    def frame(self, i: int) -> Float32[np.ndarray, "z y x"]:
        """Return frame ``i``. Materializes on demand when lazy."""

    def cumulative_field_at(self, i: int) -> DisplacementFunction:
        """Closed-form GT for ``(reference, frame_i)``: ``λc: u(c, times[i])``."""

    def incremental_field_at(self, i: int) -> DisplacementFunction:
        """Closed-form GT for ``(frame_{i-1}, frame_i)`` (or
        ``(reference, frame_0)`` when ``i == 0``):
        ``λc: u(c, times[i]) − u(c, t_prev)`` with
        ``t_prev = 0`` for ``i == 0``."""

    def pair(
        self,
        i: int,
        mode: PairingMode = PairingMode.REF_ANCHORED,
    ) -> TimeSeriesPair:
        """Build the ``i``-th pair under ``mode``."""

    def pairs(
        self,
        mode: PairingMode = PairingMode.REF_ANCHORED,
    ) -> Iterator[TimeSeriesPair]:
        """Iterate all ``n`` pairs under ``mode`` in increasing ``i``."""
```

A few choices baked into the shape:

- Frame storage lives on `TimeSeriesGT`, not on `TimeSeriesPair`. Pairs
  are cheap views; iterating doesn't duplicate volumes.
- `gt_field` on a pair is a `DisplacementFunction` closure (same
  Protocol the single-pair harness already uses). One Protocol, two
  contexts.
- `a_mask` for sequential mode falls back to the ref-frame `mask` in
  v1. The deformed-frame mask is the v2 hook called out in
  `overview.md` §10.

## 5. Native time-field constructors

Mirror the single-pair catalog so callers don't reach for a different
idiom. Linear-in-t magnitude is the only schedule in v1; saturating /
non-monotone schedules are the v2 hook. No random-field family in v1.

```python
def t_rigid_shift(
    velocity: tuple[float, float, float],
) -> TimeDisplacementFunction:
    """``u(x, t) = t · velocity`` (constant per axis, zero at t=0)."""


def t_uniform_dilation(
    strain_rate: float,
    center: tuple[float, float, float],
) -> TimeDisplacementFunction:
    """``u(x, t) = strain_rate · t · (x − center)`` — radial expansion
    from ``center`` with magnitude growing linearly in ``t``."""


def t_simple_shear(
    gamma_rate: float,
    shear_axis: int,
    gradient_axis: int,
) -> TimeDisplacementFunction:
    """``u[shear_axis](x, t) = gamma_rate · t · x[gradient_axis]``;
    other components zero."""


def t_sinusoidal(
    amplitude_rate: tuple[float, float, float],
    wavelength: tuple[float, float, float],
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TimeDisplacementFunction:
    """Per-axis sinusoid with linearly growing amplitude:
    ``u_i(x, t) = (Ȧ_i · t) · sin(2π · x_i / λ_i + φ_i)``."""


def t_compose(*fields: TimeDisplacementFunction) -> TimeDisplacementFunction:
    """Pointwise additive composition, mirroring
    :func:`mamba_dvc.validate.synthetic.compose`. Time is passed
    through to every component; the result satisfies
    ``compose(f, g)(c, t) == f(c, t) + g(c, t)``."""
```

Validation rules (parameter shapes, positivity of `wavelength`,
distinctness of `shear_axis`/`gradient_axis`) mirror the static
constructors in `validate/synthetic.py` to keep error messages
familiar.

## 6. Generator

```python
def make_series(
    shape: tuple[int, int, int],
    field: TimeDisplacementFunction,
    times: Sequence[float],
    *,
    texture_sigma: float = 1.5,
    seed: int = 0,
    mask: Bool[np.ndarray, "z y x"] | None = None,
    order: int = 3,
    reference: Float32[np.ndarray, "z y x"] | None = None,
    materialize: Literal["eager", "lazy"] = "eager",
    cache_capacity: int | None = 4,
) -> TimeSeriesGT:
    """Produce a ``(reference, frame_0, ..., frame_{n-1})`` bundle.

    Parameters mirror :func:`mamba_dvc.validate.synthetic.make_pair`
    where they overlap; only ``materialize`` and ``cache_capacity`` are
    new. ``times`` is required to be monotone non-decreasing; identity
    at ``t == 0`` is expected by convention but not enforced.
    """
```

### Memory model

A single full-resolution frame at `(960, 1280, 1280) float32` is
≈6.29 GB. The two modes:

| Mode | Frames stored | Per-call cost |
|---|---|---|
| `"eager"` | all `n` materialized at construction | one warp per frame, up front |
| `"lazy"` | zero at construction; LRU(`cache_capacity`) on access | one warp per cache miss |

**Eager** is the default for small synthetic shapes (≤ ~512³); fast,
predictable, no surprise allocations during evaluation. **Lazy** is
recommended above 512³: the evaluator iterates pairs and at any moment
only needs `frame(i-1)` and `frame(i)` resident, so `cache_capacity=4`
caps host RAM at ≈32 GB regardless of `n` while leaving headroom for
the parent process. Reference and mask are always resident.

The LRU lives in a private mutable attribute that does **not** break
``frozen=True`` semantics for the public state — the dataclass fields
remain immutable; only an internal cache map is mutated.

## 7. Wiring into the existing evaluator

The single-pair scoring layer in `validate/known_fields.py` already
does what's needed at frame granularity:
`evaluate_pair(EvaluationPair, DisplacementField) → ErrorReport`.

### 7.1 Type seam (load-bearing one-line change)

`io/dataset.py`:

```python
# before
gt_field: GroundTruthField | None

# after
gt_field: GroundTruthField | DisplacementFunction | None
```

`evaluate_pair`'s body only ever calls `pair.gt_field(field.positions)`
— a contract both `GroundTruthField` (sampled, interpolated) and
`DisplacementFunction` (closed-form) satisfy. No behavior change; one
type widening avoids a parallel evaluator.

### 7.2 Series-level evaluator

```python
# mamba_dvc/validate/series_evaluate.py

@dataclass(frozen=True)
class TimeSeriesReport:
    """Aggregate accuracy of a :class:`TimeSeriesGT` under one mode."""

    name: str
    mode: PairingMode
    times: Float32[np.ndarray, "n"]
    per_frame: tuple[ErrorReport, ...]
    """One :class:`mamba_dvc.validate.known_fields.ErrorReport` per
    pair, in increasing ``i``."""

    rmse_curve: Float32[np.ndarray, "n"]
    p95_curve: Float32[np.ndarray, "n"]
    bias_curve: Float32[np.ndarray, "n 3"]
    """Mean signed error per axis ``(dz, dy, dx)``."""

    drift_rmse_curve: Float32[np.ndarray, "n"] | None
    """Populated only when ``mode == SEQUENTIAL`` and
    ``compose_drift=True``. Cumulative ref-frame error after composing
    predicted increments, scored against
    :meth:`TimeSeriesGT.cumulative_field_at`."""


def evaluate_series(
    series: TimeSeriesGT,
    *,
    correlate_fn: CorrelateFn,
    mode: PairingMode = PairingMode.REF_ANCHORED,
    distance_bins: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 50.0),
    compose_drift: bool = False,
    name: str | None = None,
    progress: bool = False,
) -> TimeSeriesReport:
    """Run ``correlate_fn`` over ``series.pairs(mode)`` and accumulate.

    Per pair: build an ``EvaluationPair`` view, call
    ``correlate_fn(a, b, a_mask) → DisplacementField``, score with
    ``evaluate_pair``. Curves are derived from the per-frame reports.

    ``compose_drift`` is sequential-only and adds the drift-accumulation
    leg (see :func:`compose_predictions`); raises ``ValueError`` when
    set under ``REF_ANCHORED``.
    """
```

`CorrelateFn` reuses the alias from `validate/known_fields.py` —
``Callable[[ref, def, mask], DisplacementField]`` — so the existing
`run_and_evaluate` / `sweep` orchestrators and the new `evaluate_series`
share one signature shape.

### 7.3 Drift composition

```python
def compose_predictions(
    u_prev: Float32[np.ndarray, "points 3"],
    delta_pred: Float32[np.ndarray, "points 3"],
    grid: GridSpec,
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> Float32[np.ndarray, "points 3"]:
    """Compose two displacement fields sampled on the same POI lattice.

    ``u_total(x) = u_prev(x) + delta_pred(x + u_prev(x))``: takes a
    cumulative ref-frame estimate at the original POIs plus an
    increment sampled at the *deformed*-frame POI lattice, and produces
    the new cumulative ref-frame estimate at the original POIs.
    Off-grid sampling uses ``scipy.ndimage.map_coordinates`` on the
    reshaped ``(nz, ny, nx, 3)`` lattice; cheap because ``points`` is
    O(10 k), not voxel count.
    """
```

This is the only utility the drift study needs that doesn't already
exist; the rest is bookkeeping over `evaluate_pair`.

## 8. Conventions and invariants

Pin these in tests so future contributors don't drift on convention:

- **Identity at `t=0`.** With `times[0] == 0`, `frame(0)` equals
  `reference` to within `map_coordinates`'s integer-grid floor (exact
  for cubic spline at integer coordinates with `prefilter=True`).
- **Ref-anchored zero-time error ≈ correlator floor.** Under
  `REF_ANCHORED` at `t == 0`, RMSE should be at the correlator's
  noise floor, not at 0 — the warp is identity but the FFT NCC + Gauss
  fit still introduces sub-voxel fluctuations on textured noise.
- **Sequential-mode pair count.** Always `n`, with the `i == 0` pair
  being `(reference, frame_0)`. For `times[0] == 0` it is a trivial
  identity pair (zero GT); reports stay index-aligned with
  `REF_ANCHORED`.
- **Closed-form-invertible families round-trip.** For `t_rigid_shift`
  (the only family with an exactly-invertible forward map in v1),
  drift-composition under `compose_drift=True` should recover the
  cumulative GT to floating-point precision when the correlator is
  replaced by an oracle that returns the GT increment.
- **Coordinate convention.** Inherits `validate/synthetic.py`: GT is
  Eulerian-at-output-grid, evaluated at integer ref-frame POI
  positions; pull-back warp `def(x) = ref(x − u(x))`. No frame
  conversions in either evaluator path.

## 9. End-to-end usage

```python
# Per-frame error growth under linearly accelerating shift.
field = t_rigid_shift(velocity=(0.0, 0.7, 0.5))
series = make_series(
    shape=(256, 256, 256),
    field=field,
    times=np.linspace(0.0, 10.0, 11),   # frame_0 == reference (t=0)
    materialize="eager",
)
report = evaluate_series(
    series,
    correlate_fn=my_correlate,
    mode=PairingMode.REF_ANCHORED,
)
# report.rmse_curve[i] vs report.times[i] → absolute error growth.

# Drift study under the same field.
report_seq = evaluate_series(
    series,
    correlate_fn=my_correlate,
    mode=PairingMode.SEQUENTIAL,
    compose_drift=True,
)
# report_seq.drift_rmse_curve − report.rmse_curve → cost of chaining
# vs anchoring; expected to be small for shift-only, larger under
# strain or shear.
```

For full-resolution series swap `materialize="lazy"` and let LRU(4)
cap host RAM:

```python
series = make_series(
    shape=(960, 1280, 1280),
    field=t_uniform_dilation(strain_rate=0.001, center=(480, 640, 640)),
    times=np.arange(0.0, 11.0),
    materialize="lazy",
    cache_capacity=4,
)
```

## 10. v2 hooks (already aligned)

| v2 feature | v1 hook that supports it |
|---|---|
| Per-frame deformed masks | `TimeSeriesGT.deformed_masks: tuple[BoolArray, ...] \| None` slotted next to `mask`; `TimeSeriesPair.a_mask` reads from it when populated. API unchanged. |
| Warm-start time-series driver | `TimeSeriesPair` plus a `correlate_fn` accepting `initial_guess` is the interface. A future `correlate_with_warmstart` becomes a drop-in `correlate_fn`. |
| Saturating / non-monotone schedules | New constructor arguments on existing time-fields (e.g. `t_sinusoidal(..., growth="saturating", saturation_t=…)`). The Protocol is unchanged. |
| Random smooth-field family | Add `t_random_smooth(amplitude, sigma, *, seed)` — closed-form spatial structure with time-scaled magnitude. Same Protocol. |
| On-disk frame spool | Add `frame_store: zarr.Array | None` to `TimeSeriesConfig`; `TimeSeriesGT.frame(i)` reads from the store before warping. |
| Pyramid / iterative warp inside the inner correlator | Orthogonal — only `correlate_fn` changes; the series harness is untouched. |

## 11. Implementation slicing

Land in this order; each step is independently testable.

1. **`validate/series.py`**: `TimeDisplacementFunction` Protocol,
   `PairingMode`, `TimeSeriesPair`, `TimeSeriesConfig`, `TimeSeriesGT`
   with eager + lazy materialization (LRU), the four native field
   constructors, `t_compose`, and `make_series`. Pure host-side NumPy;
   no CuPy.
2. **Type seam widening** in `io/dataset.py`:
   `EvaluationPair.gt_field: GroundTruthField | DisplacementFunction | None`.
   Add a focused test confirming `evaluate_pair` accepts a closure-typed
   `gt_field` end-to-end.
3. **`validate/series_evaluate.py`**: `TimeSeriesReport`,
   `evaluate_series`, `compose_predictions`. Reuses `evaluate_pair` per
   pair; drift composition guarded by `compose_drift` flag.
4. **Tests** (see §12). Both `validate/` modules are `D`-rules-exempt
   per `pyproject.toml`'s `per-file-ignores`, but everything still
   runs through `ruff format`, `ruff check`, `pyright`, and `pytest`.

Each step keeps the v1 single-pair contract untouched.

## 12. Tests

Tier 1 — analytical (CI-eligible, small shapes, sub-second):

- **Identity at `t=0`.** `make_series(..., times=[0.0, 1.0])`;
  `frame(0)` matches `reference` within float32 epsilon.
- **Ref-anchored RMSE growth on `t_rigid_shift`.** Monotone
  non-decreasing `rmse_curve`; final-frame RMSE within the
  ±0.1-voxel target the plan calls out.
- **Sequential drift on `t_rigid_shift`.** With `compose_drift=True`
  and an oracle correlator that returns the GT increment,
  `drift_rmse_curve` is at floating-point floor across all frames.
- **Sinusoid as heterogeneous-field smoke test.** Confirms the
  sequential-vs-anchored gap is non-zero under spatially varying
  fields.
- **Lazy LRU correctness.** With `cache_capacity=2`, iterating a
  10-frame series under `REF_ANCHORED` produces identical results to
  eager, with the cache bounded.
- **Pair count alignment.** Both modes emit exactly `n` pairs;
  `report.times` aligns 1:1 across modes.

Tier 2 — full resolution (manual, not in CI):

- `(960, 1280, 1280)` series with `materialize="lazy"`,
  `cache_capacity=4`; resident-set check via OS counters; one
  `correlate_multi_gpu` pass per pair.

## 13. Out of scope

Explicitly **not** part of this buildout — flag if you want them
relocated:

- A new top-level evaluator that bypasses `evaluate_pair`. The widening
  seam in §7.1 keeps everything on one path.
- Persisting `TimeSeriesGT` to disk. The closed-form field is the
  canonical artifact; serializing materialized frames is the v2
  on-disk-spool hook.
- GPU-side warp. Generation stays host-side NumPy + SciPy
  `map_coordinates`, matching the existing `validate/synthetic.py`
  policy. Moving to GPU is orthogonal and lands when the warp itself
  becomes the bottleneck.
- Time-axis interpolation between frames. `times` is a discrete index;
  callers wanting sub-step resolution should add intermediate `times`
  entries.
