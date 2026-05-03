# Time-step pipeline — plan & contract

The driver layer that lifts the v1 single-pair `correlate()` to an
ordered sequence of frames. Implements the time-series hook reserved
in `docs/plans/overview.md` §10 ("wrap `correlate()` in `series.py`
with prefetch and optional warm-start") and consumes the persistent
multi-GPU pool reserved in `docs/plans/dispatch.md` §10
(`MultiGPUDispatcher`). Lives under `mamba_dvc/pipeline/series.py`
with supporting modules in `io/`, `core/`, `gpu/`, and `validate/`.

This document is structured to mirror `docs/plans/correlate.md` and
`docs/plans/dispatch.md`: position → invariants → API → flow →
data structures → error model → test surface → deferrals.

## 1. Position in the codebase

```
                 ┌──────────────────────────────────────────────┐
                 │ pipeline.series.correlate_series(frames, …)  │
                 │                                               │
io/series.py ───►│   pairing topology  (REF | SEQ | UPDATED)     │
                 │   warm-start hookup                            │
                 │   per-pair callback (on_pair=)                 │
                 └────┬─────────────────────────────────────────┬┘
                      │                                          │
                      ▼                                          ▼
        ┌─────────────────────────────┐         ┌─────────────────────────────┐
        │ gpu.dispatch                │         │ pipeline.correlate          │
        │   MultiGPUDispatcher (pool) │         │   single-device pure path   │
        └─────────────┬───────────────┘         └─────────────┬───────────────┘
                      │                                       │
                      └───────► correlate_admitted_subset ◄───┘
                                  (pipeline._internal)

                      DisplacementField per pair  ──────────►  DisplacementSeries
                                                                     │
                                                                     ▼
                                                  validate.series_error.evaluate(...)
                                                  ──► SeriesErrorReport
```

`pipeline/series.py` is **pure**, like `pipeline/correlate.py`. It
takes an iterator of frames already in memory and returns a
`DisplacementSeries`. All I/O, prefetch, double-buffering, and
disk-streaming live in `io/series.py` and the `on_pair` callback.

`gpu/dispatch.py` gains a `MultiGPUDispatcher` context manager so the
~6 s spawn + CUDA-context cost is paid once per series instead of once
per pair (`dispatch.md` §6 puts the per-call overhead at ~1.5 s spawn ×
N + ~3 s CUDA init per device).

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Driver is a pure function | Plan §4 pure-core rule extends to series. v2 IC-GN-inside-series and v3 dynamic re-anchoring layer on top; both fail if the driver owns hidden state. |
| Series driver does no I/O | Frame supply is an iterator. Prefetch lives in `io/series.py`. The driver runs deterministically against in-memory `list[ndarray]` for tests. |
| Pairing topology is data, not branching | `PairingStrategy` enum + a tiny per-strategy `_pair_provider`. New strategies (e.g. v3 dynamic re-anchoring) plug in without touching the driver body. |
| Grid is fixed across the series | All frames share `volume_shape`, `mask`, `window`, `overlap`. The grid is built once, mask admission runs once, every pair reuses the resulting `GridSpec`. POI alignment across `t` is what makes cumulative composition and error stratification meaningful. |
| Per-pair output is a v1 `DisplacementField` | `DisplacementSeries` is a sequence of v1 fields plus topology metadata. No new compute primitive — just sequencing. |
| Failed pairs do not abort the series | A pair that errors out (CUDA OOM, propagated worker exception) records a `SeriesPairStatus.FAILED` entry and the driver continues. The `on_pair` callback sees the failure; cumulative composition halts at the failure boundary. |
| Cumulative composition is opt-in and lazy | Storing `u_total(t)` for every `t` doubles the memory of `DisplacementSeries`. Make it a method that materializes on demand, optionally cached. |
| Persistent worker pool is the **only** way to amortize spawn cost | Series with a fresh dispatcher per pair regresses to v1 cost. The driver accepts an optional pre-built `MultiGPUDispatcher`; without one it falls through to single-device `correlate()`. |

## 3. API

### Driver (`mamba_dvc/pipeline/series.py`)

```python
from collections.abc import Callable, Iterable, Iterator

from jaxtyping import Bool, Float32

from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
from mamba_dvc.types import (
    DisplacementField,
    DisplacementSeries,
    PairingStrategy,
)


def correlate_series(
    frames: Iterable[tuple[int, Float32[np.ndarray, "z y x"]]],
    mask: Bool[np.ndarray, "z y x"] | None = None,
    *,
    strategy: PairingStrategy = PairingStrategy.SEQUENTIAL,
    dispatcher: MultiGPUDispatcher | None = None,
    on_pair: Callable[[int, DisplacementField], None] | None = None,
    # forwarded to correlate() / dispatcher.correlate() unchanged:
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float | None = None,
    search_radius: int | None = None,
    batch_size: int = 256,
    eps: float = 1e-12,
    ncc_mode: NCCMode = "linear",
    ncc_normalization: NCCNormalization = "overlap",
) -> DisplacementSeries: ...
```

- `frames` is an iterator of `(t, volume_t)`. The driver consumes it
  exactly once; rewinding is the caller's job. `t` is an integer
  frame index (not a timestamp) so the driver stays unitless.
- `mask` is the **shared screw mask** (overview §3). One mask for the
  whole series in v1; v2 per-frame deformed masks are a separate
  driver feature.
- `strategy` selects the pairing topology (§4).
- `dispatcher`: if provided, the driver routes each pair through the
  pool; if `None`, it calls `correlate()` directly. The driver does
  **not** create or manage the dispatcher — that's the caller's
  responsibility (so the same dispatcher can span multiple series).
- `on_pair(t, field)` fires after each pair completes. Use it for
  streaming writes to disk, progress bars, per-frame error capture.
  Sole side-channel from inside the driver.
- All remaining kwargs are forwarded to the per-pair correlator
  unchanged. The driver introduces no new tuning knobs in v1.

### Persistent pool (`mamba_dvc/gpu/dispatch.py`, new context manager)

```python
class MultiGPUDispatcher:
    """Persistent multi-GPU worker pool. v1 single-pair dispatch
    spawns + tears down per call; this keeps the workers alive
    across pairs so the spawn + CUDA init cost is paid once.

    Replicates ``mask`` (and, for REFERENCE_ANCHORED strategy, the
    fixed reference volume) on every device at ``__enter__``. Each
    ``correlate(...)`` call uploads only the new deformed frame.
    """

    def __init__(
        self,
        device_ids: Sequence[int] | None = None,
        *,
        mask: Bool[np.ndarray, "z y x"] | None = None,
        anchored_reference: Float32[np.ndarray, "z y x"] | None = None,
    ) -> None: ...

    def __enter__(self) -> "MultiGPUDispatcher": ...
    def __exit__(self, *exc) -> None: ...

    def correlate(
        self,
        reference: Float32[np.ndarray, "z y x"] | None,
        deformed: Float32[np.ndarray, "z y x"],
        **kwargs,
    ) -> DisplacementField: ...
```

- `reference=None` is legal only when `anchored_reference` was passed
  to the constructor — the workers reuse the resident reference. The
  driver uses this for `REFERENCE_ANCHORED` strategy to skip a 6.3 GB
  upload per pair.
- `__exit__` joins all workers, releases SHM, suppresses
  `terminate()` storms on `KeyboardInterrupt` (mirrors current
  `_run_multi_process` cleanup in `dispatch.py`).

### Frame supply (`mamba_dvc/io/series.py`)

```python
def iter_frames(
    paths: Sequence[Path],
    *,
    prefetch: int = 1,
) -> Iterator[tuple[int, Float32[np.ndarray, "z y x"]]]: ...
```

Lazy iterator over volumes on disk. Uses a thread (or
`tensorstore.Future`) to read frame `t+prefetch` while the driver
processes frame `t`. `prefetch=0` is the synchronous baseline used in
tests. The iterator is the **only** stateful component; the driver
remains pure.

### Field composition (`mamba_dvc/core/field_ops.py`, new)

```python
def compose_displacement_fields(
    total: DisplacementField,
    increment: DisplacementField,
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> DisplacementField: ...
```

Implements `u_total_new(x) = increment(x + u_total(x)) + u_total(x)`.
Required only by `SEQUENTIAL` strategy when the caller asks for
cumulative fields. Densification (overview §10) will reuse the same
interpolation scaffolding, so the module name is general.

## 4. Pairing topologies

The strategy selects a *pair provider* — a small generator over the
frame stream that emits `(reference_t, deformed_t)` pairs the
correlator runs.

### A. `REFERENCE_ANCHORED` (Lagrangian total)

```
for t = 1, 2, …:  yield (frame_0, frame_t)
```

- Each `DisplacementField` is the absolute field at `t`.
- Per-frame errors are independent (no temporal drift).
- **Failure mode:** displacement grows unbounded. Once `|u| > W/3`
  (overview §2: ~32 vx at `W=96`) the correlation SNR collapses.
  Practical for short series, sanity checks, single-load experiments.
- Dispatcher integration: `anchored_reference=frame_0` at pool
  construction; per-pair upload is one volume not two.

### B. `SEQUENTIAL` (Lagrangian incremental) — **default**

```
for t = 1, 2, …:  yield (frame_{t-1}, frame_t)
```

- Each `DisplacementField` is the **incremental** field between
  consecutive frames.
- Stays comfortably inside `search_radius` so long as inter-frame
  motion does so (the case for typical loading-step bone+screw
  experiments).
- Cumulative absolute field requires composition (§3
  `compose_displacement_fields`); errors random-walk over time.
- Dispatcher integration: each pair uploads two volumes (no resident
  reference). The previous deformed becomes the next reference; a v2
  optimization can keep the previous frame on-device, but v1 ships
  without that.

### C. `UPDATED_REFERENCE` (warm-start, v2)

Reserved. Anchored to `frame_0` but warps `frame_t` by the previous
cumulative field before correlating, so each call sees only the
residual. Requires the v2 warper (overview §10 IC-GN row). The v1
driver accepts the enum value and raises `NotImplementedError` so
the API is stable.

### Selection summary

| Strategy | Per-pair displacement | Drift | Search-radius pressure | Lands in |
|---|---|---|---|---|
| `REFERENCE_ANCHORED` | grows with `t` | none | high (cap series length) | v1 |
| `SEQUENTIAL` | small, bounded | accumulates | low | v1 (default) |
| `UPDATED_REFERENCE` | residual only | none | minimal | v2 (with warper) |

The choice between A and B is empirical: run both on the experimental
GT for a short series; the per-pair vs cumulative error split (§7)
tells you whether B's drift dominates A's SNR loss. Default is B
because v1's `correlate()` is unmodified.

## 5. Driver flow

```
correlate_series(frames, mask, *, strategy, dispatcher, on_pair, **kw):

    1. validate kwargs once (window, overlap, ncc_mode, …)
    2. consume first frame: (0, ref_or_seed)
       - REFERENCE_ANCHORED: stash as anchored_reference
       - SEQUENTIAL:         stash as prev_frame
    3. build GridSpec from ref_or_seed.shape + window + overlap (one time)
    4. resolve mask, run filter_by_mask once → admitted_idx (one time)
    5. for each subsequent (t, frame):
           pair_ref, pair_def = strategy.pick(prev, anchored, frame)
           try:
               field_t = _run_one(pair_ref, pair_def, mask, dispatcher, **kw)
               status_t = OK
           except Exception as exc:
               field_t = _failed_field(grid, status=FAILED, reason=repr(exc))
               status_t = FAILED
           records.append((t, field_t, status_t))
           if on_pair: on_pair(t, field_t)
           if strategy is SEQUENTIAL: prev = frame   # promote
           if status_t is FAILED and strategy is SEQUENTIAL:
               break out of cumulative composition; subsequent pairs
               keep computing (incremental) but cumulative is undefined
    6. return DisplacementSeries(records, grid, strategy, …)
```

`_run_one` is one branch:

```python
if dispatcher is not None:
    return dispatcher.correlate(pair_ref, pair_def, mask=mask, **kw)
return correlate(pair_ref, pair_def, mask=mask, **kw)
```

The grid + admission re-use across pairs is the *only* substantive
optimization in the driver. Everything else is plumbing that v1
already has (per-pair) — series gets it amortized.

## 6. Data structures (additions to `mamba_dvc/types.py`)

```python
class PairingStrategy(StrEnum):
    REFERENCE_ANCHORED = "reference_anchored"
    SEQUENTIAL         = "sequential"
    UPDATED_REFERENCE  = "updated_reference"   # v2; raises in driver


class SeriesPairStatus(IntEnum):
    OK     = 0
    FAILED = 1   # correlator raised; field carries zeros + status=MASKED everywhere


@dataclass(frozen=True)
class DisplacementSeries:
    """Ordered DVC results across a frame sequence.

    All fields share ``grid`` (built once at series start). Any
    per-frame metadata that varies (timestamps, file paths, status)
    lives in this dataclass, not duplicated into each
    ``DisplacementField``.
    """

    fields:        tuple[DisplacementField, ...]   # length == n_pairs
    pair_indices:  Int64[np.ndarray, "pairs 2"]    # (t_ref, t_def) per pair
    pair_status:   UInt8[np.ndarray, "pairs"]      # SeriesPairStatus
    grid:          GridSpec
    strategy:      PairingStrategy
    timestamps:    tuple[float, ...] | None = None # optional, caller-supplied

    def cumulative(
        self,
        *,
        interpolation: Literal["linear", "cubic"] = "linear",
    ) -> tuple[DisplacementField, ...]:
        """Lazy: composes incremental fields into absolute fields.

        SEQUENTIAL: composes step-by-step via
        ``core.field_ops.compose_displacement_fields``.
        REFERENCE_ANCHORED: returns ``self.fields`` as-is (already absolute).
        Halts at the first ``SeriesPairStatus.FAILED`` boundary.
        """
        ...
```

`DisplacementSeries` is `frozen=True`. `cumulative()` returns a fresh
tuple; series-level callers cache the result if they need it twice.

## 7. Error evaluation across time (`mamba_dvc/validate/series_error.py`)

Two distinct error notions, both useful, neither subsumes the other.
The module emits both inside one `SeriesErrorReport`.

### Per-pair error

For each `t` in the series, evaluate the per-pair field against the
ground-truth field at `t`:

| Stat | What it answers |
|---|---|
| `mae[t]`, `rmse[t]`, `p95[t]` | Overall accuracy at frame `t`. |
| `error_by_distance[t]` | MAE stratified by distance from screw mask boundary. Tells you whether errors localize near the boundary as the experiment progresses. |
| `confidence_vs_error[t]` | Spearman correlation between `field.confidence` and `|estimated − gt|` at OK POIs. Falls when the confidence signal stops being predictive — early warning for a strategy switch. |

Per-pair stats run as the `on_pair` callback. They are independent
across `t` and parallel-trivial; nothing the driver does affects them.

### Cumulative / drift error (sequential strategy only)

The new thing the series enables. For each `t`:

```
U_t       = compose(U_{t-1}, field_t)        # incremental composition
gt_total  = gt_field_at_t                    # absolute GT at frame t
drift[t]  = stats(U_t − gt_total)            # MAE / RMSE / p95
```

This separates two error sources that per-pair stats conflate:
1. **Per-step measurement noise** — caught by `mae[t]`.
2. **Cumulative drift** — caught by `drift[t]`. If `drift[t] − drift[t-1]`
   exceeds the per-pair MAE meaningfully, the increments are biased,
   not just noisy: the system is random-walking away from the truth.

For `REFERENCE_ANCHORED`, `drift[t] == mae[t]` by construction;
report only the per-pair stats.

### Report shape

```python
@dataclass(frozen=True)
class SeriesErrorReport:
    per_pair:   PerPairErrorTable          # rows of (t, mae, rmse, p95, …)
    cumulative: CumulativeDriftTable | None  # None for REFERENCE_ANCHORED
    strategy:   PairingStrategy
    n_pairs:    int

def evaluate(
    series: DisplacementSeries,
    gt_fields: Sequence[DisplacementField],
    *,
    distance_bins: Sequence[float] = (0, 5, 10, 20, 50),
) -> SeriesErrorReport: ...
```

`evaluate` is pure-CPU (per-pair stats) plus one composition pass
(cumulative). Cheap; safe to run in CI on small synthetic series.

## 8. I/O integration (`mamba_dvc/io/series.py`)

Out of scope for the driver but part of the buildout:

- **Frame iterator** with thread-prefetch (`prefetch=N` future
  reads). Default reader is zarr v3 (overview §8); tensorstore as an
  async alternative for GDS / SSD-bound runs.
- **Streaming writer** for `DisplacementSeries`. Each
  `DisplacementField` is ~100s of KB; serializing to a single zarr
  group with one array per attribute (`displacements`, `confidence`,
  `status`) and a leading `pairs` axis is the minimum-fuss option.
  Writer hooks in via `on_pair`.
- **GT loader** that yields `(t, gt_field_t)` aligned with the frame
  iterator. Held in memory for the whole series; total size ≈ n_pairs
  × points × 12 B (≈ 1.4 MB per pair at v1 grid scales).

The driver never touches these directly. They're orchestrated at the
CLI / notebook level.

## 9. CLI integration (`mamba_dvc/cli.py`)

```
mamba-dvc series \
    --frames PATH [PATH ...] \
    --mask PATH \
    --strategy {reference_anchored,sequential} \
    --gt-fields PATH [PATH ...] \
    --out PATH \
    [--devices 0,1,2,3] \
    [--window 96] [--overlap 0.5] ...
```

Behavior:
1. Open frames + mask + GT via `io/series.py`.
2. If `--devices`: open `MultiGPUDispatcher` as a context manager.
3. Call `correlate_series(...)` with an `on_pair` that streams fields
   to `--out` and accumulates into a `SeriesErrorReport`.
4. Write the report alongside the field zarr.
5. Print summary: per-pair mean MAE, max cumulative drift,
   FAILED-pair count.

The CLI is the single integration test for the whole stack;
everything below is unit-testable without it.

## 10. Test surface

Pure logic (no GPU, no I/O):

- `tests/pipeline/test_series.py::TestStrategySelection` — each
  `PairingStrategy` value selects the right `(ref, def)` pairs for a
  3-frame stub.
- `tests/pipeline/test_series.py::TestGridReuse` — driver builds the
  grid once; instrument `build_grid` to assert it ran exactly once
  for an N-pair series.
- `tests/pipeline/test_series.py::TestOnPairCallback` — `on_pair`
  fires once per pair, in order, with the correct `t`.
- `tests/pipeline/test_series.py::TestFailureIsolation` — a single
  failing pair (synthetic exception inside a stub correlator) marks
  one row `FAILED`; subsequent pairs still run.
- `tests/types/test_series.py::TestDisplacementSeriesCumulative` —
  composition over a synthetic 3-frame integer-shift series matches
  the analytical total field.
- `tests/core/test_field_ops.py::TestComposeIdempotent` —
  `compose(field, zero_field) == field`; `compose(zero, field) == field`.
- `tests/validate/test_series_error.py::TestPerPairAndDriftSplit` —
  on a synthetic series with known per-step error, the per-pair
  stats match the input, and `drift[t]` reflects cumulative
  composition (random-walk variance check).

Dispatch-level (no GPU):

- `tests/gpu/test_dispatcher_lifetime.py::TestEnterExitNoSpawn` —
  `MultiGPUDispatcher(device_ids=[0])` happy path runs without
  spawning when only one device is requested (mirrors current
  `device_ids=[0]` in-process path in `dispatch.py`).

End-to-end with GPU (`@pytest.mark.gpu @pytest.mark.slow`):

- `tests/integration/test_series_smoke.py` — 5-frame synthetic series,
  `device_ids=[0, 0]`, both strategies. Asserts series MAE matches
  single-pair MAE within float32 noise.
- Full multi-GPU production path (`device_ids=[0, 1, 2, 3]`, real
  experimental data) is not in CI; canonical performance smoke test
  before tagging.

## 11. Performance estimate

Defaults (overview §7): `W=96`, ~12 k POIs, single pair on 4× A6000
runs in ~3 s of compute + ~6 s of dispatcher overhead = ~9 s/pair
cold.

For a 100-frame `SEQUENTIAL` series:

| Path | Per-pair | Total |
|---|---|---|
| v1 dispatch (spawn per call) | ~9 s | ~15 min |
| `MultiGPUDispatcher` (spawn once) | ~3 s | ~5 min |
| + `REFERENCE_ANCHORED` (resident ref) | ~2.5 s | ~4 min |
| + I/O prefetch (overlap with compute) | ~2 s | ~3.5 min |

Each step is independent; the buildout lands them in this order so
each can be measured against the previous on the experimental data
before adding the next.

## 12. Known non-features (by design, v1)

- **No dynamic re-anchoring.** A supervisor that switches
  `REFERENCE_ANCHORED → SEQUENTIAL` mid-run when SNR collapses is
  v3. Pick one strategy per run.
- **No per-frame deformed masks.** Same single shared mask as v1.
  Per-frame masks are an orthogonal v2 feature (overview §10) that
  can be added to the driver signature later without refactor.
- **No iterative-warp inside series.** Series and warp are
  orthogonal: warp is intra-pair, series is inter-pair. Adding IC-GN
  later means the driver calls `correlate_iterative()` instead of
  `correlate()` — no series-level change.
- **No checkpoint / resume.** The driver consumes the iterator once.
  A failed run restarts from frame 0. The streaming writer makes
  partial output recoverable; full resume is v2.
- **No async writer.** `on_pair` runs synchronously inside the
  driver. If write throughput becomes the bottleneck, wrap the
  callback in a thread queue at the CLI level.

## 13. Open questions (non-blocking)

- **Should the driver expose a warm-start parameter for the v2
  `UPDATED_REFERENCE` path now?** Probably yes — add
  `initial_field: DisplacementField | None = None` to the per-pair
  forwarded kwargs, raise on use until the warper lands. Stabilizes
  the signature.
- **Should `DisplacementSeries.cumulative()` cache?** Default no
  (frozen + lazy); add an explicit `with_cumulative()` constructor if
  callers churn on it.
- **Should failure mode be configurable (abort vs continue)?**
  Likely yes via `on_pair` returning a sentinel. Defer until a real
  use case appears.
- **Do we want `pair_indices` as `int64` or named tuples?** Going
  `int64[N, 2]` for jaxtyping consistency; named-tuple wrappers can
  be a thin convenience method on `DisplacementSeries`.

## 14. Buildout order

Each step is its own commit, runs independently of the next, and
ships with the test additions called out in §10.

1. `types.py`: `PairingStrategy`, `SeriesPairStatus`,
   `DisplacementSeries` (no cumulative method yet).
2. `pipeline/series.py`: driver + strategy selector. No dispatcher
   integration; routes through `correlate()`. Synthetic-only tests.
3. `core/field_ops.py`: `compose_displacement_fields` + tests.
4. `DisplacementSeries.cumulative()` wired to `field_ops`.
5. `validate/series_error.py`: `evaluate(...)` + report types.
6. `gpu/dispatch.py`: `MultiGPUDispatcher` context manager. Refactor
   existing `_run_multi_process` into the dispatcher; the per-call
   `correlate_multi_gpu()` becomes a thin "spawn-and-tear-down"
   wrapper around it for back-compat.
7. `pipeline/series.py`: dispatcher branch in `_run_one`. End-to-end
   GPU smoke test.
8. `io/series.py`: lazy frame iterator + zarr writer. Integration via
   `on_pair`.
9. `cli.py`: `series` subcommand.
10. `docs/buildout/timestep-pipeline.md`: mark superseded sections;
    move stable parts into `docs/plans/series.md`.

The split between `docs/buildout/` (this doc — sequencing, intent,
open questions) and `docs/plans/` (post-landing contracts) mirrors
how `correlate.md` and `dispatch.md` solidified after their initial
buildout.
