# Timestep pipeline — triage

Snapshot of the `temporal-buildout` worktree against
`docs/buildout/timestep-pipeline.md` §14. Captured 2026-05-13; the
working tree at the time held the eight files listed under "Landed"
below as uncommitted changes (2 modified, 6 new).

## Landed

| Buildout step | Path | Status | Notes |
|---|---|---|---|
| §14.1 `types.py` additions | `mamba_dvc/types.py` | ✅ | `PairingStrategy`, `SeriesPairStatus`, `DisplacementSeries`. The `cumulative()` method is implemented inline (step §14.4 fused). |
| §14.2 driver | `mamba_dvc/pipeline/series.py` | ✅ partial | Single-device only. Every pair routes through `pipeline.correlate.correlate()`; the `dispatcher` kwarg from doc §3 is not exposed. Failure isolation via per-pair `try/except` + `RuntimeWarning`. Series grid built once from the seed frame. |
| §14.3 field composition | `mamba_dvc/core/field_ops.py` | ✅ | `compose_displacement_fields` (semi-Lagrangian, linear/cubic). Bonus: `sample_field_at` (off-lattice resampler reusing the same interpolator). |
| §14.4 `DisplacementSeries.cumulative()` | (in `types.py`) | ✅ | Halts at the first `SeriesPairStatus.FAILED` boundary; `REFERENCE_ANCHORED` returns the OK-prefix verbatim. |
| §14.5 series error | `mamba_dvc/validate/series_error.py` | ⚠ scope drift | See "Scope drift" below. |
| (out of §14) synthetic temporal | `mamba_dvc/validate/synthetic.py` | ✅ | Added `SyntheticSeries`, `TemporalDisplacementFunction`, `make_series`, `linear_motion`, `normalize_temporal_form`, `temporal_from_curried`. Required by `evaluate_synthetic`. |

Tests landed: `tests/test_types_series.py`, `tests/pipeline/test_series.py`,
`tests/core/test_field_ops.py`, `tests/validate/test_series_error.py`,
`tests/validate/test_synthetic_series.py`.

## Not landed

| Buildout step | Path | Status |
|---|---|---|
| §14.6 `MultiGPUDispatcher` | `mamba_dvc/gpu/dispatch.py` | ❌ — class absent (only the per-call `correlate_multi_gpu` from v1). |
| §14.7 dispatcher branch in `_run_one` | `mamba_dvc/pipeline/series.py` | ❌ — `correlate_series` has no `dispatcher` kwarg. |
| §14.8 frame iterator + zarr writer | `mamba_dvc/io/series.py` | ❌ — module not present. |
| §14.9 CLI `series` subcommand | `mamba_dvc/cli.py` | ❌ — no `cli.py` exists at all yet. |
| §14.10 doc move into `docs/plans/series.md` | — | ❌ |

Test gaps relative to §10:

- `TestGridReuse` (assert `build_grid` runs exactly once per series) is
  not implemented. The current driver demonstrably builds once but
  there is no regression guard.
- GPU smoke test `tests/integration/test_series_smoke.py` requires
  §14.6–7 to land first.

## Scope drift: `validate/series_error.py`

Doc §7 spec:

```python
def evaluate(
    series: DisplacementSeries,
    gt_fields: Sequence[DisplacementField],
    *,
    distance_bins: Sequence[float] = (0, 5, 10, 20, 50),
) -> SeriesErrorReport
```

What landed:

```python
def evaluate_synthetic(
    reference, u_of_t, timesteps,
    *, strategies, lags,
    ... per-correlator kwargs ...,
) -> SyntheticEvalReport
```

Differences:

- **Contract inverted.** Doc form is a *pure* evaluator over an
  already-computed series + GT fields. What landed is an end-to-end
  runner that synthesizes frames, runs `correlate_series` internally,
  and stats the result. Two distinct jobs are fused.
- **Missing diagnostics.** `error_by_distance[t]` (distance-from-mask-
  boundary stratification) and `confidence_vs_error[t]` (Spearman
  correlation of `field.confidence` against `|estimated − gt|`) are
  absent. Current report carries MAE / RMSE / p95 / `n_valid` only.
- **Gained sweep semantics.** Strategy × lag matrix lives inside the
  function instead of in callers. Convenient for a single-script
  evaluation; less reusable for series that originate elsewhere
  (experimental GT, future dispatcher path).

### Reconciliation options

1. **Keep `evaluate_synthetic` as a thin sweep harness; add a separate
   pure `evaluate(series, gt_fields, …)` per §7.** Lowest churn. The
   harness becomes one of several entry points.
2. **Refactor into `(materialize_synthetic_series, evaluate)` and have
   the harness call both.** Cleaner separation; reuses the evaluator
   for experimental GT and for any future dispatcher-routed series.
3. **Update the buildout doc §7 to match the as-built shape and drop
   the original signature.** Acknowledges reality; loses the
   distance/confidence diagnostics until they get added in a follow-up.

Option 2 is the cleanest path forward — the synthesis half doubles as
the test harness for the dispatcher integration when it lands.

## Recommended order

1. Run a first temporal-vs-DVC-strategy evaluation with the current
   implementation to confirm end-to-end output is sensible. Script:
   `scripts/eval_temporal_strategies.py` (supports `--ephemeral` to
   skip filesystem writes for smoke testing).
2. Resolve §7 contract drift (option 2 above) — splits the pure
   evaluator from the synthetic harness so experimental GT can reuse
   the same stats path.
3. Land `MultiGPUDispatcher` (§14.6) and the dispatcher branch in
   `correlate_series` (§14.7). Biggest remaining performance unlock
   per doc §11 (~9 s → ~3 s per pair for the default workload).
4. `io/series.py` (§14.8) and `cli.py` (§14.9) once the dispatcher
   path is verified end-to-end.
5. Move stable sections of `docs/buildout/timestep-pipeline.md` into
   `docs/plans/series.md` (§14.10).
