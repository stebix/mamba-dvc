"""Programmatic error evaluation for the time-series pipeline.

The top-level entry point :func:`evaluate_synthetic` takes a reference
volume plus an analytical temporal displacement field
``u(coords, t)`` and produces error curves across a configurable
sweep of pairing strategies and frame lags. It is the rapid-iteration
deliverable for the time-step buildout — single-device, in-memory,
no I/O, no CLI.

Two distinct error notions are reported, each keyed by
``(strategy, lag)``:

* **Per-pair error** (every strategy): MAE / RMSE / p95 of the
  recovered field vs. the analytical inter-frame GT
  ``u(x, t_def) - u(x, t_ref)``. Independent across pairs;
  captures per-step measurement noise.
* **Cumulative drift** (:data:`PairingStrategy.SEQUENTIAL` only):
  MAE / RMSE / p95 of the *composed* cumulative field vs. the
  analytical absolute GT ``u(x, t_def) - u(x, t_start)``. Separates
  random-walk drift from per-step noise.

The module deliberately does no plotting — output is flat arrays
indexed by ``(strategy, lag)`` so the caller (notebook, script) can
plot, save, or further process as they see fit.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32, Int64, UInt8

from mamba_dvc.core.grid import build_grid
from mamba_dvc.pipeline._internal import NCCMode, NCCNormalization
from mamba_dvc.pipeline.series import correlate_series
from mamba_dvc.types import (
    DisplacementField,
    DisplacementSeries,
    PairingStrategy,
    SeriesPairStatus,
)
from mamba_dvc.validate.synthetic import make_series, normalize_temporal_form

__all__ = [
    "CumulativeDriftTable",
    "PerPairErrorTable",
    "SyntheticEvalReport",
    "evaluate_synthetic",
]


@dataclass(frozen=True)
class PerPairErrorTable:
    """Per-pair error stats for one ``(strategy, lag)`` configuration.

    Parameters
    ----------
    t_ref, t_def
        ``(pairs,)`` int64 arrays — pair indices from the driver.
    pair_status
        ``(pairs,)`` uint8 :class:`SeriesPairStatus` per pair. Failed
        pairs carry NaN in the stat columns and ``0`` in
        :attr:`n_valid`.
    mae, rmse, p95
        ``(pairs,)`` float32 statistics of the per-POI Euclidean error
        magnitude ``||u_hat - u_gt||``, restricted to POIs with
        :attr:`DisplacementField.valid` True. NaN where no POIs were
        valid or the pair failed.
    n_valid
        ``(pairs,)`` int64 count of valid POIs that contributed to
        each row's statistics.
    """

    t_ref: Int64[np.ndarray, "pairs"]
    t_def: Int64[np.ndarray, "pairs"]
    pair_status: UInt8[np.ndarray, "pairs"]
    mae: Float32[np.ndarray, "pairs"]
    rmse: Float32[np.ndarray, "pairs"]
    p95: Float32[np.ndarray, "pairs"]
    n_valid: Int64[np.ndarray, "pairs"]


@dataclass(frozen=True)
class CumulativeDriftTable:
    """Cumulative-drift stats for one SEQUENTIAL ``(strategy, lag)`` run.

    Parameters
    ----------
    t
        ``(n,)`` int64 absolute deformed-frame timestep at each
        cumulative step. ``n`` equals the number of consecutive OK
        pairs in the source series (composition halts at the first
        failure).
    mae, rmse, p95
        ``(n,)`` float32 stats of ``||u_cumulative - u_gt_absolute||``
        over valid POIs. NaN where no POIs remained valid through the
        cumulative chain.
    n_valid
        ``(n,)`` int64 count of valid POIs that contributed.
    """

    t: Int64[np.ndarray, "n"]
    mae: Float32[np.ndarray, "n"]
    rmse: Float32[np.ndarray, "n"]
    p95: Float32[np.ndarray, "n"]
    n_valid: Int64[np.ndarray, "n"]


@dataclass(frozen=True)
class SyntheticEvalReport:
    """Aggregated report from :func:`evaluate_synthetic`.

    Parameters
    ----------
    per_pair
        Mapping ``(strategy, lag) -> PerPairErrorTable``. Populated
        for every ``(strategy, lag)`` combo that yielded at least one
        pair.
    cumulative
        Mapping ``(strategy, lag) -> CumulativeDriftTable``. Populated
        for :data:`PairingStrategy.SEQUENTIAL` only (anchored strategies
        have ``drift == mae`` by construction and the table would be
        redundant with :attr:`per_pair`).
    timesteps
        Full set of timesteps materialized by
        :func:`make_series` — superset of the timesteps used by any
        individual ``(strategy, lag)`` run.
    lags
        Lag values evaluated, in input order.
    strategies
        Strategy values evaluated, in input order.
    """

    per_pair: dict[tuple[PairingStrategy, int], PerPairErrorTable]
    cumulative: dict[tuple[PairingStrategy, int], CumulativeDriftTable]
    timesteps: tuple[int, ...]
    lags: tuple[int, ...]
    strategies: tuple[PairingStrategy, ...]


_DEFAULT_STRATEGIES: tuple[PairingStrategy, ...] = (
    PairingStrategy.REFERENCE_ANCHORED,
    PairingStrategy.SEQUENTIAL,
)


def evaluate_synthetic(
    reference: Float32[np.ndarray, "z y x"],
    u_of_t: Callable[..., object],
    timesteps: Sequence[int],
    *,
    strategies: Sequence[PairingStrategy] = _DEFAULT_STRATEGIES,
    lags: Sequence[int] = (1,),
    mask: Bool[np.ndarray, "z y x"] | None = None,
    warp_order: int = 3,
    cumulative_interpolation: Literal["linear", "cubic"] = "linear",
    window: int | tuple[int, int, int] = 96,
    overlap: float = 0.5,
    mask_threshold: float = 0.9,
    tukey_alpha: float | None = None,
    search_radius: int | None = None,
    batch_size: int | Literal["auto"] = "auto",
    eps: float = 1e-12,
    ncc_mode: NCCMode = NCCMode.LINEAR,
    ncc_normalization: NCCNormalization = NCCNormalization.OVERLAP,
) -> SyntheticEvalReport:
    """Sweep pairing strategies and frame lags on a synthetic series and report errors.

    Workflow:

    1. Materialize warped frames at every ``t`` in ``timesteps`` via
       :func:`make_series`. Frames are reused across the strategy and
       lag sweep — total host memory ~ ``len(timesteps) * prod(shape) * 4 B``.
    2. For each ``lag``, subsample ``timesteps[::lag]`` and the
       corresponding frames.
    3. For each ``(strategy, lag)`` pair, run
       :func:`correlate_series` against the subsampled stream and
       compute per-pair stats versus the analytical inter-frame GT.
    4. For :data:`PairingStrategy.SEQUENTIAL` runs additionally
       compose cumulative fields via
       :meth:`DisplacementSeries.cumulative` and compute drift stats
       versus the analytical absolute GT.

    Parameters
    ----------
    reference
        ``(z, y, x)`` float32 reference volume. The temporal field
        ``u(coords, timesteps[0])`` is the displacement applied to
        this reference to produce the first frame; in the canonical
        convention that field is zero so ``frames[0]`` matches
        ``reference`` up to cubic-spline prefilter noise.
    u_of_t
        Temporal displacement field. Pointwise
        ``(coords, t) -> displacements`` or curried
        ``t -> DisplacementFunction`` — dispatch is by positional
        arity. See :func:`mamba_dvc.validate.synthetic.normalize_temporal_form`.
    timesteps
        Integer frame indices to materialize. Must be non-empty,
        strictly increasing, with no duplicates.
    strategies
        Pairing strategies to evaluate. Defaults to
        ``(REFERENCE_ANCHORED, SEQUENTIAL)``.
        :data:`PairingStrategy.UPDATED_REFERENCE` is rejected — it is
        reserved for v2 and the driver raises on it.
    lags
        Lag values (integer strides through ``timesteps``). ``lag=k``
        means the run consumes ``timesteps[::k]``; for
        :data:`PairingStrategy.SEQUENTIAL` this pairs ``(timesteps[0],
        timesteps[k]), (timesteps[k], timesteps[2k]), …``; for
        :data:`PairingStrategy.REFERENCE_ANCHORED` it pairs
        ``(timesteps[0], timesteps[k]), (timesteps[0], timesteps[2k]), …``.
        Each lag must be a positive int. Lags that yield fewer than 2
        frames are skipped with a :class:`RuntimeWarning`.
    mask
        Optional shared validity mask. Forwarded to both
        :func:`make_series` and :func:`correlate_series`.
    warp_order
        Spline order for :func:`scipy.ndimage.map_coordinates` used
        inside frame synthesis. Default ``3`` (cubic); set to ``1`` for
        faster frame generation when absolute MAE accuracy is not
        being reported (the trilinear path biases the Gaussian
        subvoxel fit and inflates the measurable error floor below
        ~0.2 vx).
    cumulative_interpolation
        Interpolation order for
        :func:`mamba_dvc.core.field_ops.compose_displacement_fields`.
        ``"linear"`` (default) or ``"cubic"``.
    window, overlap, mask_threshold, tukey_alpha, search_radius
        Forwarded to :func:`correlate_series` (and through to
        :func:`correlate`). Window + overlap also drive the GT
        sampling grid via :func:`build_grid` so error stats compare
        like-with-like.
    batch_size, eps, ncc_mode, ncc_normalization
        Forwarded to :func:`correlate_series` unchanged.

    Returns
    -------
    SyntheticEvalReport
        Per-pair and cumulative-drift tables keyed by ``(strategy, lag)``,
        plus the materialized ``timesteps``/``lags``/``strategies``.

    Raises
    ------
    ValueError
        If ``timesteps`` is empty / non-monotonic / has duplicates;
        if ``strategies`` or ``lags`` is empty; if any lag is
        non-positive; if any strategy is
        :data:`PairingStrategy.UPDATED_REFERENCE`.

    Warnings
    --------
    Emits a :class:`RuntimeWarning` if a requested lag yields fewer
    than 2 frames (no pair possible). Per-pair correlator failures
    surface as warnings from inside the driver, not here.

    Notes
    -----
    The warp convention is hardcoded to ``"pull_back"`` so the DVC
    output sign matches the analytical field directly. Callers
    wanting ``"push_forward"`` synthesis need to bypass this entry
    point and orchestrate :func:`make_series` +
    :func:`correlate_series` manually with a manual sign flip on the
    GT.
    """
    if reference.ndim != 3:
        raise ValueError(f"reference must be 3D, got ndim={reference.ndim}")
    if reference.dtype != np.float32:
        raise ValueError(f"reference must be float32, got {reference.dtype}")

    ts = tuple(int(t) for t in timesteps)
    if not ts:
        raise ValueError("timesteps must be non-empty")
    if len(set(ts)) != len(ts):
        raise ValueError(f"timesteps must have no duplicates, got {ts}")
    if any(b <= a for a, b in itertools.pairwise(ts)):
        raise ValueError(f"timesteps must be strictly increasing, got {ts}")

    strategies_tuple = tuple(strategies)
    if not strategies_tuple:
        raise ValueError("strategies must be non-empty")
    for s in strategies_tuple:
        # ``Sequence[PairingStrategy]`` is the declared input type, but the
        # runtime check is a defensive guard against callers that ignore the
        # annotation (passing raw strings, ints, etc.).
        if not isinstance(s, PairingStrategy):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(
                f"strategies entries must be PairingStrategy instances, got {type(s).__name__}"
            )
        if s is PairingStrategy.UPDATED_REFERENCE:
            raise ValueError(
                "PairingStrategy.UPDATED_REFERENCE is reserved for v2; the driver"
                " will raise NotImplementedError on it"
            )

    lags_tuple = tuple(int(lag) for lag in lags)
    if not lags_tuple:
        raise ValueError("lags must be non-empty")
    if any(lag <= 0 for lag in lags_tuple):
        raise ValueError(f"lags entries must be positive, got {lags_tuple}")

    shape: tuple[int, int, int] = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
    )

    temporal = normalize_temporal_form(u_of_t)

    series_bundle = make_series(
        shape,
        temporal,
        ts,
        reference=reference,
        mask=mask,
        order=warp_order,
        convention="pull_back",
    )

    # Build the GT sampling grid once — same parameters the driver uses
    # so per-pair and cumulative error stats compare like-with-like.
    grid = build_grid(shape, window=window, overlap=overlap)

    # Pre-evaluate the analytical field on the GT grid at every requested
    # ``t``. Each entry is a (points, 3) float32 array. Reuse across
    # strategies and lags rather than re-computing.
    gt_per_t: dict[int, Float32[np.ndarray, "points 3"]] = {
        t: np.asarray(temporal(grid.positions, float(t)), dtype=np.float32) for t in ts
    }

    per_pair_out: dict[tuple[PairingStrategy, int], PerPairErrorTable] = {}
    cumulative_out: dict[tuple[PairingStrategy, int], CumulativeDriftTable] = {}

    for lag in lags_tuple:
        selected_ts = ts[::lag]
        selected_frames = series_bundle.frames[::lag]
        if len(selected_ts) < 2:
            warnings.warn(
                f"lag={lag} subsamples timesteps to {len(selected_ts)} frame(s);"
                " no pairs are possible — skipping every strategy at this lag.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        for strategy in strategies_tuple:
            frame_iter = zip(selected_ts, selected_frames, strict=True)
            series_result = correlate_series(
                frame_iter,
                mask=mask,
                strategy=strategy,
                window=window,
                overlap=overlap,
                mask_threshold=mask_threshold,
                tukey_alpha=tukey_alpha,
                search_radius=search_radius,
                batch_size=batch_size,
                eps=eps,
                ncc_mode=ncc_mode,
                ncc_normalization=ncc_normalization,
            )

            per_pair_out[(strategy, lag)] = _per_pair_table(series_result, gt_per_t)

            if strategy is PairingStrategy.SEQUENTIAL:
                cumulative_fields = series_result.cumulative(
                    interpolation=cumulative_interpolation
                )
                cumulative_out[(strategy, lag)] = _cumulative_table(
                    cumulative_fields, series_result, gt_per_t
                )

    return SyntheticEvalReport(
        per_pair=per_pair_out,
        cumulative=cumulative_out,
        timesteps=ts,
        lags=lags_tuple,
        strategies=strategies_tuple,
    )


def _error_magnitude_stats(
    estimated: Float32[np.ndarray, "points 3"],
    gt: Float32[np.ndarray, "points 3"],
    valid: Bool[np.ndarray, "points"],
) -> tuple[float, float, float, int]:
    """Return ``(mae, rmse, p95, n_valid)`` for the Euclidean error magnitude.

    Stats are computed only over POIs with ``valid`` True. Returns
    ``(nan, nan, nan, 0)`` when no POIs are valid so downstream tables
    do not silently coerce empty arrays to zeros.
    """
    if not bool(np.any(valid)):
        return (float("nan"), float("nan"), float("nan"), 0)
    err = (estimated - gt).astype(np.float64, copy=False)
    err_mag = np.linalg.norm(err, axis=1)
    em = err_mag[valid]
    mae = float(np.mean(em))
    rmse = float(np.sqrt(np.mean(em**2)))
    p95 = float(np.percentile(em, 95))
    return (mae, rmse, p95, int(np.sum(valid)))


def _per_pair_table(
    series: DisplacementSeries,
    gt_per_t: dict[int, Float32[np.ndarray, "points 3"]],
) -> PerPairErrorTable:
    """Assemble the per-pair error table from a finished series."""
    n = len(series.fields)
    t_ref = series.pair_indices[:, 0].astype(np.int64, copy=True)
    t_def = series.pair_indices[:, 1].astype(np.int64, copy=True)
    pair_status = series.pair_status.astype(np.uint8, copy=True)

    mae = np.full(n, np.nan, dtype=np.float32)
    rmse = np.full(n, np.nan, dtype=np.float32)
    p95 = np.full(n, np.nan, dtype=np.float32)
    n_valid = np.zeros(n, dtype=np.int64)

    for i, field in enumerate(series.fields):
        if int(pair_status[i]) != int(SeriesPairStatus.OK):
            continue
        gt = gt_per_t[int(t_def[i])] - gt_per_t[int(t_ref[i])]
        m, r, p, k = _error_magnitude_stats(field.displacements, gt, field.valid)
        mae[i] = np.float32(m)
        rmse[i] = np.float32(r)
        p95[i] = np.float32(p)
        n_valid[i] = k

    return PerPairErrorTable(
        t_ref=t_ref,
        t_def=t_def,
        pair_status=pair_status,
        mae=mae,
        rmse=rmse,
        p95=p95,
        n_valid=n_valid,
    )


def _cumulative_table(
    cumulative_fields: tuple[DisplacementField, ...],
    series: DisplacementSeries,
    gt_per_t: dict[int, Float32[np.ndarray, "points 3"]],
) -> CumulativeDriftTable:
    """Assemble the cumulative-drift table from composed sequential fields."""
    n = len(cumulative_fields)
    if n == 0:
        return CumulativeDriftTable(
            t=np.empty(0, dtype=np.int64),
            mae=np.empty(0, dtype=np.float32),
            rmse=np.empty(0, dtype=np.float32),
            p95=np.empty(0, dtype=np.float32),
            n_valid=np.empty(0, dtype=np.int64),
        )

    # cumulative() truncates at the first FAILED pair, so the first ``n``
    # rows of pair_indices are all OK by construction.
    t_start = int(series.pair_indices[0, 0])
    t_def_arr = series.pair_indices[:n, 1].astype(np.int64, copy=True)

    mae = np.full(n, np.nan, dtype=np.float32)
    rmse = np.full(n, np.nan, dtype=np.float32)
    p95 = np.full(n, np.nan, dtype=np.float32)
    n_valid = np.zeros(n, dtype=np.int64)

    for i, cumfield in enumerate(cumulative_fields):
        t_end = int(t_def_arr[i])
        gt = gt_per_t[t_end] - gt_per_t[t_start]
        m, r, p, k = _error_magnitude_stats(cumfield.displacements, gt, cumfield.valid)
        mae[i] = np.float32(m)
        rmse[i] = np.float32(r)
        p95[i] = np.float32(p)
        n_valid[i] = k

    return CumulativeDriftTable(
        t=t_def_arr,
        mae=mae,
        rmse=rmse,
        p95=p95,
        n_valid=n_valid,
    )
