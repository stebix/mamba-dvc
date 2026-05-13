"""Score ``DisplacementField`` results against stored ground-truth fields.

CPU-only for the scoring layer. ``run_and_evaluate`` and ``sweep`` orchestrate
``correlate_multi_gpu`` calls and feed each result through ``evaluate_pair``;
``evaluate_pair`` itself never spawns or touches a GPU.

See ``docs/plans/zarr-interface.md`` §3 / §4 for the API and flow.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Bool, Float32
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree

from mamba_dvc.gpu.dispatch import correlate_multi_gpu
from mamba_dvc.instrument import timed
from mamba_dvc.io.dataset import DvcDataset, EvaluationPair
from mamba_dvc.types import DisplacementField

__all__ = [
    "BoundaryDistanceIndex",
    "BoundaryStratifiedTable",
    "CorrelateFn",
    "ErrorReport",
    "evaluate_pair",
    "run_and_evaluate",
    "sweep",
]


CorrelateFn = Callable[
    [
        Float32[np.ndarray, "z y x"],
        Float32[np.ndarray, "z y x"],
        Bool[np.ndarray, "z y x"] | None,
    ],
    DisplacementField,
]
"""Signature of a function turning ``(reference, deformed, mask)`` into a field.

Allows callers (and tests) to substitute a custom dispatch path while keeping
``run_and_evaluate`` / ``sweep`` agnostic of GPU details. The default wraps
``correlate_multi_gpu`` with the kwargs the caller supplied.
"""


@dataclass(frozen=True)
class BoundaryStratifiedTable:
    """MAE / RMSE binned by distance from the mask boundary.

    Bins are right-closed except the terminal bin, which is open
    (``d > edges[-1]``). With default edges ``(0, 5, 10, 20, 50)``:

    - bin 0: ``0 <= d <= 5`` (left edge inclusive when ``edges[0] == 0``)
    - bin 1: ``5 < d <= 10``
    - bin 2: ``10 < d <= 20``
    - bin 3: ``20 < d <= 50``
    - bin 4: ``d > 50`` (open terminal)

    Empty bins carry ``mae = rmse = NaN`` so empty-vs-zero is
    distinguishable.
    """

    edges: tuple[float, ...]
    counts: Float32[np.ndarray, "bins"]
    mae: Float32[np.ndarray, "bins"]
    rmse: Float32[np.ndarray, "bins"]


@dataclass(frozen=True)
class BoundaryDistanceIndex:
    """Amortized "distance to the valid-mask region" lookup for POIs.

    Build once per mask (:meth:`from_mask`); :meth:`query` it once per
    parameter variant evaluated against that mask. Reproduces
    ``scipy.ndimage.distance_transform_edt(~mask)`` sampled at rounded
    POI voxels, exactly:

    - a POI whose rounded center voxel is *inside* the mask (``True`` --
      the common case, windows are admitted on tissue) gets ``0.0``;
    - a POI whose center is *outside* the mask (``False`` -- possible
      when ``mask_threshold < 1``, since ``filter_by_mask`` tests the
      window's mask support, not the center voxel) gets the Euclidean
      distance from that voxel to the nearest ``True`` voxel.

    The nearest ``True`` voxel to any ``False`` voxel lies on the
    26-connected *foreground* boundary shell
    ``S = mask & binary_dilation(~mask, ones((3, 3, 3)))`` -- the
    discretized segment from a ``False`` voxel to *any* ``True`` voxel
    enters ``mask`` at a voxel of ``S`` no farther away -- so a KDTree
    over ``argwhere(S)`` gives the EDT value exactly while touching only
    the thin shell (order 1e6 to 1e7 voxels at a 0.1 to 0.2 foreground
    fraction) instead of the whole volume.

    (Note: ``docs/plans/kdtree-impl-plan.md`` §2 describes the EDT
    running the other way -- distance from a foreground POI to the
    nearest *background* voxel. That is the transform of ``mask``, not
    of ``~mask``; the code in ``_stratify_by_distance`` has always used
    ``~mask``, and this index matches it. See the test suite and
    perf-doc R5 for the behavior being preserved.)

    Parameters
    ----------
    shape
        ``(z, y, x)`` shape of the mask the index was built from.
    mask
        The source mask, kept so :meth:`query` can return ``0.0`` for a
        POI whose rounded center lands on a ``True`` voxel.
    tree
        ``scipy.spatial.KDTree`` over the foreground-shell voxel
        coordinates, or ``None`` when the shell is empty (mask all
        ``True`` -- every POI is on tissue, so all distances are ``0.0``
        -- or all ``False`` -- no tissue, so all distances are ``inf``).
    """

    shape: tuple[int, int, int]
    mask: Bool[np.ndarray, "z y x"]
    tree: Any  # scipy.spatial.KDTree | None (scipy.spatial ships no type stub)

    @classmethod
    def from_mask(cls, mask: Bool[np.ndarray, "z y x"]) -> BoundaryDistanceIndex:
        """Build the boundary index for ``mask``.

        Parameters
        ----------
        mask
            3D boolean mask (``True`` = valid tissue).

        Returns
        -------
        BoundaryDistanceIndex

        Raises
        ------
        ValueError
            If ``mask`` is not 3D or does not have boolean dtype.
        """
        if mask.ndim != 3:
            raise ValueError(f"mask must be 3D, got {mask.ndim}D")
        if mask.dtype != np.bool_:
            raise ValueError(f"mask must have bool dtype, got {mask.dtype}")
        with timed("evaluate.boundary_index_build", n_voxels=int(mask.size)):
            shell = mask & binary_dilation(~mask, structure=np.ones((3, 3, 3), dtype=bool))
            coords = np.argwhere(shell)
            tree = KDTree(coords) if coords.size else None
        shape = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
        return cls(shape=shape, mask=mask, tree=tree)

    def query(
        self, positions: Float32[np.ndarray, "points 3"]
    ) -> Float32[np.ndarray, "points"]:
        """Distance from each POI's rounded center voxel to the mask region.

        Parameters
        ----------
        positions
            ``(points, 3)`` POI centers in voxel coordinates. Rounded
            and clipped into ``shape`` the same way the EDT path does.

        Returns
        -------
        numpy.ndarray
            ``(points,)`` float32 distances: ``0.0`` for a POI whose
            rounded center voxel is inside the mask; otherwise the
            distance to the nearest ``True`` voxel (``inf`` when the
            mask is empty, matching a no-tissue volume).
        """
        rounded = np.clip(
            np.round(positions).astype(np.int64),
            a_min=0,
            a_max=np.asarray(self.shape, dtype=np.int64) - 1,
        )
        on_tissue = self.mask[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
        if self.tree is None:
            dist_to_tissue = np.full(len(rounded), np.inf)
        else:
            dist_to_tissue, _ = self.tree.query(rounded, k=1)
        return np.where(on_tissue, 0.0, dist_to_tissue).astype(np.float32)


@dataclass(frozen=True)
class ErrorReport:
    """Aggregate accuracy of one ``DisplacementField`` vs its GT."""

    name: str
    n_total: int
    n_valid: int
    mae: float
    rmse: float
    p95: float
    per_axis_mae: tuple[float, float, float]
    by_distance: BoundaryStratifiedTable | None
    confidence_vs_error: float
    estimated: DisplacementField
    truth: Float32[np.ndarray, "points 3"]


def evaluate_pair(
    pair: EvaluationPair,
    field: DisplacementField,
    *,
    distance_bins: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 50.0),
    truth: Float32[np.ndarray, "points 3"] | None = None,
    boundary_index: BoundaryDistanceIndex | None = None,
) -> ErrorReport:
    """Score ``field`` against ``pair.gt_field`` at the field's POIs.

    Parameters
    ----------
    pair
        Materialized inputs from :meth:`DvcDataset.load_pair`. Must
        carry a ground-truth field (``pair.gt_field is not None``)
        unless ``truth`` is supplied directly; otherwise raises.
    field
        Result of running ``correlate()`` / ``correlate_multi_gpu()``
        on ``pair``. POIs marked invalid (``field.valid == False``)
        are excluded from every aggregate.
    distance_bins
        Right-closed bin edges (with an open terminal bin) for the
        boundary-stratified table. Pass ``()`` to skip the table.
        When ``pair.mask is None``, the table is silently set to
        ``None`` (after a warning) regardless of this argument.
    truth
        Optional precomputed ground-truth displacements at
        ``field.positions``, shape ``(points, 3)``. When given, the
        internal ``pair.gt_field(field.positions)`` call is skipped —
        this lets a caller running many parameter variants against one
        materialized pair (same POI lattice) evaluate the GT field
        once and reuse it. The caller is responsible for it matching
        ``field.positions``; only the shape is checked here.
    boundary_index
        Optional precomputed :class:`BoundaryDistanceIndex` for
        ``pair.mask``. Mirrors ``truth``: a caller running many
        variants against one materialized pair builds it once (the
        mask is identical across, e.g., a ``mask_threshold`` sweep)
        and passes it here, avoiding the per-variant rebuild. When
        ``None`` the index is built on the fly — still much faster than
        the full-volume distance transform it replaces. Ignored when
        the boundary table is skipped (``pair.mask is None`` or
        ``distance_bins`` empty). Its ``shape`` must match
        ``pair.mask.shape``.

    Returns
    -------
    ErrorReport
        Frozen aggregate; ``ErrorReport.by_distance is None`` when
        the table was skipped or the mask was missing.

    Raises
    ------
    ValueError
        If ``pair.gt_field is None`` and ``truth`` is not supplied
        (real-deformation entry), if ``truth`` has a shape other than
        ``field.displacements.shape``, or if ``boundary_index.shape``
        does not match ``pair.mask.shape``.
    """
    if truth is None:
        if pair.gt_field is None:
            raise ValueError(
                f"pair {pair.name!r} has no ground truth (kind={pair.kind!r}); "
                f"evaluate_pair requires a synthetic entry or an explicit truth="
            )
        with timed("evaluate.gt_resample", n_poi=int(field.positions.shape[0])):
            truth = pair.gt_field(field.positions)
    else:
        truth = np.ascontiguousarray(truth, dtype=np.float32)
        if truth.shape != field.displacements.shape:
            raise ValueError(
                f"truth shape {truth.shape} does not match "
                f"field.displacements shape {field.displacements.shape}"
            )

    err = field.displacements - truth
    valid = field.valid

    if not bool(np.any(valid)):
        return _empty_report(name=pair.name, field=field, truth=truth)

    err_valid = err[valid]
    abs_err = np.abs(err_valid)
    err_norm = np.linalg.norm(err_valid, axis=1)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err_valid**2).mean()))
    p95 = float(np.percentile(err_norm, 95))
    per_axis_mae = (
        float(abs_err[:, 0].mean()),
        float(abs_err[:, 1].mean()),
        float(abs_err[:, 2].mean()),
    )

    confidence_vs_error = _spearman(field.confidence[valid], err_norm)

    by_distance: BoundaryStratifiedTable | None
    if len(distance_bins) == 0:
        by_distance = None
    elif pair.mask is None:
        warnings.warn(
            "evaluate_pair: pair.mask is None; skipping boundary-stratified table",
            stacklevel=2,
        )
        by_distance = None
    else:
        if boundary_index is not None and boundary_index.shape != pair.mask.shape:
            raise ValueError(
                f"boundary_index.shape {boundary_index.shape} does not match "
                f"pair.mask.shape {pair.mask.shape}"
            )
        index = boundary_index or BoundaryDistanceIndex.from_mask(pair.mask)
        with timed("evaluate.stratify", n_poi=int(err_norm.shape[0])):
            by_distance = _stratify_by_distance(
                index=index,
                positions=field.positions[valid],
                err_norm=err_norm,
                err=err_valid,
                edges=tuple(float(e) for e in distance_bins),
            )

    return ErrorReport(
        name=pair.name,
        n_total=int(field.status.size),
        n_valid=int(np.sum(valid)),
        mae=mae,
        rmse=rmse,
        p95=p95,
        per_axis_mae=per_axis_mae,
        by_distance=by_distance,
        confidence_vs_error=confidence_vs_error,
        estimated=field,
        truth=truth,
    )


# ----------------------------------------------------------------- helpers


def _empty_report(
    *,
    name: str,
    field: DisplacementField,
    truth: Float32[np.ndarray, "points 3"],
) -> ErrorReport:
    nan = float("nan")
    return ErrorReport(
        name=name,
        n_total=int(field.status.size),
        n_valid=0,
        mae=nan,
        rmse=nan,
        p95=nan,
        per_axis_mae=(nan, nan, nan),
        by_distance=None,
        confidence_vs_error=nan,
        estimated=field,
        truth=truth,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; ``nan`` if degenerate."""
    if a.size < 2:
        return float("nan")
    ar = _rankdata(a)
    br = _rankdata(b)
    a_mean = ar.mean()
    b_mean = br.mean()
    a_centered = ar - a_mean
    b_centered = br - b_mean
    denom = np.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank assignment matching ``scipy.stats.rankdata`` semantics.

    Ties get the average of the ranks they would receive. Handles the
    only inputs we care about (1D float arrays); kept inline so this
    module does not pull in ``scipy.stats``.
    """
    arr = np.asarray(a)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(arr))
    sorted_arr = arr[sorter]
    obs = np.r_[True, sorted_arr[1:] != sorted_arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(arr)]
    return 0.5 * (count[dense] + count[dense - 1] + 1).astype(np.float64)


def _stratify_by_distance(
    *,
    index: BoundaryDistanceIndex,
    positions: Float32[np.ndarray, "points 3"],
    err_norm: np.ndarray,
    err: Float32[np.ndarray, "points 3"],
    edges: tuple[float, ...],
) -> BoundaryStratifiedTable:
    poi_dist = index.query(positions)

    n_bins = len(edges)  # k edges -> k bins (k-1 right-closed + 1 terminal)
    counts = np.zeros(n_bins, dtype=np.float32)
    mae = np.full(n_bins, np.nan, dtype=np.float32)
    rmse = np.full(n_bins, np.nan, dtype=np.float32)

    for i in range(n_bins):
        if i == 0:
            # First bin: left-inclusive when edges[0] == 0 to capture
            # boundary-incident POIs.
            lo_mask = poi_dist >= edges[0] if edges[0] == 0.0 else poi_dist > edges[0]
            bin_mask = lo_mask if n_bins == 1 else lo_mask & (poi_dist <= edges[1])
        elif i == n_bins - 1:
            bin_mask = poi_dist > edges[-1]
        else:
            bin_mask = (poi_dist > edges[i]) & (poi_dist <= edges[i + 1])

        n = int(bin_mask.sum())
        counts[i] = n
        if n > 0:
            bin_err = err[bin_mask]
            mae[i] = float(np.abs(bin_err).mean())
            rmse[i] = float(np.sqrt((bin_err**2).mean()))

    return BoundaryStratifiedTable(edges=edges, counts=counts, mae=mae, rmse=rmse)


# ----------------------------------------------------------- orchestration


def run_and_evaluate(
    dataset: DvcDataset,
    deformation: str,
    *,
    correlate_fn: CorrelateFn | None = None,
    dry_shape: tuple[int, int, int] | None = None,
    distance_bins: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 50.0),
    mask: Any = None,
    **correlate_kwargs: Any,
) -> ErrorReport:
    """Materialize → correlate → evaluate, in one call.

    Parameters
    ----------
    dataset
        Open :class:`DvcDataset`.
    deformation
        Name of a synthetic deformation entry (must carry GT). Real
        entries cannot be evaluated; ``evaluate_pair`` will raise.
    correlate_fn
        Optional callable with signature ``(reference, deformed, mask)
        -> DisplacementField``. ``None`` (default) routes to
        :func:`correlate_multi_gpu` with ``**correlate_kwargs`` bound.
    dry_shape
        Optional centered-subblock size, applied identically to all
        loaded volumes.
    distance_bins
        Right-closed bin edges (open terminal) for boundary
        stratification. Pass ``()`` to skip the table.
    mask
        Forwarded to :meth:`DvcDataset.load_pair`.
    **correlate_kwargs
        Extra keyword arguments forwarded to ``correlate_multi_gpu``
        when ``correlate_fn is None``. Ignored otherwise.
    """
    pair = dataset.load_pair(deformation, mask=mask, dry_shape=dry_shape)
    fn = correlate_fn or _default_correlate_fn(correlate_kwargs)
    field = fn(pair.reference, pair.deformed, pair.mask)
    return evaluate_pair(pair, field, distance_bins=distance_bins)


def sweep(
    dataset: DvcDataset,
    *,
    names: Sequence[str] | None = None,
    correlate_fn: CorrelateFn | None = None,
    dry_shape: tuple[int, int, int] | None = None,
    distance_bins: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 50.0),
    mask: Any = None,
    **correlate_kwargs: Any,
) -> list[ErrorReport]:
    """Iterate over synthetic deformations and return one report per name.

    Parameters
    ----------
    dataset
        Open :class:`DvcDataset`.
    names
        Optional explicit list of deformation names. ``None`` uses
        ``dataset.list_synthetic()`` so only entries with GT are run.
    correlate_fn, dry_shape, distance_bins, mask, correlate_kwargs
        Forwarded to :func:`run_and_evaluate` for each entry. The
        ``correlate_fn`` is shared across the sweep — wrap a
        long-lived dispatcher in a closure to amortize spawn cost.
    """
    selected = list(names) if names is not None else dataset.list_synthetic()
    fn = correlate_fn or _default_correlate_fn(correlate_kwargs)
    out: list[ErrorReport] = []
    for name in selected:
        pair = dataset.load_pair(name, mask=mask, dry_shape=dry_shape)
        field = fn(pair.reference, pair.deformed, pair.mask)
        out.append(evaluate_pair(pair, field, distance_bins=distance_bins))
    return out


def _default_correlate_fn(correlate_kwargs: dict[str, Any]) -> CorrelateFn:
    """Bind extra kwargs into a ``CorrelateFn`` calling ``correlate_multi_gpu``."""

    def _fn(
        reference: Float32[np.ndarray, "z y x"],
        deformed: Float32[np.ndarray, "z y x"],
        mask: Bool[np.ndarray, "z y x"] | None,
    ) -> DisplacementField:
        return correlate_multi_gpu(reference, deformed, mask=mask, **correlate_kwargs)

    return _fn
