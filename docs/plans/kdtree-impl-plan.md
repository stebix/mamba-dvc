# Boundary-distance via surface KDTree — implementation plan

Replace the full-volume `scipy.ndimage.distance_transform_edt` in the
boundary-stratified error table with a surface KDTree, and compute it
once per materialized pair instead of once per `mask_threshold` variant.
This is recommendation **R1** of
[`general-perf-improvements.md`](general-perf-improvements.md): per
scored synthetic pair the evaluate cost drops from ~460 s to ~seconds,
and the campaign evaluate phase from ~8 h to ~minutes.

Scope is *only* the boundary-distance computation. No change to the
correlation path, the I/O path, the outlier test, the JSON schema, or
the `BoundaryStratifiedTable` contract. R2–R5 in the perf doc are
separate work.

## 1. Goal and invariant

**`by_distance` stays numerically equivalent to today.** Same
`BoundaryStratifiedTable` (`edges`, `counts`, `mae`, `rmse`), same
binning semantics, same handling of out-of-range POIs. Only the
internal "distance from each POI to the mask boundary" computation
changes, and it is constructed to reproduce the EDT value exactly on the
inputs that occur.

## 2. Why a surface KDTree is exactly equivalent to the EDT here

`_stratify_by_distance` currently does `distance_transform_edt(~mask)`
and samples it at the rounded POI voxels. For a voxel `p`:

- If `p` is **foreground** (`mask[p] == True`) — the common case, POIs
  are admitted on foreground — the EDT value is the Euclidean distance
  from `p` to the nearest *background* voxel. That nearest background
  voxel always lies on the **background boundary shell**

  ```
  S = (~mask) & binary_dilation(mask, structure=ones((3,3,3), bool))
  ```

  i.e. background voxels within Chebyshev distance 1 of any foreground
  voxel: the discretized straight line from `p` to any background voxel
  `q` passes through an `S` voxel that is no farther from `p` than `q`
  is, so `min_{s∈S} ‖p−s‖ == EDT(p)`. (The 3×3×3 / 26-connected
  dilation is what makes this exact regardless of which direction the
  boundary runs; the default 6-connected structure would slightly
  overestimate near diagonal boundaries.)

- If `p` is **background** (`mask[p] == False`) — possible for a POI
  *center* when `mask_threshold < 1`, since `filter_by_mask` tests the
  *window's* mask support, not the center voxel — the EDT value is `0`
  by definition (distance to nearest background = itself). The KDTree
  would instead report the distance to the nearest boundary background
  voxel, which can be > 0. Handled with a one-line lookup:

  ```
  poi_dist[i] = 0.0 if not mask[rounded[i]] else tree.query(rounded[i], k=1)[0]
  ```

`S` is the thin shell — with foreground fraction 0.1–0.2 that is
~10⁶–10⁷ voxels — so `cKDTree(np.argwhere(S))` is a few hundred MB and
builds in ~a second; querying the few-hundred POIs is instant. The only
full-volume operation that remains is `binary_dilation` on one bool
array (a few seconds, once per pair); it can be bbox-cropped later if
ever needed (R3) but is fine as-is.

Degenerate masks (matching today's behavior loosely):
- mask all `True` → `S` empty → no tree → query returns all `inf`
  (today's `distance_transform_edt(~mask)` of an all-`False` argument
  is degenerate the same way; downstream all POIs fall in the open
  terminal bin).
- mask all `False` → every POI is background → query returns all `0.0`.
- `pair.mask is None` → unchanged: `evaluate_pair` already warns and
  sets `by_distance = None` before any of this runs.

## 3. Changes

### 3.1 `mamba_dvc/validate/known_fields.py`

**Imports.** Drop `from scipy.ndimage import distance_transform_edt`.
Add `from scipy.ndimage import binary_dilation` and `from scipy.spatial
import cKDTree`. (`cKDTree` has no scipy type stub — wrap in a tiny
typed shim or annotate the field `Any` / `# pyright: ignore`; decide
when running `pyright`.)

**New `BoundaryDistanceIndex`** — `@dataclass(frozen=True)`, added to
`__all__`:

```python
@dataclass(frozen=True)
class BoundaryDistanceIndex:
    """Amortized nearest-background-voxel distance lookup for POIs.

    Build once per mask (``from_mask``); ``query`` it for each
    parameter variant evaluated against that mask. Reproduces
    ``scipy.ndimage.distance_transform_edt(~mask)`` sampled at
    rounded POI voxels, exactly, for foreground POIs (and 0 for
    POIs whose center voxel is background, matching the EDT).
    """

    shape: tuple[int, int, int]
    mask: Bool[np.ndarray, "z y x"]   # kept for the background-voxel -> 0 case
    tree: Any                         # scipy.spatial.cKDTree | None

    @classmethod
    def from_mask(cls, mask: Bool[np.ndarray, "z y x"]) -> "BoundaryDistanceIndex":
        # validate dtype/ndim (bool, 3D) consistent with filter_by_mask's checks
        surface = (~mask) & binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool))
        coords = np.argwhere(surface)            # (n_surface, 3) int64
        tree = cKDTree(coords) if coords.size else None
        return cls(shape=tuple(mask.shape), mask=mask, tree=tree)

    def query(
        self, positions: Float32[np.ndarray, "points 3"]
    ) -> Float32[np.ndarray, "points"]:
        rounded = np.clip(
            np.round(positions).astype(np.int64),
            a_min=0,
            a_max=np.array(self.shape, dtype=np.int64) - 1,
        )                                        # same clip as today
        if self.tree is None:
            d = np.full(len(rounded), np.inf)
        else:
            d, _ = self.tree.query(rounded, k=1)
        on_fg = self.mask[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
        return np.where(on_fg, d, 0.0).astype(np.float32)
```

**`_stratify_by_distance`** — change the signature from `mask:
np.ndarray` to `index: BoundaryDistanceIndex`; delete the
`distances_volume = distance_transform_edt(...)` block and the manual
clip+index; replace with `poi_dist = index.query(positions)`. The
binning loop below is untouched.

**`evaluate_pair`** — add keyword `boundary_index: BoundaryDistanceIndex
| None = None`. In the `by_distance` branch, when `pair.mask is not
None` and `len(distance_bins) > 0`:

```python
index = boundary_index if boundary_index is not None else BoundaryDistanceIndex.from_mask(pair.mask)
by_distance = _stratify_by_distance(index=index, positions=..., err_norm=..., err=..., edges=...)
```

So standalone callers (`run_and_evaluate`, `sweep`, tests) keep working
unchanged — and are *already* faster, since even a freshly-built index
beats the scipy EDT; passing a precomputed one just removes the 4×
rebuild in the batch driver. Docstring: note `boundary_index=` mirrors
`truth=` (amortize across variants run against one materialized pair).

**`run_and_evaluate` / `sweep`** — no signature change; they call
`evaluate_pair` without the new arg.

### 3.2 `mamba_dvc/run/batch.py`

- Import `BoundaryDistanceIndex` from `mamba_dvc.validate.known_fields`
  (alongside the existing `ErrorReport, evaluate_pair`).
- In `run_batch`, in the `for load_key, load_jobs in
  _group_by(conv_jobs, _load_group_key)` body, right after `truth_cache:
  dict[...] = {}` (≈ `batch.py:356`):

  ```python
  boundary_index = (
      BoundaryDistanceIndex.from_mask(pair.mask)
      if pair.gt_field is not None and pair.mask is not None and spec.distance_bins
      else None
  )
  ```

  Built once per `load_pair`, only when it will actually be used: real
  entries have no `gt_field` → skipped; `distance_bins: []` → skipped.
- Thread `boundary_index` through `_run_variant(...)` and into the
  `evaluate_pair` call:

  ```python
  report = evaluate_pair(
      pair, field,
      distance_bins=spec.distance_bins,
      truth=truth_cache[gk],
      boundary_index=boundary_index,
  )
  ```
- The existing `del pair; gc.collect()` at the end of the loop already
  drops `boundary_index` (out of scope); the cKDTree is small.

No change to `_summarize` / `_write_sidecar` / the `by_distance`
serialization at `batch.py:658` — the `BoundaryStratifiedTable` it
receives is unchanged.

## 4. Tests

`tests/validate/test_known_fields.py` (new cases) — all CPU, no `gpu`
marker:

- **EDT equivalence.** For a handful of small synthetic masks — solid
  ball in a 32³ cube; off-center axis-aligned slab; hollow shell —
  build `BoundaryDistanceIndex.from_mask(m)` and assert
  `index.query(P)` equals `distance_transform_edt(~m)[tuple(round(P).T)]`
  (the old computation) via `np.testing.assert_allclose` for random
  *foreground* `P`. Should be exact up to float rounding.
- **Background-center POIs.** `P` that rounds onto a background voxel →
  expect `0.0` (and confirm the old EDT also gives `0.0` there).
- **Out-of-range POIs.** `P` with coords < 0 or ≥ shape → same clipped
  result as the old path.
- **Degenerate masks.** All-`True` → `tree is None`, `query` all `inf`;
  all-`False` → `query` all `0.0`.
- **`evaluate_pair` parity.** Same `pair` / `field`, evaluated with vs
  without `boundary_index=` → identical `by_distance` (`counts`, `mae`,
  `rmse` arrays element-wise, NaNs in the same slots).
- **Index reuse.** One index, two `query` calls on different position
  arrays → independent correct results (no hidden state).

`tests/run/test_batch.py`:

- Confirm nothing mocks `distance_transform_edt` (if it does, retarget
  to `BoundaryDistanceIndex` / leave it real — it's cheap on the tiny
  test masks).
- A scored synthetic run still produces a populated `by_distance` in the
  sidecar; the `boundary_index` is built once per `load_pair` (assert
  via a spy/counter on `BoundaryDistanceIndex.from_mask` that it's
  called once for a multi-variant pair, zero times for a real-only
  pair).

## 5. Gate

`uv run ruff format .` · `uv run ruff check --fix .` · `uv run pyright`
(watch the untyped `cKDTree`) · `uv run pytest -m "not gpu"`. Then a
spot re-run of one scored synthetic pair to eyeball that `wall_evaluate_s`
collapsed and `by_distance` matches a pre-change run.

## 6. Risk

Low. The new path is provably equal to the old on the inputs that occur
(foreground POIs → EDT identity above; background-center POIs → explicit
0; out-of-range → same clip). The standalone API is backward-compatible
(self-builds the index when not supplied). Blast radius is
`validate/known_fields.py` + `run/batch.py` + tests; the only other
consumer of `by_distance` is the JSON serialization in `batch.py`, which
sees an unchanged `BoundaryStratifiedTable`. Memory: cKDTree on ~10⁶–10⁷
× 3 float64 ≈ a few hundred MB, plus one transient full-volume bool for
`binary_dilation` (~1.6 GB) during `from_mask` — well within the 128 GB
host budget.

## 7. Future hooks (not now)

- Anisotropic voxel spacing: scale `coords` (and the query points) by
  the per-axis spacing before building the tree — mirrors
  `distance_transform_edt`'s `sampling=` argument. v1 is isotropic unit
  spacing.
- Bbox-cropped `binary_dilation` to drop the one remaining full-volume
  op (folds into R3's bbox work).
- If a *dense* boundary-distance field is ever wanted (visualization,
  finer stratification), that is the point to revisit the GPU-EDT
  alternative noted in the perf doc — but the sparse query here does not
  need it.
