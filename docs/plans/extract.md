# Subvolume extraction — plan & mathematical strategy

Step 2 of the single-pass DVC pipeline (`docs/plans/overview.md` §2).
Consumes the admitted POI lattice from `core/grid.py`, emits the
batched subvolume tensors that `core/window.py`, `core/ncc.py`, and
downstream stages operate on.

## 1. Position in the pipeline

```
build_grid ──► filter_by_mask ──► extract_subvolumes ──► window ──► ncc
  (lattice)      (admission)        (THIS STEP)         (Tukey)    (FFT)
```

Inputs, from upstream:

- `volume : Float32[(z, y, x)]` — the full reference, deformed, or mask
  volume (same module — NumPy or CuPy — as wherever the pipeline is
  running).
- `starts : Int64[(batch, 3)]` — admitted POI start indices, already
  filtered by `filter_by_mask` and already chunked by the caller into
  batches that fit the per-GPU memory budget.
- `window : (wz, wy, wx)` — cubic or anisotropic subvolume size.

Output:

- `Float32[(batch, wz, wy, wx)]`, C-contiguous, fresh allocation.

The module is a single verb. Callers run it three times per batch —
once for `reference`, once for `deformed`, once for `mask` — and pass
all three to the preprocessing stage.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Single verb, one tensor at a time | Keeps the module boring and reusable. Debug tooling and future v2 densification need the same gather; mask and volume should not be special-cased. |
| Caller owns batching | The per-batch budget (§6: ~0.43 GB for ref+def at batch 256, W = 96) is a pipeline-level concern. Extract is stateless and does not know the GPU budget. |
| Same array module in/out | `_array_module(volume)` decides the backend (NumPy/CuPy) via `cupy.get_array_module`; `starts` must agree. No implicit host↔device transfer — the pipeline does the one-time H2D after load. |
| Float32 C-contiguous output | cuFFT (step 4) requires it. The gather materializes through `ascontiguousarray(..., dtype=float32)` unconditionally. |
| Eager bounds check | `as_strided` performs no bounds checking and would silently return garbage. Subvolume-in-range is validated before indexing. |
| Read-only view | The rolling-window view shares storage with the source volume; writing through it would clobber the reference. Documented; tests assert that mutating the output does not mutate the source. |

## 3. Mathematical strategy: rolling-window view + fancy index

### 3.1 Notation

- Volume shape `(Z, Y, X)`; strides `(s_Z, s_Y, s_X)` in bytes (for a
  C-contiguous float32 volume, `s_X = 4`, `s_Y = 4X`, `s_Z = 4XY`).
- Window shape `(w_Z, w_Y, w_X)`.
- Batch of admitted starts `S ∈ ℤ^{B×3}`, rows `(z_b, y_b, x_b)`.
- Desired output tensor `U ∈ ℝ^{B×w_Z×w_Y×w_X}` with

  ```
  U[b, Δz, Δy, Δx] = V[z_b + Δz, y_b + Δy, x_b + Δx]
  ```

  for all `0 ≤ b < B`, `0 ≤ Δz < w_Z`, `0 ≤ Δy < w_Y`, `0 ≤ Δx < w_X`.

### 3.2 The six-axis rolling-window view

Memory is a flat buffer. NumPy/CuPy render a multidimensional array
over it through a `(shape, strides)` pair. `as_strided` lets us pick
both freely, reinterpreting the same bytes through a different index
scheme with zero copy and zero allocation.

We construct

```
view_shape   = (Z − w_Z + 1,  Y − w_Y + 1,  X − w_X + 1,   w_Z, w_Y, w_X)
view_strides = (s_Z,           s_Y,          s_X,           s_Z, s_Y, s_X)
```

so that for any `(z, y, x)` in the outer three axes and `(Δz, Δy, Δx)`
in the inner three,

```
view[z, y, x, Δz, Δy, Δx] = V[z + Δz, y + Δy, x + Δx].              (†)
```

The outer and inner stride triples are identical because, in the
underlying buffer, advancing one step along "window start" traverses
exactly one voxel — the same thing advancing one step along "position
inside the window" does. The two groups of axes are distinguished only
by how we index them, not by how they're laid out in memory.

**1D sanity check.** Buffer `[a b c d e f]`, `stride = 1`, window 3:

```
shape   = (4, 3)
strides = (1, 1)
```

yields the rolling view

```
view[0] = [a b c]    view[1] = [b c d]    view[2] = [c d e]    view[3] = [d e f]
```

Element `b` appears in three "rows" — they are distinct index paths
into the same bytes. The 3D case generalizes one axis at a time.

### 3.3 Fancy indexing the outer three axes

Given the view with property (†), the whole batch gather is

```
U = view[S[:, 0], S[:, 1], S[:, 2]]                                 (‡)
```

which is standard NumPy/CuPy integer-array (fancy) indexing on the
first three axes, broadcasting trivially onto the preserved inner
three axes. The library materializes exactly one result tensor of
shape `(B, w_Z, w_Y, w_X)` in a single kernel launch on CuPy — no
per-axis coordinate tensors are ever built.

Contrast the alternatives:

| Approach | Launches | Extra allocation | Notes |
|---|---|---|---|
| Python `for b in range(B): out[b] = V[z_b:z_b+w_Z, ...]` | `O(B)` | 0 | Fine correctness; too many launches on GPU. |
| Build `(B, w_Z, w_Y, w_X, 3)` coordinate tensor; gather | 1 | ~3× output size as int64 indices | Works, but extra ~0.7 GB at batch 256 / W = 96. |
| **Rolling view + fancy index on outer axes (ours)** | **1** | **0** | Uses strides to encode the inner-axis coordinates implicitly. |

### 3.4 Complexity

Let `N = Z·Y·X`, `B = |starts|`, `W = w_Z·w_Y·w_X`.

- View construction: `O(1)` (no data touched; only a new descriptor).
- Gather kernel: reads and writes `Θ(B·W)` float32 elements. Memory
  traffic is the dominant cost, and it's optimal — the output tensor
  genuinely has `B·W` elements and each must be written once.
- No `O(N)` full-volume pass anywhere in the extract step.

### 3.5 Memory safety

The view shares storage with the source. Two consequences:

1. **Never write through the view.** We index it and pass the indexed
   result through `ascontiguousarray(...)`, which for a CuPy fancy-
   index result is already a fresh copy; the view itself never leaves
   the function. Callers receive a standalone array.
2. **`as_strided` trusts its inputs.** A malformed `shape` or `strides`
   can read beyond the buffer and return garbage or segfault. We
   bracket the strided view with an eager bounds check:

   ```
   0 ≤ min(S[:, i])    and    max(S[:, i]) + w_i  ≤  dim_i
   ```

   for `i ∈ {0, 1, 2}`. `build_grid` already guarantees this for the
   lattice it produces; the extract module re-validates because it is
   a public entry point.

## 4. Interaction with the broader pipeline

### 4.1 Why the output lives on the same device as the input

Plan §6 puts the entire reference, deformed, and mask volume on each
GPU (replicated). Extract therefore runs **on device**. Duck-typed
support for NumPy exists only to let CPU-only machines run the same
correctness tests that CI runs, not because a CPU pipeline is a
production target.

### 4.2 Why extract is separated from windowing

`core/window.py` (step 3) performs masked mean-subtract, Tukey
multiplication, and masked zeroing. Those operations need:

- the ref/deformed subvolumes (to compute and subtract their masked
  mean),
- the mask subvolumes (to define "valid voxels"),
- axis-aligned 1D Tukey windows.

All of that assumes the `(batch, w_z, w_y, w_x)` layout is already in
hand. Keeping extract separate means:

- Window is a pure tensor operation, not an indexing op.
- Extract is reusable for paths that want raw subvolumes (e.g., debug
  cut-outs, validation reporting, potential v2 densification).
- The test surface for each is small.

### 4.3 Relationship to the FFT NCC inputs

Extract emits `(B, w_Z, w_Y, w_X)` float32. cuFFT's batched R2C plan
operates on exactly this layout, with the inner three axes as the
transform axes and `B` as the batch dimension. No reshape, no axis
permutation is needed between `window` and `ncc` — the layout carries
through the pipeline unchanged until the peak localization step.

## 5. Shape-vocabulary compliance

Per `CLAUDE.md`, annotations use lowercase shape symbols:

- `z`, `y`, `x` — full-volume axes,
- `batch` — the leading axis introduced by extraction,
- `w` — per-window extent symbol for annotated outputs.

The module's signatures are:

```python
def extract_subvolumes(
    volume: Float32[ndarray, "z y x"],
    starts: Int64[ndarray, "batch 3"],
    window: tuple[int, int, int],
) -> Float32[ndarray, "batch w w w"]: ...
```

Consistent with `GridSpec` (`positions: Float32[ndarray, "N 3"]` —
noting that the `types.py` vocabulary predates the lowercase
convention and will be migrated when the next round of types is added;
until then, do not mix the two in a single signature).

## 6. Validation strategy (tests that must pass)

Mirrors `tests/core/test_grid.py`. Full list in
`tests/core/test_extract.py`. Key invariants asserted:

- **Correctness vs. naive reference.** A Python-loop implementation
  sees the same random volume and starts, element-wise equality is
  required.
- **Shape, dtype, C-contiguity of the output.**
- **Anisotropic window correctness** — the three axes are not
  accidentally transposed.
- **Empty-batch shortcut** — `starts.shape[0] == 0` returns an
  `(0, w_Z, w_Y, w_X)` array without indexing the view.
- **Output does not alias source** — mutating `U` leaves `V`
  untouched.
- **Integration with `build_grid`** — feeding the admitted starts of a
  real grid yields `(N, W, W, W)` subvolumes at the right positions.
- **Runtime shape check** via `@jaxtyped(typechecker=beartype)`.
- **Exhaustive input rejection** — non-3D volume, non-float32 volume,
  non-int64 starts, wrong `starts.shape[-1]`, non-positive window,
  oversize window, out-of-bounds start.
- **GPU parity** (`@pytest.mark.gpu`) — CuPy inputs yield the same
  values as NumPy inputs; mixed host/device inputs are rejected with a
  ValueError mentioning "same array module".

## 7. Future-proofing (v2 hooks)

No explicit hooks in the v1 API — the function is small enough that
any v2 feature layers on top rather than into it.

| v2 need | How extract stays relevant |
|---|---|
| IC-GN refinement | Consumes the same `(B, W, W, W)` layout. |
| Padfield masked FFT | Same. The masked-NCC change is inside `core/ncc.py`. |
| Iterative warp loop | Outer driver calls extract each iteration with updated `starts` (shifted by the current displacement estimate). Nothing in this module needs to change. |
| Multi-level pyramid | Extract is reused at each level; the volume and window change, the gather does not. |
| Dense field densification | `GridSpec.grid_shape + spacing` handles the reshape; extract plays no role there. |
| Per-frame deformed masks | Extract is called a fourth time per batch (for `deformed_mask`). Trivial. |

## 8. Known non-features (by design)

- No prefetch or async H2D — that is a pipeline / `gpu/memory.py`
  concern.
- No CUDA-graph capture — extract is not in the hot loop often enough
  at v1 scales to justify it. One gather per batch for 11 875 POIs
  across 4 GPUs is ~50 calls per pair; negligible overhead.
- No implicit dtype promotion — callers are expected to store volumes
  as float32 on device (plan §6). A `float64` volume is a bug, not a
  case to handle.

## 9. Open questions (non-blocking)

- **Should we expose a `RawKernel` fallback?** Profiling will tell.
  The stride-trick gather is one kernel launch per batch; if that
  shows up as a bottleneck behind the ~tens-of-ms FFTs, we write a
  2-line custom gather kernel. Not needed for v1.
- **Should extract accept a `dest` buffer for zero-allocation runs?**
  Tempting for the hot path, but the output is freshly allocated by
  the rolling-window gather anyway. Revisit only if `gpu/memory.py`
  shows a measurable benefit from pool reuse.
