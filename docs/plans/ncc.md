# Batched FFT NCC — plan & mathematical strategy

Steps 4–5 of the single-pass DVC pipeline (`docs/plans/overview.md` §2).
Consumes the masked, windowed, zero-mean subvolume tensors emitted by
`core/window.py`, returns the batched cross-correlation map and the
integer-lag peak that `core/peakfit.py` refines to subvoxel accuracy.

## 1. Position in the pipeline

```
build_grid ──► extract ──► preprocess ──► ncc ──► peakfit ──► outlier
                                          (THIS STEP)
```

Inputs, from upstream:

- `reference : Float32[(batch, w, w, w)]` — preprocessed reference
  subvolumes (`window.preprocess_subvolumes`). Already mean-subtracted
  over valid voxels, Tukey-windowed, masked voxels zeroed.
- `deformed  : Float32[(batch, w, w, w)]` — same, for the deformed
  volume.
- (Optional) the matching mask subvolumes are not consumed here; their
  effect has already been applied during preprocessing.

Outputs:

- `displacements : Float32[(batch, 3)]` — integer-voxel lag of the
  correlation peak per POI, wrapped to `[-w/2, w/2)` along each axis,
  in `(dz, dy, dx)` order matching the rest of the codebase.
- `correlation  : Float32[(batch, w, w, w)]` — the full normalized
  cross-correlation volume per POI, returned so `core/peakfit.py` can
  do its 3×3×3 neighborhood fit without recomputation.
- `peak_value   : Float32[(batch,)]` — the NCC value at the integer
  peak. Becomes the `confidence` channel of the final
  `DisplacementField`.

The module is two verbs:

- `correlate(reference, deformed) -> Float32[(batch, w, w, w)]` — the
  pure NCC kernel.
- `peak_displacement(correlation) -> (displacements, peak_value)` —
  argmax + circular wrap.

Splitting them keeps `correlate` reusable for v2 (Padfield masked NCC
will swap *only* the kernel) and lets tests assert correlation-volume
properties independently of peak detection.

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| Single batched FFT pair per call | One forward `rfftn(ref)`, one forward `rfftn(def)`, one inverse `irfftn`. Per-batch overhead is amortized; cuFFT plans cache by shape. |
| Real-to-complex transforms (R2C) | Inputs are real float32. cuFFT's R2C halves the spectral memory along the last axis (`X//2 + 1`) and roughly halves transform time. |
| Output is the **normalized** cross-correlation, not the raw cross-power | Plan §2 step 4: "normalize by the per-subvolume L2 norms". A raw cross-correlation peak's magnitude is dominated by texture energy and useless as a confidence score. |
| Normalization is over the **full window**, not the valid-mask support | After preprocessing, masked voxels are exactly zero, so they contribute nothing to the L2 norm regardless. Special-casing the mask here would duplicate work and introduce a v1/v2 fork point unnecessarily. The principled valid-support correction is Padfield (v2). |
| Same array module in/out | NumPy or CuPy; decided by `cupy.get_array_module`. Reference and deformed must agree. No implicit transfer. |
| Cyclic correlation, lag wrapped to `[-w/2, w/2)` | The plan caps `\|d\| ≤ w/3`, so circular wrapping is unambiguous. Same convention as `scipy.signal.correlate` with `mode="same"` on a periodic signal. |
| No zero-padding | Padding to `2w` would defeat the cyclic-correlation memory savings without buying anything: at our displacement budget, aliasing does not occur. |

## 3. Mathematical strategy

### 3.1 Notation

- Subvolume window shape `(w, w, w)`, batch size `B`.
- `f_b, g_b ∈ ℝ^{w³}` — preprocessed reference and deformed subvolumes
  for POI `b`. Each has zero mean over its valid support and is
  zero-padded over the masked-out region.
- `F_b = rfftn(f_b)`, `G_b = rfftn(g_b)` — batched forward R2C
  transforms.
- `‖f_b‖₂² = Σ f_b[i]²` — the L2 norm squared, summed across the full
  window. Because preprocessing zeroed the invalid voxels, this equals
  the sum over the valid support automatically.

### 3.2 The cross-correlation theorem (cyclic form)

For any two real signals,

```
(f ⋆ g)[n] = Σ_m f[m + n] · g[m]   (cross-correlation, cyclic in n)
           = irfftn( conj(F) · G )                                      (*)
```

The conjugate goes on `F` because cross-correlation flips one signal's
index; the FFT of the flipped signal is the conjugate of the FFT of the
unflipped signal. We adopt the convention that the lag `n` represents
"deformed minus reference" — equivalently, "by how much the deformed
sample is shifted relative to the reference". With `f = reference` and
`g = deformed`, the displacement at peak corresponds to the deformed
volume's local translation in voxel coordinates.

Applied batchwise with R2C cuFFT, (*) is one kernel per stack:

```
F_b = rfftn(reference[b]),   G_b = rfftn(deformed[b])     (forward, batched)
H_b = conj(F_b) * G_b                                     (elementwise, batched)
C_b = irfftn(H_b)                                         (inverse, batched)
```

`C_b` has shape `(w, w, w)` and is real.

### 3.3 Normalization

We use the **whole-window L2 normalization**:

```
NCC_b[n] = C_b[n] / sqrt(‖f_b‖₂² · ‖g_b‖₂² + ε)                       (**)
```

with a small `ε = 1e-12` floor that defends against the all-zero
subvolume case (entirely masked POI, which `filter_by_mask` should have
rejected upstream — `ε` is the safety net, not the contract).

Why whole-window L2 and not the textbook "sliding-window NCC" (which
recomputes the local mean and variance at every lag):

1. After preprocessing, both subvolumes are *globally* zero-mean over
   their valid support. The local-mean correction at each lag is
   already absorbed into the preprocessing step.
2. The local-variance correction is what Padfield's masked FFT (v2)
   exists to do. Doing it correctly under a mask requires 6 FFTs per
   batch (Padfield 2010 Eq. 4–5); we are explicitly deferring that.
3. At our displacement target (`|d| ≤ w/3`, overlap fraction ≥ 0.5),
   the overlap-fraction error in the whole-window L2 normalization is
   ≤ 50% — large in absolute terms, but a *constant scaling* of the
   correlation peak across all candidate lags within a single POI. It
   shifts confidence values, not peak locations. The Gaussian subvoxel
   fit (`peakfit.py`) is invariant to a global multiplicative scale on
   the 3×3×3 neighborhood (the fit is in log-space, and a scale becomes
   an additive constant that drops out of the maximum). So the integer
   peak and the subvoxel refinement are both unaffected.

The bias **does** affect cross-POI confidence comparisons. We document
this in the docstring and note it as a v2 concern.

### 3.4 Integer peak localization

```
flat_peak  = argmax(C_b.reshape(B, w³), axis=-1)
peak_value = max(C_b.reshape(B, w³), axis=-1)
peak_idx   = unravel_index(flat_peak, (w, w, w))
displacement = wrap_to_signed(peak_idx, w)
```

with

```
wrap_to_signed(i, w) = i           if i < w/2
                     = i - w       otherwise
```

For even `w` (the only case used in v1 — defaults are `w = 96` and
`w = 128`), this is exact. The returned displacement has signed integer
values in `[-w/2, w/2)`; `core/peakfit.py` adds a fractional refinement
with absolute value `< 1`.

### 3.5 Complexity and memory

Let `B` be the batch size and `w` the (cubic) window size.

| Step | FLOPs | Memory (peak, complex64) |
|---|---|---|
| `rfftn(reference)` | `O(B · w³ · log w)` | `B · w² · (w/2 + 1) · 8 B` |
| `rfftn(deformed)`  | `O(B · w³ · log w)` | same, in addition |
| `conj * mul`       | `O(B · w² · (w/2+1))` | reuses one of the above |
| `irfftn`           | `O(B · w³ · log w)` | `B · w³ · 4 B` real out |

At `B = 256`, `w = 96`: each spectral tensor is 418 MB; in-flight
working set ≈ 1.7 GB. Plan §6 reserves ~17 GB total per GPU including
volume residency; this fits comfortably.

cuFFT plan caching: CuPy's `cupyx.scipy.fft` interface caches plans by
`(shape, dtype, axes, batched)`. Once warmed by the first batch, every
later batch of the same shape pays only the kernel-launch cost.

## 4. Interaction with the broader pipeline

### 4.1 Why correlation is returned, not just the peak

`core/peakfit.py` needs the 3×3×3 neighborhood around the integer peak
to fit a separable log-parabola. Returning the full correlation volume
(rather than gathering 27 voxels in `ncc.py`) keeps the FFT module a
pure tensor op and lets `peakfit.py` decide on the neighborhood
extraction policy (with toroidal wrapping at face boundaries — see the
peakfit plan).

The memory cost is `B · w³ · 4 B` per batch — already paid by the
inverse FFT's output buffer; we are not allocating extra.

### 4.2 Why we don't fold the mask in here

The plan keeps masking strictly inside `core/window.py`:

- "Mean-subtract over unmasked voxels only", then "Zero out masked
  voxels after windowing" (plan §2 step 3).

By the time tensors reach `core/ncc.py`, both subvolumes have already
encoded the mask through their numerical content. Routing the mask
into the FFT kernel would either:

1. Duplicate the zeroing (no-op in v1; harmless but pointless), or
2. Trigger the Padfield 6-FFT branch (v2), which we are not building
   yet.

Keeping the boundary clean means swapping in Padfield is a one-file
change to `core/ncc.py` with no upstream churn.

### 4.3 Relationship to peakfit

`peakfit.py` will accept `correlation: Float32[(batch, w, w, w)]` and
the integer peak indices computed here. The returned subvoxel
displacement is **added** to this module's integer displacement,
producing the final per-POI `(dz, dy, dx)` in voxel units.

Rationale for the split: the integer peak is needed not only as a
starting guess but also for sanity bounds (caller can reject any POI
with `|integer peak| > search_radius` before paying the fit cost).

## 5. API

```python
# mamba_dvc/core/ncc.py

def correlate(
    reference: Float32[ndarray, "batch w w w"],
    deformed:  Float32[ndarray, "batch w w w"],
    *,
    eps: float = 1e-12,
) -> Float32[ndarray, "batch w w w"]: ...


def peak_displacement(
    correlation: Float32[ndarray, "batch w w w"],
) -> tuple[
    Int64[ndarray, "batch 3"],   # integer displacement, signed
    Float32[ndarray, "batch"],   # peak NCC value
]: ...
```

Both functions are pure, duck-typed across NumPy and CuPy. `correlate`
allocates one fresh `(B, w, w, w)` real-valued output. The intermediate
spectral tensors are the FFT library's, not exposed.

## 6. Validation strategy (tests that must pass)

Mirrors `tests/core/test_window.py`. Full list in
`tests/core/test_ncc.py`.

- **Self-correlation peak at zero.** `correlate(f, f)` has its argmax
  at `(0, 0, 0)` and `peak_value == 1` (within float32 tolerance) for
  any non-trivial `f`.
- **Integer-shift recovery.** For `f` random and `g = circshift(f, s)`
  with `s ∈ {(1,0,0), (-2, 5, -3), (10, -7, 12), …}`, `peak_displacement`
  must recover exactly `s` (after wrap to `[-w/2, w/2)`).
- **Linearity check.** Scaling either input by a positive constant
  leaves the peak location unchanged and rescales the peak value by
  exactly that constant (within tolerance). Confirms the L2
  normalization holds.
- **Zero input degeneracy.** All-zero `reference` does not produce
  NaN; `peak_value` is finite (zero, modulo `eps`).
- **Output shape, dtype, contiguity.** `(B, w, w, w)` float32 C-contig.
- **Anisotropic windows** are not used in v1 — assert that the kernel
  handles non-cubic `(wz, wy, wx)` correctly anyway, since the
  signature does not forbid it and v2 pyramid levels will exercise it.
- **`peak_displacement` wrap convention.** A correlation volume with a
  peak at integer index `(w-1, 0, 0)` returns `(-1, 0, 0)`, not
  `(w-1, 0, 0)`.
- **GPU parity** (`@pytest.mark.gpu`) — CuPy inputs match NumPy outputs
  bit-for-bit-ish (atol ~1e-4 for the FFT accumulation difference).
- **Round-trip integration** (`@pytest.mark.slow`) — combine with
  `validate.synthetic.rigid_shift` and assert integer recovery on a
  small textured volume.

## 7. Future-proofing (v2 hooks)

| v2 need | How `ncc.py` stays relevant |
|---|---|
| Padfield masked FFT | Replace the body of `correlate()` with the 6-FFT formulation. The `(reference, deformed) → corr_map` contract is unchanged; callers keep working. The `mask` argument re-enters via the function signature; `core/window.py`'s zeroing becomes a no-op for the Padfield path. |
| IC-GN refinement | `peakfit.py` swaps; `ncc.py` is not touched. |
| Multi-level pyramid | `correlate` is reused at each level with smaller `w`; cuFFT plan cache amortizes across levels. |
| Iterative warp loop | The outer driver re-runs `correlate` after each warp update. No state lives inside the module. |

## 8. Known non-features (by design)

- **No phase correlation variant.** Phase-only is more robust under
  illumination changes but trades SNR for that robustness; our µCT
  intensity is stable, and Bar-Kochba 2014 specifically argued for
  amplitude-weighted normalized correlation.
- **No GPU plan-cache management.** CuPy handles it. Profiling will
  reveal if we need to pin a plan by hand.
- **No subvoxel fit here.** `peakfit.py` is a separate module (plan
  §4); coupling them would prevent the IC-GN swap.
- **No outlier handling.** That is `core/outlier.py` (plan §2 step 7),
  which runs after the full grid is assembled.

## 9. Open questions (non-blocking)

- **Should `peak_displacement` accept a `search_radius` clamp?**
  Currently it returns whatever the global argmax is. Adding a clamp
  here would couple it to the pipeline's `search_radius` parameter
  (plan §5). Cleaner to enforce that downstream — failed POIs become
  outliers, not silent zeros.
- **Should `correlate` accept a precomputed `rfftn` of either input?**
  Time-series workflows in v2 will reuse the reference's spectrum
  across many deformed volumes. Re-shape this only when the time
  series driver materializes; do not pre-engineer it now.
