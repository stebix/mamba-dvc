# Error minimization — diagnosing the showcase accuracy gap

The DVC showcase notebook (`notebooks/dvc-showcase.ipynb`) reports per-axis MAE
in the 0.5–0.6 voxel range against a plan budget of ~0.1 voxels. This
document records the diagnostic chain that traced the gap to its root cause
and lists the concrete fixes ordered by intrusiveness.

## Plan target vs. observed

`docs/plans/overview.md` §2 budgets per-axis error at ~0.1 voxel for the v1
single-pass FFT NCC + Gaussian sub-voxel fit, calibrated against:

- band-limited noise textures (full spectral content),
- window 96 voxels (the v1 default),
- low-spatial-frequency warps (intra-window strain ≪ 1 voxel).

The showcase intentionally uses `(96, 128, 128)` volumes with `window=32` so
PyVista stays interactive — three regimes outside the calibrated envelope at
once.

## Diagnostic chain

A debug notebook (`notebooks/dvc-debug-texture.ipynb`) replaces the bone
phantom with `validate.synthetic.make_texture` and isolates each
contributor:

1. **Phantom-only effects**: cases 1–4 with band-limited noise reference.
   Phantom contribution dropped MAE from ~0.5 to ~0.13 voxels per non-zero
   axis. So roughly two thirds of the showcase error comes from the
   piecewise-constant scaffold + weak multiplicative texture and one third
   from something else.

2. **Apodization-mismatch hypothesis** (rejected). Tukey-alpha sweep
   `{0.0, 0.05, 0.25}` at `window=32`, shift `(1.5, 0, -2)`:

   | alpha | MAE_z | MAE_x |
   |---|---|---|
   | 0.0  | 0.170 | 0.159 |
   | 0.05 | 0.171 | 0.161 |
   | 0.25 | 0.127 | 0.147 |

   Bias is essentially flat — at `alpha=0` the Tukey window is rectangular,
   yet the bias remains. Apodization mismatch between reference and deformed
   crops is not the dominant cause.

3. **Window-size scaling** (confirms a window-driven bias). Sweep `{32, 48,
   64}` at `alpha=0.25`, same shift:

   | window | MAE_z | MAE_x |
   |---|---|---|
   | 32 | 0.127 | 0.147 |
   | 48 | 0.062 | 0.074 |
   | 64 | 0.038 | 0.042 |

   Doubling the window cuts bias by ~3.5×. By `window=64` both per-axis MAEs
   are inside the plan's 0.05-voxel target on band-limited noise.

4. **Warp-bypass test** (exonerates the warp). Substituted `np.roll` for the
   cubic-spline warp on three integer shifts:

   ```
   warp vs np.roll  shift=(2, -3, 1):  L_inf=0.000e+00  RMS=0.000e+00
   avg max(MAE_z, MAE_x) — np.roll : 0.1616
   avg max(MAE_z, MAE_x) — warp    : 0.1617
   ```

   `scipy.ndimage.map_coordinates(order=3, prefilter=True, mode="reflect")`
   produces bit-identical output to `np.roll` for integer shifts in the
   interior. Correlator MAE is identical to four decimals. The bias is **not
   in `warp()`** — it lives inside `correlate()`.

5. **Signed-bias direction** (identifies the bias mechanism). Across every
   case, the recovered displacement is biased *toward zero*:

   | truth | bias_z | bias_x |
   |---|---|---|
   | (2, -3, 1)  | -0.151 | -0.085 |
   | (1, 0, -2)  | -0.088 | +0.149 |
   | (3, 0, -2)  | -0.185 | +0.149 |

   Sign of `bias` is always opposite to the sign of truth. Magnitude grows
   sub-linearly with `|u|` (z=1 → 0.088, z=3 → 0.185, ratio 2.1× for 3×
   shift). Classic shrinkage estimator pattern.

## Root cause — cyclic FFT NCC + whole-window L2 normalization

`mamba_dvc/core/ncc.py:124-138` implements a **cyclic** cross-correlation
followed by a **single-constant** L2 normalization per POI:

```python
cross_power = xp.conj(spectrum_ref) * spectrum_def
corr = xp.fft.irfftn(cross_power, s=reference.shape[1:], axes=fft_axes)
...
ref_sq = (reference.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
def_sq = (deformed.astype(xp.float64) ** 2).sum(axis=(1, 2, 3))
norms = xp.sqrt(ref_sq * def_sq + eps)
corr_normalized = corr.astype(xp.float32) / norms_f32[:, None, None, None]
```

Two compounding effects:

1. **Cyclic wrap-around.** At true lag `k = u`, the cyclic sum
   `Σ_n ref[n] · def[(n+k) mod W]` pairs `W − |u|` voxels correctly and
   `|u|` voxels via the wrap. The wrap pairs uncorrelated content for
   band-limited noise — zero-mean *at lag k = u* but not symmetric in `±ε`
   around the peak. Lags between 0 and `u` collect more wrap pairings with
   nearby-shifted content; lags beyond `u` collect fewer. The peak shape is
   skewed toward `k = 0`, which the log-parabola sub-voxel fit then
   converts into a fractional-voxel bias toward zero.

2. **Full-window L2 denominator.** The denominator
   `sqrt(‖ref‖² · ‖def‖²)` is a single constant per POI — *not* the
   overlap-aware Lewis (1995) denominator. The cross-correlation peak at
   lag `u` has true-overlap numerator ≈ `(W − |u|) · ⟨ref²⟩` — peak height
   `(W − |u|)/W < 1`, observed directly in the single-POI introspection
   (peak value 0.93 ≈ 30/32). At lag 0 the numerator picks up
   `W · autocorr(u)` from the texture's autocorrelation. A constant
   denominator cannot normalize away the systematic difference, so peak
   height shrinks linearly with `|u|/W`.

Both effects vanish as `|u|/W → 0`, matching the observed window scaling.
The y-axis (truth `0`) shows essentially zero bias because no wrap occurs.

The plan's §2 claim "aliasing does not occur for `|d| ≤ w/3`" is correct in
the *spectral* sense — there is no fold-back. It does not address the
*statistical* skew described above, which is what produces the bias.

## Proposed fixes, ordered by intrusiveness

### 0. Showcase only — bump `window`

Change the showcase notebook to `window=64`. Per the table above MAE drops
to ~0.04 voxels on the texture-only reference, inside the plan budget. The
phantom adds another ~0.05 voxels on top so the showcase number lands near
0.1 — qualitatively correct without touching any production code.

This is a notebook-only patch and unblocks the demo while we land the
proper fix.

### 1. Zero-pad the FFT NCC — surgical, in-scope for v1

In `mamba_dvc/core/ncc.py`, pad each subvolume to `(2 W_z, 2 W_y, 2 W_x)`
before `rfftn`, conjugate-multiply in the padded spectrum, take `irfftn` at
the padded size, then crop the correlation map to the central
`±search_radius` band before argmax. This converts cyclic correlation into
linear correlation — wrap contributions vanish identically, and the peak
shape becomes purely a function of the texture's autocorrelation.

Cost: per-POI FFT volume grows by ~8× (3D). For the production
`window = 96` plan that means `192³ = 7.1 M` complex bins per POI, which
fits comfortably in the v1 memory budget (`docs/plans/overview.md` §6).

The L2 normalization should also switch from whole-window to
**overlap-aware**: divide each lag `k` by `sqrt(Σ_overlap ref² · Σ_overlap def²)`
where the overlap is the in-bounds intersection at lag `k`. Overlap sums
can be precomputed via integral images (one per POI) so the per-lag cost
is `O(1)`. This is the Lewis (1995) NCC formulation; it adds two more
integral-image passes but keeps everything within the v1 single-pass
budget.

These two changes together should drop the showcase MAE from ~0.13 to
**well under 0.05 voxels** at `window=32` on band-limited noise, and put
the production `window=96` setting comfortably inside the 0.1-voxel plan
target across the full `|d| ≤ W/3` displacement budget.

Estimated diff: ~30 lines in `core/ncc.py`, plus a regression test in
`tests/core/test_ncc.py` that asserts the bias-toward-zero pattern is
absent for integer shifts. Public API (`correlate`, `peak_displacement`)
unchanged.

### 2. Padfield masked FFT NCC — the v2 path

`docs/plans/overview.md` §10 already lists Padfield (2010) as the v2
upgrade for masked correlation. Padfield's six-FFT formulation computes the
exact masked cross-correlation, including overlap-aware normalization, in
a single pass. It subsumes both the zero-pad fix and the Lewis denominator
change, and additionally lets the deformed-frame mask differ from the
reference mask (already wired into the v1 API as
`deformed_mask=None`).

Cost: 6 FFTs per pair instead of 2. Drops in as a swap-in implementation
of `core.ncc.correlate` — pipeline unchanged. Worth doing for v2 when the
masking story matters; not required to hit the v1 accuracy budget.

## Plan revisions to consider

`docs/plans/overview.md` should be updated alongside fix #1:

- §2: replace "Cyclic correlation, no zero padding" with a note that v1
  zero-pads to 2W per axis and uses overlap-aware L2 normalization. The
  `|d| ≤ W/3` rule still applies but for SNR reasons, not aliasing.
- §2: document the linear scaling of bias with `|u|/W` so future readers
  know why the budget assumed a window much larger than the displacement.
- §9: add a regression test asserting MAE < 0.05 on integer-shift inputs
  via `np.roll` — this is the cheapest possible test that would have
  caught the current bias on day one.

## What did *not* explain the gap

For future reference, these were ruled out by the diagnostic notebook:

- **Sign-convention bug in `peak_displacement` or `_gather_neighbors`** —
  case 2 (all-fractional shift) hits the budget cleanly, which it could
  not if the wrap or sign were wrong.
- **Cubic-spline warp artifacts** — `warp − np.roll` interior residual is
  exactly zero; replacing the warp with `np.roll` does not change the
  correlator MAE.
- **Apodization mismatch from the Tukey window** — bias persists at
  `tukey_alpha = 0` (rectangular).
- **Peak-localization bug in `peakfit.gaussian_subvoxel_fit`** — case 2's
  sub-voxel fit is clean; the fit faithfully reports the asymmetric peak
  shape that the cyclic NCC produces. The fit is doing its job; the input
  peak is wrong.

The single-POI introspection cell in the debug notebook is the artifact
that pinned this — the asymmetric correlation neighborhood
`[0.81, 0.93, 0.86]` for an integer-truth peak is not noise, it is the
cyclic-correlation skew.
