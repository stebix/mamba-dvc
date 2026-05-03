# v1 Backcheck: Core Algorithm vs. `docs/theory/fft-dvc.md`

Cross-validation of the landed v1 implementation (`core/ncc.py`,
`core/peakfit.py`, `core/window.py`, `pipeline/correlate.py`,
`pipeline/_internal.py`, `core/outlier.py`) against the canonical
DIC/DVC literature summarized in `docs/theory/fft-dvc.md`.

The first two sections audit what is mathematically correct and what
diverges. The third section enumerates **low-hanging fruit** — each
item carries a checkbox so the buildout pipeline can pick them up
incrementally without re-deriving the rationale.

---

## 1. What is mathematically correct

| Theory claim | Implementation site | Status |
|---|---|---|
| Linear CC via zero-padding to `M ≥ 2N − 1` (§3.2, §5(3c), §9 table) | `correlate_linear` pads to exactly `2W` per axis, FFTs in padded domain, crops central `W` block | OK |
| Lewis 1995 per-lag overlap-aware denominator (§2.3) | `normalization="overlap"`: FFT of `ref²` and `def²` correlated against an indicator box, yielding `√(S_ref(s) · S_def(s))` per lag | OK (structurally) |
| Implicit triangle-bias cancellation via Lewis denominator (§3.2 last ¶, §5(3e)) | Falls out of overlap-aware normalization automatically; explicitly noted in `ncc.py` module docstring | OK |
| Real-FFT for memory/runtime halving (§2.2 last sentence) | `xp.fft.rfftn` / `irfftn` throughout both kernels | OK |
| Sign convention `lag = deformed − reference` (§1: `g(x + u) ≈ f(x)`) | `conj(rfftn(ref)) · rfftn(def)` peaks at `s = u` | OK |
| Mean subtraction before windowing (DC removal, §5(4b)) | `preprocess_subvolumes`: masked mean → window → re-zero. Order documented | OK |
| 3-point Gaussian sub-voxel (§4.1) | `_log_parabola_offset` is the closed-form log-parabolic fit (= Gaussian fit); plain-parabolic fallback when any sample ≤ 0 | OK (exact closed form) |
| Per-axis separable fit valid for separable Gaussian peaks (§4.1) | `gaussian_subvoxel_fit` does three independent 1D fits | OK (exact for the model) |
| Sub-voxel clamp `[-0.5, 0.5]`; defer overflow to outlier filter | `_log_parabola_offset` final `clip` | OK (matches §6 philosophy) |
| Universal Median Test, ε = 2.0 (§5(6), §7, §9 table) | `detect_outliers` uses Westerweel & Scarano 2005 with default ε = 2.0 | OK (canonical) |
| Tukey α = 0.25 reserved as opt-in only (§3.3, §3.7) | `TUKEY_DEFAULTS = {"linear": 0.0, "cyclic": 0.25}` — defaults to no taper in linear mode | OK (matches §3.6: "almost universally no apodization") |

---

## 2. Mathematical divergences worth knowing about

These are not bugs — they are explicit simplifications baked into v1.
They are documented here so future contributors do not "fix" them
without thinking.

### 2.1 The current kernel is Lewis NCC, not strictly Lewis ZNCC

The numerator subtracts a single per-subset mean during preprocessing
rather than the per-lag rolling mean `ḡ_p(s)`. The denominator does
compute `Σg²` over the overlap (via the indicator-box correlation) but
does **not** compute `Σg` over the overlap, so the per-lag mean
correction `−|Ω(s)| · f̄_p · ḡ_p(s)` from §2.3 is missing.

- Effect is small when `ḡ_p(s) ≈ ḡ_p` (well-conditioned data).
- Becomes load-bearing when `g`'s local mean varies meaningfully
  across the overlap region (e.g. POIs straddling a strong gradient).
- See low-hanging-fruit item **LHF-4** for the upgrade path.

### 2.2 §3.5 caveat is latent, not active

Theory warns that subset windowing breaks the upsampled-DFT sub-voxel
refinement — the upsampled DFT recovers the peak of (windowed-CC), not
(CC). With `tukey_alpha = 0` in linear mode this never fires, but if a
future caller sets `tukey_alpha > 0` *and* v2 swaps the Gaussian fit
for Guizar-Sicairos, the bias warning kicks in.

- Worth a docstring note in `peakfit.py` (and later in the
  Guizar-Sicairos module) reminding the integrator to either re-extract
  unwindowed subsets for refinement or stay on `tukey_alpha = 0`.

### 2.3 Padding to exactly `2W` is not necessarily a smooth FFT size

§3.2 / §5(3c) recommend `M ≥ 2N − 1` rounded up to the next 2-3-5
composite. `2 · 96 = 192 = 2⁶ · 3` happens to be smooth, but a generic
`next_fast_len`-style helper would future-proof the kernel for
non-power-of-2 windows. See **LHF-5**.

---

## 3. Low-hanging fruit (checkable)

Pull these into v1 in order of value/effort. Each item is local,
additive, and does not require restructuring the pipeline.

### LHF-1 — Pan 2013 Gaussian pre-filter on full volumes

- [ ] Add `prefilter_sigma: float | None = 0.7` to `pipeline.correlate.correlate()`.
- [ ] Apply isotropic Gaussian (`scipy.ndimage.gaussian_filter` on host,
      `cupyx.scipy.ndimage.gaussian_filter` on device) to both `reference`
      and `deformed` once before subset extraction.
- [ ] Document σ ≈ 0.5–1 voxel default, citing Pan 2013 (`OLE` 51, 1161–1167).
- [ ] Note that the filter is safe vs. §3.5 (acts on full volumes, not subsets).
- [ ] Add a unit test that a virtual sub-voxel-shifted pair shows
      reduced systematic bias with the filter on (compare to
      `validate/synthetic.py` baseline).

**Why:** Single most-cited bias reduction in the literature (30–50 %
on interpolation bias for sub-voxel). Cheap, safe, complementary to
zero-padding.

**Refs:** §3.7, §5(1), §7 (third bullet).

---

### LHF-2 — Secondary-peak ratio per POI

- [ ] Compute `R = C* / C^(2)` per POI from the existing correlation map
      (mask out a small neighborhood around the integer peak, take the
      next argmax).
- [ ] Surface as a new field on `DisplacementField` (e.g.
      `peak_ratio: Float32[..., "points"]`) or fold into a richer
      `quality` struct.
- [ ] Add to `peak_displacement` or as a sibling helper in `core/ncc.py`.
- [ ] Threshold guidance in docstring: `R ≳ 1.5` unique, `R → 1`
      ambiguous (Charonko & Vlachos 2013).
- [ ] Optional: feed into outlier rejection as a secondary criterion
      (UMT alone catches lattice-anomalous POIs but not locally
      ambiguous ones).

**Why:** Cheap to compute from data already on device; UMT alone does
not catch locally ambiguous peaks.

**Refs:** §6.1 (second bullet), Charonko & Vlachos 2013 (`MST` 24, 065301).

---

### LHF-3 — SSSIG / MIG per POI (Cramér-Rao a-priori uncertainty)

- [ ] Compute `SSSIG(x_p) = Σ ‖∇f(x_p + ξ)‖²` over the reference subset
      (one `xp.gradient` + sum, on the already-extracted, pre-window
      reference subvolume).
- [ ] Surface `sssig: Float32[..., "points"]` on `DisplacementField`.
- [ ] Compute and report MIG = `(1/|Ω|) · Σ ‖∇f‖` as a global summary
      in validation runs.
- [ ] Document the conversion `σ_u² = 2·σ_n² / SSSIG` so callers can
      back out a per-POI displacement noise floor given an estimate
      of `σ_n`.
- [ ] Wire into validation reports (§6.3 baselines all quote it).

**Why:** Most rigorous per-POI a-priori uncertainty available, costs
one gradient pass per subset, immediately useful for picking subset
size.

**Refs:** §6.1 (third bullet), Pan/Lu/Xie 2010 (`OLE` 48, 469–477).

---

### LHF-4 — Canonical Lewis ZNCC numerator/denominator

- [ ] Add one extra box-correlation per pair to compute `Σg` (and
      symmetrically `Σf`) over the per-lag overlap.
- [ ] Subtract `|Ω(s)| · f̄(s) · ḡ(s)` from the numerator.
- [ ] Use `√((S_g²(s) − |Ω(s)| · ḡ(s)²) · (S_f²(s) − |Ω(s)| · f̄(s)²))`
      for the denominator variance.
- [ ] Bump from 5 FFTs to 7 FFTs per pair (one extra box-correlation
      each side); no other restructuring.
- [ ] Gate behind `ncc_normalization="zncc"` so existing `"overlap"`
      callers remain bit-stable.
- [ ] Add a regression test on a virtually-shifted pair with a
      strong gradient where the per-lag mean differs from the
      whole-subset mean.

**Why:** Closes the divergence in §2.1 — turns the kernel into the
canonical Lewis ZNCC. Theoretically cleaner; matters when local mean
varies across the overlap.

**Refs:** §2.3 (Lewis 1995), §3.2 last paragraph, §9 table.

---

### LHF-5 — Smooth-FFT-size padding helper

- [ ] Replace literal `2 * W` in `correlate_linear` with
      `next_fast_len(2 * W - 1)` (use `scipy.fft.next_fast_len` /
      `cupyx.scipy.fft.next_fast_len` or roll a small 2-3-5 sieve).
- [ ] Update the cropping helper to use the resolved padded shape.
- [ ] Validate that throughput improves on representative
      non-power-of-2 windows (e.g. `W = 80`, `W = 100`).

**Why:** Pure performance win on non-power-of-2 windows; zero
algorithmic risk.

**Refs:** §3.2, §5(3c).

---

### LHF-6 — Peak FWHM into the output

- [ ] Reuse the per-axis `log_den = −2/σ²` already computed in
      `_log_parabola_offset` to derive per-axis Gaussian σ.
- [ ] Convert to per-axis FWHM = `2 · sqrt(2 · ln 2) · σ`.
- [ ] Surface as `peak_fwhm: Float32[..., "points 3"]` on
      `DisplacementField`.
- [ ] Document threshold guidance: a broadened FWHM diagnoses blur or
      de-correlation (e.g. local rotation that the translation-only
      kernel cannot match).

**Why:** Essentially free — the quantity is a by-product of the
existing Gaussian sub-voxel fit. Adds a third per-POI quality channel
(alongside `confidence` and the proposed `peak_ratio`).

**Refs:** §6.1 (fourth bullet).

---

### LHF-7 — Docstring note: §3.5 incompatibility with future Guizar-Sicairos

- [ ] Add a "Caveats for v2" paragraph to `peakfit.py` flagging that
      a Guizar-Sicairos upsampled-DFT replacement requires either
      `tukey_alpha = 0` or a re-extracted unwindowed subset for the
      refinement stage.
- [ ] Add the same note to the v2 sub-voxel module when it lands.

**Why:** Latent bug-trap. Cheap to inoculate now; expensive to debug
once the upsampled-DFT lands.

**Refs:** §3.5, §4.2 closing paragraph.

---

## 4. Explicitly out of scope for v1

These are correctly listed as v2 work in `overview.md` §10 and are
**not** low-hanging fruit — they are real engineering, but the seams
exist for a clean later swap.

| Theory feature | Plan §10 hook | Why deferred |
|---|---|---|
| Guizar-Sicairos upsampled DFT (§4.2) | Replaces `peakfit` | Biggest sub-voxel accuracy win (0.05 → 0.005–0.01 vox) but a real new module |
| IC-GN refinement (§4.3) | Layered after `peakfit` | Recovers shape-function parameters; non-trivial |
| Iterative deformation (FIDVC, §5(7)) | Wraps `correlate()` in a fixed-point loop | Needs warp + interpolation infrastructure |
| Multi-scale pyramid (§7) | Pre-pass before `correlate()` | Needs downsampling infrastructure |
| Padfield masked FFT (§7, refs Padfield 2011) | Replaces `correlate_linear` | Real algorithmic redesign |
| Mechanical regularization / global DVC (§7) | Out of `mamba_dvc` scope entirely | Different paradigm |

---

## 5. Pickup order recommendation

If a single PR has budget for two items, pick **LHF-1** (Pan
pre-filter) and **LHF-3** (SSSIG). They are independent, both
local-additive, and together give the biggest jump in v1's measurable
quality: lower interpolation bias plus a defensible per-POI noise
floor for the validation report.

**LHF-2** and **LHF-6** together form a "richer quality channel" PR
once `DisplacementField` gets extended.

**LHF-4** is the right time to revisit the NCC kernel if a
zero-strain test on real data shows the per-lag mean assumption
biting.

**LHF-5** and **LHF-7** are housekeeping; bundle into the next
unrelated touch of the respective files.
