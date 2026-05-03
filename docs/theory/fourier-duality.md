# Fourier duality cheat sheet

A pedagogic reference for the conjugate pairs that show up throughout the
DVC pipeline — translation/phase, convolution/multiplication, and the
specialisations to magnitude vs. phase. Written to build intuition, not
to be exhaustive.

The Fourier transform has a beautiful symmetry: operations in one domain
map to *paired* operations in the other. Once you internalise a handful
of these "conjugate pairs," a lot of FFT-based image processing stops
feeling like magic.

Notation: 1D for clarity (everything generalises to 3D componentwise).
Write $f(x) \leftrightarrow F(\xi)$ for "$f$ has Fourier transform $F$."

---

## The core pairs

### 1. Translation ↔ linear phase ramp

$$f(x - x_0) \;\;\longleftrightarrow\;\; e^{-2\pi i\,\xi x_0}\,F(\xi)$$

Shifting the signal by $x_0$ in space multiplies the spectrum by a
**complex exponential** whose phase grows linearly with frequency $\xi$.

- **Magnitude:** $|e^{-2\pi i\xi x_0}\,F(\xi)| = |F(\xi)|$.
  **Translation does not change the magnitude spectrum at all.** This is
  why $|F|^2$ (the power spectrum) is *translation-invariant* — and why
  pure-magnitude features lose all positional information.
- **Phase:** $\arg(\text{shifted}) = \arg F(\xi) - 2\pi\xi x_0$. The
  shift adds a frequency-proportional ramp to the phase. Steeper
  ramp = bigger shift.

**This is the single most important pair for DVC.** Phase-correlation
registration works by isolating exactly this ramp: divide
$\overline{F_{\text{ref}}}\cdot F_{\text{def}}$ by its magnitude, IFFT,
and the result is a delta at the shift. No magnitude content survives —
only the ramp.

### 2. Modulation ↔ translation (the dual of #1)

$$e^{2\pi i\,\xi_0 x}\,f(x) \;\;\longleftrightarrow\;\; F(\xi - \xi_0)$$

Multiplying by a complex exponential in space *shifts* the spectrum.
This is the radio engineer's "carrier" trick. It's also why `fftshift`
exists: multiplying a real signal by $(-1)^n$ (= sampling $e^{i\pi n}$)
shifts the spectrum by half the sampling rate, moving DC from index 0
to the centre.

### 3. Convolution ↔ multiplication

$$(f * g)(x) \;\;\longleftrightarrow\;\; F(\xi)\,G(\xi)$$

And its sibling, **correlation ↔ conjugate multiplication**:

$$(f \star g)(x) := \int \overline{f(t)}\,g(t+x)\,dt \;\;\longleftrightarrow\;\; \overline{F(\xi)}\,G(\xi)$$

This is what `correlate_linear` exploits. Two FFTs, one elementwise
product, one IFFT — instead of an $O(W^6)$ sum.

- **Magnitude:** $|FG| = |F|\cdot|G|$. Magnitudes multiply.
- **Phase:** $\arg(FG) = \arg F + \arg G$. Phases add. (And for
  correlation, $\arg(\overline{F}G) = \arg G - \arg F$ — phases
  *subtract*, which is why the cross-power phase encodes the
  *displacement* between the two signals.)

### 4. Multiplication ↔ convolution (dual of #3)

$$f(x)\,g(x) \;\;\longleftrightarrow\;\; (F * G)(\xi)$$

A windowed signal has a *blurred* spectrum (convolved with the window's
transform). This is why apodising a subvolume with a Hann window before
FFT trades sharper main lobes for wider skirts — you're convolving the
spectrum with the window's spectrum.

### 5. Scaling ↔ inverse scaling

$$f(a x) \;\;\longleftrightarrow\;\; \frac{1}{|a|}\,F\!\left(\frac{\xi}{a}\right)$$

Squeeze in space → stretch in frequency, with an amplitude factor. The
**uncertainty principle** lives here: you can't have a signal that's
narrow in both domains.

### 6. Differentiation ↔ multiplication by frequency

$$\frac{d f}{d x} \;\;\longleftrightarrow\;\; (2\pi i\,\xi)\,F(\xi)$$

Derivatives boost high frequencies linearly. Iterating:
$\partial^n \leftrightarrow (2\pi i\xi)^n$. This is the basis of
spectral derivatives and Wirtinger / Sobolev norms.

### 7. Conjugation ↔ conjugation-and-flip

$$\overline{f(x)} \;\;\longleftrightarrow\;\; \overline{F(-\xi)}$$

A consequence: if $f$ is **real**, then $F(-\xi) = \overline{F(\xi)}$ —
the **Hermitian symmetry** that `rfftn` exploits to store half the
spectrum.

### 8. Reflection ↔ reflection

$$f(-x) \;\;\longleftrightarrow\;\; F(-\xi)$$

For real signals combined with #7: flipping in space conjugates the
spectrum. This is why correlation $f \star g$ equals convolution
$\overline{f(-\cdot)} * g$, and why `conj(F)` shows up wherever
correlation does.

### 9. Parseval / Plancherel ↔ energy preservation

$$\int |f(x)|^2\,dx \;=\; \int |F(\xi)|^2\,d\xi$$

Total energy is the same in both domains. This is what makes the L2
normalisation in `correlate_cyclic` work: the per-POI norm $\|f\|_2$ can
be computed in either domain.

---

## How magnitude and phase carry information separately

A useful mental model:

| Property                | Magnitude $\|F\|$                                      | Phase $\arg F$                                  |
| ----------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Encodes                 | "what frequencies are present, and how strong"         | "where things are in space"                     |
| Translation invariant?  | **Yes**                                                | No (gains a linear ramp)                        |
| Determines image alone? | No (phase-swap experiments)                            | Largely yes (Oppenheim & Lim, 1981)             |
| Used by                 | power spectrum, autocorrelation, $\|F\|^2$             | phase correlation, registration, structure      |

The classic **Oppenheim–Lim experiment**: take two images, swap their
phases while keeping each one's magnitude. The reconstructions look
like the *phase donor*, not the magnitude donor. **Phase carries the
geometry.** Magnitude alone gives you texture statistics with no
spatial structure (this is also why the autocorrelation $f \star f$,
whose FT is $|F|^2$, throws away all positional information).

---

## Specialisations for NCC and DVC

Putting the pairs together, here is what `correlate_linear` is actually
doing:

1. **Pair #3 (correlation ↔ conjugate multiplication):**
   $\overline{F_{\text{ref}}}\cdot F_{\text{def}}$ in frequency
   = correlation in space.
2. **Pair #1 (translation ↔ phase ramp):** if
   $\text{def} = \text{ref}(\cdot - u)$, then
   $F_{\text{def}} = e^{-2\pi i\xi u} F_{\text{ref}}$, so
   $\overline{F_{\text{ref}}}\,F_{\text{def}} = |F_{\text{ref}}|^2 e^{-2\pi i\xi u}$.
   IFFT of that is an autocorrelation peak *shifted by $u$*. The peak
   location reads off the shift.
3. **Pair #4 (multiplication ↔ convolution):** the indicator box for
   the overlap denominator works because
   $\mathcal{F}\{\text{ref}^2 \cdot \mathbb{1}_{\text{shift}}\}
   = \mathcal{F}\{\text{ref}^2\} * \mathcal{F}\{\mathbb{1}\}$
   — the per-lag overlap sum is itself a correlation.

**Phase-only correlation** (which v1 doesn't use, but is worth knowing)
is the limit of #1 in pure form: divide
$\overline{F_{\text{ref}}}\,F_{\text{def}}$ by its magnitude before
IFFT, leaving only the phase ramp. The IFFT is then a near-delta at
$u$, robust to illumination changes (since those mostly affect $|F|$)
but fragile to noise (since you've thrown away the SNR-bearing
magnitude weighting). NCC is the noise-robust compromise: keep
magnitudes, normalise globally.

---

## One-line summary

**Shift in space = phase ramp in frequency; magnitude is shift-blind;
phase holds the geometry.** Almost every DVC, registration, and
phase-correlation result follows from those three sentences plus
convolution-multiplication duality.

---

## References

- **Bracewell, *The Fourier Transform and Its Applications*** — the
  most pedagogic source for these pairs; the "Pictorial Dictionary of
  Fourier Transforms" chapter is the gold standard.
- **Oppenheim & Schafer, *Discrete-Time Signal Processing*, ch. 2 & 8**
  — discrete versions, including the DFT specifics and circular vs.
  linear convolution.
- **Oppenheim & Lim, "The importance of phase in signals,"
  *Proc. IEEE* 69(5), 1981** — the phase-swap experiment.
- **Kuglin & Hines, "The phase correlation image alignment method,"
  1975** — original phase-correlation registration paper, directly
  applies pair #1.
- **Lewis, "Fast Normalized Cross-Correlation," 1995** — applies pairs
  #3 and #4 in the form `correlate_linear` uses.
- **Foroosh, Zerubia & Berthod, "Extension of phase correlation to
  subpixel registration," *IEEE TIP* 11(3), 2002** — combines pair #1
  with subpixel fitting, the closest published cousin to what
  `peakfit` does.
