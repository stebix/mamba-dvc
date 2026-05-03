# Digital Volume Correlation with FFT-Based Cross-Correlation
## Design document for a Python implementation

*Scope.* Mathematical specification of a local, subset-based DVC pipeline
whose integer-voxel kernel is FFT-based cross-correlation, with sub-voxel
refinement and quality metrics. The document is grounded in the canonical
literature (Bay 1999; Lewis 1995; Foroosh et al. 2002; Guizar-Sicairos et al.
2008; Westerweel et al. 1997; Astarita 2007; Pan & co-workers 2010–2015;
Bar-Kochba et al. 2014; Buljac et al. 2018) and the dominant open-source
reference implementations (SPAM, TomoWarp2, FIDVC, scikit-image).

---

## 1. Notation and problem statement

Let $f, g \colon \mathbb{Z}^3 \to \mathbb{R}$ be the reference and deformed
3-D volumes, sampled on an isotropic grid with voxel size $\Delta$. We seek
the displacement field $\mathbf{u}\colon \Omega \to \mathbb{R}^3$ such that

$$
g(\mathbf{x} + \mathbf{u}(\mathbf{x})) \approx a(\mathbf{x})\, f(\mathbf{x}) + b(\mathbf{x})
$$

where $a, b$ accommodate slowly-varying multiplicative and additive grayscale
drift between scans (X-ray attenuation drift, beam hardening, gain
fluctuations). In **local DVC**, $\mathbf{u}$ is sampled on a regular grid
of points of interest (POIs) $\{\mathbf{x}_p\}$ with node spacing $h$
voxels; around each POI a cubic subset $\Omega_N = [-N/2,N/2)^3$ of $N$
voxels per side is correlated against a search window in $g$. Strain is
recovered by post-differentiation of the discrete displacement field. Global
DVC formulations (Roux–Hild–Leclerc voxel-scale DVC, Mendoza–Hild–Roux 2019
mechanically regularized DVC) replace the per-subset problem with a single
weak-form minimization over the whole domain, regularized by elasticity;
they are out of scope here but cited for completeness.

The per-POI sub-problem is to find the translation
$\mathbf{s}^\star \in \mathbb{R}^3$ that maximizes a similarity functional
between $f_p(\boldsymbol{\xi}) := f(\mathbf{x}_p + \boldsymbol{\xi})$ and
$g_p(\boldsymbol{\xi}; \mathbf{s}) := g(\mathbf{x}_p + \boldsymbol{\xi} + \mathbf{s})$
over $\boldsymbol{\xi} \in \Omega_N$. Higher-order shape functions
(rigid 6-DOF: Smith et al. 2002; affine 12-DOF: Gates et al. 2011) extend
this to local rotation and stretch, but the FFT kernel itself yields only
a translation; rotation and stretch are recovered either by iterative
warping (FIDVC) or by chaining FFT-CC with an inverse-compositional
Gauss–Newton (IC-GN) optimizer (Pan/Jiang 2015; Wang et al. 2016).

---

## 2. Similarity criterion

### 2.1 The three canonical criteria

The three criteria with by far the most use in the DIC/DVC literature are
plain cross-correlation (CC), normalized cross-correlation (NCC), and
zero-mean normalized cross-correlation (ZNCC). For a translation $\mathbf{s}$:

$$
\mathrm{CC}(\mathbf{s}) = \sum_{\boldsymbol{\xi}} f_p(\boldsymbol{\xi})\, g_p(\boldsymbol{\xi}; \mathbf{s})
$$

$$
\mathrm{ZNCC}(\mathbf{s}) =
\frac{\sum_{\boldsymbol{\xi}} \bigl[f_p(\boldsymbol{\xi}) - \bar f_p\bigr]\bigl[g_p(\boldsymbol{\xi};\mathbf{s}) - \bar g_p(\mathbf{s})\bigr]}
     {\sigma_f\, \sigma_g(\mathbf{s})\, |\Omega_N|}
$$

with $\bar f_p$, $\sigma_f$ the local mean and standard deviation in
$\Omega_N$; $\bar g_p(\mathbf{s})$, $\sigma_g(\mathbf{s})$ are the same in
the moving subset of $g$ at offset $\mathbf{s}$.

Pan, Wu & Wang (2010, *Appl. Opt.* 49, 5501–5509) showed that ZNCC, ZNSSD,
and PZNSSD share identical extrema, so any of them is an equally sound
primary criterion. ZNCC is preferred in practice because it is invariant
to affine grayscale change $g \mapsto a g + b$; this matters in
μCT and synchrotron μCT where beam intensity drift and reconstruction
artifacts modulate gray levels between successive scans.

### 2.2 FFT computation of CC

By the Fourier convolution theorem, the linear cross-correlation is

$$
\mathrm{CC}(\mathbf{s}) \;=\; \mathfrak{F}^{-1}\!\bigl[F_p \cdot G_p^{\ast}\bigr](\mathbf{s})
$$

where $F_p = \mathfrak{F}\{f_p\}$, $G_p = \mathfrak{F}\{g_p\}$ are the
3-D DFTs and $\mathbf{s}$ is the spatial-domain output index. Cost is
$O(N^3 \log N)$ per POI versus $O(N^6)$ for spatial brute force; this is the
central performance argument for FFT-CC and is unchanged from Bay et al.
(1999). The DFT yields a *circular* correlation; the standard remedy is
zero-padding both subsets to a length $M \ge 2N - 1$ rounded up to a smooth
FFT size (typically $M = 2N$ or the next power of 2). The choice of pad,
the bias incurred by skipping it, and the alternative of subset apodization
are detailed in §3. Real-input FFTs (`rfftn`) halve memory and runtime since
both inputs are real; the spectrum $F_p \cdot G_p^{\ast}$ is then Hermitian,
which the inverse `irfftn` exploits.

### 2.3 FFT computation of ZNCC (Lewis 1995)

The ZNCC numerator is itself a cross-correlation
$\sum_{\boldsymbol{\xi}}(f_p - \bar f_p)(g_p - \bar g_p) = \mathrm{CC}(\mathbf{s}) - |\Omega_N|\,\bar f_p\,\bar g_p(\mathbf{s})$
and is therefore FFT-friendly. The denominator's $\sigma_f$ is a scalar
precomputed once per subset; $\sigma_g(\mathbf{s})$ depends on the moving
window in $g$. Lewis (1995, *Vision Interface*) showed that running sums
(integral images) of $g$ and $g^2$ make $\sigma_g(\mathbf{s})$ available
for *all* $\mathbf{s}$ in $O(N^3)$. This integral-image construction is
exactly what MATLAB's `normxcorr2` and OpenCV's `matchTemplate` (mode
`CV_TM_CCOEFF_NORMED`) implement; in 3-D it generalizes by adding one
cumulative-sum axis. Briechle & Hanebeck (2001) and Padfield (2011) extend
it to masked correlation for ROIs of arbitrary shape, useful when the bone
or cell ROI does not fill the subset cube.

A specific consequence of the Lewis construction is that the integral-image
denominator sums *only over the overlap region* between reference and
search subsets at each shift, so the partial-overlap weighting that
otherwise biases CC toward zero displacement (§3.1) is implicitly cancelled
in the normalization. This is the main reason FFT-based ZNCC is robust in
practice and why the DIC/DVC literature spends much less time on edge
effects than the PIV literature. Plain CC and phase correlation do *not*
have this safety net.

### 2.4 Phase correlation as an alternative

Phase-only correlation (Kuglin & Hines 1975; Foroosh, Zerubia & Berthod
2002) replaces the cross-power magnitude with unity:

$$
\mathrm{PC}(\mathbf{s}) \;=\; \mathfrak{F}^{-1}\!\left[\frac{F_p \cdot G_p^{\ast}}{|F_p \cdot G_p^{\ast}|}\right]\!(\mathbf{s})
$$

It produces a quasi-Dirac peak under pure translation and is robust to
broadband illumination changes, but is *not* robust to noise: skimage's
own documentation notes that under high noise, unnormalized
cross-correlation outperforms phase correlation. Eckstein & Vlachos (2009)
proposed an explicit spectral filter shaped to the expected
particle-image autocorrelation ("Robust Phase Correlation") that
recovers ~40% noise sensitivity versus unfiltered. In the present design,
phase correlation is recommended only as a fast, optional *coarse*
localizer — not the primary criterion.

---

## 3. Edge effects: zero-padding vs apodization

The DIC/DVC literature is thin on this topic; it has been worked out much
more thoroughly in PIV, where it has been a methodological focus for some
30 years. The mechanisms transfer directly to DVC.

### 3.1 Circular vs linear cross-correlation: the bias

The DFT computes circular cross-correlation: $\mathfrak{F}^{-1}[F_p G_p^\ast]$
implicitly assumes both $f_p$ and $g_p$ are periodic, so for any shift
$\mathbf{s}$ a feature within distance $|\mathbf{s}|$ of the subset
boundary is correlated against its periodic image from the opposite side.
The result is a contaminated correlation map plus a peak-position bias
toward $\mathbf{s} = \mathbf{0}$.

A second, subtler effect survives even in *linear* cross-correlation: the
number of samples that genuinely overlap between $f_p$ and a shifted copy
of $g_p$ decreases as $(N - |s_i|)$ along each axis $i$, so the linear CC
inherits a triangular (Bartlett) weighting

$$
W_{\text{tri}}(\mathbf{s}) = \prod_{i=1}^{3}\left(1 - \frac{|s_i|}{N}\right)_{+}
$$

unless explicitly compensated. Westerweel, Dabiri & Gharib (1997,
*Exp. Fluids* 23, 20–28) is the canonical PIV-context derivation; under
Gaussian image noise the bias scales fractionally as $|\mathbf{s}|/N$ and
is a leading systematic error when displacements are comparable to a
significant fraction of the subset size. Keane & Adrian's
"one-quarter rule" is the practical PIV summary: with circular CC and no
compensation, the maximum reliable displacement is $\lesssim N/4$.

### 3.2 Zero-padding: linear CC

Padding each subset to a length $M \ge 2N - 1$ converts circular into linear
cross-correlation; this is textbook DSP (Oppenheim & Schafer). The standard
choice is $M = 2N$ rounded up to a smooth FFT size (power of 2, or
2-3-5-composite). The cost is $\sim 8\times$ the unpadded FFT volume per
POI in 3-D, rarely a binding constraint on modern hardware. Sub-search-range
optimizations (extracting the search subset at size $N + 2S$, padding to
$M \ge 2N + 2S - 1$) save FFT cost when $S \ll N$, but the simpler
"pad to $2N$" convention works for any displacement up to $N - 1$ voxels
and is the recommended default.

PIVlab (Thielicke 2021, *JORS*) reported, on adding a linear-CC option in
version 2.2, that "linear correlation decreases bias and random (RMS) error,
and also enhances the robustness", at $\sim 2.5\times$ compute over circular.
OpenPIV has used zero-padded linear from the start, citing Raffel et al.'s
PIV practical guide. In modern DVC implementations, zero-padding is the
safe default; the "circular plus stay-away-from-edges" approach is a legacy
choice driven by memory budgets that no longer bind.

When ZNCC is computed Lewis-style (§2.3), the integral-image denominator
already corrects the partial-overlap triangle in the normalization, so the
residual bias from skipping zero-padding is small even with circular CC.
Plain CC and phase correlation lack this safety net; if either is used as
the primary criterion without zero-padding, an explicit triangle correction
$W_{\text{tri}}^{-1}(\mathbf{s})$ should be applied to the correlation map
before peak extraction.

### 3.3 Apodization: subset windowing

Multiplying the subset by a window function before the FFT does two things.
First, it suppresses spectral leakage from the boundary discontinuity (a
rectangular subset has a $\mathrm{sinc}$ frequency response with high
sidelobes). Second, it down-weights features near the subset edge —
exactly the features that disappear under a translation, so it also
reduces the displacement-dependent bias of §3.1. Standard windows in order
of increasing aggressiveness are rectangular, Tukey
(cosine-tapered rectangle with parameter $\alpha \in [0,1]$), Hann
($\alpha = 1$ Tukey), Hamming, Blackman, Nuttall.

The reference comparison for image-correlation contexts is Astarita (2007,
*Exp. Fluids* 43, 859–872), who tested these windows in iterative-deformation
PIV. The headline tradeoff: apodization improves modulation transfer at
small wavelengths (better resolution of small-scale flow features) but
reduces the *effective* subset size and broadens the correlation peak. Hann
is cleanest spectrally but sacrifices ~50% of the subset energy; Tukey with
$\alpha \approx 0.25$–$0.5$ is the usual compromise, keeping a flat central
region and tapering only the edges.

Eckstein, Charonko & Vlachos (2008, *Exp. Fluids* 45, 485–500) and Eckstein
& Vlachos (2009, *MST* 20, 055401) treat the analogous question in the
*spectral* domain — applying an explicit, autocorrelation-shaped filter
directly to $F_p \cdot G_p^\ast / |F_p \cdot G_p^\ast|$ — and report
~40% noise-sensitivity reduction versus unfiltered phase correlation.

A modern alternative worth noting is *periodic plus smooth decomposition*
(Moisan 2011, *J. Math. Imaging Vis.* 39, 161–179; Mahmood et al. 2016/2018
for a fast implementation): decompose $f = f_p + f_s$ into a periodic
component $f_p$ and a smooth boundary-correcting component $f_s$, then
FFT only $f_p$. Avoids both apodization energy loss and zero-padding cost.
Not yet standard in DIC/DVC but conceptually clean and inexpensive.

### 3.4 Cost of apodization in the SSSIG framework

Apodization is not a free improvement; it directly degrades the per-POI
noise floor. The Pan/Lu/Xie 2010 displacement uncertainty bound for a
windowed subset becomes

$$
\sigma_u^2 \;=\; \frac{2\sigma_n^2}{\mathrm{SSSIG}_w}, \qquad
\mathrm{SSSIG}_w \;=\; \sum_{\boldsymbol{\xi} \in \Omega_N} w(\boldsymbol{\xi})^2 \, \|\nabla f(\mathbf{x}_p + \boldsymbol{\xi})\|^2
$$

with $w$ the apodization window. Aggressive windowing reduces
$\mathrm{SSSIG}_w$ and raises the noise floor; it also broadens the
correlation peak and degrades sub-voxel localization. The choice trades
bias against variance and should be informed by the dominant error in the
data — a zero-strain test on the actual imaging modality is the right way
to decide.

### 3.5 Interaction with sub-voxel refinement

Apodization does *not* commute with the Guizar-Sicairos upsampled DFT
(§4.2). The upsampled DFT recovers the location of the peak of whatever
spectrum it is given; if $f_p, g_p$ have been windowed before the FFT, the
upsampled DFT estimates the peak of (windowed-CC), not (CC), and the bias
depends on the window shape rather than vanishing. Two clean options:
window only the integer-voxel coarse pass and re-extract unwindowed
subsets for the refinement, or skip apodization entirely and rely on
zero-padding plus ZNCC normalization. The pre-filter of Pan 2013 (§5,
step 1) does *not* have this problem because it acts on the full volumes
before subset extraction; it is therefore the preferred mechanism for
suppressing high-frequency content.

### 3.6 What the DIC/DVC literature actually does

Almost universally: zero-pad to convert to linear CC, ZNCC normalization
for partial-overlap correction, and Pan 2013 Gaussian *pre-filtering* of
the full volumes (not subset apodization) for high-frequency leakage and
interpolation bias. The exception is in low-SNR or cropped-ROI settings
(e.g. some speckle-tracking X-ray DIC studies) where an explicit subset
window $w(\mathbf{x})$ enters the ZNCC formula, primarily for edge-artifact
suppression in noisy data rather than for bias correction. FFT-CC followed
by IC-GN (Pan/Jiang 2015; Wang et al. 2016), FIDVC (Bar-Kochba 2014),
SPAM-register, and TomoWarp2 all skip subset apodization.

### 3.7 Recommended defaults

Linear CC via zero-padding to $M = 2N$ rounded to a smooth FFT size, with
ZNCC Lewis-style normalization that implicitly handles the partial-overlap
triangle. Pan 2013 Gaussian pre-filter ($\sigma \approx 0.7$ voxel) on the
full volumes addresses high-frequency leakage *before* subset extraction
and is complementary to zero-padding. No subset apodization by default.
Add Tukey ($\alpha \approx 0.25$) only if edge-driven peak distortion is
observed in the specific data, and adjust the SSSIG uncertainty estimate
to $\mathrm{SSSIG}_w$ accordingly. If raw CC or phase correlation is used
instead of ZNCC, apply the explicit triangle correction
$W_{\text{tri}}^{-1}(\mathbf{s})$ to the correlation map before peak
extraction.

For an MRI-physics audience these issues map directly onto FFT-based
navigator motion correction in $k$-space: the circular-shift assumption is
the same, the partial-overlap bias toward zero motion is the same. The
twist in DVC is that subsets ($N \sim 16$–$48$) are much smaller than
typical $k$-space matrices, so the $|\mathbf{s}|/N$ fractional bias bites
earlier and zero-padding matters more.

---

## 4. Sub-voxel refinement

After locating the integer-voxel maximum $\mathbf{s}_{\mathrm{int}}$, the
sub-voxel residual must be estimated. Three techniques span the
literature; their typical achievable precision, on textured speckle and
moderate noise, lies in $0.005$–$0.1$ voxel.

### 4.1 Local peak fitting (3-point parabolic / Gaussian)

Per axis, fit $C(\mathbf{s})$ in the integer triple
$\{C_{-}, C_0, C_{+}\}$:

$$
\delta_{\text{parab}} = \frac{C_+ - C_-}{2(2C_0 - C_+ - C_-)}, \qquad
\delta_{\text{gauss}} = \frac{\ln C_+ - \ln C_-}{2(2\ln C_0 - \ln C_+ - \ln C_-)}.
$$

Gaussian fit is preferred when the subset autocorrelation is approximately
Gaussian (the typical case for tomographic speckle and PIV particle
images; Willert & Gharib 1991, Westerweel 1997). Cost is $O(1)$ per axis.
Both schemes produce *peak-locking bias* at the $\le 0.05$-voxel level
(Heid & Kääb 2012, Sutton et al. 2009 ch. 5), with bias preferentially
toward integer values; this bias is the dominant systematic component when
the underlying displacement is much smaller than $1$ voxel.

### 4.2 Upsampled DFT (Guizar-Sicairos, Thurman & Fienup 2008)

The cleanest, lowest-bias subvoxel refinement is the matrix-multiply
upsampled DFT of Guizar-Sicairos et al. (*Opt. Lett.* 33, 156–158, 2008),
also the algorithm in `skimage.registration.phase_cross_correlation`. It
is a two-stage procedure:

**Stage 1 — coarse.** Compute the integer-voxel peak by standard
zero-padded FFT cross-correlation with upsample factor $\kappa=2$.

**Stage 2 — local upsample.** In a small window of size $\kappa_w$ around
$\mathbf{s}_{\mathrm{int}}$, evaluate the inverse DFT of $F_p G_p^{\ast}$
on a grid upsampled by $u$ via direct matrix-multiply DFT:

$$
\widehat{C}_u[\mathbf{k}] \;=\; \bigl(D_u^{(1)} \otimes D_u^{(2)} \otimes D_u^{(3)}\bigr)\,\bigl[F_p \odot G_p^{\ast}\bigr]
$$

where $D_u^{(i)}$ is the $\kappa_w \times N$ DFT-matrix that evaluates the
inverse transform only on the $\kappa_w$ samples of interest along axis
$i$, on a grid of spacing $1/u$ voxel. This avoids the
$O((uN)^3 \log uN)$ cost of full zero-pad upsampling; total cost is
$O(N^3 + \kappa_w^3 u^3 + 3\,\kappa_w u N)$. With $u=10\!-\!100$, sub-voxel
precision is $0.01$–$0.1$ voxel — well below the noise floor for almost any
DVC application. The 3-D generalization is immediate and dense
(separable matmul along each axis), making it well-suited to JAX/GPU
vectorization. scikit-image's internal `_upsampled_dft` is the canonical
reference and is dimension-agnostic, hence directly usable for 3-D.

As noted in §3.5, this stage requires the spectrum to be that of the
unwindowed subset; if apodization is in use, window only the integer-voxel
coarse pass and re-extract the unwindowed subset for the refinement.

### 4.3 IC-GN with FFT-CC initial guess (Pan/Jiang 2015; Wang et al. 2016)

For maximum accuracy and to fit shape-function parameters (rotation,
stretch, shear), feed the FFT-CC integer guess into an
inverse-compositional Gauss–Newton (IC-GN) optimization of the ZNSSD
criterion. IC-GN converges in $4$–$8$ iterations from a reasonable initial
guess; under SSD with linear interpolation, Pan (2014, *Strain* 50, 48–56)
proved IC-GN has identical noise robustness to forward-additive
Newton–Raphson (FA-NR) but lower per-iteration cost because the Hessian is
constant in the inverse-compositional formulation. The combination
"FFT-CC → IC-GN" is the *path-independent DIC/DVC* paradigm: each POI is
refined from its own FFT-CC seed, eliminating the propagation of seeding
errors that plagues reliability-guided schemes and trivially parallelizing
across POIs.

For an FFT-only implementation with parabolic/Gaussian peak fits, sub-voxel
precision of $\sim 0.05$ voxel is realistic (Liu & Morgan 2007, Bay 1999);
adding IC-GN brings it to $0.005$–$0.02$ voxel under good speckle/noise
(Wang et al. 2016).

---

## 5. End-to-end pipeline

Inputs: $f, g$ on an isotropic grid; POI grid spacing $h$; subset size $N$;
search half-width $S$; upsampling $u$; optional ROI mask $M$.

**(1) Preprocessing.** Apply isotropic Gaussian blur with
$\sigma \approx 0.5$–$1$ voxel to both volumes. Pan (2013, *OLE* 51,
1161–1167) showed that Gaussian pre-filtering reduces the interpolation bias
of subvoxel registration by 30–50%; the residual bias is bounded by the
Nyquist content above $1/(2\sigma)$ cycles/voxel, which the filter has now
suppressed. Do *not* over-filter: aggressive smoothing destroys the speckle
gradients the algorithm needs (Pan 2010 SSSIG argument; see §6.1).

**(2) POI grid.** Generate $\{\mathbf{x}_p\}$ on a regular Cartesian
lattice with spacing $h$. Discard POIs whose subset $\mathbf{x}_p + \Omega_N$
leaves the ROI mask $M$.

**(3) FFT-CC per POI.** For each $\mathbf{x}_p$:

  a. Extract reference subset $f_p \in \mathbb{R}^{N^3}$ and search subset
     $g_p \in \mathbb{R}^{(N+2S)^3}$.
  b. Subtract local means to remove the DC term (this guarantees the
     correlation peak is bounded and Hermitian symmetry holds).
  c. Zero-pad both subsets to a common smooth FFT size
     $M \ge \max(N, N + 2S) + N - 1$, typically $M = 2N$ rounded up to a
     power of 2, for linear cross-correlation (§3.2). When $S \ll N$,
     padding only to $M \ge 2N + 2S - 1$ saves FFT cost.
  d. Compute real FFTs $F_p = \mathrm{rfftn}(f_p)$, $G_p = \mathrm{rfftn}(g_p)$.
  e. Compute $C = \mathrm{irfftn}(F_p \cdot G_p^{\ast})$, optionally
     normalized by $\sigma_f \sigma_g(\mathbf{s})$ via integral images of
     $g$ and $g^2$ for ZNCC. The integral-image denominator implicitly
     corrects the partial-overlap triangle, so no separate
     $W_{\text{tri}}^{-1}$ correction is needed for ZNCC; raw CC and phase
     correlation require it (§3.2).
  f. $\mathbf{s}_{\mathrm{int}} = \arg\max_{\mathbf{s}} C(\mathbf{s})$
     restricted to $\|\mathbf{s}\|_\infty \le S$.

**(4) Subvoxel refinement.** Apply Guizar-Sicairos upsampled DFT (§4.2)
in a $\kappa_w^3$ window around $\mathbf{s}_{\mathrm{int}}$ with factor $u$.
For shape-function recovery, optionally chain IC-GN (§4.3) with
$\mathbf{s}_{\mathrm{int}} + \boldsymbol\delta$ as initial guess.

**(5) Per-POI metrics.** Record peak ZNCC*, secondary-peak ratio,
correlation FWHM, SSSIG (§6). These drive outlier rejection.

**(6) Outlier rejection.** Apply the Universal Median Test (Westerweel &
Scarano 2005, *Exp. Fluids* 39, 1096–1100): for each POI, compute the
normalized fluctuation against the median displacement of its
$3 \times 3 \times 3$ POI neighborhood; flag if it exceeds threshold
$\varepsilon \approx 2$. Replace by neighborhood median or NaN. This is the
same scheme FIDVC and SPAM both use.

**(7) (Optional) Iterative deformation method.** When displacements exceed
the FFT search radius $S$ or when local stretch is appreciable: warp $g$
by the current displacement field (B-spline or trilinear interpolation),
recompute FFT-CC on the residual, and add the increment. Iterate until
$\|\Delta \mathbf{u}\|_\infty < \mathrm{tol}$. This is the FIDVC scheme
(Bar-Kochba et al. 2014, *Exp. Mech.* 55, 261–274) and is the cleanest way
to recover finite deformations of $\sim 30\%$ strain with a single
implementation.

**(8) Strain.** Compute $F = I + \nabla \mathbf{u}$ and the Green–Lagrange
strain $E = \tfrac{1}{2}(F^\top F - I)$ from the discrete displacement
field. Two standard schemes:

- Centred finite differences on the regular POI grid: simple, but
  noise-amplifying (the strain noise floor is $\sim \sqrt{2}\,\sigma_u/h$
  per component for nearest-neighbor differences).
- FE smoothing on a tetrahedral mesh of POIs (Geers, De Borst &
  Brekelmans 1996; SPAM's `regularStrain` script): better noise behavior,
  natural support for irregular POI grids and large strain.

For most applications a $3 \times 3 \times 3$ centered scheme with a
post-hoc Gaussian smoothing of the strain field is a reasonable default;
move to FE-based smoothing once SDER becomes the limiting metric (§6).

---

## 6. Quality metrics

Two metric families are tracked. Per-POI metrics drive online outlier
rejection. Global metrics on zero-strain or virtually-shifted test pairs
characterize the implementation as a whole.

### 6.1 Per-POI runtime metrics

**Peak ZNCC.** $C^\star = \max_{\mathbf{s}} \mathrm{ZNCC}(\mathbf{s})$. Should
be close to 1 for a good match. Empirical thresholds used in the bone
literature (Palanca et al. 2015; Tozzi et al. 2017): $C^\star > 0.9$
reliable, $0.7$–$0.9$ marginal, $< 0.7$ outlier candidate.

**Secondary-peak ratio.** $R = C^\star / C^{(2)}$ where $C^{(2)}$ is the
next-largest *local* maximum (excluding a neighborhood of $C^\star$).
$R \gtrsim 1.5$ indicates a unique peak; $R \to 1$ flags ambiguity from
insufficient texture or aliased speckle. Charonko & Vlachos (2013,
*MST* 24, 065301) develop this metric quantitatively for PIV uncertainty
estimation; the same construction transfers to DVC.

**SSSIG / MIG.** Pan, Xie, Wang, Qian, Wang (2008) and Pan, Lu, Xie (2010,
*OLE* 48, 469–477) derive the displacement uncertainty of SSD-based DIC as

$$
\sigma_u^2 \;=\; \frac{2\sigma_n^2}{\mathrm{SSSIG}}, \qquad
\mathrm{SSSIG}(\mathbf{x}_p) = \sum_{\boldsymbol{\xi} \in \Omega_N} \|\nabla f(\mathbf{x}_p+\boldsymbol{\xi})\|^2
$$

with $\sigma_n$ the image noise std. This is a Cramér–Rao-style bound for
translation under Gaussian image noise and is the most rigorous per-POI
*a priori* uncertainty estimate available; it costs one gradient pass per
subset. Apodized variants use $\mathrm{SSSIG}_w$ as in §3.4. The Mean
Intensity Gradient
$\mathrm{MIG} = (1/|\Omega_N|)\sum \|\nabla f\|$ is the global counterpart.
For 8-bit $\mu$CT, MIG below $\sim 5$ grayscale-units/voxel typically
predicts unreliable convergence; for 16-bit data the threshold scales
linearly with dynamic range.

**Correlation peak FWHM.** Width of the upsampled correlation peak. A
broadened FWHM diagnoses blur or de-correlation (e.g., severe local
rotation that the translation-only kernel cannot match).

### 6.2 Global validation metrics (zero-strain and virtually-shifted tests)

The DVC community has converged on two complementary aggregate metrics
(Liu & Morgan 2007; Palanca et al. 2015, 2016; Dall'Ara et al. 2014, 2017):

- **MAER** (Mean Absolute Error of strain): $\mathrm{MAER} = \langle |\varepsilon_{ij}^{\text{meas}} - \varepsilon_{ij}^{\text{true}}| \rangle$ — accuracy / systematic.
- **SDER** (Standard Deviation of Error of strain): $\mathrm{SDER} = \mathrm{std}(\varepsilon_{ij}^{\text{meas}} - \varepsilon_{ij}^{\text{true}})$ — precision / random.
- **Displacement random error**: $\mathrm{std}(\mathbf{u}_{\text{meas}} - \mathbf{u}_{\text{true}})$, in voxels or μm.
- **Displacement systematic error / peak-locking curve**: $\langle \mathbf{u}_{\text{meas}} - \mathbf{u}_{\text{true}}\rangle$ as a function of true sub-voxel position.

Two test types are used (Buljac et al. 2018, §4):

1. **Virtually-shifted** test: take a single scan, apply a known
   sub-voxel shift via Fourier shift theorem, run DVC against the original.
   Captures algorithmic bias in the absence of physical noise — gives
   *optimistic* numbers.
2. **Zero-strain repeated-scan** test: physically rescan the same
   undeformed specimen, run DVC. Captures the realistic combination of
   image noise, reconstruction artifacts, and algorithmic bias.

Both should be reported. A virtually-shifted test in particular is the
right tool for diagnosing residual circular-CC bias (§3.1): apply known
sub-voxel and integer shifts of varying magnitude and inspect the
displacement-vs-true-displacement curve for the characteristic bias
toward zero.

### 6.3 Realistic reference values from literature

Numbers below anchor what "good" looks like and assume ZNCC-based subset
DVC with sub-voxel refinement and a zero-strain test on real
$\mu$CT/SR-$\mu$CT pairs (the harder test condition). All values are quoted
*per strain component* unless stated otherwise.

**Trabecular bone, lab μCT (∼20–40 μm voxel).**
- Liu & Morgan (2007, *J. Biomech.* 40, 3516): displacement error 1.86–3.39 μm
  ($\approx$ 0.05–0.10 voxel), strain error 345–794 με at $30^3$–$50^3$ voxel
  subsets. Foundational benchmark across six bone types.
- Dall'Ara et al. (2014) on porcine vertebrae, 39 μm voxel: random strain
  error 64–221 με at $48^3$ voxel ($1872$ μm) subset, rising sharply to
  359–1203 με at $16^3$ voxel ($624$ μm) subset. Demonstrates the strong
  $\mathrm{SDER}\propto N^{-\alpha}$ trend with $\alpha \in [1,2]$.
- Palanca et al. (2016, *J. Biomech.* 49, 3882) on augmented vertebrae:
  MAER 100–400 με and SDER ~200 με at 1872 μm subset (FFT+DC and global
  ShIRT-FE).

**Cortical bone, synchrotron μCT (1.6 μm voxel).**
- Palanca et al. (2017) and Dall'Ara et al. (2017, *Front. Mater.* 4:31):
  displacement random error 0.024–0.226 μm; strain SDER below 200 με at
  most subset sizes; under favorable conditions, single-BSU spatial
  resolution of ~300 μm with errors as low as 9–46 με. SR-μCT remains the
  gold standard for DVC precision.

**HR-pQCT, long human bone (∼40 μm voxel).**
- Du et al. (2024, *Comput. Methods Biomech. Biomed. Imaging Vis.*):
  with a modified stitching protocol, displacement precision $\sim 0.10$
  voxel, MAER 527 με, SDER 589 με, LSC (least-significant change) 1633 με.
  Clinically achievable performance.

**Clinical 3T MRI of bone (DVC at coarser resolution).**
- Dall'Ara et al. (2020, *Materials* 13:5354): in-vivo precision 200 με at
  7.5 mm spatial resolution, 133 με at 10 mm. Direct evidence that an
  FFT initial guess plus direct-correlation refinement (FFT+DC) beats
  either alone: displacement random error $9.5$–$25.4$ μm with FFT+DC,
  $9.5$–$29.7$ μm with FFT only, $9.5$–$46.5$ μm with DC only, across
  subset sizes 16–56 voxels.

**Soft materials / cells, confocal microscopy.**
- Bar-Kochba et al. (2014) FIDVC validation: displacement noise floor
  $\sim 0.02$ voxel, strain errors below 10% for applied stretches up to
  $\sim 30\%$. The iterative-deformation method is the differentiator at
  finite deformation.

**Implementation acceptance targets.** A new Python FFT-CC DVC should
target, at the *zero-strain* condition on real $\mu$CT-class data:

- Displacement precision $\le 0.05$ voxel with sub-voxel refinement at
  $u=10$–$20$. ($\le 0.02$ voxel is achievable with FFT-CC + IC-GN.)
- Strain SDER 200–400 με at subset sizes $32^3$–$48^3$ in well-textured,
  trabecular-like data. Larger subsets and FE-smoothed strains can push
  this below 100 με.
- 1000+ με at $16^3$ subsets and degraded data — to be expected; do not
  pursue smaller subsets without strong regularization.
- Throughput $\sim 10^4$ POIs/s on a single modern GPU is a realistic
  JAX target for $N=32$ subsets (the FFT cost is ~ms per POI, the
  bottleneck is data movement).

These are not aspirational — they are baselines reached by FIDVC, SPAM,
TomoWarp2, and DaVis. Material shortfall would indicate an implementation
bug rather than an algorithmic limit.

---

## 7. Key improvements over the bare Bay 1999 single-pass FFT-CC

In recommended order of payoff for an MVP:

**ZNCC instead of plain CC.** Adds invariance to grayscale offset/gain, at
the cost of one integral-image pass. Lewis (1995) is the canonical
reference; this is mandatory in practice for X-ray data. Bonus benefit:
the integral-image denominator implicitly corrects the partial-overlap
triangle (§3.2), removing one source of bias.

**Zero-padding to linear CC.** Pad both subsets to $M = 2N$ on a smooth
FFT size to convert circular to linear cross-correlation (§3.2). Without
this, displacements suffer the bias toward zero documented by Westerweel,
Dabiri & Gharib (1997); residual bias persists at all scales of $|s|/N$.
Cheap on modern hardware ($\sim 8\times$ FFT volume in 3-D), and standard
in PIVlab (since v2.2), OpenPIV, and FIDVC.

**Gaussian pre-filter, $\sigma\!\approx\!0.5$–$1$ voxel.** Halves the
interpolation bias in sub-voxel refinement (Pan 2013). Acts on the full
volumes before subset extraction, so it does not bias the upsampled DFT
the way subset apodization would (§3.5).

**Guizar-Sicairos upsampled DFT for sub-voxel.** Bias-free to numerical
precision, $0.005$–$0.01$ voxel readily, GPU-friendly. State-of-the-art in
scikit-image.

**Universal Median Test outlier filter.** Standard PIV/DVC robustification
(Westerweel & Scarano 2005); cheap, effective.

**FFT-CC integer guess fed to IC-GN refinement.** The
"path-independent DVC" of Pan/Jiang (2015) and Wang et al. (2016).
Recovers shape-function parameters (local affine), drops
sub-voxel error to 0.005–0.02 voxel, parallelizes trivially across POIs.

**Iterative deformation method (FIDVC).** Warp-and-recorrelate to capture
displacements larger than the FFT search radius and finite stretches
without enlarging the subset (Bar-Kochba et al. 2014). Open-source MATLAB
reference at `FranckLab/FIDVC`.

**Multi-scale (image pyramid).** Coarse FFT-CC at 4× downsampled volumes
for a wide-radius initial estimate, then refine at full resolution with
small search. Standard in TomoWarp2, FIDVC, and SPAM `register`.

**Optional subset apodization (Tukey, $\alpha\!\approx\!0.25$).** Worth
adding only if a zero-strain test on the actual data exhibits edge-driven
peak distortion that zero-padding alone does not resolve (§3.3–3.4). Not
the default; trades bias against variance and complicates interaction
with sub-voxel refinement (§3.5).

**3-D SIFT-aided initial guess.** For very large or discontinuous
deformations where neither FFT-CC nor warping converges (Yang et al. 2021,
*OLE* 136, 106323; open-source at `SCUT-CCNL/3DSIFT-PiDVC`). Adds
substantial complexity, only worth it when the iterative-deformation
method fails.

**Mechanical regularization (global DVC).** Out of scope for the local
FFT-CC pipeline but worth noting: Mendoza, Neggers, Hild, Roux (2019,
*CMAME* 355, 27–43) penalize the elastic equilibrium gap; with strong
regularization, SDER below 50 με is reported on synchrotron μCT data.

---

## 8. Reference Python implementations

Three open-source codebases are worth direct study before re-implementation:

**SPAM** (Stamati et al. 2020, *JOSS* 5, 2286). The most actively maintained
Python DVC codebase. Local (`spam-ldic`), discrete (`spam-ddic`), and
global (`spam-gdic`, beta) variants. The core `register` function is a
non-rigid, 12-DOF Lucas–Kanade-style iterative registration, *not* FFT-CC,
but the per-POI driver, ROI handling, strain calculator and outlier filter
are excellent reference. `pip install spam`; source at
`gitlab.com/spam-project/spam`.

**TomoWarp2** (Tudisco et al. 2017, *SoftwareX* 6, 267). Pure-Python local
DIC/DVC with GUI; computes the correlation in image space (sum-table NCC),
hence does *not* use FFT — useful as a sanity-check reference precisely
because its algorithm is independent of yours. Source at
`github.com/ElsevierSoftwareX/SOFTX-D-17-00061`.

**scikit-image `registration.phase_cross_correlation`**. The canonical
Python implementation of Guizar-Sicairos. Works in arbitrary $n$D,
directly usable as the per-POI sub-voxel kernel for 3-D translation.
Inspect `_phase_cross_correlation.py` and `_upsampled_dft`; both are
short, BSD-licensed, and well-commented. The signature exposes
`upsample_factor`, `normalization` (`'phase'` or `None`), and masked
correlation via `reference_mask` / `moving_mask` (Padfield 2011).

**PIVlab** (Thielicke 2021, *JORS* 9, e334). MATLAB, not Python, but its
v2.2+ side-by-side comparison of circular vs linear CC is the cleanest
empirical demonstration that zero-padding matters in practice; relevant
for validating the §3 design choices.

**FIDVC** (Bar-Kochba et al. 2014; `github.com/FranckLab/FIDVC`). MATLAB,
not Python, but the reference implementation of the iterative deformation
method. The warp-correlate-update logic transfers to Python in $\sim$200
lines.

**3DSIFT-PiDVC** (Yang et al. 2021; `github.com/SCUT-CCNL/3DSIFT-PiDVC`).
C++/multi-thread reference for FFT-CC + IC-GN + 3-D SIFT initial guess.
The configuration-driven layout is a useful template.

For a JAX-based implementation in the existing `mritools` workflow, the
mapping is straightforward:

- `jax.numpy.fft.rfftn` / `irfftn` for the per-POI 3-D real FFT (memory
  halved versus complex FFT).
- `jax.vmap` over the POI axis to batch across subsets;
  `jax.pmap` across devices for multi-GPU.
- The Guizar-Sicairos upsampled DFT is two separable matmuls per axis
  (Hermitian on $F_p G_p^{\ast}$); naturally `vmap`-friendly and reuses
  the same DFT matrices across all POIs at fixed $u, \kappa_w, N$.
- ZNCC denominator: integral images via `jnp.cumsum` along three axes
  with stride-tricks, or equivalently a separable box-filter convolution.
- Outlier filtering (Universal Median Test) and strain computation are
  small-kernel operations: either `scipy.ndimage` on host or
  `jax.scipy.signal` on device.
- The IC-GN refinement, if added, is a $12$-parameter Gauss–Newton with a
  constant Hessian — `jax.lax.scan` over iterations is the natural
  pattern, identical to the JAX MRF reconstruction loops in
  `mritools/recon`.

The `tissue-properties`-style YAML-as-source-of-truth pattern adapts
cleanly to DVC algorithm parameters (subset size, search radius, upsampling
factor, peak threshold, UMT threshold, padding strategy, optional window
type), keeping documented references to the literature defaults next to
the values.

---

## 9. Summary specification

A minimal-but-defensible Python FFT-CC DVC consists of:

| Component         | Choice                                                                | Reference                            |
| ---               | ---                                                                   | ---                                  |
| Pre-filter        | Isotropic Gaussian, $\sigma = 0.7$ vox, on full volumes               | Pan 2013                             |
| Similarity        | ZNCC via FFT numerator + integral-image denominator                   | Lewis 1995; Pan 2010                 |
| Padding strategy  | Zero-pad each subset to $M = 2N$ rounded to smooth FFT size (linear CC) | Westerweel et al. 1997; PIVlab 2.2 |
| Apodization       | Off by default; Tukey $\alpha = 0.25$ only if edge artifacts persist  | Astarita 2007                        |
| Coarse search     | Real-FFT 3-D cross-correlation                                        | Bay 1999                             |
| Triangle correction | Implicit via ZNCC denominator; explicit $W_\text{tri}^{-1}$ if using raw CC | §3.2                          |
| Sub-voxel         | Guizar-Sicairos upsampled DFT, $u = 20$, $\kappa_w = 3$               | Guizar-Sicairos et al. 2008          |
| Shape function    | Translation only at FFT level; optional IC-GN stage                   | Pan/Jiang 2015; Wang et al. 2016     |
| Large deformation | Iterative deformation method (warp + recorrelate)                     | Bar-Kochba et al. 2014               |
| Outlier filter    | Universal Median Test, $\varepsilon = 2$                              | Westerweel & Scarano 2005            |
| Strain            | Centered FD or tet-mesh FE smoothing                                  | Geers et al. 1996; SPAM              |
| Per-POI quality   | ZNCC*, secondary-peak ratio, SSSIG-CRLB                               | Pan/Lu/Xie 2010                      |
| Global validation | MAER, SDER on virtual-shift + zero-strain pairs                       | Liu & Morgan 2007; Buljac 2018       |

Default subset size $N = 32$ for low-contrast data, $N = 24$ for
well-textured trabecular speckle, $N = 48$+ for clinical-resolution data.
Default search radius $S = 4$ voxels with multi-scale pyramid for larger
displacements.

---

## 10. References

[Astarita07]  Astarita, T. (2007). Analysis of weighting windows for image
deformation methods in PIV. *Exp. Fluids* 43, 859–872.

[BarKochba14]  Bar-Kochba, E., Toyjanova, J., Andrews, E., Kim, K.-S.,
Franck, C. (2014). A fast iterative digital volume correlation algorithm
for large deformations. *Exp. Mech.* 55(1), 261–274.

[Bay99]  Bay, B.K., Smith, T.S., Fyhrie, D.P., Saad, M. (1999).
Digital volume correlation: Three-dimensional strain mapping using X-ray
tomography. *Exp. Mech.* 39(3), 217–226.

[Briechle01]  Briechle, K., Hanebeck, U.D. (2001). Template matching using
fast normalized cross correlation. *SPIE* 4387.

[Buljac18]  Buljac, A., Jailin, C., Mendoza, A., Neggers, J.,
Taillandier-Thomas, T., Bouterf, A., Smaniotto, B., Hild, F., Roux, S.
(2018). Digital Volume Correlation: Review of Progress and Challenges.
*Exp. Mech.* 58, 661–708.

[Charonko13]  Charonko, J.J., Vlachos, P.P. (2013). Estimation of
uncertainty bounds for individual particle image velocimetry measurements
from cross-correlation peak ratio. *Meas. Sci. Tech.* 24, 065301.

[DallAra14]  Dall'Ara, E., Barber, D., Viceconti, M. (2014). About the
inevitable compromise between spatial resolution and accuracy of strain
measurement for bone tissue. *J. Biomech.* 47(12), 2956–2963.

[DallAra17]  Dall'Ara, E., Peña-Fernández, M., Palanca, M., Giorgi, M.,
Cristofolini, L., Tozzi, G. (2017). Precision of Digital Volume
Correlation Approaches for Strain Analysis in Bone Imaged with Micro-CT
at Different Dimensional Levels. *Front. Mater.* 4:31.

[DallAra20]  Dall'Ara, E., Bodey, A.J., Isaksson, H., Tozzi, G. (2020).
*In Vivo* Deformation and Strain Measurements in Human Bone Using DVC and
3T Clinical MRI. *Materials* 13, 5354.

[Du24]  Du et al. (2024). Accuracy and precision of internal displacement
and strain measurements in long human bones using HR-pQCT and DVC.
*Comput. Methods Biomech. Biomed. Imaging Vis.* (Taylor & Francis).

[Eckstein08]  Eckstein, A.C., Charonko, J., Vlachos, P. (2008). Phase
correlation processing for DPIV measurements. *Exp. Fluids* 45, 485–500.

[Eckstein09]  Eckstein, A., Vlachos, P.P. (2009). Digital particle image
velocimetry (DPIV) robust phase correlation. *Meas. Sci. Tech.* 20,
055401.

[Foroosh02]  Foroosh, H., Zerubia, J., Berthod, M. (2002). Extension of
phase correlation to subpixel registration. *IEEE Trans. Image Process.*
11(3), 188–200.

[Geers96]  Geers, M.G.D., De Borst, R., Brekelmans, W.A.M. (1996).
Computing strain fields from discrete displacement fields in 2D solids.
*Int. J. Solids Struct.* 33, 4293–4307.

[GuizarSicairos08]  Guizar-Sicairos, M., Thurman, S.T., Fienup, J.R. (2008).
Efficient subpixel image registration algorithms. *Opt. Lett.* 33, 156–158.

[Heid12]  Heid, T., Kääb, A. (2012). Evaluation of existing image matching
methods for deriving glacier surface displacements globally from optical
satellite imagery. *Remote Sens. Environ.* 118, 339–355.

[Jiang15]  Jiang, Z., Kemao, Q., Miao, H., Yang, J., Tang, L. (2015).
Path-independent digital image correlation with high accuracy, speed and
robustness. *Opt. Lasers Eng.* 65, 93–102.

[KeaneAdrian92]  Keane, R.D., Adrian, R.J. (1992). Theory of
cross-correlation analysis of PIV images. *Appl. Sci. Res.* 49, 191–215.

[Lewis95]  Lewis, J.P. (1995). Fast Normalized Cross-Correlation.
*Vision Interface*, Canadian Image Processing and Pattern Recognition Society.

[LiuMorgan07]  Liu, L., Morgan, E.F. (2007). Accuracy and precision of
digital volume correlation in quantifying displacements and strains in
trabecular bone. *J. Biomech.* 40(15), 3516–3520.

[Mahmood18]  Mahmood, F., Toots, M., Öfverstedt, L.-G., Skoglund, U. (2018).
Algorithm and architecture optimization for 2D Discrete Fourier Transforms
with simultaneous edge artifact removal. *Int. J. Reconfigurable Computing*
2018, 1403181.

[Mendoza19]  Mendoza, A., Neggers, J., Hild, F., Roux, S. (2019). Complete
mechanical regularization applied to digital image and volume correlation.
*CMAME* 355, 27–43.

[Moisan11]  Moisan, L. (2011). Periodic plus smooth image decomposition.
*J. Math. Imaging Vis.* 39, 161–179.

[Padfield11]  Padfield, D. (2011). Masked object registration in the
Fourier domain. *IEEE Trans. Image Process.* 21(5), 2706–2718.

[Palanca15]  Palanca, M., Tozzi, G., Cristofolini, L., Viceconti, M.,
Dall'Ara, E. (2015). Three-dimensional local measurements of bone strain
and displacement: comparison of three DVC approaches. *J. Biomech. Eng.*
137, 071006.

[Palanca16]  Palanca, M., Brugo, T.M., Cristofolini, L. (2016). Strain
uncertainties from two DVC approaches in prophylactically augmented
vertebrae. *J. Mech. Behav. Biomed. Mater.* 67, 165–175.

[Pan10]  Pan, B., Lu, Z., Xie, H. (2010). Mean intensity gradient: An
effective global parameter for quality assessment of speckle patterns
used in digital image correlation. *Opt. Lasers Eng.* 48, 469–477.

[Pan10b]  Pan, B., Wu, D., Wang, Z. (2010). Equivalence of digital image
correlation criteria for pattern matching. *Appl. Opt.* 49(28), 5501–5509.

[Pan13]  Pan, B. (2013). Bias error reduction of digital image correlation
using Gaussian pre-filtering. *Opt. Lasers Eng.* 51(10), 1161–1167.

[Pan14]  Pan, B. (2014). An evaluation of convergence criteria for digital
image correlation using inverse compositional Gauss-Newton algorithm.
*Strain* 50, 48–56.

[Smith02]  Smith, T.S., Bay, B.K., Rashid, M.M. (2002). Digital volume
correlation including rotational degrees of freedom. *Exp. Mech.* 42(3),
272–278.

[Stamati20]  Stamati, O., Andò, E., Roubin, E., et al. (2020). spam:
Software for Practical Analysis of Materials. *JOSS* 5(51), 2286.

[Sun18]  Sun, Y. (2018). Computation of the normalized cross-correlation
by fast Fourier transform. *PLoS ONE* 13(9), e0203434.

[Thielicke21]  Thielicke, W., Sonntag, R. (2021). Particle Image Velocimetry
for MATLAB: Accuracy and enhanced algorithms in PIVlab. *J. Open Res. Soft.*
9, e334.

[Tudisco17]  Tudisco, E., Andò, E., Cailletaud, R., Hall, S.A. (2017).
TomoWarp2: A local digital volume correlation code. *SoftwareX* 6, 267–270.

[Wang16]  Wang, T., Jiang, Z., Kemao, Q., Huang, F., Liu, Y., Liu, Z.
(2016). GPU Accelerated Digital Volume Correlation. *Exp. Mech.* 56,
297–309.

[Westerweel97]  Westerweel, J., Dabiri, D., Gharib, M. (1997). The effect
of a discrete window offset on the accuracy of cross-correlation analysis
of digital PIV recordings. *Exp. Fluids* 23, 20–28.

[Westerweel05]  Westerweel, J., Scarano, F. (2005). Universal outlier
detection for PIV data. *Exp. Fluids* 39(6), 1096–1100.

[Yang21]  Yang, J., Huang, J., Jiang, Z., Dong, S., Tang, L., Liu, Y.,
Liu, Z., Zhou, L. (2021). 3D SIFT aided path independent digital volume
correlation and its GPU acceleration. *Opt. Lasers Eng.* 136, 106323.
