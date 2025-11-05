# Plot Series Plan

## 1. Wavelength-wise variance heatmaps (before vs after denoising)
- **What:** 2 side-by-side heatmaps of per-wavelength variance across replicates for each treatment (rows = treatments/samples, cols = wavelength).
- **Expect:** Global variance shrinkage after denoising; a narrow preserved variance ridge only in the scytonemin band if biology is real (not noise).
- **Computation:** For each treatment t, compute \operatorname{Var}_k[R_{t,k}(\lambda)] and \operatorname{Var}_k[\widetilde{R}_{t,k}(\lambda)]. Clip color scale by robust quantiles (e.g., 5-95%) so the 370-382 nm structure is visible.
- **Figure callouts:** Add a vertical band (370-382 nm). Annotate percent variance reduction per treatment outside that band.

## 2. Variance ratio ribbon
- **What:** A single line plot of \text{VR}(\lambda) = \frac{\operatorname{Var}(R)}{\operatorname{Var}(\widetilde{R})}.
- **Expect:** VR >> 1 across most wavelengths; VR ~ 1 inside 370-382 nm (signal preserved).
- **Add:** Shaded CI via bootstrap over samples.

## 3. SAM (Spectral Angle Mapper) difference plots
- **What:** For each treatment, compute median spectrum pre/post denoise; reference a clean control (e.g., your best SNR white-normalized control).
- SAM(a,b) = \cos^{-1}\!\left(\frac{a \cdot b}{\lVert a \rVert \lVert b \rVert}\right)
- **Panels:**
  - SAM to control (raw) vs SAM to control (denoised) per treatment (bar or dot-line).
  - \Delta SAM = SAM_{den} - SAM_{raw} across wavelengths using a sliding window (e.g., 8-12 nm) to localize effects.
- **Expect:** \Delta SAM strongly negative (improved similarity) outside 370-382 nm; near-zero \Delta SAM inside the scytonemin band.

## 4. Band-restricted preservation index
- **What:** Two numbers per treatment:
  - Noise-collapse: mean VR outside [370, 382] nm.
  - Pigment-preservation: 1 - |\Delta SAM|_{370-382}.
- **Show:** Small table or lollipop chart; this becomes your "at-a-glance" preservation vs collapse summary.

## 5. ROI micro-panels (zoomed spectra overlays)
- **What:** Small multiples of median ± IQR spectra, zoomed to 350-400 nm, before/after.
- **Expect:** Same dip depth/shape around 370-382 nm; reduced spread elsewhere.

## 6. Treatment-wise effect sizes
- **What:** For each treatment, Cohen's d (or Cliff's \delta) for pre- vs post-denoise outside the ROI vs inside the ROI (two points per treatment).
- **Expect:** Large effect outside; negligible inside.

## Minimal how-to (metrics you can compute directly)
- Variance heatmaps: groupby(treatment) -> np.var over replicates per \lambda.
- VR line: elementwise ratio of those variances (clip to avoid div-by-zero).
- SAM: normalize spectra to unit length per sample; angle to control median (or robust reference).
- \Delta SAM windowed: compute SAM on rolling-mean spectra (window ~11 nm).
- Preservation index: PI = 1 - \text{median}(|\Delta SAM|) within 370-382 nm.

## Suggested figure captions (ready to paste)
- **Fig. A** — ΔSNR heatmap. Denoised spectra show subtle dB gains across the spectrum; annotations highlight the average improvement inside 370–382 nm.
- **Fig. B** — ΔSNR summary (outside vs inside ROI). Lollipop chart spells out the per-treatment dB gains so small differences are readable.
- **Fig. C** — Wavelength-wise variance heatmaps. Denoising collapses variance broadly while preserving a narrow high-fidelity band at ~370–382 nm (scytonemin). Vertical band shows the ROI.
- **Fig. D** — Variance ratio (raw/denoised) vs wavelength. Ratios ≫1 outside the scytonemin window indicate strong noise suppression; ratios ≈1 within 370–382 nm indicate pigment-specific preservation.
- **Fig. E** — SAM to reference, before vs after. Spectral angles shrink after denoising overall, with negligible change in the ROI, confirming biologically meaningful curvature is retained.

- **Fig. F** — Preservation vs Collapse indices. Treatments show high noise-collapse outside the ROI and high preservation inside, demonstrating target-feature integrity post-denoise.
- **Fig. G** — UV-A ROI overlays (median ± IQR). The characteristic dip remains intact post-denoise with reduced spread elsewhere.
- **Fig. H** — Treatment-wise effect sizes. Cohen’s d contrasts outside vs inside the ROI to confirm large off-band gains with minimal ROI drift.
