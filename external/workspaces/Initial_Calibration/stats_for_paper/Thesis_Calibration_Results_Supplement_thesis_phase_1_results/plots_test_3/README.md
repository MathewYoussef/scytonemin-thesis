# Plots Test 3 — Production Checklist

This folder scaffolds the figure suite requested for the updated Results narrative.  
Each subdirectory mirrors the deliverables so plotting scripts and outputs stay organised.

## Core Figures

1. **fig01_calibration/** → save files as `fig01_calibration_{assay}_{analyte}.{png,pdf}`
   - 1A–C Chromatogram (total/oxidised/reduced) AUC vs concentration with 1/x-weighted fit, 95% CI band, and residual inset (relative residual vs fitted concentration with ±0.1/±0.2 guides).
   - 1D–F DAD (total/oxidised/reduced) with the same structure.
   - Annotate slope, intercept, slope SE, intercept SE, R², max |relative residual|, df on each panel.

2. **fig02_dose_structure/** → `fig02_dose_structure.{png,pdf}`
   - 2A UVA vs UVB scatter for n = 30 (mW·cm⁻²), show Pearson r = 0.9498.
   - 2B–C Marginal UVA/UVB histograms (or rugs); include an info box with VIFs (UVA ≈ 37.9, UVB ≈ 10.4, UVA×UVB ≈ 25.1).

3. **fig03_single_factor_chrom/** → `fig03_single_factor_chrom_{analyte}_{dose}.{png,pdf}`
   - 3A–C Chromatogram mg·gDW⁻¹ vs UVB (total, oxidised, reduced); OLS line + 95% CI, annotate slope ± SE, R², n = 30.
   - 3D–F Chromatogram mg·gDW⁻¹ vs UVA (same outputs, shared axis limits).

4. **fig04_single_factor_dad/** → `fig04_single_factor_dad_total_{dose}.{png,pdf}`
   - 4A DAD total mg·mL⁻¹ vs UVB; annotate slope 0.0829 ± 0.0286, R² = 0.2303, n = 30.
   - 4B DAD total mg·mL⁻¹ vs UVA; annotate slope 0.0194 ± 0.00733, R² = 0.2008, n = 30.
   - Caption reminder: “Scale verified from DAD_derived_concentrations_corrected.csv”.

5. **fig05_emm/** → `fig05_emm_{endpoint}.{png,pdf}`
   - 5A–B Chromatogram z-score concentration estimated marginal means (EMMs) vs UVA/UVB with 95% ribbons; note classical UVA main-effect p = 0.0288.
   - 5C–D Chromatogram raw amount EMMs; mention “no classical interaction; raw interaction p ≈ 0.10–0.11”.

6. **fig06_pvals_methods/** → `fig06_pvals_methods_{endpoint}.{png,pdf}`
   - Forest-style plots of interaction p-values (Classical, HC3, Rank) on a −log₁₀ scale for raw, Δ, and z endpoints.  
     Highlight nominal p < 0.05 for HC3/Rank on raw metrics, and reference Freedman–Lane in captions.

7. **fig07_freedman_lane/** → `fig07_freedman_lane_{endpoint}.{png,pdf}`
   - Freedman–Lane permutation histograms (2,000 permutations) for Δ concentration, Δ amount, z concentration, z amount, DAD total raw.  
     Overlay observed F and annotate p_FL (0.7271, 0.7136, 0.7406, 0.7366, 0.7871).

8. **fig08_cv_performance/** → `fig08_cv_performance.{png,pdf}`
   - Ridge and PLS cross-validated R² bars for key endpoints (chromatogram Δ-amount, z-amount, DAD total raw).
   - Actual vs cross-validated prediction scatter plots (one chromatogram endpoint, one DAD) with 45° line and CV R² labels (5-fold).

9. **fig09_bootstrap_coeffs/** → `fig09_bootstrap_coeffs.{png,pdf}`
   - Bootstrap distributions for β_UVB (Δ-amount; mean 1.347, 95% CI [0.0549, 2.623]) and the interaction coefficient (mean −0.3613, 95% CI [−0.848, 0.1987]); show CI ticks and vertical zero line.

10. **fig10_outlier_path/** → `fig10_outlier_path.{png,pdf}`
    - Interaction F and p trajectories as high-residual observations are removed (0 → k).  
      Annotate baseline F = 3.087, p = 0.01316 and tightened range F ≈ 3.22–3.32, p ≈ 0.010–0.0088; caption reminds that Freedman–Lane remains non-significant.

## Supplemental Figures

- **supplemental/figS01_diagnostics/** → residual vs fitted, Q–Q, leverage/Cook’s plots for representative endpoints (n = 30, no flagged leverage points).
- **supplemental/figS07_cross_assay_alignment/** → Chromatogram mg·gDW⁻¹ converted to mg·mL⁻¹ vs DAD total mg·mL⁻¹ with OLS and 45° reference line.

## Metadata Checklist

- Save both `.png` (publication) and `.pdf` (vector) outputs once generated.
- Each plotting script should emit a sidecar `.yml` capturing data sources (file + SHA256), script version, timestamp, and key statistics used in annotations.
- Maintain consistent colour/line schemes by outcome family and synchronise axis limits across comparable panels.
- Whenever highlighting HC3/Rank positives, reiterate the Freedman–Lane outcome in-caption to prevent over-interpretation.

