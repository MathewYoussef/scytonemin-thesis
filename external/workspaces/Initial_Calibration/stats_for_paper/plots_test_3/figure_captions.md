# Figure Captions — SeaBase/Matplotlib Output (plots_test_3)

**Figure 1. Calibration integrity (Chromatogram & DAD).**  
Concentration (mg·mL⁻¹) versus blank-corrected AUC for chromatogram (panels 1A–C) and DAD (1D–F) standards. Lines are the validated 1/x-weighted fits (chromatogram) or reported DAD calibration parameters; shaded bands show 95 % CIs. Insets display relative residuals with reference bands at ±0.10 and ±0.20. Annotated slope, intercept, SEs, R², max |relative residual| and df agree with `calibration_summary.md` and `calibration_*.json`.

**Figure 2. Dose structure and collinearity diagnostics.**  
UVA vs UVB scatter (panel 2A) for all n = 30 samples, annotated with Pearson r = 0.9498 and the VIFs (UVA ≈ 37.9; UVB ≈ 10.4; UVA×UVB ≈ 25.1). Panels 2B–C convert the unique dose levels into replicate-count bars (mW·cm⁻²), highlighting the non-crossed design that motivates robust/permutation inference.

**Figure 3. Chromatogram single-factor trends.**  
Outcome (mg·gDW⁻¹) versus UVB (panels 3A–C: total, oxidized, reduced) and versus UVA (3D–F) for n = 30 samples. Points show sample means; black lines are OLS fits with 95 % CIs. Embedded annotations list slope ± SE, R², and n (e.g., UVB→total slope 0.5607 mg·gDW⁻¹·(mW·cm⁻²)⁻¹, R² = 0.2445).

**Figure 4. DAD single-factor trends (dry-weight normalised).**  
DAD total concentrations converted to mg·gDW⁻¹ are plotted against UVB (4A) and UVA (4B). OLS fits with 95 % CIs yield UVB slope 1.034 ± 0.384 mg·gDW⁻¹·(mW·cm⁻²)⁻¹, R² = 0.205, and UVA slope 0.221 ± 0.100 mg·gDW⁻¹·(mW·cm⁻²)⁻¹, R² = 0.149 (n = 30). Underlying mg·mL⁻¹ calibration is validated in Fig. 1; here the dry-weight scale aligns the DAD trends with the chromatogram analyses.

**Figure 5. Estimated marginal means (chromatogram endpoints — total form).**  
EMMs vs UVA (5A, 5C) and UVB (5B, 5D) for total z-score concentration and total raw amount, averaging over the other factor. Each panel annotates the classical UVA and UVB main-effect p-values (e.g., z concentration p_UVA ≈ 0.0288, p_UVB ≈ 0.066); panels 5C–D additionally remind that the classical interaction sits near p ≈ 0.10–0.11.

**Figure 6. Interaction p-values across methods.**  
Forest-style −log₁₀(p) comparison of Classical ANOVA, HC3, and rank-transform tests for raw, Δ, and z chromatogram metrics. A dashed vertical line marks p = 0.05; panels highlight nominal HC3/Rank positives that motivated permutation checks.

**Figure 7. Freedman–Lane permutation tests (2 000 permutations).**  
Horizontal bars show the Freedman–Lane interaction p-values reported in `freedman_lane_interaction.csv` for Δ concentration, Δ amount, z concentration, z amount, and DAD total mg·gDW⁻¹; accompanying labels list the observed F statistics (e.g., F = 3.09, p = 0.727). A dashed vertical line at p = 0.05 reinforces that all interactions remain non-significant.

**Figure 8. Dose-only predictive performance is weak.**  
Panels 8A–B: bar charts of ridge R² and PLS cross-validated R² for chromatogram Δ/z amount and DAD total (mg·gDW⁻¹) endpoints. Panels 8C–D: observed vs 5-fold CV ridge predictions (chromatogram Δ amount; DAD total mg·gDW⁻¹) with CV R² annotations (≈ 0.16 and ≈ 0.30 respectively), underscoring the limited explanatory power of dose alone.

**Figure 9. Bootstrap stability of Δ-amount ridge coefficients.**  
Synthetic density plots derived from `ridge_bootstrap_summary.csv` show β_UVB (mean ≈ 1.347; 95 % CI [0.0549, 2.623]) and the interaction coefficient (mean ≈ −0.361; 95 % CI [−0.848, 0.199]). The UVB effect remains positive whereas the interaction continues to straddle zero.

**Figure 10. Outlier sensitivity trajectory (Δ concentration interaction).**  
F statistic and p-value versus number of high studentized residual removals (0 → 2). Baseline F = 3.087 (p = 0.01316) rises modestly (F ≈ 3.22–3.32; p ≈ 0.010–0.0088); caption reiterates that Freedman–Lane results stay non-significant.

**Figure S1. Regression diagnostics.**  
Residual vs fitted, normal Q–Q, and leverage vs residual² plots for Chrom Δ amount and DAD total mg·gDW⁻¹ OLS fits. Threshold lines mark studentized residual, leverage (2p/n), and Cook’s distance (4/(n−p)); no observation exceeds prespecified influence cutoffs.

**Figure S7. Cross-assay alignment sanity check.**  
Chromatogram totals converted to mg·mL⁻¹ via the median estimated extraction volume (≈ 0.512 mL) plotted against DAD total mg·mL⁻¹. Includes 1:1 reference line, OLS trend, and Pearson r annotation, clarifying that assays corroborate but are not identical.
