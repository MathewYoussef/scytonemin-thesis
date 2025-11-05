# Core Figures

**Fig. 1 — Calibration integrity (Chromatogram & DAD)**  
Calibrations parameterize concentration as a function of blank-corrected AUC. Each panel shows standards, the validated fit with 95 % CI, and a residual inset with ±0.10/±0.20 anchors; annotated slopes/intercepts/SEs/R²/max |relative residual|/df match the repo calibration summaries.

**Fig. 2 — Dose structure & collinearity**  
UVA vs UVB scatter (r = 0.9498) plus marginal histograms illustrate the strong covariance that inflates VIFs (UVA ≈ 37.9, UVB ≈ 10.4, UVA×UVB ≈ 25.1), motivating collinearity-tolerant inference.

**Fig. 3 — Chromatogram single-factor trends (mg·gDW⁻¹)**  
OLS fits with 95 % CI for total/oxidized/reduced amounts versus UVB (3A–C) and UVA (3D–F); annotated slopes/R²/n show modest monotone patterns (e.g., UVB→total slope ≈ 0.5607, R² ≈ 0.2445).

**Fig. 4 — DAD single-factor trends (mg·gDW⁻¹)**  
Dry-weight–normalised DAD totals versus UVB/UVA with OLS fits (slopes ≈ 1.034 ± 0.384 and 0.221 ± 0.100; R² = 0.205/0.149; n = 30), providing the dose–response view in biologically relevant units; the mg·mL⁻¹ calibration remains documented in Fig. 1 and the text.

**Fig. 5 — Estimated marginal means (chromatogram)**  
EMMs for z-score concentration and raw amount versus UVA/UVB. The UVA main effect in z concentration (classical p = 0.0288) and the non-replicating raw interaction (classical p≈0.10–0.11) are called out alongside the borderline DAD Δ total mg·gDW⁻¹ p≈0.0575.

**Fig. 6 — Interaction p-values across estimation methods**  
−log₁₀(p) bars compare classical, HC3, and rank ANOVA interaction tests for raw/Δ/z endpoints; dashed line marks p = 0.05, highlighting that HC3/rank nominal hits appear mainly in raw chromatogram metrics.

**Fig. 7 — Permutation tests for UVA×UVB interaction**  
Bar chart comparing simple-response shuffle p-values with Freedman–Lane p-values for each dataset/variant/metric combination; reproduces the repository values (p_FL ≈ 0.71–0.79) to highlight that structured permutations eliminate the nominal interactions.

**Fig. 8 — Predictive value is low**  
Ridge and PLS cross-validated R² remain modest (< 0.30), and observed-vs-predicted scatters for chrom Δ amount and DAD total mg·gDW⁻¹ show limited explanatory power, underscoring weak dose predictivity.

**Fig. 9 — Bootstrap stability of Δ amount coefficients**  
Mean ±95 % CI error bars from the ridge bootstrap summary: β_UVB remains positive (mean ≈ 1.347; 95 % CI [0.055, 2.623]) while the interaction spans zero (mean ≈ −0.361; 95 % CI [−0.848, 0.199]).

**Fig. 10 — Outlier sensitivity (Δ concentration interaction)**  
Interaction F and p traced as high studentized residuals are removed; nominal p-values tighten (≈ 0.013→0.009), but interactions remain non-significant under Freedman–Lane.

# Supplemental Figures

**Fig. S1 — Chromatogram raw concentration diagnostics**  
Residual vs fitted, Q–Q, and leverage plots for the raw concentration ANOVA model (n = 90); checks show non-normal residuals but no leverage outliers beyond preset thresholds.

**Fig. S2 — Chromatogram raw amount diagnostics**  
Same diagnostics for the raw mg·gDW⁻¹ model (n = 90), confirming leverage is within limits despite heavy-tailed residuals.

**Fig. S3 — Chromatogram Δ concentration diagnostics**  
Δ concentration ANOVA residual/Q–Q/leverage plots (n = 90) illustrate the deviations that prompted robust and permutation analyses.

**Fig. S4 — Chromatogram Δ amount diagnostics**  
Diagnostics for Δ amount mg·gDW⁻¹ (n = 90); leverage remains below the predetermined Cook’s distance threshold.

**Fig. S5 — Chromatogram z concentration diagnostics**  
Z-score concentration residual assessments (n = 90) highlight kurtosis relative to normality assumptions.

**Fig. S6 — DAD total mg·gDW⁻¹ diagnostics**  
DAD main-effect model diagnostics (n = 30), showing acceptable residual behavior with no influential leverage points.

**Fig. S7 — Cross-assay concentration alignment**  
Chromatogram-derived mg·mL⁻¹ (via dry-weight scaling) versus DAD mg·mL⁻¹ for matched samples; 45° line and Pearson r quantify cross-assay agreement while noting assays are complementary, not identical.
