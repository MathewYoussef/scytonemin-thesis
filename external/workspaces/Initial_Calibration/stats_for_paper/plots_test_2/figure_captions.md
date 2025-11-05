# Figure Captions for `plots_test_2`

**Fig. 1 — Calibration integrity.** Scatterplots of chromatogram (top row) and DAD (bottom row) standards with validated fits expressed as concentration = β₀ + β₁·AUC. Shaded bands show 95 % confidence intervals; insets plot relative residuals with guides at ±0.1 and ±0.2, and annotations report slope, intercept, standard errors, R², max |relative residual|, and degrees of freedom. Chromatogram fits use the archived 1/x‑weighted regressions; DAD panels reproduce the validated slopes/intercepts (calibration_summary.md).

**Fig. 2 — Dose structure & collinearity.** UVA vs. UVB scatter for all n = 30 chromatogram samples (Pearson r = 0.9498) with annotated variance‑inflation factors (UVA 37.9; UVB 10.4; interaction 25.1) alongside histograms of the marginal UVA and UVB dose grids. The strong collinearity motivates robust and permutation-based inference.

**Fig. 3 — Chromatogram single-factor trends.** Outcome (mg·gDW⁻¹) vs. UVB (top row) and UVA (bottom row) for total, oxidized, and reduced endpoints. Lines show ordinary least-squares fits with 95 % confidence bands; annotations list slopes and R² (e.g., UVB→total slope 0.5607, R² = 0.2445; UVA→total slope 0.115, R² = 0.163; n = 30).

**Fig. 4 — DAD single-factor trends (dry-weight normalized).** DAD total concentration (mg·gDW⁻¹) vs. UVB and UVA with OLS fits and 95 % confidence intervals. Slopes (1.0337 ± 0.3841 for UVB; 0.2212 ± 0.0999 for UVA) and R² (0.2055, 0.1490) match `DAD_derived_concentrations_corrected.csv`; n = 30. Panel titles make clear that values are dry-weight normalized to mirror the chromatogram comparisons.

**Fig. 5 — Estimated marginal means (chromatogram).** Marginal mean profiles from two-way ANOVA (no interaction term) for z-score concentration (top) and raw amount (bottom) against UVA and UVB. The UVA main effect on z-score concentration (p = 0.0288) is annotated, whereas raw endpoints note the absence of classical interaction support (p ≈ 0.10–0.11).

**Fig. 6 — Interaction p-values across methods.** −log₁₀(p) for interaction terms under classical ANOVA, HC3 robust ANOVA, and rank-transformed ANOVA across raw, Δ, and z endpoints. The dashed line marks p = 0.05; HC3 and rank hits on raw endpoints are flagged to explain why permutation tests were pursued.

**Fig. 7 — Freedman–Lane permutation tests (2 000 permutations).** Null distributions of the interaction F-statistic for chromatogram Δ concentration, Δ amount, z concentration, z amount, and DAD total (mg·gDW⁻¹), preserving UVA/UVB main effects. Observed F (red dashed line) lies well within each null; p_FL values (0.713–0.787) are reported in-panel.

**Fig. 8 — Predictive value is low.** Bar plots of ridge regression R² and PLS cross-validated R² for key chromatogram and DAD endpoints, plus observed vs. fitted scatter plots (OLS proxies) for chromatogram Δ amount and DAD total mg·gDW⁻¹. Results emphasize that dose-only models explain little variance (e.g., ridge R² ≤ 0.30; PLS CV R² ≤ 0.12).

**Fig. 9 — Bootstrap stability of Δ amount coefficients.** Density plots for 2 000-bootstrap ridge coefficients of β_UVB (mean 1.347, 95 % CI [0.055, 2.623]) and the UVA×UVB interaction (mean −0.361, 95 % CI [−0.848, 0.199]). Vertical lines mark zero and the bootstrap confidence limits, illustrating that the interaction is not stably different from zero.

**Fig. 10 — Outlier sensitivity (Δ concentration).** Interaction F-statistic and p-value vs. number of removed high studentized residuals for the chromatogram Δ concentration model. While trimming slightly tightens nominal p-values (≈ 0.013→0.009), Freedman–Lane permutation tests remain non-significant, reinforcing the exploratory nature of the signal.
