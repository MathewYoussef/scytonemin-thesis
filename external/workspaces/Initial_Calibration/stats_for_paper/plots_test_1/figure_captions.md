# Figure Captions

**Fig. 1 — Calibration integrity.** Calibrations parameterize concentration as a function of blank-corrected AUC. Points are standards; lines show the validated fits (1/x-weighted for chromatogram, DAD calibrations as stored). Insets plot relative residuals with ±0.1 and ±0.2 guides. Numerical annotations report slope, intercept, standard errors, R², maximum |relative residual|, and degrees of freedom.

**Fig. 2 — Dose structure and collinearity.** Observed UVA and UVB intensities (n=30). Panel A shows the scatter with Pearson r ≈ 0.95; Panels B and C count unique UVA/UVB dose levels. Variance inflation factors (UVA ≈ 37.9, UVB ≈ 10.4, interaction ≈ 25.1) highlight severe multicollinearity.

**Fig. 3 — Chromatogram single-factor trends.** Single-factor regressions for chromatogram total/oxidized/reduced (mg·gDW⁻¹) vs UVB (top) and UVA (bottom). Lines show OLS fits with 95% confidence bands; annotations give slope ± SE, R², and n = 30. Trends are monotone but modest.

**Fig. 4 — DAD single-factor trends.** Dry-weight–normalized DAD totals (mg·gDW⁻¹) versus UVB (left) and UVA (right). OLS slopes are 1.03 ± 0.38 and 0.221 ± 0.100 mg·gDW⁻¹·(mW·cm⁻²)⁻¹ with R² = 0.205 and 0.149 (n = 30). The volumetric (mg·mL⁻¹) calibration is shown in Fig. 1.

**Fig. 5 — Estimated marginal means (Chromatogram total endpoints).** Estimated marginal means (means ± SEM) for chromatogram total z-score concentration and total mg·gDW⁻¹ across UVA or UVB categories. Main-effect p-values from classical ANOVA are annotated on each panel.

**Fig. 6 — Interaction p-values across methods.** Comparison of classical, HC3, and rank-based interaction tests (−log₁₀ p). HC3 and rank show nominal p < 0.05 only for raw endpoints, motivating confirmatory permutation testing.

**Fig. 7 — Freedman–Lane permutation tests (2000 permutations).** Distribution of interaction F-statistics under Freedman–Lane permutations (categorical UVA/UVB, 2,000 draws). Observed F (vertical line) lies well inside the null, yielding p_FL ≈ 0.72–0.79 for all tested endpoints.

**Fig. 8 — Cross-validated predictive performance.** Five-fold cross-validation results for ridge (with interaction term) and two-component PLS on key endpoints. Bars show mean CV R² (≤ 0.16 chromatogram, ≈ 0.30 DAD). Scatter panels compare observed vs predicted (Chrom Δ amount) under ridge and PLS, underscoring weak predictive power.

**Fig. 9 — Ridge bootstrap coefficients.** Bootstrap coefficient distributions for Δ endpoints, reconstructed from the stored summary statistics. β_UVB is positive for Δ amount (mean 1.347; 95% CI [0.055, 2.623]), whereas interaction coefficients center near zero (Δ amount mean −0.361; 95% CI [−0.848, 0.199]).

**Fig. 10 — Outlier sensitivity path (Δ concentration).** Interaction F-statistic and p-value as high-residual samples are removed (baseline, drop 2B, drop 1J, drop 2B+1J). Classical p-values tighten modestly but remain above 0.008; Freedman–Lane inference stays non-significant.
