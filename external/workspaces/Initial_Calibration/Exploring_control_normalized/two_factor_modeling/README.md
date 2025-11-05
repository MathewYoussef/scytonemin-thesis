# Two-Factor UVA/UVB Modeling

## Objective
Evaluate whether modeling UVA and UVB together (additive and interaction effects) reveals dose–concentration relationships that were missed in single-factor tests.

## Approach
- Fit two-way ANOVA: `concentration ~ UVA + UVB + UVA:UVB`.
- Fit multiple regression (continuous predictors) for both raw and control-normalized metrics.
- Run analyses on chromatogram and DAD datasets (mg/mL, mg/gDW; delta/ratio/etc. as needed).

## Planned Workflow
1. Prepare tidy datasets with UVA, UVB, and target metrics (raw and normalized).
2. Fit models with/without interaction; record F-statistics, p-values, partial eta-squared.
3. Compare models across normalization methods; highlight any statistically significant main or interaction effects.
4. Archive results (tables, plots, model diagnostics) here and summarize findings in the main documentation.

## Results Summary (August 2025)
- Chromatogram raw metrics: no UVA, UVB, or interaction terms crossed α = 0.05 (all p ≥ 0.13; `chromatogram_two_way_anova.csv`), suggesting no clear dose dependence in the unscaled data.
- Chromatogram normalized metrics: delta and z-score endpoints show UVA×UVB interactions (p ≈ 0.013–0.040) with moderate η² (0.33–0.43) and marginal UVA/UVB main effects; ratio and percent variants echo weaker interaction signals (p ≈ 0.07–0.09). Continuous regressions highlight positive UVB coefficients for amount metrics (p ≈ 0.020–0.084) but with low R² (≤ 0.16).
- DAD metrics: only predicted total per gDW exhibits a borderline UVA×UVB interaction (p ≈ 0.058, η² ≈ 0.71), replicated across delta/ratio/percent/z-score tables; other DAD responses remain non-significant in both ANOVA and regression outputs.
- Robust HC3 ANOVA (`chromatogram_two_way_anova_robust.csv:2-24`, `dad_two_way_anova_robust.csv:2-8`) keeps UVA×UVB effects significant for chromatogram raw (p ≈ 0.022–0.023) and normalized metrics (p ≈ 0.002–0.004) while attenuating DAD interactions (p ≈ 0.25), indicating that heteroskedasticity/outliers are not solely responsible for the chromatogram signal.
- Rank-transformed two-way ANOVA (`chromatogram_two_way_anova_rank.csv:2-24`, `dad_two_way_anova_rank.csv:2-8`) likewise retains interaction significance for normalized chromatogram metrics (p ≈ 0.003–0.011) and yields a marginal raw amount effect (p ≈ 0.047) but no DAD signal, suggesting the interaction hints persist under a non-parametric treatment despite limited power.
- Complementary modeling (`run_complementary_models.py`): Ridge regression keeps the positive UVB effect but still yields low explanatory power (chromatogram delta amount R² ≈ 0.16 with α = 1; `chromatogram_ridge_results.csv:11-13`), and DAD ridge fits reach R² ≈ 0.30 yet rely on strong negative interaction shrinkage (`dad_ridge_results.csv:2-7`). PLS cross-validated R² remains ≤ 0.12 for chromatogram metrics and ≈ 0.02 for DAD (`chromatogram_pls_results.csv:2-37`, `dad_pls_results.csv:1-4`). Permutation-based ANOVA confirms earlier signals—normalized chromatogram interactions retain p_perm ≈ 0.009–0.038 and DAD totals stay marginal at p_perm ≈ 0.062 (`chromatogram_two_way_anova_permutation.csv:1-25`, `dad_two_way_anova_permutation.csv:1-8`).
- Resampling sensitivity (`run_resampling_sensitivity.py`): Freedman–Lane permutations produce high interaction p-values (p ≥ 0.71; `freedman_lane_interaction.csv:1-5`), implying the factorial signal weakens when the reduced-model structure is preserved. Bootstrap ridge intervals show UVB effects remain positive for chromatogram amount metrics (delta/z-score 95% CI excludes zero; `ridge_bootstrap_summary.csv:3-11`) but interaction coefficients straddle zero across all endpoints.
- Outlier checks (`run_outlier_sensitivity.py`): Removing high studentized residuals (|r| ≥ 3; `outlier_sensitivity.csv:0-14`) leaves UVA×UVB p-values below 0.05 and often strengthens them (e.g., z_amount_mg_per_gDW drops from p=0.040 to 0.010), indicating the interaction hints are not driven by a single extreme observation.
- Current interpretation: normalized chromatogram metrics hint at a UVA×UVB interaction across classical, HC3, rank, and simple permutation ANOVAs (`chromatogram_two_way_anova_delta.csv:2-16`, `chromatogram_two_way_anova_permutation.csv:1-25`), but Freedman–Lane permutations no longer reject the null (p ≥ 0.71; `freedman_lane_interaction.csv:1-5`) and DAD responses stay null or marginal (`dad_two_way_anova_permutation.csv:1-8`). Diagnostics highlight the principle caveats—non-normal residuals and leverage (`anova_residual_diagnostics.csv:1-8`), extreme UVA/UVB collinearity (`uva_uvb_correlations.csv:1-5`, `regression_vif.csv:1-24`), and low predictive power in ridge/PLS (`chromatogram_ridge_results.csv:11-13`, `chromatogram_pls_results.csv:2-37`, `dad_ridge_results.csv:2-7`). Conclusion: the UVA×UVB signal in normalized chromatogram data remains exploratory and sensitive to modeling choices.
- Diagnostics (see `anova_residual_diagnostics.csv:1-8`): Shapiro–Wilk and Jarque–Bera tests flag non-normal residuals for all chromatogram models (p ≤ 4e-5) with studentized outliers up to |4.8|, while DAD residuals pass normality (p ≈ 0.59) and show moderate outliers (|2.3|); leverage remains below the 2p/n threshold for every fit.
- Collinearity (see `regression_vif.csv:1-24`, `uva_uvb_correlations.csv:1-5`): UVA and UVB doses are highly correlated (ρ ≈ 0.95) and produce severe VIFs (≈38 for UVA, ≈10 for UVB, ≈25 for the interaction term) across all regression variants, limiting interpretability of main-effect coefficients.
- Interpretation: the two-factor analyses surface interaction hints in select normalized chromatogram measures, yet effects are inconsistent, rely on limited replicates, and violate normality assumptions with substantial UVA/UVB collinearity. Even with HC3-robust and rank-based ANOVA, the results remain exploratory; treat them as leads rather than definitive UV dose–concentration relationships.
- Key limitations to revisit before declaring success: (1) severe UVA/UVB multicollinearity that destabilizes coefficient estimates, (2) non-normal chromatogram residuals with high-leverage observations, and (3) small cell counts that make interaction terms rank-deficient (see statsmodels warnings during robust/rank analyses).
- Next steps (analysis-focused, still desk-based):
  - Diagnose the divergence between simple and Freedman–Lane permutations by reviewing dose-level balance and replicate counts per UVA/UVB cell; quantify how cell sparsity affects the factorial F-statistic.
  - Explore variance-stabilizing transforms or generalized linear/hierarchical variants that can accommodate the heavy-tailed chromatogram residuals while preserving two-factor structure.
- Deferred work (requires future data collection or chamber redesign): addressing UVA/UVB collinearity via new dose grids or orthogonal contrasts; not feasible under current laboratory constraints.

## Notes
- Use the existing control-normalization outputs as needed.
- Consider additional covariates (batch, dry mass, chamber position) if two-factor modeling still fails.
- Document model assumptions (normality, equal variance) via residual plots and tests before drawing conclusions.
