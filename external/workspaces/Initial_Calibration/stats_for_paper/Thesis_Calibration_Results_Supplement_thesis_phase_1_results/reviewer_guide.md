# Reviewer Guide: Thesis Calibration Results Supplement

All paths are relative to the `Thesis_Calibration_Results_Supplement/` folder. The packet groups every file cited in the Results narrative so a reviewer can recreate calibrations, regressions, permutation tests, and figures.

## Primary Reports & Summaries
- `calibration_summary.md` – Tabulates the DAD and chromatogram calibration coefficients (slope, intercept, SE, R², max |relative residual|, df) that anchor all concentration conversions cited in the Results section.
- `results_appendix.md` – Consolidates every statistical output referenced in the main text, including regressions, ANOVA tables, permutation tests, ridge/PLS performance, and diagnostic metrics.
- `single_factor_summary.md` – Lists single-dose OLS regressions and dose–response correlations that motivate the descriptive trend statements for UVA/UVB.
- `two_factor_summary.md` – Aggregates classical/robust/perm ANOVA outputs and interaction models for chromatogram and DAD endpoints.
- `diagnostics_and_sensitivity_summary.md` – Provides normality tests, leverage/cook’s distance checks, variance inflation factors, outlier-removal scenarios, and UVA–UVB correlation diagnostics that support the design discussion.
- `STATISTICAL_JOURNEY.md` – Narrative log of the analysis workflow, outlining versions, sensitivity checks, and rationale for each modeling layer.
- `regression_summary.csv` – Machine-readable version of the single-factor OLS fits (slopes, intercepts, R²) used in figures/tables.
- `missing_info.md` – Documents unavailable metadata (e.g., blank chromatograms) explaining the absence of LOD/LOQ estimates.

## Calibration & Standards
- `calibration_total.json`, `calibration_oxidized.json`, `calibration_reduced.json` – Raw JSON exports from the calibration routine, including weighting scheme, slope/intercept with SEs, residual diagnostics, and blank statistics.
- `standards_fitted_total.csv`, `standards_fitted_oxidized.csv`, `standards_fitted_reduced.csv` – Standard curve points (AUC vs. known concentrations) used to generate the chromatogram calibration plots.
- `dad_calibration_totals.csv` – DAD standards table with AUC/concentration pairs for re-fitting or spot-checking the reported DAD calibrations.

## Dose & Response Data
- `Chromatogram_derived_concentrations.csv`, `DAD_derived_concentrations_corrected.csv` – Full quantified concentration tables (mg·mL⁻¹ and mg·gDW⁻¹) with dose assignments for both assays.
- `Combined_Scytonemin_Concentrations.csv`, `chrom_dad_scytonemin_all_variants.csv` – Merge views combining chromatogram and DAD outputs across normalization schemes.
- `uva_uvb_ratio_concentration_data_total.csv`, `uva_uvb_ratio_concentration_data_oxidized.csv`, `uva_uvb_ratio_concentration_data_reduced.csv` – Ratio-based dose configurations supplying the N=25 analyses described in the text.
- `uva_uvb_correlations.csv` – Pearson correlations (and counts) for UVA vs. UVB across assay variants, supporting the collinearity discussion.
- `raw_total_scytonemin.csv`, `raw_oxidized_scytonemin.csv`, `raw_reduced_scytonemin.csv` – Source peak-area tables prior to normalization, enabling independent recalculation.

## ANOVA, OLS, and Permutation Outputs
- `chromatogram_two_way_anova*.csv`, `dad_two_way_anova*.csv` – Full suite of classical, delta, percent, ratio, robust, and z-score ANOVA tables for both assays; includes sum of squares, F, p-values, and η².
- `chromatogram_two_way_anova_permutation.csv`, `dad_two_way_anova_permutation.csv` – Simple permutation ANOVA results (2000 draws) matching the exploratory interaction claims.
- `freedman_lane_interaction.csv` – Structure-preserving permutation (Freedman–Lane) outcomes demonstrating that nominal interactions do not survive proper null resampling.
- `chromatogram_regression_coeffs_delta.csv`, `chromatogram_regression_coeffs_zscore.csv`, `dad_regression_coeffs_delta.csv` – OLS coefficient tables with p-values for delta/z-score formulations, aligning with effect-size statements.
- `anova_residual_diagnostics.csv` – Residual distribution checks (Shapiro, Jarque–Bera, leverage, Cook’s D) for factor models.
- `regression_vif.csv` – Variance inflation factors illustrating the UVA/UVB aliasing and interaction collinearity.
- `outlier_sensitivity.csv` – Interaction F/p trajectories under drop-one/drop-set scenarios, supporting the outlier robustness discussion.

## Predictive Modeling Artifacts
- `chromatogram_ridge_results.csv`, `dad_ridge_results.csv` – Ridge regression coefficients and R² values, matching the low predictive power reported.
- `ridge_bootstrap_summary.csv` – Bootstrap coefficient summaries (means, CIs) demonstrating UVB stability and interaction uncertainty.
- `chromatogram_pls_results.csv`, `dad_pls_results.csv` – Partial least squares loadings and cross-validated R², documenting cross-validated under-performance.

## Figure Captions & Plotting Assets
- `plots_test_2/figure_captions.md`, `plots_test_3/figure_captions.md`, `_plots_test_4/figure_captions.md` – Caption text and statistical references for each figure release.
- `plots_test_1/`, `plots_test_2/`, `plots_test_3/`, `_plots_test_4/` – Full figure directories (PNG/SVG outputs, data caches, and styles) allowing a reviewer to check rendered plots against the manuscript.

## Reproduction / Plotting Scripts
- `generate_plots_test_2.py` – Batch driver for the `_test_2` figure set (calibrations, dose structure, regressions, permutations, predictive checks).
- `plots_test_3/generate_figures.py`, `plots_test_3/make_fig09.py`, `plots_test_3/provenance_template.yml` – Modular plot generation plus provenance template for final supplements.
- `_plots_test_4/generate_figures.py` – Latest iteration of the plotting harness (mirrors figure captions in `_plots_test_4`).
- `plots_test_1/fig01_calibration/make_fig01.py` through `fig08_cv_performance/make_fig08.py` – Per-figure scripts for the initial figure suite (calibration curves, dose layout, single-factor panels, ANOVA marginal means, permutation comparisons, CV performance).

## Additional Notes for Reviewers
- All CSV/JSON files retain original column names and units from the analysis pipeline; no post-processing has been applied during packaging.
- Directory copies exclude embedded virtual environments (`venv`/`.venv`) to keep the supplement lightweight; recreate them with the provided requirements if reproducibility of the Python environment is necessary.
- To re-run any script, activate an environment with the dependencies listed in the respective project README files located inside each `plots_test_*` directory.
