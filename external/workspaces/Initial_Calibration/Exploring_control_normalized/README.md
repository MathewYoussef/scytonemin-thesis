# Exploring Control-Normalized Concentrations

## Motivation
Our replicate-level analyses (chromatogram and DAD) showed no statistically robust relationship between dose and concentration—even after robust medians, correlation checks, and ANOVA/Kruskal tests. The control cohort (UVA = UVB = 0) therefore becomes an important anchor: It represents baseline biological/method variability and may serve as a normalization reference.

This folder collects experiments where we normalize concentrations relative to that control baseline and re-run the dose-separation statistics.

## Normalization Approaches
Each subfolder explores one of five baseline-adjustment strategies:

1. `delta_from_control/` – Subtract the control mean/median from each dose (additive baseline correction).
2. `ratio_to_control/` – Divide dose values by the control mean/median (multiplicative baseline correction).
3. `percent_change_from_control/` – Convert delta to percent change relative to the control baseline.
4. `zscore_using_control/` – Standardize using control mean and standard deviation (Z-scores).
5. `control_summary/` – Document control statistics themselves (for reference and any additional methods that rely solely on control parameters).

Each approach gets its own README describing assumptions, formulas, and testing workflow.

## Planned Workflow
1. Populate each subfolder with a README (method overview, equations, planned scripts, and expected outputs).
2. Implement scripts to compute control-normalized chromatogram and DAD concentrations for that method.
3. Re-run robust summaries and statistical tests (ANOVA/Kruskal) on the normalized values.
4. Compare results against the original unnormalized analysis; document whether normalization improves dose separation.
5. Integrate findings back into the main documentation if any method proves useful.

## Notes
- Source data will remain the same (`treatments_concentration_raw.csv`, `treatments_corrected_amounts.csv`, `DAD_derived_concentrations_corrected.csv`); normalization scripts produce derived tables in each subfolder.
- Keep citations and formulas clear in each README (especially when definitions differ, e.g., median vs. mean control baselines).
- All plots/statistics should explicitly state the normalization applied to avoid confusion.
- Current artifacts:
  - `control_summary/control_baselines.json`, `chromatogram_control_summary.csv`, `dad_control_summary.csv`
  - `delta_from_control/chromatogram_delta.csv`, `dad_delta.csv`, plus ANOVA/Kruskal outputs `_dose_stats_delta.csv`
  - `ratio_to_control/chromatogram_ratio.csv`, `dad_ratio.csv`, plus `_dose_stats_ratio.csv`
  - `percent_change_from_control/chromatogram_percent_change.csv`, `dad_percent_change.csv`, plus `_dose_stats_pct.csv`
  - `zscore_using_control/chromatogram_zscores.csv`, `dad_zscores.csv`, plus `_dose_stats_zscore.csv`
- Statistical outcome: all normalizations reproduce the same p-values as the raw analysis (best-case ANOVA p ≈ 0.044, but corresponding Kruskal p ≈ 0.17). No control-based transform produced a statistically significant dose–concentration relationship.
