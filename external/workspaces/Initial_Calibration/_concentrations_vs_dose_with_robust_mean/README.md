# Concentrations vs Dose with Robust Mean Aggregation

## Objective
Evaluate whether dose-level robust averages (median or Huber) reveal clearer relationships between UVA/UVB dose and scytonemin concentrations than replicate-level scatter plots.

## Scope
- **Datasets**: chromatogram-derived concentrations (`treatments_concentration_raw.csv`, `treatments_corrected_amounts.csv`) and DAD-derived concentrations (`DAD_derived_concentrations_corrected.csv`).
- **Metrics**: raw concentrations (mg/mL) and biomass-normalized amounts (mg/gDW).
- **Dose descriptors**: UVA intensity, UVB intensity, UVA/UVB ratio, and UVA×UVB.

## Planned Workflow
1. Compute robust summaries per dose (e.g., median, Huber location) for chromatogram concentrations/amounts; produce dose vs concentration tables/plots.
2. Repeat for DAD-derived concentrations/amounts.
3. Compare robust-mean results against replicate-level correlations to determine whether aggregation clarifies trends.
4. Document findings here and cross-reference conclusions in `docs/`.
- Run one-way ANOVA and Kruskal-Wallis tests to quantify whether dose groups show statistically significant separation.

## Notes
- Maintain outputs (summary CSVs, plots) in this folder for traceability.
- Record methodology choices (robust statistic, tuning parameters) alongside results.
- If robust averaging improves interpretability, recommend aggregating doses in future analyses and note the implication in the final report.
- Current artifacts:
  - `chromatogram_robust_summary_mg_mL.csv` / `chromatogram_robust_summary_mg_per_gDW.csv`
  - `dad_robust_summary_mg_mL.csv` / `dad_robust_summary_mg_per_gDW.csv`
  - `chromatogram_dose_stats.csv` / `chromatogram_dose_group_stats.csv`
  - `dad_dose_stats.csv` / `dad_dose_group_stats.csv`
  - (Plots skipped in this environment because matplotlib is unavailable; rerun the scripts on a plotting-enabled machine to generate PNGs.)
- Initial statistics (`chromatogram_dose_stats.csv`, `dad_dose_stats.csv`) show p-values ≥ 0.04 even after robust aggregation (e.g., total mg/gDW ANOVA p = 0.044, but Kruskal p = 0.17), so no dosing metric clears both parametric and non-parametric thresholds; there is no evidence of a dose–concentration relationship. The six predefined dose levels with five replicates each provide 30 observations per form, so the tests exhaustively cover the dataset.
