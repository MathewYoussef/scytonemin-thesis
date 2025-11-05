# Stage A–C Status Summary

## Stage A — Calibration Diagnostics
- QC thresholds relaxed to R² ≥ 0.98 and max |relative residual| ≤ 0.15 per updated config.
- **total**: weighting `1/x`, intercept mode `free`; slope `7.21e-06`, intercept `-0.01599`. R² `0.9934`, max |relative residual| `0.127` → **QC pass**.
- **oxidized**: weighting `1/x`, intercept mode `free`; slope `1.10e-05`, intercept `-0.01612`. R² `0.9847`, max |relative residual| `0.142` → **QC pass**.
- **reduced**: weighting `1/x`, intercept mode `free`; slope `2.06e-05`, intercept `-0.01473`. R² `0.9988`, max |relative residual| `0.120` → **QC pass**.
- Deliverables: `DAD_to_Concentration_AUC/calibration_{form}.json`, `.../standards_fitted_{form}.csv`, and fallback SVG plots at `Chromatogram_Calibration_Plots/calibration_{form}.svg` (generated without matplotlib). No blanks available → LOD/LOQ remain undefined.
- Plot markers on calibration charts represent **standards only** (blue points). Treatments have dedicated overlays: `Chromatogram_Calibration_Plots/{form}_treatment_overlay_grouped_by_uva.{svg|csv}` (colored by UVA dose), `..._grouped_by_uvb.{svg|csv}` (colored by UVB dose), and `..._grouped_by_uva_uvb_ratio.{svg|csv}` (colored by UVA/UVB ratio; ratio is NaN where UVB dose is zero).

## Stage B — Treatment Concentrations
- Results saved to `DAD_to_Concentration_AUC/treatments_concentration_raw.csv` (30 samples × 3 forms).
- Consistency gate now uses a total-vs-(oxidized+reduced) regression (`total_vs_parts_model.json`) fitted on standards: slope `0.497 ± 0.006`, intercept `0.0016 ± 0.0032`, RMSE `0.0050`.
- Applying this model still flags `25/30` treatments (`total_vs_parts_consistency.csv`) because oxidized/reduced concentrations remain roughly double the total prediction.
- Dose correlations remain weak: Pearson r tops out at ~0.54 (total mg/mL vs UVB), while mg/gDW correlations stay ≤0.50 (`Chromatogram_derived_concentration_patterns_plots/chromatogram_dose_correlations.csv`).
- One-way ANOVA and Kruskal–Wallis tests across the six dose groups (30 observations per form) yield p ≥ 0.04 (`_concentrations_vs_dose_with_robust_mean/chromatogram_dose_stats.csv`); the lone borderline ANOVA hit (total mg/gDW p ≈ 0.044) is not corroborated by the matching Kruskal p ≈ 0.17, so we still lack statistically robust dose separation.
- LOQ flags unavailable (blank variance missing); column retained for downstream compatibility.

## Stage C — Biomass-Normalized Amounts
- Output `DAD_to_Concentration_AUC/treatments_corrected_amounts.csv` contains concentrations plus mg/gDW values after merging with `sample_id_truth.csv`.
- Amount SEs scale directly with concentration SEs (extraction volume 1.0 mL, dilution factor 1.0).
- Samples lacking `dry_biomass_g` retain `NaN` for normalized amounts; review `sample_id_truth.csv` for missing biomass entries.
- Consolidated table `chromatogram_derived_concentrations.csv` captures (per sample) the total concentration, peak area, UVA/UVB doses, ratio, biomass, corrected oxidized/reduced/total amounts, and change-from-baseline deltas. Regression diagnostics (including ΔUVA vs Δconcentration) live in `Chromatogram_derived_concentration_patterns_plots/regression_summary.csv`, with driver-specific point clouds in accompanying SVGs.
