# Figure Captions and Statistical References

## 01 – DAD spectrum (sample vs. blank)
Highest-total treatment spectrum (sample ID chosen from `DAD_derived_concentrations_corrected.csv`, predicted total mg·gDW⁻¹ = 2.53) overlaid with the blank mean in the 320–480 nm band. Illustrates the raw optical signal that underpins the DAD calibration fits in `Diode_Array_Derived_Calibration_Plots/standards_fitted_total.csv`.

## 02 – Chromatogram calibration scatter (total form)
Chromatogram peak area vs. known concentration for total scytonemin standards with treatment overlays coloured by UVA/UVB doses. The fit (slope 7.21 × 10⁻⁶, intercept −0.01599 mg·mL⁻¹) matches the calibration parameters in `calibration_total.json`; all treatment concentrations in `Chromatogram_derived_concentrations.csv` fall on this regression.

## 03 – DAD calibration scatter (total 320–480 nm)
DAD AUC vs. known concentration for total scytonemin standards with treatment overlays. Fit (slope 4.53 × 10⁻⁸, intercept −0.0240 mg·mL⁻¹) agrees with `Diode_Array_Derived_Calibration_Plots/calibration_total.json`, confirming that the predicted totals in `DAD_derived_concentrations_corrected.csv` are consistent with the validated calibration curve.

## 04 – Example concentration table (chromatogram vs. DAD)
Ten representative samples listing UVA/UVB doses, chromatogram concentrations (mg·mL⁻¹ and mg·gDW⁻¹) and DAD predictions. Data are pulled directly from `Chromatogram_derived_concentrations.csv` and `DAD_derived_concentrations_corrected.csv`.

## 05 – Full single-factor panels (chromatogram & DAD, total/oxidized/reduced)
Six-row figure showing chromatogram and DAD total/oxidized/reduced mg·gDW⁻¹ against UVA, UVB, UVB grouped by UVA, and UVA×UVB scatter. These distributions mirror the single-factor regression outputs in `Chromatogram_derived_concentration_patterns_plots/regression_summary.csv` and the OLS/two-way ANOVA results in `chromatogram_two_way_anova*.csv` and `dad_two_way_anova*.csv`, which show clear UVA×UVB effects for delta and z-score chromatogram endpoints (p ≈ 0.008–0.040) but only marginal signals for ratio/percent variants (p ≈ 0.072–0.12) and none for the DAD responses.

## 05 (prev) – Prior single-factor layout (kept for comparison)
Legacy version of the single-factor panel prior to the expanded 6×4 layout. Retained only for historical comparison—statistical interpretation matches the current figure.

## 05b – Total-only single-factor view (chromatogram & DAD)
Concentrates on total mg·gDW⁻¹ for chromatogram (top row) and DAD (bottom row), plotted against UVA and UVB independently. Captures the key result that single-factor dose trends remain modest, consistent with the slopes and R² values in `regression_summary.csv` (e.g., UVB→chromatogram total slope 0.56 mg·gDW⁻¹ per mW·cm⁻², R² = 0.24; UVB→DAD total slope 1.03 mg·gDW⁻¹ per mW·cm⁻², R² = 0.21).

## 06a – Chromatogram z-score heatmap (mean per UVA×UVB cell)
Mean z-score mg·gDW⁻¹ for each UVA×UVB dose cell, annotated with replicate counts. Supports the interaction hints observed in `chromatogram_two_way_anova_zscore.csv` (interaction p = 0.034, η² = 0.34) while reminding that elevated z-scores occur in specific combinations.

## 06b – DAD total heatmap (mean per UVA×UVB cell)
Mirrors 06a using predicted DAD totals. The heatmap shows modest variation, aligning with `dad_two_way_anova.csv` where the interaction term is only marginal (p ≈ 0.058) and the Freedman–Lane test in `freedman_lane_interaction.csv` yields p ≥ 0.71.

## 06c – Additional chromatogram heatmaps (raw, delta, ratio)
Row of heatmaps for raw mg·gDW⁻¹, delta vs. control, and ratio-to-control metrics. These panels illustrate that delta and z-score normalizations emphasize UVA×UVB hot spots (p ≈ 0.008–0.040 in `chromatogram_two_way_anova_delta.csv`), whereas ratio and percent variants retain only marginal interaction evidence (p ≈ 0.07–0.12 in `chromatogram_two_way_anova_ratio.csv` and `chromatogram_two_way_anova_pct.csv`).

## 07 – Simple vs. Freedman–Lane permutation comparison
Bar chart contrasting simple response shuffles (p_perm) with Freedman–Lane structured permutations (p_freedman_lane) for each dataset/variant/metric. Confirms that interaction signals vanish under Freedman–Lane (p ≥ 0.71; see `freedman_lane_interaction.csv`).

## 08 – Ridge regression bootstrap intervals
Median ±95 % bootstrap intervals for ridge regression coefficients across chromatogram and DAD metrics (`ridge_bootstrap_summary.csv`). Highlights the stable, modest UVB coefficients and uncertainty in the UVA×UVB interaction terms.

## 09 – Outlier sensitivity scenarios
Interaction p-values for baseline, drop-one, and drop-set outlier removal (`outlier_sensitivity.csv`). Shows that trimming influential observations leaves interaction p-values essentially unchanged, reinforcing that the earlier hints remain exploratory.
