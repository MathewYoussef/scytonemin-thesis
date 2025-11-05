# Control Summary

## Purpose
Document control (UVA = UVB = 0) statistics that underpin all normalization approaches. This subfolder holds the baseline values used elsewhere so we can track assumptions, recalc thresholds, or reuse control parameters in new methods.

## Planned Outputs
- Control medians, means, standard deviations per form for both chromatogram and DAD datasets (mg/mL and mg/gDW).
- Plots illustrating control variability (e.g., boxplots or distribution histograms).
- Notes on any control outliers or data quality concerns.

## Workflow
1. Extract control rows from chromatogram and DAD concentration tables.
2. Compute stats (mean, median, SD, IQR) per form/metric.
3. Save summary tables (`control_baseline_stats_chromatogram.csv`, `..._dad.csv`).
4. Generate optional plots for inspection.
5. Reference these stats in the delta/ratio/percent/z-score normalizations.

## Notes
- Keep a record of which control statistics were used in each downstream method.
- The control summary itself does not perform normalization; it just centralizes the baseline numbers so other subfolders stay in sync.
