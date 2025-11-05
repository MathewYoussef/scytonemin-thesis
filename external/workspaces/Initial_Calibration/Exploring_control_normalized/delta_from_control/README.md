# Delta from Control

## Idea
Subtract the control (UVA = UVB = 0) baseline from each measurement. This captures dose-induced deviations in absolute units (mg/mL, mg/gDW). If the control carries systematic offsets, this normalization removes them, leaving only the relative changes.

## Formula
For each form and measurement:

```
delta_value = value - baseline_control
```

Where `baseline_control` is typically the control median (robust to outliers). We will compute deltas for:
- Chromatogram `conc_mg_ml` and `amount_mg_per_gDW`
- DAD `predicted_*_mg_ml` and `predicted_*_mg_per_gDW`

## Workflow
1. Calculate control medians per form using chromatogram and DAD datasets.
2. Subtract the median from each corresponding measurement to produce delta columns.
3. Produce robust medians/IQR plots for the deltas vs dose.
4. Re-run ANOVA/Kruskal, correlations, and effect size on the delta values.
5. Save outputs (`*_delta_summary.csv`, plots) here.

## Notes
- If control variance is high, consider also reporting control mean and IQR.
- Negative deltas indicate the dose measurement sits below the control baseline.
- Document the baseline statistics alongside the derived delta tables for traceability.
