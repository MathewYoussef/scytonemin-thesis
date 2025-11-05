# Percent Change from Control

## Idea
Express each dose measurement as a percent change relative to the control baseline. This frames the effect in intuitive percentage units while retaining control baseline structure.

## Formula

```
percent_change = (value - baseline_control) / baseline_control * 100
```

Where `baseline_control` is the control median for the same form/metric. Percent change will be computed for:
- Chromatogram `conc_mg_ml`, `amount_mg_per_gDW`
- DAD `predicted_*_mg_ml`, `predicted_*_mg_per_gDW`

## Workflow
1. Calculate control medians per form.
2. Compute percent change for each measurement (handle zero control baselines carefully).
3. Produce robust median/IQR plots for percent change.
4. Re-run ANOVA/Kruskal, correlations, and effect-size analyses on the percent-change values.
5. Save tables and plots here.

## Notes
- Equivalent to ratio minus 1 expressed in percentage.
- If control baseline is zero (or near zero), percent change is undefined; use thresholds or fallback logic.
- Report baseline statistics alongside the percent-change results for context.
