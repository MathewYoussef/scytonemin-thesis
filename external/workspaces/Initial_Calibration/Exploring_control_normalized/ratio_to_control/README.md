# Ratio to Control

## Idea
Divide each measurement by the control baseline to express dose responses as fold changes relative to the control. This highlights proportional changes (e.g., >1 means above control, <1 means below control).

## Formula

```
ratio_value = value / baseline_control
```

Where `baseline_control` is the control median for the same form/metric. Ratios will be computed for:
- Chromatogram `conc_mg_ml`, `amount_mg_per_gDW`
- DAD `predicted_*_mg_ml`, `predicted_*_mg_per_gDW`

## Workflow
1. Calculate per-form control medians.
2. Normalize each measurement by its control median (handle zero/near-zero denominators carefully).
3. Produce robust medians/IQR plots for the ratios.
4. Re-run ANOVA/Kruskal, correlations, and effect-size analyses on the ratios.
5. Save outputs (ratio tables, stats, plots) here.

## Notes
- If control median is near zero, ratio can be unstable. Flag any cases where the denominator is < tolerance.
- Ratios can be log-transformed later if needed (e.g. log2 fold change).
- Document all control baselines used for transparency.
