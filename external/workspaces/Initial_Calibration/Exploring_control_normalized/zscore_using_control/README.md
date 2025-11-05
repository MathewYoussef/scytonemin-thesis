# Z-Score Using Control

## Idea
Standardize each measurement by the control mean and standard deviation to express dose responses in “control standard deviation units.” This helps assess how far each dose deviates from the baseline relative to baseline variability.

## Formula

```
z_score = (value - control_mean) / control_std
```

Where `control_mean` and `control_std` are computed from control replicates for the same form/metric. Z-scores will be calculated for:
- Chromatogram `conc_mg_ml`, `amount_mg_per_gDW`
- DAD `predicted_*_mg_ml`, `predicted_*_mg_per_gDW`

## Workflow
1. Calculate control mean and standard deviation per form (optionally include robust alternatives for comparison).
2. Convert each measurement into a z-score.
3. Produce robust medians/IQR (or median |z|) plots versus dose.
4. Re-run ANOVA/Kruskal analyses on the z-scores.
5. Save the tables, plots, and baseline stats here.

## Notes
- If control variance is near zero, z-scores can become unstable; set a minimum std threshold or fall back to robust std.
- Z-scores allow direct comparison across forms with different scales.
- Keep track of the control mean/std used for reproducibility.
