# Dose-Level Mean Analysis Notes

## Estimators
- Dose-level summaries use both simple means and 20% trimmed means (drop min/max of n=5) alongside Huber M-estimates (fallback to simple mean if convergence fails).
- Dispersion per dose is captured via trimmed-sample SE, MAD-derived SE, and bootstrap (1000 draws) percentile 95% confidence intervals.

## Dose Trend Models
- Linear trends against UVA (UVB is aliased) are reported for both ordinary and variance-weighted least squares.
- Weights default to inverse bootstrap variance; if unstable they fall back to MAD-based variance estimates.

## Assay Alignment
- Chromatogram vs. DAD dose means compared via Deming regression using variance ratios estimated from replicate-level MAD.
- Bootstrap (5000 resamples) provides slope/intercept confidence intervals; correlations (Pearson/Spearman/Kendall) summarise rank agreement.

## Next Steps
- Integrate replicate-level context by overlaying trimmed means ± CI on stripplots.
- Update narrative to contrast replicate-model null findings with dose-mean monotonic trends.
## Figures
- `dose_level_stripplot.png`: replicate-level points with 20% trimmed mean ± bootstrap CI for chromatogram and DAD totals as a visual contrast to the dose-level trend analysis.
- `dose_level_stripplot_all_forms.png`: six-panel view (chromatogram/DAD × total/oxidized/reduced) showing replicates with trimmed mean ± CI per dose.
