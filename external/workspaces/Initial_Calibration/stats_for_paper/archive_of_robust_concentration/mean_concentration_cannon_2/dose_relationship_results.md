# Dose–Concentration Pattern Summary

## Framing
Replicate-level analyses (n = 30) were dominated by biological variance, so we focus on 20 % trimmed means (middle three replicates) to examine dose-level behaviour. This file aligns the quantitative evidence with the updated narrative.

## Data Assets & Reminders
- Replicates: `Combined_Scytonemin_Concentrations.csv`.
- Trimmed means: `dose_level_summary.csv` (20 % trimmed mean, trimmed SD, 95 % percentile bootstrap CI with 2 000 resamples).
- Trend diagnostics: `dose_trend_stats.csv`, `dose_pattern_summary.csv`.
- Sequential deltas: `dose_pattern_sequential_deltas.csv`.
- Assay concordance: `chrom_dad_alignment.csv`.
- “Total” values stem from direct chromatogram/DAD quantification; oxidized and reduced pools are spectral components that are not additive. Reduced concentrations can therefore exceed the reported totals.
- DAD oxidized control values are slightly negative because of baseline subtraction—retain them but note the artefact.
- Use `dose_level_summary.csv` as the single source of trimmed means and reference it from other workflows instead of duplicating CSVs.

## Mean-Level Regressions (UVA axis)
Weighted least squares (weights = 1/trimmed-SD²) from `dose_trend_stats.csv` show:
- Chrom total slope: 0.091 mg·gDW⁻¹·(mW·cm⁻²)⁻¹ (SE 0.075, R² = 0.27).
- Chrom reduced slope: 0.390 (SE 0.085, R² = 0.84).
- DAD total slope: 0.134 (SE 0.132, R² = 0.20).
- DAD reduced slope: 0.472 (SE 0.122, R² = 0.79).

Pearson r with UVA spans roughly 0.64–0.75 for totals and 0.69–0.75 for reduced pools. Slopes are positive but the final dose declines, so describe the behaviour as “mostly increasing with a late dip.”

## Quadratic Fits (UVA + UVA²)
Coefficients (`dose_pattern_summary.csv`):

| metric | β₁ (± SE) | p(β₁) | β₂ (± SE) | p(β₂) | Kendall τ (replicates) | Kendall p |
|:--|--:|--:|--:|--:|--:|--:|
| Chrom Total | 0.632 ± 0.132 | 0.017 | −0.149 ± 0.035 | 0.024 | 0.280 | 0.040 |
| Chrom Oxidized | 0.171 ± 0.208 | 0.470 | −0.039 ± 0.061 | 0.567 | 0.132 | 0.336 |
| Chrom Reduced | 0.653 ± 0.222 | 0.060 | −0.127 ± 0.100 | 0.294 | 0.260 | 0.057 |
| DAD Total | 0.927 ± 0.383 | 0.094 | −0.231 ± 0.108 | 0.122 | 0.255 | 0.062 |
| DAD Oxidized | 0.354 ± 0.535 | 0.556 | −0.073 ± 0.155 | 0.668 | 0.201 | 0.142 |
| DAD Reduced | 0.770 ± 0.238 | 0.048 | −0.145 ± 0.103 | 0.254 | 0.285 | 0.037 |

Interpretation:
- β₁ reflects the initial slope at low UVA; the negative β₂ terms encode the softening between dose₄ and dose₅ while UVB is still increasing. The UVB drop only occurs between dose₅ and dose₆.
- Replicate-level Kendall τ values (≈0.26–0.29 for totals/reduced) show modest ordering even before aggregation.

## Sequential Deltas
`dose_pattern_sequential_deltas.csv` documents the stepwise changes (95 % CIs):
- Chrom total: +0.40, +0.08, +0.24, −0.18, +0.02.
- Chrom reduced: +0.60 [0.23, 1.30], −0.16, +0.34, +0.27, −0.38.
- DAD total: +0.89, +0.09, +0.35, −0.40, +0.15.
- DAD reduced: +0.89 [0.39, 1.76], −0.23, +0.32, +0.07, −0.18.

Only the first jump (dose₁→₂) has an interval that excludes zero; the remaining deltas overlap zero, so curvature claims stay exploratory. Sign patterns: totals `+ + + − +`, reduced `+ − + + −`.

## Interpretation
- Chromatogram totals climb through dose₄ (0.986 mg·gDW⁻¹), dip at dose₅ (0.807 mg·gDW⁻¹), and end at 0.831 mg·gDW⁻¹.
- Chromatogram reduced peaks at 1.790 mg·gDW⁻¹ (dose₅) and drops to 1.411 mg·gDW⁻¹ at dose₆.
- DAD totals peak at 1.844 mg·gDW⁻¹ (dose₄), fall to 1.440 mg·gDW⁻¹, and partially rebound to 1.585 mg·gDW⁻¹.
- DAD reduced peaks at 2.093 mg·gDW⁻¹ (dose₅) before declining to 1.913 mg·gDW⁻¹.
- Describe the trend as “mostly increasing with a late dip” and note the non-additive component definitions and negative oxidized baseline.

## Follow-Up
- When comparing with reflectance analyses, match the delta sign patterns and regression slopes/Deming parameters.
- Document the baseline correction that drives negative oxidized values.
- If more confirmation is required, consider leave-one-out sensitivity on the trimmed means or monotonic trend tests such as Page’s test.
