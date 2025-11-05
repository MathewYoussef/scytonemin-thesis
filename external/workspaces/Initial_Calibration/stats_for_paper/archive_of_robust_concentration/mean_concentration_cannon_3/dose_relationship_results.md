
# Dose–Concentration Pattern Summary

## Framing
Replicate-level tests (n = 30) did not isolate convincing UVA/UVB effects, so we focus on 20 % trimmed means (middle three replicates) to examine the dose-level response within `mean_concentration_cannon_3/`.

## Data Assets & Reminders
- Replicates: `Combined_Scytonemin_Concentrations.csv`.
- Trimmed means: `dose_level_summary.csv` (20 % trimmed mean, trimmed SD, 95 % percentile bootstrap CI built from 2 000 resamples).
- Trend diagnostics: `dose_trend_stats.csv`, `dose_pattern_summary.csv`.
- Sequential deltas: `dose_pattern_sequential_deltas.csv`.
- Assay concordance: `chrom_dad_alignment.csv`.
- Component definitions: “total” columns come from direct chromatogram/DAD quantification. Oxidized/reduced pools are spectral components that are *not* additive, so reduced estimates can exceed the total.
- Baseline artefact: DAD oxidized control values are slightly negative because of baseline subtraction; flag this when publishing.
- Canonical source: treat `dose_level_summary.csv` as the single trimmed-mean table and reference it elsewhere instead of duplicating files.

## Mean-Level Regressions (UVA axis)
Weighted least squares (weights = 1/trimmed-SD²; `dose_trend_stats.csv`) yield:
- Chrom total slope: 0.091 mg·gDW⁻¹·(mW·cm⁻²)⁻¹ (SE 0.075, R² = 0.27).
- Chrom reduced slope: 0.390 (SE 0.085, R² = 0.84).
- DAD total slope: 0.134 (SE 0.132, R² = 0.20).
- DAD reduced slope: 0.472 (SE 0.122, R² = 0.79).

Pearson correlations with UVA fall between 0.64–0.75 for totals and 0.69–0.75 for reduced pools. Slopes are positive, but the highest dose declines, so the behaviour is “mostly increasing with a late dip.”

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
- β₁ values reflect the initial slope near UVA = 0. The negative β₂ terms encode the softening between dose₄ and dose₅ while UVB is still increasing; UVB only declines between dose₅ and dose₆.
- Replicate-level Kendall τ values (≈0.26–0.29 for totals/reduced) show modest ordering even before aggregation.

## Sequential Deltas
`dose_pattern_sequential_deltas.csv` summarises the bootstrap 95 % CIs for dose changes:
- Chrom total: +0.40, +0.08, +0.24, −0.18, +0.02.
- Chrom reduced: +0.60 [0.23, 1.30], −0.16, +0.34, +0.27, −0.38.
- DAD total: +0.89, +0.09, +0.35, −0.40, +0.15.
- DAD reduced: +0.89 [0.39, 1.76], −0.23, +0.32, +0.07, −0.18.

Only the first step (dose₁→₂) has a CI that excludes zero; later intervals overlap zero, so curvature claims stay exploratory. Sign patterns: totals `+ + + − +`, reduced `+ − + + −`.

## Interpretation
- Chromatogram totals climb through dose₄ (0.986 mg·gDW⁻¹), dip at dose₅ (0.807 mg·gDW⁻¹), and end at 0.831 mg·gDW⁻¹.
- Chromatogram reduced peaks at 1.790 mg·gDW⁻¹ (dose₅) before dropping to 1.411 mg·gDW⁻¹.
- DAD totals peak at 1.844 mg·gDW⁻¹ (dose₄), decline to 1.440 mg·gDW⁻¹, and partially rebound to 1.585 mg·gDW⁻¹.
- DAD reduced reaches 2.093 mg·gDW⁻¹ (dose₅) and falls to 1.913 mg·gDW⁻¹ at dose₆.
- Describe the response as “mostly increasing with a late dip,” and remind readers that totals vs. oxidized/reduced are non-additive with small-sample uncertainty.

## Follow-Up
- Match the delta sign patterns and regression/Deming summaries when comparing to reflectance outputs or other models.
- Document the baseline subtraction that causes negative oxidized controls.
- Future robustness checks could include leave-one-out analyses of the trimmed means or monotonic trend tests (e.g., Page’s test) if additional confirmation is required.
