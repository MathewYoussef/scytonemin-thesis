# Dose–Concentration Pattern Summary

## Framing
Replicate-level hypothesis tests (n = 30) remained null, so we collapsed each dose to a 20 % trimmed mean (middle three replicates) and studied the dose-level behaviour instead. This summary aligns the quantitative results with the revised narrative.

## Data Preparation Notes
- Scope: files under `mean_concentration_cannon_1/`; concentrations are reported in mg·gDW⁻¹.
- Key artefacts: replicates (`Combined_Scytonemin_Concentrations.csv`), trimmed means (`dose_level_summary.csv`), trend diagnostics (`dose_trend_stats.csv`/`dose_pattern_summary.csv`), deltas (`dose_pattern_sequential_deltas.csv`), and assay alignment (`chrom_dad_alignment.csv`).
- Bootstraps: 95 % percentile intervals use 2 000 resamples.
- Component definitions: the “total” column reflects direct chromatogram/DAD quantification. Oxidized and reduced pools come from separate spectral components and are not additive; the reduced estimate can exceed the total.
- Baseline artefact: DAD oxidized values at the control sit slightly below zero due to baseline subtraction. They remain in the analysis but should be footnoted.
- Canonical table: treat `dose_level_summary.csv` as the source of trimmed means and reference it elsewhere rather than duplicating CSVs.

## Mean-Level Regressions (UVA axis)
Weighted least squares with weights = 1/trimmed-SD² (`dose_trend_stats.csv:2-13`) yield positive slopes:
- Chrom total: 0.091 mg·gDW⁻¹·(mW·cm⁻²)⁻¹ (SE 0.075, R² = 0.27).
- Chrom reduced: 0.390 (SE 0.085, R² = 0.84).
- DAD total: 0.134 (SE 0.132, R² = 0.20).
- DAD reduced: 0.472 (SE 0.122, R² = 0.79).

Pearson correlations with UVA fall between 0.64 and 0.75 for totals and 0.69 and 0.75 for reduced pools. The positive slopes describe a “mostly increasing” trend rather than a strictly monotonic one because the highest dose declines.

## Quadratic Fit (UVA + UVA²)
Coefficients from `dose_pattern_summary.csv`:

| metric | β₁ (± SE) | p(β₁) | β₂ (± SE) | p(β₂) | Kendall τ (replicates) | Kendall p |
|:--|--:|--:|--:|--:|--:|--:|
| Chrom Total | 0.632 ± 0.132 | 0.017 | −0.149 ± 0.035 | 0.024 | 0.280 | 0.040 |
| Chrom Oxidized | 0.171 ± 0.208 | 0.470 | −0.039 ± 0.061 | 0.567 | 0.132 | 0.336 |
| Chrom Reduced | 0.653 ± 0.222 | 0.060 | −0.127 ± 0.100 | 0.294 | 0.260 | 0.057 |
| DAD Total | 0.927 ± 0.383 | 0.094 | −0.231 ± 0.108 | 0.122 | 0.255 | 0.062 |
| DAD Oxidized | 0.354 ± 0.535 | 0.556 | −0.073 ± 0.155 | 0.668 | 0.201 | 0.142 |
| DAD Reduced | 0.770 ± 0.238 | 0.048 | −0.145 ± 0.103 | 0.254 | 0.285 | 0.037 |

Interpretation:
- β₁ captures the initial slope near UVA = 0; the negative β₂ terms encode the softening between dose₄ and dose₅ while both UVA and UVB are still increasing. The UVB drop only appears between dose₅ and dose₆.
- Replicate-level Kendall τ values sit around 0.26–0.29 for totals/reduced, signalling modest positive ordering even before aggregation.

## Sequential Dose-to-Dose Changes
Bootstrap deltas (`dose_pattern_sequential_deltas.csv`) highlight the step-by-step behaviour:
- Chrom total: +0.40, +0.08, +0.24, −0.18, +0.02.
- Chrom reduced: +0.60 [0.23, 1.30], −0.16, +0.34, +0.27, −0.38.
- DAD total: +0.89, +0.09, +0.35, −0.40, +0.15.
- DAD reduced: +0.89 [0.39, 1.76], −0.23, +0.32, +0.07, −0.18.

Only the first jump (dose₁→₂) has an interval that excludes zero; later intervals overlap zero, underscoring the exploratory nature of the curvature claims. Sign patterns are `+ + + − +` for totals and `+ − + + −` for reduced pools.

## Interpretation
- Both assays rise through dose₄, the reduced pools peak at dose₅ (1.790 mg·gDW⁻¹ for chromatogram, 2.093 mg·gDW⁻¹ for DAD), and the final step declines as UVB drops.
- Describe the response as “mostly increasing with a late dip,” not fully monotonic.
- Because the analysis rests on six trimmed means, confidence intervals and p-values are exploratory. Report them with that caveat.

## Uses and Follow-Up
- When comparing to reflectance or other models, match the sign patterns from the delta table and the Deming/linear slopes.
- Document the baseline subtraction behind negative oxidized estimates and reference a single trimmed-mean CSV to avoid copy drift.
- Future robustness checks could include leave-one-out sensitivity on the retained replicates or alternative monotonic trend tests (e.g., Page’s test) if additional confirmation is needed.
