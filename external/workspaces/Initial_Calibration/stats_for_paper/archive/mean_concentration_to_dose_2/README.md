# Dose-Level Robust Mean Summary

This directory stores the aggregated dose summaries derived from `Combined_Scytonemin_Concentrations.csv`. Each row in the source file is a replicate with paired chromatogram and DAD concentrations expressed per gram dry weight.

## Files

- `plan.md` — analysis design and task breakdown.
- `dose_level_summary.csv` — one row per UVA/UVB dose, with robust means, trimmed standard deviations, bootstrap estimates, and 95 % CIs for total/oxidized/reduced chromatogram and DAD concentrations (mg·gDW⁻¹).
- `dose_level_relationships.csv` — Pearson/Spearman/Kendall correlations and weighted least-squares regressions between the dose rank and each concentration metric.
- `dose_level_summary_stats.md` — human-readable summary of WLS slopes/CIs and Deming regression parameters (all in mg/gDW).
- `dose_level_polyfits.csv` / `dose_level_poly_coeffs.csv` — quadratic predictions with bootstrap bands and fitted coefficients for each metric.
- `dose_level_deltas.csv` / `dose_level_delta_patterns.csv` — adjacent dose differences (20 % trimmed means) with bootstrap CIs and sign probabilities.
- `dose_level_permutation_quadratic.csv` — permutation tests comparing quadratic vs. flat models.
- `dose_level_order_tests.csv` — Jonckheere–Terpstra trend tests with optional single-slack removal per interior dose.
- `dose_level_monotone_slack.csv` — diagnostic counts of monotonicity violations (prototype order-restricted check).
- `chrom_dad_alignment.csv` — dose-level comparisons between chromatogram and DAD outputs (correlations plus Deming regression slopes/intercepts).
- `plots/` — replicate stripplots overlaid with 20 % trimmed mean ± bootstrap CI (annotations include slope, 95 % CI, and p-values) plus combined quadratic/delta figures (`*_poly_delta.png`).
- `Combined_Scytonemin_Concentrations.csv` — the replicate-level inputs copied from the main analysis directory (included here for reproducibility).

## Dose definition

Each dose corresponds to a unique `(UVA, UVB)` pair. Doses are ranked by increasing UVA:

| dose_id | dose_rank | UVA (mW·cm⁻²) | UVB (mW·cm⁻²) |
|---------|-----------|---------------|---------------|
| D1 | 1 | 0.000 | 0.000 |
| D2 | 2 | 0.647 | 0.246 |
| D3 | 3 | 1.095 | 0.338 |
| D4 | 4 | 1.692 | 0.584 |
| D5 | 5 | 2.488 | 0.768 |
| D6 | 6 | 3.185 | 0.707 |

Note the UVB peak at rank 5; the highest UVA dose (rank 6) has UVB = 0.707 mW·cm⁻², slightly lower than rank 5.

## Robust aggregation

- Location estimator: 20 % trimmed mean (Huber fallback was not required; bootstrap distributions were symmetric for all doses).
- Spread: standard deviation of the trimmed sample; also reported via bootstrap SE and 95 % percentile CIs (200 resamples).
- Bootstrap seed: NumPy default RNG; reproducibility can be enforced by setting `np.random.seed(...)` before running the scripts.

## Regression summaries

- Weighted least squares use weights `w = 1 / σ²`, where σ² is the bootstrap variance of the trimmed mean per dose (small/zero variances are replaced with the smallest positive finite variance).
- Correlations (Pearson/Spearman/Kendall) are computed on the six robust-mean observations.
- Deming/orthogonal regression between chromatogram and DAD means uses the ratio of average trimmed variances to set the error ratio λ; 1 000 bootstrap resamples provide 95 % CIs for slope/intercept.

## Usage notes

- The replicate-level analyses (ANOVA, permutations, ridge/PLS) remain the authoritative inference for biological variability.
- The dose-level summaries highlight latent monotone trends once biological variance is collapsed; treat associated p-values as descriptive (n = 6).
- Downstream reflectance models should use `dose_level_summary.csv` as the canonical concentration lookup table, keyed by `dose_id` or (`UVA`, `UVB`).
