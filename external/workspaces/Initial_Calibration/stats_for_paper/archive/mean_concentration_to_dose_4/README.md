# Dose-Level Robust Mean Outputs

Generated from `Combined_Scytonemin_Concentrations.csv` using 500 bootstrap resamples per dose and a
Huber M-estimate (fallback to 20% trimmed mean on convergence failure).

## Files
- `dose_summary.csv`: Per-dose robust means, MAD, bootstrap SD/variance, and 95% CI for chromatogram and DAD metrics (mg·gDW⁻¹ and mg·mL⁻¹).
- `dose_level_correlations.csv`: Pearson/Spearman/Kendall correlations of dose axes (UVA, UVB) versus robust means.
- `dose_level_regressions.csv`: Weighted least squares (weights = 1 / bootstrap variance) slopes/intercepts vs. UVA/UVB.
- `chrom_dad_alignment.csv`: Cross-assay correlations and Deming regression (λ = 1) with bootstrap CIs.

Bootstrap seed: 20240907
Bootstrap draws per dose: 500
Deming bootstrap draws: 500

Note: UVB levels are strongly but not perfectly monotone with UVA (0.768 → 0.707 swap in the top two doses).