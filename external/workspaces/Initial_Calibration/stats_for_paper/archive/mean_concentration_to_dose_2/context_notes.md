# Replicate vs. Dose-Level Perspective

- **Replicate-level (n = 30)**: retains biological variance; previous ANOVA, permutation, and ridge/PLS analyses show no robust UVA/UVB effects once collinearity and variance structure are accounted for.
- **Dose-level robust means (n = 6)**: 20 % trimmed means collapse within-dose variability and reveal monotone increases in chromatogram and DAD totals with the UVA/UVB ladder (see `dose_level_relationships.csv`).
- **Interpretation**: the latent trend is visible when technical replicates are aggregated, but the small sample of independent doses and the diagonal design (UVA and UVB co-vary, with UVB peaking at dose rank 5) require descriptive wording.
- **Visual summary**: replicate stripplots with trimmed mean ± bootstrap CI overlays live in `mean_concentration_to_dose_2/plots/`, illustrating how the broad replicate spread collapses to clear dose-level trends (annotations report slopes, CIs, and Spearman correlations; numeric values are tabulated in `dose_level_summary_stats.md`).
- **Shape diagnostics**: quadratic fits (`dose_level_polyfits.csv`), delta fingerprints (`dose_level_deltas.csv`), permutation tests (`dose_level_permutation_quadratic.csv`), and JT order tests with optional slack (`dose_level_order_tests.csv`) confirm a rise–dip–rebound pattern tied to the UVB-heavy D5 dose.
- **Recommendation**: use dose-level summaries for exploratory plots and reflectance calibration, while citing replicate-level inference for statements about statistical significance under biological variance.
