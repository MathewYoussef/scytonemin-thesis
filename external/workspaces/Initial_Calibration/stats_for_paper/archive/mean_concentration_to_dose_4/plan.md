# Dose-Level Mean Concentration Analysis Plan

## Objective
Summarize chromatogram and DAD concentrations at the dose level using robust means, quantify trends along the UVA/UVB axis, and contrast dose-level behavior with replicate-level inference while preparing artifacts for downstream reflectance modeling.

## Context Check
- The UVA grid is strictly increasing across the six dose cells; UVB co-varies strongly but is not perfectly monotone (the top two doses swap 0.768 → 0.707), so both axes should be retained in summaries.
- Replicate-level analyses (n = 30) show no robust interaction effects once biological variance is modeled; the mean-level view is exploratory and should be framed as complementary, not confirmatory.

## 1. Build Dose-Level Summaries
1. Extract chromatogram and DAD totals/oxidized/reduced from the replicate tables.
2. For each unique UVA/UVB pair (six cells):
   - Compute a robust center (Huber M-estimate or 20% trimmed mean; default to Huber given n = 5). Record the estimator and tuning constant in metadata.
   - Quantify spread per dose using trimmed SD or MAD.
   - Run a replicate-level bootstrap (≥200 draws) of the robust statistic to obtain SD and percentile 95% CI bounds. Retain bootstrap variance for weighting.
3. Assemble a tidy table with columns such as `dose_id`, `uva_mw_cm2`, `uvb_mw_cm2`, `chrom_total_mean_huber`, `chrom_total_ci_low/high`, …, including latent/precision-weighted concentrations when available.
4. Save the table to `mean_concentration_to_dose_4/dose_summary.csv` (and later to the shared `aggregated_reflectance/` location for reflectance workflows).

## 2. Quantify Dose Relationships with Robust Means
1. Treat the six robust means as the analytic set.
2. Compute Pearson, Spearman, and Kendall correlations versus both UVA and UVB to show agreement and highlight the small UVB reversal.
3. Fit weighted least squares of concentration vs. UVA (primary axis) using weights = 1 / bootstrap_variance. Report slope, SE, 95% CI, and R²; note that UVB slopes are redundant but can be quoted for completeness.
4. Repeat the regressions for latent/precision-weighted concentrations to emphasise magnitude while documenting the weighting scheme.
5. Flag explicitly that inference is limited (n = 6) and exploratory.

## 3. Maintain Replicate-Level Context
1. Keep existing replicate-level ANOVA/OLS/permutation summaries referenced side-by-side.
2. Summarize the contrast: “Replicate-level models (n = 30) show no robust dose–concentration effect once biological variance is retained; dose-level robust means (n = 6) exhibit a monotonic trend along the UVA axis.”
3. Prepare figures that overlay stripplots of all replicates with dose-level robust mean ± CI to visualise biological spread vs. mean trend.

## 4. Chromatogram vs. DAD Alignment at Dose Level
1. Using the dose-level robust means, compute Pearson/Spearman correlations between chromatogram and DAD totals (and oxidized/reduced).
2. Fit a Deming or Passing–Bablok regression (note the assumed error ratio); provide slope/intercept with bootstrap 95% CI to demonstrate cross-assay agreement.
3. Report any notable slope differences to guide assay selection for reflectance models.

## 5. Prepare Outputs for Reflectance Integration
1. Store the master dose summary table plus metadata (estimator spec, bootstrap settings, dose definitions) under `aggregated_reflectance/` once finalized.
2. Create a small README documenting dose IDs, UVA/UVB values, and pointers to the replicate sources so downstream scripts use identical definitions.

## 6. Narrative Updates
1. Update the Results/Discussion to present both views:
   - Replicate-level inference retains the “no robust dose effect” conclusion.
   - Dose-level robust means (n = 6) suggest a monotonic increase when biological variance is collapsed.
2. Explain why both perspectives matter (biological heterogeneity vs. mean structure) and how reflectance modeling will rely on the dose-level summaries for pattern recognition while retaining replicate-level caution for inference.

## Next Steps
- Confirm estimator choice (Huber vs. trimmed mean) before coding.
- Implement the summarization scripts/notebooks, respecting read/write constraints.
- Validate bootstrap reproducibility (set seeds) and document assumptions in the metadata.
