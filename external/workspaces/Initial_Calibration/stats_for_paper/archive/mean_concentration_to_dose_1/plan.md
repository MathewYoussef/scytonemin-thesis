# Dose-Level Concentration Means — Plan

## Objectives
- Summarise chromatogram and DAD concentrations at the dose level using robust location estimates.
- Quantify mean-level dose trends while preserving replicate-level context and assay alignment.
- Deliver a tidy aggregation ready for reflectance integration plus clear narrative guidance.

## Data Inputs
- `Chromatogram_derived_concentrations.csv`, `DAD_derived_concentrations_corrected.csv`
- Existing latent/precision-weighted summaries (if needed) from prior analyses.

## Step 1 – Robust Dose-Level Summaries
1. Enumerate the six UVA/UVB dose combinations (confirm diagonal structure).
2. For each assay (total/oxidized/reduced):
   - Compute both the simple mean and a 20% trimmed mean (equivalent to dropping min/max with n=5) plus Huber M-estimate for comparison.
   - Capture dispersion via:
     - Trimmed standard deviation converted to SE (sd_trim / sqrt(n_eff)).
     - MAD-based SE (1.4826 × MAD / sqrt(n)).
     - Bootstrap (≥200 resamples) percentile CI for the preferred estimator; guard against zero-variance resamples.
3. Store per-dose summaries and metadata (dose_id, UVA, UVB, n, estimator type, mean, SE variants, CI bounds) in a tidy DataFrame.
4. Write the final table to `mean_concentration_to_dose_1/dose_level_summary.csv`.

## Step 2 – Dose-Mean Trend Assessment
1. Treat trimmed means as six observations ordered by UVA (noting UVB collinearity).
2. Fit simple regressions against UVA:
   - Ordinary least squares on trimmed means.
   - Weighted least squares using 1 / bootstrap_variance weights (fallback to MAD-based variance if bootstrap is degenerate).
3. Report slope, intercept, SE, 95% CI, R², and p-value; confirm UVB gives the same slope to document aliasing.
4. Compute Pearson, Spearman, and Kendall correlations for transparency.

## Step 3 – Replicate vs. Dose-Level Narrative
1. Reconcile new mean-level findings with existing replicate-level ANOVA/permutation outcomes.
2. Draft comparison bullets highlighting:
   - “Dose-level trimmed means show monotone increase (slope …).”
   - “Replicate-level models (n=30) remain nonsignificant under structure-preserving tests.”
3. Plan figure concept: strip plot of replicates with overlaid trimmed mean ± CI.

## Step 4 – Chromatogram vs. DAD Alignment
1. Using dose-level trimmed means, compute assay-to-assay comparisons:
   - Pearson/Spearman correlations.
   - Deming regression with variance ratio estimated from replicate MAD² between chromatogram and DAD to reflect relative assay noise.
2. Summarise slope, intercept, and 95% CI to confirm consistent dose ranking.

## Step 5 – Deliverables & Documentation
- `dose_level_summary.csv` with full metadata.
- Optional `assay_alignment.csv` capturing Deming regression output.
- Updated narrative notes (appended to `mean_concentration_to_dose_1/notes.md` or existing results memo) describing how mean-level trends coexist with replicate-level variance.
- Reminder to cite the 1/AUC weighting in DAD calibration when referencing slopes/SEs in text.

## Open Questions
- Need to choose final “headline” estimator (trimmed vs. Huber) once diagnostics are reviewed.
- Confirm bootstrap stability with only five replicates; may need to rely on MAD-based SE if resampling proves noisy.
- Decide whether latent concentration estimates should be included alongside raw assay values.
