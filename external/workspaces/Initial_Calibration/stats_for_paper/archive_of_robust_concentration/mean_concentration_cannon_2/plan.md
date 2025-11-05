# Dose-Level Robust Mean Analysis Plan

## Overview
We will explore dose-dependent behaviour after collapsing biological replicates with robust mean estimators. The goal is to contrast dose-level trends against the previously reported replicate-level null findings, and to prepare inputs for downstream reflectance modelling.

## Step 1 — Build Dose-Level Summaries
- Enumerate the six unique (UVA, UVB) dose pairs and note that UVB only decreases at the final step (0.768 → 0.707 mW·cm⁻²).
- Compute 20 % trimmed means (middle three replicates) and trimmed SDs for chromatogram and DAD totals/oxidized/reduced.
- Produce 95 % percentile bootstrap CIs with 2 000 resamples and document the sensitivity that comes from retaining only three data points per dose.
- Keep `dose_level_summary.csv` as the single source of trimmed means; reference it elsewhere rather than duplicating CSVs.

## Step 2 — Quantify Trends Using Robust Means
- Treat the dose summary as *n* = 6, compute Pearson/Spearman/Kendall correlations versus UVA and UVB, and fit weighted least squares regressions (weights = 1/trimmed-SD²).
- Report slopes, SEs, R², and 95 % CIs, explicitly describing the trend as “mostly increasing with a mid/late dip.”

## Step 3 — Maintain Replicate-Level Context
- Retain the replicate-level ANOVA, permutation, and robust tests (n = 30) so we can contrast variance-dominated results with dose-level ordering.
- Summarise the contrast: replicate analyses stay null whereas trimmed means climb through dose₄/₅ then soften.
- Build strip/line plots overlaying replicates with trimmed means ± CI to visualise the variance collapse and late decline.

## Step 4 — Align Chromatogram and DAD at Dose Level
- Using the dose-level robust means, compute Pearson/Spearman correlations and run Deming regression between chromatogram and DAD totals/oxidized/reduced.
- Document the assumed error variance ratio for Deming (default 1:1 unless assay-specific variances are estimated) and provide slopes/intercepts with 95 % CIs to demonstrate assay concordance.

## Step 5 — Prepare Outputs for Reflectance Integration
- Reference `dose_level_summary.csv` from reflectance workflows instead of keeping redundant per-directory copies. Document the (UVA, UVB) mapping and the UVB decline in the existing metadata table.

## Step 6 — Narrative Updates
- Refresh manuscript prose, captions, and figures so they:
  - Characterise the trend as “mostly increasing with a mid/late dip” for totals and reduced pools.
  - Explain that total vs. oxidized/reduced are non-additive component estimates and mention the negative oxidized control.
- Highlight why both views matter (replicate-level heterogeneity vs. dose-level ordering) and remind readers that n = 6 means keep inference exploratory.
