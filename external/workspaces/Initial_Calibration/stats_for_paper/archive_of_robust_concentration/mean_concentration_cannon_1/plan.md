# Dose-Level Robust Mean Analysis Plan

## Overview
We will explore dose-dependent behaviour after collapsing biological replicates with robust mean estimators. The goal is to contrast dose-level trends against the previously reported replicate-level null findings, and to prepare inputs for downstream reflectance modelling.

## Step 1 — Build Dose-Level Summaries
- Enumerate the six unique (UVA, UVB) dose pairs and document the fact that UVB declines only on the final step (0.768 → 0.707 mW·cm⁻²).
- For each chromatogram and DAD metric (total, oxidized, reduced), compute 20 % trimmed means (middle three replicates) and trimmed SDs.
- Generate 95 % percentile bootstrap confidence intervals using 2 000 resamples; capture the sensitivity to three contributing replicates in the write-up.
- Publish a single canonical CSV (`dose_level_summary.csv`) and link to it from any reflectance or manuscript resources instead of duplicating the table.

## Step 2 — Quantify Trends Using Robust Means
- Treat the dose summary as *n* = 6 observations and compute Pearson/Spearman/Kendall correlations versus UVA and UVB.
- Fit weighted least squares regressions of concentration ~ UVA and concentration ~ UVB with weights = 1/trimmed-SD².
- Report slopes, SEs, R², and 95 % CIs, emphasising that the pattern is “mostly increasing with a mid/late dip” rather than perfectly monotone.

## Step 3 — Maintain Replicate-Level Context
- Retain the replicate-level ANOVA, permutation, and robust tests (n = 30) to highlight how biological variance masks dose effects.
- Summarise the contrast: replicate-level models remain null, while dose-level robust means rise through dose₄/₅ before softening.
- Prepare visuals overlaying replicates with trimmed means ± CI to illustrate the variance collapse and the late decline.

## Step 4 — Align Chromatogram and DAD at Dose Level
- Using the dose-level robust means, compute Pearson/Spearman correlations and run Deming regression between chromatogram and DAD totals/oxidized/reduced.
- Document the assumed error variance ratio for Deming (default 1:1 unless assay-specific variances are estimated) and provide slopes/intercepts with 95 % CIs to demonstrate assay concordance.

## Step 5 — Prepare Outputs for Reflectance Integration
- Point reflectance workflows to `dose_level_summary.csv` instead of keeping separate CSV copies; reference the existing metadata table for (UVA, UVB) mapping and the UVB drop at dose₆.

## Step 6 — Narrative Updates
- Update the manuscript and figure captions so they consistently describe the trend as “mostly increasing with a mid/late dip,” clarify that totals are not additive with oxidized/reduced components, and mention the negative oxidized controls.
- Highlight why both perspectives matter: replicate-level heterogeneity versus dose-level ordering, and the exploratory nature of results based on six dose means.
