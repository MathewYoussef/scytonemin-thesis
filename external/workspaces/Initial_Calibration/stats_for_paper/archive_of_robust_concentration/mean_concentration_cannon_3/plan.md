# Dose-Level Robust Mean Analysis Plan

## Overview
We will explore dose-dependent behaviour after collapsing biological replicates with robust mean estimators. The goal is to contrast dose-level trends against the previously reported replicate-level null findings, and to prepare inputs for downstream reflectance modelling.

## Step 1 — Build Dose-Level Summaries
- Enumerate the six unique (UVA, UVB) dose pairs and note that UVB only drops at the final step (0.768 → 0.707 mW·cm⁻²).
- Compute 20 % trimmed means (middle three replicates) and trimmed SDs for chromatogram and DAD totals/oxidized/reduced.
- Generate 95 % percentile bootstrap CIs with 2 000 resamples, emphasising the sensitivity introduced by only three contributing replicates.
- Keep `dose_level_summary.csv` as the canonical trimmed-mean table and reference it elsewhere rather than duplicating CSVs.

## Step 2 — Quantify Trends Using Robust Means
- Treat the dose summary as *n* = 6. Compute Pearson/Spearman/Kendall correlations versus UVA and UVB, and fit weighted least squares models (weights = 1/trimmed-SD²).
- Report slopes, SEs, R², and 95 % CIs, consistently describing the behaviour as “mostly increasing with a mid/late dip.”

## Step 3 — Maintain Replicate-Level Context
- Preserve ANOVA, permutation, and robust tests on the replicate data (n = 30) to show how variance washes out dose effects.
- Summarise the contrast: replicate-level analyses stay null, while trimmed means rise through dose₄/₅ before softening.
- Prepare visuals overlaying replicates with trimmed means ± CI to show variance collapse plus the late decline.

## Step 4 — Align Chromatogram and DAD at Dose Level
- Using the dose-level robust means, compute Pearson/Spearman correlations and run Deming regression between chromatogram and DAD totals/oxidized/reduced.
- Document the assumed error variance ratio for Deming (default 1:1 unless assay-specific variances are estimated) and provide slopes/intercepts with 95 % CIs to demonstrate assay concordance.

## Step 5 — Prepare Outputs for Reflectance Integration
- Reference `dose_level_summary.csv` directly from reflectance and manuscript workflows; avoid redundant CSV copies. Use the existing metadata table to communicate the (UVA, UVB) mapping and the UVB drop at dose₆.

## Step 6 — Narrative Updates
- Align summaries, captions, and figure callouts so they:
  - Describe the trend as “mostly increasing with a mid/late dip.”
  - Clarify that totals and oxidized/reduced components are not additive and mention the negative oxidized control.
- Highlight the contrast between replicate-level heterogeneity and dose-level ordering, stressing that n = 6 dose means keep inference exploratory.
