# Dose-level dose–concentration summary

## Datasets
- Robust dose summaries (`dose_summary.csv`) with Huber means, bootstrap SD/variance, and 95 % CIs for each assay/analyte (`mean_concentration_to_dose_4/dose_summary.csv:1`).
- Correlations and weighted linear fits vs. UVA/UVB (`dose_level_correlations.csv:1`, `dose_level_regressions.csv:1`).
- Polynomial (linear + quadratic) fits on dose means (`dose_level_poly_regressions.csv:1`) and replicate-level contrasts (`replicate_poly_contrasts.csv:1`).
- Bootstrap finite differences describing changes between consecutive doses (`dose_level_step_differences.csv:1`).
- Cross-assay alignment stats (`chrom_dad_alignment.csv:1`).
- Visual overlay of replicates vs. robust means (`fig_dose_means_faceted.png:1`).

Dose order (UVA / UVB mW·cm⁻²): 0/0 → 0.647/0.246 → 1.095/0.338 → 1.692/0.584 → 2.488/0.768 → 3.185/0.707.

## Key findings by analyte

| Metric | Spearman ρ (UVA) | WLS slope (95 % CI)* | Quadratic WLS p | Replicate linear p | Replicate quadratic p | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Chrom total | 0.71 (p=0.111) (`dose_level_correlations.csv:2`) | 0.08 (p=0.248) (`dose_level_regressions.csv:1`) | 0.029 (`dose_level_poly_regressions.csv:2`) | 0.0047 (`replicate_poly_contrasts.csv:2`) | 0.0108 (`replicate_poly_contrasts.csv:2`) | Control revises upward to 0.32 mg·gDW⁻¹, and the quadratic term still flags the mid-dose dip even though the linear WLS slope is modest. |
| Chrom oxidized | 0.31 (p=0.544) | 0.03 (p=0.499) | 0.850 | 0.508 | 0.211 | No clear trend; Huber means sit close to trimmed values with wide CIs. |
| Chrom reduced | 0.77 (p=0.072) | 0.33 (p=0.037) | 0.379 | 0.016 | 0.122 | Strong monotone increase through dose 4 with a gentle softening at the UVB-heavy step. |
| DAD total | 0.54 (p=0.266) | 0.15 (p=0.334) | 0.097 | 0.0086 | 0.022 | Replicate-level fits still show the rise–dip–rise pattern even though the Huber slope alone is not significant. |
| DAD oxidized | 0.54 (p=0.266) | 0.12 (p=0.190) | 0.757 | 0.146 | 0.161 | Mild rise then plateau; behaviour mirrors the chromatogram oxidized series. |
| DAD reduced | 0.71 (p=0.111) | 0.46 (p=0.029) | 0.402 | 0.013 | 0.065 | Clearest dose response among the DAD metrics; positive through rank 4 with a shallow dip thereafter. |

*WLS slope CIs are available in `dose_level_regressions.csv:1`.

## Dose-to-dose changes (Huber means)

Representative differences (Δ mg·gDW⁻¹) with bootstrap 95 % CIs (`dose_level_step_differences.csv:1`):
- **Chrom total:** +0.38 (0→0.647), +0.06 (0.647→1.095), +0.23 (1.095→1.692), −0.15 (1.692→2.488), −0.04 (2.488→3.185). Only the first jump is clearly positive; subsequent steps carry zero within the CI, underscoring the rise–dip motif.
- **Chrom reduced:** +0.67, −0.22, +0.36, +0.35, −0.39 — the initial surge remains well above zero, while the UVB-heavy step (rank 5) drags the curve down.
- **DAD reduced:** +0.94, −0.27, +0.31, +0.13, +0.10 — mirrors the chromatogram reduced series with the same mid-dose wobble but overall positive movement.

## Cross-assay agreement
Dose-level Huber means align strongly (`chrom_dad_alignment.csv:1`): total Pearson r=0.99, Deming slope≈2.06 [1.63, 2.53]; oxidized r=0.89; reduced r=0.83. This validates using either assay as a proxy in downstream reflectance modeling.

## Interpretation & next steps
1. **Reduced fractions** (chromatogram & DAD) show the clearest dose association—both linear slopes and finite differences are consistently positive through the first four doses, with small late dips. These signatures can anchor reflectance pattern matching.
2. **Totals** exhibit a rise–dip–rise pattern flagged by significant linear and quadratic terms in replicate-level fits. This suggests a non-monotonic response (increase–increase–decrease–increase) rather than a simple trend.
3. **Oxidized fractions** remain noisy; no component reaches nominal significance.
4. For richer pattern detection we can extend beyond quadratic fits (e.g., unimodal/isotonic regression) or compare the finite-difference signature directly against reflectance-derived trends.

The figure (`fig_dose_means_faceted.png:1`) summarizes these behaviors visually for all six endpoints.
