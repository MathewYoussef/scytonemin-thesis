# Dose-Level Robust Mean Findings

## Objective
Prior replicate-level tests (n = 30) could not isolate clear UVA/UVB effects, so we re-analysed the experiment using 20 % trimmed means to see whether dose-level averages reveal structure once biological outliers are down-weighted.

## Data & Robust Aggregation
- `Combined_Scytonemin_Concentrations.csv` holds six dose pairs with five replicates each.
- `dose_level_summary.csv` stores 20 % trimmed means, trimmed SDs, and 95 % percentile bootstrap CIs from 2 000 resamples.
- “Total” denotes the direct chromatogram/DAD quantity; oxidized and reduced pools come from independent spectral decompositions and therefore need not sum to the total.
- Baseline corrections make the DAD oxidized control slightly negative. Keep the values, but call out the artefact when reporting them.

## UVA–UVB Structure
- UVA is monotone from 0 to 3.185 mW·cm⁻²; UVB rises alongside UVA through dose₅ (0.768 mW·cm⁻²) and then falls to 0.707 mW·cm⁻² at dose₆.
- The dominant softening in means happens between dose₄ and dose₅ while UVB is still rising; the UVB drop only affects the final step.
- Replicate IDs 1–6 correspond to descending dose order; pair with `aggregated_reflectance/dose_metadata.md` to avoid mislabelling.

## Dose-Level Behaviour
- Chromatogram totals: 0.274 → 0.670 → 0.747 → 0.986 → 0.807 → 0.831 mg·gDW⁻¹.
- Chromatogram reduced peaks at 1.790 mg·gDW⁻¹ (dose₅) and ends at 1.411 mg·gDW⁻¹ (dose₆).
- DAD totals top out at 1.844 mg·gDW⁻¹ (dose₄), decrease to 1.440 mg·gDW⁻¹, then reach 1.585 mg·gDW⁻¹.
- DAD reduced peaks at 2.093 mg·gDW⁻¹ (dose₅) before settling at 1.913 mg·gDW⁻¹.
- WLS slopes for UVA remain positive (chrom total 0.091, chrom reduced 0.390, DAD reduced 0.472 mg·gDW⁻¹·(mW·cm⁻²)⁻¹) but the late decline keeps the pattern “mostly increasing with a late dip.”
- Pearson correlations with UVA span roughly 0.64–0.75 for totals and about 0.69–0.75 for reduced pools (`dose_trend_stats.csv:2-13`).

## Sequential Deltas
`dose_pattern_sequential_deltas.csv` shows the sign patterns: totals follow + + + − +, whereas reduced pools follow + − + + −. Only the first step has a CI that excludes zero; later steps include the null, so curvature must be described cautiously.

## Chromatogram ↔ DAD Alignment
Cross-assay agreement is strong (Pearson r ≥ 0.95, Deming slopes 1.88/2.03/1.09). Either assay can anchor downstream analyses once dose means are established.

## Interpretation & Follow-Up
Robust means expose a pronounced climb through dose₄, a peak at dose₅ for the reduced pools, and a subsequent decline as UVB steps down. Describe the behaviour as “mostly increasing with a late dip,” emphasise the non-additive component definitions, and remember the evidence is exploratory with n = 6 means.

Next steps:
1. Document the handling of negative oxidized values (baseline subtraction or future clipping).
2. Treat `dose_level_summary.csv` as the single source of trimmed means; link to it instead of duplicating CSVs across directories.
3. Align figures and prose so they highlight the mid-course softening and correct sign patterns.
