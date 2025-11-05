# Dose-Level Robust Mean Findings

## Objective
Our replicate-level models (n = 30) struggled to detect dose effects, so we reframed the experiment around 20 % trimmed means to ask whether mean pigment concentrations change with UVA/UVB once biological spread is suppressed.

## Data & Robust Aggregation
- Replicates: `Combined_Scytonemin_Concentrations.csv` (six dose pairs × five replicates).
- Trimmed summaries: `dose_level_summary.csv` with 20 % trimmed means, trimmed SDs, and 95 % percentile bootstrap intervals generated from 2 000 resamples.
- “Total” columns represent the direct chromatogram/DAD quantification. Oxidized and reduced pools are determined from spectral components and are not additive; exceeding the total is expected.
- Baseline correction pushes the DAD oxidized control slightly below zero; those values remain in the table but require a short note when presented.

## UVA–UVB Structure
- UVA advances from 0 to 3.185 mW·cm⁻² while UVB rises from 0 to 0.768 mW·cm⁻², then drops to 0.707 mW·cm⁻² at the last dose (`aggregated_reflectance/dose_metadata.md:2-7`).
- The mean-level softening appears between dose₄ and dose₅ even though both UV components are still increasing; the UVB reversal only happens on the final step.
- Sample IDs in the replicate sheet run opposite the dose index (1* corresponds to dose₆). Cross-reference the metadata table during joins.

## Dose-Level Behaviour
- Chromatogram totals: 0.274 → 0.670 → 0.747 → 0.986 → 0.807 → 0.831 mg·gDW⁻¹.
- Chromatogram reduced peaks at 1.790 mg·gDW⁻¹ (dose₅) before falling to 1.411 mg·gDW⁻¹ (dose₆).
- DAD totals crest at 1.844 mg·gDW⁻¹ (dose₄), dip to 1.440 mg·gDW⁻¹, and finish at 1.585 mg·gDW⁻¹. DAD reduced reaches 2.093 mg·gDW⁻¹ (dose₅) and declines to 1.913 mg·gDW⁻¹.
- Weighted linear fits against UVA remain positive (chrom total 0.091, chrom reduced 0.390, DAD reduced 0.472 mg·gDW⁻¹·(mW·cm⁻²)⁻¹) but wide intervals coupled with the late decline mean the pattern is “mostly increasing with a late dip.”
- Pearson r with UVA spans roughly 0.64–0.75 for totals and 0.69–0.75 for reduced pools (`dose_trend_stats.csv:2-13`).

## Sequential Deltas
Bootstrap deltas (`dose_pattern_sequential_deltas.csv`) emphasise the early jump—dose₁→₂ is strongly positive across metrics—followed by modest changes whose intervals overlap zero. Reduced pools follow a + − + + − sign pattern; totals follow + + + − +, underscoring the need for nuanced language.

## Chromatogram ↔ DAD Alignment
Chromatogram and DAD trimmed means remain tightly coupled (Pearson r ≥ 0.95, Deming slopes 1.88/2.03/1.09 for total/oxidized/reduced), so either assay can stand in once mean values are established.

## Interpretation & Next Steps
Robust means reveal a pronounced rise through dose₄, stronger peaks for the reduced pools at dose₅, and a late decline once UVB steps down. Treat the evidence as exploratory (n = 6 means) and describe the trend as “mostly increasing with a late dip.”

Action items:
1. Annotate the oxidized baseline behaviour (negative control) before external release.
2. Use `dose_level_summary.csv` as the single source for trimmed means; reference it from other directories instead of keeping redundant copies.
3. Harmonise manuscript prose, figures, and captions so they all highlight the mid-course dip and the non-additive definition of “total.”
