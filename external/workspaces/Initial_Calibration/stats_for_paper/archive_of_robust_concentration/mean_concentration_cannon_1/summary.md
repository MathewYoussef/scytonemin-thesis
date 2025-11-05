# Dose-Level Robust Mean Findings

## Objective
Replicate-level models (n = 30) did not recover convincing UVA/UVB effects, so we collapsed each UV dose to a robust mean (20 % trimmed) to ask whether mean-level pigment concentrations change with exposure once biological variance is muted.

## Data & Robust Aggregation
- Replicate concentrations live in `Combined_Scytonemin_Concentrations.csv` (six UVA/UVB pairs × five replicates).
- Dose summaries (`dose_level_summary.csv`) hold 20 % trimmed means, trimmed SDs, and 95 % percentile bootstrap intervals computed from 2 000 resamples (middle three replicates retained).
- The columns labelled “total” come directly from chromatogram/DAD quantification; oxidized and reduced pools are derived from independent spectral components and are **not** constrained to sum to the total, which explains reduced values exceeding totals.
- DAD oxidized values occasionally dip slightly below zero at the control because the spectra are baseline corrected. We retain those estimates but flag them for follow-up.

## UVA–UVB Structure
- UVA increases monotonically from 0 to 3.185 mW·cm⁻²; UVB climbs in tandem through dose₅ (0.768 mW·cm⁻²) and then steps down to 0.707 mW·cm⁻² at dose₆ (`aggregated_reflectance/dose_metadata.md:2-7`).
- The principal softening in pigment means occurs between dose₄ and dose₅ while both UVA and UVB are still increasing; the subsequent UVB drop only affects the final step.
- Sample IDs in the replicate sheet run in descending dose order (1* ≈ dose₆ … 6* ≈ dose₁); the dose metadata table should be referenced when joining to avoid rank inversions.

## Dose-Level Behaviour
- Chromatogram totals climb from 0.274 mg·gDW⁻¹ (dose₁) to 0.986 mg·gDW⁻¹ (dose₄), dip to 0.807 mg·gDW⁻¹ (dose₅), and settle near 0.831 mg·gDW⁻¹ at dose₆.
- Chromatogram reduced means peak at 1.790 mg·gDW⁻¹ (dose₅) before falling to 1.411 mg·gDW⁻¹ (dose₆); totals and oxidized components never exceed this peak.
- DAD totals reach 1.844 mg·gDW⁻¹ at dose₄, drop to 1.440 mg·gDW⁻¹, and partially rebound to 1.585 mg·gDW⁻¹. DAD reduced means crest at 2.093 mg·gDW⁻¹ (dose₅) and decline to 1.913 mg·gDW⁻¹ at dose₆.
- Weighted linear fits against UVA (weights = 1/trimmed-SD²) therefore show directionally positive slopes—chrom total 0.091, chrom reduced 0.390, DAD reduced 0.472 mg·gDW⁻¹·(mW·cm⁻²)⁻¹—but the intervals are wide and the late decline keeps the trend “mostly increasing” rather than strictly monotonic (`dose_trend_stats.csv:2-13`).
- Pearson correlations with UVA span ~0.64–0.75 for totals and ~0.69–0.75 for reduced pools; the DAD reduced lower bound sits at 0.69 (`dose_trend_stats.csv:6,12`).

## Sequential Deltas
- Bootstrap deltas (`dose_pattern_sequential_deltas.csv`) follow + + + − + for chromatogram totals, + − + + − for chromatogram reduced, and an analogous pattern for DAD totals/reduced. The strong positive jump between doses₁→₂ (e.g., +0.60 mg·gDW⁻¹ for chrom reduced) is the only step whose 95 % CI excludes zero; later steps overlap zero, so curvature should be interpreted cautiously.

## Chromatogram ↔ DAD Alignment
- Dose-level trimmed means are tightly aligned: Pearson r ≥ 0.95 for totals and ≥ 0.94 for oxidized/reduced, with Deming slopes of 1.88 (total), 2.03 (oxidized), and 1.09 (reduced) and intercepts that straddle zero (`chrom_dad_alignment.csv:1-3`).

## Interpretation and Follow-Up
Collapsing to robust means exposes a clear rise through dose₄ and an even stronger peak for the reduced pools at dose₅, followed by a late decline once UVB steps down. The evidence is exploratory (n = 6 dose means), but it supports a “mostly increasing with a late dip” interpretation rather than a fully monotonic response.

Next steps:
1. Document the baseline-correction logic (or clip negatives) for oxidized DAD estimates.
2. Designate `dose_level_summary.csv` as the canonical trimmed-mean table and reference it elsewhere instead of shipping duplicate CSV copies.
3. Harmonise manuscript language so it mirrors the mid-course softening and the non-additive definition of “total” versus oxidized/reduced pools.
