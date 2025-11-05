# Dose–Concentration Results Summary (mg/gDW unless noted)

This document synthesizes the dose-level analyses derived from `Combined_Scytonemin_Concentrations.csv` and the robust aggregation pipeline in `mean_concentration_to_dose_2/`.

## Key data products

- `dose_level_summary.csv` — 20 % trimmed means, trimmed SDs, bootstrap estimates, and 95 % CIs for total/oxidized/reduced scytonemin (chromatogram & DAD) across the six UVA/UVB doses.
- `dose_level_relationships.csv` — Pearson/Spearman/Kendall correlations and weighted least-squares (WLS) fits against dose rank.
- `chrom_dad_alignment.csv` — Deming regression comparing chromatogram and DAD dose means.
- `dose_level_summary_stats.md` — human-readable excerpt of slopes, confidence intervals, and assay-alignment metrics.
- `plots/` — replicate stripplots with trimmed mean ± 95 % bootstrap CI overlays and annotated WLS statistics.

## Dose ordering

| rank | dose_id | UVA (mW·cm⁻²) | UVB (mW·cm⁻²) | note |
|------|---------|---------------|---------------|------|
| 1 | D1 | 0.000 | 0.000 | dark control |
| 2 | D2 | 0.647 | 0.246 | lowest UV exposure |
| 3 | D3 | 1.095 | 0.338 | — |
| 4 | D4 | 1.692 | 0.584 | — |
| 5 | D5 | 2.488 | 0.768 | maximal UVB |
| 6 | D6 | 3.185 | 0.707 | highest UVA but UVB dips below rank 5 |

The diagonal design (each UVA level paired with a single UVB level) yields strong co-correlation (r ≈ 0.95) but not strict monotonicity: rank 6 has lower UVB than rank 5.

## Replicate-level vs. dose-level perspectives

- **Replicates (n = 30)** retain biological variance. Classical ANOVA, permutation tests, and ridge/PLS fits (see existing `stats_for_paper` results) show no robust main effects or interactions after accounting for severe collinearity.
- **Dose-level robust means (n = 6)** collapse within-dose variability. 20 % trimmed means indicate latent monotone trends, especially for the reduced component. Treat p-values as descriptive given the small sample of independent doses.

## Dose–response findings

Weighted least-squares slopes use weights = 1 / bootstrap variance of the trimmed mean (see `dose_level_summary_stats.md` for full detail).

| Metric | WLS slope (95 % CI) | p (slope) | Spearman r (p) | Interpretation |
|--------|--------------------|-----------|-----------------|----------------|
| Chromatogram total | 0.071 (−0.034, 0.175) | 0.134 | 0.829 (0.042) | Mean-level increase, but CI includes zero once weights applied. |
| Chromatogram oxidized | 0.026 (−0.047, 0.099) | 0.377 | 0.371 (0.468) | Weak evidence of a monotone pattern. |
| Chromatogram reduced | 0.186 (0.015, 0.356) | 0.039 | 0.771 (0.072) | Clear positive trend; reduced fraction drives the total. |
| DAD total | 0.125 (−0.119, 0.369) | 0.228 | 0.714 (0.111) | Similar incline to chromatogram total; CI spans zero. |
| DAD oxidized | 0.079 (−0.052, 0.210) | 0.169 | 0.371 (0.468) | Mirrors chromatogram oxidized variability. |
| DAD reduced | 0.250 (0.028, 0.473) | 0.035 | 0.543 (0.266) | Strongest dose-level increase among DAD metrics. |

**Visual evidence:** see `plots/chrom_reduced_dose_strip.png` and `plots/dad_reduced_dose_strip.png` — trimmed means rise across dose ranks with 95 % CI entirely above zero.

## Chromatogram vs. DAD alignment

Deming regression (error ratio λ based on average trimmed variances):

| Pair | λ | Slope (95 % CI) | Intercept (95 % CI) | Notes |
|------|---|-----------------|---------------------|-------|
| Total | 0.229 | 1.885 (1.119, 2.186) | 0.026 (−0.255, 0.646) | DAD totals ≈ 1.9× chromatogram totals; intercept near zero. |
| Oxidized | 0.426 | 2.052 (1.430, 2.917) | −0.209 (−0.376, 0.058) | Higher scaling but consistent ranking. |
| Reduced | 1.148 | 1.090 (0.375, 1.408) | 0.338 (0.021, 1.430) | Close to 1:1 scaling; reduced forms align strongly. |

## Shape-aware pattern checks

The UVA/UVB ladder produces more nuance than a purely monotone increase. We quantified several complementary views; all artefacts live in `mean_concentration_to_dose_2/`.

### 1. Quadratic (shape-aware) fits

- Files: `dose_level_polyfits.csv`, `dose_level_poly_coeffs.csv`.
- Method: quadratic regression (degree‑2 polynomial, fit with intercept incorporated) with 1 000 bootstrap resamples for pointwise 95 % prediction bands.
- Observations:
  - All metrics share a concave-down profile (`coef_quadratic < 0`; e.g., total chromatogram −0.051, DAD total −0.101), reflecting the slight dip at the UVB-heavy D5 → D6 transition.
  - Reduced forms peak around ranks 3–4 (UVA ≈ 1.1–1.7 mW·cm⁻²) before flattening.
- Visuals: see `plots/*_poly_delta.png` for overlays combining trimmed means, quadratic fits (with bands), and delta fingerprints.

### 2. Permutation against flat baseline

- File: `dose_level_permutation_quadratic.csv`.
- We compared the quadratic RSS to a constant (flat-dose) model under 1 000 permutations of the six means (preserving the diagonal design).
- Key results:
  - Chromatogram total: ΔRSS = 0.268, p_perm ≈ 0.026.
  - DAD total: ΔRSS = 0.859, p_perm ≈ 0.050.
  - Reduced metrics show moderate evidence (p ≈ 0.089 for chromatogram, 0.103 for DAD).
- Interpretation: total concentrations display statistically detectable curvature relative to a flat response, consistent with an “increase–flatten–dip” pattern.

### 3. Adjacent dose deltas (fingerprints)

- Files: `dose_level_deltas.csv`, `dose_level_delta_patterns.csv`.
- For each dose step, we bootstrap 20 % trimmed means (1 000 draws) and report Δ(D_i→D_{i+1}) with 95 % CIs and the probability the change is positive/negative.
- Highlights:
  - Early steps (D1→D2) are strongly positive for reduced forms (chrom: Δ ≈ +0.60, CI > 0; DAD: Δ ≈ +0.89, CI > 0).
  - Transitions around D4→D5 frequently go negative (UVB peak), with > 50 % of bootstrap replicates reporting a drop.
  - `dose_level_delta_patterns.csv` shows the chance of at least one negative step is high (> 0.98 for all metrics), while “exactly one negative” occurs about 19–47 % of the time depending on the assay/form.
- Takeaway: the canonical fingerprint is “large rise at low dose, plateau/decline near the UVB-heavy step, mild recovery at the highest UVA dose.”

### 4. Order-restricted diagnostics with slack

- Files: `dose_level_order_tests.csv` (Jonckheere–Terpstra with optional single slack), `dose_level_monotone_slack.csv`, `dose_level_delta_patterns.csv`.
- Results:
  - Without slack, JT z-scores sit below ≈1.0 (e.g., chrom_total z ≈ 0.96, p ≈ 0.17).
  - Allowing one interior dose to float free (drop D5 for totals, drop D2 for chrom_reduced) reduces p-values to the 0.12–0.19 range, still descriptive but consistent with a limited reversal rather than pervasive noise.
  - Bootstrap pattern table confirms ≥ 1 negative step occurs in > 98 % of draws for every metric, cementing the “single dip” motif.
- Interpretation: a monotone increase with one slack (near the UVB-heavy D5) matches the data better than strict monotonicity, but evidence remains exploratory (p ≳ 0.12).

### Summary for downstream use

- Total concentrations (both assays) exhibit significant curvature relative to a flat response (permutation p ≲ 0.05) and show large positive deltas at low dose followed by a UVB-associated dip.
- Reduced fractions provide the clearest monotone component (positive WLS slope with CI > 0) but still display the mid-dose sag in the delta analysis.
- These fingerprints supply concrete patterns to compare against reflectance-derived signals (e.g., do reflectance features follow the same rise–dip–rebound sequence).

### Next steps

- Implement a formal order-restricted test with one slack (Tukey short-cut or Hettmansperger–Norton) and integrate results.
- Produce plots overlaying the quadratic fits and delta fingerprints alongside the replicate stripplots for manuscript figures.
- Apply the same analyses to reflectance-derived concentrations to test whether the spectral proxies inherit the dose pattern quantified here.
