# DAD Feature Extraction & Replicate Exploration

## Purpose
We extend the DAD workflow beyond raw AUC by extracting spectral descriptors (centroid, peak ratios, curvature, FWHM, derivatives) to test whether richer features separate UV dose levels or correlate with chromatogram/biomass-normalized concentrations. This folder tracks experiments, scripts, and results specific to that exploratory work.

## Rationale
- Chromatogram calibrations exhibit a consistent bias (total ≈ 0.5 × oxidized + reduced) and weak correlations with UVA/UVB dose, even after biomass normalization.
- DAD-derived concentrations show similar limitations when relying on AUC alone (Pearson r ≤ ~0.48). Feature engineering may reveal dose-response structure obscured in single-metric models.
- If advanced features remain uninformative, we will evaluate robust means per dose group to understand whether replicate averaging clarifies trends, and document those findings here.

## Planned Steps
1. Implement Stage D feature extraction scripts (centroid, ratios, curvature, FWHM, derivatives) and produce `dad_features.csv`.
2. Fit features vs chromatogram concentrations on standards and assess treatment parity.
3. Compute Pearson/Spearman correlations of features (both concentrations and raw descriptors) vs UVA/UVB metrics.
4. Explore robust mean aggregation at the dose level if replicate scatter overwhelms feature signals.
5. Summarize outcomes and fold successful features back into the main analysis/report.

## Current Outputs
- `dad_features.csv` – per-sample/form feature table (AUC, centroid, λₘₐₓ, FWHM, band/derivative metrics).
- `dad_features_standards_vs_chromatogram.csv`, `dad_features_treatments_vs_chromatogram.csv` – merged feature/concentration tables for regression diagnostics.
- `dad_feature_regression_summary.csv` – linear regression results of each feature against chromatogram concentrations (standards and treatments).
- `dad_feature_dose_correlations.csv` – Pearson/Spearman correlations of feature values vs UVA/UVB descriptors.

## Interim Findings
- Feature regressions mirror AUC performance: for standards, R² for area-based metrics remains ≈0.84–0.89; alternative descriptors (centroid, ratio, derivatives) do not exceed that benchmark.
- Dose correlations across features remain weak (|r| ≤ ~0.48), reinforcing the need to evaluate robust means or additional modeling.
- Inspecting `dad_feature_regression_summary.csv` and `dad_feature_dose_correlations.csv` shows no single-feature transformation (e.g., 384/400 ratio, centroid, FWHM) recovers a clear UVA/UVB trend—the replicate-level scatter is the limiting factor.
- Next steps: investigate robust mean aggregation per dose (median/Huber) to see if averaging suppresses replicate noise; if successful, archive both replicate-level and aggregated analyses for comparison, then proceed to multivariate models (PCR/PLSR). Document each finding in this README and cross-reference in `docs/` once conclusions are drawn.

## Audit Notes
- Keep references to generated scripts, tables, and plots within this directory.
- Record decisions (e.g., feature selection criteria, robust-mean thresholds) for traceability.
- Any conclusions that feed the main report should be cross-linked in `docs/` once validated.
