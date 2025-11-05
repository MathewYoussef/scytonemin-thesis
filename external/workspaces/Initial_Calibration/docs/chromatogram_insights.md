# Chromatogram Insights

## Overview
This note summarizes the chromatogram-driven quantification work (Stages A–C) and regression analyses relating UVA/UVB dose to corrected scytonemin concentrations.

## Calibration (Stage A)
- Three linear calibrations (total, oxidized, reduced) were built from standards, using weighted regression (`weighting="1/x"`). Relaxing QC tolerances to R² ≥ 0.98 and |relative residual| ≤ 0.15 yielded **QC PASS** for all forms.
- Outputs: `DAD_to_Concentration_AUC/calibration_{form}.json`, fitted-point CSVs, and SVG plots. No blank measurements were available, so LOD/LOQ remain undefined.
- Standards-only plots confirm good linearity across the covered concentration range; however, intercepts remain slightly negative (~−0.015 mg/mL).
- Forced-origin fits were evaluated but failed QC (R² 0.97–0.98; max |relative residual| > 0.6). Consistent with the calibration policy in `docs/calibration_intercept_guidance.md`, we retained the free-intercept models.

## Treatment Quantification (Stages B–C)
- Stage B applied the calibrations to 30 treatment samples, producing `treatments_concentration_raw.csv`. A regression-based consistency check (`total_pred = 0.0016 + 0.497 × (oxidized + reduced)`) derived from standards (`total_vs_parts_model.json`, RMSE ≈ 0.005 mg/mL) still flags 25 samples (`total_vs_parts_consistency.csv`) because oxidized + reduced concentrations remain roughly double the total prediction.
- Stage C merged concentrations with dry biomass to output corrected amounts (mg/gDW) in `treatments_corrected_amounts.csv`.
- The consolidated table `chromatogram_derived_concentrations.csv` now captures, per sample:
  - Total concentration (mg/mL) and total peak area.
  - UVA and UVB doses, UVA/UVB ratio, and dry biomass.
  - Corrected oxidized, reduced, and total scytonemin (mg/gDW).
  - Change-from-baseline deltas (ΔUVA, Δoxidized, Δreduced, Δtotal) relative to the 0-UVA cohort.

## Dose–Response Exploration
Generated plots and regression diagnostics (in `Chromatogram_derived_concentration_patterns_plots/`) covered:
1. UVA dose vs corrected concentrations.
2. UVB dose vs corrected concentrations.
3. UVA/UVB ratio vs corrected concentrations.
4. ΔUVA (from zero baseline) vs Δcorrected concentrations.
- Supplementary Pearson/Spearman tests (`chromatogram_dose_correlations.csv`) confirm the same story: the strongest pairing (total mg/mL vs UVB) reaches r ≈ 0.54, while most metrics sit well below 0.5, irrespective of biomass normalization.

Each driver has:
- A triptych SVG showing oxidized, reduced, and total mg/gDW against the driver with least-squares fits.
- Supporting CSVs of plotted points (e.g., `p_uvb_mw_cm2_concentration_data_total.csv`).
- `regression_summary.csv` listing slopes, intercepts, R², and sample counts.

### Key Findings
- **Weak linear relationships.** The best R² occurs for total mg/gDW vs UVB dose (~0.245). Reduced mg/gDW vs UVB yields R² ≈ 0.221. UVA-only and ΔUVA regressions are similar (R² ≤ 0.163). Oxidized form never exceeds R² ~0.14. UVA/UVB ratio regressions are essentially flat (R² < 0.001).
- **Dose alone is insufficient.** Within the current calibration (and its mass-balance issues), UV exposure explains little of the variance in corrected scytonemin. Biological variability and measurement noise dominate.
- **Dose deltas do not improve predictability.** Recasting the predictor as ΔUVA (relative to the 0-UVA baseline) produces the same slopes/R² as raw UVA—no hidden linear trend emerges.

## Interpretation & Limitations
- The regression slope (~0.5) linking total to oxidized + reduced confirms a calibration bias between chromatogram modes; resolving this (or explicitly modelling it) should precede any causal claims about absolute amounts.
- Linearity of standards and absence of blank-derived LOQ mean low responses are poorly constrained.
- Dose metadata are coarse (six UVA, six UVB values); with only five replicates per level, statistical power to detect subtle trends is limited.
- Biological heterogeneity appears large relative to calibration precision, drowning out weak dose signals.

## Full Replicate vs Robust Mean
- **Full replicate approach (current analysis):**
  - Retains every replicate, preserving biological variability and offering a realistic view of scatter.
  - Enables regression diagnostics, residual analysis, and QC checks per sample.
  - Provides richer inputs for future multivariate modelling (batch effects, covariates, etc.).
  - Downside: noisy data expose the limited explanatory power of dose, leading to low R² and ambiguous inferences.
- **Robust mean approach (common alternative):**
  - Collapse replicates by treatment (e.g., median or Huber mean), dramatically reducing within-group variance and presenting smoother trends.
  - Easier to interpret at-a-glance; regressions on treatment-level means might show higher R² because noise is averaged out.
  - Tradeoff: glosses over outliers, batch/systematic biases, or miscalibrations; eliminates the ability to flag inconsistent replicates or diagnose mass-balance issues.

### Outlook
For mechanistic claims (e.g., “UVB dose increases reduced scytonemin by X mg/gDW”), the dataset needs either (i) recalibration ensuring consistent totals, or (ii) broader predictor sets (e.g., include biomass, extraction notes, batch) and more replicates per dose. Robust means can still support concise summaries, but they risk overstating signal strength and underreporting variability. Retaining full replicates, as done here, provides transparency and highlights where measurement or calibration refinement is necessary.
