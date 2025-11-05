# Diode-Array Calibration Notes

## Data Preparation
- Blank spectra were missing from the tidy export and are now appended via `Scripts/append_blank_spectra.py`. The script ingests `DAD_RAW_FILES/Youssef_Abs_Spectra_per_pigment_3_10_25.xls` (Sheet1) and adds three blank replicates (`BLANK`, `BLANK.1`, `BLANK.2`) to `Compiled_DAD_DATA/Scytonemin/scytonemin_spectra_tidy.csv` with `sample_category="blank"`.
- `Scripts/compute_dad_auc.py` integrates absorbance between 320–480 nm for all scytonemin spectra, storing results (raw AUC, optional blank-corrected AUC, wavelength coverage) in `Diode_Array_AUC/diode_array_auc.csv`. With `spectral.subtract_blank_auc=true`, the current calibration uses blank-subtracted AUC values (Blanks → 0).

## Calibration
- `Scripts/fit_dad_calibration.py` reads the AUC table, merges known standard concentrations from `DAD_RAW_FILES/scytonemin_standard_concentrations.csv`, and fits forced-origin linear regressions per form (oxidized/reduced/total) using the configuration `modeling.weighting` (currently `1/x`).
- Outputs per form are written to `Diode_Array_Derived_Calibration_Plots/`:
  - `calibration_{form}.json` with slope, SE, R², RSS, max relative residual, and LOD/LOQ estimates (derived from blank AUC SD).
  - `standards_fitted_{form}.csv` listing blank-corrected AUC, predicted concentration, and residuals for each standard level.
  - `calibration_{form}.png` (PNG when matplotlib is available; falls back to SVG) showing AUC vs concentration with the forced-origin fit.
- `Scripts/predict_dad_concentrations.py` generates `DAD_derived_concentrations.csv` (treatment metadata, per-form AUCs, and predicted concentrations).
- `Scripts/plot_dad_treatments.py` overlays treatment predictions onto the calibration curves; outputs are saved as `{form}_treatment_overlay_grouped_by_uva.{csv|png}` when matplotlib is available (fallback: SVG).

## Current Fit Metrics (mg/mL vs corrected AUC)
| Form     | Slope (×10⁻⁸) | Intercept (mg/mL) | R²     | Max |relative residual| |
|----------|----------------|-------------------|--------|-------------------------|
| Total    | 4.526          | -0.0240 ± 0.0245  | 0.908  | 0.307                   |
| Oxidized | 3.007          | -0.0342 ± 0.0279  | 0.931  | 0.266                   |
| Reduced  | 4.711          | -0.0217 ± 0.0244  | 0.979  | 0.469                   |

Notes:
- Forced-origin fits expose larger scatter than chromatogram-based calibrations; oxidized/reduced show non-trivial residuals (>100%).
- Allowing the intercept to float shrinks residuals dramatically (compare to forced-origin fits with R² ≈ 0.91–0.98 and max |rel residual| ≥ 0.74). The fitted intercepts are small (~0.02–0.03 mg/mL) and within ~1 SE of zero, reflecting residual baseline absorbance after subtracting blanks.
- Blank AUC SD remains large (~4.66×10⁴), so LOD/LOQ estimates derived from raw blanks are extremely high. Investigate blank acquisition consistency if tighter detection limits are desired.
- Even though each form is calibrated independently, treatment predictions in `DAD_derived_concentrations.csv` show the same pattern as the chromatograms: `predicted_oxidized + predicted_reduced` roughly doubles `predicted_total` (e.g., sample `1D`: 0.0499 + 0.1548 ≈ 0.205 > 0.1137 mg/mL). This reinforces that the total peak should be interpreted alongside the shared-fit QC rather than as an absolute truth.
- A no-blank variant (`analysis_config_no_blank.yaml`) was run for comparison; slopes, R², and residuals were indistinguishable from the blank-subtracted fits (intercepts shifted by ≤3×10⁻³ mg/mL and treatment predictions changed by ≤5×10⁻⁵ mg/mL). Because the blank variance remains high in either case, we retain the blank-subtracted workflow but treat LOQ/LOD as “not determined” pending more stable blanks.
- Dose-response correlations mirror the chromatogram findings: Pearson r peaks around 0.48 (reduced mg/mL vs UVB) and mg/gDW values stay ≤0.43 (`Diode_Array_Derived_Calibration_Plots/dad_dose_correlations.csv`), so AUC alone does not cleanly separate doses.
- ANOVA and Kruskal–Wallis across the six dose groups (30 observations per form) still produce p-values ≥ 0.09 (`_concentrations_vs_dose_with_robust_mean/dad_dose_stats.csv`); DAD-derived concentrations do not show a statistically significant dose effect under current replicates.
- Diode-array spectra integrate everything in the 320–480 nm window (solvent, other pigments, instrument drift). Even a scytonemin-free sample carries residual absorbance, so a non-zero intercept is biologically plausible and statistically justified. Forcing the calibration through the origin would instead bias the slope and inflate residuals.
- Calibration intercept policy follows the guidance summarised in `docs/calibration_intercept_guidance.md` (EPA 8000D, ICH Q2(R2), ISO 8466, Eurachem): retain the intercept unless blanks and low-end standards prove a zero intercept model improves accuracy across the range.
- All intermediate artefacts (augmented tidy spectra, AUC table, calibration CSV/JSON/SVG) are versioned in-repo for traceability.
