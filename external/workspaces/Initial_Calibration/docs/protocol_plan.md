# UPLC-DAD Scytonemin Analysis Implementation Plan

## Confirmed Parameters
- Extraction volume: 1.0 mL (chemistry.extraction_volume_ml)
- Dilution factors encoded via calibration standards (see `DAD_RAW_FILES/scytonemin_standard_concentrations.csv`)
- Reflectance source: `Synthetic_Reflectance_data/raw_normalized`

## Key Data Assets
- Chromatogram areas (total/oxidized/reduced): `DAD_RAW_FILES/scytonemin_chromatogram_areas/`
- Standard concentration table: `DAD_RAW_FILES/scytonemin_standard_concentrations.csv`
- DAD spectra (long format): `Compiled_DAD_DATA/Scytonemin/scytonemin_spectra_tidy.csv`
- Biomass and dose metadata inputs: `DAD_RAW_FILES/randomization_by_column.csv`, `DAD_RAW_FILES/chamber_dose_schedule.csv`, `DAD_RAW_FILES/sample_biomass.csv`
- Reflectance spectra: `Synthetic_Reflectance_data/raw_normalized/**/normalized.csv`

## Config Skeleton (`analysis_config.yaml`)
```yaml
paths:
  chrom_areas_dir: "${REPO_ROOT}/DAD_RAW_FILES/scytonemin_chromatogram_areas"
  dad_spectra_csv: "${REPO_ROOT}/Compiled_DAD_DATA/Scytonemin/scytonemin_spectra_tidy.csv"
  reflectance_root: "${REPO_ROOT}/Synthetic_Reflectance_data/raw_normalized"
  biomass_csv: "${REPO_ROOT}/DAD_RAW_FILES/sample_biomass.csv"
  chamber_csv: "${REPO_ROOT}/DAD_RAW_FILES/chamber_dose_schedule.csv"
  randomization_csv: "${REPO_ROOT}/DAD_RAW_FILES/randomization_by_column.csv"
  standards_csv: "${REPO_ROOT}/DAD_RAW_FILES/scytonemin_standard_concentrations.csv"
  outputs:
    calibrations_dir: "${REPO_ROOT}/DAD_to_Concentration_AUC"
    chromatogram_plots_dir: "${REPO_ROOT}/Chromatogram_Calibration_Plots"
    reflectance_plots_dir: "${REPO_ROOT}/Reflectance_to_Concentration_Plots"
    dad_auc_dir: "${REPO_ROOT}/Diode_Array_AUC"
    dad_calibration_plots_dir: "${REPO_ROOT}/Diode_Array_Derived_Calibration_Plots"

scytonemin:
  forms: ["total", "oxidized", "reduced"]
  expected_rt_min: {total: 3.5, oxidized: 3.1, reduced: 3.9}
  rt_tolerance_min: 0.2

chemistry:
  extraction_volume_ml: 1.0
  dilution_factor: 1.0

spectral:
  wl_min_nm: 320
  wl_max_nm: 480
  wl_step_nm: 1
  sg_window: 11
  sg_polyorder: 3

modeling:
  use_weighted_regression: true
  weighting: "1/x"
  kfold: 5

qc:
  calib_r2_min: 0.98
  max_abs_rel_residual: 0.15
  total_vs_parts_tolerance: 0.05
```

## Implementation Stages
1. **Data contracts**: Harmonize chromatogram, DAD, reflectance, and biomass metadata per schema; rerun `Scripts/sample_ID_truth.py` to produce `Compiled_DAD_DATA/Scytonemin/sample_id_truth.csv`.
2. **Calibration (Stage A)**: Fit per-form Beer-Lambert models using standards + weights; archive coefficients in `calibration_{form}.json` and diagnostics in plots & CSVs. Incorporate concentrations from `scytonemin_standard_concentrations.csv` where vendor export lacks values.
3. **Treatment quantification (Stage B)**: Apply calibrations to treatment areas; propagate SEs, enforce total vs (oxidized+reduced) tolerance, label `<LOQ` using Stage A results.
4. **Biomass normalization (Stage C)**: Merge concentrations with dry mass from truth table; compute mg/gDW using extraction volume, output corrected tables.
5. **Diode-array calibration staging**: Extract blank spectra from the XLS workbook, append to the tidy CSV, integrate 320–480 nm AUCs into `Diode_Array_AUC/diode_array_auc.csv`, and fit forced-origin calibration curves (outputs in `Diode_Array_Derived_Calibration_Plots/`).
6. **DAD feature modeling (Stages D–E)**: Resample spectra to config grid, compute feature suite, validate vs chromatogram concentrations (standards and treatments), then relate to biomass-normalized amounts.
7. **Reflectance linkage (Stage F)**: Aggregate raw-normalized reflectance spectra by treatment group, compute analogous features (including pseudo-absorbance), correlate with group-level corrected amounts and dose metadata.
8. **Advanced modeling (Stage G)**: Deploy derivative indices and PCR/PLSR only after QC gates pass; document settings and CV diagnostics.
9. **QC & reporting**: Automate thresholds (R², residuals, mg/gDW completeness); compile markdown/PDF report summarizing config, QC outcomes, calibrations, spectral relationships, and interpretive guidance.

## Outstanding Checks
- Confirm whether chromatogram exports already include retention time drift flags.
- Decide if vendor DAD files need re-import for future batches (store instructions alongside this plan if yes).
