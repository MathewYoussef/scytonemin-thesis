# Initial Calibration — Raw & Intermediate Assets

Raw instrument exports and intermediate artifacts copied from `UPLC_DAD_Scytonemin_Concentrations/Initial_Calibration`. These inputs are the source of truth for rebuilding Stage A/B calibrations inside `src/chromatography/`.

## Contents
- `DAD_RAW_FILES/` — original chromatogram integrals, biomass metadata, randomisation tables, and standard concentration sheets.
- `Compiled_DAD_DATA/` — harmonised diode-array spectral tables generated during the original workflow.
- `Diode_Array_AUC/` & `Diode_Array_AUC_no_blank/` — absorbance integrals used when fitting and validating the DAD calibrations.
- `DAD_to_Concentration_AUC/` — staging area targeted by Stage B/C scripts for treatment-level concentration outputs.

The `analysis_config.yaml` under `data/reference/initial_calibration/` references these paths via relative locations. No files in this directory are modified in-place by the new pipeline; regenerated results should land in `data/reference/**` or `scaffold/**`.
