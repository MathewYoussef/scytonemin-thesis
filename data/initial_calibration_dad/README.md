# DAD-Derived Concentrations (Corrected)

Outputs from the diode array detector (DAD) calibration workflow executed in `Initial_Calibration/Scripts`. The corrected concentration tables align chromatogram-derived and DAD-predicted concentrations with dry biomass metadata.

## Contents
- `DAD_derived_concentrations_corrected.csv` — corrected concentration predictions per sample after blank removal and normalization.
- `DAD_derived_concentrations.csv` — raw model outputs prior to correction.
- `diode_array_auc.csv` — integrated absorbance curves (AUC) for each sample.
- `treatments_corrected_amounts.csv` — treatment-level concentration corrections.
- `sample_id_truth.csv` — mapping between sample IDs, treatments, and UV dose schedule.

See `schema.json` for inferred schemas and `provenance.yaml` for the calibration pipeline lineage.
