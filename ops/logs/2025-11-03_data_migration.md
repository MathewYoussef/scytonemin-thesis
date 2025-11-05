# Data Migration Log — 2025-11-03

## Summary
- Imported Initial Calibration raw instrument exports, intermediate AUC dumps, and DAD-to-concentration staging directories into `data/raw/initial_calibration/`.
- Captured Initial Calibration reference tables (calibration JSON/CSV, configs, tidy spectra) in `data/reference/initial_calibration/` and lifted Stage scripts into `src/chromatography/`.
- Migrated Reflectance denoised spectra and upstream staging data into `data/raw/reflectance/`, canonical exports into `data/reference/reflectance/`, modules into `src/reflectance/`, supporting IMAS helpers into `src/imas_pipeline/`, and seeded scaffold figures/tests.
- Brought Act-of-God/Mamba SSM spectra, staging dumps, manifests, metadata, checkpoints, configs, scripts, and evaluation outputs into the new `data/raw/mamba_ssm/`, `data/reference/mamba_ssm/`, `models/mamba_ssm/`, `src/mamba_ssm/`, and `ops/output/data/mamba_checks/` destinations. Scaffold now contains ROI and plot series figures.

## Detailed Actions
- `rsync` copy: `Initial_Calibration/DAD_RAW_FILES/`, `Compiled_DAD_DATA/`, `Diode_Array_AUC*/`, `DAD_to_Concentration_AUC/` → `data/raw/initial_calibration/`.
- Reference tables/configs (*.csv, *.json, *.yaml) from `Initial_Calibration/` → `data/reference/initial_calibration/`.
- Stage scripts from `Initial_Calibration/Scripts/` → `src/chromatography/` (includes `run_stage_a.py`, `run_stage_bc.py`, utilities, plotting helpers).
- Reflectance canonical outputs (`reflectance/canonical_dataset/`), crosswalk tables, and modules/tests moved into `data/reference/reflectance/`, `src/reflectance/`, and `tests/reflectance/verify_canonical_dataset.py`.
- Raw reflectance staging directories (`denoised_full_run_staging/`, `from_dose_concentration_chromo_dad_work_upstream/`) → `data/raw/reflectance/`; IMAS helpers → `src/imas_pipeline/`; figures (`chrom_vs_dad_outputs/`) → `scaffold/reflectance/figures/`.
- Act-of-God assets: copied spectra staging (`data/spectra_for_fold/`, `_staging/`, `denoised_full_run_staging/`) → `data/raw/mamba_ssm/`; metadata/manifests/panel summaries → `data/reference/mamba_ssm/`; checkpoints → `models/mamba_ssm/`; configs + scripts + modules → `src/mamba_ssm/`; panel eval + logs → `ops/output/data/mamba_checks/`; ROI & plot series figures → `scaffold/mamba_ssm/figures/`.

## Follow-up
- Add provenance `README.md` + schema/checksum entries in each `data/raw/**` and `data/reference/**` directory.
- Refactor migrated scripts into package entry-points (`make reproduce`) and wire tests (e.g., port Stage A/B assertions, reflectance canonical dataset validation).
- Confirm large directories commit cleanly via Git (no DVC layer) and monitor repo size before pushing upstream.
- Review scaffold figure drops and trim superseded plots; document any omitted large logs in this log file or a follow-up entry.
- Mamba denoising is **not** re-run by default; ship checkpoints + evaluation artefacts and note that auditors may rerun `evaluate_validation_panel` only if they opt into installing Torch/mamba-ssm locally.
