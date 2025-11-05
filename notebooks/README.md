# Notebooks Overview

A reviewer can start with these sample-driven notebooks to inspect each pipeline without running the full dataset:

- `00_env_and_schema.ipynb` — pins Python package versions, records dataset schemas, and hashes the `src/` tree before any analysis.
- `initial_calibration_overview.ipynb` — loads sample chromatogram/DAD exports, reproduces Stage A/B aggregation, and plots dose-versus-concentration using trimmed means.
- `01_dosimetry_mdv_benchmark.ipynb` — recomputes UVA/UVB doses from raw Solarmeter readings and verifies %MDV scaling.
- `03_spectra_ingest_and_rel_reflectance.ipynb` — converts raw counts to relative reflectance using dark/white references and writes sample arrays.
- `reflectance_overview.ipynb` — inspects the canonical reflectance tables (precision-weighted totals, dose summaries) and visualises chromatogram vs DAD cross-plots.
- `mamba_validation_overview.ipynb` — examines the validation panel coverage for the Act_of_God Mamba run and mirrors the readiness aggregations.
- `04_mamba_denoising_QC.ipynb` — inspects the shipped denoised spectra, reproduces pass/fail gate summaries, and visualises ΔSNR heatmaps without re-running the model.
- `05_continuum_removal_and_occupancy.ipynb` — performs continuum removal, quadratic bowl fitting, and scytonemin occupancy calculations in the 320–480 and 360–410 nm windows.
- `06_uplc_processing_and_calibration.ipynb` — replays the chromatogram/DAD calibrations with 1/x weighting and derives dry-weight concentrations (mg·gDW⁻¹).
- `07_concentration_profiles_and_cross_assay.ipynb` — summarises dose-response curves with trimmed means/CIs and performs Deming regression between chromatogram and DAD assays.
- `08_reflectance_to_concentration_mapping.ipynb` — maps Σ occupancy (320–480 nm, 360–410 nm) to Chrom_total mg·gDW⁻¹, reproducing the linear fit reported in Figures 12–13.
- `09_geometry_and_orientation_effects.ipynb` — compares occupancy-to-Chrom_total fits across viewing geometries (12 o’clock, 6 o’clock, Σ) to highlight BRDF limitations.

Scripts alongside the notebooks:

- `run_samples.py` — executes lightweight validation pipelines (used by CI `make quickstart`).
- `run_full_pipeline.py` — orchestrates the full reproduction of processed tables.
- `render_figures.py` — regenerates figures for insertion into `scaffold/**/figures`.
