# Notebooks Overview

A reviewer can start with these sample-driven notebooks to inspect each pipeline without running the full dataset:

- `00_env_and_schema.ipynb` — pins Python package versions, records dataset schemas, and hashes the `src/` tree before any analysis.
- `initial_calibration_overview.ipynb` — loads sample chromatogram/DAD exports, reproduces Stage A/B aggregation, and plots dose-versus-concentration using trimmed means.
- `01_dosimetry_mdv_benchmark.ipynb` — recomputes UVA/UVB doses from raw Solarmeter readings and verifies %MDV scaling.
- `03_spectra_ingest_and_rel_reflectance.ipynb` — converts raw counts to relative reflectance using dark/white references and writes sample arrays.
- `reflectance_overview.ipynb` — inspects the canonical reflectance tables (precision-weighted totals, dose summaries) and visualises chromatogram vs DAD cross-plots.
- `mamba_validation_overview.ipynb` — examines the validation panel coverage for the Act_of_God Mamba run and mirrors the readiness aggregations.

Scripts alongside the notebooks:

- `run_samples.py` — executes lightweight validation pipelines (used by CI `make quickstart`).
- `run_full_pipeline.py` — orchestrates the full reproduction of processed tables.
- `render_figures.py` — regenerates figures for insertion into `scaffold/**/figures`.
