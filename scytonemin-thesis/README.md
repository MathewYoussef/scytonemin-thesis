# Scytonemin Thesis Supplementary Repository

All supplementary data, code, and figures needed to audit the Scytonemin thesis live here. Reviewers can start from raw instrument exports and reproduce every processed table and figure using the pipelines in this repo.

## What’s Inside

- **`data/raw/**`** – canonical inputs (chromatograms, reflectance spectra, UV calibration tables, denoised Mamba spectra) each with a README and checksum manifest. The original working directories are archived under `external/workspaces/` for reference only.
- **`src/**`** – refactored Python modules and CLI scripts: chromatography stage A/B, reflectance canonical builders, and Mamba SSM evaluation tools.
- **`notebooks/`** – reviewer walkthroughs (`00_env_and_schema.ipynb`, `01_dosimetry_mdv_benchmark.ipynb`, `03_spectra_ingest_and_rel_reflectance.ipynb`, `initial_calibration_overview.ipynb`, `reflectance_overview.ipynb`, `mamba_validation_overview.ipynb`) plus CLI runners that push outputs into `scaffold/**`.
- **`data-sample/**`** – small CSV/NPY slices safe for CI and quick review; the Makefile regenerates them from raw data when needed.
- **`data-processed/**`** – regenerated tables produced by `make reproduce` (reflectance dose summaries, DAD dose response, validation panel counts, UV dose schedule).
- **`scaffold/**`** – destination folders for figures/tables embedded in MkDocs pages; each block has `figures/`, `tables/`, `notebooks/`, and `assets/` subfolders.
- **`models/mamba_ssm/`** – shipped Mamba checkpoints plus checksums; training is not rerun, but evaluation scripts reproduce panel metrics.
- **`docs/`** – MkDocs site backing the public audit hub: block overviews, methods, media stubs, and claim pages.
- **`manifest/claims.yml`** + **`ops/output/data/`** – provenance catalogues, checksum reports, and agent handoffs.

## Getting Started

```bash
make setup        # create local virtualenv / install deps
make quickstart   # run pytest quick markers + sample pipelines
make reproduce    # rebuild all processed tables and scaffold figures
make docs         # build the MkDocs documentation locally
```

- See `docs/index.md` for the “Audit in 10 Minutes” walkthrough and per-block expectations.
- Consult `ops/output/data/uplc_resource_map.md` if you need to reference the archived workspaces under `external/workspaces/`.
- Heavy Mamba assets can be evaluated via `python src/mamba_ssm/scripts/evaluate_validation_panel.py` using the shipped checkpoints.

## Repository Etiquette

- Do not modify files in `external/workspaces/**`; treat them as historical snapshots.
- Place new raw inputs under `data/raw/**` with a README + checksum update, and document provenance in the matching `data/*/provenance.yaml`.
- Keep processed outputs reproducible via `make reproduce`; never check in ad-hoc artefacts outside the sanctioned directories.
