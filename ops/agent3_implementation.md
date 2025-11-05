# Agent 3 — Implementation Plan (Step‑by‑Step)

> Branch: `agent3/data-harvest-setup`  
> Write scope: `data/**`, `models/**`, `ops/output/data/**`, and `scaffold/**/figures` only.

## 0) Bootstrap DVC/LFS
```bash
git checkout -b agent3/data-harvest-setup
dvc init
# Document remote in ops/output/data/remote.md (Zenodo/OSF/S3). Do not commit credentials.
```

## 1) Catalog sources
- Parse `ops/output/inventory/inventory.csv` and filter for the four blocks:
  - `Reflectance/**`
  - `Initial_Calibration/**`
  - `Act_of_God_Mamba_Results/**`
  - `Supplements/**`
- For each file, compute `sha256`, size, and license (or `UNKNOWN`). Write `ops/output/data/catalog.csv`.

## 2) Data dictionary & provenance
- For each dataset/model:
  - `data/<dataset_id>/README.md` (description, units, columns)
  - `data/<dataset_id>/schema.json`
  - `data/<dataset_id>/provenance.yaml` (origin path, commit, transforms)

## 3) Samples & processed outputs
- Create `data-sample/<dataset_id>/` subsets that can reproduce flagship figures within minutes.
- Export processed tables to `data-processed/<block>/...` (CSV/Parquet) with checksums.
- Render required figures/tables to `scaffold/<block>/figures` following Agent 2 naming conventions.

## 4) Health checks
- `ops/output/data/checks.md` summarizing:
  - row/column counts
  - schema matches
  - checksum verification
  - simple stat checks (e.g., correlation r within expected range)

## 5) PR
- Ensure no media dumps in repo; raw heavy data lives in DVC/LFS only.
- Open PR with catalog, dictionaries, samples, processed tables, and figures.
