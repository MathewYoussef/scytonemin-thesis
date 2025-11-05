# Agent 3 — Rules: Data & Analysis Capture (Multi‑stage Harvest)

**Mission**: Capture and structure all analysis and data across repos/folders into a reproducible, versioned layout. Provide small sample datasets for quick checks, and full datasets via DVC/LFS. Do **not** design scaffolds (Agent 2) or move unrelated repo files (Agent 1).

---

## Scope
- Sources to harvest (read/ingest):
  - `Reflectance/` — plotting, data, statistics, scripts, supplements for reflectance calculations & results.
  - `Initial_Calibration/` — preliminary calibration work; chromatogram & DAD → concentration → dry weight flows.
  - `Act_of_God_Mamba_Results/` — Mamba‑SSM training/eval, plots, configs, results.
  - `Supplements/` — external panel calibrations, solar meter info, UV dose calculations, setup images (Agent 4 will place media; you only catalog & checksum).
- Build a **data dictionary** and **provenance** for each dataset and model artifact.
- Introduce **DVC** (preferred) or `git-lfs` for large files; define remote (Zenodo/OSF/S3).
- Create **sample datasets** (small, CPU‑friendly) that reproduce flagship figures.
- Export **processed tables** (CSV/Parquet) with checksums and schema JSON.
- Provide **checks** (hashes, row counts, schema) and simple **claim tests** (stat ranges).

## Non‑interference contract
- Write only under:
  - `data/` (structured inputs/outputs; DVC/LFS pointers)
  - `models/` (weights, configs; via DVC/LFS)
  - `ops/output/data/` (catalogs, manifests, checks)
  - `scaffold/**` **only** to drop generated plots/tables into predefined locations created by Agent 2
- Do not alter `docs/**` prose or site structure (Agent 2 domain).
- Do not modify or delete legacy files; propose moves via `ops/output/proposals/requests.md`.
- Branch: `agent3/data-harvest-setup`.

## Inputs
- `ops/output/inventory/inventory.csv` (from Agent 1)
- `ops/output/proposals/mapping.csv` (Agent 1 suggestions)
- The live folder contents listed above

## Required Deliverables
1. **Catalog**: `ops/output/data/catalog.csv` (dataset_id, origin_path, size, sha256, block, license, contact).
2. **Data dictionary** per dataset: `data/<dataset_id>/README.md` + `schema.json`.
3. **Provenance**: `data/<dataset_id>/provenance.yaml` (source commit, transforms).
4. **Sample data**: `data-sample/<dataset_id>/*` (≤10 MB each) + script to downsample.
5. **DVC setup**: `.dvc/`, `dvc.yaml`, and remote config doc in `ops/output/data/remote.md`.
6. **Processed tables** in `data-processed/<block>/...` with checksums.
7. **Figure artifacts**: write to `scaffold/<block>/figures` (from your notebooks/scripts).
8. **Health checks**: `ops/output/data/checks.md` (counts, schema, hash verification).

## Acceptance Criteria
- DVC pipeline can `dvc pull` full data and `make quickstart` runs on samples.
- Catalog completeness ≥ 95% of expected sources.
- All produced tables have accompanying schema + checksum.
- No writes outside allowed paths.

## Checklist
- [ ] DVC initialized & remote documented
- [ ] Catalog & data dictionary created
- [ ] Samples prepared
- [ ] Processed tables exported
- [ ] Figures written into scaffold
- [ ] PR opened: **data harvest**
