# Reflectance Scaffold — Purpose & Write API

**Purpose:** Stage denoised spectra visuals, summary tables, and notebooks for the reflectance analysis prior to publication in the docs site.

## Writers
- **Agent 3:** Drops generated plots, processed tables, and executed notebooks derived from `Reflectance/`.
- **Agent 4:** Contributes curated media that pair with the reflectance story before moving assets into `docs/media/`.

## Expected Outputs
- `figures/` — Spectra panels, SNR heatmaps, QC dashboards.
- `tables/` — CSV/Parquet exports (ΔSNR summaries, treatment stats, manifest excerpts).
- `notebooks/` — Clean notebooks that reproduce figures/tables; ensure deterministic writes.
- `assets/images/`, `assets/videos/` — Raw media staging before derivative creation.

## Conventions
- Prefix assets with the thesis figure ID when known (e.g., `fig04a_snr-gains.png`).
- Store tables alongside a `*_schema.json` and checksum entry maintained in `ops/output/data`.
- Avoid large raw arrays here; use DVC/LFS pointers under `data/` and materialise lightweight derivatives only.

