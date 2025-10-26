# Supplements Scaffold — Purpose & Write API

**Purpose:** Stage condensed derivatives of supplemental materials (dose calculations, instrument certificates, galleries) before they are linked from the docs site.

## Writers
- **Agent 3:** Converts raw supplemental assets into audit-friendly tables, plots, or notebooks.
- **Agent 4:** Curates photo/video derivatives and ensures captions/alt text align with claim coverage.

## Expected Outputs
- `figures/` — Charts distilled from supplemental spreadsheets (e.g., UV schedule summaries).
- `tables/` — CSV/Parquet exports of dose schedules, instrument metadata, or calibration logs.
- `notebooks/` — Scripts demonstrating how supplements feed the core analysis.
- `assets/images/`, `assets/videos/` — Staging area for media prior to placement in `docs/media/`.

## Conventions
- Use filenames that indicate the source document (e.g., `uv-dose_schedule_v1.png`).
- Capture provenance (source path, timestamp) in README snippets or accompanying metadata files.
- When large files are necessary, store them via DVC/LFS outside this scaffold and link to the pointer.

