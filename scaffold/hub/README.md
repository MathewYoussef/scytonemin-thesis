# Hub Scaffold — Purpose & Write API

**Purpose:** Central landing artefacts shared across blocks (claims ledger, cross-block tables, hub notebooks).

## Writers
- **Agent 3:** Generates hub-level tables (e.g., `claims.csv`, inventory rollups) and lightweight sample notebooks for quickstart flows.
- **Agent 4:** Provides thumbnails/posters for the landing page when cross-block media is required.

## Expected Outputs
- `figures/` — Overview diagrams or navigation charts referenced from `docs/index.md`.
- `tables/` — Aggregated CSVs/Parquet files (claims ledger, audit matrices).
- `notebooks/` — Hub notebooks powering quickstart demos or summary stats.
- `assets/images/` & `assets/videos/` — Staging area for media before promotion to `docs/media/`.

## Conventions
- Use lowercase kebab-case filenames (e.g., `claims-coverage.png`, `audit-readiness.csv`).
- Every table must ship with a schema definition and checksum recorded in `ops/output/data`.
- Notebook outputs should overwrite existing files deterministically to keep diffs clean.

