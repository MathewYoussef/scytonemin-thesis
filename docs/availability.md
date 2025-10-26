# Code & Data Availability

This audit hub consolidates code, data pointers, and media locations so reviewers can reproduce figures and verify claims without hunting across folders.

## Code

- Reflectance analysis: `Reflectance/`
- Calibration pipelines (Chromatogram → DAD → Concentration → Dry Weight): `Initial_Calibration/`
- Mamba‑SSM experiment: `Act_of_God_Mamba_Results/`
- Shared helpers (to be centralized): `IMAS-portfolio/imas_pipeline/`

## Data

- Full datasets and models: to be managed by Agent 3 via DVC/LFS; documented under `ops/output/data/catalog.csv` with checksums and provenance.
- Sample (small) datasets: to be provided under `data-sample/**` for quick checks (`make quickstart`).

## Figures, Tables, Notebooks

- Generated figures/tables/notebooks are staged under `scaffold/<block>/**` following the write APIs described in each block README.

## Media

- Photos and videos: curated by Agent 4 as web-weight derivatives under `docs/media/**` with captions and alt text; high‑res originals referenced via DVC/LFS.

## Releases and DOIs

- Upon integration, archive the audit hub (docs + scaffolds + DVC/LFS pointers) to Zenodo and cite the release DOI in the manuscript’s Code & Data Availability statement.

