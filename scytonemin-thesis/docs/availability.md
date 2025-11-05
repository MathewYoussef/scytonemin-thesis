# Code & Data Availability

This audit hub packages every asset required to reproduce the calibration, reflectance, and Act‑of‑God validation workflows. The `scytonemin-thesis/` repository is self-contained: raw drops, canonical tables, checkpoints, and derived figures are tracked inside the tree with provenance README files and checksum manifests.

## Code Layout

- Chromatogram/DAD pipeline: `src/chromatography/` (invoked via `python -m chromatography.run_stage_a` / `run_stage_bc`).
- Reflectance harmonisation: `src/reflectance/` (canonical builder + aggregation utilities).
- Mamba evaluation tooling: `src/mamba_ssm/` (inference/eval scripts; warm-start training is not re-run by default).
- Shared helpers: `src/imas_pipeline/` hosts utilities referenced by multiple blocks.

Use the Makefile targets for orchestration:

```bash
make setup        # create .venv and install dependencies (Torch optional)
make reproduce    # rerun Stage A/B + reflectance canonical build; skips Mamba eval if Torch/mamba-ssm absent
make quickstart   # sample smoke checks (to be filled by Agents 3/4)
make docs         # build MkDocs site
```

## Data & Models

- Raw instrument exports and staging drops live under `data/raw/<block>/` with README + `CHECKSUMS.sha256` manifests.
- Canonical tables/configs sit in `data/reference/<block>/` (calibration CSV/JSON, reflectance crosswalks, Mamba manifests).
- Act-of-God checkpoints reside in `models/mamba_ssm/` and evaluation artefacts in `ops/output/data/mamba_checks/`.
- All checksum manifests regenerate deterministically after a successful `make reproduce` run.

We intentionally **do not** re-run the AoG denoising by default. Auditors have the option to install `torch`, `torchvision`, and `mamba-ssm[causal-conv1d]` locally, then rerun `python -m mamba_ssm.scripts.evaluate_validation_panel` to regenerate panel metrics. The packaged CSV/JSON summaries already capture the canonical results used in the thesis.

## Figures, Tables, Notebooks

- Regenerated tables/plots are staged under `scaffold/<block>/{figures,tables,notebooks}/` according to each block’s README write API.
- Reflectance + Initial Calibration figures are refreshed automatically by `make reproduce`; Mamba figures ship from the recorded AoG run.

## Media

- Media derivatives (photos, videos) belong in `docs/media/**`; high-resolution originals are archived separately and referenced via the media README files when published.

## Releases and DOIs

- Prior to submission, archive the repository (including raw data, reference tables, models, and docs site) to Zenodo and cite the DOI in the thesis/manuscript. Mention that AoG training is fixed to the provided checkpoints while evaluation is reproducible via the recorded manifests.
