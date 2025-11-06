# Scytonemin Thesis Audit Repository

This repository packages **all supplementary material** needed to review the
Scytonemin thesis: raw instrument exports, processed tables, documentation,
automation scripts, and reviewer notebooks. Everything required to rebuild the
audit hub locally is tracked here so reviewers can reproduce the artefacts and
browse the MkDocs site without chasing external resources.

---

## Repository Layout

| Path | What lives here |
| --- | --- |
| `data/raw/**` | Canonical raw inputs (chromatograms, diode-array spectra, denoised reflectance, UV calibration logs). Each folder has a README plus `CHECKSUMS.sha256`. |
| `data/reference/**` | Regenerated canonical tables/configs ready for notebooks (calibration JSON, reflectance canonical dataset, validation manifests, etc.). |
| `data-processed/**` | Outputs produced by `make reproduce` (dose summaries, validation-panel counts, UV schedules). |
| `models/mamba_ssm/` | Shipped Mamba checkpoints plus checksums. Training is **not** rerun; we provide the production weights. |
| `ops/` | Resource maps, auditor workflows, and provenance logs lifted from the legacy workspaces. |
| `notebooks/` | Reviewer walkthroughs (00–09) and CLI helpers (`run_full_pipeline.py`, `run_samples.py`, `render_figures.py`). |
| `scaffold/**` | Destination folders for figures/tables embedded in MkDocs pages; each block owns a `figures/`, `tables/`, `notebooks/`, and `assets/` subfolder. |
| `docs/` | MkDocs site powering the public audit hub. |
| `external/workspaces/**` | Immutable snapshots of the legacy environments Agent 1 catalogued. Treat these as read-only references. |


## Quick Start

```bash
make setup        # create .venv and install lightweight dependencies
make reproduce    # rebuild calibration + reflectance outputs (Stage A/B + canonical builder)
make docs         # build the MkDocs site into ./site/
make quickstart   # (optional) run fast pytest markers + sample pipelines
```

- `make reproduce` re-executes Stage A/B chromatogram calibrations and the
  reflectance canonical builder. The Act-of-God denoiser is **not** rerun—we ship
  the denoised spectra and evaluation artefacts under `data/reference/mamba_ssm/`.
- `make docs` now includes `mkdocs-material` in the shared requirements so CI and
  local builds use the same theme stack.


## Optional Heavy Dependencies

CI and the notebooks run with CPU-only packages. If you want to experiment with
the denoiser yourself, install the GPU stack manually:

```bash
pip install torch==2.3.0 torchvision==0.18.0 mamba-ssm[causal-conv1d]==2.2.5 tensorboard
```

These packages are intentionally absent from `env/requirements.txt` so GitHub
runners are not forced to build CUDA wheels on every audit run.


## Notebooks & Automation

| Notebook | Purpose |
| --- | --- |
| `00_env_and_schema.ipynb` | Capture package versions, dataset schemas, and git hashes before analysis. |
| `01_dosimetry_mdv_benchmark.ipynb` | Recompute UVA/UVB dose schedules and verify %MDV mappings. |
| `04_mamba_denoising_QC.ipynb` | Inspect the shipped denoised spectra and reproduce readiness-gate pass/fail tallies. |
| `05_continuum_removal_and_occupancy.ipynb` | Demonstrate continuum removal, quadratic bowl fitting, and occupancy calculations. |
| `06_uplc_processing_and_calibration.ipynb` | Replay Stage A/B fits and derive dry-weight concentrations. |
| `07_concentration_profiles_and_cross_assay.ipynb` | Summarise dose-response behaviour and run Deming regression between assays. |
| `08_reflectance_to_concentration_mapping.ipynb` | Map Σ occupancy to Chrom_total (mg·gDW⁻¹) and expose a prediction helper. |
| `09_geometry_and_orientation_effects.ipynb` | Compare fits by viewing angle and discuss BRDF limitations. |

Automation scripts under `notebooks/` dispatch into the modules in `src/` and
mirror what CI executes.


## Data & Provenance

- **Checksums** – Every folder under `data/raw/**`, `data/reference/**`, and
  `models/**` ships with a checksum manifest. Regenerate them whenever you update
  artefacts via the pipelines.
- **Resource maps** – See `ops/output/data/uplc_resource_map.md` and related notes
  for how the legacy workspaces map into this repository.
- **Snapshots** – Everything under `external/workspaces/**` is archival and kept
  out of the pipelines. Do not commit changes there.


## Continuous Integration

`.github/workflows/ci-audit.yml` runs on every push/PR to `main` and verifies:

1. Python 3.11 environment + dependency installation (CPU-only stack)
2. Quick pytest markers (`-m quick`)
3. `mkdocs build`

Use GitHub Actions’ “Re-run jobs” button if you update pipelines or dependencies.


## Large Files

GitHub will warn about three large artefacts that ship with the repo:

- `data/reference/mamba_ssm/denoised_full_run.csv` (~59 MB)
- `models/mamba_ssm/checkpoints/god_run/mamba_tiny_uv_best.pt` (~62 MB)
- `models/mamba_ssm/prod/Track_H_fold_02/mamba_tiny_uv_best.pt` (~62 MB)

They remain below the 100 MB hard limit, so pushes succeed. Use Git LFS if you
prefer to offload them.


## Questions

Open an issue or start a discussion if you need additional guidance. Happy auditing!
