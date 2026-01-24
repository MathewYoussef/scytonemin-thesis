# Scytonemin Thesis Audit Repository

[![CI Status](https://github.com/MathewYoussef/scytonemin-thesis/workflows/CI%20Audit/badge.svg)](https://github.com/MathewYoussef/scytonemin-thesis/actions)

This repository contains the complete audit trail and reproducible research materials for a thesis on **scytonemin biosynthesis and optical remote sensing in cyanobacteria**.

---

## Table of Contents

- [About This Research](#about-this-research)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Research Workflow](#research-workflow)
  - [1. Hypothesis & Background](#1-hypothesis--background)
  - [2. Data Collection](#2-data-collection)
  - [3. Analysis Methods](#3-analysis-methods)
  - [4. Results & Validation](#4-results--validation)
- [Documentation](#documentation)
- [For Contributors](#for-contributors)
- [Technical Details](#technical-details)
- [Citation](#citation)

---

## About This Research

**Research Question**: Can optical reflectance spectroscopy be used to non-invasively quantify scytonemin concentrations in cyanobacterial cultures?

**Background**: Scytonemin is a UV-protective pigment produced by cyanobacteria in response to radiation stress. Traditional quantification methods (HPLC, chromatography) are destructive and time-consuming. This research explores using optical remote sensing techniques—similar to those used in planetary geology—to measure scytonemin concentration through reflectance spectroscopy.

**Key Innovation**: Development of a machine learning denoising pipeline (Mamba-SSM) to process noisy reflectance spectra, combined with multi-assay validation (chromatography, diode-array detection) to establish robust concentration mapping.

**Practical Impact**: Non-destructive monitoring of UV stress response in cyanobacteria for astrobiology, bioproduction, and environmental sensing applications.

---

## Repository Structure

This repository is organized to support **complete reproducibility** of the research. All directories serve a specific purpose in the analysis pipeline:

### Core Analysis Directories

| Directory | Purpose | Key Contents |
|-----------|---------|--------------|
| **`src/`** | Analysis code | `chromatography/` - UPLC processing<br>`reflectance/` - Spectroscopy analysis<br>`mamba_ssm/` - ML denoising model |
| **`notebooks/`** | Interactive analysis | Jupyter notebooks 00-09 (see [Research Workflow](#research-workflow))<br>Helper scripts for automation |
| **`data/`** | Raw experimental data | Chromatogram exports, reflectance spectra, calibration logs |
| **`data-processed/`** | Computed results | Outputs from analysis pipelines (`make reproduce`) |
| **`tests/`** | Quality assurance | Unit tests for core modules, integration tests |

### Documentation & Support

| Directory | Purpose | Key Contents |
|-----------|---------|--------------|
| **`docs/`** | Research documentation | MkDocs site with methodology, claims verification |
| **`models/`** | Pre-trained models | Mamba-SSM denoiser checkpoints (62 MB each) |
| **`scaffold/`** | Output organization | Structured folders for generated figures and tables |
| **`env/`** | Dependencies | Python requirements, setup configurations |

### Reference & Operations

| Directory | Purpose | Notes |
|-----------|---------|-------|
| **`external/`** | Legacy workspaces | Archived historical data (read-only) |
| **`ops/`** | Operations & provenance | Resource maps, audit logs, workflow documentation |
| **`claims/`** | Thesis claims | Individual claim verification files |

### Configuration Files

- **`Makefile`** - Automation: `make setup`, `make reproduce`, `make docs`, `make test`
- **`mkdocs.yml`** - Documentation site structure
- **`pytest.ini`** - Test configuration with custom markers
- **`ruff.toml`** - Code linting and quality rules
- **`CITATION.cff`** - Citation metadata for this repository

---

## Getting Started

### Quick Start Commands

```bash
# 1. Set up the environment
make setup        # Creates virtual environment and installs dependencies

# 2. Run quick validation
make quickstart   # Runs fast tests and sample pipelines

# 3. Reproduce full analysis
make reproduce    # Rebuilds all processed data from raw inputs

# 4. Build documentation
make docs         # Generates MkDocs site in ./site/

# 5. Run all tests
pytest tests -v   # Execute full test suite
```

### Installation

**Prerequisites**: Python 3.11+, Git

**Basic Setup** (for reviewing and testing):
```bash
git clone https://github.com/MathewYoussef/scytonemin-thesis.git
cd scytonemin-thesis
make setup
```

**Full Setup** (for running ML denoising - optional):
```bash
# After basic setup, install GPU dependencies
pip install torch==2.3.0 torchvision==0.18.0 mamba-ssm[causal-conv1d]==2.2.5 tensorboard
```

**Note**: GPU dependencies are intentionally excluded from default requirements to avoid long CI build times.

---

## Research Workflow

This section describes the chronological flow of the research from hypothesis to validation.

### 1. Hypothesis & Background

**Core Hypothesis**: Reflectance spectroscopy can quantify scytonemin non-destructively with accuracy comparable to established chromatographic methods.

**Theoretical Foundation**:
- Scytonemin exhibits characteristic absorption in UV-visible range (320-480 nm)
- Optical remote sensing (ORS) successfully used in planetary science for mineral detection
- Machine learning can denoise spectra while preserving biologically relevant features

**Documented in**:
- `docs/index.md` - Research overview
- `docs/claims/` - Individual claim verification
- `notebooks/00_env_and_schema.ipynb` - Experimental setup documentation

### 2. Data Collection

**Experimental Design**:
- Cyanobacterial cultures exposed to 6 UV dose levels (0 to ~3.2 mW/cm² UVA)
- Multiple measurement modalities for cross-validation:
  - **UPLC Chromatography** - Gold standard quantification
  - **Diode Array Detection (DAD)** - Alternative chromatographic method
  - **Reflectance Spectroscopy** - Non-destructive optical measurement

**Raw Data Location**:
- `data/` - Raw instrument exports
  - Chromatogram files
  - Reflectance spectra (320-700 nm)
  - UV dosimetry logs
- `data-sample/` - Example datasets for quick testing

**Quality Control**:
- Checksums for all raw data (`CHECKSUMS.sha256` in each folder)
- Dark/white reference calibrations
- Replicate measurements for precision estimation

**Documented in**:
- `notebooks/01_dosimetry_mdv_benchmark.ipynb` - UV dose verification
- `notebooks/03_spectra_ingest_and_rel_reflectance.ipynb` - Data ingestion

### 3. Analysis Methods

The analysis pipeline consists of multiple stages, each with dedicated code and notebooks:

#### Stage A/B: Chromatography Calibration
- **Purpose**: Establish concentration standards from chromatography
- **Code**: `src/chromatography/`
- **Notebooks**: `06_uplc_processing_and_calibration.ipynb`
- **Methods**: Weighted linear regression (1/x weighting), trimmed mean aggregation
- **Output**: Calibrated concentrations in mg/gDW (milligrams per gram dry weight)

#### Stage C: Reflectance Processing
- **Purpose**: Convert raw spectra to relative reflectance
- **Code**: `src/reflectance/`
- **Notebooks**: `03_spectra_ingest_and_rel_reflectance.ipynb`
- **Methods**: Dark/white normalization, continuum removal, occupancy calculation

#### Stage D: ML Denoising (Mamba-SSM)
- **Purpose**: Denoise reflectance spectra while preserving UV absorption features
- **Code**: `src/mamba_ssm/`
- **Notebooks**: `04_mamba_denoising_QC.ipynb`
- **Methods**: State-space model (Mamba) trained on validation panel
- **Model**: Pre-trained checkpoints in `models/mamba_ssm/`

#### Stage E: Occupancy & Concentration Mapping
- **Purpose**: Map spectral features to scytonemin concentration
- **Code**: Combined in reflectance and analysis modules
- **Notebooks**: 
  - `05_continuum_removal_and_occupancy.ipynb` - Feature extraction
  - `08_reflectance_to_concentration_mapping.ipynb` - Linear mapping
- **Methods**: Integration over 320-480 nm and 360-410 nm windows

**Documented in**: Each notebook contains detailed methodology with inline citations.

### 4. Results & Validation

#### Cross-Assay Validation
- **Comparison**: Chromatography vs DAD vs Reflectance
- **Statistical Methods**: Deming regression, trimmed means, confidence intervals
- **Notebook**: `07_concentration_profiles_and_cross_assay.ipynb`

#### Geometric Effects
- **Analysis**: BRDF (bidirectional reflectance) limitations across viewing angles
- **Notebook**: `09_geometry_and_orientation_effects.ipynb`

#### Quality Metrics
- **SNR improvement**: Quantified in denoising QC
- **Prediction accuracy**: R² and residual analysis
- **Reproducibility**: All analyses automated via `make reproduce`

#### Claims Verification
- Each thesis claim mapped to:
  - Specific notebook cell
  - Generated figure/table
  - Automated test (where applicable)
- See `docs/claims/` for claim-by-claim evidence

---

## Documentation

### Interactive Notebooks

Sequential walkthrough of the analysis (start here for understanding the research):

| Notebook | Topic | Purpose |
|----------|-------|---------|
| `00_env_and_schema.ipynb` | Environment setup | Package versions, data schemas, git hashes |
| `01_dosimetry_mdv_benchmark.ipynb` | UV dosimetry | Dose calculations and %MDV verification |
| `03_spectra_ingest_and_rel_reflectance.ipynb` | Data ingestion | Raw to relative reflectance conversion |
| `04_mamba_denoising_QC.ipynb` | ML denoising | Quality control for Mamba-SSM outputs |
| `05_continuum_removal_and_occupancy.ipynb` | Feature extraction | Continuum removal, occupancy windows |
| `06_uplc_processing_and_calibration.ipynb` | Chromatography | UPLC calibration and concentration derivation |
| `07_concentration_profiles_and_cross_assay.ipynb` | Cross-validation | Dose-response curves, Deming regression |
| `08_reflectance_to_concentration_mapping.ipynb` | Main result | Spectral occupancy → concentration mapping |
| `09_geometry_and_orientation_effects.ipynb` | BRDF analysis | Viewing angle effects and limitations |

### MkDocs Site

Comprehensive documentation site (build with `make docs`):

- **Home** (`docs/index.md`) - Audit hub overview
- **Claims** (`docs/claims/`) - Thesis claim verification
- **Methods** - Detailed methodology for each analysis block:
  - Initial Calibration
  - Reflectance Processing
  - Mamba-SSM Denoising
  - Cross-Assay Validation

View locally: `mkdocs serve` then open `http://localhost:8000`

---

## For Contributors

We welcome contributions! This repository supports:

- **Reviewers**: Audit reproducibility, verify claims
- **Researchers**: Extend methods, apply to new datasets
- **Developers**: Improve code quality, add tests

### Contribution Guidelines

Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Development workflow
- Testing requirements
- Code style guidelines
- Pull request process

### Code of Conduct

All contributors must follow our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

### Issue Templates

Use GitHub issue templates for:
- **Bug reports** - Report issues with reproducibility
- **Feature requests** - Suggest enhancements
- **Questions** - Ask via GitHub Discussions

---

## Technical Details

### Continuous Integration

`.github/workflows/ci-audit.yml` runs on every push/PR:

1. Python 3.11 environment setup
2. Dependency installation (CPU-only stack)
3. Linting with ruff
4. Quick tests (`pytest -m quick`)
5. Test coverage reporting
6. Documentation build (`mkdocs build`)

### Data Provenance

- **Checksums**: SHA-256 manifests for all raw data
- **Version Control**: Git tracks all code and configuration
- **Resource Maps**: `ops/output/` documents data lineage
- **Automated Tests**: Validate data integrity and analysis correctness

### Large Files

Three files exceed 50 MB (all below 100 MB GitHub limit):

- `data/reference/mamba_ssm/denoised_full_run.csv` (~59 MB)
- `models/mamba_ssm/checkpoints/god_run/mamba_tiny_uv_best.pt` (~62 MB)
- `models/mamba_ssm/prod/Track_H_fold_02/mamba_tiny_uv_best.pt` (~62 MB)

Optional: Use Git LFS for these files if preferred.

### Testing

```bash
# Run all tests
pytest tests -v

# Run only quick tests (used in CI)
pytest tests -m quick

# Run with coverage
pytest tests --cov=src --cov-report=html
```

---

## Citation

If you use this repository or build upon this research, please cite:

```bibtex
@software{scytonemin_thesis,
  author = {Youssef, Mathew},
  title = {Scytonemin Thesis Audit Hub},
  year = {2025},
  url = {https://github.com/MathewYoussef/scytonemin-thesis}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## Questions & Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions via GitHub Discussions
- **Email**: For private inquiries, contact the maintainer

---

**License**: See [LICENSE](LICENSE) file for details.

**Last Updated**: 2026-01-24
