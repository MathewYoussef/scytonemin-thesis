# Scytonemin Thesis Audit Repository

[![CI Status](https://github.com/MathewYoussef/scytonemin-thesis/workflows/CI%20Audit/badge.svg)](https://github.com/MathewYoussef/scytonemin-thesis/actions)

This repository contains the complete audit trail and reproducible research materials for a thesis on **scytonemin biosynthesis and optical remote sensing in cyanobacteria**.

---

## Table of Contents

- [About This Research](#about-this-research)
- [Getting Started](#getting-started)
- [Research Workflow](#research-workflow)
- [Key Components](#key-components)
- [Documentation](#documentation)
- [For Contributors](#for-contributors)
- [Citation](#citation)

---

## About This Research

**Research Question**: Can optical reflectance spectroscopy be used to non-invasively quantify scytonemin concentrations in cyanobacterial cultures?

**Background**: Scytonemin is a UV-protective pigment produced by cyanobacteria in response to radiation stress. Traditional quantification methods (HPLC, chromatography) are destructive and time-consuming. This research explores using optical remote sensing techniques—similar to those used in planetary geology—to measure scytonemin concentration through reflectance spectroscopy.

**Key Innovation**: Development of a machine learning denoising pipeline (Mamba-SSM) to process noisy reflectance spectra, combined with multi-assay validation (chromatography, diode-array detection) to establish robust concentration mapping.

**Practical Impact**: Non-destructive monitoring of UV stress response in cyanobacteria for astrobiology, bioproduction, and environmental sensing applications.

---

## Getting Started

### Quick Start (3 minutes)

```bash
# Clone and set up
git clone https://github.com/MathewYoussef/scytonemin-thesis.git
cd scytonemin-thesis
make setup        # Install dependencies

# Verify everything works
make quickstart   # Run quick tests and sample pipelines
```

### I Want To...

**Understand the research**  
→ Start with [Research Workflow](#research-workflow) below, then explore `notebooks/` starting with `00_env_and_schema.ipynb`

**Reproduce the analysis**  
→ Run `make reproduce` to rebuild all processed data from raw inputs (requires ~30 min)

**View results & figures**  
→ Run `make docs` then open `site/index.html` in your browser

**Run tests**  
→ Run `pytest tests -v` to execute the full test suite (25 tests)

**Contribute code**  
→ Read [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow

---

## Research Workflow

This section describes the chronological flow of the research from hypothesis to validation.

### 1. Hypothesis & Background

**Core Question**: Can we measure scytonemin non-destructively?

**Theory**: Scytonemin absorbs strongly in the UV-visible range (320-480 nm). By measuring reflectance spectra and applying machine learning denoising, we should be able to quantify concentration without destroying the sample.

**Start here**: `notebooks/00_env_and_schema.ipynb` - Documents experimental setup

### 2. Data Collection

**Experiment**: Cyanobacterial cultures exposed to 6 UV dose levels (0 to 3.2 mW/cm²)

**Three measurement methods for cross-validation**:
- **UPLC Chromatography** - Gold standard (destructive)
- **Diode Array Detection** - Alternative chromatography
- **Reflectance Spectroscopy** - Non-destructive (our innovation!)

**Raw data**: See `data/` directory  
**Quality checks**: `notebooks/01_dosimetry_mdv_benchmark.ipynb`

### 3. Analysis Methods

The analysis happens in **5 stages** (A through E):

**Stage A/B**: Chromatography Calibration  
→ Code: `src/chromatography/`  
→ Notebook: `06_uplc_processing_and_calibration.ipynb`  
→ Output: Standard concentrations in mg/gDW

**Stage C**: Reflectance Processing  
→ Code: `src/reflectance/`  
→ Notebook: `03_spectra_ingest_and_rel_reflectance.ipynb`  
→ Output: Clean reflectance spectra

**Stage D**: ML Denoising (Mamba-SSM)  
→ Code: `src/mamba_ssm/`  
→ Notebook: `04_mamba_denoising_QC.ipynb`  
→ Output: Denoised spectra (pre-trained model included)

**Stage E**: Concentration Mapping  
→ Notebooks: `05_continuum_removal_and_occupancy.ipynb`, `08_reflectance_to_concentration_mapping.ipynb`  
→ Output: Spectral features → concentration prediction

### 4. Results & Validation

**Cross-validation**: Compare all three measurement methods  
→ Notebook: `07_concentration_profiles_and_cross_assay.ipynb`

**Geometric effects**: Test viewing angle limitations  
→ Notebook: `09_geometry_and_orientation_effects.ipynb`

**Quality metrics**: R², SNR improvement, residual analysis  
→ See individual notebooks and `docs/claims/`

---

## Key Components

### Where Things Live

**Analysis Code** → `src/`  
Python modules for chromatography, reflectance, and ML denoising

**Interactive Notebooks** → `notebooks/`  
9 Jupyter notebooks documenting each analysis step (numbered 00-09)

**Experimental Data** → `data/`  
Raw instrument outputs (chromatograms, spectra, calibration logs)

**Generated Results** → `data-processed/`  
Outputs from running `make reproduce`

**Tests** → `tests/`  
Unit tests ensuring correctness (run with `pytest`)

**Documentation Site** → `docs/`  
Detailed methodology (build with `make docs`)

**Pre-trained Model** → `models/mamba_ssm/`  
ML denoiser checkpoints (no training needed)

<details>
<summary><b>📁 Complete Directory Reference</b> (click to expand)</summary>

| Directory | Purpose |
|-----------|---------|
| `src/chromatography/` | UPLC processing code |
| `src/reflectance/` | Spectroscopy analysis code |
| `src/mamba_ssm/` | ML denoising model |
| `notebooks/` | Jupyter analysis walkthroughs |
| `data/` | Raw experimental data |
| `data-processed/` | Generated analysis outputs |
| `data-sample/` | Small sample datasets for testing |
| `tests/` | Unit & integration tests |
| `docs/` | MkDocs documentation site |
| `models/` | Pre-trained model weights |
| `scaffold/` | Output folders for figures/tables |
| `env/` | Python dependencies |
| `external/` | Legacy/archived workspaces (read-only) |
| `ops/` | Operations, resource maps, audit logs |
| `claims/` | Individual thesis claim verifications |

**Configuration files**: `Makefile`, `pytest.ini`, `ruff.toml`, `mkdocs.yml`, `CITATION.cff`

</details>

---

## Documentation

### Sequential Notebook Walkthrough

Follow these notebooks in order to understand the complete analysis:

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 00 | `env_and_schema.ipynb` | Package versions, data schemas, setup |
| 01 | `dosimetry_mdv_benchmark.ipynb` | UV dose calculations |
| 03 | `spectra_ingest_and_rel_reflectance.ipynb` | Raw → clean spectra |
| 04 | `mamba_denoising_QC.ipynb` | ML denoising quality control |
| 05 | `continuum_removal_and_occupancy.ipynb` | Feature extraction |
| 06 | `uplc_processing_and_calibration.ipynb` | Chromatography calibration |
| 07 | `concentration_profiles_and_cross_assay.ipynb` | Cross-method validation |
| 08 | `reflectance_to_concentration_mapping.ipynb` | **Main result**: spectra → concentration |
| 09 | `geometry_and_orientation_effects.ipynb` | Viewing angle effects |

### Documentation Website

Build and view the full documentation site:

```bash
make docs           # Build with MkDocs
mkdocs serve        # View at http://localhost:8000
```

The site includes:
- Detailed methodology for each analysis stage
- Individual thesis claim verifications
- Data provenance documentation

---

## For Contributors

### How to Contribute

1. **Read the guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Follow the code of conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
3. **Use issue templates** for bug reports or feature requests
4. **Submit pull requests** following the PR template

### Development Workflow

```bash
# Set up development environment
make setup

# Run tests before making changes
pytest tests -v

# Make your changes, then test again
pytest tests -v

# Check code quality
ruff check src/

# Build docs to verify
make docs
```

### Running Tests

```bash
pytest tests -v              # All tests
pytest tests -m quick        # Quick tests only (for CI)
pytest --cov=src             # With coverage report
```

---

## Technical Details

<details>
<summary><b>🔧 CI/CD Pipeline</b> (click to expand)</summary>

The GitHub Actions workflow (`.github/workflows/ci-audit.yml`) runs on every push:

1. Python 3.11 environment setup
2. Dependency installation
3. Code linting with ruff
4. Quick tests (`pytest -m quick`)
5. Test coverage reporting
6. Documentation build

</details>

<details>
<summary><b>📦 Large Files</b> (click to expand)</summary>

Three files exceed 50 MB (all below 100 MB GitHub limit):

- `data/reference/mamba_ssm/denoised_full_run.csv` (~59 MB)
- `models/mamba_ssm/checkpoints/god_run/mamba_tiny_uv_best.pt` (~62 MB)
- `models/mamba_ssm/prod/Track_H_fold_02/mamba_tiny_uv_best.pt` (~62 MB)

</details>

<details>
<summary><b>🔐 Data Provenance</b> (click to expand)</summary>

- **Checksums**: SHA-256 manifests for all raw data
- **Version Control**: Git tracks all code and configuration
- **Resource Maps**: `ops/output/` documents data lineage
- **Automated Tests**: Validate data integrity and analysis correctness

</details>

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

- **Bug reports**: Use GitHub Issues with the bug report template
- **Feature requests**: Use GitHub Issues with the feature request template
- **Questions**: Start a GitHub Discussion
- **Private inquiries**: Contact the maintainer

---

**License**: See [LICENSE](LICENSE) file  
**Last Updated**: 2026-01-24
