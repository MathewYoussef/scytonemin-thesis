# Mamba-SSM — Raw Spectra & Staging Assets

Training and evaluation inputs copied from `Act_of_God_Mamba_Results/data`. These directories must remain unchanged so auditors can reproduce validation metrics with the published checkpoints.

## Contents
- `spectra_for_fold/` — per-fold denoised spectra used during cross-validation and warm-start training.
- `_staging/` — intermediate drops (preprocessed spectra, metadata joins) consumed by CLI utilities.
- `denoised_full_run_staging/` — all-spectra denoised outputs ready for downstream analytics.

Pipelines in `src/mamba_ssm/` expect these paths to exist relative to the project root.
