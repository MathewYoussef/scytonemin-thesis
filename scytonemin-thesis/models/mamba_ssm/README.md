# Mamba-SSM — Checkpoints

Trained checkpoints copied from `Act_of_God_Mamba_Results/checkpoints`. These weights are required to reproduce the validation panel metrics without retraining.

## Layout
- `prod/Track_H_fold_02/mamba_tiny_uv_best.pt` — warm-start fold-02 model promoted to production.
- `checkpoints/god_run/` — Act-of-God fine-tuning outputs (include failed attempts under `god_run_failed_*`).

Reference the manifest and metadata tables under `data/reference/mamba_ssm/` when loading these models.
