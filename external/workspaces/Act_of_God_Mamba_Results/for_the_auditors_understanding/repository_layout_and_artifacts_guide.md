# Act_of_God Repository Overview

This guide orients you (or anyone inheriting the work) to the contents of
`Act_of_God_Mamba_Results`. It explains what lives where, why those artifacts
exist, and how to reproduce the core workflows.

> Last updated: 2025-10-10  
> Primary run: Act_of_God warm-start from fold-02 checkpoint (`mamba_tiny_uv_best.pt`)

## Auditors bundle quick links

| File | What you'll find |
| ---- | ---------------- |
| `act_of_god_warm_start_plan.md` | Warm-start plan, readiness gates, and training/evaluation commands. |
| `validation_panel_audit_results.md` | Latest fixed-panel gate outcomes (pass/fail reasoning). |
| `validation_panel_split_summary.md` | Composition of the 750-spectrum validation panel. |
| `training_manifest_coverage_summary.md` | Training manifest coverage by treatment/sample/angle. |
| `training_group_counts_reference.md` | Replicate counts for every training group_id. |
| `fold02_training_coverage_snapshot.md` | Baseline fold-02 coverage used for warm-start context. |
| `dose_metadata_documentation.md` | Description of FiLM conditioning features and dose stats. |

---

## Top-level layout

| Path / File | Purpose |
| ----------- | ------- |
| `act_of_god_warm_start_plan.md` | Warm-start plan; high-level objectives, validation gates, training command (bundle copy of repo README). |
| `repository_layout_and_artifacts_guide.md` | *This* document – directory tour + usage guide. |
| `requirements.txt` | Python dependencies pinned to match the VM environment. |
| `configs/` | JSON / configs for fold manifests and model settings inherited from production runs. |
| `manifests/` | Train/val/test CSV manifests for each cross-validation fold (reference). |
| `data/` | Spectra, metadata, wavelength grid, manifests used in the Act_of_God run. |
| `scripts/` | Orchestration scripts (training, evaluation, downstream proxy, etc.). |
| `src/` | Training + model source code shared with cross-validation project. |
| `checkpoints/` | Saved warm-start checkpoint(s). |
| `logs/` | Training logs from the Act_of_God run. |
| `final_analytics/` | Post-run analytics, including the latest panel evaluation outputs. |
| `artifacts/` | Placeholder for exported metrics/plots (empty unless populated manually). |

---

## Run history & latest outputs

### Dataset coverage notes

- Treatments 1–5 each contribute five samples; treatment_6 only includes samples G and I.
- Every sample/angle pair has 150 replicates, so the unique spectra total 8,100 (150 × 2 angles × 27 sample-angle combos).
- `manifest_full.csv` contains 11,100 rows because Noise2Noise pairing duplicates captures (for pairing, not new data). Duplicates point to the same `.npy` on disk; denoising overwrites the same `_denoised.npy`.

- **Latest Act_of_God run (successful)**  
  Checkpoint: `checkpoints/god_run/mamba_tiny_uv_best.pt` (EMA warm-start promoted after QA).  
  Logs: `logs/god_run/` (training log + auditing_manifest log).  
  QA artifacts: `final_analytics/panel_eval/` (summary JSON, per-spectrum/per-group CSVs, denoised panel spectra, downstream proxy metrics).
- **Aborted warm-start (pre-validation fix)**  
  Preserved under `checkpoints/god_run_failed_20251009_010956/` and `logs/god_run_failed_20251009_010956/`. Useful if you need to audit the initial run that lacked the validation loader.
- **Original fold checkpoints (reference)**  
  `checkpoints/prod/Track_H_fold_0#/` contain the five production cross-validation models. Corresponding logs live in the original Production_Mamba_Results repo; they are not mirrored here.

Use this section as the quick answer to “where are the final stats/weights?” – almost always `checkpoints/god_run/` + `final_analytics/panel_eval/`.

## Key directories & files

### Denoising the full dataset

1. Activate your environment (see *Re-running the panel QA*).
2. Run:
   ```bash
   python scripts/run_denoise_from_manifest.py \
     --manifest data/spectra_for_fold/manifest_full.csv \
     --root-dir data/spectra_for_fold \
     --checkpoint checkpoints/god_run/mamba_tiny_uv_best.pt \
     --output-root denoised_full_run \
     --dose-features-csv data/metadata/dose_features.csv \
     --dose-stats-json data/metadata/dose_stats.json \
     --batch-size 128 \
     --num-workers 4
   ```
   This writes `_denoised.npy` files mirroring the manifest structure (treatment/sample/angle/replicate).
3. Inspect `denoised_full_run/` for denoised spectra or archive as needed.

- `data/_staging/manifest.csv` contains the full 9,000-spectra manifest (5 samples × 6 treatments × 2 angles × 150 replicates). Pass it to `scripts/run_denoise_from_manifest.py` to generate `denoised_full_run_staging/`.
- `data/denoised_full_run_staging/` now contains the denoised spectra produced with the staging manifest:
  ```bash
  python scripts/run_denoise_from_manifest.py \
    --manifest data/_staging/manifest.csv \
    --root-dir data/_staging \
    --checkpoint checkpoints/god_run/mamba_tiny_uv_best.pt \
    --output-root denoised_full_run_staging \
    --dose-features-csv data/metadata/dose_features.csv \
    --dose-stats-json data/metadata/dose_stats.json \
    --batch-size 128 \
    --num-workers 4 \
    --device cuda
  ```
- Now anyone reading the repo sees both the 8,100 training manifest and the new 9,000-spectrum staging manifest alongside the denoised staging spectra.

### `scripts/`
- `plot_roi_windows.py` — legacy utility for ROI overlays (raw vs denoised vs baseline). Combine it with the denoised outputs (see 'Denoising the full dataset') to mirror the `a4b1_T1_roi_local_plots` visuals.
- `run_cross_validation.py`, `generate_fold_manifests.py` – legacy cross-val automation.
- `run_denoise_from_manifest.py` – batch inference helper.
- `evaluate_validation_panel.py` – **primary QA script** (Act_of_God version banner printed at runtime). Evaluates all readiness gates, writes reports to `final_analytics/panel_eval/`.
- `downstream_proxy_eval.py` – Gate-4 generic downstream proxy check (replicate variance, separability, dose monotonicity / peak stability).

### `src/`
- `train.py` – training loop (similar to cross-val project with warm-start logic).
- `models/mamba_uv.py` – Mamba-UV model, FiLM conditioning, etc.
- `roi_metrics.py`, `metrics.py` – ROI integrity calculations, PSNR/SAM helpers.
- `utils/calibration.py` – calibration utilities (unused in QA but part of training kit).

### `data/`
- `spectra_for_fold/` – entire spectra corpus (raw `.npy` files by treatment/sample/angle) plus `manifest_full.csv`.
- `metadata/` – dose metadata, sampling weights for FiLM conditioning and downstream proxy.
- `validation_panel.csv` – 750-spectrum fixed hold-out panel (hash-checked during QA).
- `validation_panel_summary.md`, `train_manifest_summary.md`, `training_group_counts.md` – audit notes verifying coverage.

### `checkpoints/`
- `god_run/mamba_tiny_uv_best.pt` — best early-stopped checkpoint from the Act_of_God run (EMA weights embedded).
- `god_run_failed_20251009_010956/` — partial run captured before the validation loader fix (forensics reference).
- `prod/Track_H_fold_0#/` — original cross-validation fold checkpoints (reference only).

### `logs/`
- `god_run/` — training logs and audit log (commands + per-epoch metrics).
- `god_run_failed_20251009_010956/` — logs from the aborted warm-start run.
- *(Cross-validation logs remain in the Production_Mamba_Results repo.)*

### `final_analytics/panel_eval/`
- `panel_eval_summary.json` – machine-readable QA results (per-gate pass/fail, manifest hash, git hash, ranked checkpoints). See `validation_panel_audit_results.md` in this bundle for the curated human summary.
- `*_per_spectrum_metrics.csv` – PSNR/SAM/ROI stats for each of the 750 panel spectra.
- `*_per_group_summary.csv` – aggregated metrics for each held-out group.
- `*_panel_preds/` – denoised panel spectra (`*_denoised.npy`) corresponding to current best checkpoint.
- `downstream_metrics.json` (inside each `*_panel_preds` directory) – downstream proxy results.

---

## Re-running the panel QA locally

1. **Install dependencies** (Python ≥ 3.10 recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --no-cache-dir -r requirements.txt
   ```

2. **Run the audit** (recreates all four gates):
   ```bash
   cd path/to/Act_of_God_Mamba_Results
   PANEL_HASH=$(python -c "import hashlib, pathlib;print(hashlib.sha256(pathlib.Path('data/validation_panel.csv').read_bytes()).hexdigest())")

   python scripts/evaluate_validation_panel.py \
     --checkpoint-dir checkpoints/god_run \
     --manifest data/validation_panel.csv \
     --root-dir data/spectra_for_fold \
     --baseline-dir data/spectra_for_fold \
     --metadata data/metadata/dose_features.csv \
     --wavelength-grid data/spectra_for_fold/wavelength_grid.npy \
     --expected-hash $PANEL_HASH \
     --output-dir final_analytics/panel_eval \
     --top-k 5 \
     --downstream-script scripts/downstream_proxy_eval.py \
     --batch-size 256
   ```

3. **Check outputs** under `final_analytics/panel_eval/`. The version banners in the scripts are logged to the console so you can confirm which revision ran.

---

## Reproducing training (optional)

1. Install dependencies (as above).
2. Ensure GPU + CUDA 12.1 (or adjust `requirements.txt` for your hardware).
3. Run:
   ```bash
   python -m src.train \
     --train_dir data/spectra_for_fold \
     --train_manifest data/spectra_for_fold/manifest_train.csv \
     --val_dir data/spectra_for_fold \
     --val_manifest data/validation_panel.csv \
     --init_checkpoint checkpoints/god_run/mamba_tiny_uv_best.pt \
     ...  # see act_of_god_warm_start_plan.md “Training recipe” for full argument list
   ```
4. Monitor logs in `logs/god_run/` (override `--log_dir` / `--checkpoint_dir` to avoid overwriting).

---

## Notes & Gotchas

- **Manifest hash**: QA script enforces the SHA-256 hash of `data/validation_panel.csv`. If you intentionally edit the panel, update the hash or run without `--expected-hash`.
- **Downstream proxy**: Gate-4 uses `scripts/downstream_proxy_eval.py`. It writes `downstream_metrics.json` beside the denoised spectra and returns pass/fail to the main QA script.
- **Virtual environment**: `/mnt/spectra-env` is not included. Recreate it locally using `requirements.txt` before running training or evaluation.
- **Logs vs analytics**: `logs/` contains raw training output; `final_analytics/panel_eval/` is the curated QA snapshot. Keep both for reproducibility.

---

Need deeper analytics? Start with `final_analytics/panel_eval/mamba_tiny_uv_best_per_spectrum_metrics.csv` – every panel spectrum’s PSNR/SAM/ROI stats are there, keyed by `relative_path`.
