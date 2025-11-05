# Act_of_God Warm Start Plan

This experiment tests whether the best-performing fold 02 checkpoint can be promoted into a production-ready, all-data model via an “Act_of_God” run. We warm start from `Track_H_fold_02` weights, train on every spectrum, and police progress with a fixed validation panel. If the panel passes the readiness gates we ship the model; if not, we fall back to Act B/C.

## Why attempt this?
- Fold 02 already cleared the production bar (PSNR ≈ 22.4 dB, SAM ≈ 6.7°), so its weights encode the behaviour we want.
- One consolidated training pass is faster than iterating on five folds—if we can guard against drift.
- A fixed validation panel lets us detect overfit or bias as soon as it appears.

**Risks:** Fold 02 was tuned on a narrower 3/1/1 split. Training on all spectra could reintroduce fold-specific quirks, so the validation panel is mandatory. If the panel fails any gate, abort and pivot to Act B.

## Validation panel (fixed hold-out)
- Size: 750 spectra (~6.8 % of the corpus) drawn from three group_ids.
- Composition:
  - `treatment_5/sample_H/6Oclock` — hard case from folds 01/05.
  - `treatment_3/sample_J/12Oclock` — representative “easy” case from the strong fold 02 regime (300 spectra).
  - `treatment_6/sample_I/6Oclock` — gold-standard control (zero-dose reference, 150 spectra).
- Rules: hold out entire groups (all replicates, both polarities). No augmentation in this split. Use the manifest `data/validation_panel.csv` recorded below.

## Training set coverage
- Manifest: `data/spectra_for_fold/manifest_train.csv` (10 350 spectra).
- Every treatment includes all five production samples (treatment 6 has its two controls).  
- Within each sample, both 6 O’clock and 12 O’clock angles remain, except where a full group has been reserved for validation (see summary in `data/validation_panel_summary.md`).
- Compared with the fold 3/1/1 splits, no samples are omitted from training; only the three validation group_ids (750 spectra) are excluded to prevent leakage.
- Audit helpers:
  - `data/train_manifest_summary.md` — per-treatment sample/angle coverage.
  - `data/training_group_counts.md` — replicate counts by group_id.
  - `data/validation_panel_summary.md` — validation hold-outs.

## Readiness gates
Evaluate on the fixed panel every epoch (or via early-stopping hook). The run only ships if **all** of the following hold:
1. `PSNR_mean ≥ 18 dB`, `PSNR_std ≤ 4 dB`, `SAM_mean ≤ 9°`.
2. Tail safety: ≥95 % of spectra with `PSNR ≥ 14 dB` **and** `SAM ≤ 12°`.
3. ROI integrity: ≤10 % error for ≥90 % of spectra on peak/area metrics; zero spurious peaks.
4. Downstream proxy (baseline pipeline) performs no worse than the raw+baseline reference.

If any gate fails → mark the run **Fail**, archive diagnostics, and immediately resume the Act B plan.

## Training recipe
```
python -m src.train \
  --model mamba_tiny_uv \
  --train_dir data/spectra_for_fold \
  --train_manifest data/spectra_for_fold/manifest_train.csv \
  --val_manifest data/validation_panel.csv \
  --sequence_length 601 \
  --epochs 400 \
  --early_stop_patience 30 \
  --bs 600 \
  --lr 3e-4 \
  --lr_min 3e-5 \
  --weight_decay 1e-4 \
  --noise2noise --noise2noise_pairwise \
  --geometry_film --film_hidden_dim 64 \
  --film_features cos_theta UVA_total UVB_total UVA_over_UVB P_UVA_mW_cm2 P_UVB_mW_cm2 UVA_norm UVB_norm \
  --dose_features_csv data/metadata/dose_features.csv \
  --dose_sampling_weights_csv data/metadata/dose_sampling_weights.csv \
  --dose_stats_json data/metadata/dose_stats.json \
  --lambda_weights data/spectra_for_fold/lambda_stats.npz \
  --dip_loss --dip_weight 1.0 --dip_m 6 \
  --dip_window_half_nm 7.0 --dip_min_area 5e-4 \
  --dip_w_area 2.5 --dip_w_equivalent_width 1.5 --dip_w_centroid 2.0 --dip_w_depth 0.3 \
  --dip_underfill_factor 2.0 --dip_detect_sigma_nm 1.0 --baseline local --baseline_guard_nm 10.0 \
  --derivative_weight 0.3 --deriv_weight_roi 0.6 --deriv_roi_min 320.0 --deriv_roi_max 500.0 \
  --curvature_weight_roi 0.3 \
  --d_model 384 --n_layers 8 --d_state 16 \
  --amp --amp_dtype bf16 --ema --ema_decay 0.999 --cudnn_benchmark \
  --init_checkpoint checkpoints/prod/Track_H_fold_02/mamba_tiny_uv_best.pt \
  --log_dir logs/god_run \
  --checkpoint_dir checkpoints/god_run \
  --audit_log logs/god_run/auditing_manifest.log
```
- Start from the saved fold 02 checkpoint.
- Apply the same deterministic baseline centring + scaling transform to train and validation splits; jitter/cutout remain train-only.
- Skip validation Noise2Noise (`--val_noise2noise` off) and reverse-time TTA (`--tta_reverse` omitted) so panel metrics reflect clean spectra.
- Keep Act A preprocessing (baseline centring, jitter, cutout) for the training data; never touch the panel spectra.
- Early stopping monitors the validation panel metrics (patience ≈ 30 epochs). Allow the max 400 epochs so the warm start can converge.

## Monitoring & evaluation
- Watch GPU with `watch -n 1 nvidia-smi`; monitor logs via `tail -f logs/god_run/train.log`.
- Capture validation metrics to `logs/god_run/metrics.csv` (existing training script already emits this).
- After training, run QA notebooks/scripts to compute ROI integrity and downstream proxy performance on the fixed panel.

## Fallback plan
If any readiness gate fails:
1. Archive run artefacts under `artifacts/god_run_failed_YYYYMMDD`.
2. Document failure mode in `final_analytics/reviewing_the_5fold_run.md`.
3. Resume the Act B schedule (calibrator head + loss rebalance). The warm checkpoint from this run can still seed Act B.

## Quick checklist
- [ ] Validation panel CSV present and versioned.
- [ ] Fold 02 checkpoint copied into `checkpoints/prod/Track_H_fold_02/`.
- [ ] Training manifest (`data/spectra_for_fold/manifest_train.csv`) excludes the validation panel.
- [ ] Validation panel normalized identically to training (no jitter/cutout, no val Noise2Noise, no TTA).
- [ ] Early stopping + logging configured for the panel.
- [ ] QA scripts ready for PSNR/SAM tail metrics, ROI checks, downstream proxy.

## Panel QA & Shipping
After early stopping, verify the fixed panel before shipping:
```
python scripts/evaluate_validation_panel.py \
  --checkpoint-dir checkpoints/god_run \
  --manifest data/validation_panel.csv \
  --root-dir data/spectra_for_fold \
  --baseline-dir data/spectra_for_fold \
  --metadata data/metadata/dose_features.csv \
  --wavelength-grid data/spectra_for_fold/wavelength_grid.npy \
  --output-dir final_analytics/panel_eval \
  --downstream-script scripts/downstream_proxy_eval.py
```

- The script replays inference with the same deterministic normalization as training (no eval-time aug/TTA, EMA weights if present) and evaluates every readiness gate (aggregates, tails, ROI integrity, downstream proxy).
- Supply `--expected-hash $(shasum -a 256 data/validation_panel.csv | cut -d' ' -f1)` to fail fast if the panel changed.
- Optionally pass `--downstream-script scripts/downstream_proxy_eval.py --baseline-dir data/spectra_for_fold --metadata data/metadata/dose_features.csv --wavelength-grid data/spectra_for_fold/wavelength_grid.npy` to exercise the downstream proxy gate.
- Outputs live in `final_analytics/panel_eval/`, including per-spectrum metrics, per-group summaries, manifest/code hashes, and the top-ranked checkpoint selected by composite panel score.
