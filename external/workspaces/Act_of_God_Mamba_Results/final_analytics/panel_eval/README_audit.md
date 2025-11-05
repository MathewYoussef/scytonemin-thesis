Act_of_God Panel Audit (Gate Results)
====================================

Audit run: 2025-10-10 07:03:46Z UTC
Scripts: evaluate_validation_panel.py & downstream_proxy_eval.py (2025-10-09 version banners)
Checkpoint(s) evaluated: /mnt/Act_of_God/checkpoints/god_run/mamba_tiny_uv_best.pt (top-1 composite)

Gate Summary:
- Gate 1 — Aggregate metrics: **FAIL**
  * PSNR_mean = 44.153 dB (pass threshold ≥ 18 dB)
  * PSNR_std = 4.968 dB (**fails** threshold ≤ 4 dB)
  * SAM_mean = 1.039° (pass threshold ≤ 9°)

- Gate 2 — Tail metrics: **PASS**
  * ≥95% spectra satisfy PSNR ≥ 14 dB and SAM ≤ 12°

- Gate 3 — ROI integrity: **FAIL**
  * Peak/area error limit (≤10% for ≥90% spectra) not satisfied;
    zero spurious peaks criterion also violated per summary JSON.

- Gate 4 — Downstream proxy: **PASS**
  * Replicate variance reduction target met,
  * Separability ratio ≥ baseline,
  * Dose monotonicity / fallback peak-F1 satisfied.

Conclusion: Readiness gates NOT satisfied (aggregate variance & ROI integrity failed).
Refer to panel_eval_summary.json, per-spectrum metrics, and downstream_metrics.json for detailed breakdown.
