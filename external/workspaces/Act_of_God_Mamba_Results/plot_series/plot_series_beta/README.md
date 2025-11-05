# Plot Series Beta

Prototype workspace for the second-generation diagnostics plots described in `../README.md`.  
Everything here is self-contained so you can iterate without touching the main code paths.

## What lives here
- `generate_plots.py`: CLI entry point that orchestrates data loading, metric computation, and figure creation.
- `data_loader.py`: Helpers for pulling raw + denoised spectra and wavelength grids from the staging areas.
- `metrics.py`: Numerical routines (SNR, variance, variance ratios, SAM windows, preservation indices, effect sizes).
- `plotting.py`: Matplotlib figure builders for the SNR/variance diagnostics and downstream panels.
- `paths.py`: Centralises default locations for the staging data and output drop-site.

## Quickstart
```bash
cd plot_series/plot_series_beta
python generate_plots.py --output-dir ./outputs
```

Key flags:
- `--raw-dir` / `--denoised-dir`: Override the defaults if you stage alternative runs.
- `--control-treatment`: Which treatment acts as the high-SNR reference (defaults to `treatment_6`).
- `--bootstrap-samples`: Number of bootstrap draws for the variance ratio ribbon CI (default 500).
- `--group-by`: Choose `treatment` (default) to collapse samples/angles into six tidy bands, or `group` to inspect every sample×angle combination.
- `--snr-delta-max`: Clamp the ΔSNR heatmap colour range to ±value (dB) so subtle gains stay visible.

Figures are written to the chosen output directory with descriptive filenames (e.g. `fig_A_snr_heatmap.png`).

### Outputs
- `fig_A_snr_heatmap.png`
- `fig_B_snr_summary.png`
- `fig_C_variance_heatmaps.png`
- `fig_D_variance_ratio.png`
- `fig_E_sam_panels.png`
- `fig_F_preservation_indices.png`
- `fig_G_roi_micro_panels.png`
- `fig_H_effect_sizes.png`

## Status
Initial plumbing for data IO, analytics kernels, and figure scaffolding is in progress. Expect incremental commits as we refine metric definitions and styling.
