# Plot Concepts for Thesis Means

- Dose trajectories with 95% CIs: line plots of trimmed means for Chrom and DAD totals/reduced/oxidized vs dose (UVA index) to highlight dose₄/dose₅ peaks and the reduced>oxidized gap (`dose_level_summary.csv`).
- Sequential delta bars: dose-to-dose trimmed-mean changes with CI whiskers per metric, exposing the + + + − + and + − + + − signatures and significant jumps (`dose_pattern_sequential_deltas.csv`).
- Cross-assay concordance: scatterplots of Chrom vs DAD trimmed means for Total/Reduced/Oxidized with Deming regression and 1:1 reference plus Pearson r annotations (`chrom_dad_alignment.csv`).
- UV regime context: dual-axis figure plotting UVA/UVB intensities alongside Total and Reduced trajectories, pinpointing the dose₅→dose₆ UVB step-down while UVA rises (`dose_level_summary.csv`).
- Replicates vs trimmed means: jittered replicate points per dose overlaid with trimmed means and 95% CIs for each assay’s Total pool to show robustness vs raw spread (`Combined_Scytonemin_Concentrations.csv`, `dose_level_summary.csv`).
- Reduced vs oxidized panels: split charts for each assay comparing reduced and oxidized trajectories, reinforcing reduced-pool dominance at higher doses (`dose_level_summary.csv`).
- Peak comparison panels: bar or lollipop charts of peak doses with adjacent-dose values to visualize late softening and the reduced rebound (`dose_level_summary.csv`).
- Quadratic-fit curvature or residual plot: display fitted vs observed or second-derivative cues to illustrate the concave-down dose₄–dose₅ behavior (`dose_pattern_summary.csv`).
- UVA/UVB trend regression plots: scatter with weighted regression overlays for totals and reduced pools, annotated with slope/Kendall τ to emphasize “mostly increasing with late dip” character (`dose_trend_stats.csv`).
