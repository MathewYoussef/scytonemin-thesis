import os
from pathlib import Path

# Ensure matplotlib writes cache/control files inside the workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path("figures_journey").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def weighted_line(x, intercept, slope):
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    return xs, intercept + slope * xs


def main():
    summary = pd.read_csv("dose_level_summary.csv").sort_values("p_uva_mw_cm2")
    stats = (
        pd.read_csv("dose_trend_stats.csv")
        .query("axis == 'UVA'")
        .set_index("metric")
    )

    panels = [
        ("Chromatogram Total", "chrom_total_mg_per_gDW"),
        ("Chromatogram Reduced", "chrom_reduced_mg_per_gDW"),
        ("DAD Total", "dad_total_mg_per_gDW"),
        ("DAD Reduced", "dad_reduced_mg_per_gDW"),
    ]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    chrom_total_y = None
    chrom_total_x = None

    for ax, (title, prefix) in zip(axes, panels):
        x = summary["p_uva_mw_cm2"].values
        y = summary[f"{prefix}_trimmed_mean"].values
        ci_low = summary[f"{prefix}_ci_low"].values
        ci_high = summary[f"{prefix}_ci_high"].values

        ax.errorbar(
            x,
            y,
            yerr=[y - ci_low, ci_high - y],
            fmt="o",
            color="#1f77b4",
            ecolor="#4d4d4d",
            capsize=3,
            label="Trimmed mean ± 95% CI",
        )

        metric_key = prefix.replace("_trimmed_mean", "")
        metric_stats = stats.loc[metric_key]
        xs, ys = weighted_line(x, metric_stats["intercept"], metric_stats["slope"])
        ax.plot(xs, ys, color="#d62728", label="Weighted linear fit")

        text = (
            f"Slope = {metric_stats['slope']:.3f} ± {metric_stats['slope_se']:.3f}\n"
            f"R² = {metric_stats['r_squared']:.2f}\n"
            f"Pearson r = {metric_stats['pearson_r']:.2f}"
        )
        ax.text(
            0.02,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
        )

        ax.set_title(title)
        ax.set_ylabel("Concentration (mg·gDW⁻¹)")
        ax.legend(loc="upper left", fontsize=8)

        if prefix.startswith("chrom_total"):
            chrom_total_x = x
            chrom_total_y = y

    for ax in axes[2:]:
        ax.set_xlabel("UVA (mW·cm⁻²)")

    # Annotate the UVB decrease on the first panel
    final_idx = -1
    ax0 = axes[0]
    assert chrom_total_x is not None and chrom_total_y is not None
    ax0.annotate(
        "UVB decreases\n(0.768→0.707) with rebound",
        xy=(chrom_total_x[final_idx], chrom_total_y[final_idx]),
        xytext=(chrom_total_x[final_idx] - 0.9, chrom_total_y.max() + 0.2),
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=9,
        ha="center",
    )

    fig.tight_layout()
    output_path = Path("figures_journey") / "fig2_trend_lines.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
