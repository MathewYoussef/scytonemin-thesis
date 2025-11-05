import os
from pathlib import Path

MPLCONFIG_PATH = Path("figures_journey") / "mplconfig"
try:
    MPLCONFIG_PATH.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # On read-only sandboxes this fails; assume directory already exists.
    pass
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def trimmed_mean(values, proportion=0.2):
    values = np.asarray(values, dtype=float)
    m = len(values)
    k = int(np.floor(proportion * m))
    if k >= m // 2:
        return values.mean()
    sorted_vals = np.sort(values)
    return sorted_vals[k : m - k].mean()


def load_data():
    reps = pd.read_csv("Combined_Scytonemin_Concentrations.csv")
    summary = pd.read_csv("dose_level_summary.csv").sort_values("p_uva_mw_cm2")
    deltas = pd.read_csv("dose_pattern_sequential_deltas.csv")
    trend = pd.read_csv("dose_trend_stats.csv")
    alignment = pd.read_csv("chrom_dad_alignment.csv")
    return reps, summary.reset_index(drop=True), deltas, trend, alignment


def attach_dose_metadata(reps, summary):
    summary = summary.copy()
    summary["dose_index"] = np.arange(1, len(summary) + 1)
    dose_lookup = summary.set_index("dose_id")["dose_index"].to_dict()
    merge_cols = ["p_uva_mw_cm2", "p_uvb_mw_cm2"]
    reps = reps.merge(summary[merge_cols + ["dose_id", "dose_index"]], on=merge_cols, how="left")
    return reps, summary, dose_lookup


def fig_replicates_vs_trimmed(reps, summary, fig_dir):
    rng = np.random.default_rng(42)

    def panel(metric_prefix, axis, point_color):
        metric = f"{metric_prefix}_mg_per_gDW"
        mean_col = f"{metric_prefix}_mg_per_gDW_trimmed_mean"
        low_col = f"{metric_prefix}_mg_per_gDW_ci_low"
        high_col = f"{metric_prefix}_mg_per_gDW_ci_high"
        for dose_idx, group in reps.groupby("dose_index"):
            jitter = rng.uniform(-0.12, 0.12, len(group))
            axis.scatter(
                np.full(len(group), dose_idx) + jitter,
                group[metric],
                color=point_color,
                alpha=0.6,
                s=25,
            )
        axis.errorbar(
            summary["dose_index"],
            summary[mean_col],
            yerr=[
                summary[mean_col] - summary[low_col],
                summary[high_col] - summary[mean_col],
            ],
            fmt="o-",
            color="black",
            ecolor="black",
            capsize=3,
            label="Trimmed mean ± CI",
        )
        axis.set_xlabel("Dose index (ordered by UVA)")
        axis.set_ylabel("Concentration (mg·gDW⁻¹)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    panel("chrom_total", axes[0], "#2E86AB")
    axes[0].set_title("Chromatogram Totals")
    panel("dad_total", axes[1], "#B23A48")
    axes[1].set_title("DAD Totals")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_replicates_vs_trimmed.png", dpi=300)
    plt.close(fig)


def fig_dose_trajectories(summary, fig_dir):
    metrics = {
        "Chromatogram": ["chrom_total", "chrom_oxidized", "chrom_reduced"],
        "DAD": ["dad_total", "dad_oxidized", "dad_reduced"],
    }
    colors = {"total": "#1f77b4", "oxidized": "#ff7f0e", "reduced": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, (title, metric_list) in zip(axes, metrics.items()):
        for metric in metric_list:
            suffix = metric.split("_")[1]
            ax.plot(
                summary["p_uva_mw_cm2"],
                summary[f"{metric}_mg_per_gDW_trimmed_mean"],
                marker="o",
                label=suffix.title(),
                color=colors[suffix],
            )
        ax.axvline(
            summary.loc[4, "p_uva_mw_cm2"],
            color="grey",
            linestyle="--",
            linewidth=1,
            label="Dose₅ UVA" if title == "Chromatogram" else None,
        )
        ax.set_xlabel("UVA (mW·cm⁻²)")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.set_title(f"{title} Dose Trajectories")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure2_dose_trajectories.png", dpi=300)
    plt.close(fig)


def weighted_fit(x, y, sd):
    w = 1.0 / (sd ** 2)
    X = np.vstack([np.ones_like(x), x]).T
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta


def fig_weighted_trends(summary, fig_dir):
    metrics = {
        "Chromatogram": ["chrom_total", "chrom_reduced"],
        "DAD": ["dad_total", "dad_reduced"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, (title, metric_list) in zip(axes, metrics.items()):
        for metric in metric_list:
            x = summary["p_uva_mw_cm2"].values
            y = summary[f"{metric}_mg_per_gDW_trimmed_mean"].values
            sd = summary[f"{metric}_mg_per_gDW_trimmed_sd"].values
            beta = weighted_fit(x, y, sd)
            xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
            ys = beta[0] + beta[1] * xs
            label = metric.split("_")[1].title()
            ax.scatter(x, y, s=35, label=f"{label} means")
            ax.plot(xs, ys, label=f"{label} fit (slope {beta[1]:.3f})")
        ax.set_xlabel("UVA (mW·cm⁻²)")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.set_title(f"Weighted UVA Trends — {title}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_weighted_trends.png", dpi=300)
    plt.close(fig)


def fig_dose_deltas(deltas, fig_dir):
    focus_metrics = [
        "Chrom Total (mg·gDW⁻¹)",
        "Chrom Reduced",
        "DAD Total (mg·gDW⁻¹)",
        "DAD Reduced",
    ]
    subset = deltas[deltas["metric"].isin(focus_metrics)].copy()
    start_idx = pd.to_numeric(
        subset["transition"].str.extract(r"dose_(\\d)")[0],
        errors="coerce",
    )
    subset = subset.assign(start_dose=start_idx).dropna(subset=["start_dose"])
    subset["start_dose"] = subset["start_dose"].astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()
    for ax, metric in zip(axes, focus_metrics):
        mdata = subset[subset["metric"] == metric]
        ax.bar(
            mdata["transition"],
            mdata["delta_trimmed_mean"],
            color=np.where(mdata["delta_trimmed_mean"] >= 0, "#2ca02c", "#d62728"),
        )
        ax.errorbar(
            mdata["transition"],
            mdata["delta_trimmed_mean"],
            yerr=[
                mdata["delta_trimmed_mean"] - mdata["delta_ci_low"],
                mdata["delta_ci_high"] - mdata["delta_trimmed_mean"],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title(metric)
        ax.set_ylabel("Delta (mg·gDW⁻¹)")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure4_dose_deltas.png", dpi=300)
    plt.close(fig)


def fig_assay_concordance(summary, alignment, fig_dir):
    pairs = [
        ("total", "Chrom ↔ DAD Totals"),
        ("oxidized", "Chrom ↔ DAD Oxidized"),
        ("reduced", "Chrom ↔ DAD Reduced"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (suffix, title) in zip(axes, pairs):
        x = summary[f"chrom_{suffix}_mg_per_gDW_trimmed_mean"]
        y = summary[f"dad_{suffix}_mg_per_gDW_trimmed_mean"]
        ax.scatter(x, y, color="#1f77b4", s=35)
        match = alignment[alignment["metric"] == suffix]
        if not match.empty:
            slope = match["deming_slope"].iloc[0]
            intercept = match["deming_intercept"].iloc[0]
            xs = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
            ax.plot(xs, intercept + slope * xs, color="black", linestyle="--", label=f"Slope {slope:.2f}")
        ax.set_xlabel("Chrom trimmed mean (mg·gDW⁻¹)")
        ax.set_ylabel("DAD trimmed mean (mg·gDW⁻¹)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_assay_concordance.png", dpi=300)
    plt.close(fig)


def fig_trimmed_sensitivity(reps, dose_lookup, fig_dir):
    metrics = ["chrom_total_mg_per_gDW", "dad_total_mg_per_gDW"]
    records = []
    for (dose_id, dose_index), group in reps.groupby(["dose_id", "dose_index"]):
        for metric in metrics:
            baseline = trimmed_mean(group[metric].values)
            records.append(
                {
                    "metric": metric,
                    "dose_index": dose_index,
                    "type": "Baseline",
                    "value": baseline,
                }
            )
            for idx in group.index:
                loo_values = group.loc[group.index != idx, metric].values
                if len(loo_values) == 0:
                    continue
                loo_mean = trimmed_mean(loo_values)
                records.append(
                    {
                        "metric": metric,
                        "dose_index": dose_index,
                        "type": "Leave-one-out",
                        "value": loo_mean,
                    }
                )
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    titles = {
        "chrom_total_mg_per_gDW": "Sensitivity — Chrom Totals",
        "dad_total_mg_per_gDW": "Sensitivity — DAD Totals",
    }
    for ax, metric in zip(axes, metrics):
        subset = df[df["metric"] == metric]
        for dose_idx in sorted(subset["dose_index"].unique()):
            dose_data = subset[subset["dose_index"] == dose_idx]
            baseline = dose_data[dose_data["type"] == "Baseline"]["value"].iloc[0]
            loo_vals = dose_data[dose_data["type"] == "Leave-one-out"]["value"]
            x_vals = np.full(len(loo_vals), dose_idx)
            ax.scatter(
                x_vals,
                loo_vals,
                color="#FFB74D",
                alpha=0.6,
                label="Leave-one-out" if dose_idx == 1 else None,
            )
            ax.scatter(
                [dose_idx],
                [baseline],
                color="#0D47A1",
                s=45,
                label="Baseline trimmed" if dose_idx == 1 else None,
            )
            if not loo_vals.empty:
                ax.vlines(dose_idx, loo_vals.min(), loo_vals.max(), color="grey", linewidth=1)
        ax.set_xlabel("Dose index")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.set_title(titles[metric])
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure6_trimmed_mean_sensitivity.png", dpi=300)
    plt.close(fig)


def main():
    ensure_style()
    fig_dir = Path("figures_journey")
    fig_dir.mkdir(exist_ok=True)
    reps, summary, deltas, trend, alignment = load_data()
    reps, summary, dose_lookup = attach_dose_metadata(reps, summary)

    fig_replicates_vs_trimmed(reps, summary, fig_dir)
    fig_dose_trajectories(summary, fig_dir)
    fig_weighted_trends(summary, fig_dir)
    fig_dose_deltas(deltas, fig_dir)
    fig_assay_concordance(summary, alignment, fig_dir)
    fig_trimmed_sensitivity(reps, dose_lookup, fig_dir)


if __name__ == "__main__":
    main()
