from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve()
REPO_DIR = HERE
while not (REPO_DIR / "dose_level_summary.csv").exists():
    if REPO_DIR.parent == REPO_DIR:
        raise FileNotFoundError(
            "Could not locate repository root containing dose_level_summary.csv"
        )
    REPO_DIR = REPO_DIR.parent

PLOT_DIR = REPO_DIR / "plots_for_thesis_means_3"
OUTPUT_DIR = PLOT_DIR / "figures"


def ensure_environment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reps = pd.read_csv(REPO_DIR / "Combined_Scytonemin_Concentrations.csv")
    summary = pd.read_csv(REPO_DIR / "dose_level_summary.csv")
    deltas = pd.read_csv(REPO_DIR / "dose_pattern_sequential_deltas.csv")
    trend = pd.read_csv(REPO_DIR / "dose_trend_stats.csv")
    alignment = pd.read_csv(REPO_DIR / "chrom_dad_alignment.csv")
    return reps, summary, deltas, trend, alignment


def _long_summary(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    metric_map = [
        ("chrom_total", "Chromatogram", "Total"),
        ("chrom_oxidized", "Chromatogram", "Oxidized"),
        ("chrom_reduced", "Chromatogram", "Reduced"),
        ("dad_total", "DAD", "Total"),
        ("dad_oxidized", "DAD", "Oxidized"),
        ("dad_reduced", "DAD", "Reduced"),
    ]
    ordered = summary.reset_index(drop=True)
    for idx, row in ordered.iterrows():
        for base, assay, pool in metric_map:
            records.append(
                {
                    "dose_id": row["dose_id"],
                    "dose_index": idx + 1,
                    "p_uva_mw_cm2": row["p_uva_mw_cm2"],
                    "p_uvb_mw_cm2": row["p_uvb_mw_cm2"],
                    "assay": assay,
                    "pool": pool,
                    "mean": row[f"{base}_mg_per_gDW_trimmed_mean"],
                    "ci_low": row[f"{base}_mg_per_gDW_ci_low"],
                    "ci_high": row[f"{base}_mg_per_gDW_ci_high"],
                }
            )
    return pd.DataFrame.from_records(records)


def plot_dose_trajectories(summary: pd.DataFrame) -> None:
    tidy = _long_summary(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, (assay, chunk) in zip(axes, tidy.groupby("assay"), strict=False):
        for pool, pdata in chunk.groupby("pool"):
            sns.lineplot(
                data=pdata,
                x="p_uva_mw_cm2",
                y="mean",
                marker="o",
                ax=ax,
                label=pool,
            )
            ax.fill_between(
                pdata["p_uva_mw_cm2"],
                pdata["ci_low"],
                pdata["ci_high"],
                alpha=0.15,
            )
        ax.set_title(f"{assay} trimmed means vs UVA")
        ax.set_xlabel("UVA (mW/cm^2)")
        ax.set_ylabel("Concentration (mg/gDW)")
    fig.suptitle("Dose trajectories with 95% bootstrap CIs", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dose_trajectories.png", dpi=300)
    plt.close(fig)


def plot_sequential_deltas(deltas: pd.DataFrame) -> None:
    focus_metrics = [
        "Chrom Total (mg·gDW⁻¹)",
        "Chrom Reduced",
        "DAD Total (mg·gDW⁻¹)",
        "DAD Reduced",
    ]
    focus = deltas[deltas["metric"].isin(focus_metrics)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()
    for ax, (metric, chunk) in zip(axes, focus.groupby("metric"), strict=False):
        sns.barplot(
            data=chunk,
            x="transition",
            y="delta_trimmed_mean",
            ax=ax,
            color="#4C72B0",
            saturation=0.8,
        )
        for patch, val in zip(ax.patches, chunk["delta_trimmed_mean"], strict=False):
            patch.set_facecolor("#2ca02c" if val >= 0 else "#d62728")
        ax.errorbar(
            chunk["transition"],
            chunk["delta_trimmed_mean"],
            yerr=[
                chunk["delta_trimmed_mean"] - chunk["delta_ci_low"],
                chunk["delta_ci_high"] - chunk["delta_trimmed_mean"],
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Delta trimmed mean (mg/gDW)")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Sequential dose-to-dose changes with 95% CIs", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dose_sequential_deltas.png", dpi=300)
    plt.close(fig)


def plot_replicates_vs_trimmed(reps: pd.DataFrame, summary: pd.DataFrame) -> None:
    work_summary = summary.copy()
    if "dose_index" not in work_summary.columns:
        work_summary = work_summary.sort_values("p_uva_mw_cm2").reset_index(drop=True)
        work_summary["dose_index"] = work_summary.index + 1
    merged = reps.merge(
        work_summary[["p_uva_mw_cm2", "p_uvb_mw_cm2", "dose_id", "dose_index"]],
        on=["p_uva_mw_cm2", "p_uvb_mw_cm2"],
        how="left",
    )
    metrics = [
        ("chrom_total_mg_per_gDW", "Chromatogram total"),
        ("dad_total_mg_per_gDW", "DAD total"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        sns.stripplot(
            data=merged,
            x="dose_index",
            y=metric,
            jitter=0.25,
            alpha=0.6,
            ax=ax,
        )
        mean_col = metric + "_trimmed_mean"
        low_col = metric + "_ci_low"
        high_col = metric + "_ci_high"
        ax.errorbar(
            work_summary["dose_index"],
            work_summary[mean_col],
            yerr=[
                work_summary[mean_col] - work_summary[low_col],
                work_summary[high_col] - work_summary[mean_col],
            ],
            fmt="o-",
            color="black",
            ecolor="black",
            capsize=3,
            label="Trimmed mean +/- CI",
        )
        ax.set_title(title)
        ax.set_xlabel("Dose index (ordered by UVA)")
        ax.set_ylabel("Concentration (mg/gDW)")
        ax.legend()
    fig.suptitle("Replicate spread versus trimmed means", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "replicates_vs_trimmed.png", dpi=300)
    plt.close(fig)


def plot_cross_assay_concordance(summary: pd.DataFrame, alignment: pd.DataFrame) -> None:
    pairs = [
        ("total", "Total pools"),
        ("oxidized", "Oxidized pools"),
        ("reduced", "Reduced pools"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    for ax, (suffix, title) in zip(axes, pairs, strict=False):
        chrom_col = f"chrom_{suffix}_mg_per_gDW_trimmed_mean"
        dad_col = f"dad_{suffix}_mg_per_gDW_trimmed_mean"
        x = summary[chrom_col]
        y = summary[dad_col]
        sns.scatterplot(x=x, y=y, ax=ax, s=50, color="#1f77b4")
        min_bound = min(x.min(), y.min()) * 0.9
        max_bound = max(x.max(), y.max()) * 1.1
        grid = np.linspace(min_bound, max_bound, 100)
        ax.plot(grid, grid, linestyle=":", color="grey", label="1:1 line")
        match = alignment[alignment["metric"] == suffix]
        if not match.empty:
            slope = match["deming_slope"].iloc[0]
            intercept = match["deming_intercept"].iloc[0]
            pearson = match["pearson_r"].iloc[0]
            ax.plot(
                grid,
                intercept + slope * grid,
                color="#d62728",
                label=f"Deming slope {slope:.2f}",
            )
            ax.text(0.05, 0.92, f"Pearson r = {pearson:.2f}", transform=ax.transAxes)
        ax.set_title(f"{title}")
        ax.set_xlabel("Chrom trimmed mean (mg/gDW)")
        ax.set_ylabel("DAD trimmed mean (mg/gDW)")
        ax.legend()
    fig.suptitle("Chromatogram versus DAD concordance", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "assay_concordance.png", dpi=300)
    plt.close(fig)


def plot_uv_context(summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    dose_idx = summary["dose_index"]
    ax1.plot(dose_idx, summary["p_uva_mw_cm2"], marker="o", color="#1f77b4", label="UVA")
    ax1.plot(dose_idx, summary["p_uvb_mw_cm2"], marker="o", color="#ff7f0e", label="UVB")
    ax1.set_xlabel("Dose index (ordered by UVA)")
    ax1.set_ylabel("Irradiance (mW/cm^2)")
    ax1.set_xticks(dose_idx)
    ax1.set_xticklabels(summary["dose_id"], rotation=0)
    ax2 = ax1.twinx()
    ax2.plot(
        dose_idx,
        summary["dad_total_mg_per_gDW_trimmed_mean"],
        marker="s",
        color="#2ca02c",
        label="DAD total",
    )
    ax2.plot(
        dose_idx,
        summary["dad_reduced_mg_per_gDW_trimmed_mean"],
        marker="s",
        color="#9467bd",
        label="DAD reduced",
    )
    ax2.set_ylabel("Concentration (mg/gDW)")
    ax1.axvline(dose_idx.iloc[-1] - 0.5, color="grey", linestyle="--", linewidth=1)
    ax1.annotate(
        "UVB step-down",
        xy=(dose_idx.iloc[-1], summary["p_uvb_mw_cm2"].iloc[-1]),
        xytext=(-50, -25),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="grey"),
        ha="right",
    )
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.suptitle("UV regimen and DAD concentrations", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uv_context.png", dpi=300)
    plt.close(fig)


def plot_reduced_vs_oxidized(summary: pd.DataFrame) -> None:
    configs = [
        ("chrom", "Chromatogram"),
        ("dad", "DAD"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    x = summary["p_uva_mw_cm2"]
    for ax, (prefix, title) in zip(axes, configs, strict=False):
        reduced = summary[f"{prefix}_reduced_mg_per_gDW_trimmed_mean"]
        oxidized = summary[f"{prefix}_oxidized_mg_per_gDW_trimmed_mean"]
        ax.plot(x, reduced, marker="o", color="#2ca02c", label="Reduced")
        ax.plot(x, oxidized, marker="o", color="#d62728", label="Oxidized")
        ax.fill_between(x, oxidized, reduced, color="#2ca02c", alpha=0.15)
        ax.set_title(f"{title} reduced vs oxidized")
        ax.set_xlabel("UVA (mW/cm^2)")
        ax.set_ylabel("Concentration (mg/gDW)")
        ax.legend()
    fig.suptitle("Reduced pools dominate at higher UVA", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reduced_vs_oxidized.png", dpi=300)
    plt.close(fig)


def plot_peak_comparison(summary: pd.DataFrame) -> None:
    metrics = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom total"),
        ("chrom_reduced_mg_per_gDW_trimmed_mean", "Chrom reduced"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD total"),
        ("dad_reduced_mg_per_gDW_trimmed_mean", "DAD reduced"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=False)
    highlight_color = "#d62728"
    for ax, (metric, title) in zip(axes.flatten(), metrics, strict=False):
        peak_idx = int(summary[metric].idxmax())
        start = max(0, peak_idx - 1)
        end = min(len(summary) - 1, peak_idx + 1)
        window = summary.iloc[start : end + 1]
        ax.plot(
            window["dose_index"],
            window[metric],
            marker="o",
            color="#1f77b4",
            label="Local window",
        )
        ax.scatter(
            summary.loc[peak_idx, "dose_index"],
            summary.loc[peak_idx, metric],
            color=highlight_color,
            s=70,
            label="Peak dose",
            zorder=3,
        )
        for xi, yi, dose in zip(
            window["dose_index"],
            window[metric],
            window["dose_id"],
            strict=False,
        ):
            ax.text(xi, yi, dose, ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Dose index")
        ax.set_ylabel("Trimmed mean (mg/gDW)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
    fig.suptitle("Peak doses versus neighbours", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "peak_comparison.png", dpi=300)
    plt.close(fig)


def plot_quadratic_curvature(summary: pd.DataFrame) -> None:
    metrics = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom total"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD total"),
    ]
    x = summary["p_uva_mw_cm2"].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=False)
    for col, (metric, title) in enumerate(metrics):
        y = summary[metric].values
        coeffs = np.polyfit(x, y, 2)
        xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
        fit = np.polyval(coeffs, xs)
        fitted_points = np.polyval(coeffs, x)
        curvature = coeffs[0]
        upper = axes[0, col]
        lower = axes[1, col]
        upper.plot(x, y, marker="o", linestyle="-", color="#1f77b4", label="Observed")
        upper.plot(xs, fit, linestyle="--", color="#d62728", label="Quadratic fit")
        upper.set_title(f"{title} quadratic fit")
        upper.set_xlabel("UVA (mW/cm^2)")
        upper.set_ylabel("Trimmed mean (mg/gDW)")
        upper.text(0.05, 0.9, f"a = {curvature:.3f}", transform=upper.transAxes)
        upper.legend()
        residuals = y - fitted_points
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in residuals]
        lower.bar(summary["dose_index"], residuals, color=colors, width=0.4)
        lower.axhline(0, color="grey", linewidth=1)
        lower.set_xlabel("Dose index")
        lower.set_ylabel("Residual (mg/gDW)")
        lower.set_title(f"{title} residuals")
        lower.set_xticks(summary["dose_index"])
        lower.set_xticklabels(summary["dose_id"])
    fig.suptitle("Quadratic curvature and residual diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quadratic_curvature.png", dpi=300)
    plt.close(fig)


def plot_trend_regressions(summary: pd.DataFrame, trend: pd.DataFrame) -> None:
    configs = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom total"),
        ("chrom_reduced_mg_per_gDW_trimmed_mean", "Chrom reduced"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD total"),
        ("dad_reduced_mg_per_gDW_trimmed_mean", "DAD reduced"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    x = summary["p_uva_mw_cm2"]
    for ax, (metric, title) in zip(axes.flatten(), configs, strict=False):
        base_metric = metric.replace("_trimmed_mean", "")
        row = trend[(trend["metric"] == base_metric) & (trend["axis"] == "UVA")]
        sns.scatterplot(x=x, y=summary[metric], ax=ax, s=50, color="#1f77b4")
        if not row.empty:
            slope = row["slope"].iloc[0]
            intercept = row["intercept"].iloc[0]
            pearson = row["pearson_r"].iloc[0]
            kendall = row["kendall_tau"].iloc[0]
            xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
            ax.plot(xs, intercept + slope * xs, color="#d62728", label=f"Slope {slope:.3f}")
            ax.text(
                0.05,
                0.08,
                f"Pearson r = {pearson:.2f}\nKendall tau = {kendall:.2f}",
                transform=ax.transAxes,
            )
        ax.set_title(f"{title} vs UVA")
        ax.set_xlabel("UVA (mW/cm^2)")
        ax.set_ylabel("Trimmed mean (mg/gDW)")
        ax.legend()
    fig.suptitle("Weighted regression summaries", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "uva_trend_regressions.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_environment()
    reps, summary, deltas, trend, alignment = load_data()
    summary = summary.sort_values("p_uva_mw_cm2").reset_index(drop=True)
    summary["dose_index"] = summary.index + 1

    plot_dose_trajectories(summary)
    plot_sequential_deltas(deltas)
    plot_replicates_vs_trimmed(reps, summary)
    plot_cross_assay_concordance(summary, alignment)
    plot_uv_context(summary)
    plot_reduced_vs_oxidized(summary)
    plot_peak_comparison(summary)
    plot_quadratic_curvature(summary)
    plot_trend_regressions(summary, trend)


if __name__ == "__main__":
    main()
