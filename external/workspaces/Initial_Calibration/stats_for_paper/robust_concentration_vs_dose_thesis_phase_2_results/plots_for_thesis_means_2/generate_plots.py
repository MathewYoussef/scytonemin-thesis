"""
Plot generation script for thesis mean concentration figures.
Uses seaborn/matplotlib to create figures described in plot_plan.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "build"

ASSAY_LABELS = {"chrom": "Chromatogram", "dad": "DAD"}
POOL_LABELS = {"total": "Total", "reduced": "Reduced", "oxidized": "Oxidized"}
POOL_ORDER = ["Total", "Reduced", "Oxidized"]
POOL_COLORS = {
    "Total": "#1f77b4",
    "Reduced": "#d62728",
    "Oxidized": "#2ca02c",
}


@dataclass(frozen=True)
class DatasetBundle:
    dose_summary: pd.DataFrame
    sequential_deltas: pd.DataFrame
    trend_stats: pd.DataFrame
    chrom_dad_alignment: pd.DataFrame
    replicates: pd.DataFrame


def load_data() -> DatasetBundle:
    """Load all CSV inputs required for plot generation."""
    dose_summary = pd.read_csv(DATA_ROOT / "dose_level_summary.csv")
    sequential_deltas = pd.read_csv(DATA_ROOT / "dose_pattern_sequential_deltas.csv")
    trend_stats = pd.read_csv(DATA_ROOT / "dose_trend_stats.csv")
    chrom_dad_alignment = pd.read_csv(DATA_ROOT / "chrom_dad_alignment.csv")
    replicates = pd.read_csv(DATA_ROOT / "Combined_Scytonemin_Concentrations.csv")
    return DatasetBundle(
        dose_summary=dose_summary,
        sequential_deltas=sequential_deltas,
        trend_stats=trend_stats,
        chrom_dad_alignment=chrom_dad_alignment,
        replicates=replicates,
    )


def get_dose_order(dose_summary: pd.DataFrame) -> list[str]:
    """Return dose IDs sorted numerically."""
    return (
        dose_summary.copy()
        .assign(order=dose_summary["dose_id"].str.extract(r"(\d+)").astype(int))
        .sort_values("order")["dose_id"]
        .tolist()
    )


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
        }
    )


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    ensure_output_directory(OUTPUT_DIR)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def tidy_dose_summary(dose_summary: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form dataframe with assay/pool columns."""
    records = []
    for assay_key, assay_label in ASSAY_LABELS.items():
        for pool_key, pool_label in POOL_LABELS.items():
            base = f"{assay_key}_{pool_key}_mg_per_gDW"
            records.append(
                dose_summary[["dose_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]].assign(
                    assay=assay_label,
                    pool=pool_label,
                    mean=dose_summary[f"{base}_trimmed_mean"],
                    ci_low=dose_summary[f"{base}_ci_low"],
                    ci_high=dose_summary[f"{base}_ci_high"],
                )
            )
    return pd.concat(records, ignore_index=True)


def plot_dose_trajectories(data: DatasetBundle) -> None:
    """Line plots of trimmed means with CI ribbons for each assay."""
    long_df = tidy_dose_summary(data.dose_summary)
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(7.2, 6.0), constrained_layout=True
    )
    uva_col = "p_uva_mw_cm2"
    dose5_uva = (
        data.dose_summary.loc[data.dose_summary["dose_id"] == "dose_5", uva_col]
        .squeeze()
        .item()
    )
    for ax, (assay, subset) in zip(axes, long_df.groupby("assay", sort=False)):
        for pool in POOL_ORDER:
            pool_df = subset[subset["pool"] == pool]
            ax.plot(
                pool_df[uva_col],
                pool_df["mean"],
                label=pool,
                color=POOL_COLORS[pool],
                marker="o",
            )
            ax.fill_between(
                pool_df[uva_col],
                pool_df["ci_low"],
                pool_df["ci_high"],
                color=POOL_COLORS[pool],
                alpha=0.2,
            )
        ax.axvline(dose5_uva, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_ylabel("20% trimmed mean (mg·gDW⁻¹)")
        ax.set_title(f"{assay} pools vs UVA dose")
        ax.grid(True, axis="y", alpha=0.4)
    axes[-1].set_xlabel("UVA dose (mW·cm⁻²)")
    axes[0].legend(title="Pool", loc="upper left")
    fig.suptitle("Dose trajectories with 95% bootstrap CIs", y=1.02)
    save_figure(fig, "dose_trajectories.png")


def parse_metric_name(metric: str) -> tuple[str, str]:
    assay_token, remainder = metric.split(" ", 1)
    assay_label = "Chromatogram" if assay_token == "Chrom" else "DAD"
    pool_label = remainder.replace("(mg·gDW⁻¹)", "").strip()
    return assay_label, pool_label.capitalize()


def tidy_sequential_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, row in delta_df.iterrows():
        assay_label, pool_label = parse_metric_name(row["metric"])
        parts.append(
            {
                "assay": assay_label,
                "pool": pool_label,
                "transition": row["transition"],
                "delta": row["delta_trimmed_mean"],
                "ci_low": row["delta_ci_low"],
                "ci_high": row["delta_ci_high"],
            }
        )
    tidy = pd.DataFrame(parts)
    tidy["transition"] = pd.Categorical(
        tidy["transition"],
        categories=sorted(delta_df["transition"].unique()),
        ordered=True,
    )
    return tidy


def plot_sequential_deltas(data: DatasetBundle) -> None:
    tidy = tidy_sequential_deltas(data.sequential_deltas)
    assays = ["Chromatogram", "DAD"]
    pools = ["Total", "Reduced", "Oxidized"]
    fig, axes = plt.subplots(
        nrows=len(assays),
        ncols=len(pools),
        figsize=(10, 6),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    for i, assay in enumerate(assays):
        for j, pool in enumerate(pools):
            ax = axes[i, j]
            subset = tidy[(tidy["assay"] == assay) & (tidy["pool"] == pool)]
            if subset.empty:
                ax.set_visible(False)
                continue
            colors = [POOL_COLORS.get(pool, "#888888")] * len(subset)
            ax.bar(
                subset["transition"],
                subset["delta"],
                color=colors,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.errorbar(
                subset["transition"],
                subset["delta"],
                yerr=[
                    subset["delta"] - subset["ci_low"],
                    subset["ci_high"] - subset["delta"],
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=0.8,
            )
            ax.axhline(0, color="0.4", linewidth=0.8)
            if i == len(assays) - 1:
                ax.set_xlabel("Dose transition")
            if j == 0:
                ax.set_ylabel(f"{assay}\nΔ trimmed mean (mg·gDW⁻¹)")
            ax.set_title(f"{pool} pool")
            ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Sequential dose-to-dose trimmed-mean changes (95% CIs)", y=1.02)
    save_figure(fig, "sequential_deltas.png")


def plot_replicates_vs_trimmed(data: DatasetBundle) -> None:
    summary = data.dose_summary[
        [
            "dose_id",
            "p_uva_mw_cm2",
            "chrom_total_mg_per_gDW_trimmed_mean",
            "chrom_total_mg_per_gDW_ci_low",
            "chrom_total_mg_per_gDW_ci_high",
            "dad_total_mg_per_gDW_trimmed_mean",
            "dad_total_mg_per_gDW_ci_low",
            "dad_total_mg_per_gDW_ci_high",
        ]
    ].copy()

    replicates = data.replicates.copy()
    merge_cols = ["p_uva_mw_cm2", "p_uvb_mw_cm2"]
    replicates = replicates.merge(
        data.dose_summary[["dose_id"] + merge_cols], on=merge_cols, how="left"
    )
    order = get_dose_order(data.dose_summary)

    figure_specs = [
        {
            "assay": "Chromatogram",
            "rep_column": "chrom_total_mg_per_gDW",
            "mean_column": "chrom_total_mg_per_gDW_trimmed_mean",
            "ci_low_column": "chrom_total_mg_per_gDW_ci_low",
            "ci_high_column": "chrom_total_mg_per_gDW_ci_high",
        },
        {
            "assay": "DAD",
            "rep_column": "dad_total_mg_per_gDW",
            "mean_column": "dad_total_mg_per_gDW_trimmed_mean",
            "ci_low_column": "dad_total_mg_per_gDW_ci_low",
            "ci_high_column": "dad_total_mg_per_gDW_ci_high",
        },
    ]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4.5), constrained_layout=True)
    for ax, spec in zip(axes, figure_specs):
        assay = spec["assay"]
        reps = replicates.rename(columns={spec["rep_column"]: "value"})
        sns.stripplot(
            data=reps,
            x="dose_id",
            y="value",
            order=order,
            ax=ax,
            jitter=0.2,
            color="#888888",
            alpha=0.7,
            size=4,
        )
        mean_df = summary.rename(
            columns={
                spec["mean_column"]: "mean",
                spec["ci_low_column"]: "ci_low",
                spec["ci_high_column"]: "ci_high",
            }
        )
        mean_df = mean_df.set_index("dose_id").loc[order].reset_index()
        ax.errorbar(
            mean_df["dose_id"],
            mean_df["mean"],
            yerr=[
                mean_df["mean"] - mean_df["ci_low"],
                mean_df["ci_high"] - mean_df["mean"],
            ],
            fmt="o",
            color=POOL_COLORS["Total"],
            ecolor=POOL_COLORS["Total"],
            elinewidth=1.2,
            capsize=4,
            label="20% trimmed mean ± 95% CI",
        )
        ax.set_title(f"{assay} totals")
        ax.set_xlabel("Dose")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Concentration (mg·gDW⁻¹)")
    axes[1].legend(loc="upper right")
    fig.suptitle("Replicates vs trimmed means (totals)", y=1.02)
    save_figure(fig, "replicates_vs_trimmed.png")


def prepare_assay_pair_data(dose_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a tidy dataframe with Chromatogram vs DAD trimmed means by pool."""
    frames = []
    for pool_key, pool_label in POOL_LABELS.items():
        chrom_col = f"chrom_{pool_key}_mg_per_gDW_trimmed_mean"
        dad_col = f"dad_{pool_key}_mg_per_gDW_trimmed_mean"
        frames.append(
            dose_summary[["dose_id", "p_uva_mw_cm2", chrom_col, dad_col]]
            .rename(
                columns={
                    chrom_col: "Chromatogram",
                    dad_col: "DAD",
                }
            )
            .assign(pool=pool_label)
        )
    return pd.concat(frames, ignore_index=True)


def plot_cross_assay_concordance(data: DatasetBundle) -> None:
    pair_df = prepare_assay_pair_data(data.dose_summary)
    alignment = (
        data.chrom_dad_alignment.set_index("metric")
        if not data.chrom_dad_alignment.empty
        else pd.DataFrame()
    )
    pool_metric_map = {"Total": "total", "Reduced": "reduced", "Oxidized": "oxidized"}

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), sharex=False, sharey=False, constrained_layout=True
    )

    for ax, pool in zip(axes, ["Total", "Reduced", "Oxidized"]):
        subset = pair_df[pair_df["pool"] == pool]
        ax.scatter(
            subset["Chromatogram"],
            subset["DAD"],
            color=POOL_COLORS.get(pool, "#333333"),
            s=45,
        )
        lims = [
            min(subset["Chromatogram"].min(), subset["DAD"].min()),
            max(subset["Chromatogram"].max(), subset["DAD"].max()),
        ]
        buffer = (lims[1] - lims[0]) * 0.08 if lims[1] != lims[0] else 0.1
        x_min, x_max = lims[0] - buffer, lims[1] + buffer
        ax.plot(
            [x_min, x_max],
            [x_min, x_max],
            color="0.5",
            linestyle="--",
            linewidth=1.0,
            label="1:1 line" if ax is axes[0] else None,
        )
        metric_key = pool_metric_map[pool]
        if not alignment.empty and metric_key in alignment.index:
            row = alignment.loc[metric_key]
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = row["deming_intercept"] + row["deming_slope"] * x_vals
            ax.plot(
                x_vals,
                y_vals,
                color=POOL_COLORS.get(pool, "#333333"),
                linewidth=1.3,
                label="Deming fit" if ax is axes[0] else None,
            )
            ax.text(
                0.04,
                0.92,
                f"slope {row['deming_slope']:.2f}\nr {row['pearson_r']:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        for _, point in subset.iterrows():
            ax.annotate(
                point["dose_id"].replace("dose_", "d"),
                (point["Chromatogram"], point["DAD"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="0.35",
            )
        ax.set_title(f"{pool} pools")
        ax.set_xlabel("Chromatogram trimmed mean (mg·gDW⁻¹)")
        if ax is axes[0]:
            ax.set_ylabel("DAD trimmed mean (mg·gDW⁻¹)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
    axes[0].legend(loc="upper left")
    fig.suptitle("Chromatogram vs DAD trimmed means (Deming regression)", y=1.02)
    save_figure(fig, "cross_assay_concordance.png")


def plot_uv_regime_context(data: DatasetBundle) -> None:
    summary = data.dose_summary.copy()
    order = get_dose_order(summary)
    summary = summary.set_index("dose_id").loc[order].reset_index()
    x = np.arange(len(order))

    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(
        x,
        summary["p_uva_mw_cm2"],
        marker="o",
        color="#9467bd",
        label="UVA",
    )
    ax1.plot(
        x,
        summary["p_uvb_mw_cm2"],
        marker="o",
        color="#8c564b",
        linestyle="--",
        label="UVB",
    )
    ax1.set_ylabel("Irradiance (mW·cm⁻²)")
    ax1.set_xticks(x, labels=order)
    ax1.set_xlabel("Dose ID")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    chrom_total = summary["chrom_total_mg_per_gDW_trimmed_mean"]
    chrom_reduced = summary["chrom_reduced_mg_per_gDW_trimmed_mean"]
    ax2.plot(
        x,
        chrom_total,
        marker="s",
        color=POOL_COLORS["Total"],
        label="Chrom Total",
    )
    ax2.plot(
        x,
        chrom_reduced,
        marker="^",
        color=POOL_COLORS["Reduced"],
        label="Chrom Reduced",
    )
    ax2.set_ylabel("Trimmed mean concentration (mg·gDW⁻¹)")
    ax2.legend(loc="upper right")

    dose5_idx = order.index("dose_5")
    ax1.axvspan(dose5_idx - 0.25, dose5_idx + 0.75, color="grey", alpha=0.15)
    ax2.axvline(dose5_idx, color="0.4", linestyle="--", linewidth=1.0)
    target_idx = min(dose5_idx + 1, len(order) - 1)
    ax2.annotate(
        "UVB dips\nwhile UVA rises",
        xy=(dose5_idx + 0.4, chrom_total.iloc[target_idx]),
        xytext=(dose5_idx + 0.6, chrom_total.max() + 0.15),
        arrowprops=dict(arrowstyle="->", color="0.3"),
        ha="center",
    )
    fig.suptitle("UV regime vs Chromatogram totals/reduced pools", y=1.02)
    save_figure(fig, "uv_regime_context.png")


def plot_reduced_vs_oxidized_panels(data: DatasetBundle) -> None:
    summary = data.dose_summary.copy()
    order = get_dose_order(summary)
    summary = summary.set_index("dose_id").loc[order].reset_index()
    x = summary["p_uva_mw_cm2"]
    dose5_uva = summary.loc[summary["dose_id"] == "dose_5", "p_uva_mw_cm2"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, assay_key in zip(axes, ["chrom", "dad"]):
        assay_label = ASSAY_LABELS[assay_key]
        reduced = summary[f"{assay_key}_reduced_mg_per_gDW_trimmed_mean"]
        oxidized = summary[f"{assay_key}_oxidized_mg_per_gDW_trimmed_mean"]
        ax.plot(
            x,
            reduced,
            marker="o",
            color=POOL_COLORS["Reduced"],
            label="Reduced",
        )
        ax.plot(
            x,
            oxidized,
            marker="o",
            color=POOL_COLORS["Oxidized"],
            label="Oxidized",
        )
        ax.fill_between(
            x,
            oxidized,
            reduced,
            where=reduced >= oxidized,
            color=POOL_COLORS["Reduced"],
            alpha=0.1,
        )
        ax.set_title(f"{assay_label} reduced vs oxidized")
        ax.set_xlabel("UVA dose (mW·cm⁻²)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axvline(dose5_uva, color="0.4", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Trimmed mean (mg·gDW⁻¹)")
    axes[1].legend(loc="upper left")
    fig.suptitle("Reduced dominance over oxidized pools", y=1.02)
    save_figure(fig, "reduced_vs_oxidized.png")


def plot_peak_comparison(data: DatasetBundle) -> None:
    summary = data.dose_summary.copy()
    order = get_dose_order(summary)
    summary = summary.set_index("dose_id").loc[order]
    combos = [
        ("chrom", "total", "Chrom Total", "Total"),
        ("chrom", "reduced", "Chrom Reduced", "Reduced"),
        ("dad", "total", "DAD Total", "Total"),
        ("dad", "reduced", "DAD Reduced", "Reduced"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    axes = axes.flatten()
    for ax, (assay_key, pool_key, title, pool_label) in zip(axes, combos):
        series = summary[f"{assay_key}_{pool_key}_mg_per_gDW_trimmed_mean"]
        peak_dose = series.idxmax()
        peak_pos = order.index(peak_dose)
        neighbor_positions = [idx for idx in [peak_pos - 1, peak_pos, peak_pos + 1] if 0 <= idx < len(order)]
        neighbor_doses = [order[idx] for idx in neighbor_positions]
        subset = series.loc[neighbor_doses]
        x = np.arange(len(subset))
        color = POOL_COLORS[pool_label]
        ax.plot(x, subset.values, marker="o", color=color)
        peak_mask = subset.index == peak_dose
        ax.scatter(
            x[peak_mask],
            subset[peak_mask],
            s=70,
            color=color,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.set_xticks(x, labels=neighbor_doses)
        ax.set_title(f"{title} peak window")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.grid(True, axis="y", alpha=0.3)
        if len(subset) > 1:
            delta = subset.iloc[-1] - subset.iloc[-2]
            label = f"Δ to next: {delta:+.2f}"
        else:
            label = "Δ to next: n/a"
        ax.annotate(
            label,
            xy=(0.65, 0.15),
            xycoords="axes fraction",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    fig.suptitle("Peak comparison windows around late-dose softening", y=1.02)
    save_figure(fig, "peak_comparison.png")


def plot_quadratic_fit_diagnostics(data: DatasetBundle) -> None:
    summary = data.dose_summary.copy()
    order = get_dose_order(summary)
    summary = summary.set_index("dose_id").loc[order].reset_index()
    uva = summary["p_uva_mw_cm2"].values
    combos = [
        ("chrom", "total", "Chrom Total"),
        ("chrom", "reduced", "Chrom Reduced"),
        ("dad", "total", "DAD Total"),
        ("dad", "reduced", "DAD Reduced"),
    ]

    dose5_uva = summary.loc[summary["dose_id"] == "dose_5", "p_uva_mw_cm2"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (assay_key, pool_key, title) in zip(axes.flatten(), combos):
        values = summary[f"{assay_key}_{pool_key}_mg_per_gDW_trimmed_mean"].values
        coeffs = np.polyfit(uva, values, deg=2)
        grid = np.linspace(uva.min(), uva.max(), 200)
        fitted = np.polyval(coeffs, grid)
        ax.scatter(
            uva,
            values,
            color=POOL_COLORS[POOL_LABELS[pool_key]],
            label="Observed",
        )
        ax.plot(
            grid,
            fitted,
            color="black",
            linewidth=1.2,
            label="Quadratic fit",
        )
        curvature = coeffs[0]
        concavity = "down" if curvature < 0 else "up"
        ax.text(
            0.04,
            0.92,
            f"a={curvature:.3f} (concave {concavity})",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set_title(title)
        ax.set_xlabel("UVA dose (mW·cm⁻²)")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.grid(True, alpha=0.3)
        ax.axvline(dose5_uva, color="0.4", linestyle="--", linewidth=1.0)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Quadratic curvature diagnostics (observed vs fitted)", y=1.02)
    save_figure(fig, "quadratic_fit_diagnostics.png")


def plot_trend_regressions(data: DatasetBundle) -> None:
    summary = data.dose_summary.copy()
    order = get_dose_order(summary)
    summary = summary.set_index("dose_id").loc[order].reset_index()
    uva = summary["p_uva_mw_cm2"].values
    dose5_uva = summary.loc[summary["dose_id"] == "dose_5", "p_uva_mw_cm2"].iloc[0]
    trend = data.trend_stats.set_index(["metric", "axis"])
    combos = [
        {
            "metric": "chrom_total_mg_per_gDW",
            "column": "chrom_total_mg_per_gDW_trimmed_mean",
            "label": "Chrom Total",
            "pool": "Total",
        },
        {
            "metric": "chrom_reduced_mg_per_gDW",
            "column": "chrom_reduced_mg_per_gDW_trimmed_mean",
            "label": "Chrom Reduced",
            "pool": "Reduced",
        },
        {
            "metric": "dad_total_mg_per_gDW",
            "column": "dad_total_mg_per_gDW_trimmed_mean",
            "label": "DAD Total",
            "pool": "Total",
        },
        {
            "metric": "dad_reduced_mg_per_gDW",
            "column": "dad_reduced_mg_per_gDW_trimmed_mean",
            "label": "DAD Reduced",
            "pool": "Reduced",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, spec in zip(axes.flatten(), combos):
        values = summary[spec["column"]].values
        color = POOL_COLORS.get(spec["pool"], "#1f77b4")
        ax.scatter(uva, values, color=color)
        if (spec["metric"], "UVA") in trend.index:
            row = trend.loc[(spec["metric"], "UVA")]
            line_grid = np.linspace(uva.min(), uva.max(), 100)
            fit = row["intercept"] + row["slope"] * line_grid
            ax.plot(
                line_grid,
                fit,
                color="black",
                linewidth=1.2,
            )
            ax.text(
                0.04,
                0.90,
                f"slope {row['slope']:.3f}\nR² {row['r_squared']:.2f}\nr {row['pearson_r']:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        ax.axvline(dose5_uva, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_title(spec["label"])
        ax.set_xlabel("UVA dose (mW·cm⁻²)")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("UVA trend regressions with slope annotations", y=1.02)
    save_figure(fig, "uva_trend_regressions.png")


def main() -> None:
    configure_style()
    data = load_data()
    plot_dose_trajectories(data)
    plot_sequential_deltas(data)
    plot_replicates_vs_trimmed(data)
    plot_cross_assay_concordance(data)
    plot_uv_regime_context(data)
    plot_reduced_vs_oxidized_panels(data)
    plot_peak_comparison(data)
    plot_quadratic_fit_diagnostics(data)
    plot_trend_regressions(data)
    print(f"Saved figures to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
