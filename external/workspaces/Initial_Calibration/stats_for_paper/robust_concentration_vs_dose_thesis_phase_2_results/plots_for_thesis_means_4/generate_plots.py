from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLORS: Dict[str, str] = {
    "total": "#1f77b4",
    "oxidized": "#ff7f0e",
    "reduced": "#2ca02c",
    "uva": "#4c72b0",
    "uvb": "#dd8452",
    "chrom_reps": "#2E86AB",
    "dad_reps": "#B23A48",
}

DATA_FILES: Dict[str, str] = {
    "reps": "Combined_Scytonemin_Concentrations.csv",
    "summary": "dose_level_summary.csv",
    "deltas": "dose_pattern_sequential_deltas.csv",
    "trend_stats": "dose_trend_stats.csv",
    "alignment": "chrom_dad_alignment.csv",
    "pattern_summary": "dose_pattern_summary.csv",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_datasets() -> Dict[str, pd.DataFrame]:
    root = project_root()
    return {name: pd.read_csv(root / rel_path) for name, rel_path in DATA_FILES.items()}


def configure_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 120,
        }
    )


def ensure_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def add_dose_index(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values("p_uva_mw_cm2").reset_index(drop=True)
    ordered["dose_index"] = np.arange(1, len(ordered) + 1)
    return ordered


def plot_dose_trajectories(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("total", "Total"), ("reduced", "Reduced"), ("oxidized", "Oxidized")]
    assays = [("Chromatogram", "chrom"), ("DAD", "dad")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (title, prefix) in zip(axes, assays):
        for suffix, label in metrics:
            mean_col = f"{prefix}_{suffix}_mg_per_gDW_trimmed_mean"
            low_col = f"{prefix}_{suffix}_mg_per_gDW_ci_low"
            high_col = f"{prefix}_{suffix}_mg_per_gDW_ci_high"
            sns.lineplot(
                data=summary,
                x="p_uva_mw_cm2",
                y=mean_col,
                marker="o",
                label=label,
                color=COLORS[suffix],
                ax=ax,
            )
            ax.fill_between(
                summary["p_uva_mw_cm2"],
                summary[low_col],
                summary[high_col],
                alpha=0.15,
                color=COLORS[suffix],
            )
        dose5 = summary.loc[summary["dose_id"] == "dose_5", "p_uva_mw_cm2"]
        if not dose5.empty:
            ax.axvline(dose5.iloc[0], linestyle="--", color="grey", linewidth=1)
        ax.set_title(f"{title} Trimmed Means ± 95% CI")
        ax.set_xlabel("UVA (mW·cm⁻²)")
        ax.set_ylabel("Concentration (mg·gDW⁻¹)")
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "dose_trajectories.png")
    plt.close(fig)


def plot_sequential_deltas(deltas: pd.DataFrame, out_dir: Path) -> None:
    focus_metrics = [
        "Chrom Total (mg·gDW⁻¹)",
        "Chrom Reduced",
        "DAD Total (mg·gDW⁻¹)",
        "DAD Reduced",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, focus_metrics):
        mdata = deltas[deltas["metric"] == metric].copy()
        mdata["start_dose"] = (
            mdata["transition"].str.extract(r"dose_(\d)", expand=False).astype(int)
        )
        mdata.sort_values("start_dose", inplace=True)
        positions = np.arange(len(mdata))
        bar_colors = [
            COLORS["reduced"] if val >= 0 else "#d62728"
            for val in mdata["delta_trimmed_mean"]
        ]
        ax.bar(positions, mdata["delta_trimmed_mean"], color=bar_colors, alpha=0.85)
        ax.errorbar(
            positions,
            mdata["delta_trimmed_mean"],
            yerr=[
                mdata["delta_trimmed_mean"] - mdata["delta_ci_low"],
                mdata["delta_ci_high"] - mdata["delta_trimmed_mean"],
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(mdata["transition"], rotation=40, ha="right")
        ax.set_ylabel("Δ Trimmed Mean (mg·gDW⁻¹)")
        ax.set_title(metric)

    fig.tight_layout()
    fig.savefig(out_dir / "sequential_deltas.png")
    plt.close(fig)


def plot_assay_concordance(
    summary: pd.DataFrame, alignment: pd.DataFrame, out_dir: Path
) -> None:
    metrics = [("total", "Totals"), ("oxidized", "Oxidized"), ("reduced", "Reduced")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (suffix, label) in zip(axes, metrics):
        x = summary[f"chrom_{suffix}_mg_per_gDW_trimmed_mean"]
        y = summary[f"dad_{suffix}_mg_per_gDW_trimmed_mean"]
        sns.scatterplot(x=x, y=y, ax=ax, color=COLORS[suffix], s=70)

        stats_row = alignment.loc[alignment["metric"] == suffix].iloc[0]
        pearson = stats_row["pearson_r"]
        slope = stats_row["deming_slope"]
        intercept = stats_row["deming_intercept"]
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        grid = np.linspace(min_val * 0.9, max_val * 1.1, 100)
        ax.plot(grid, grid, linestyle=":", color="grey", linewidth=1, label="1:1")
        ax.plot(
            grid,
            intercept + slope * grid,
            linestyle="--",
            color="black",
            linewidth=1.2,
            label=f"Deming slope {slope:.2f}",
        )
        ax.set_title(f"{label} — Chrom vs DAD (r={pearson:.3f})")
        ax.set_xlabel("Chrom trimmed mean (mg·gDW⁻¹)")
        ax.set_ylabel("DAD trimmed mean (mg·gDW⁻¹)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "assay_concordance.png")
    plt.close(fig)


def plot_uv_context(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax_uv = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=summary,
        x="dose_index",
        y="p_uva_mw_cm2",
        marker="o",
        label="UVA",
        color=COLORS["uva"],
        ax=ax_uv,
    )
    sns.lineplot(
        data=summary,
        x="dose_index",
        y="p_uvb_mw_cm2",
        marker="s",
        label="UVB",
        color=COLORS["uvb"],
        ax=ax_uv,
    )
    ax_uv.set_xlabel("Dose index (ordered by UVA)")
    ax_uv.set_ylabel("Irradiance (mW·cm⁻²)")
    ax_uv.set_xticks(summary["dose_index"])
    ax_uv.axvspan(5, 6, color="grey", alpha=0.1)

    ax_conc = ax_uv.twinx()
    conc_metrics = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom Total", COLORS["total"]),
        ("chrom_reduced_mg_per_gDW_trimmed_mean", "Chrom Reduced", COLORS["reduced"]),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD Total", "#8c564b"),
        ("dad_reduced_mg_per_gDW_trimmed_mean", "DAD Reduced", "#9467bd"),
    ]
    for col, label, color in conc_metrics:
        sns.lineplot(
            data=summary,
            x="dose_index",
            y=col,
            marker="o",
            label=label,
            color=color,
            ax=ax_conc,
        )
    ax_conc.set_ylabel("Trimmed mean (mg·gDW⁻¹)")

    handles1, labels1 = ax_uv.get_legend_handles_labels()
    handles2, labels2 = ax_conc.get_legend_handles_labels()
    if ax_uv.legend_:
        ax_uv.legend_.remove()
    if ax_conc.legend_:
        ax_conc.legend_.remove()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.95),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "uv_context.png")
    plt.close(fig)


def plot_replicates_vs_trimmed(
    reps: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    specs = [
        ("chrom_total_mg_per_gDW", "Chrom Totals", COLORS["chrom_reps"]),
        ("dad_total_mg_per_gDW", "DAD Totals", COLORS["dad_reps"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (metric, title, color) in zip(axes, specs):
        sns.stripplot(
            data=reps,
            x="dose_index",
            y=metric,
            ax=ax,
            color=color,
            jitter=0.25,
            alpha=0.65,
            size=6,
        )
        mean_col = f"{metric}_trimmed_mean"
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        ax.errorbar(
            summary["dose_index"],
            summary[mean_col],
            yerr=[
                summary[mean_col] - summary[low_col],
                summary[high_col] - summary[mean_col],
            ],
            fmt="o-",
            color="black",
            capsize=4,
            label="Trimmed mean ± CI",
        )
        ax.set_title(title)
        ax.set_xlabel("Dose index (ordered by UVA)")
        ax.set_ylabel("Concentration (mg·gDW⁻¹)")
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "replicates_vs_trimmed.png")
    plt.close(fig)


def plot_reduced_vs_oxidized(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    assays = [("Chromatogram", "chrom"), ("DAD", "dad")]

    for ax, (title, prefix) in zip(axes, assays):
        for suffix, label in [("reduced", "Reduced"), ("oxidized", "Oxidized")]:
            mean_col = f"{prefix}_{suffix}_mg_per_gDW_trimmed_mean"
            sns.lineplot(
                data=summary,
                x="dose_index",
                y=mean_col,
                marker="o",
                label=label,
                color=COLORS[suffix],
                ax=ax,
            )
        ax.set_xlabel("Dose index")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.set_title(f"{title} — Reduced vs Oxidized")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "reduced_vs_oxidized.png")
    plt.close(fig)


def compute_peak_windows(
    summary: pd.DataFrame,
    metric_cols: Iterable[Tuple[str, str]],
) -> pd.DataFrame:
    records = []
    for col, label in metric_cols:
        peak_idx = summary[col].idxmax()
        for idx in [peak_idx - 1, peak_idx, peak_idx + 1]:
            if idx < summary.index.min() or idx > summary.index.max():
                continue
            row = summary.loc[idx]
            records.append(
                {
                    "metric": label,
                    "dose_index": row["dose_index"],
                    "dose_id": row["dose_id"],
                    "value": row[col],
                }
            )
    return pd.DataFrame(records)


def plot_peak_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom Total"),
        ("chrom_reduced_mg_per_gDW_trimmed_mean", "Chrom Reduced"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD Total"),
        ("dad_reduced_mg_per_gDW_trimmed_mean", "DAD Reduced"),
    ]
    peak_df = compute_peak_windows(summary, metric_cols)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metric_cols):
        mdata = peak_df[peak_df["metric"] == label].sort_values("dose_index")
        ax.plot(
            mdata["dose_index"],
            mdata["value"],
            linestyle="--",
            marker="o",
            color=COLORS["total"],
        )
        ax.set_title(label)
        ax.set_xlabel("Dose index")
        ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")
        ax.set_xticks(mdata["dose_index"])
        ax.set_xticklabels(mdata["dose_id"], rotation=30)

    fig.tight_layout()
    fig.savefig(out_dir / "peak_comparison.png")
    plt.close(fig)


def plot_quadratic_curvature(
    summary: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    metrics = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom Total"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD Total"),
    ]
    stat_lookup = pattern_summary.set_index("metric")
    mapping = {
        "chrom_total_mg_per_gDW_trimmed_mean": "Chrom Total (mg·gDW⁻¹)",
        "dad_total_mg_per_gDW_trimmed_mean": "DAD Total (mg·gDW⁻¹)",
    }

    fig, axes = plt.subplots(2, len(metrics), figsize=(12, 8), sharex=True)
    uv = summary["p_uva_mw_cm2"].values

    for col_idx, (col, label) in enumerate(metrics):
        y = summary[col].values
        coeffs = np.polyfit(uv, y, 2)
        xs = np.linspace(uv.min(), uv.max(), 200)
        axes[0, col_idx].scatter(uv, y, color=COLORS["total"], s=70)
        axes[0, col_idx].plot(xs, np.polyval(coeffs, xs), color=COLORS["oxidized"], linewidth=2)
        quad_coef = stat_lookup.loc[mapping[col], "quadratic_coef"]
        axes[0, col_idx].set_title(f"{label} — quadratic β₂ = {quad_coef:.3f}")
        axes[0, col_idx].set_ylabel("Trimmed mean (mg·gDW⁻¹)")

        residuals = y - np.polyval(coeffs, uv)
        axes[1, col_idx].axhline(0, color="grey", linewidth=1)
        axes[1, col_idx].stem(
            uv,
            residuals,
            linefmt="grey",
            markerfmt="o",
            basefmt=" ",
        )
        axes[1, col_idx].set_xlabel("UVA (mW·cm⁻²)")
        axes[1, col_idx].set_ylabel("Residual (mg·gDW⁻¹)")

    fig.tight_layout()
    fig.savefig(out_dir / "quadratic_curvature.png")
    plt.close(fig)


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    X = np.column_stack((np.ones_like(x), x))
    w = np.sqrt(weights)
    WX = w[:, None] * X
    wy = w * y
    beta, _, _, _ = np.linalg.lstsq(WX, wy, rcond=None)
    intercept, slope = beta
    return intercept, slope


def plot_trend_regressions(
    summary: pd.DataFrame,
    trend_stats: pd.DataFrame,
    out_dir: Path,
) -> None:
    metrics = [
        ("chrom_total_mg_per_gDW_trimmed_mean", "Chrom Total"),
        ("chrom_reduced_mg_per_gDW_trimmed_mean", "Chrom Reduced"),
        ("dad_total_mg_per_gDW_trimmed_mean", "DAD Total"),
        ("dad_reduced_mg_per_gDW_trimmed_mean", "DAD Reduced"),
    ]
    axis_specs = [("UVA", "p_uva_mw_cm2"), ("UVB", "p_uvb_mw_cm2")]
    fig, axes = plt.subplots(len(metrics), len(axis_specs), figsize=(14, 16), sharey=False)

    for row_idx, (col, label) in enumerate(metrics):
        sd_col = col.replace("_trimmed_mean", "_trimmed_sd")
        sd = np.clip(summary[sd_col].values, 1e-6, None)
        weights = 1.0 / np.square(sd)
        base_metric = col.replace("_trimmed_mean", "")
        y = summary[col].values

        for col_idx, (axis_name, axis_col) in enumerate(axis_specs):
            x = summary[axis_col].values
            intercept, slope = weighted_linear_fit(x, y, weights)
            span = np.ptp(x)
            padding = 0.05 * span if span != 0 else 0.05
            xs = np.linspace(x.min() - padding, x.max() + padding, 100)
            ax = axes[row_idx, col_idx]
            sns.scatterplot(x=x, y=y, ax=ax, color=COLORS["total"], s=60)
            ax.plot(xs, intercept + slope * xs, color=COLORS["oxidized"], linewidth=2)
            stats_row = trend_stats[
                (trend_stats["metric"] == base_metric) & (trend_stats["axis"] == axis_name)
            ]
            if not stats_row.empty:
                stats_data = stats_row.iloc[0]
                ax.set_title(
                    f"{label} vs {axis_name} — slope {stats_data['slope']:.3f}, τ {stats_data['kendall_tau']:.2f}"
                )
            else:
                ax.set_title(f"{label} vs {axis_name}")
            ax.set_xlabel(f"{axis_name} (mW·cm⁻²)")
            ax.set_ylabel("Trimmed mean (mg·gDW⁻¹)")

    fig.tight_layout()
    fig.savefig(out_dir / "trend_regressions.png")
    plt.close(fig)


def main() -> None:
    configure_theme()
    out_dir = ensure_output_dir()
    data = load_datasets()
    summary = add_dose_index(data["summary"])
    reps = data["reps"].merge(
        summary[["p_uva_mw_cm2", "p_uvb_mw_cm2", "dose_id", "dose_index"]],
        on=["p_uva_mw_cm2", "p_uvb_mw_cm2"],
        how="left",
    )
    reps = reps.dropna(subset=["dose_index"]).copy()
    reps["dose_index"] = reps["dose_index"].astype(int)

    plot_dose_trajectories(summary, out_dir)
    plot_sequential_deltas(data["deltas"], out_dir)
    plot_assay_concordance(summary, data["alignment"], out_dir)
    plot_uv_context(summary, out_dir)
    plot_replicates_vs_trimmed(reps, summary, out_dir)
    plot_reduced_vs_oxidized(summary, out_dir)
    plot_peak_comparison(summary, out_dir)
    plot_quadratic_curvature(summary, data["pattern_summary"], out_dir)
    plot_trend_regressions(summary, data["trend_stats"], out_dir)


if __name__ == "__main__":
    main()
