from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DOSE_ORDER = [f"dose_{i}" for i in range(1, 7)]


@dataclass(frozen=True)
class TrajectoryMetric:
    assay: str
    pool: str
    prefix: str

    @property
    def mean_col(self) -> str:
        return f"{self.prefix}_trimmed_mean"

    @property
    def ci_low_col(self) -> str:
        return f"{self.prefix}_ci_low"

    @property
    def ci_high_col(self) -> str:
        return f"{self.prefix}_ci_high"

    @property
    def label(self) -> str:
        return f"{self.assay} {self.pool}"


TRAJECTORY_METRICS: List[TrajectoryMetric] = [
    TrajectoryMetric("Chrom", "Total", "chrom_total_mg_per_gDW"),
    TrajectoryMetric("Chrom", "Reduced", "chrom_reduced_mg_per_gDW"),
    TrajectoryMetric("Chrom", "Oxidized", "chrom_oxidized_mg_per_gDW"),
    TrajectoryMetric("DAD", "Total", "dad_total_mg_per_gDW"),
    TrajectoryMetric("DAD", "Reduced", "dad_reduced_mg_per_gDW"),
    TrajectoryMetric("DAD", "Oxidized", "dad_oxidized_mg_per_gDW"),
]


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    tables["dose_summary"] = pd.read_csv(PROJECT_ROOT / "dose_level_summary.csv")
    tables["deltas"] = pd.read_csv(PROJECT_ROOT / "dose_pattern_sequential_deltas.csv")
    tables["replicates"] = pd.read_csv(PROJECT_ROOT / "Combined_Scytonemin_Concentrations.csv")
    tables["alignment"] = pd.read_csv(PROJECT_ROOT / "chrom_dad_alignment.csv")
    tables["trend_stats"] = pd.read_csv(PROJECT_ROOT / "dose_trend_stats.csv")
    return tables


def sort_doses(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["dose_num"] = ordered["dose_id"].str.extract(r"(\d+)").astype(int)
    ordered = ordered.sort_values("dose_num").drop(columns="dose_num")
    ordered = ordered.reset_index(drop=True)
    return ordered


def prepare_dose_long(dose_df: pd.DataFrame, metrics: Iterable[TrajectoryMetric]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric in metrics:
        for _, row in dose_df.iterrows():
            records.append(
                {
                    "dose_id": row["dose_id"],
                    "p_uva_mw_cm2": row["p_uva_mw_cm2"],
                    "p_uvb_mw_cm2": row["p_uvb_mw_cm2"],
                    "assay": metric.assay,
                    "pool": metric.pool,
                    "label": metric.label,
                    "mean": row[metric.mean_col],
                    "ci_low": row[metric.ci_low_col],
                    "ci_high": row[metric.ci_high_col],
                }
            )
    long_df = pd.DataFrame.from_records(records)
    long_df["label"] = pd.Categorical(long_df["label"], [m.label for m in metrics])
    return long_df


def plot_dose_trajectories(dose_df: pd.DataFrame) -> Path:
    tidy = prepare_dose_long(dose_df, TRAJECTORY_METRICS)
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11), sharex=True)
    axes = axes.flatten()

    for ax, (label, group) in zip(axes, tidy.groupby("label", observed=True)):
        ax.plot(group["p_uva_mw_cm2"], group["mean"], marker="o", color="#1f77b4")
        ax.fill_between(
            group["p_uva_mw_cm2"],
            group["ci_low"],
            group["ci_high"],
            color="#1f77b4",
            alpha=0.25,
        )
        ax.set_title(label)
        ax.set_ylabel("mg per gDW")
        ax.axhline(0, color="grey", linewidth=0.5)
        for uva, mean, dose in zip(group["p_uva_mw_cm2"], group["mean"], group["dose_id"]):
            ax.annotate(
                dose.replace("dose_", "d"),
                xy=(uva, mean),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    axes[-1].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    axes[-2].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    for ax in axes:
        ax.set_xlim(dose_df["p_uva_mw_cm2"].min(), dose_df["p_uva_mw_cm2"].max())
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)

    fig.suptitle("Dose Trajectories of Trimmed Means with 95% CIs", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "dose_trajectories.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sequential_deltas(delta_df: pd.DataFrame) -> Path:
    metrics = delta_df["metric"].unique()
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=False)
    axes = axes.flatten()
    cmap = {True: "#2ca02c", False: "#d62728"}

    for ax, metric in zip(axes, metrics):
        data = delta_df[delta_df["metric"] == metric].copy()
        data["transition_label"] = data["transition"].str.replace("dose_", "d", regex=False)
        positions = np.arange(len(data))
        colors = [cmap[val > 0] for val in data["delta_trimmed_mean"]]
        ax.bar(positions, data["delta_trimmed_mean"], color=colors)
        ax.vlines(
            positions,
            data["delta_ci_low"],
            data["delta_ci_high"],
            colors="black",
            linewidth=1.2,
        )
        ax.scatter(
            positions,
            data["delta_ci_low"],
            color="black",
            s=20,
            zorder=3,
        )
        ax.scatter(
            positions,
            data["delta_ci_high"],
            color="black",
            s=20,
            zorder=3,
        )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(metric.replace("⁻¹", "^-1"))
        ax.set_ylabel("Δ trimmed mean")
        ax.set_xticks(positions)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
        ax.set_xticklabels(data["transition_label"], rotation=45)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Sequential Change in Trimmed Means (95% CI whiskers)", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "sequential_deltas.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def get_metric(assay: str, pool: str) -> TrajectoryMetric:
    for metric in TRAJECTORY_METRICS:
        if metric.assay == assay and metric.pool == pool:
            return metric
    raise KeyError(f"No metric for {assay} {pool}")


def plot_cross_assay_concordance(dose_df: pd.DataFrame, alignment_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    metric_map = {
        "total": (get_metric("Chrom", "Total"), get_metric("DAD", "Total"), "Total pool"),
        "reduced": (get_metric("Chrom", "Reduced"), get_metric("DAD", "Reduced"), "Reduced pool"),
        "oxidized": (get_metric("Chrom", "Oxidized"), get_metric("DAD", "Oxidized"), "Oxidized pool"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes = axes.flatten()

    for idx, (metric_key, (chrom_metric, dad_metric, title)) in enumerate(metric_map.items()):
        ax = axes[idx]
        x = ordered[chrom_metric.mean_col]
        y = ordered[dad_metric.mean_col]
        ax.scatter(x, y, color="#1f77b4", s=70, label="dose means")
        for xi, yi, dose in zip(x, y, ordered["dose_id"]):
            ax.annotate(
                dose.replace("dose_", "d"),
                xy=(xi, yi),
                xytext=(6, -6),
                textcoords="offset points",
                fontsize=9,
            )
        stats_row = alignment_df[alignment_df["metric"] == metric_key].iloc[0]
        slope = stats_row["deming_slope"]
        intercept = stats_row["deming_intercept"]
        min_val = float(min(x.min(), y.min()))
        max_val = float(max(x.max(), y.max()))
        span = max(1e-6, max_val - min_val)
        lower = min_val - 0.1 * span
        upper = max_val + 0.1 * span
        line_x = np.array([lower, upper])
        ax.plot(line_x, line_x, linestyle="--", color="grey", linewidth=1, label="1:1 reference")
        ax.plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=1.5, label="Deming fit")
        note = (
            f"Pearson r = {stats_row['pearson_r']:.2f}\n"
            f"Deming slope = {slope:.2f}"
        )
        ax.text(
            0.05,
            0.95,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_title(title)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Chrom trimmed mean (mg per gDW)")
        if idx == 0:
            ax.set_ylabel("DAD trimmed mean (mg per gDW)")
        ax.grid(True, linewidth=0.5, alpha=0.4)
        if idx == 0:
            ax.legend(loc="lower right")

    fig.suptitle("Chrom vs DAD trimmed means with Deming regression", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "cross_assay_concordance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_uv_regime_context(dose_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    x = np.arange(len(ordered))
    labels = ordered["dose_id"].str.replace("dose_", "d")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(x, ordered["p_uva_mw_cm2"], marker="o", color="#1f77b4", label="UVA")
    ax1.plot(x, ordered["p_uvb_mw_cm2"], marker="s", color="#ff7f0e", label="UVB")
    ax1.set_xlabel("Dose")
    ax1.set_ylabel("Irradiance (mW cm$^{-2}$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y", linewidth=0.5, alpha=0.4)
    if len(x) >= 2:
        ax1.axvspan(x[-2] + 0.5, x[-1] + 0.5, color="grey", alpha=0.1)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        ordered["dad_total_mg_per_gDW_trimmed_mean"],
        marker="o",
        color="#2ca02c",
        label="DAD total",
    )
    ax2.plot(
        x,
        ordered["dad_reduced_mg_per_gDW_trimmed_mean"],
        marker="o",
        color="#9467bd",
        label="DAD reduced",
    )
    ax2.set_ylabel("Trimmed mean (mg per gDW)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.suptitle("UV regime and concentration trajectories", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "uv_regime_context.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reduced_vs_oxidized_panels(dose_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    uva = ordered["p_uva_mw_cm2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    assays = ["Chrom", "DAD"]

    for ax, assay in zip(axes, assays):
        reduced_metric = get_metric(assay, "Reduced")
        oxidized_metric = get_metric(assay, "Oxidized")
        ax.plot(
            uva,
            ordered[reduced_metric.mean_col],
            marker="o",
            color="#2ca02c",
            label="Reduced",
        )
        ax.fill_between(
            uva,
            ordered[reduced_metric.ci_low_col],
            ordered[reduced_metric.ci_high_col],
            color="#2ca02c",
            alpha=0.2,
        )
        ax.plot(
            uva,
            ordered[oxidized_metric.mean_col],
            marker="o",
            color="#d62728",
            label="Oxidized",
        )
        ax.fill_between(
            uva,
            ordered[oxidized_metric.ci_low_col],
            ordered[oxidized_metric.ci_high_col],
            color="#d62728",
            alpha=0.2,
        )
        for x_val, y_val, dose in zip(uva, ordered[reduced_metric.mean_col], ordered["dose_id"]):
            ax.annotate(
                dose.replace("dose_", "d"),
                xy=(x_val, y_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(f"{assay} pools vs UVA")
        ax.set_xlabel("UVA irradiance (mW cm$^{-2}$)")
        if assay == "Chrom":
            ax.set_ylabel("Trimmed mean (mg per gDW)")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
        ax.legend(loc="upper left")

    fig.suptitle("Reduced and oxidized pools by assay", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "reduced_vs_oxidized.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_replicates_vs_trimmed_means(
    dose_df: pd.DataFrame, replicates_df: pd.DataFrame
) -> Path:
    uva_uvb_to_dose = {
        (row.p_uva_mw_cm2, row.p_uvb_mw_cm2): row.dose_id for row in dose_df.itertuples()
    }
    replicates_df = replicates_df.copy()
    replicates_df["dose_id"] = replicates_df.apply(
        lambda row: uva_uvb_to_dose.get((row.p_uva_mw_cm2, row.p_uvb_mw_cm2)), axis=1
    )
    if replicates_df["dose_id"].isna().any():
        missing = replicates_df[replicates_df["dose_id"].isna()][
            ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
        ]
        raise ValueError(f"Unmapped dose rows:\n{missing}")

    tidy_records = []
    replicate_metrics = [
        ("Chrom", "Total", "chrom_total_mg_per_gDW"),
        ("Chrom", "Reduced", "chrom_reduced_mg_per_gDW"),
        ("DAD", "Total", "dad_total_mg_per_gDW"),
        ("DAD", "Reduced", "dad_reduced_mg_per_gDW"),
    ]
    for assay, pool, column in replicate_metrics:
        for row in replicates_df.itertuples():
            tidy_records.append(
                {
                    "assay": assay,
                    "pool": pool,
                    "label": f"{assay} {pool}",
                    "dose_id": row.dose_id,
                    "value": getattr(row, column),
                }
            )
    replicate_tidy = pd.DataFrame.from_records(tidy_records)
    replicate_tidy["label"] = pd.Categorical(
        replicate_tidy["label"], [f"{a} {p}" for a, p, _ in replicate_metrics]
    )

    trimmed_lookup = prepare_dose_long(
        dose_df,
        [m for m in TRAJECTORY_METRICS if m.label in replicate_tidy["label"].unique()],
    )
    order = DOSE_ORDER

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, label in zip(axes, replicate_tidy["label"].cat.categories):
        subset = replicate_tidy[replicate_tidy["label"] == label]
        sns.stripplot(
            data=subset,
            x="dose_id",
            y="value",
            order=order,
            ax=ax,
            jitter=0.2,
            color="#9467bd",
            alpha=0.7,
        )
        trimmed = trimmed_lookup[trimmed_lookup["label"] == label].set_index("dose_id").reindex(order)
        ax.errorbar(
            x=np.arange(len(order)),
            y=trimmed["mean"],
            yerr=[
                trimmed["mean"] - trimmed["ci_low"],
                trimmed["ci_high"] - trimmed["mean"],
            ],
            fmt="-o",
            color="#1f77b4",
            capsize=4,
            linewidth=2,
        )
        ax.set_title(label)
        ax.set_xlabel("Dose")
        ax.set_ylabel("mg per gDW")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"d{i}" for i in range(1, 7)])
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)

    fig.suptitle("Replicate Spread vs Trimmed Means (95% CIs)", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "replicates_vs_trimmed_means.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

def plot_peak_comparison_panels(dose_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    selected = [
        get_metric("Chrom", "Total"),
        get_metric("Chrom", "Reduced"),
        get_metric("DAD", "Total"),
        get_metric("DAD", "Reduced"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, selected):
        series = ordered[metric.mean_col].to_numpy()
        peak_idx = int(np.argmax(series))
        window_indices = [i for i in [peak_idx - 1, peak_idx, peak_idx + 1] if 0 <= i < len(ordered)]
        window = ordered.iloc[window_indices]
        x = np.arange(len(window))
        bars = window[metric.mean_col]
        colors = ["#1f77b4" if idx == peak_idx else "#7f7f7f" for idx in window_indices]
        ax.bar(x, bars, color=colors)
        err_low = bars - window[metric.ci_low_col]
        err_high = window[metric.ci_high_col] - bars
        ax.errorbar(x, bars, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4, linewidth=1)
        labels = window["dose_id"].str.replace("dose_", "d")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Trimmed mean (mg per gDW)")
        ax.set_title(f"{metric.label} peak window")
        if peak_idx + 1 < len(ordered):
            drop = series[peak_idx] - series[min(len(series) - 1, peak_idx + 1)]
            ax.text(
                0.05,
                0.9,
                f"Peak drop to next dose = {drop:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)

    fig.suptitle("Peak-dose comparisons with adjacent doses", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "peak_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_quadratic_fit_curvature(dose_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    uva = ordered["p_uva_mw_cm2"].to_numpy()
    dense_uva = np.linspace(uva.min(), uva.max(), 200)
    selected = [
        get_metric("Chrom", "Total"),
        get_metric("Chrom", "Reduced"),
        get_metric("DAD", "Total"),
        get_metric("DAD", "Reduced"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, selected):
        y = ordered[metric.mean_col].to_numpy()
        coeffs = np.polyfit(uva, y, 2)
        fit_line = np.polyval(coeffs, dense_uva)
        curvature = 2 * coeffs[0]
        ax.scatter(uva, y, color="#1f77b4", s=60)
        ax.plot(dense_uva, fit_line, color="#d62728", linewidth=1.5)
        for x_val, y_val, dose in zip(uva, y, ordered["dose_id"]):
            ax.annotate(
                dose.replace("dose_", "d"),
                xy=(x_val, y_val),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.text(
            0.05,
            0.9,
            f"Quadratic curvature = {curvature:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_title(f"{metric.label} vs UVA")
        ax.set_ylabel("Trimmed mean (mg per gDW)")
        ax.grid(True, linewidth=0.5, alpha=0.4)

    axes[-1].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    axes[-2].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    fig.suptitle("Quadratic fits highlight concave-down behavior", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "quadratic_fit_curvature.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    sqrt_w = np.sqrt(weights)
    X = np.vstack([x, np.ones_like(x)]).T
    X_w = X * sqrt_w[:, None]
    y_w = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
    slope, intercept = beta
    return float(slope), float(intercept)


def plot_uva_trend_regressions(dose_df: pd.DataFrame, trend_df: pd.DataFrame) -> Path:
    ordered = sort_doses(dose_df)
    uva = ordered["p_uva_mw_cm2"].to_numpy()
    selected = [
        get_metric("Chrom", "Total"),
        get_metric("Chrom", "Reduced"),
        get_metric("DAD", "Total"),
        get_metric("DAD", "Reduced"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, selected):
        y = ordered[metric.mean_col].to_numpy()
        sd = ordered[f"{metric.prefix}_trimmed_sd"].to_numpy()
        weights = np.where(sd > 0, 1 / (sd ** 2), 0.0)
        slope, intercept = weighted_linear_fit(uva, y, weights)
        line_x = np.linspace(uva.min(), uva.max(), 100)
        ax.scatter(uva, y, color="#1f77b4", s=60)
        ax.plot(line_x, slope * line_x + intercept, color="#ff7f0e", linewidth=1.5, label="WLS fit")
        stats_row = trend_df[(trend_df["metric"] == metric.prefix) & (trend_df["axis"] == "UVA")].iloc[0]
        note = (
            f"Slope = {stats_row['slope']:.3f} (CI {stats_row['slope_ci_low']:.3f} to {stats_row['slope_ci_high']:.3f})\n"
            f"Pearson r = {stats_row['pearson_r']:.2f}, Kendall tau = {stats_row['kendall_tau']:.2f}"
        )
        ax.text(
            0.05,
            0.9,
            note,
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_title(f"{metric.label} vs UVA")
        ax.set_ylabel("Trimmed mean (mg per gDW)")
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    axes[-2].set_xlabel("UVA irradiance (mW cm$^{-2}$)")
    fig.suptitle("UVA trend regressions with weighted fits", fontsize=14, y=0.98)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "uva_trend_regressions.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path



def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    dose_df = tables["dose_summary"]
    delta_df = tables["deltas"]
    replicates_df = tables["replicates"]
    alignment_df = tables["alignment"]
    trend_df = tables["trend_stats"]

    generated = [
        plot_dose_trajectories(dose_df),
        plot_sequential_deltas(delta_df),
        plot_cross_assay_concordance(dose_df, alignment_df),
        plot_uv_regime_context(dose_df),
        plot_reduced_vs_oxidized_panels(dose_df),
        plot_replicates_vs_trimmed_means(dose_df, replicates_df),
        plot_peak_comparison_panels(dose_df),
        plot_quadratic_fit_curvature(dose_df),
        plot_uva_trend_regressions(dose_df, trend_df),
    ]

    for path in generated:
        print(f"Saved {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
