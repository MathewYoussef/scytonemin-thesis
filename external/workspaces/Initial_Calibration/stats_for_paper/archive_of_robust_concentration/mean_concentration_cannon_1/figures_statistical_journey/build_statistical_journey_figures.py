"""
Generate the figure suite that illustrates the statistical journey from
replicate-level null findings to dose-level robust mean trends.

Each figure is saved into the same directory as this script:
1. dose_overview.png
2. replicate_stripplots.png
3. dose_mean_trends.png
4. sequential_deltas.png
5. chrom_dad_concordance.png
6. hypothesis_synthesis.png

Assumes the working directory is the project root containing the CSV inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parent
RNG = np.random.default_rng(42)


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "replicates": pd.read_csv(PROJECT_ROOT / "Combined_Scytonemin_Concentrations.csv"),
        "dose_summary": pd.read_csv(PROJECT_ROOT / "dose_level_summary.csv"),
        "delta": pd.read_csv(PROJECT_ROOT / "dose_pattern_sequential_deltas.csv"),
        "trend": pd.read_csv(PROJECT_ROOT / "dose_trend_stats.csv"),
        "alignment": pd.read_csv(PROJECT_ROOT / "chrom_dad_alignment.csv"),
    }
    return data


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def figure_dose_overview(dose_summary: pd.DataFrame) -> None:
    base = (
        dose_summary[["dose_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        .drop_duplicates()
        .sort_values("p_uva_mw_cm2")
    )

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(
        base["p_uva_mw_cm2"],
        base["p_uvb_mw_cm2"],
        c="#2a6f97",
        s=70,
        zorder=3,
    )
    ax.plot(
        base["p_uva_mw_cm2"],
        base["p_uvb_mw_cm2"],
        color="#264653",
        linestyle="--",
        alpha=0.6,
        zorder=2,
    )

    # highlight the non-monotonic UVB drop
    if len(base) >= 2:
        prev = base.iloc[-2]
        curr = base.iloc[-1]
        ax.annotate(
            "",
            xy=(curr["p_uva_mw_cm2"], curr["p_uvb_mw_cm2"]),
            xytext=(prev["p_uva_mw_cm2"], prev["p_uvb_mw_cm2"]),
            arrowprops=dict(arrowstyle="->", color="#e76f51", lw=2),
        )
        ax.text(
            curr["p_uva_mw_cm2"],
            curr["p_uvb_mw_cm2"] - 0.05,
            "UVB dip",
            color="#e76f51",
            ha="right",
        )

    for _, row in base.iterrows():
        ax.text(
            row["p_uva_mw_cm2"],
            row["p_uvb_mw_cm2"] + 0.035,
            f"{row['dose_id']}:\n({row['p_uva_mw_cm2']:.3f}, {row['p_uvb_mw_cm2']:.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

    ax.set_xlabel("UVA (mW·cm$^{-2}$)")
    ax.set_ylabel("UVB (mW·cm$^{-2}$)")
    ax.set_title("Dose Overview: UVA vs. UVB Pairings")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dose_overview.png", dpi=300)
    plt.close(fig)


def figure_replicate_stripplots(
    replicates: pd.DataFrame, dose_summary: pd.DataFrame
) -> None:
    metrics = [
        ("Chrom Total", "chrom_total_mg_per_gDW"),
        ("Chrom Reduced", "chrom_reduced_mg_per_gDW"),
        ("DAD Total", "dad_total_mg_per_gDW"),
        ("DAD Reduced", "dad_reduced_mg_per_gDW"),
    ]

    unique_uva = sorted(replicates["p_uva_mw_cm2"].unique())
    dose_summary = dose_summary.copy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for ax, (title, col) in zip(axes, metrics):
        # jittered replicate points
        grouped = replicates.groupby("p_uva_mw_cm2")
        for idx, (dose, subset) in enumerate(grouped):
            jitter = RNG.uniform(-0.3, 0.3, size=len(subset)) * 0.15
            ax.scatter(
                np.full(len(subset), idx) + jitter,
                subset[col],
                alpha=0.6,
                color="#577590",
                edgecolor="white",
                linewidth=0.5,
            )

        # overlay trimmed mean with CI
        trimmed = dose_summary.sort_values("p_uva_mw_cm2")[
            [
                "p_uva_mw_cm2",
                f"{col}_trimmed_mean",
                f"{col}_trimmed_sd",
                f"{col}_ci_low",
                f"{col}_ci_high",
            ]
        ].dropna()

        ax.errorbar(
            range(len(trimmed)),
            trimmed[f"{col}_trimmed_mean"],
            yerr=np.vstack(
                (
                    trimmed[f"{col}_trimmed_mean"] - trimmed[f"{col}_ci_low"],
                    trimmed[f"{col}_ci_high"] - trimmed[f"{col}_trimmed_mean"],
                )
            ),
            fmt="o-",
            color="#bc4749",
            ecolor="#bc4749",
            capsize=4,
            label="Trimmed mean ± bootstrap CI",
        )

        ax.set_title(title)
        ax.grid(alpha=0.2)

        for dose_idx, row in enumerate(trimmed.itertuples()):
            ax.text(
                dose_idx,
                getattr(row, f"{col}_trimmed_mean") + 0.05,
                f"{getattr(row, f'{col}_trimmed_mean'):.2f}\n[{getattr(row, f'{col}_ci_low'):.2f}, {getattr(row, f'{col}_ci_high'):.2f}]",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            )

    axes[0].legend(loc="upper left")
    axes[-1].set_xticks(range(len(unique_uva)))
    axes[-1].set_xticklabels(
        [f"{v:.3f}" for v in unique_uva],
        rotation=45,
        ha="right",
    )
    fig.suptitle("Replicates vs. Trimmed Means by Dose", y=0.995)
    fig.text(0.5, 0.04, "UVA dose (mW·cm$^{-2}$; ascending)", ha="center")
    fig.text(0.01, 0.5, "Concentration (mg·gDW$^{-1}$)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "replicate_stripplots.png", dpi=300)
    plt.close(fig)


def weighted_polyfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int):
    # normal equation solution with weights
    return np.polynomial.polynomial.Polynomial.fit(x, y, deg=degree, w=w).convert()


def figure_dose_mean_trends(dose_summary: pd.DataFrame, trend: pd.DataFrame) -> None:
    metrics = [
        ("Chrom Total", "chrom_total_mg_per_gDW"),
        ("Chrom Reduced", "chrom_reduced_mg_per_gDW"),
        ("DAD Total", "dad_total_mg_per_gDW"),
        ("DAD Reduced", "dad_reduced_mg_per_gDW"),
    ]

    x = dose_summary["p_uva_mw_cm2"].values

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for ax, (title, base_col) in zip(axes, metrics):
        mean_col = f"{base_col}_trimmed_mean"
        sd_col = f"{base_col}_trimmed_sd"
        y = dose_summary[mean_col].values
        sd_vals = dose_summary[sd_col].to_numpy(copy=True)
        positive = sd_vals[sd_vals > 0]
        fallback = positive.mean() if len(positive) else 1.0
        sd_vals = np.where(sd_vals == 0, fallback, sd_vals)
        w = 1 / np.square(sd_vals)

        ci_low = dose_summary[f"{base_col}_ci_low"].values
        ci_high = dose_summary[f"{base_col}_ci_high"].values
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - ci_low, ci_high - y)),
            fmt="o",
            color="#2a9d8f",
            capsize=4,
        )

        # linear fit line
        poly1 = weighted_polyfit(x, y, w, degree=1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, poly1(x_line), color="#264653", label="Weighted linear fit")

        # quadratic fit
        poly2 = weighted_polyfit(x, y, w, degree=2)
        ax.plot(
            x_line,
            poly2(x_line),
            color="#e76f51",
            linestyle="--",
            label="Weighted quadratic fit",
        )

        ax.set_title(title)
        ax.grid(alpha=0.2)

        trend_row = trend[(trend["metric"] == f"{base_col}") & (trend["axis"] == "UVA")].iloc[0]
        ax.text(
            0.02,
            0.95,
            (
                f"Slope = {trend_row['slope']:.3f}\n"
                f"95% CI [{trend_row['slope_ci_low']:.2f}, {trend_row['slope_ci_high']:.2f}]\n"
                f"R² = {trend_row['r_squared']:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

    axes[0].legend(loc="upper left", fontsize=9)
    fig.text(0.5, 0.04, "UVA dose (mW·cm$^{-2}$)", ha="center")
    fig.text(0.01, 0.5, "Trimmed mean (mg·gDW$^{-1}$)", va="center", rotation="vertical")
    fig.suptitle("Dose-Level Trimmed Means with Weighted Fits", y=0.995)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "dose_mean_trends.png", dpi=300)
    plt.close(fig)


def figure_sequential_deltas(delta: pd.DataFrame) -> None:
    focus_metrics = [
        "Chrom Total (mg·gDW⁻¹)",
        "Chrom Reduced",
        "DAD Total (mg·gDW⁻¹)",
        "DAD Reduced",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, focus_metrics):
        sub = delta[delta["metric"] == metric]
        x = np.arange(len(sub))
        y = sub["delta_trimmed_mean"].values
        lower = y - sub["delta_ci_low"].values
        upper = sub["delta_ci_high"].values - y

        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            fmt="o",
            capsize=4,
            color="#1d3557",
            ecolor="#457b9d",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["transition"], rotation=45, ha="right")
        ax.set_title(metric)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Delta (mg·gDW$^{-1}$)")

        for idx, row in enumerate(sub.itertuples()):
            delta_value = getattr(row, "delta_trimmed_mean")
            ci_low = getattr(row, "delta_ci_low")
            ci_high = getattr(row, "delta_ci_high")
            offset = 0.05 if delta_value >= 0 else -0.05
            ax.text(
                idx,
                delta_value + offset,
                f"{delta_value:.2f}\n[{ci_low:.2f}, {ci_high:.2f}]",
                ha="center",
                va="bottom" if delta_value >= 0 else "top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
            )

    fig.suptitle("Sequential Dose-to-Dose Deltas (Trimmed Means)", y=0.995)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "sequential_deltas.png", dpi=300)
    plt.close(fig)


def figure_chrom_dad_concordance(dose_summary: pd.DataFrame, alignment: pd.DataFrame):
    cols = [
        ("Total", "chrom_total_mg_per_gDW_trimmed_mean", "dad_total_mg_per_gDW_trimmed_mean"),
        ("Oxidized", "chrom_oxidized_mg_per_gDW_trimmed_mean", "dad_oxidized_mg_per_gDW_trimmed_mean"),
        ("Reduced", "chrom_reduced_mg_per_gDW_trimmed_mean", "dad_reduced_mg_per_gDW_trimmed_mean"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (label, chrom_col, dad_col) in zip(axes, cols):
        ax.scatter(
            dose_summary[chrom_col],
            dose_summary[dad_col],
            color="#6a4c93",
            s=60,
        )
        ax.set_xlabel(f"Chromatogram {label} (mg·gDW$^{-1}$)")
        ax.set_ylabel(f"DAD {label} (mg·gDW$^{-1}$)")
        ax.grid(alpha=0.2)

        row = alignment[alignment["metric"] == label.lower()].iloc[0]
        slope = row["deming_slope"]
        intercept = row["deming_intercept"]
        x_line = np.linspace(
            dose_summary[chrom_col].min(), dose_summary[chrom_col].max(), 100
        )
        ax.plot(
            x_line,
            intercept + slope * x_line,
            color="#ff7b00",
            label=f"Deming fit (slope={slope:.2f})",
        )
        ax.legend(fontsize=8)
        ax.text(
            0.02,
            0.95,
            (
                f"Pearson r = {row['pearson_r']:.2f}\n"
                f"Spearman r = {row['spearman_r']:.2f}\n"
                f"Slope CI [{row['deming_slope_ci_low']:.2f}, {row['deming_slope_ci_high']:.2f}]"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

    fig.suptitle("Chromatogram vs. DAD Concordance (Dose Means)", y=0.995)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "chrom_dad_concordance.png", dpi=300)
    plt.close(fig)


def kendall_tau(x: Iterable[float], y: Iterable[float]) -> float:
    x = list(x)
    y = list(y)
    n = len(x)
    concordant = discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 or dy == 0:
                continue
            concordant += int(dx * dy > 0)
            discordant += int(dx * dy < 0)

    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else np.nan


def figure_hypothesis_synthesis(
    replicates: pd.DataFrame,
    dose_summary: pd.DataFrame,
    trend: pd.DataFrame,
) -> None:
    metrics = [
        ("Chrom Total", "chrom_total_mg_per_gDW"),
        ("Chrom Reduced", "chrom_reduced_mg_per_gDW"),
        ("DAD Total", "dad_total_mg_per_gDW"),
        ("DAD Reduced", "dad_reduced_mg_per_gDW"),
    ]

    uva_ranks = replicates["p_uva_mw_cm2"].rank(method="dense")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: replicate-level Kendall tau for each metric
    taus = []
    for label, col in metrics:
        taus.append(
            {
                "Metric": label,
                "Tau": kendall_tau(uva_ranks.values, replicates[col].values),
            }
        )
    tau_df = pd.DataFrame(taus)
    axes[0].bar(tau_df["Metric"], tau_df["Tau"], color="#4361ee")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Kendall τ (replicate-level)")
    axes[0].set_title("Replicate-Level Monotonicity Checks")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.2)
    for idx, row in tau_df.iterrows():
        axes[0].text(
            idx,
            row["Tau"] + np.sign(row["Tau"]) * 0.02,
            f"{row['Tau']:.2f}",
            ha="center",
            va="bottom" if row["Tau"] >= 0 else "top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

    # right: dose-level slopes with 95% CI
    metric_map = {col: label for label, col in metrics}
    subset = (
        trend[(trend["axis"] == "UVA")]
        .copy()
        .assign(
            Metric=lambda df: df["metric"].map(metric_map).fillna(df["metric"])
        )
    )
    order = [label for label, _ in metrics]
    subset = subset[subset["Metric"].isin(order)]
    subset["Metric"] = pd.Categorical(subset["Metric"], categories=order, ordered=True)
    subset = subset.sort_values("Metric")
    y_pos = np.arange(len(subset))
    axes[1].errorbar(
        subset["slope"],
        y_pos,
        xerr=np.vstack(
            (
                subset["slope"] - subset["slope_ci_low"],
                subset["slope_ci_high"] - subset["slope"],
            )
        ),
        fmt="o",
        color="#e63946",
        ecolor="#e63946",
        capsize=4,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(subset["Metric"])
    axes[1].set_xlabel("Weighted slope (mg·gDW$^{-1}$ per mW·cm$^{-2}$)")
    axes[1].set_title("Dose-Level Slopes from Trimmed Means")
    axes[1].grid(alpha=0.2)
    for idx, row in subset.iterrows():
        axes[1].text(
            row["slope"],
            idx + 0.05,
            f"{row['slope']:.3f}\n[{row['slope_ci_low']:.2f}, {row['slope_ci_high']:.2f}]",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

    fig.suptitle("Hypothesis Synthesis: From Replicates to Robust Means", y=0.995)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "hypothesis_synthesis.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    data = load_data()

    figure_dose_overview(data["dose_summary"])
    figure_replicate_stripplots(data["replicates"], data["dose_summary"])
    figure_dose_mean_trends(data["dose_summary"], data["trend"])
    figure_sequential_deltas(data["delta"])
    figure_chrom_dad_concordance(data["dose_summary"], data["alignment"])
    figure_hypothesis_synthesis(
        data["replicates"], data["dose_summary"], data["trend"]
    )


if __name__ == "__main__":
    main()
