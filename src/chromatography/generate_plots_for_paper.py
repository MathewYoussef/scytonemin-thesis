#!/usr/bin/env python3
"""Regenerate publication figures from precomputed stats tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "plots_for_paper"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=300)
    plt.close(fig)


def plot_dad_spectrum() -> None:
    spectra = pd.read_csv(PROJECT_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "scytonemin_spectra_tidy.csv")
    window = spectra[(spectra["wavelength_nm"] >= 320) & (spectra["wavelength_nm"] <= 480)]
    samples = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")

    panels = [
        ("total", "predicted_total_mg_ml", "predicted_total_mg_per_gDW"),
        ("oxidized", "predicted_oxidized_mg_ml", "predicted_oxidized_mg_per_gDW"),
        ("reduced", "predicted_reduced_mg_ml", "predicted_reduced_mg_per_gDW"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    blank_mean = (
        window[(window["sample_category"] == "blank") & (window["spectrum_state"] == "blank")] \
        .groupby("wavelength_nm")["intensity_abs"].mean()
    )

    for ax, (state, mg_ml_col, mg_gdw_col) in zip(axes, panels):
        top = samples.sort_values(mg_gdw_col, ascending=False).iloc[0]
        sample_id = top["sample_id"]
        trace = window[
            (window["sample_category"] == "sample")
            & (window["spectrum_state"] == state)
            & (window["sample_id"] == sample_id)
        ]
        if trace.empty:
            continue
        ax.plot(trace["wavelength_nm"], trace["intensity_abs"], label=f"Sample {sample_id}", color="#1f77b4")
        ax.plot(blank_mean.index, blank_mean.values, label="Blank mean", color="#ff7f0e")
        ax.set_title(f"{state.capitalize()} spectrum")
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.text(
            0.03,
            0.92,
            f"Sample {sample_id}\n{top[mg_ml_col]:.3f} mg·mL⁻¹\n{top[mg_gdw_col]:.2f} mg·gDW⁻¹",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
        if ax is axes[0]:
            ax.set_ylabel("Absorbance (a.u.)")
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("DAD spectra (320–480 nm) for representative treatments")
    save(fig, "01_dad_spectrum_sample_vs_blank.png")


def plot_chrom_calibration() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "DAD_RAW_FILES" / "scytonemin_chromatogram_areas" / "raw_total_scytonemin.csv")
    standards_frame = (
        raw[raw["sample_category"] == "standard"]
        [["area", "standard_concentration_mg_ml"]]
        .dropna()
        .rename(columns={"standard_concentration_mg_ml": "known"})
        .astype(float)
    )

    samples = raw[raw["sample_category"] == "sample"].copy()
    samples["sample_id"] = samples["sample_name"].str.split().str[-1]
    derived = pd.read_csv(PROJECT_ROOT / "chromatogram_derived_concentrations.csv")
    samples = samples.merge(derived[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2", "conc_total_mg_ml"]], on="sample_id", how="left")
    samples = samples.dropna(subset=["area", "conc_total_mg_ml"])

    slope, intercept = np.polyfit(standards_frame["area"], standards_frame["known"], 1)
    x_line = np.linspace(min(samples["area"].min(), standards_frame["area"].min()), max(samples["area"].max(), standards_frame["area"].max()), 200)
    y_line = slope * x_line + intercept

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cmap = sns.color_palette("viridis", as_cmap=True)
    for ax, colour, title in [
        (axes[0], "p_uva_mw_cm2", "Treatments coloured by UVA (mW·cm⁻²)"),
        (axes[1], "p_uvb_mw_cm2", "Treatments coloured by UVB (mW·cm⁻²)"),
    ]:
        sc = ax.scatter(samples["area"], samples["conc_total_mg_ml"], c=samples[colour], cmap=cmap, s=14, alpha=0.9)
        ax.scatter(standards_frame["area"], standards_frame["known"], color="#1f77b4", s=35, label="Standards")
        ax.plot(x_line, y_line, color="#444444", linewidth=1.2, label="Least squares fit")
        ax.set_xlabel("Chromatogram peak area (counts)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.2)
        sns.despine(ax=ax)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=colour.replace("p_", "").replace("_", " ") + " (mW·cm⁻²)")
        if ax is axes[0]:
            ax.set_ylabel("Concentration (mg·mL⁻¹)")
            ax.legend(frameon=False)
    axes[0].text(0.04, 0.92, f"Fit: y={slope:.3e}·x + {intercept:.3f}", transform=axes[0].transAxes, fontsize=8.5,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    fig.suptitle("Chromatogram calibration (total form)")
    save(fig, "02_chromatogram_calibration_scatter.png")


def plot_dad_calibration() -> None:
    standards = pd.read_csv(PROJECT_ROOT / "Diode_Array_Derived_Calibration_Plots" / "standards_fitted_total.csv")[["auc_corrected", "known_concentration_mg_ml"]].dropna()
    standards.rename(columns={"auc_corrected": "auc"}, inplace=True)
    treatments = pd.read_csv(PROJECT_ROOT / "Diode_Array_Derived_Calibration_Plots" / "total_treatment_overlay_grouped_by_uva.csv")
    samples = treatments[treatments["sample_type"] == "treatment"].dropna(subset=["auc", "concentration_mg_ml"])
    dad_meta = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")
    samples = samples.drop(columns=[col for col in samples.columns if col.startswith("p_uva_mw_cm2")], errors="ignore")
    samples = samples.merge(dad_meta[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]], on="sample_id", how="left")
    samples = samples.dropna(subset=["p_uva_mw_cm2", "p_uvb_mw_cm2"])

    slope, intercept = np.polyfit(standards["auc"], standards["known_concentration_mg_ml"], 1)
    x_line = np.linspace(min(standards["auc"].min(), samples["auc"].min()), max(standards["auc"].max(), samples["auc"].max()), 200)
    y_line = slope * x_line + intercept

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = sns.color_palette("plasma", as_cmap=True)
    for ax, colour, title in [
        (axes[0], "p_uva_mw_cm2", "Treatments coloured by UVA (mW·cm⁻²)"),
        (axes[1], "p_uvb_mw_cm2", "Treatments coloured by UVB (mW·cm⁻²)"),
    ]:
        sc = ax.scatter(samples["auc"], samples["concentration_mg_ml"], c=samples[colour], cmap=cmap, s=18, alpha=0.9, label="Treatments")
        ax.scatter(standards["auc"], standards["known_concentration_mg_ml"], color="#2ca02c", s=42, label="Standards")
        ax.plot(x_line, y_line, color="#444444", linewidth=1.2, label="Least squares fit")
        ax.set_xlabel("Integrated absorbance (AUC)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.2)
        sns.despine(ax=ax)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=colour.replace("p_", "").replace("_", " ") + " (mW·cm⁻²)")
        if ax is axes[0]:
            ax.set_ylabel("Concentration (mg·mL⁻¹)")
            ax.legend(frameon=False)
    axes[0].text(0.04, 0.91, f"Fit: y={slope:.3e}·x + {intercept:.3f}", transform=axes[0].transAxes, fontsize=8.5,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    fig.suptitle("DAD calibration (total 320–480 nm)")
    save(fig, "03_dad_calibration_scatter.png")


def plot_concentration_table() -> None:
    chrom = pd.read_csv(PROJECT_ROOT / "chromatogram_derived_concentrations.csv")
    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")
    table = chrom.merge(
        dad[["sample_id", "predicted_total_mg_ml", "predicted_total_mg_per_gDW"]],
        on="sample_id",
        how="inner",
    )
    table = table.sort_values("sample_id").head(10)
    table = table[[
        "sample_id",
        "p_uva_mw_cm2",
        "p_uvb_mw_cm2",
        "conc_total_mg_ml",
        "total_mg_per_gDW",
        "predicted_total_mg_ml",
        "predicted_total_mg_per_gDW",
    ]].round(4)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.35)
    ax.set_title("Example concentrations derived from calibrations", pad=12)
    save(fig, "04_concentration_table.png")


def plot_single_factor_panels() -> None:
    chrom = pd.read_csv(PROJECT_ROOT / "chromatogram_derived_concentrations.csv")
    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")

    def _panel(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
        panel = df[["p_uva_mw_cm2", "p_uvb_mw_cm2", value_col]].dropna().copy()
        panel.rename(columns={value_col: "value"}, inplace=True)
        panel["label"] = label
        return panel

    panels = [
        _panel(chrom, "total_mg_per_gDW", "Chromatogram total mg·gDW⁻¹"),
        _panel(chrom, "oxidized_mg_per_gDW", "Chromatogram oxidized mg·gDW⁻¹"),
        _panel(chrom, "reduced_mg_per_gDW", "Chromatogram reduced mg·gDW⁻¹"),
        _panel(dad, "predicted_total_mg_per_gDW", "DAD total mg·gDW⁻¹"),
        _panel(dad, "predicted_oxidized_mg_per_gDW", "DAD oxidized mg·gDW⁻¹"),
        _panel(dad, "predicted_reduced_mg_per_gDW", "DAD reduced mg·gDW⁻¹"),
    ]

    fig, axes = plt.subplots(len(panels), 4, figsize=(16, 3.2 * len(panels)))

    for row, panel in enumerate(panels):
        title_prefix = panel["label"].iloc[0]

        for col_idx, (dose_col, axis_label) in enumerate([
            ("p_uvb_mw_cm2", "UVB dose (mW·cm⁻²)"),
            ("p_uva_mw_cm2", "UVA dose (mW·cm⁻²)"),
        ]):
            ax = axes[row, col_idx]
            sns.boxplot(data=panel, x=dose_col, y="value", color="#4C72B0", ax=ax)
            sns.stripplot(data=panel, x=dose_col, y="value", color="#111111", size=3, alpha=0.6, jitter=True, ax=ax)
            ax.set_xlabel(axis_label)
            ax.set_title(f"{title_prefix} vs {axis_label.split()[0]}")
            if col_idx == 0:
                ax.set_ylabel(title_prefix)
            else:
                ax.set_ylabel("")

        grouped_ax = axes[row, 2]
        sns.boxplot(
            data=panel,
            x="p_uvb_mw_cm2",
            y="value",
            hue="p_uva_mw_cm2",
            palette="Blues",
            ax=grouped_ax,
        )
        grouped_ax.legend(title="UVA", frameon=False, fontsize=8)
        grouped_ax.set_xlabel("UVB dose (mW·cm⁻²)")
        grouped_ax.set_title(f"{title_prefix}: UVB grouped by UVA")
        grouped_ax.set_ylabel("")

        scatter_ax = axes[row, 3]
        norm = colors.Normalize(vmin=panel["value"].min(), vmax=panel["value"].max())
        sc = scatter_ax.scatter(
            panel["p_uvb_mw_cm2"],
            panel["p_uva_mw_cm2"],
            c=panel["value"],
            cmap="coolwarm",
            s=45,
            alpha=0.85,
            norm=norm,
        )
        scatter_ax.set_xlabel("UVB dose (mW·cm⁻²)")
        scatter_ax.set_ylabel("UVA dose (mW·cm⁻²)")
        scatter_ax.set_title(f"{title_prefix}: UVA vs UVB")
        fig.colorbar(sc, ax=scatter_ax, label=title_prefix)

    fig.suptitle("Single-factor distributions for chromatogram and DAD metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "05_single_factor_panels.png", dpi=300)
    plt.close(fig)


def plot_single_factor_variation() -> None:
    chrom = pd.read_csv(PROJECT_ROOT / "chromatogram_derived_concentrations.csv")
    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")

    chrom_panel = chrom[["p_uva_mw_cm2", "p_uvb_mw_cm2", "total_mg_per_gDW"]].dropna()
    dad_panel = dad[["p_uva_mw_cm2", "p_uvb_mw_cm2", "predicted_total_mg_per_gDW"]].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False)

    sns.boxplot(data=chrom_panel, x="p_uvb_mw_cm2", y="total_mg_per_gDW", color="#4C72B0", ax=axes[0, 0])
    sns.stripplot(data=chrom_panel, x="p_uvb_mw_cm2", y="total_mg_per_gDW", color="#111111", size=3, alpha=0.6, jitter=True, ax=axes[0, 0])
    axes[0, 0].set_xlabel("UVB dose (mW·cm⁻²)")
    axes[0, 0].set_ylabel("Chromatogram total mg·gDW⁻¹")
    axes[0, 0].set_title("Chromatogram total vs UVB")

    sns.boxplot(data=chrom_panel, x="p_uva_mw_cm2", y="total_mg_per_gDW", color="#4C72B0", ax=axes[0, 1])
    sns.stripplot(data=chrom_panel, x="p_uva_mw_cm2", y="total_mg_per_gDW", color="#111111", size=3, alpha=0.6, jitter=True, ax=axes[0, 1])
    axes[0, 1].set_xlabel("UVA dose (mW·cm⁻²)")
    axes[0, 1].set_ylabel("Chromatogram total mg·gDW⁻¹")
    axes[0, 1].set_title("Chromatogram total vs UVA")

    sns.boxplot(data=dad_panel, x="p_uvb_mw_cm2", y="predicted_total_mg_per_gDW", color="#55A868", ax=axes[1, 0])
    sns.stripplot(data=dad_panel, x="p_uvb_mw_cm2", y="predicted_total_mg_per_gDW", color="#111111", size=3, alpha=0.6, jitter=True, ax=axes[1, 0])
    axes[1, 0].set_xlabel("UVB dose (mW·cm⁻²)")
    axes[1, 0].set_ylabel("DAD total mg·gDW⁻¹")
    axes[1, 0].set_title("DAD total vs UVB")

    sns.boxplot(data=dad_panel, x="p_uva_mw_cm2", y="predicted_total_mg_per_gDW", color="#55A868", ax=axes[1, 1])
    sns.stripplot(data=dad_panel, x="p_uva_mw_cm2", y="predicted_total_mg_per_gDW", color="#111111", size=3, alpha=0.6, jitter=True, ax=axes[1, 1])
    axes[1, 1].set_xlabel("UVA dose (mW·cm⁻²)")
    axes[1, 1].set_ylabel("DAD total mg·gDW⁻¹")
    axes[1, 1].set_title("DAD total vs UVA")

    fig.suptitle("Single-factor view: total mg·gDW⁻¹ vs UV doses")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "05b_single_factor_total_only.png", dpi=300)
    plt.close(fig)


def plot_heatmaps() -> None:
    zscores = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "zscore_using_control" / "chromatogram_zscores.csv")
    total = zscores[zscores["form"] == "total"]
    chrom_pivot = total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])["z_amount_mg_per_gDW"].mean().unstack()
    chrom_counts = total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"]).size().unstack().fillna(0).astype(int)
    chrom_annot = chrom_pivot.copy()
    for uva in chrom_pivot.index:
        for uvb in chrom_pivot.columns:
            chrom_annot.loc[uva, uvb] = f"{chrom_pivot.loc[uva, uvb]:.2f}\n(n={chrom_counts.loc[uva, uvb]})"
    fig, ax = plt.subplots(figsize=(6, 4.3))
    sns.heatmap(chrom_pivot, annot=chrom_annot, fmt="", cmap="coolwarm", ax=ax, cbar_kws={"label": "Mean z-score (mg·gDW⁻¹)"})
    ax.set_xlabel("UVB dose (mW·cm⁻²)")
    ax.set_ylabel("UVA dose (mW·cm⁻²)")
    ax.set_title("Chromatogram z-score mg·gDW⁻¹ (mean ± n)")
    save(fig, "06a_heatmap_chrom_zscore.png")

    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")
    dad_pivot = dad.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])["predicted_total_mg_per_gDW"].mean().unstack()
    dad_counts = dad.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"]).size().unstack().fillna(0).astype(int)
    dad_annot = dad_pivot.copy()
    for uva in dad_pivot.index:
        for uvb in dad_pivot.columns:
            dad_annot.loc[uva, uvb] = f"{dad_pivot.loc[uva, uvb]:.2f}\n(n={dad_counts.loc[uva, uvb]})"
    fig, ax = plt.subplots(figsize=(6, 4.3))
    sns.heatmap(dad_pivot, annot=dad_annot, fmt="", cmap="coolwarm", ax=ax, cbar_kws={"label": "Mean predicted total (mg·gDW⁻¹)"})
    ax.set_xlabel("UVB dose (mW·cm⁻²)")
    ax.set_ylabel("UVA dose (mW·cm⁻²)")
    ax.set_title("DAD predicted total mg·gDW⁻¹ (mean ± n)")
    save(fig, "06b_heatmap_dad_total.png")

    datasets = []
    raw = pd.read_csv(PROJECT_ROOT / "chromatogram_derived_concentrations.csv")
    datasets.append((raw.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])["total_mg_per_gDW"].mean().unstack(),
                     raw.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"]).size().unstack().fillna(0),
                     "Raw mg·gDW⁻¹"))
    delta = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "delta_from_control" / "chromatogram_delta.csv")
    delta_total = delta[delta["form"] == "total"]
    datasets.append((delta_total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])["delta_amount_mg_per_gDW"].mean().unstack(),
                     delta_total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"]).size().unstack().fillna(0),
                     "Delta mg·gDW⁻¹"))
    ratio = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "ratio_to_control" / "chromatogram_ratio.csv")
    ratio_total = ratio[ratio["form"] == "total"]
    datasets.append((ratio_total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])["ratio_amount_mg_per_gDW"].mean().unstack(),
                     ratio_total.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"]).size().unstack().fillna(0),
                     "Ratio to control mg·gDW⁻¹"))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, (pivot, counts, title) in zip(axes, datasets):
        annot = pivot.copy()
        for uva in pivot.index:
            for uvb in pivot.columns:
                annot.loc[uva, uvb] = f"{pivot.loc[uva, uvb]:.2f}\n(n={counts.loc[uva, uvb]})"
        sns.heatmap(pivot, annot=annot, fmt="", cmap="coolwarm", cbar=False, ax=ax)
        ax.set_xlabel("UVB dose (mW·cm⁻²)")
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("UVA dose (mW·cm⁻²)")
    fig.suptitle("Chromatogram mg·gDW⁻¹ (non-significant interaction views)")
    save(fig, "06c_heatmaps_chrom_other.png")


def plot_permutation_comparison() -> None:
    simple = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "two_factor_modeling" / "chromatogram_two_way_anova_permutation.csv")
    freedman = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "two_factor_modeling" / "freedman_lane_interaction.csv")
    merged = simple.merge(freedman, on=["dataset", "variant", "measurement"], how="inner", suffixes=("_simple", "_freedman"))
    if merged.empty:
        return
    merged["descriptor"] = merged["dataset"] + " • " + merged["variant"] + " • " + merged["measurement"]
    df = merged.melt(id_vars="descriptor", value_vars=["p_perm", "p_freedman_lane"], var_name="test", value_name="p_value")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=df, x="p_value", y="descriptor", hue="test", ax=ax,
                palette={"p_perm": "#4C72B0", "p_freedman_lane": "#DD8452"})
    ax.axvline(0.05, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Empirical p-value")
    ax.set_ylabel("Dataset • variant • metric")
    ax.set_title("Permutation tests for UVA×UVB interaction")
    ax.set_xlim(0, min(1.0, df["p_value"].max() * 1.15))
    ax.legend(title="Permutation method", labels=["Simple shuffle", "Freedman–Lane"], frameon=False, loc="upper right")
    sns.despine(ax=ax)
    save(fig, "07_permutation_comparison.png")


def plot_ridge_bootstrap() -> None:
    boot = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "two_factor_modeling" / "ridge_bootstrap_summary.csv")
    boot = boot.sort_values(["dataset", "variant", "measurement", "term"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y_positions = range(len(boot))
    colors = {"p_uva_mw_cm2": "#1f77b4", "p_uvb_mw_cm2": "#2ca02c", "p_uva_mw_cm2:p_uvb_mw_cm2": "#d62728"}
    for y, (_, row) in zip(y_positions, boot.iterrows()):
        ax.plot([row["coef_p2_5"], row["coef_p97_5"]], [y, y], color=colors.get(row["term"], "#333333"), linewidth=3)
        ax.plot(row["coef_median"], y, marker="o", color=colors.get(row["term"], "#333333"))
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"{row['dataset']} • {row['variant']} • {row['measurement']} ({row['term']})" for _, row in boot.iterrows()], fontsize=8)
    ax.axvline(0, color="#444444", linestyle="--")
    ax.set_xlabel("Bootstrap coefficient interval (median ± 95% range)")
    ax.set_title("Ridge regression stability (2,000 bootstrap draws)")
    save(fig, "08_ridge_bootstrap_intervals.png")


def plot_outlier_sensitivity() -> None:
    outlier = pd.read_csv(PROJECT_ROOT / "Exploring_control_normalized" / "two_factor_modeling" / "outlier_sensitivity.csv")
    outlier["descriptor"] = outlier["variant"] + " • " + outlier["measurement"]
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    sns.barplot(data=outlier, x="scenario", y="p_value", hue="descriptor", ax=ax)
    ax.axhline(0.05, color="#444444", linestyle="--", linewidth=1)
    ax.set_ylabel("Interaction p-value")
    ax.set_xlabel("Removal scenario")
    ax.set_title("Outlier sensitivity of UVA×UVB interaction")
    ax.legend(loc="upper right", fontsize=9, frameon=False, title="Dataset • metric")
    sns.despine(ax=ax)
    save(fig, "09_outlier_sensitivity.png")


def main() -> None:
    ensure_output_dir()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    plot_dad_spectrum()
    plot_chrom_calibration()
    plot_dad_calibration()
    plot_concentration_table()
    plot_single_factor_panels()
    plot_single_factor_variation()
    plot_heatmaps()
    plot_permutation_comparison()
    plot_ridge_bootstrap()
    plot_outlier_sensitivity()


if __name__ == "__main__":
    main()
