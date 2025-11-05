#!/usr/bin/env python3
"""
Generate publication figures summarising calibration integrity, dose structure,
single-factor trends, two-factor diagnostics, and supplemental robustness
analyses for the scytonemin dataset.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tools.sm_exceptions import ValueWarning
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --------------------------------------------------------------------------- #
# Paths & styling
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXTERNAL_ROOT = PROJECT_ROOT.parent

CORE_DIR = SCRIPT_DIR / "figures_core"
SUPP_DIR = SCRIPT_DIR / "figures_supplement"
for directory in (CORE_DIR, SUPP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def weighted_linear_model(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    weight_mode: str | None = None,
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict[str, float]]:
    """Fit OLS/WLS and return model + summary stats."""
    X = sm.add_constant(df[x_col])
    if weight_mode == "1/x":
        weights = 1.0 / df[x_col].values
        model = sm.WLS(df[y_col], X, weights=weights).fit()
    elif weight_mode == "1/y":
        weights = 1.0 / df[y_col].values
        model = sm.WLS(df[y_col], X, weights=weights).fit()
    else:
        model = sm.OLS(df[y_col], X).fit()

    fitted = model.fittedvalues
    resid = df[y_col] - fitted
    rel_resid = (resid / df[y_col]).replace([np.inf, -np.inf], np.nan).dropna()
    stats = {
        "slope": model.params[x_col],
        "intercept": model.params["const"],
        "slope_se": model.bse[x_col],
        "intercept_se": model.bse["const"],
        "r2": model.rsquared,
        "max_rel": float(np.abs(rel_resid).max()) if not rel_resid.empty else np.nan,
        "df": float(model.df_resid),
    }
    return model, stats


def annotate_calibration(ax: plt.Axes, stats: Dict[str, float], loc: str = "upper left") -> None:
    """Place calibration statistics inside a subplot."""
    text = "\n".join(
        [
            f"slope = {stats['slope']:.3e} ± {stats['slope_se']:.2e}",
            f"intercept = {stats['intercept']:.3e} ± {stats['intercept_se']:.2e}",
            f"R² = {stats['r2']:.4f}",
            f"max |rel resid| = {stats['max_rel']:.3f}",
            f"df = {stats['df']:.0f}",
        ]
    )
    ha = "left" if loc.endswith("left") else "right"
    va = "top" if loc.startswith("upper") else "bottom"
    ax.text(
        0.02 if ha == "left" else 0.98,
        0.98 if va == "top" else 0.02,
        text,
        ha=ha,
        va=va,
        fontsize=11,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )


def add_residual_inset(ax: plt.Axes, fitted: Iterable[float], residuals: Iterable[float]) -> None:
    """Create residual plot inset with ±0.1/±0.2 anchors."""
    inset = inset_axes(ax, width="45%", height="45%", loc="lower right")
    inset.axhline(0, color="black", linewidth=1)
    for level, style, color in [(0.1, "--", "#d62728"), (0.2, ":", "#ff7f0e")]:
        inset.axhline(level, color=color, linestyle=style, linewidth=1)
        inset.axhline(-level, color=color, linestyle=style, linewidth=1)

    inset.scatter(fitted, residuals, color="#1f77b4", s=30, alpha=0.85)
    inset.set_xlabel("Fitted", fontsize=8)
    inset.set_ylabel("Rel resid", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_title("Residuals", fontsize=8)


def freedman_lane_interaction(
    y: np.ndarray,
    uva: np.ndarray,
    uvb: np.ndarray,
    n_perm: int = 2000,
    seed: int = 123,
) -> Tuple[float, np.ndarray, float]:
    """Freedman–Lane permutation test treating UVA/UVB as continuous doses."""
    X_full = np.column_stack([np.ones_like(uva), uva, uvb, uva * uvb])
    X_reduced = np.column_stack([np.ones_like(uva), uva, uvb])

    full_model = sm.OLS(y, X_full).fit()
    reduced_model = sm.OLS(y, X_reduced).fit()

    resid = reduced_model.resid
    fitted = reduced_model.fittedvalues

    rss_full = float(np.sum(full_model.resid**2))
    rss_reduced = float(np.sum(reduced_model.resid**2))

    df_full = full_model.df_resid
    df_reduced = reduced_model.df_resid
    df_diff = df_reduced - df_full
    f_obs = ((rss_reduced - rss_full) / df_diff) / (rss_full / df_full)

    rng = np.random.default_rng(seed)
    f_perm = np.empty(n_perm)
    for idx in range(n_perm):
        permuted = rng.permutation(resid)
        y_perm = fitted + permuted
        perm_full = sm.OLS(y_perm, X_full).fit()
        perm_reduced = sm.OLS(y_perm, X_reduced).fit()
        rss_full_p = float(np.sum(perm_full.resid**2))
        rss_reduced_p = float(np.sum(perm_reduced.resid**2))
        f_perm[idx] = ((rss_reduced_p - rss_full_p) / df_diff) / (rss_full_p / df_full)

    p_value = (np.count_nonzero(f_perm >= f_obs) + 1) / (n_perm + 1)
    return float(f_obs), f_perm, float(p_value)


def freedman_lane_categorical(
    data: pd.DataFrame,
    response: str,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> Tuple[float, np.ndarray, float]:
    """Freedman–Lane permutation mirroring run_resampling_sensitivity.py."""
    df = data.copy()
    df["p_uva_mw_cm2"] = df["p_uva_mw_cm2"].astype("category")
    df["p_uvb_mw_cm2"] = df["p_uvb_mw_cm2"].astype("category")

    full_formula = (
        f"{response} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )
    reduced_formula = f"{response} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ValueWarning)
        full_model = smf.ols(full_formula, data=df).fit()
        reduced_model = smf.ols(reduced_formula, data=df).fit()
        observed_table = sm.stats.anova_lm(full_model, typ=2)
    f_obs = float(observed_table.loc["C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", "F"])

    residuals = reduced_model.resid.to_numpy(copy=True)
    fitted = reduced_model.fittedvalues.to_numpy(copy=True)

    local_rng = rng or np.random.default_rng(12345)
    exceed = 1
    f_perm = []
    for _ in range(n_perm):
        local_rng.shuffle(residuals)
        perm_y = fitted + residuals
        perm_df = df.copy()
        perm_df["_perm_response"] = perm_y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            perm_model = smf.ols(
                full_formula.replace(response, "_perm_response"), data=perm_df
            ).fit()
            perm_table = sm.stats.anova_lm(perm_model, typ=2)
        perm_f = perm_table.loc["C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", "F"]
        if np.isnan(perm_f):
            continue
        f_perm.append(float(perm_f))
        if perm_f >= f_obs:
            exceed += 1
    f_perm_array = np.array(f_perm, dtype=float)
    p_value = exceed / (n_perm + 1)
    return f_obs, f_perm_array, p_value


def set_common_limits(axes: Iterable[plt.Axes], axis: str) -> None:
    """Enforce identical axis limits across comparable subplots."""
    limit_pairs = [ax.get_xlim() if axis == "x" else ax.get_ylim() for ax in axes]
    min_val = min(pair[0] for pair in limit_pairs)
    max_val = max(pair[1] for pair in limit_pairs)
    for ax in axes:
        if axis == "x":
            ax.set_xlim(min_val, max_val)
        else:
            ax.set_ylim(min_val, max_val)


# --------------------------------------------------------------------------- #
# Figure generators (core)
# --------------------------------------------------------------------------- #

def figure01_calibrations() -> None:
    """Calibration integrity (chromatogram + DAD)."""
    chrom_sources = {
        "Total": PROJECT_ROOT / "standards_fitted_total.csv",
        "Oxidized": PROJECT_ROOT / "standards_fitted_oxidized.csv",
        "Reduced": PROJECT_ROOT / "standards_fitted_reduced.csv",
    }
    dad_sources = {
        "Total": EXTERNAL_ROOT / "Diode_Array_Derived_Calibration_Plots" / "standards_fitted_total.csv",
        "Oxidized": EXTERNAL_ROOT / "Diode_Array_Derived_Calibration_Plots" / "standards_fitted_oxidized.csv",
        "Reduced": EXTERNAL_ROOT / "Diode_Array_Derived_Calibration_Plots" / "standards_fitted_reduced.csv",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
    # Chromatogram panels
    for col, (label, path) in enumerate(chrom_sources.items()):
        ax = axes[0, col]
        df = pd.read_csv(path)
        df = df[df["sample_category"] == "standard"].rename(
            columns={"response": "auc", "known_concentration_mg_ml": "conc"}
        )
        model, stats = weighted_linear_model(df, "auc", "conc", weight_mode="1/x")

        auc_grid = np.linspace(df["auc"].min() * 0.95, df["auc"].max() * 1.05, 200)
        X_pred = sm.add_constant(auc_grid)
        pred = model.get_prediction(X_pred)

        ax.scatter(df["auc"], df["conc"], s=90, color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.plot(auc_grid, pred.predicted_mean, color="black", linewidth=2)
        ax.fill_between(
            auc_grid,
            pred.conf_int()[:, 0],
            pred.conf_int()[:, 1],
            color="#1f77b4",
            alpha=0.25,
        )
        ax.set_title(f"1{chr(ord('A') + col)} — Chrom {label}")
        ax.set_xlabel("AUC (blank-corrected)")
        ax.set_ylabel("Concentration (mg·mL$^{-1}$)" if col == 0 else "")
        annotate_calibration(ax, stats)
        rel_resid = (df["conc"] - model.fittedvalues) / df["conc"]
        add_residual_inset(ax, model.fittedvalues, rel_resid)

    # DAD panels
    for col, (label, path) in enumerate(dad_sources.items()):
        ax = axes[1, col]
        df = pd.read_csv(path).rename(
            columns={"auc_corrected": "auc", "known_concentration_mg_ml": "conc"}
        )
        model, stats = weighted_linear_model(df, "auc", "conc", weight_mode="1/x")
        auc_grid = np.linspace(df["auc"].min() * 0.95, df["auc"].max() * 1.05, 200)
        X_pred = sm.add_constant(auc_grid)
        pred = model.get_prediction(X_pred)
        ax.scatter(df["auc"], df["conc"], s=90, color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax.plot(auc_grid, pred.predicted_mean, color="black", linewidth=2)
        ax.fill_between(
            auc_grid,
            pred.conf_int()[:, 0],
            pred.conf_int()[:, 1],
            color="#ff7f0e",
            alpha=0.25,
        )
        ax.set_title(f"1{chr(ord('D') + col)} — DAD {label}")
        ax.set_xlabel("AUC (blank-corrected)")
        ax.set_ylabel("Concentration (mg·mL$^{-1}$)" if col == 0 else "")
        annotate_calibration(ax, stats)
        rel_resid = (df["conc"] - model.fittedvalues) / df["conc"]
        add_residual_inset(ax, model.fittedvalues, rel_resid)

    fig.suptitle("Fig. 1 — Calibration integrity (Chromatogram & DAD)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(CORE_DIR / "fig01_calibration.png", dpi=300)
    plt.close(fig)


def figure02_dose_structure() -> None:
    """UVA/UVB lattice and marginals."""
    df = pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(df["p_uva_mw_cm2"], df["p_uvb_mw_cm2"], s=70, alpha=0.9)
    pearson_r = df[["p_uva_mw_cm2", "p_uvb_mw_cm2"]].corr().iloc[0, 1]
    axes[0].text(
        0.02,
        0.96,
        f"Pearson r = {pearson_r:.4f}",
        transform=axes[0].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    axes[0].text(
        0.02,
        0.78,
        "VIFs\nUVA ≈ 37.9\nUVB ≈ 10.4\nUVA×UVB ≈ 25.1",
        transform=axes[0].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    axes[0].set_xlabel("UVA (mW·cm$^{-2}$)")
    axes[0].set_ylabel("UVB (mW·cm$^{-2}$)")
    axes[0].set_title("2A — UVA vs UVB")

    uva_levels = sorted(df["p_uva_mw_cm2"].unique())
    sns.countplot(x="p_uva_mw_cm2", data=df, order=uva_levels, ax=axes[1], color="#1f77b4")
    axes[1].set_xlabel("UVA (mW·cm$^{-2}$)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("2B — UVA marginal (replicates)")

    uvb_levels = sorted(df["p_uvb_mw_cm2"].unique())
    sns.countplot(x="p_uvb_mw_cm2", data=df, order=uvb_levels, ax=axes[2], color="#ff7f0e")
    axes[2].set_xlabel("UVB (mW·cm$^{-2}$)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("2C — UVB marginal (replicates)")

    fig.suptitle("Fig. 2 — Dose structure & collinearity", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CORE_DIR / "fig02_dose_structure.png", dpi=300)
    plt.close(fig)


def figure03_single_factor_chrom() -> None:
    """Chromatogram single-factor trends (mg·gDW⁻¹)."""
    df = pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv")
    outcomes = {
        "Total": "total_mg_per_gDW",
        "Oxidized": "oxidized_mg_per_gDW",
        "Reduced": "reduced_mg_per_gDW",
    }
    forms = list(outcomes.keys())
    n = len(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
    for col, (form, y_col) in enumerate(outcomes.items()):
        # UVB row
        ax_uvb = axes[0, col]
        sns.regplot(
            x="p_uvb_mw_cm2",
            y=y_col,
            data=df,
            ax=ax_uvb,
            ci=95,
            scatter_kws=dict(s=70, alpha=0.85),
            line_kws=dict(color="black"),
        )
        model_uvb = sm.OLS(df[y_col], sm.add_constant(df["p_uvb_mw_cm2"])).fit()
        ax_uvb.set_title(f"3{chr(ord('A') + col)} — {form} vs UVB")
        ax_uvb.set_xlabel("UVB (mW·cm$^{-2}$)")
        ax_uvb.set_ylabel("mg·gDW$^{-1}$" if col == 0 else "")
        ax_uvb.text(
            0.02,
            0.94,
            f"slope = {model_uvb.params['p_uvb_mw_cm2']:.3f}\nR² = {model_uvb.rsquared:.3f}\nn = {n}",
            transform=ax_uvb.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        # UVA row
        ax_uva = axes[1, col]
        sns.regplot(
            x="p_uva_mw_cm2",
            y=y_col,
            data=df,
            ax=ax_uva,
            ci=95,
            scatter_kws=dict(s=70, alpha=0.85),
            line_kws=dict(color="black"),
        )
        model_uva = sm.OLS(df[y_col], sm.add_constant(df["p_uva_mw_cm2"])).fit()
        ax_uva.set_title(f"3{chr(ord('D') + col)} — {form} vs UVA")
        ax_uva.set_xlabel("UVA (mW·cm$^{-2}$)")
        ax_uva.set_ylabel("mg·gDW$^{-1}$" if col == 0 else "")
        ax_uva.text(
            0.02,
            0.94,
            f"slope = {model_uva.params['p_uva_mw_cm2']:.3f}\nR² = {model_uva.rsquared:.3f}\nn = {n}",
            transform=ax_uva.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    # Harmonise axes within each row
    set_common_limits(axes[0], "x")
    set_common_limits(axes[1], "x")

    fig.suptitle("Fig. 3 — Chromatogram single-factor trends (mg·gDW⁻¹)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CORE_DIR / "fig03_single_factor_chrom.png", dpi=300)
    plt.close(fig)


def figure04_single_factor_dad() -> None:
    """DAD total concentration vs dose metrics after dry-weight normalisation."""
    df = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")
    response = "predicted_total_mg_per_gDW"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (label, x_col) in zip(axes, [("UVB", "p_uvb_mw_cm2"), ("UVA", "p_uva_mw_cm2")]):
        sns.regplot(
            x=x_col,
            y=response,
            data=df,
            ax=ax,
            ci=95,
            scatter_kws=dict(s=70, alpha=0.85),
            line_kws=dict(color="black"),
        )
        model = sm.OLS(df[response], sm.add_constant(df[x_col])).fit()
        slope = model.params[x_col]
        slope_se = model.bse[x_col]
        r2 = model.rsquared
        ax.set_title(f"4{label[0]} — DAD total vs {label}")
        ax.set_xlabel(f"{label} (mW·cm$^{-2}$)")
        ax.text(
            0.02,
            0.94,
            f"slope = {slope:.3f} ± {slope_se:.3f}\nR² = {r2:.3f}\nn = {len(df)}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    axes[0].set_ylabel("DAD total (mg·gDW$^{-1}$)")
    fig.suptitle(
        "Fig. 4 — DAD single-factor trends (dry-weight normalised)",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CORE_DIR / "fig04_single_factor_dad.png", dpi=300)
    plt.close(fig)


def compute_emms(df: pd.DataFrame, response: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate marginal means over UVA and UVB (categorical)."""
    df = df.copy()
    df["UVA"] = df["p_uva_mw_cm2"].astype("category")
    df["UVB"] = df["p_uvb_mw_cm2"].astype("category")
    model = smf.ols(f"{response} ~ C(UVA) + C(UVB)", data=df).fit()

    emms_uva = (
        df.groupby("UVA")[response]
        .mean()
        .rename("EMM")
        .reset_index()
        .assign(UVA=lambda d: d["UVA"].astype(float))
    )
    emms_uvb = (
        df.groupby("UVB")[response]
        .mean()
        .rename("EMM")
        .reset_index()
        .assign(UVB=lambda d: d["UVB"].astype(float))
    )
    return emms_uva, emms_uvb, model


def figure05_emms() -> None:
    """Estimated marginal means for chromatogram endpoints."""
    chrom_z = pd.read_csv(PROJECT_ROOT / "chromatogram_zscores.csv")
    chrom_z = chrom_z[chrom_z["form"] == "total"]
    chrom_raw = pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv")

    emms_z_uva, emms_z_uvb, model_z = compute_emms(chrom_z, "z_conc_mg_ml")
    emms_raw_uva, emms_raw_uvb, model_raw = compute_emms(
        chrom_raw[["p_uva_mw_cm2", "p_uvb_mw_cm2", "total_mg_per_gDW"]],
        "total_mg_per_gDW",
    )

    anova_z = pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova_zscore.csv")
    z_uva_p = anova_z[
        (anova_z["measurement"] == "z_conc_mg_ml")
        & (anova_z["term"] == "C(p_uva_mw_cm2)")
    ]["PR(>F)"].astype(float).iloc[0]
    z_uvb_p = anova_z[
        (anova_z["measurement"] == "z_conc_mg_ml")
        & (anova_z["term"] == "C(p_uvb_mw_cm2)")
    ]["PR(>F)"].astype(float).iloc[0]

    anova_raw = pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova.csv")
    raw_uva_p = anova_raw[
        (anova_raw["measurement"] == "amount_mg_per_gDW")
        & (anova_raw["term"] == "C(p_uva_mw_cm2)")
    ]["PR(>F)"].astype(float).iloc[0]
    raw_uvb_p = anova_raw[
        (anova_raw["measurement"] == "amount_mg_per_gDW")
        & (anova_raw["term"] == "C(p_uvb_mw_cm2)")
    ]["PR(>F)"].astype(float).iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(emms_z_uva["UVA"], emms_z_uva["EMM"], marker="o")
    axes[0, 0].set_title("5A — Z concentration EMM vs UVA")
    axes[0, 0].set_xlabel("UVA (mW·cm$^{-2}$)")
    axes[0, 0].set_ylabel("EMM (z)")
    axes[0, 0].text(
        0.02,
        0.92,
        f"Classical UVA p = {z_uva_p:.4f}\nClassical UVB p = {z_uvb_p:.4f}",
        transform=axes[0, 0].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    axes[0, 1].plot(emms_z_uvb["UVB"], emms_z_uvb["EMM"], marker="o")
    axes[0, 1].set_title("5B — Z concentration EMM vs UVB")
    axes[0, 1].set_xlabel("UVB (mW·cm$^{-2}$)")
    axes[0, 1].set_ylabel("EMM (z)")
    axes[0, 1].text(
        0.02,
        0.9,
        f"Classical UVA p = {z_uva_p:.4f}\nClassical UVB p = {z_uvb_p:.4f}",
        transform=axes[0, 1].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    axes[1, 0].plot(emms_raw_uva["UVA"], emms_raw_uva["EMM"], marker="o", color="#2ca02c")
    axes[1, 0].set_title("5C — Raw amount EMM vs UVA")
    axes[1, 0].set_xlabel("UVA (mW·cm$^{-2}$)")
    axes[1, 0].set_ylabel("EMM (mg·gDW$^{-1}$)")
    axes[1, 0].text(
        0.02,
        0.88,
        f"Classical UVA p = {raw_uva_p:.4f}\nClassical UVB p = {raw_uvb_p:.4f}\nInteraction p ≈ 0.10–0.11",
        transform=axes[1, 0].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    axes[1, 1].plot(emms_raw_uvb["UVB"], emms_raw_uvb["EMM"], marker="o", color="#d62728")
    axes[1, 1].set_title("5D — Raw amount EMM vs UVB")
    axes[1, 1].set_xlabel("UVB (mW·cm$^{-2}$)")
    axes[1, 1].set_ylabel("EMM (mg·gDW$^{-1}$)")
    axes[1, 1].text(
        0.02,
        0.9,
        f"Classical UVA p = {raw_uva_p:.4f}\nClassical UVB p = {raw_uvb_p:.4f}",
        transform=axes[1, 1].transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.suptitle(
        "Fig. 5 — Estimated marginal means (chromatogram endpoints)\n"
        "Note: DAD Δ total mg·gDW⁻¹ classical p ≈ 0.0575 (borderline).",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(CORE_DIR / "fig05_emm.png", dpi=300)
    plt.close(fig)


def figure06_pvalues() -> None:
    """Classical vs HC3 vs rank interaction p-values."""
    classical_sources = {
        "raw": pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova.csv"),
        "delta": pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova_delta.csv"),
        "zscore": pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova_zscore.csv"),
    }
    robust = pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova_robust.csv")
    rank = pd.read_csv(PROJECT_ROOT / "chromatogram_two_way_anova_rank.csv")

    def get_classical_p(df: pd.DataFrame, measurement: str) -> float:
        mask = (df["measurement"] == measurement) & df["term"].str.contains(
            "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", regex=False
        )
        value = df.loc[mask, "PR(>F)"].astype(float).values
        return float(value[0]) if len(value) else np.nan

    def get_variant_p(df: pd.DataFrame, variant: str, measurement: str, key: str) -> float:
        mask = (df["variant"] == variant) & (df["measurement"] == measurement)
        mask &= df["term"].str.contains("C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", regex=False)
        value = df.loc[mask, key].astype(float).values
        return float(value[0]) if len(value) else np.nan

    records = []
    for variant, measurements in [
        ("raw", ["conc_mg_ml", "amount_mg_per_gDW"]),
        ("delta", ["delta_conc_mg_ml", "delta_amount_mg_per_gDW"]),
        ("zscore", ["z_conc_mg_ml", "z_amount_mg_per_gDW"]),
    ]:
        for measurement in measurements:
            records.extend(
                [
                    {
                        "variant": variant,
                        "measurement": measurement,
                        "method": "Classical",
                        "p": get_classical_p(classical_sources[variant], measurement),
                    },
                    {
                        "variant": variant,
                        "measurement": measurement,
                        "method": "HC3",
                        "p": get_variant_p(robust, variant, measurement, "PR(>F)"),
                    },
                    {
                        "variant": variant,
                        "measurement": measurement,
                        "method": "Rank",
                        "p": get_variant_p(rank, variant, measurement, "PR(>F)"),
                    },
                ]
            )
    plot_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, variant in zip(axes, ["raw", "delta", "zscore"]):
        subset = plot_df[plot_df["variant"] == variant].copy()
        subset["neglogp"] = -np.log10(subset["p"])
        sns.barplot(data=subset, x="measurement", y="neglogp", hue="method", ax=ax)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
        ax.set_title(f"6{variant[0].upper()} — {variant} endpoints")
        ax.set_ylabel("-log10(p)" if ax is axes[0] else "")
        ax.set_xlabel("Measurement")
        ax.tick_params(axis="x", rotation=35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Fig. 6 — Interaction p-values (classical vs robust variants)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.95, 0.94])
    fig.savefig(CORE_DIR / "fig06_pvals_methods.png", dpi=300)
    plt.close(fig)


def figure07_freedman_lane() -> None:
    """Freedman–Lane permutation null distributions with observed F."""
    configs = [
        {
            "csv": "chromatogram_delta.csv",
            "column": "delta_conc_mg_ml",
            "form": "total",
            "title": "Chrom Δ concentration",
        },
        {
            "csv": "chromatogram_delta.csv",
            "column": "delta_amount_mg_per_gDW",
            "form": "total",
            "title": "Chrom Δ amount",
        },
        {
            "csv": "chromatogram_zscores.csv",
            "column": "z_conc_mg_ml",
            "form": "total",
            "title": "Chrom z concentration",
        },
        {
            "csv": "chromatogram_zscores.csv",
            "column": "z_amount_mg_per_gDW",
            "form": "total",
            "title": "Chrom z amount",
        },
        {
            "csv": "DAD_derived_concentrations_corrected.csv",
            "column": "predicted_total_mg_per_gDW",
            "form": None,
            "title": "DAD total (mg·gDW$^{-1}$)",
        },
    ]
    letters = "ABCDE"
    n_cols = 2
    n_rows = (len(configs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.6 * n_cols, 3.6 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes = axes.flatten()

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        df = pd.read_csv(PROJECT_ROOT / cfg["csv"])
        if cfg.get("form"):
            df = df[df["form"] == cfg["form"]].copy()
        df = df.reset_index(drop=True)

        rng = np.random.default_rng(20240 + idx)
        f_obs, f_perm, p_value = freedman_lane_categorical(
            df,
            cfg["column"],
            n_perm=2000,
            rng=rng,
        )

        if f_perm.size == 0 or np.isnan(f_obs):
            ax.text(
                0.5,
                0.5,
                "Permutation failed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        sns.histplot(
            f_perm,
            bins=30,
            ax=ax,
            color="#4C72B0",
            edgecolor="white",
        )
        ax.axvline(
            f_obs,
            color="#DD8452",
            linewidth=2,
            label=f"Observed F = {f_obs:.2f}",
        )
        ax.set_xlabel("Permutation F-statistic")
        ax.set_ylabel("Count")
        ax.set_title(
            f"7{letters[idx]} — {cfg['title']}\n$p_{{FL}} = {p_value:.3f}$"
        )
        ax.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax)

    for ax in axes[len(configs) :]:
        ax.axis("off")

    fig.suptitle("Fig. 7 — Freedman–Lane permutation nulls", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CORE_DIR / "fig07_freedman_lane.png", dpi=300)
    plt.close(fig)


def figure08_predictive_value() -> None:
    """Predictive performance (ridge & PLS) and observed vs fitted scatter."""
    ridge = pd.read_csv(PROJECT_ROOT / "chromatogram_ridge_results.csv")
    ridge = ridge[
        (ridge["variant"].isin(["delta", "zscore"])) & (ridge["term"] == "p_uvb_mw_cm2")
    ][["variant", "measurement", "r_squared"]].drop_duplicates()
    ridge = ridge[ridge["measurement"].isin(["delta_amount_mg_per_gDW", "z_amount_mg_per_gDW"])]
    ridge_total = pd.read_csv(PROJECT_ROOT / "dad_ridge_results.csv")
    ridge_total = ridge_total[
        (ridge_total["variant"] == "raw")
        & (ridge_total["measurement"] == "predicted_total_mg_per_gDW")
        & (ridge_total["term"] == "p_uvb_mw_cm2")
    ][["measurement", "r_squared"]].drop_duplicates()
    ridge_total["variant"] = "raw"
    ridge = pd.concat([ridge, ridge_total], ignore_index=True)

    pls = pd.read_csv(PROJECT_ROOT / "chromatogram_pls_results.csv")
    pls = pls[
        (pls["variant"].isin(["delta", "zscore", "raw"]))
        & (pls["term"] == "p_uvb_mw_cm2")
    ][["variant", "measurement", "cv_r_squared"]].drop_duplicates()
    pls = pls[
        (
            (pls["variant"] == "delta") & (pls["measurement"] == "delta_amount_mg_per_gDW")
        )
        | ((pls["variant"] == "zscore") & (pls["measurement"] == "z_amount_mg_per_gDW"))
        | ((pls["variant"] == "raw") & (pls["measurement"] == "conc_mg_ml"))
    ]
    pls_dad = pd.read_csv(PROJECT_ROOT / "dad_pls_results.csv")
    pls_dad = pls_dad[
        (pls_dad["variant"] == "raw")
        & (pls_dad["measurement"] == "predicted_total_mg_per_gDW")
        & (pls_dad["term"] == "p_uvb_mw_cm2")
    ][["variant", "measurement", "cv_r_squared"]].drop_duplicates()
    pls = pd.concat([pls, pls_dad], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.barplot(data=ridge, x="measurement", y="r_squared", hue="variant", ax=axes[0, 0])
    axes[0, 0].set_title("8A — Ridge R²")
    axes[0, 0].set_ylabel("R²")
    axes[0, 0].tick_params(axis="x", rotation=35)

    sns.barplot(data=pls, x="measurement", y="cv_r_squared", hue="variant", ax=axes[0, 1])
    axes[0, 1].set_title("8B — PLS CV R²")
    axes[0, 1].set_ylabel("CV R²")
    axes[0, 1].tick_params(axis="x", rotation=35)

    chrom = pd.read_csv(PROJECT_ROOT / "chromatogram_delta.csv")
    chrom_total = chrom[chrom["form"] == "total"]
    chrom_X = sm.add_constant(
        chrom_total[["p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    )
    chrom_X["interaction"] = chrom_total["p_uva_mw_cm2"] * chrom_total["p_uvb_mw_cm2"]
    chrom_model = sm.OLS(chrom_total["delta_amount_mg_per_gDW"], chrom_X).fit()
    chrom_preds = chrom_model.fittedvalues
    axes[1, 0].scatter(
        chrom_total["delta_amount_mg_per_gDW"],
        chrom_preds,
        s=65,
        alpha=0.85,
        color="#1f77b4",
    )
    lims = [
        min(chrom_total["delta_amount_mg_per_gDW"].min(), chrom_preds.min()),
        max(chrom_total["delta_amount_mg_per_gDW"].max(), chrom_preds.max()),
    ]
    axes[1, 0].plot(lims, lims, color="black", linestyle="--")
    axes[1, 0].set_title("8C — Chrom Δ-amount (OLS proxy)")
    axes[1, 0].set_xlabel("Observed")
    axes[1, 0].set_ylabel("Predicted")

    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")
    dad_X = sm.add_constant(dad[["p_uva_mw_cm2", "p_uvb_mw_cm2"]])
    dad_X["interaction"] = dad["p_uva_mw_cm2"] * dad["p_uvb_mw_cm2"]
    dad_model = sm.OLS(dad["predicted_total_mg_per_gDW"], dad_X).fit()
    dad_preds = dad_model.fittedvalues
    axes[1, 1].scatter(dad["predicted_total_mg_per_gDW"], dad_preds, s=65, color="#d62728", alpha=0.85)
    lims = [
        min(dad["predicted_total_mg_per_gDW"].min(), dad_preds.min()),
        max(dad["predicted_total_mg_per_gDW"].max(), dad_preds.max()),
    ]
    axes[1, 1].plot(lims, lims, color="black", linestyle="--")
    axes[1, 1].set_title("8D — DAD total mg·gDW$^{-1}$ (OLS proxy)")
    axes[1, 1].set_xlabel("Observed")
    axes[1, 1].set_ylabel("Predicted")

    fig.suptitle("Fig. 8 — Predictive value is low (ridge & PLS)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CORE_DIR / "fig08_predictive_value.png", dpi=300)
    plt.close(fig)


def figure09_bootstrap() -> None:
    """Bootstrap coefficient stability (Δ amount model)."""
    df = pd.read_csv(PROJECT_ROOT / "ridge_bootstrap_summary.csv")
    delta = df[(df["variant"] == "delta") & (df["measurement"] == "delta_amount_mg_per_gDW")]
    beta_uvb = delta[delta["term"] == "p_uvb_mw_cm2"].iloc[0]
    beta_inter = delta[delta["term"] == "p_uva_mw_cm2:p_uvb_mw_cm2"].iloc[0]

    samples_uvb = np.random.default_rng(42).normal(
        beta_uvb["coef_mean"], beta_uvb["coef_std"], size=10000
    )
    samples_inter = np.random.default_rng(43).normal(
        beta_inter["coef_mean"], beta_inter["coef_std"], size=10000
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, samples, row, title in [
        (axes[0], samples_uvb, beta_uvb, "β_UVB (Δ amount)"),
        (axes[1], samples_inter, beta_inter, "Interaction (Δ amount)"),
    ]:
        sns.kdeplot(samples, ax=ax, color="#1f77b4")
        ax.axvline(0, color="black", linestyle="--")
        ax.axvline(row["coef_p2_5"], color="#d62728", linestyle=":")
        ax.axvline(row["coef_p97_5"], color="#d62728", linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Coefficient value (mg·gDW$^{-1}$·(mW·cm$^{-2}$)$^{-1}$)")
        ax.text(
            0.05,
            0.9,
            f"mean = {row['coef_mean']:.3f}\n95% CI [{row['coef_p2_5']:.3f}, {row['coef_p97_5']:.3f}]",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    fig.suptitle("Fig. 9 — Bootstrap stability of key coefficients", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CORE_DIR / "fig09_bootstrap_coeffs.png", dpi=300)
    plt.close(fig)


def figure10_outlier_path() -> None:
    """Outlier removal sensitivity (Δ concentration interaction F and p)."""
    df = pd.read_csv(PROJECT_ROOT / "outlier_sensitivity.csv")
    subset = df[(df["variant"] == "delta") & (df["measurement"] == "delta_conc_mg_ml")].copy()
    subset["n_removed"] = subset["n_removed"].fillna(0).astype(int)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(subset["n_removed"], subset["F"], marker="o", color="#1f77b4", label="F statistic")
    ax1.set_xlabel("Number of removed high residuals")
    ax1.set_ylabel("Interaction F", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(subset["n_removed"], subset["p_value"], marker="s", color="#d62728", label="p-value")
    ax2.set_ylabel("p-value", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Fig. 10 — Outlier sensitivity (Δ concentration interaction)", fontsize=18)
    fig.tight_layout()
    fig.savefig(CORE_DIR / "fig10_outlier_path.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Supplemental figures
# --------------------------------------------------------------------------- #

def diagnostic_triptych(
    data: pd.DataFrame,
    formula: str,
    label: str,
    outfile: Path,
) -> None:
    """Residual vs fitted, QQ plot, leverage-residual diagnostics."""
    model = smf.ols(formula, data=data).fit()
    influence = OLSInfluence(model)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Residuals vs fitted
    axes[0].scatter(model.fittedvalues, model.resid, s=60, alpha=0.85)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{label}: Residual vs fitted")

    # QQ plot
    qqplot(model.resid, line="45", ax=axes[1], alpha=0.8, markerfacecolor="#1f77b4")
    axes[1].set_title(f"{label}: Q–Q plot")

    # Leverage vs studentized residuals
    axes[2].scatter(influence.hat_matrix_diag, influence.resid_studentized_internal, s=60, alpha=0.85)
    axes[2].axhline(0, color="black", linestyle="--")
    axes[2].axhline(2, color="#d62728", linestyle=":", linewidth=1)
    axes[2].axhline(-2, color="#d62728", linestyle=":", linewidth=1)
    axes[2].axvline(2 * (model.df_model + 1) / len(data), color="#ff7f0e", linestyle="--")
    axes[2].set_xlabel("Leverage")
    axes[2].set_ylabel("Studentized residual")
    axes[2].set_title(f"{label}: Leverage diagnostic")

    fig.suptitle(f"{label} diagnostics (n={len(data)})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def supplemental_diagnostics() -> None:
    """Build Fig. S1–S6 diagnostics."""
    diagnostics_specs = [
        (
            pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv"),
            "conc_total_mg_ml ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S1 — Chrom raw concentration",
            SUPP_DIR / "figS01_chrom_raw_conc.png",
        ),
        (
            pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv"),
            "total_mg_per_gDW ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S2 — Chrom raw amount",
            SUPP_DIR / "figS02_chrom_raw_amount.png",
        ),
        (
            pd.read_csv(PROJECT_ROOT / "chromatogram_delta.csv").query("form == 'total'"),
            "delta_conc_mg_ml ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S3 — Chrom Δ concentration",
            SUPP_DIR / "figS03_chrom_delta_conc.png",
        ),
        (
            pd.read_csv(PROJECT_ROOT / "chromatogram_delta.csv").query("form == 'total'"),
            "delta_amount_mg_per_gDW ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S4 — Chrom Δ amount",
            SUPP_DIR / "figS04_chrom_delta_amount.png",
        ),
        (
            pd.read_csv(PROJECT_ROOT / "chromatogram_zscores.csv").query("form == 'total'"),
            "z_conc_mg_ml ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S5 — Chrom z concentration",
            SUPP_DIR / "figS05_chrom_z_conc.png",
        ),
        (
            pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv"),
            "predicted_total_mg_per_gDW ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            "Fig. S6 — DAD total mg·gDW⁻¹",
            SUPP_DIR / "figS06_dad_total_mg_per_gdw.png",
        ),
    ]

    for data, formula, label, outfile in diagnostics_specs:
        data = data.copy()
        data["p_uva_mw_cm2"] = data["p_uva_mw_cm2"].astype("category")
        data["p_uvb_mw_cm2"] = data["p_uvb_mw_cm2"].astype("category")
        diagnostic_triptych(data, formula, label, outfile)


def supplemental_cross_assay_alignment() -> None:
    """Fig. S7 — Chromatogram vs DAD concentration alignment."""
    chrom = pd.read_csv(PROJECT_ROOT / "Chromatogram_derived_concentrations.csv")
    dad = pd.read_csv(PROJECT_ROOT / "DAD_derived_concentrations_corrected.csv")

    merged = chrom.merge(
        dad[["sample_id", "predicted_total_mg_ml"]],
        on="sample_id",
        suffixes=("_chrom", "_dad"),
    )
    merged["chrom_mg_ml"] = merged["total_mg_per_gDW"] * merged["dry_biomass_g"]  # V_extract ≈ 1 mL

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["chrom_mg_ml"], merged["predicted_total_mg_ml"], s=70, alpha=0.85)
    lims = [
        min(merged["chrom_mg_ml"].min(), merged["predicted_total_mg_ml"].min()),
        max(merged["chrom_mg_ml"].max(), merged["predicted_total_mg_ml"].max()),
    ]
    ax.plot(lims, lims, color="black", linestyle="--")
    ax.set_xlabel("Chromatogram (mg·mL$^{-1}$)")
    ax.set_ylabel("DAD (mg·mL$^{-1}$)")
    r = merged[["chrom_mg_ml", "predicted_total_mg_ml"]].corr().iloc[0, 1]
    ax.text(
        0.05,
        0.9,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    fig.suptitle("Fig. S7 — Cross-assay concentration alignment", fontsize=16)
    fig.tight_layout()
    fig.savefig(SUPP_DIR / "figS07_cross_assay_alignment.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    figure01_calibrations()
    figure02_dose_structure()
    figure03_single_factor_chrom()
    figure04_single_factor_dad()
    figure05_emms()
    figure06_pvalues()
    figure07_freedman_lane()
    figure08_predictive_value()
    figure09_bootstrap()
    figure10_outlier_path()

    supplemental_diagnostics()
    supplemental_cross_assay_alignment()


if __name__ == "__main__":
    main()
