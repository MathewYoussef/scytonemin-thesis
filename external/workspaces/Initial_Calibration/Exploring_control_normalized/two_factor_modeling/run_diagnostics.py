#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import load_chromatogram, load_dad  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUTH_PATH = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"


@dataclass(frozen=True)
class ModelSpec:
    dataset: str
    variant: str
    metric: str


CHROM_SPECS: List[ModelSpec] = [
    ModelSpec("chromatogram", "raw", "conc_mg_ml"),
    ModelSpec("chromatogram", "raw", "amount_mg_per_gDW"),
    ModelSpec("chromatogram", "delta", "delta_conc_mg_ml"),
    ModelSpec("chromatogram", "delta", "delta_amount_mg_per_gDW"),
    ModelSpec("chromatogram", "zscore", "z_conc_mg_ml"),
    ModelSpec("chromatogram", "zscore", "z_amount_mg_per_gDW"),
]

CHROM_PATHS = {
    "raw": None,
    "delta": REPO_ROOT
    / "Exploring_control_normalized"
    / "delta_from_control"
    / "chromatogram_delta.csv",
    "zscore": REPO_ROOT
    / "Exploring_control_normalized"
    / "zscore_using_control"
    / "chromatogram_zscores.csv",
}

DAD_SPECS: List[ModelSpec] = [
    ModelSpec("dad", "raw", "predicted_total_mg_per_gDW"),
    ModelSpec("dad", "delta", "predicted_total_mg_per_gDW_delta"),
]

DAD_PATHS = {
    "raw": None,
    "delta": REPO_ROOT
    / "Exploring_control_normalized"
    / "delta_from_control"
    / "dad_delta.csv",
}


def ensure_uv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"p_uva_mw_cm2", "p_uvb_mw_cm2"}.issubset(df.columns):
        return df
    truth_df = pd.read_csv(TRUTH_PATH)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    return df.merge(truth_df, on="sample_id", how="left")


def load_dataset(dataset: str, variant: str) -> pd.DataFrame:
    if dataset == "chromatogram":
        if variant == "raw":
            df = load_chromatogram(REPO_ROOT)
        else:
            df = pd.read_csv(CHROM_PATHS[variant])
    elif dataset == "dad":
        if variant == "raw":
            df = load_dad(REPO_ROOT)
        else:
            df = pd.read_csv(DAD_PATHS[variant])
    else:
        msg = f"Unsupported dataset '{dataset}'"
        raise ValueError(msg)
    return ensure_uv_columns(df)


def dropna_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = [metric, "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"Missing columns {missing} for metric '{metric}'"
        raise KeyError(msg)
    clean = df.dropna(subset=cols).copy()
    return clean


def fit_two_way_anova(df: pd.DataFrame, metric: str):
    formula = (
        f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )
    model = smf.ols(formula, data=df).fit()
    return model


def fit_continuous_regression(df: pd.DataFrame, metric: str):
    formula = f"{metric} ~ p_uva_mw_cm2 + p_uvb_mw_cm2 + p_uva_mw_cm2:p_uvb_mw_cm2"
    model = smf.ols(formula, data=df).fit()
    return model


def compute_anova_diagnostics(
    model, dataset: str, variant: str, metric: str
) -> dict[str, float]:
    resid = model.resid
    n_obs = resid.shape[0]
    shapiro_p = np.nan
    if 3 <= n_obs <= 5000:
        shapiro_p = stats.shapiro(resid)[1]
    jb_result = stats.jarque_bera(resid)
    if isinstance(jb_result, (tuple, list)):
        jb_stat, jb_p = jb_result[0], jb_result[1]
    else:
        jb_stat = getattr(jb_result, "statistic", np.nan)
        jb_p = getattr(jb_result, "pvalue", np.nan)
    influence = model.get_influence()
    hat = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    studentized = influence.resid_studentized_internal
    leverage_threshold = 2 * (model.df_model + 1) / n_obs if n_obs else np.nan
    return {
        "dataset": dataset,
        "variant": variant,
        "metric": metric,
        "n_obs": n_obs,
        "shapiro_p": shapiro_p,
        "jarque_bera": jb_stat,
        "jarque_bera_p": jb_p,
        "max_abs_studentized_resid": np.nanmax(np.abs(studentized)),
        "max_leverage": np.nanmax(hat),
        "leverage_threshold": leverage_threshold,
        "n_high_leverage": int(np.sum(hat > leverage_threshold)),
        "max_cooks_distance": np.nanmax(cooks),
    }


def compute_vif(model, dataset: str, variant: str, metric: str) -> Iterable[dict]:
    exog = model.model.exog
    names = model.model.exog_names
    for i, name in enumerate(names):
        if name == "Intercept":
            continue
        yield {
            "dataset": dataset,
            "variant": variant,
            "metric": metric,
            "term": name,
            "vif": variance_inflation_factor(exog, i),
        }


def correlation_summary(df: pd.DataFrame, dataset: str, variant: str) -> dict:
    corr = df["p_uva_mw_cm2"].corr(df["p_uvb_mw_cm2"])
    return {
        "dataset": dataset,
        "variant": variant,
        "corr_uva_uvb": corr,
        "n_obs": len(df),
    }


def main() -> None:
    anova_records: list[dict] = []
    vif_records: list[dict] = []
    corr_records: list[dict] = []

    for spec in CHROM_SPECS + DAD_SPECS:
        df = load_dataset(spec.dataset, spec.variant)
        clean = dropna_for_metric(df, spec.metric)
        if clean.empty:
            continue

        # two-way ANOVA diagnostics
        aov_model = fit_two_way_anova(clean, spec.metric)
        anova_records.append(
            compute_anova_diagnostics(aov_model, spec.dataset, spec.variant, spec.metric)
        )

        # continuous regression collinearity checks
        reg_model = fit_continuous_regression(clean, spec.metric)
        vif_records.extend(
            compute_vif(reg_model, spec.dataset, spec.variant, spec.metric)
        )
        corr_records.append(
            correlation_summary(clean, spec.dataset, spec.variant)
        )

    out_dir = REPO_ROOT / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    if anova_records:
        anova_df = pd.DataFrame.from_records(anova_records)
        try:
            anova_df.to_csv(out_dir / "anova_residual_diagnostics.csv", index=False)
        except PermissionError:
            print("=== anova_residual_diagnostics.csv (write blocked) ===")
            print(anova_df.to_csv(index=False))
    if vif_records:
        vif_df = pd.DataFrame.from_records(vif_records)
        try:
            vif_df.to_csv(out_dir / "regression_vif.csv", index=False)
        except PermissionError:
            print("=== regression_vif.csv (write blocked) ===")
            print(vif_df.to_csv(index=False))
    if corr_records:
        corr_df = pd.DataFrame.from_records(corr_records).drop_duplicates()
        try:
            corr_df.to_csv(out_dir / "uva_uvb_correlations.csv", index=False)
        except PermissionError:
            print("=== uva_uvb_correlations.csv (write blocked) ===")
            print(corr_df.to_csv(index=False))

    print("Diagnostics complete.")


if __name__ == "__main__":
    main()
