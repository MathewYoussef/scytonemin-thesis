#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ValueWarning

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

ALPHAS = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
N_PERMUTATIONS = 2000
RANDOM_SEED = 42


def ensure_uv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"p_uva_mw_cm2", "p_uvb_mw_cm2"}.issubset(df.columns):
        return df
    truth_df = pd.read_csv(TRUTH_PATH)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    return df.merge(truth_df, on="sample_id", how="left")


def load_dataset(dataset: str, variant: str) -> pd.DataFrame:
    if dataset == "chromatogram":
        df = (
            load_chromatogram(REPO_ROOT)
            if variant == "raw"
            else pd.read_csv(CHROM_PATHS[variant])
        )
    elif dataset == "dad":
        df = load_dad(REPO_ROOT) if variant == "raw" else pd.read_csv(DAD_PATHS[variant])
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


def build_design(df: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    uva = df["p_uva_mw_cm2"].to_numpy(dtype=float)
    uvb = df["p_uvb_mw_cm2"].to_numpy(dtype=float)
    interaction = uva * uvb
    X = np.column_stack([uva, uvb, interaction])
    feature_names = ["p_uva_mw_cm2", "p_uvb_mw_cm2", "p_uva_mw_cm2:p_uvb_mw_cm2"]
    return X, interaction, feature_names


def fit_ridge(X: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, object]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", RidgeCV(alphas=ALPHAS, store_cv_results=True)),
        ]
    )
    pipeline.fit(X, y)
    ridge = pipeline.named_steps["ridge"]
    scaler = pipeline.named_steps["scaler"]

    coef_scaled = ridge.coef_
    intercept_scaled = ridge.intercept_
    coef_orig = coef_scaled / scaler.scale_
    intercept_orig = intercept_scaled - np.dot(coef_orig, scaler.mean_)

    predictions = pipeline.predict(X)
    r_squared = 1.0 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)

    return {
        "alpha": float(ridge.alpha_),
        "coef_scaled": coef_scaled,
        "intercept_scaled": float(intercept_scaled),
        "coef_orig": coef_orig,
        "intercept_orig": float(intercept_orig),
        "r_squared": float(r_squared),
    }


def fit_pls(X: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, object]:
    n_components = min(2, X.shape[1], X.shape[0] - 1)
    if n_components < 1:
        raise ValueError("PLS requires at least two observations.")
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(X, y)

    y_pred = pls.predict(X).ravel()
    r_squared = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    k = min(5, X.shape[0])
    if k < 2:
        cv_r2 = np.nan
    else:
        cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        y_cv = cross_val_predict(pls, X, y, cv=cv).ravel()
        cv_r2 = 1.0 - np.sum((y - y_cv) ** 2) / np.sum((y - y.mean()) ** 2)

    return {
        "n_components": n_components,
        "x_weights": pls.x_weights_,
        "x_loadings": pls.x_loadings_,
        "y_loadings": pls.y_loadings_,
        "r_squared": float(r_squared),
        "cv_r_squared": float(cv_r2),
    }


def anova_formula(metric: str) -> str:
    return (
        f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )


def compute_permutation_pvalues(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    formula = anova_formula(metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ValueWarning)
        model = ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)
    observed_f = table["F"].to_dict()

    rng = np.random.default_rng(RANDOM_SEED)
    f_exceed_counts = {term: 1 for term in observed_f}

    y_original = df[metric].to_numpy(copy=True)

    for _ in range(N_PERMUTATIONS):
        rng.shuffle(y_original)
        df["_perm_metric"] = y_original
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            perm_model = ols(
                anova_formula("_perm_metric"),
                data=df,
            ).fit()
            perm_table = anova_lm(perm_model, typ=2)
        for term in observed_f:
            perm_f = perm_table.loc[term, "F"]
            if np.isnan(perm_f):
                continue
            if perm_f >= observed_f[term]:
                f_exceed_counts[term] += 1

    results = []
    for term, f_value in observed_f.items():
        p_perm = f_exceed_counts[term] / (N_PERMUTATIONS + 1)
        results.append({"term": term, "F_obs": f_value, "p_perm": p_perm})

    return pd.DataFrame(results)


def linearize_results(
    spec: ModelSpec,
    feature_names: List[str],
    ridge_result: dict[str, object],
    pls_result: dict[str, object],
) -> Tuple[List[dict[str, object]], List[dict[str, object]]]:
    ridge_rows = []
    for name, coef_scaled, coef_orig in zip(
        feature_names,
        ridge_result["coef_scaled"],
        ridge_result["coef_orig"],
        strict=True,
    ):
        ridge_rows.append(
            {
                "dataset": spec.dataset,
                "variant": spec.variant,
                "measurement": spec.metric,
                "term": name,
                "alpha": ridge_result["alpha"],
                "coef_scaled": float(coef_scaled),
                "coef_orig": float(coef_orig),
                "intercept_orig": ridge_result["intercept_orig"],
                "r_squared": ridge_result["r_squared"],
            }
        )

    pls_rows = []
    x_weights = pls_result["x_weights"]
    x_loadings = pls_result["x_loadings"]
    y_loadings = pls_result["y_loadings"]
    n_components = x_weights.shape[1]

    for feat_idx, name in enumerate(feature_names):
        for comp_idx in range(n_components):
            pls_rows.append(
                {
                    "dataset": spec.dataset,
                    "variant": spec.variant,
                    "measurement": spec.metric,
                    "term": name,
                    "component": comp_idx + 1,
                    "n_components": pls_result["n_components"],
                    "x_weight": float(x_weights[feat_idx, comp_idx]),
                    "x_loading": float(x_loadings[feat_idx, comp_idx]),
                    "y_loading": float(y_loadings[0, comp_idx]),
                    "r_squared": pls_result["r_squared"],
                    "cv_r_squared": pls_result["cv_r_squared"],
                }
            )

    return ridge_rows, pls_rows


def process_specs(specs: Iterable[ModelSpec]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ridge_records: list[dict[str, object]] = []
    pls_records: list[dict[str, object]] = []
    perm_records: list[dict[str, object]] = []

    for spec in specs:
        df = load_dataset(spec.dataset, spec.variant)
        clean = dropna_for_metric(df, spec.metric)
        if clean.empty:
            continue

        X, _, feature_names = build_design(clean)
        y = clean[spec.metric].to_numpy(dtype=float)

        ridge_result = fit_ridge(X, y)
        pls_result = fit_pls(X, y)
        ridge_rows, pls_rows = linearize_results(spec, feature_names, ridge_result, pls_result)
        ridge_records.extend(ridge_rows)
        pls_records.extend(pls_rows)

        perm_df = compute_permutation_pvalues(clean, spec.metric)
        perm_df.insert(0, "measurement", spec.metric)
        perm_df.insert(0, "variant", spec.variant)
        perm_df.insert(0, "dataset", spec.dataset)
        perm_df["n_permutations"] = N_PERMUTATIONS
        perm_records.extend(perm_df.to_dict("records"))

    ridge_df = pd.DataFrame.from_records(ridge_records) if ridge_records else pd.DataFrame()
    pls_df = pd.DataFrame.from_records(pls_records) if pls_records else pd.DataFrame()
    perm_df = pd.DataFrame.from_records(perm_records) if perm_records else pd.DataFrame()
    return ridge_df, pls_df, perm_df


def main() -> None:
    ridge_chrom, pls_chrom, perm_chrom = process_specs(CHROM_SPECS)
    ridge_dad, pls_dad, perm_dad = process_specs(DAD_SPECS)

    out_dir = REPO_ROOT / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ridge_chrom.empty:
        ridge_chrom.to_csv(out_dir / "chromatogram_ridge_results.csv", index=False)
    if not pls_chrom.empty:
        pls_chrom.to_csv(out_dir / "chromatogram_pls_results.csv", index=False)
    if not perm_chrom.empty:
        perm_chrom.to_csv(out_dir / "chromatogram_two_way_anova_permutation.csv", index=False)

    if not ridge_dad.empty:
        ridge_dad.to_csv(out_dir / "dad_ridge_results.csv", index=False)
    if not pls_dad.empty:
        pls_dad.to_csv(out_dir / "dad_pls_results.csv", index=False)
    if not perm_dad.empty:
        perm_dad.to_csv(out_dir / "dad_two_way_anova_permutation.csv", index=False)

    print("Complementary modeling complete.")


if __name__ == "__main__":
    main()
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
