#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import load_chromatogram, load_dad  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUTH_PATH = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"

N_PERM = 2000
N_BOOT = 2000
RANDOM_SEED = 12345


@dataclass(frozen=True)
class ModelSpec:
    dataset: str
    variant: str
    metric: str


CHROM_SPECS = [
    ModelSpec("chromatogram", "delta", "delta_conc_mg_ml"),
    ModelSpec("chromatogram", "delta", "delta_amount_mg_per_gDW"),
    ModelSpec("chromatogram", "zscore", "z_conc_mg_ml"),
    ModelSpec("chromatogram", "zscore", "z_amount_mg_per_gDW"),
]

DAD_SPECS = [
    ModelSpec("dad", "raw", "predicted_total_mg_per_gDW"),
]

CHROM_PATHS = {
    "delta": REPO_ROOT
    / "Exploring_control_normalized"
    / "delta_from_control"
    / "chromatogram_delta.csv",
    "zscore": REPO_ROOT
    / "Exploring_control_normalized"
    / "zscore_using_control"
    / "chromatogram_zscores.csv",
}

DAD_PATHS = {
    "raw": None,
}


def ensure_uv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"p_uva_mw_cm2", "p_uvb_mw_cm2"}.issubset(df.columns):
        return df
    truth_df = pd.read_csv(TRUTH_PATH)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    return df.merge(truth_df, on="sample_id", how="left")


def load_dataset(spec: ModelSpec) -> pd.DataFrame:
    if spec.dataset == "chromatogram":
        if spec.variant == "delta":
            df = pd.read_csv(CHROM_PATHS["delta"])
        elif spec.variant == "zscore":
            df = pd.read_csv(CHROM_PATHS["zscore"])
        else:
            df = load_chromatogram(REPO_ROOT)
    else:
        if spec.variant == "raw":
            df = load_dad(REPO_ROOT)
        else:
            df = pd.read_csv(DAD_PATHS[spec.variant])
    return ensure_uv_columns(df)


def clean_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = [metric, "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing columns {missing} for metric {metric}"
        raise KeyError(msg)
    return df.dropna(subset=required).copy()


def full_formula(metric: str) -> str:
    return (
        f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )


def reduced_formula(metric: str) -> str:
    return f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2)"


def freedman_lane_pvalue(
    data: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ValueWarning)
        full_model = ols(full_formula(metric), data=data).fit()
        reduced_model = ols(reduced_formula(metric), data=data).fit()
        observed_table = anova_lm(full_model, typ=2)
    f_obs = observed_table.loc["C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", "F"]

    residuals = reduced_model.resid.to_numpy(copy=True)
    fitted = reduced_model.fittedvalues.to_numpy(copy=True)

    exceed = 1  # include observed
    for _ in range(n_perm):
        rng.shuffle(residuals)
        perm_y = fitted + residuals
        perm_df = data.copy()
        perm_df["_perm_metric"] = perm_y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ValueWarning)
            perm_model = ols(
                full_formula("_perm_metric"),
                data=perm_df,
            ).fit()
            perm_table = anova_lm(perm_model, typ=2)
        perm_f = perm_table.loc["C(p_uva_mw_cm2):C(p_uvb_mw_cm2)", "F"]
        if np.isnan(perm_f):
            continue
        if perm_f >= f_obs:
            exceed += 1
    p_value = exceed / (n_perm + 1)
    return f_obs, p_value


def build_design_matrix(df: pd.DataFrame) -> NDArray[np.float64]:
    uva = df["p_uva_mw_cm2"].to_numpy(dtype=float)
    uvb = df["p_uvb_mw_cm2"].to_numpy(dtype=float)
    interaction = uva * uvb
    return np.column_stack([uva, uvb, interaction])


def bootstrap_ridge(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    n_samples = X.shape[0]
    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    pipeline = Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    records: list[dict[str, float]] = []

    # determine alpha using full data with CV-style search
    best_alpha = select_best_alpha(X, y, alphas)

    for _ in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot = X[idx]
        y_boot = y[idx]
        pipeline.named_steps["ridge"].set_params(alpha=best_alpha)
        pipeline.fit(X_boot, y_boot)
        scaler = pipeline.named_steps["scale"]
        ridge = pipeline.named_steps["ridge"]
        coef_scaled = ridge.coef_
        intercept_scaled = ridge.intercept_
        coef_orig = coef_scaled / scaler.scale_
        intercept_orig = intercept_scaled - np.dot(coef_orig, scaler.mean_)
        for name, coef in zip(
            ["p_uva_mw_cm2", "p_uvb_mw_cm2", "p_uva_mw_cm2:p_uvb_mw_cm2"],
            coef_orig,
            strict=True,
        ):
            records.append(
                {
                    "term": name,
                    "coef": float(coef),
                    "intercept": float(intercept_orig),
                    "alpha": float(best_alpha),
                }
            )
    return pd.DataFrame.from_records(records)


def select_best_alpha(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alphas: NDArray[np.float64],
) -> float:
    from sklearn.linear_model import RidgeCV

    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X, y)
    return float(ridge_cv.alpha_)


def summarize_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("term")
        .coef.agg(
            [
                ("coef_median", "median"),
                ("coef_mean", "mean"),
                ("coef_std", "std"),
                ("coef_p2_5", lambda x: np.quantile(x, 0.025)),
                ("coef_p97_5", lambda x: np.quantile(x, 0.975)),
            ]
        )
    )
    summary.reset_index(inplace=True)
    summary["alpha"] = df["alpha"].iloc[0] if not df.empty else np.nan
    return summary


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    perm_records: list[dict[str, float]] = []
    boot_summaries: list[pd.DataFrame] = []

    for spec in CHROM_SPECS + DAD_SPECS:
        df = clean_df(load_dataset(spec), spec.metric)
        if df.empty:
            continue

        f_obs, p_freedman = freedman_lane_pvalue(df, spec.metric, rng)
        perm_records.append(
            {
                "dataset": spec.dataset,
                "variant": spec.variant,
                "measurement": spec.metric,
                "F_obs": f_obs,
                "p_freedman_lane": p_freedman,
                "n_permutations": N_PERM,
            }
        )

        X = build_design_matrix(df)
        y = df[spec.metric].to_numpy(dtype=float)
        boot_df = bootstrap_ridge(X, y, rng)
        summary = summarize_bootstrap(boot_df)
        summary.insert(0, "measurement", spec.metric)
        summary.insert(0, "variant", spec.variant)
        summary.insert(0, "dataset", spec.dataset)
        boot_summaries.append(summary)

    out_dir = REPO_ROOT / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    if perm_records:
        pd.DataFrame.from_records(perm_records).to_csv(
            out_dir / "freedman_lane_interaction.csv", index=False
        )
    if boot_summaries:
        pd.concat(boot_summaries, ignore_index=True).to_csv(
            out_dir / "ridge_bootstrap_summary.csv", index=False
        )

    print("Resampling sensitivity analyses complete.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=ValueWarning)
    main()
