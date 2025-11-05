#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ValueWarning

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import load_chromatogram  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUTH_PATH = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    metric: str


SPECS = [
    ModelSpec("delta", "delta_conc_mg_ml"),
    ModelSpec("delta", "delta_amount_mg_per_gDW"),
    ModelSpec("zscore", "z_conc_mg_ml"),
    ModelSpec("zscore", "z_amount_mg_per_gDW"),
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


def ensure_uv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"p_uva_mw_cm2", "p_uvb_mw_cm2"}.issubset(df.columns):
        return df
    truth_df = pd.read_csv(TRUTH_PATH)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    return df.merge(truth_df, on="sample_id", how="left")


def load_variant(variant: str) -> pd.DataFrame:
    if variant == "delta":
        df = pd.read_csv(CHROM_PATHS["delta"])
    elif variant == "zscore":
        df = pd.read_csv(CHROM_PATHS["zscore"])
    else:
        df = load_chromatogram(REPO_ROOT)
    return ensure_uv_columns(df)


def clean_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = [metric, "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"Missing {missing} for {metric}"
        raise KeyError(msg)
    return df.dropna(subset=cols).copy()


def formula(metric: str) -> str:
    return (
        f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )


def compute_anova(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ValueWarning)
        model = ols(formula(metric), data=df).fit()
        table = anova_lm(model, typ=2)
    term = "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    return (
        table.loc[term, "F"],
        table.loc[term, "PR(>F)"],
    )


def identify_flagged(df: pd.DataFrame, metric: str, max_flags: int = 5) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ValueWarning)
        model = ols(formula(metric), data=df).fit()
        influence = model.get_influence()
    studentized = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]

    info = pd.DataFrame(
        {
            "studentized": studentized,
            "leverage": leverage,
            "cooks": cooks,
        },
        index=df.index,
    )
    if {"sample_id"}.issubset(df.columns):
        info["sample_id"] = df["sample_id"]
    if {"sample_name"}.issubset(df.columns):
        info["sample_name"] = df["sample_name"]

    info["abs_studentized"] = info["studentized"].abs()
    flagged = info.sort_values("abs_studentized", ascending=False)
    flagged = flagged[flagged["abs_studentized"] >= 3.0].head(max_flags)
    if flagged.empty:
        flagged = info.sort_values("abs_studentized", ascending=False).head(3)
    return flagged


def drop_indices(df: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    return df.drop(index=indices)


def main() -> None:
    results = []

    for spec in SPECS:
        df = clean_df(load_variant(spec.variant), spec.metric)
        if df.empty:
            continue

        base_F, base_p = compute_anova(df, spec.metric)
        flagged = identify_flagged(df, spec.metric)

        results.append(
            {
                "variant": spec.variant,
                "measurement": spec.metric,
                "scenario": "baseline",
                "removed": "",
                "n_removed": 0,
                "F": base_F,
                "p_value": base_p,
            }
        )

        for idx, row in flagged.iterrows():
            label = row.get("sample_id", row.get("sample_name", str(idx)))
            reduced_df = drop_indices(df, [idx])
            F_val, p_val = compute_anova(reduced_df, spec.metric)
            results.append(
                {
                    "variant": spec.variant,
                    "measurement": spec.metric,
                    "scenario": "drop_single",
                    "removed": str(label),
                    "n_removed": 1,
                    "F": F_val,
                    "p_value": p_val,
                }
            )

        # drop all flagged together
        reduced_df = drop_indices(df, flagged.index)
        F_val, p_val = compute_anova(reduced_df, spec.metric)
        results.append(
            {
                "variant": spec.variant,
                "measurement": spec.metric,
                "scenario": "drop_flagged_set",
                "removed": ";".join(
                    [str(r.get("sample_id", r.get("sample_name", idx))) for idx, r in flagged.iterrows()]
                ),
                "n_removed": len(flagged),
                "F": F_val,
                "p_value": p_val,
            }
        )

    out_dir = REPO_ROOT / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(results).to_csv(
        out_dir / "outlier_sensitivity.csv",
        index=False,
    )
    print("Outlier sensitivity analyses complete.")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=ValueWarning)
    main()
