#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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


def anova_formula(metric: str) -> str:
    return (
        f"{metric} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + "
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    )


def run_hc3_anova(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    model = ols(anova_formula(spec.metric), data=df).fit()
    table = anova_lm(model, typ=2, robust="hc3")
    table = table.reset_index().rename(columns={"index": "term", "Pr(>F)": "PR(>F)"})
    table.insert(0, "measurement", spec.metric)
    table.insert(0, "variant", spec.variant)
    table.insert(0, "dataset", spec.dataset)
    return table


def run_rank_anova(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    rank_col = "__rank_metric"
    df = df.copy()
    df[rank_col] = rankdata(df[spec.metric])
    model = ols(anova_formula(rank_col), data=df).fit()
    table = anova_lm(model, typ=2)
    table = table.reset_index().rename(columns={"index": "term", "Pr(>F)": "PR(>F)"})
    table.insert(0, "measurement", spec.metric)
    table.insert(0, "variant", spec.variant)
    table.insert(0, "dataset", spec.dataset)
    return table


def process_specs(specs: Iterable[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust_tables: list[pd.DataFrame] = []
    rank_tables: list[pd.DataFrame] = []

    for spec in specs:
        df = load_dataset(spec.dataset, spec.variant)
        clean = dropna_for_metric(df, spec.metric)
        if clean.empty:
            continue
        robust_tables.append(run_hc3_anova(clean, spec))
        rank_tables.append(run_rank_anova(clean, spec))

    robust_df = pd.concat(robust_tables, ignore_index=True) if robust_tables else pd.DataFrame()
    rank_df = pd.concat(rank_tables, ignore_index=True) if rank_tables else pd.DataFrame()
    return robust_df, rank_df


def main() -> None:
    robust_chrom, rank_chrom = process_specs(CHROM_SPECS)
    robust_dad, rank_dad = process_specs(DAD_SPECS)

    out_dir = REPO_ROOT / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not robust_chrom.empty:
        robust_chrom.to_csv(out_dir / "chromatogram_two_way_anova_robust.csv", index=False)
    if not rank_chrom.empty:
        rank_chrom.to_csv(out_dir / "chromatogram_two_way_anova_rank.csv", index=False)
    if not robust_dad.empty:
        robust_dad.to_csv(out_dir / "dad_two_way_anova_robust.csv", index=False)
    if not rank_dad.empty:
        rank_dad.to_csv(out_dir / "dad_two_way_anova_rank.csv", index=False)

    print("Robust and rank-based ANOVA complete.")


if __name__ == "__main__":
    main()
