#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import load_chromatogram, load_dad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-factor UVA/UVB models on chromatogram and DAD datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--chrom-file",
        type=Path,
        help="Optional chromatogram CSV (default uses raw data).",
    )
    parser.add_argument(
        "--chrom-metrics",
        type=str,
        default="conc_mg_ml,amount_mg_per_gDW",
        help="Comma-separated chromatogram columns to model.",
    )
    parser.add_argument(
        "--dad-file",
        type=Path,
        help="Optional DAD CSV (default uses raw data).",
    )
    parser.add_argument(
        "--dad-metrics",
        type=str,
        default="predicted_total_mg_ml,predicted_total_mg_per_gDW,"
        "predicted_oxidized_mg_ml,predicted_oxidized_mg_per_gDW,"
        "predicted_reduced_mg_ml,predicted_reduced_mg_per_gDW",
        help="Comma-separated DAD columns to model.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional suffix for output filenames (e.g., 'delta').",
    )
    return parser.parse_args()


def load_custom_or_default(repo: Path, chrom_file: Optional[Path], dad_file: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    chrom_df = load_chromatogram(repo) if chrom_file is None else pd.read_csv(
        chrom_file if chrom_file.is_absolute() else repo / chrom_file
    )
    dad_df = load_dad(repo) if dad_file is None else pd.read_csv(
        dad_file if dad_file.is_absolute() else repo / dad_file
    )

    if "p_uva_mw_cm2" not in chrom_df.columns or "p_uvb_mw_cm2" not in chrom_df.columns:
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        chrom_df = chrom_df.merge(truth_df, on="sample_id", how="left")

    if "p_uva_mw_cm2" not in dad_df.columns or "p_uvb_mw_cm2" not in dad_df.columns:
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        dad_df = dad_df.merge(truth_df, on="sample_id", how="left")

    return chrom_df, dad_df


def run_two_way_anova(df: pd.DataFrame, response: str) -> dict:
    formula = f"{response} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + C(p_uva_mw_cm2):C(p_uvb_mw_cm2)"
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return {
        "formula": formula,
        "anova_table": anova_table,
        "model_summary": model.summary().as_text(),
    }


def summarize_anova(anova_table: pd.DataFrame) -> pd.DataFrame:
    table = anova_table.reset_index().rename(columns={"index": "term"})
    table["eta_squared"] = table["sum_sq"] / table["sum_sq"].sum()
    return table[["term", "sum_sq", "df", "F", "PR(>F)", "eta_squared"]]


def run_continuous_regression(df: pd.DataFrame, response: str) -> dict:
    formula = f"{response} ~ p_uva_mw_cm2 + p_uvb_mw_cm2 + p_uva_mw_cm2:p_uvb_mw_cm2"
    model = smf.ols(formula, data=df).fit()
    return {
        "formula": formula,
        "summary": model.summary().as_text(),
        "coefficients": model.params,
        "pvalues": model.pvalues,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def process_metrics(df: pd.DataFrame, metrics: List[str], label: str, out_dir: Path, tag: Optional[str]) -> None:
    records = []
    reg_records = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        clean = df.dropna(subset=[metric, "p_uva_mw_cm2", "p_uvb_mw_cm2"]).copy()
        if clean.empty:
            continue
        # two-way ANOVA (categorical doses)
        aov = run_two_way_anova(clean, metric)
        summary = summarize_anova(aov["anova_table"])
        summary.insert(0, "measurement", metric)
        summary.insert(0, "dataset", label)
        records.append(summary)

        # continuous regression
        reg = run_continuous_regression(clean, metric)
        for term, coef in reg["coefficients"].items():
            reg_records.append(
                {
                    "dataset": label,
                    "measurement": metric,
                    "term": term,
                    "coef": coef,
                    "pvalue": reg["pvalues"][term],
                    "r_squared": reg["r_squared"],
                    "adj_r_squared": reg["adj_r_squared"],
                }
            )

    if records:
        stats_df = pd.concat(records, ignore_index=True)
        stats_df.to_csv(out_dir / f"{label}_two_way_anova{'_' + tag if tag else ''}.csv", index=False)
    if reg_records:
        reg_df = pd.DataFrame.from_records(reg_records)
        reg_df.to_csv(out_dir / f"{label}_regression_coeffs{'_' + tag if tag else ''}.csv", index=False)


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent
    chrom_metrics = [m.strip() for m in args.chrom_metrics.split(",") if m.strip()]
    dad_metrics = [m.strip() for m in args.dad_metrics.split(",") if m.strip()]
    chrom_df, dad_df = load_custom_or_default(repo_root, args.chrom_file, args.dad_file)

    out_dir = repo_root / "Exploring_control_normalized" / "two_factor_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    process_metrics(chrom_df, chrom_metrics, "chromatogram", out_dir, args.tag)
    process_metrics(dad_df, dad_metrics, "dad", out_dir, args.tag)
    print("Two-factor modeling complete.")


if __name__ == "__main__":
    main()
