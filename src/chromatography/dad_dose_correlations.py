#!/usr/bin/env python3
"""
Compute Pearson and Spearman correlations between DAD-derived concentrations
(mg/mL and mg/gDW) and UV dose metadata (UVA, UVB, UVA/UVB ratio, UVA×UVB).

Outputs:
    DAD_Derived_Calibration_Plots/dad_dose_correlations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from scipy import stats  # type: ignore

    HAVE_SCIPY = True
except ModuleNotFoundError:  # pragma: no cover
    HAVE_SCIPY = False


FORMS = ["total", "oxidized", "reduced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DAD dose correlations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def correlations(values: pd.Series, doses: pd.Series) -> tuple[float, float, float, float, int]:
    mask = (~values.isna()) & (~doses.isna())
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    x = doses[mask].to_numpy(dtype=float)
    y = values[mask].to_numpy(dtype=float)
    if HAVE_SCIPY:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        pearson_p = np.nan
        rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        spearman_r = float(np.corrcoef(rank_x, rank_y)[0, 1])
        spearman_p = np.nan
    return pearson_r, pearson_p, spearman_r, spearman_p, n


def main() -> None:
    args = parse_args()
    repo = args.config.resolve().parent
    with args.config.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)  # config currently unused but validates path

    dad_df = pd.read_csv(repo / "DAD_derived_concentrations_corrected.csv")
    if {"p_uva_mw_cm2", "p_uvb_mw_cm2"}.issubset(dad_df.columns):
        dose_df = dad_df.copy()
    else:
        truth_df = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")[
            ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
        ]
        dose_df = dad_df.merge(truth_df, on="sample_id", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        dose_df["uva_uvb_ratio"] = dose_df["p_uva_mw_cm2"] / dose_df["p_uvb_mw_cm2"]
    dose_df["uva_uvb_ratio"] = dose_df["uva_uvb_ratio"].replace([np.inf, -np.inf], np.nan)
    dose_df["uva_times_uvb"] = dose_df["p_uva_mw_cm2"] * dose_df["p_uvb_mw_cm2"]

    records = []
    for form in FORMS:
        conc_col = f"predicted_{form}_mg_ml"
        amount_col = f"predicted_{form}_mg_per_gDW"
        for label, series in {
            "mg_mL": dose_df[conc_col],
            "mg_per_gDW": dose_df[amount_col],
        }.items():
            for metric, dose_series in {
                "UVA_mW_cm2": dose_df["p_uva_mw_cm2"],
                "UVB_mW_cm2": dose_df["p_uvb_mw_cm2"],
                "UVA_div_UVB": dose_df["uva_uvb_ratio"],
                "UVA_times_UVB": dose_df["uva_times_uvb"],
            }.items():
                pearson_r, pearson_p, spearman_r, spearman_p, n = correlations(series, dose_series)
                records.append(
                    {
                        "form": form,
                        "measurement": label,
                        "dose_metric": metric,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "n": n,
                    }
                )

    output_dir = repo / "Diode_Array_Derived_Calibration_Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dad_dose_correlations.csv"
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    print(f"Wrote DAD correlation summary to {out_path}")


if __name__ == "__main__":
    main()
