#!/usr/bin/env python3
"""
Correlate DAD spectral features (treatments only) with UV dose metrics.

Outputs:
    DAD_feature_extraction_replicate_exploration/dad_feature_dose_correlations.csv
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
    parser = argparse.ArgumentParser(description="Compute correlations between DAD features and UV dose metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("DAD_feature_extraction_replicate_exploration/dad_features.csv"),
        help="Path to extracted features CSV (default: %(default)s)",
    )
    return parser.parse_args()


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    xv = x[mask]
    yv = y[mask]
    if HAVE_SCIPY:
        pearson_r, pearson_p = stats.pearsonr(xv, yv)
        spearman_r, spearman_p = stats.spearmanr(xv, yv)
    else:
        pearson_r = float(np.corrcoef(xv, yv)[0, 1])
        pearson_p = np.nan
        ranks_x = pd.Series(xv).rank(method="average").to_numpy(dtype=float)
        ranks_y = pd.Series(yv).rank(method="average").to_numpy(dtype=float)
        spearman_r = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        spearman_p = np.nan
    return pearson_r, pearson_p, spearman_r, spearman_p, n


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)
    repo = args.config.resolve().parent

    features_path = args.features if args.features.is_absolute() else repo / args.features
    df = pd.read_csv(features_path)
    df = df[df["sample_category"] == "sample"].copy()

    truth_df = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")[
        ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    ]
    df = df.merge(truth_df, on="sample_id", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["uva_uvb_ratio"] = df["p_uva_mw_cm2"] / df["p_uvb_mw_cm2"]
    df["uva_uvb_ratio"] = df["uva_uvb_ratio"].replace([np.inf, -np.inf], np.nan)
    df["uva_times_uvb"] = df["p_uva_mw_cm2"] * df["p_uvb_mw_cm2"]

    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "sample_id",
            "sample_category",
            "spectrum_state",
            "p_uva_mw_cm2",
            "p_uvb_mw_cm2",
            "uva_uvb_ratio",
            "uva_times_uvb",
        }
        and df[col].dtype != object
    ]

    dose_metrics = {
        "UVA_mW_cm2": "p_uva_mw_cm2",
        "UVB_mW_cm2": "p_uvb_mw_cm2",
        "UVA_div_UVB": "uva_uvb_ratio",
        "UVA_times_UVB": "uva_times_uvb",
    }

    records = []
    for form in FORMS:
        subset = df[df["spectrum_state"] == form]
        for feature in feature_cols:
            values = subset[feature].to_numpy(dtype=float)
            for metric, dose_col in dose_metrics.items():
                dose_values = df.loc[subset.index, dose_col].to_numpy(dtype=float)
                pearson_r, pearson_p, spearman_r, spearman_p, n = corr(
                    values,
                    dose_values,
                )
                records.append(
                    {
                        "form": form,
                        "feature": feature,
                        "dose_metric": metric,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "n": n,
                    }
                )

    output_path = (
        repo / "DAD_feature_extraction_replicate_exploration" / "dad_feature_dose_correlations.csv"
    )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    print(f"Wrote feature-dose correlations to {output_path}")


if __name__ == "__main__":
    main()
