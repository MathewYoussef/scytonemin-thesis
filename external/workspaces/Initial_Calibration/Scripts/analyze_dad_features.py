#!/usr/bin/env python3
"""
Stage D analysis:
    * Merge DAD spectral features with known concentrations (standards) and chromatogram-derived concentrations (treatments).
    * Compute linear regression summaries per form/feature.
Outputs:
    DAD_feature_extraction_replicate_exploration/dad_features_standards_vs_chromatogram.csv
    DAD_feature_extraction_replicate_exploration/dad_features_treatments_vs_chromatogram.csv
    DAD_feature_extraction_replicate_exploration/dad_feature_regression_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


FORMS = ["total", "oxidized", "reduced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DAD spectral features.")
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


def load_config(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)  # just validate
    return path.resolve().parent


def parse_standard_level(sample_id: str) -> float | None:
    match = re.search(r"Standard\s*([0-9]+)", sample_id)
    if match:
        return float(match.group(1))
    return None


def regression_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return slope, intercept, r_squared, max_abs_rel_residual."""
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    fitted = intercept + slope * x
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_res = np.where(y != 0, np.abs(residuals) / np.abs(y), np.nan)
    max_rel = float(np.nanmax(rel_res)) if np.any(~np.isnan(rel_res)) else float("nan")
    return float(slope), float(intercept), r_squared, max_rel


def main() -> None:
    args = parse_args()
    repo_root = load_config(args.config)

    features_path = args.features if args.features.is_absolute() else repo_root / args.features
    features_df = pd.read_csv(features_path)

    standards_df = features_df[features_df["sample_category"] == "standard"].copy()
    standards_df["standard_level"] = standards_df["sample_id"].apply(parse_standard_level)
    standard_conc = pd.read_csv(repo_root / "DAD_RAW_FILES" / "scytonemin_standard_concentrations.csv")
    standards_df = standards_df.merge(
        standard_conc.rename(columns={"concentration_mg_ml": "known_concentration_mg_ml"}),
        how="left",
        left_on="standard_level",
        right_on="standard_level",
    )

    treatments_df = features_df[features_df["sample_category"] == "sample"].copy()
    chromo_raw = pd.read_csv(repo_root / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv")
    chromo_corr = pd.read_csv(repo_root / "DAD_to_Concentration_AUC" / "treatments_corrected_amounts.csv")

    treatments_df = treatments_df.merge(
        chromo_raw[["sample_id", "form", "conc_mg_ml"]],
        how="left",
        left_on=["sample_id", "spectrum_state"],
        right_on=["sample_id", "form"],
    ).merge(
        chromo_corr[["sample_id", "form", "amount_mg_per_gDW"]],
        how="left",
        left_on=["sample_id", "spectrum_state"],
        right_on=["sample_id", "form"],
        suffixes=("", "_corr"),
    )

    feature_cols = [
        col
        for col in features_df.columns
        if col not in {"sample_id", "sample_category", "spectrum_state", "standard_level", "known_concentration_mg_ml"}
    ]

    feature_cols = [c for c in feature_cols if features_df[c].dtype != object or c.startswith("abs")]

    standards_records = []
    summary_records = []
    for form in FORMS:
        std_subset = standards_df[standards_df["spectrum_state"] == form]
        for feature in feature_cols:
            if feature not in std_subset.columns:
                continue
            x = std_subset[feature].to_numpy(dtype=float)
            y = std_subset["known_concentration_mg_ml"].to_numpy(dtype=float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 2:
                continue
            slope, intercept, r_sq, max_rel = regression_stats(x[mask], y[mask])
            summary_records.append(
                {
                    "form": form,
                    "feature": feature,
                    "data": "standards",
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_sq,
                    "max_abs_rel_residual": max_rel,
                    "n": int(mask.sum()),
                }
            )
            standards_records.extend(
                {
                    "sample_id": row.sample_id,
                    "form": form,
                    "feature": feature,
                    "feature_value": row[feature],
                    "known_concentration_mg_ml": row["known_concentration_mg_ml"],
                }
                for _, row in std_subset[mask].iterrows()
            )

    treatment_records = []
    for form in FORMS:
        tr_subset = treatments_df[treatments_df["spectrum_state"] == form]
        for feature in feature_cols:
            if feature not in tr_subset.columns:
                continue
            x = tr_subset[feature].to_numpy(dtype=float)
            y = tr_subset["conc_mg_ml"].to_numpy(dtype=float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() >= 2:
                slope, intercept, r_sq, max_rel = regression_stats(x[mask], y[mask])
                summary_records.append(
                    {
                        "form": form,
                        "feature": feature,
                        "data": "treatments_conc_mg_ml",
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_sq,
                        "max_abs_rel_residual": max_rel,
                        "n": int(mask.sum()),
                    }
                )
            y_corr = tr_subset["amount_mg_per_gDW"].to_numpy(dtype=float)
            mask_corr = ~np.isnan(x) & ~np.isnan(y_corr)
            if mask_corr.sum() >= 2:
                slope, intercept, r_sq, max_rel = regression_stats(x[mask_corr], y_corr[mask_corr])
                summary_records.append(
                    {
                        "form": form,
                        "feature": feature,
                        "data": "treatments_mg_per_gDW",
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_sq,
                        "max_abs_rel_residual": max_rel,
                        "n": int(mask_corr.sum()),
                    }
                )
            treatment_records.extend(
                {
                    "sample_id": row.sample_id,
                    "form": form,
                    "feature": feature,
                    "feature_value": row[feature],
                    "conc_mg_ml": row["conc_mg_ml"],
                    "amount_mg_per_gDW": row["amount_mg_per_gDW"],
                }
                for _, row in tr_subset.iterrows()
            )

    out_dir = repo_root / "DAD_feature_extraction_replicate_exploration"
    out_dir.mkdir(parents=True, exist_ok=True)

    if standards_records:
        pd.DataFrame.from_records(standards_records).to_csv(
            out_dir / "dad_features_standards_vs_chromatogram.csv", index=False
        )
    if treatment_records:
        pd.DataFrame.from_records(treatment_records).to_csv(
            out_dir / "dad_features_treatments_vs_chromatogram.csv", index=False
        )

    pd.DataFrame.from_records(summary_records).to_csv(
        out_dir / "dad_feature_regression_summary.csv", index=False
    )
    print(f"Wrote regression summaries to {out_dir / 'dad_feature_regression_summary.csv'}")


if __name__ == "__main__":
    main()
