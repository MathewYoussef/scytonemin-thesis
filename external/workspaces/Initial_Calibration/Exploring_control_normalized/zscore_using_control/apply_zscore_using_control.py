#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import (
    load_chromatogram,
    load_dad,
    ensure_baselines,
    FORMS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute control-based z-scores for chromatogram and DAD data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def safe_z(values: pd.Series, stats: Dict[str, Any]) -> pd.Series:
    mean = stats.get("mean", np.nan)
    std = stats.get("std", np.nan)
    if np.isnan(mean) or np.isnan(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - mean) / std


def apply_chrom_zscore(chrom_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = chrom_df.copy()
    df["z_conc_mg_ml"] = np.nan
    df["z_amount_mg_per_gDW"] = np.nan
    for form in FORMS:
        stats = baselines["chromatogram"].get(form, {})
        conc_stats = stats.get("conc_mg_ml", {})
        amount_stats = stats.get("amount_mg_per_gDW", {})
        mask = df["form"] == form
        df.loc[mask, "z_conc_mg_ml"] = safe_z(df.loc[mask, "conc_mg_ml"], conc_stats)
        df.loc[mask, "z_amount_mg_per_gDW"] = safe_z(df.loc[mask, "amount_mg_per_gDW"], amount_stats)
    return df


def apply_dad_zscore(dad_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = dad_df.copy()
    for col, stats in baselines["dad"].items():
        df[f"{col}_zscore"] = safe_z(df[col], stats)
    return df


def summarize(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    grouping = ["form", "p_uva_mw_cm2", "p_uvb_mw_cm2"] if "form" in df.columns else ["p_uva_mw_cm2", "p_uvb_mw_cm2"]
    summary = df.groupby(grouping)[value_cols].agg(["median", "mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent
    baselines = ensure_baselines(repo_root)

    chrom_df = load_chromatogram(repo_root)
    chrom_z = apply_chrom_zscore(chrom_df, baselines)
    chrom_summary = summarize(chrom_z, ["z_conc_mg_ml", "z_amount_mg_per_gDW"])

    dad_df = load_dad(repo_root)
    dad_z = apply_dad_zscore(dad_df, baselines)
    z_cols = [col for col in dad_z.columns if col.endswith("_zscore")]
    dad_summary = summarize(dad_z, z_cols)

    out_dir = repo_root / "Exploring_control_normalized" / "zscore_using_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_z.to_csv(out_dir / "chromatogram_zscores.csv", index=False)
    chrom_summary.to_csv(out_dir / "chromatogram_zscores_summary.csv", index=False)
    dad_z.to_csv(out_dir / "dad_zscores.csv", index=False)
    dad_summary.to_csv(out_dir / "dad_zscores_summary.csv", index=False)
    print(f"Wrote z-scored chromatogram data to {out_dir / 'chromatogram_zscores.csv'}")
    print(f"Wrote z-scored DAD data to {out_dir / 'dad_zscores.csv'}")


if __name__ == "__main__":
    main()
