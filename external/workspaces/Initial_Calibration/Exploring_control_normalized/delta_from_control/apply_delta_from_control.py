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
    parser = argparse.ArgumentParser(description="Apply control-subtraction (delta) normalization to chromatogram and DAD data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def apply_chromatogram_delta(chrom_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = chrom_df.copy()
    df["delta_conc_mg_ml"] = np.nan
    df["delta_amount_mg_per_gDW"] = np.nan
    for form in FORMS:
        form_mask = df["form"] == form
        stats = baselines["chromatogram"].get(form, {})
        conc_stats = stats.get("conc_mg_ml", {})
        amount_stats = stats.get("amount_mg_per_gDW", {})
        if conc_stats:
            df.loc[form_mask, "delta_conc_mg_ml"] = df.loc[form_mask, "conc_mg_ml"] - conc_stats["median"]
        if amount_stats:
            df.loc[form_mask, "delta_amount_mg_per_gDW"] = df.loc[form_mask, "amount_mg_per_gDW"] - amount_stats["median"]
    return df


def summarize_dose_groups(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    grouping = ["form", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    summary = (
        df.groupby(grouping)[value_cols]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary


def apply_dad_delta(dad_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = dad_df.copy()
    for col, stats in baselines["dad"].items():
        baseline = stats.get("median", np.nan)
        df[f"{col}_delta"] = df[col] - baseline
    return df


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent
    baselines = ensure_baselines(repo_root)

    chrom_df = load_chromatogram(repo_root)
    chrom_delta = apply_chromatogram_delta(chrom_df, baselines)
    chrom_summary = summarize_dose_groups(
        chrom_delta,
        ["delta_conc_mg_ml", "delta_amount_mg_per_gDW"],
    )

    dad_df = load_dad(repo_root)
    dad_delta = apply_dad_delta(dad_df, baselines)
    delta_cols = [col for col in dad_delta.columns if col.endswith("_delta")]
    dad_summary = dad_delta.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"])[delta_cols].agg(["median", "mean", "std", "count"]).reset_index()
    dad_summary.columns = ["_".join(col).rstrip("_") for col in dad_summary.columns.to_flat_index()]

    out_dir = repo_root / "Exploring_control_normalized" / "delta_from_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_delta.to_csv(out_dir / "chromatogram_delta.csv", index=False)
    chrom_summary.to_csv(out_dir / "chromatogram_delta_summary.csv", index=False)
    dad_delta.to_csv(out_dir / "dad_delta.csv", index=False)
    dad_summary.to_csv(out_dir / "dad_delta_summary.csv", index=False)

    print(f"Wrote chromatogram deltas to {out_dir / 'chromatogram_delta.csv'}")
    print(f"Wrote DAD deltas to {out_dir / 'dad_delta.csv'}")


if __name__ == "__main__":
    main()
