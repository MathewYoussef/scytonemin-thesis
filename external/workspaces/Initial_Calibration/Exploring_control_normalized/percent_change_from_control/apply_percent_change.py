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
    parser = argparse.ArgumentParser(description="Compute percent change from control for chromatogram and DAD data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def percent_change(values: pd.Series, baseline: float) -> pd.Series:
    if np.isnan(baseline) or baseline == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - baseline) / baseline * 100.0


def apply_chrom_percent(chrom_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = chrom_df.copy()
    df["pct_conc_mg_ml"] = np.nan
    df["pct_amount_mg_per_gDW"] = np.nan
    for form in FORMS:
        stats = baselines["chromatogram"].get(form, {})
        conc_base = stats.get("conc_mg_ml", {}).get("median", np.nan)
        amount_base = stats.get("amount_mg_per_gDW", {}).get("median", np.nan)
        mask = df["form"] == form
        df.loc[mask, "pct_conc_mg_ml"] = percent_change(df.loc[mask, "conc_mg_ml"], conc_base)
        df.loc[mask, "pct_amount_mg_per_gDW"] = percent_change(df.loc[mask, "amount_mg_per_gDW"], amount_base)
    return df


def apply_dad_percent(dad_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = dad_df.copy()
    for col, stats in baselines["dad"].items():
        base = stats.get("median", np.nan)
        if np.isnan(base) or base == 0:
            df[f"{col}_pct"] = np.nan
        else:
            df[f"{col}_pct"] = (df[col] - base) / base * 100.0
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
    chrom_pct = apply_chrom_percent(chrom_df, baselines)
    chrom_summary = summarize(chrom_pct, ["pct_conc_mg_ml", "pct_amount_mg_per_gDW"])

    dad_df = load_dad(repo_root)
    dad_pct = apply_dad_percent(dad_df, baselines)
    pct_cols = [col for col in dad_pct.columns if col.endswith("_pct")]
    dad_summary = summarize(dad_pct, pct_cols)

    out_dir = repo_root / "Exploring_control_normalized" / "percent_change_from_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_pct.to_csv(out_dir / "chromatogram_percent_change.csv", index=False)
    chrom_summary.to_csv(out_dir / "chromatogram_percent_change_summary.csv", index=False)
    dad_pct.to_csv(out_dir / "dad_percent_change.csv", index=False)
    dad_summary.to_csv(out_dir / "dad_percent_change_summary.csv", index=False)
    print(f"Wrote percent-change chromatogram data to {out_dir / 'chromatogram_percent_change.csv'}")
    print(f"Wrote percent-change DAD data to {out_dir / 'dad_percent_change.csv'}")


if __name__ == "__main__":
    main()
