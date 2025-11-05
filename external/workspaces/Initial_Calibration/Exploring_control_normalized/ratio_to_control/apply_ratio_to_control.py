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
    parser = argparse.ArgumentParser(description="Apply control ratio normalization to chromatogram and DAD data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def apply_chromatogram_ratio(chrom_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = chrom_df.copy()
    df["ratio_conc_mg_ml"] = np.nan
    df["ratio_amount_mg_per_gDW"] = np.nan
    for form in FORMS:
        form_mask = df["form"] == form
        stats = baselines["chromatogram"].get(form, {})
        conc = stats.get("conc_mg_ml", {})
        amount = stats.get("amount_mg_per_gDW", {})
        conc_base = conc.get("median", np.nan)
        amount_base = amount.get("median", np.nan)
        if conc and not np.isnan(conc_base) and conc_base != 0:
            df.loc[form_mask, "ratio_conc_mg_ml"] = df.loc[form_mask, "conc_mg_ml"] / conc_base
        elif conc and conc_base == 0:
            df.loc[form_mask, "ratio_conc_mg_ml"] = np.nan
        if amount and not np.isnan(amount_base) and amount_base != 0:
            df.loc[form_mask, "ratio_amount_mg_per_gDW"] = df.loc[form_mask, "amount_mg_per_gDW"] / amount_base
        elif amount and amount_base == 0:
            df.loc[form_mask, "ratio_amount_mg_per_gDW"] = np.nan
    return df


def apply_dad_ratio(dad_df: pd.DataFrame, baselines: Dict[str, Any]) -> pd.DataFrame:
    df = dad_df.copy()
    for col, stats in baselines["dad"].items():
        base = stats.get("median", np.nan)
        if not np.isnan(base) and base != 0:
            df[f"{col}_ratio"] = df[col] / base
        else:
            df[f"{col}_ratio"] = np.nan
    return df


def summarize(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    grouping = ["form", "p_uva_mw_cm2", "p_uvb_mw_cm2"] if "form" in df.columns else ["p_uva_mw_cm2", "p_uvb_mw_cm2"]
    summary = (
        df.groupby(grouping)[value_cols]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent
    baselines = ensure_baselines(repo_root)

    chrom_df = load_chromatogram(repo_root)
    chrom_ratio = apply_chromatogram_ratio(chrom_df, baselines)
    chrom_summary = summarize(chrom_ratio, ["ratio_conc_mg_ml", "ratio_amount_mg_per_gDW"])

    dad_df = load_dad(repo_root)
    dad_ratio = apply_dad_ratio(dad_df, baselines)
    ratio_cols = [col for col in dad_ratio.columns if col.endswith("_ratio")]
    dad_summary = summarize(dad_ratio, ratio_cols)

    out_dir = repo_root / "Exploring_control_normalized" / "ratio_to_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_ratio.to_csv(out_dir / "chromatogram_ratio.csv", index=False)
    chrom_summary.to_csv(out_dir / "chromatogram_ratio_summary.csv", index=False)
    dad_ratio.to_csv(out_dir / "dad_ratio.csv", index=False)
    dad_summary.to_csv(out_dir / "dad_ratio_summary.csv", index=False)
    print(f"Wrote ratio-normalized chromatogram data to {out_dir / 'chromatogram_ratio.csv'}")
    print(f"Wrote ratio-normalized DAD data to {out_dir / 'dad_ratio.csv'}")


if __name__ == "__main__":
    main()
