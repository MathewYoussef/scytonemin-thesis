#!/usr/bin/env python3
"""
Compute Pearson and Spearman correlations between chromatogram-derived concentrations
(mg/mL and mg/gDW) and UV dose metadata (UVA, UVB, UVA/UVB ratio, UVA×UVB).

Outputs:
    Chromatogram_derived_concentration_patterns_plots/chromatogram_dose_correlations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

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
    parser = argparse.ArgumentParser(description="Compute chromatogram dose correlations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def pivot_values(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="sample_id", columns="form", values=value_col)
    pivot.columns = [f"{col}_{value_col}" for col in pivot.columns]
    return pivot


def correlation_table(
    data: pd.DataFrame,
    values: Dict[str, pd.Series],
    dose_metrics: Dict[str, pd.Series],
) -> pd.DataFrame:
    records = []
    for form, series in values.items():
        form_name, measure = form.split(":", 1)
        y = series.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        for dose_name, dose_series in dose_metrics.items():
            x = dose_series.to_numpy(dtype=float)
            use_mask = mask & ~np.isnan(x)
            if use_mask.sum() < 3:
                pearson_r = np.nan
                pearson_p = np.nan
                spearman_r = np.nan
                spearman_p = np.nan
            else:
                if HAVE_SCIPY:
                    pearson_r, pearson_p = stats.pearsonr(x[use_mask], y[use_mask])
                    spearman_r, spearman_p = stats.spearmanr(x[use_mask], y[use_mask])
                else:
                    pearson_r = float(np.corrcoef(x[use_mask], y[use_mask])[0, 1])
                    pearson_p = np.nan
                    ranks_x = pd.Series(x[use_mask]).rank(method="average").to_numpy()
                    ranks_y = pd.Series(y[use_mask]).rank(method="average").to_numpy()
                    spearman_r = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
                    spearman_p = np.nan
            records.append(
                {
                    "form": form_name,
                    "measurement": measure,
                    "dose_metric": dose_name,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n": int(use_mask.sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent
    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    output_dir = repo_root / "Chromatogram_derived_concentration_patterns_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(repo_root / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv")
    corrected_df = pd.read_csv(repo_root / "DAD_to_Concentration_AUC" / "treatments_corrected_amounts.csv")
    truth_df = pd.read_csv(repo_root / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")

    dose_df = truth_df[
        ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2", "uva_pct_mdv", "uvb_pct_mdv"]
    ].drop_duplicates()
    with np.errstate(divide="ignore", invalid="ignore"):
        dose_df["uva_uvb_ratio"] = dose_df["p_uva_mw_cm2"] / dose_df["p_uvb_mw_cm2"]
    dose_df["uva_uvb_ratio"] = dose_df["uva_uvb_ratio"].replace([np.inf, -np.inf], np.nan)
    dose_df["uva_times_uvb"] = dose_df["p_uva_mw_cm2"] * dose_df["p_uvb_mw_cm2"]

    mg_ml_wide = pivot_values(raw_df, "conc_mg_ml")
    mg_gdw_wide = pivot_values(corrected_df, "amount_mg_per_gDW")

    merged = (
        dose_df.merge(mg_ml_wide, on="sample_id", how="left")
        .merge(mg_gdw_wide, on="sample_id", how="left")
        .drop_duplicates(subset=["sample_id"])
    )

    value_series: Dict[str, pd.Series] = {}
    for form in FORMS:
        ml_col = f"{form}_conc_mg_ml"
        gdw_col = f"{form}_amount_mg_per_gDW"
        if ml_col in merged.columns:
            value_series[f"{form}:mg_mL"] = merged[ml_col]
        if gdw_col in merged.columns:
            value_series[f"{form}:mg_per_gDW"] = merged[gdw_col]

    dose_metrics = {
        "UVA_mW_cm2": merged["p_uva_mw_cm2"],
        "UVB_mW_cm2": merged["p_uvb_mw_cm2"],
        "UVA_div_UVB": merged["uva_uvb_ratio"],
        "UVA_times_UVB": merged["uva_times_uvb"],
    }

    results = correlation_table(merged, value_series, dose_metrics)
    output_path = output_dir / "chromatogram_dose_correlations.csv"
    results.to_csv(output_path, index=False)
    print(f"Wrote correlation summary to {output_path}")


if __name__ == "__main__":
    main()
