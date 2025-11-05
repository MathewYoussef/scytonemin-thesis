#!/usr/bin/env python3
"""
Generate diode-array derived concentration predictions for treatment samples.

Outputs:
  - DAD_derived_concentrations.csv with sample metadata, AUCs, and predicted concentrations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concentrations from diode-array AUCs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo = args.config.resolve().parent

    auc_path = repo / config["paths"]["outputs"]["dad_auc_dir"] / "diode_array_auc.csv"
    auc_df = pd.read_csv(auc_path)

    truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
    truth_df = pd.read_csv(truth_path)
    truth_df = truth_df.rename(
        columns={
            "p_uva_mw_cm2": "p_uva_mw_cm2",
            "p_uvb_mw_cm2": "p_uvb_mw_cm2",
        }
    )

    # Prepare metadata
    metadata = truth_df[[
        "sample_id",
        "dry_biomass_g",
        "p_uva_mw_cm2",
        "p_uvb_mw_cm2",
    ]].copy()
    metadata["uva_uvb_ratio"] = metadata.apply(
        lambda row: row["p_uva_mw_cm2"] / row["p_uvb_mw_cm2"]
        if row["p_uvb_mw_cm2"] not in (0, np.nan)
        else np.nan,
        axis=1,
    )

    # Pivot AUCs for treatments
    treatments = auc_df[auc_df["sample_category"] == "sample"].copy()
    auc_wide = treatments.pivot(
        index="sample_id",
        columns="spectrum_state",
        values="auc_corrected",
    )
    auc_wide = auc_wide.rename(
        columns={
            "total": "auc_total_320_480",
            "oxidized": "auc_oxidized_320_480",
            "reduced": "auc_reduced_320_480",
        }
    )

    # Load calibration coefficients
    cal_dir = repo / config["paths"]["outputs"]["dad_calibration_plots_dir"]
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}
    for form in ("total", "oxidized", "reduced"):
        with (cal_dir / f"calibration_{form}.json").open("r", encoding="utf-8") as fh:
            calib = yaml.safe_load(fh)
        slopes[form] = calib["slope"]
        intercepts[form] = calib.get("intercept", 0.0)

    # Predict concentrations (mg/mL)
    predictions = pd.DataFrame(index=auc_wide.index)
    predictions["predicted_total_mg_ml"] = intercepts["total"] + slopes["total"] * auc_wide["auc_total_320_480"]
    predictions["predicted_oxidized_mg_ml"] = intercepts["oxidized"] + slopes["oxidized"] * auc_wide["auc_oxidized_320_480"]
    predictions["predicted_reduced_mg_ml"] = intercepts["reduced"] + slopes["reduced"] * auc_wide["auc_reduced_320_480"]

    # Merge everything
    combined = metadata.merge(auc_wide, how="inner", left_on="sample_id", right_index=True)
    combined = combined.merge(predictions, how="left", left_on="sample_id", right_index=True)

    combined = combined.sort_values("sample_id").reset_index(drop=True)
    out_cols = [
        "sample_id",
        "p_uva_mw_cm2",
        "p_uvb_mw_cm2",
        "uva_uvb_ratio",
        "dry_biomass_g",
        "auc_total_320_480",
        "predicted_total_mg_ml",
        "auc_oxidized_320_480",
        "predicted_oxidized_mg_ml",
        "auc_reduced_320_480",
        "predicted_reduced_mg_ml",
    ]
    combined = combined[out_cols]

    output_path = repo / "DAD_derived_concentrations.csv"
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")


if __name__ == "__main__":
    main()
