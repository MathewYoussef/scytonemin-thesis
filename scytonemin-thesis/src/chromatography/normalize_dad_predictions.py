#!/usr/bin/env python3
"""
Add biomass-normalized amounts (mg/gDW) to DAD-derived concentration predictions.

Outputs:
    DAD_derived_concentrations_corrected.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize DAD-derived concentrations by biomass.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    repo = args.config.resolve().parent

    extraction_volume = float(config["chemistry"]["extraction_volume_ml"])
    dilution_factor = float(config["chemistry"]["dilution_factor"])
    factor = extraction_volume * dilution_factor

    dad_path = repo / "DAD_derived_concentrations.csv"
    df = pd.read_csv(dad_path)

    if "dry_biomass_g" not in df.columns:
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        truth_df = pd.read_csv(truth_path)[["sample_id", "dry_biomass_g"]]
        df = df.merge(truth_df, on="sample_id", how="left")

    for form in ("total", "oxidized", "reduced"):
        conc_col = f"predicted_{form}_mg_ml"
        amount_col = f"predicted_{form}_mg_per_gDW"
        if conc_col in df.columns:
            df[amount_col] = np.where(
                (df["dry_biomass_g"] > 0) & ~df["dry_biomass_g"].isna(),
                df[conc_col] * factor / df["dry_biomass_g"],
                np.nan,
            )

    output_path = repo / "DAD_derived_concentrations_corrected.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote biomass-normalized DAD predictions to {output_path}")


if __name__ == "__main__":
    main()
