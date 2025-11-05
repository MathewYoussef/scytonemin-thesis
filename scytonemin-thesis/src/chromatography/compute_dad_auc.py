#!/usr/bin/env python3
"""
Compute diode-array spectra AUC (320–480 nm) for all scytonemin samples, including blanks.
Outputs raw and blank-corrected AUC tables for downstream calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute diode-array AUC for scytonemin spectra.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "analysis_config.yaml",
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def compute_auc(group: pd.DataFrame) -> pd.Series:
    subset = group.sort_values("wavelength_nm")
    wl = subset["wavelength_nm"].to_numpy(dtype=float)
    intensity = subset["intensity_abs"].to_numpy(dtype=float)
    auc = float(np.trapezoid(intensity, wl))
    return pd.Series(
        {
            "auc_raw": auc,
            "n_points": len(subset),
            "wavelength_min": wl.min() if len(wl) else np.nan,
            "wavelength_max": wl.max() if len(wl) else np.nan,
        }
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tidy_path = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "scytonemin_spectra_tidy.csv"
    output_dir = REPO_ROOT / config["paths"]["outputs"]["dad_auc_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_df = pd.read_csv(tidy_path)
    tidy_df = tidy_df[tidy_df["analyte"] == "Scytonemin"].copy()

    wl_min = config["spectral"]["wl_min_nm"]
    wl_max = config["spectral"]["wl_max_nm"]

    subset = tidy_df[
        (tidy_df["wavelength_nm"] >= wl_min) & (tidy_df["wavelength_nm"] <= wl_max)
    ].copy()

    grouped = subset.groupby(
        ["sample_id", "sample_category", "spectrum_state"], as_index=False, sort=False
    ).apply(compute_auc, include_groups=False).reset_index(drop=True)

    tol = 1.0
    coverage_mask = (grouped["wavelength_min"] <= wl_min + tol) & (grouped["wavelength_max"] >= wl_max - tol)
    grouped = grouped[coverage_mask].reset_index(drop=True)

    blanks = grouped[grouped["sample_category"] == "blank"]
    if blanks.empty:
        raise RuntimeError("No blank spectra with full wavelength coverage found.")

    blank_mean_raw = blanks["auc_raw"].mean()
    blank_std = blanks["auc_raw"].std(ddof=1)

    subtract_blank = config.get("spectral", {}).get("subtract_blank_auc", True)
    if subtract_blank:
        blank_baseline = blanks["auc_raw"].min()
        grouped["auc_corrected"] = grouped["auc_raw"] - blank_baseline
        grouped.loc[grouped["auc_corrected"] < 0, "auc_corrected"] = 0.0
        grouped.loc[grouped["sample_category"] == "blank", "auc_corrected"] = 0.0
        grouped["blank_baseline_auc"] = blank_baseline
    else:
        grouped["auc_corrected"] = grouped["auc_raw"]
        grouped["blank_baseline_auc"] = np.nan

    grouped["blank_mean_auc"] = blank_mean_raw
    grouped["blank_std_auc"] = blank_std

    grouped = grouped.sort_values(["sample_category", "sample_id"]).reset_index(drop=True)
    grouped.to_csv(output_dir / "diode_array_auc.csv", index=False)
    print(f"Wrote AUC table with {len(grouped)} rows to {output_dir / 'diode_array_auc.csv'}")


if __name__ == "__main__":
    main()
