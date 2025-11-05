#!/usr/bin/env python3
"""
Extract blank diode-array spectra from the XLS workbook and append them to the tidy CSV.

The blanks are required for diode-array calibration (serve as the zero-concentration anchor).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TIDY_PATH = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "scytonemin_spectra_tidy.csv"
XLS_PATH = REPO_ROOT / "DAD_RAW_FILES" / "Youssef_Abs_Spectra_per_pigment_3_10_25.xls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append blank spectra from XLS to tidy DAD CSV.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting if blank spectra already exist (default: abort to prevent duplication).",
    )
    return parser.parse_args()


def load_blank_spectra() -> pd.DataFrame:
    xl = pd.ExcelFile(XLS_PATH)
    if "Sheet1" not in xl.sheet_names:
        raise ValueError("Expected Sheet1 in the XLS workbook for blank spectra.")
    df = xl.parse("Sheet1")

    blank_pairs = [
        ("BLANK", "Unnamed: 45"),
        ("BLANK.1", "Unnamed: 47"),
        ("BLANK.2", "Unnamed: 49"),
    ]

    records: list[pd.DataFrame] = []
    for idx, (wl_col, abs_col) in enumerate(blank_pairs, start=1):
        if wl_col not in df.columns or abs_col not in df.columns:
            raise ValueError(f"Expected columns {wl_col}/{abs_col} in Sheet1.")

        sub = df[[wl_col, abs_col]].copy()
        sub.columns = ["wavelength_nm", "intensity_abs"]
        sub["wavelength_nm"] = pd.to_numeric(sub["wavelength_nm"], errors="coerce")
        sub["intensity_abs"] = pd.to_numeric(sub["intensity_abs"], errors="coerce")
        sub = sub.dropna()
        sub["sample_id"] = f"Blank {idx}"
        sub["sample_category"] = "blank"
        sub["analyte"] = "Scytonemin"
        sub["spectrum_state"] = "blank"

        sub = sub[
            [
                "sample_id",
                "sample_category",
                "analyte",
                "spectrum_state",
                "wavelength_nm",
                "intensity_abs",
            ]
        ]
        records.append(sub)

    blanks_df = pd.concat(records, ignore_index=True)
    blanks_df["wavelength_nm"] = blanks_df["wavelength_nm"].astype(float)
    blanks_df["intensity_abs"] = blanks_df["intensity_abs"].astype(float)
    return blanks_df


def append_blanks(overwrite: bool) -> None:
    tidy_df = pd.read_csv(TIDY_PATH)
    if not overwrite and (tidy_df["sample_category"] == "blank").any():
        raise RuntimeError(
            "Blank spectra already present in tidy CSV. Use --overwrite to refresh them explicitly."
        )
    tidy_df = tidy_df[tidy_df["sample_category"] != "blank"]

    blanks_df = load_blank_spectra()
    combined = pd.concat([tidy_df, blanks_df], ignore_index=True)
    combined = combined.sort_values(["sample_category", "sample_id", "wavelength_nm"]).reset_index(drop=True)
    combined.to_csv(TIDY_PATH, index=False)
    print(f"Appended {len(blanks_df)} blank spectra rows (three replicates) to {TIDY_PATH}")


def main() -> None:
    args = parse_args()
    append_blanks(args.overwrite)


if __name__ == "__main__":
    main()
