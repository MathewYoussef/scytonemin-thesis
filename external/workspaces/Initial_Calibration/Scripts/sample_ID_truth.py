#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "DAD_RAW_FILES"
DEFAULT_OUTPUT = REPO_ROOT / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sample ID truth table that combines biomass and irradiation metadata."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated CSV (default: %(default)s)",
    )
    return parser.parse_args()


def split_letters(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    tokens = []
    for token in str(value).split():
        token = token.strip().upper()
        if token:
            tokens.append(token)
    return tokens


def unique_letters(*groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
    return ordered


def make_sample_id(prefix: object, letter: str) -> str:
    if prefix is None or (isinstance(prefix, float) and pd.isna(prefix)):
        raise ValueError("Missing treatment_number/column_label for sample ID")
    prefix_str = str(prefix).strip()
    if not prefix_str:
        raise ValueError("Empty treatment_number/column_label")
    try:
        prefix_val = int(float(prefix_str))
        prefix_str = str(prefix_val)
    except ValueError:
        prefix_str = prefix_str
    return f"{prefix_str}{letter.upper()}"


def coalesce(row: dict[str, object], base: str) -> float:
    chamber_val = row.get(f"{base}_chamber")
    if chamber_val is not None and not (isinstance(chamber_val, float) and pd.isna(chamber_val)):
        return float(chamber_val)
    random_val = row.get(f"{base}_random")
    return float(random_val) if random_val is not None and not (isinstance(random_val, float) and pd.isna(random_val)) else float("nan")


def build_truth_table(random_df: pd.DataFrame, chamber_df: pd.DataFrame, biomass_df: pd.DataFrame) -> pd.DataFrame:
    random_df = random_df.rename(
        columns={
            "P_UVA (mW cm^-2)": "p_uva",
            "P_UVB (mW cm^-2)": "p_uvb",
            "%MDV_UVA_dose (%)": "uva_pct_mdv",
            "%MDV_UVB_dose (%)": "uvb_pct_mdv",
        }
    ).copy()
    chamber_df = chamber_df.rename(
        columns={
            "grid (#)": "grid",
            "P_UVA (mW cm^-2)": "p_uva",
            "P_UVB (mW cm^-2)": "p_uvb",
            "%MDV_UVA_dose (%)": "uva_pct_mdv",
            "%MDV_UVB_dose (%)": "uvb_pct_mdv",
        }
    ).copy()
    random_df["column_numeric"] = pd.to_numeric(random_df["column_label"], errors="coerce")
    chamber_subset = chamber_df[["grid", "p_uva", "p_uvb", "uva_pct_mdv", "uvb_pct_mdv"]]
    merged = random_df.merge(
        chamber_subset,
        how="left",
        left_on="column_numeric",
        right_on="grid",
        suffixes=("_random", "_chamber"),
    )

    records: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        letters = unique_letters(
            split_letters(row.get("spectroscopy_letters")),
            split_letters(row.get("storage_letters")),
        )
        for letter in letters:
            prefix = row.get("treatment_number")
            if prefix is None or (isinstance(prefix, float) and pd.isna(prefix)):
                prefix = row.get("column_label")
            sample_id = make_sample_id(prefix, letter)
            records.append(
                {
                    "sample_id": sample_id,
                    "p_uva_mw_cm2": coalesce(row, "p_uva"),
                    "p_uvb_mw_cm2": coalesce(row, "p_uvb"),
                    "uva_pct_mdv": coalesce(row, "uva_pct_mdv"),
                    "uvb_pct_mdv": coalesce(row, "uvb_pct_mdv"),
                }
            )

    truth_df = pd.DataFrame(records)
    truth_df = truth_df.merge(biomass_df, how="left", on="sample_id")
    truth_df = truth_df[
        [
            "sample_id",
            "dry_biomass_g",
            "p_uva_mw_cm2",
            "p_uvb_mw_cm2",
            "uva_pct_mdv",
            "uvb_pct_mdv",
        ]
    ].sort_values("sample_id").reset_index(drop=True)
    return truth_df


def main() -> None:
    args = parse_args()
    random_df = pd.read_csv(RAW_DIR / "randomization_by_column.csv")
    chamber_df = pd.read_csv(RAW_DIR / "chamber_dose_schedule.csv")
    biomass_df = pd.read_csv(RAW_DIR / "sample_biomass.csv")
    truth_df = build_truth_table(random_df, chamber_df, biomass_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    truth_df.to_csv(args.output, index=False)
    print(f"Wrote {len(truth_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
