#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FORMS = ["total", "oxidized", "reduced"]


def load_calibration(directory: Path) -> dict[str, tuple[float, float]]:
    coefs: dict[str, tuple[float, float]] = {}
    for form in FORMS:
        with (directory / f"calibration_{form}.json").open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        coefs[form] = (data["intercept"], data["slope"])
    return coefs


def load_predictions(auc_path: Path, coefs: dict[str, tuple[float, float]]) -> pd.DataFrame:
    auc_df = pd.read_csv(auc_path)
    treatments = auc_df[auc_df["sample_category"] == "sample"].copy()
    pivot = treatments.pivot_table(
        index="sample_id",
        columns="spectrum_state",
        values="auc_corrected",
    )
    rows = []
    for sample_id, row in pivot.iterrows():
        record = {"sample_id": sample_id}
        for form in FORMS:
            intercept, slope = coefs[form]
            auc = row.get(form)
            if pd.isna(auc):
                record[f"{form}_pred"] = float("nan")
            else:
                record[f"{form}_pred"] = intercept + slope * auc
        rows.append(record)
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def main() -> None:
    base_preds = load_predictions(
        Path("Diode_Array_AUC/diode_array_auc.csv"),
        load_calibration(Path("Diode_Array_Derived_Calibration_Plots")),
    )
    raw_preds = load_predictions(
        Path("Diode_Array_AUC_no_blank/diode_array_auc.csv"),
        load_calibration(Path("Diode_Array_Derived_Calibration_Plots_no_blank")),
    )
    merged = base_preds.merge(raw_preds, on="sample_id", suffixes=("_blank_sub", "_raw"))
    cols = ["sample_id"]
    for form in FORMS:
        base_col = f"{form}_pred_blank_sub"
        raw_col = f"{form}_pred_raw"
        delta_col = f"{form}_pred_delta"
        merged[delta_col] = merged[raw_col] - merged[base_col]
        cols.extend([base_col, raw_col, delta_col])
    print(merged[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
