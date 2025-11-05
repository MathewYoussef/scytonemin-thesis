#!/usr/bin/env python3
"""
Stages B & C: apply scytonemin calibrations to treatment chromatograms and normalize by biomass.

Outputs:
- DAD_to_Concentration_AUC/treatments_concentration_raw.csv
- DAD_to_Concentration_AUC/treatments_corrected_amounts.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage B/C quantification for scytonemin chromatograms.")
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


@dataclass
class Calibration:
    form: str
    slope: float
    intercept: float
    intercept_mode: str
    slope_se: float
    intercept_se: Optional[float]
    cov_intercept_slope: Optional[float]
    loq_mg_ml: Optional[float]
    qc_pass: bool


@dataclass
class TotalPartsModel:
    intercept: float
    slope: float
    intercept_se: Optional[float]
    slope_se: Optional[float]
    r_squared: float
    rmse: float
    dof: int
    n: int


def load_calibration(path: Path, form: str) -> Calibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    detection_limits = data.get("detection_limits", {})
    loq = detection_limits.get("LOQ_mg_ml")
    return Calibration(
        form=form,
        slope=data["slope"],
        intercept=data["intercept"],
        intercept_mode=data.get("intercept_mode", "free"),
        slope_se=data.get("slope_se", 0.0) or 0.0,
        intercept_se=data.get("intercept_se"),
        cov_intercept_slope=data.get("cov_intercept_slope"),
        loq_mg_ml=loq if loq is None or not math.isnan(loq) else None,
        qc_pass=bool(data.get("qc_pass", False)),
    )


def extract_sample_id(name: str) -> str:
    if not isinstance(name, str):
        return ""
    match = re.search(r"([0-9]+[A-Za-z]+)$", name.strip())
    return match.group(1) if match else name.strip()


def first_source_file(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.split(";")[0].strip()


def compute_concentration(area: float, calib: Calibration) -> tuple[float, float]:
    raw = calib.intercept + calib.slope * area
    conc = max(raw, 0.0)
    var = (calib.slope_se ** 2) * (area**2)
    if calib.intercept_se is not None:
        var += (calib.intercept_se ** 2)
    if calib.cov_intercept_slope is not None:
        var += 2.0 * area * calib.cov_intercept_slope
    if var < 0:
        var = 0.0
    se = math.sqrt(var)
    return conc, se


def load_chromatogram(form: str, chrom_dir: Path) -> pd.DataFrame:
    path = chrom_dir / f"raw_{form}_scytonemin.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing chromatogram file {path}")
    df = pd.read_csv(path)
    df["type"] = df["sample_category"].map({"sample": "treatment", "standard": "standard", "blank": "blank"})
    return df


def compute_stage_b(
    form: str,
    config: dict,
    repo_root: Path,
    calib: Calibration,
) -> pd.DataFrame:
    chrom_dir = repo_root / config["paths"]["chrom_areas_dir"]
    df = load_chromatogram(form, chrom_dir)
    treatments = df[df["type"] == "treatment"].copy()
    treatments = treatments.dropna(subset=["area"])
    treatments["area"] = treatments["area"].astype(float)
    records = []
    for row in treatments.itertuples(index=False):
        area = float(row.area)
        conc, se = compute_concentration(area, calib)
        loq_flag = False
        if calib.loq_mg_ml is not None and not math.isnan(calib.loq_mg_ml):
            loq_flag = conc < calib.loq_mg_ml
        records.append(
            {
                "sample_name": row.sample_name,
                "sample_id": extract_sample_id(row.sample_name),
                "form": form,
                "area": area,
                "conc_mg_ml": conc,
                "conc_se_mg_ml": se,
                "loq_flag": loq_flag,
                "batch_id": first_source_file(row.source_file),
                "intercept_mode": calib.intercept_mode,
                "calibration_qc_pass": calib.qc_pass,
            }
        )
    return pd.DataFrame.from_records(records)


def _load_standard_predictions(calibrations_dir: Path, form: str) -> pd.DataFrame:
    path = calibrations_dir / f"standards_fitted_{form}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing standards CSV for {form}: {path}")
    df = pd.read_csv(path)
    if "type" in df.columns:
        df = df[df["type"] == "standard"].copy()
    if "fitted_concentration_mg_ml" in df.columns:
        df["conc_mg_ml"] = df["fitted_concentration_mg_ml"]
    elif "known_concentration_mg_ml" in df.columns:
        df["conc_mg_ml"] = df["known_concentration_mg_ml"]
    else:
        raise ValueError(f"{path} must include fitted_concentration_mg_ml or known_concentration_mg_ml")
    df = df[["sample_name", "conc_mg_ml"]].copy()
    return df


def build_total_vs_parts_model(calibrations_dir: Path) -> TotalPartsModel:
    total_df = _load_standard_predictions(calibrations_dir, "total").rename(
        columns={"conc_mg_ml": "conc_total"}
    )
    oxidized_df = _load_standard_predictions(calibrations_dir, "oxidized").rename(
        columns={"conc_mg_ml": "conc_oxidized"}
    )
    reduced_df = _load_standard_predictions(calibrations_dir, "reduced").rename(
        columns={"conc_mg_ml": "conc_reduced"}
    )

    merged = total_df.merge(oxidized_df, on="sample_name").merge(reduced_df, on="sample_name")
    merged["sum_parts"] = merged["conc_oxidized"] + merged["conc_reduced"]

    x = merged["sum_parts"].to_numpy(dtype=float)
    y = merged["conc_total"].to_numpy(dtype=float)

    if len(x) < 3:
        raise ValueError("Not enough standards to build total-vs-parts regression.")

    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(beta[0])
    slope = float(beta[1])
    fitted = X @ beta
    residuals = y - fitted
    n = len(x)
    dof = max(n - 2, 1)
    sse = float(np.sum(residuals**2))
    mse = sse / dof
    rmse = math.sqrt(mse)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - sse / tss if tss > 0 else float("nan")

    try:
        cov = mse * np.linalg.inv(X.T @ X)
        intercept_se = math.sqrt(cov[0, 0])
        slope_se = math.sqrt(cov[1, 1])
    except np.linalg.LinAlgError:
        intercept_se = float("nan")
        slope_se = float("nan")

    return TotalPartsModel(
        intercept=intercept,
        slope=slope,
        intercept_se=intercept_se,
        slope_se=slope_se,
        r_squared=r_squared,
        rmse=rmse,
        dof=dof,
        n=n,
    )


def enforce_consistency(
    df: pd.DataFrame,
    tolerance: float,
    model: TotalPartsModel,
) -> pd.DataFrame:
    pivot = df.pivot_table(index="sample_id", columns="form", values="conc_mg_ml")
    if {"total", "oxidized", "reduced"}.issubset(pivot.columns):
        total = pivot["total"]
        parts = pivot["oxidized"].fillna(0) + pivot["reduced"].fillna(0)
        predicted_total = model.intercept + model.slope * parts
        abs_diff = np.abs(total - predicted_total)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(predicted_total != 0, abs_diff / np.abs(predicted_total), np.nan)
        allowed_abs = np.maximum(tolerance * np.abs(predicted_total), model.rmse)
        failure = (abs_diff > allowed_abs) | np.isnan(rel_diff)
        return pd.DataFrame(
            {
                "total_conc_observed": total,
                "sum_parts_conc": parts,
                "predicted_total_conc": predicted_total,
                "total_vs_parts_abs_diff": abs_diff,
                "total_vs_parts_rel_diff": rel_diff,
                "total_vs_parts_allowed_abs_diff": allowed_abs,
                "total_vs_parts_fail": failure,
            }
        )
    return pd.DataFrame(
        columns=[
            "total_conc_observed",
            "sum_parts_conc",
            "predicted_total_conc",
            "total_vs_parts_abs_diff",
            "total_vs_parts_rel_diff",
            "total_vs_parts_allowed_abs_diff",
            "total_vs_parts_fail",
        ]
    )


def compute_stage_c(
    stage_b_df: pd.DataFrame,
    truth_path: Path,
    extraction_volume_ml: float,
    dilution_factor: float,
) -> pd.DataFrame:
    truth_df = pd.read_csv(truth_path)
    merged = stage_b_df.merge(truth_df, how="left", on="sample_id", suffixes=("", "_truth"))
    factor = extraction_volume_ml * dilution_factor

    def calc_amount(row):
        mass = row.get("dry_biomass_g")
        if mass is None or (isinstance(mass, float) and math.isnan(mass)) or mass == 0:
            return math.nan, math.nan
        amount = row["conc_mg_ml"] * factor / mass
        amount_se = row["conc_se_mg_ml"] * factor / mass
        return amount, amount_se

    amounts = merged.apply(lambda row: calc_amount(row), axis=1, result_type="expand")
    merged["amount_mg_per_gDW"] = amounts[0]
    merged["amount_se_mg_per_gDW"] = amounts[1]
    cols = [
        "sample_id",
        "form",
        "conc_mg_ml",
        "conc_se_mg_ml",
        "amount_mg_per_gDW",
        "amount_se_mg_per_gDW",
        "loq_flag",
        "dry_biomass_g",
        "p_uva_mw_cm2",
        "p_uvb_mw_cm2",
        "uva_pct_mdv",
        "uvb_pct_mdv",
    ]
    return merged[cols]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo_root = args.config.resolve().parent
    calibrations_dir = repo_root / config["paths"]["outputs"]["calibrations_dir"]

    forms = config["scytonemin"]["forms"]
    concentration_frames = []
    for form in forms:
        calib_path = calibrations_dir / f"calibration_{form}.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration JSON missing for {form}: {calib_path}")
        calib = load_calibration(calib_path, form)
        frame = compute_stage_b(form, config, repo_root, calib)
        concentration_frames.append(frame)

    stage_b_df = pd.concat(concentration_frames, ignore_index=True)
    stage_b_df = stage_b_df.sort_values(["sample_id", "form"]).reset_index(drop=True)

    total_parts_model = build_total_vs_parts_model(calibrations_dir)
    model_summary_path = calibrations_dir / "total_vs_parts_model.json"
    model_summary_path.write_text(
        json.dumps(
            {
                "intercept": total_parts_model.intercept,
                "slope": total_parts_model.slope,
                "intercept_se": total_parts_model.intercept_se,
                "slope_se": total_parts_model.slope_se,
                "r_squared": total_parts_model.r_squared,
                "rmse": total_parts_model.rmse,
                "degrees_of_freedom": total_parts_model.dof,
                "n": total_parts_model.n,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tolerance = config["qc"]["total_vs_parts_tolerance"]
    consistency_df = enforce_consistency(stage_b_df, tolerance, total_parts_model)
    if not consistency_df.empty:
        failures = consistency_df["total_vs_parts_fail"].sum()
        print(
            f"[INFO] Total vs parts failures: {failures} of {len(consistency_df)} "
            f"(relative tolerance {tolerance:.3f}, RMSE {total_parts_model.rmse:.4g})"
        )
    consistency_csv = calibrations_dir / "total_vs_parts_consistency.csv"
    consistency_df.reset_index(names="sample_id").to_csv(consistency_csv, index=False)
    stage_b_df = stage_b_df.merge(
        consistency_df,
        how="left",
        left_on="sample_id",
        right_index=True,
    )

    raw_output = repo_root / config["paths"]["outputs"]["calibrations_dir"] / "treatments_concentration_raw.csv"
    stage_b_df.to_csv(raw_output, index=False)

    truth_path = repo_root / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
    extraction_volume_ml = float(config["chemistry"]["extraction_volume_ml"])
    dilution_factor = float(config["chemistry"]["dilution_factor"])
    stage_c_df = compute_stage_c(stage_b_df, truth_path, extraction_volume_ml, dilution_factor)

    corrected_output = repo_root / config["paths"]["outputs"]["calibrations_dir"] / "treatments_corrected_amounts.csv"
    stage_c_df.to_csv(corrected_output, index=False)

    print(f"[INFO] Wrote Stage B results to {raw_output}")
    print(f"[INFO] Wrote Stage C results to {corrected_output}")


if __name__ == "__main__":
    main()
