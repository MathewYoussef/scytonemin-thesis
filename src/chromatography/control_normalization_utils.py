#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

FORMS = ["total", "oxidized", "reduced"]


def load_chromatogram(repo_root: Path) -> pd.DataFrame:
    """Load chromatogram concentrations with dose metadata and corrected amounts."""
    raw_path = repo_root / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv"
    corrected_path = repo_root / "DAD_to_Concentration_AUC" / "treatments_corrected_amounts.csv"
    truth_path = repo_root / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"

    raw_df = pd.read_csv(raw_path)
    truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
    raw_df = raw_df.merge(truth_df, how="left", on="sample_id")
    raw_df["p_uva_mw_cm2"] = raw_df["p_uva_mw_cm2"].fillna(0.0)
    raw_df["p_uvb_mw_cm2"] = raw_df["p_uvb_mw_cm2"].fillna(0.0)

    corrected_df = pd.read_csv(corrected_path)[["sample_id", "form", "amount_mg_per_gDW"]]
    merged = raw_df.merge(corrected_df, how="left", on=["sample_id", "form"])
    return merged


def load_dad(repo_root: Path) -> pd.DataFrame:
    """Load DAD-derived concentrations with dose metadata."""
    dad_path = repo_root / "DAD_derived_concentrations_corrected.csv"
    truth_path = repo_root / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"

    dad_df = pd.read_csv(dad_path)
    if "p_uva_mw_cm2" not in dad_df.columns or "p_uvb_mw_cm2" not in dad_df.columns:
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        dad_df = dad_df.merge(truth_df, how="left", on="sample_id")
    dad_df["p_uva_mw_cm2"] = dad_df["p_uva_mw_cm2"].fillna(0.0)
    dad_df["p_uvb_mw_cm2"] = dad_df["p_uvb_mw_cm2"].fillna(0.0)
    return dad_df


def _summary_stats(values: pd.Series) -> Dict[str, Any]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"median": np.nan, "mean": np.nan, "std": np.nan, "iqr": np.nan, "count": 0}
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "iqr": float(q3 - q1),
        "count": int(len(values)),
    }


def compute_control_baselines(chrom_df: pd.DataFrame, dad_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute control (UVA=UVB=0) stats for chromatogram and DAD metrics."""
    baselines: Dict[str, Any] = {"chromatogram": {}, "dad": {}}

    for form in FORMS:
        mask = (
            (chrom_df["form"] == form)
            & (chrom_df["p_uva_mw_cm2"] == 0.0)
            & (chrom_df["p_uvb_mw_cm2"] == 0.0)
        )
        control = chrom_df[mask]
        baselines["chromatogram"][form] = {
            "conc_mg_ml": _summary_stats(control["conc_mg_ml"]) if "conc_mg_ml" in control else {},
            "amount_mg_per_gDW": _summary_stats(control["amount_mg_per_gDW"]) if "amount_mg_per_gDW" in control else {},
        }

    dad_control = dad_df[(dad_df["p_uva_mw_cm2"] == 0.0) & (dad_df["p_uvb_mw_cm2"] == 0.0)]
    value_cols = [
        col
        for col in dad_df.columns
        if col.startswith("predicted_") and (col.endswith("_mg_ml") or col.endswith("_mg_per_gDW"))
    ]
    for col in value_cols:
        baselines["dad"][col] = _summary_stats(dad_control[col])

    return baselines


def save_baselines(baselines: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(baselines, fh, indent=2)


def load_baselines(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_baselines(repo_root: Path) -> Dict[str, Any]:
    """Load existing baselines or compute and save new ones."""
    baseline_path = repo_root / "Exploring_control_normalized" / "control_summary" / "control_baselines.json"
    if baseline_path.exists():
        return load_baselines(baseline_path)

    chrom_df = load_chromatogram(repo_root)
    dad_df = load_dad(repo_root)
    baselines = compute_control_baselines(chrom_df, dad_df)
    save_baselines(baselines, baseline_path)
    return baselines
