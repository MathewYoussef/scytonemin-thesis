#!/usr/bin/env python3
"""
Extract spectral features from DAD tidy spectra within the configured wavelength window.

Outputs:
    DAD_feature_extraction_replicate_exploration/dad_features.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class Config:
    wl_min: float
    wl_max: float
    sg_window: int
    sg_polyorder: int
    polite_min_points: int = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DAD spectral features.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional override for the output CSV.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Tuple[Config, Path]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    spectral = raw.get("spectral", {})
    cfg = Config(
        wl_min=float(spectral.get("wl_min_nm", 320.0)),
        wl_max=float(spectral.get("wl_max_nm", 480.0)),
        sg_window=int(spectral.get("sg_window", 11)),
        sg_polyorder=int(spectral.get("sg_polyorder", 3)),
    )
    repo = path.resolve().parent
    return cfg, repo


def _interp_at(wl: np.ndarray, intensity: np.ndarray, target: float) -> float:
    if target < wl.min() or target > wl.max():
        return float("nan")
    return float(np.interp(target, wl, intensity))


def _auc(wl: np.ndarray, intensity: np.ndarray, lower: float, upper: float) -> float:
    mask = (wl >= lower) & (wl <= upper)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(intensity[mask], wl[mask]))


def _fwhm(wl: np.ndarray, intensity: np.ndarray) -> float:
    if len(intensity) == 0:
        return float("nan")
    baseline_shift = intensity - intensity.min()
    peak = baseline_shift.max()
    if peak <= 0:
        return float("nan")
    half = peak / 2.0
    idx_max = np.argmax(baseline_shift)
    left_wl = wl[: idx_max + 1]
    left_int = baseline_shift[: idx_max + 1]
    right_wl = wl[idx_max:]
    right_int = baseline_shift[idx_max:]

    if left_int.min() > half or right_int.min() > half:
        return float("nan")

    left_half = np.interp(half, left_int[::-1], left_wl[::-1])
    right_half = np.interp(half, right_int, right_wl)
    return float(right_half - left_half)


def _second_derivative(wl: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    first = np.gradient(intensity, wl)
    second = np.gradient(first, wl)
    return second


def compute_features(wl: np.ndarray, intensity: np.ndarray) -> Dict[str, float]:
    order = np.argsort(wl)
    wl = wl[order]
    intensity = intensity[order]
    features: Dict[str, float] = {}
    if len(wl) < 2:
        return features

    features["auc_320_480"] = float(np.trapezoid(intensity, wl))
    idx_max = np.argmax(intensity)
    features["abs_max"] = float(intensity[idx_max])
    features["lambda_max_nm"] = float(wl[idx_max])

    weights = np.clip(intensity, a_min=0.0, a_max=None)
    if weights.sum() > 0:
        features["centroid_nm"] = float(np.sum(wl * weights) / np.sum(weights))
    else:
        features["centroid_nm"] = float("nan")

    features["fwhm_nm"] = _fwhm(wl, intensity)

    features["abs_384_nm"] = _interp_at(wl, intensity, 384.0)
    features["abs_400_nm"] = _interp_at(wl, intensity, 400.0)
    if not np.isnan(features["abs_400_nm"]) and features["abs_400_nm"] != 0:
        features["ratio_abs_384_400"] = features["abs_384_nm"] / features["abs_400_nm"]
    else:
        features["ratio_abs_384_400"] = float("nan")

    auc_320_380 = _auc(wl, intensity, 320.0, 380.0)
    auc_380_440 = _auc(wl, intensity, 380.0, 440.0)
    features["auc_320_380"] = auc_320_380
    features["auc_380_440"] = auc_380_440
    if not np.isnan(auc_380_440) and auc_380_440 != 0:
        features["auc_ratio_320_380_over_380_440"] = auc_320_380 / auc_380_440
    else:
        features["auc_ratio_320_380_over_380_440"] = float("nan")

    second = _second_derivative(wl, intensity)
    features["second_derivative_384"] = _interp_at(wl, second, 384.0)
    features["second_derivative_400"] = _interp_at(wl, second, 400.0)

    return features


def main() -> None:
    args = parse_args()
    cfg, repo_root = load_config(args.config)

    tidy_path = repo_root / "Compiled_DAD_DATA" / "Scytonemin" / "scytonemin_spectra_tidy.csv"
    spectra_df = pd.read_csv(tidy_path)
    spectra_df = spectra_df[spectra_df["analyte"] == "Scytonemin"].copy()

    window_mask = (spectra_df["wavelength_nm"] >= cfg.wl_min) & (spectra_df["wavelength_nm"] <= cfg.wl_max)
    spectra_df = spectra_df[window_mask]

    groups = spectra_df.groupby(["sample_id", "sample_category", "spectrum_state"], sort=False)

    records = []
    for (sample_id, sample_category, spectrum_state), group in groups:
        wl = group["wavelength_nm"].to_numpy(dtype=float)
        intensity = group["intensity_abs"].to_numpy(dtype=float)
        feats = compute_features(wl, intensity)
        record = {
            "sample_id": sample_id,
            "sample_category": sample_category,
            "spectrum_state": spectrum_state,
        }
        record.update(feats)
        records.append(record)

    features_df = pd.DataFrame.from_records(records)
    output = (
        args.output
        if args.output
        else repo_root / "DAD_feature_extraction_replicate_exploration" / "dad_features.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output, index=False)
    print(f"Wrote DAD spectral features to {output}")


if __name__ == "__main__":
    main()
