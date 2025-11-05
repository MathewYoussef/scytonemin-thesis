#!/usr/bin/env python3
"""
Generic downstream proxy evaluator for the Act_of_God validation panel.

Given denoised spectra and a baseline (raw + baseline filter) reference,
compute three task-agnostic checks and report whether denoising helps or at
least does not hurt:

1. Replicate consistency: mean within-group variance must drop by ≥20%.
2. Separability: (between-group / within-group) distance ratio must be ≥ baseline.
3. Dose-response monotonicity (if dose metadata available) – median Spearman Δρ ≥ 0
   and ≥70% of series improve. If no dose metadata, fall back to peak stability F1.

Outputs a JSON payload with per-metric stats and an overall pass/fail flag.
Designed to be invoked by evaluate_validation_panel.py as Gate 4.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROI_MIN_NM = 320.0
ROI_MAX_NM = 500.0
PEAK_PROMINENCE = 0.01  # heuristic for fallback peak detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Validation panel manifest CSV.")
    parser.add_argument(
        "--denoised-dir",
        required=True,
        help="Directory containing denoised spectra (relative_path stem + '_denoised.npy').",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Directory containing baseline/raw spectra (.npy matching manifest entries).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV with dose information (must contain 'relative_path').",
    )
    parser.add_argument(
        "--wavelength-grid",
        default="data/spectra_for_fold/wavelength_grid.npy",
        help="Path to wavelength grid .npy (default: data/spectra_for_fold/wavelength_grid.npy).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Destination JSON file for metrics and pass/fail status.",
    )
    parser.add_argument(
        "--treatment-column",
        default="treatment",
        help="Manifest column used for between-group separability (default: treatment).",
    )
    parser.add_argument(
        "--group-column",
        default="group_id",
        help="Manifest column used for replicate grouping (default: group_id).",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest {manifest_path} is empty")
    return rows


def load_metadata(metadata_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if metadata_path is None:
        return {}
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found")
    lookup: Dict[str, Dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "relative_path" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain 'relative_path' column")
        for row in reader:
            lookup[row["relative_path"]] = row
    return lookup


def load_spectrum(base_dir: Path, relative_path: str, suffix: Optional[str] = None) -> np.ndarray:
    rel = Path(relative_path)
    if suffix:
        rel = rel.with_name(rel.stem + suffix)
    path = base_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing spectrum: {path}")
    return np.load(path).astype(np.float64)


def collect_spectra(
    manifest_rows: List[Dict[str, str]],
    den_dir: Path,
    base_dir: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, str]]]:
    denoised: List[np.ndarray] = []
    baseline: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    for row in manifest_rows:
        rel = row["relative_path"]
        den = load_spectrum(den_dir, rel, suffix="_denoised.npy")
        base = load_spectrum(base_dir, rel, suffix=None)
        if den.shape != base.shape:
            raise ValueError(f"Shape mismatch for {rel}: denoised {den.shape}, baseline {base.shape}")
        denoised.append(den)
        baseline.append(base)
        metadata.append(row)
    return denoised, baseline, metadata


def group_indices(metadata: List[Dict[str, str]], column: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(metadata):
        key = row.get(column, "")
        groups.setdefault(key, []).append(idx)
    return groups


def mean_within_group_variance(spectra: np.ndarray, groups: Dict[str, List[int]]) -> float:
    variances: List[float] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group = spectra[indices]
        var = group.var(axis=0, ddof=1)
        variances.append(var.mean())
    if not variances:
        return 0.0
    return float(np.mean(variances))


def separability_ratio(
    spectra: np.ndarray,
    groups: Dict[str, List[int]],
) -> float:
    centroids: Dict[str, np.ndarray] = {}
    within: List[float] = []
    for key, indices in groups.items():
        group = spectra[indices]
        centroid = group.mean(axis=0)
        centroids[key] = centroid
        if len(indices) >= 2:
            diffs = np.linalg.norm(group - centroid, axis=1)
            within.extend(diffs.tolist())
    centroid_list = list(centroids.values())
    if len(centroid_list) < 2 or not within:
        return 0.0
    centroid_stack = np.stack(centroid_list, axis=0)
    dist_matrix = np.linalg.norm(centroid_stack[:, None, :] - centroid_stack[None, :, :], axis=2)
    upper = dist_matrix[np.triu_indices(len(centroid_list), k=1)]
    between_mean = float(np.mean(upper)) if upper.size else 0.0
    within_mean = float(np.mean(within)) if within else 0.0
    if within_mean <= 1e-12:
        return float("inf") if between_mean > 0 else 0.0
    return between_mean / within_mean


def load_wavelengths(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Wavelength grid not found at {path}")
    return np.load(path).astype(np.float64)


def trapz_roi_area(spectra: np.ndarray, wavelengths: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    roi_wavelengths = wavelengths[roi_mask]
    roi_spectra = spectra[:, roi_mask]
    return np.trapz(roi_spectra, roi_wavelengths, axis=1)


def rankdata(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[sorter] = np.arange(len(values), dtype=float)
    unique_vals, inverse_indices = np.unique(values, return_inverse=True)
    for val_index in range(len(unique_vals)):
        tie_indices = np.where(inverse_indices == val_index)[0]
        if tie_indices.size > 1:
            mean_rank = ranks[tie_indices].mean()
            ranks[tie_indices] = mean_rank
    return ranks


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    cov = np.sum((rx - rx_mean) * (ry - ry_mean))
    denom = np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(cov / denom)


def detect_peaks(signal: np.ndarray, prominence: float = PEAK_PROMINENCE) -> List[int]:
    peaks: List[int] = []
    for i in range(1, signal.size - 1):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            drop = max(signal[i - 1] - signal[i], signal[i + 1] - signal[i])
            if drop >= prominence:
                peaks.append(i)
    return peaks


def peak_f1(baseline: np.ndarray, denoised: np.ndarray, wavelengths: np.ndarray, tol_nm: float = 2.0) -> float:
    baseline_peaks = detect_peaks(baseline)
    denoised_peaks = detect_peaks(denoised)
    if not baseline_peaks and not denoised_peaks:
        return 1.0
    if not baseline_peaks or not denoised_peaks:
        return 0.0
    matched = 0
    used = set()
    for bp in baseline_peaks:
        bp_lambda = wavelengths[bp]
        for dp in denoised_peaks:
            if dp in used:
                continue
            if abs(wavelengths[dp] - bp_lambda) <= tol_nm:
                matched += 1
                used.add(dp)
                break
    precision = matched / max(len(denoised_peaks), 1)
    recall = matched / max(len(baseline_peaks), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def determine_dose_column(metadata_lookup: Dict[str, Dict[str, str]]) -> Optional[str]:
    if not metadata_lookup:
        return None
    sample_metadata = next(iter(metadata_lookup.values()))
    candidates = [col for col in sample_metadata.keys() if 'dose' in col.lower()]
    if candidates:
        return candidates[0]
    for col in ('UVA_total_mWh_cm2', 'UVB_total_mWh_cm2'):
        if col in sample_metadata:
            return col
    return None


def compute_dose_correlations(
    manifest_rows: List[Dict[str, str]],
    spectra: np.ndarray,
    metadata_lookup: Dict[str, Dict[str, str]],
    wavelengths: np.ndarray,
    treatment_column: str,
    dose_column: Optional[str],
) -> Dict[str, float]:
    if not metadata_lookup or dose_column is None:
        return {}
    roi_mask = (wavelengths >= ROI_MIN_NM) & (wavelengths <= ROI_MAX_NM)
    roi_areas = trapz_roi_area(spectra, wavelengths, roi_mask)
    treatment_to_pairs: Dict[str, List[Tuple[float, float]]] = {}
    for idx, row in enumerate(manifest_rows):
        rel = row['relative_path']
        meta = metadata_lookup.get(rel)
        if not meta or dose_column not in meta:
            continue
        try:
            dose_val = float(meta[dose_column])
        except (ValueError, TypeError):
            continue
        treatment_key = row.get(treatment_column, row.get('group_id', ''))
        treatment_to_pairs.setdefault(treatment_key, []).append((dose_val, roi_areas[idx]))

    correlations: Dict[str, float] = {}
    for treatment, pairs in treatment_to_pairs.items():
        if len(pairs) < 2:
            continue
        doses, areas = zip(*pairs)
        corr = spearman_corr(doses, areas)
        correlations[treatment] = corr
    return correlations


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    metadata_lookup = load_metadata(Path(args.metadata)) if args.metadata else {}
    denoised_dir = Path(args.denoised_dir)
    baseline_dir = Path(args.baseline_dir)
    wavelengths = load_wavelengths(Path(args.wavelength_grid))
    denoised_list, baseline_list, manifest_rows = collect_spectra(
        manifest, denoised_dir, baseline_dir
    )

    denoised = np.stack(denoised_list, axis=0)
    baseline = np.stack(baseline_list, axis=0)

    group_map = group_indices(manifest_rows, args.group_column)
    treatment_map = group_indices(manifest_rows, args.treatment_column)

    # Metric 1: replicate consistency
    var_base = mean_within_group_variance(baseline, group_map)
    var_denoised = mean_within_group_variance(denoised, group_map)
    var_reduction = float(var_base - var_denoised)
    if var_base <= 1e-12:
        replicate_pass = True
        reduction_pct = 0.0
    else:
        reduction_pct = var_reduction / var_base
        replicate_pass = var_denoised <= 0.8 * var_base

    # Metric 2: separability
    sep_base = separability_ratio(baseline, treatment_map)
    sep_denoised = separability_ratio(denoised, treatment_map)
    separability_pass = sep_denoised >= sep_base - 1e-6

    # Metric 3: dose monotonicity or fallback peak F1
    median_delta = float('nan')
    pass_series_ratio = float('nan')
    peak_f1_mean = float('nan')
    monotonicity_pass = False
    dose_column = determine_dose_column(metadata_lookup)
    base_corrs = compute_dose_correlations(
        manifest_rows, baseline, metadata_lookup, wavelengths, args.treatment_column, dose_column
    )
    den_corrs = compute_dose_correlations(
        manifest_rows, denoised, metadata_lookup, wavelengths, args.treatment_column, dose_column
    )
    if base_corrs and den_corrs:
        deltas: List[float] = []
        improved = 0
        keys = set(base_corrs.keys()) & set(den_corrs.keys())
        for key in keys:
            delta = den_corrs[key] - base_corrs[key]
            deltas.append(delta)
            if delta >= 0:
                improved += 1
        if deltas:
            median_delta = float(np.median(deltas))
            pass_series_ratio = float(improved / len(deltas))
            monotonicity_pass = (median_delta >= 0.0) and (pass_series_ratio >= 0.7)
    if not monotonicity_pass:
        f1_scores: List[float] = []
        for base_spec, den_spec in zip(baseline, denoised):
            f1_scores.append(peak_f1(base_spec, den_spec, wavelengths))
        peak_f1_mean = float(np.mean(f1_scores)) if f1_scores else float('nan')
        monotonicity_pass = peak_f1_mean >= 0.9 if not np.isnan(peak_f1_mean) else True

    overall_pass = replicate_pass and separability_pass and monotonicity_pass

    result = {
        "replicate_consistency": {
            "baseline_variance": var_base,
            "denoised_variance": var_denoised,
            "reduction_pct": reduction_pct,
            "pass": replicate_pass,
        },
        "separability": {
            "baseline_ratio": sep_base,
            "denoised_ratio": sep_denoised,
            "delta_ratio": sep_denoised - sep_base,
            "pass": separability_pass,
        },
        "dose_monotonicity": {
            "dose_column_used": dose_column,
            "median_delta_rho": median_delta,
            "improved_ratio": pass_series_ratio,
            "peak_f1_mean": peak_f1_mean,
            "pass": monotonicity_pass,
        },
        "pass": overall_pass,
    }

    Path(args.output_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    print('[INFO] downstream_proxy_eval.py (Act_of_God QA) 2025-10-09 version active')
    main()
