#!/usr/bin/env python3
"""
Run one-way ANOVA, Kruskal-Wallis, and effect-size summaries to test whether
chromatogram/DAD concentrations separate by UV dose.

Outputs (saved under `_concentrations_vs_dose_with_robust_mean/`):
    chromatogram_dose_stats.csv
    chromatogram_dose_group_stats.csv
    dad_dose_stats.csv
    dad_dose_group_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import yaml

FORMS = ["total", "oxidized", "reduced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dose-effect statistics for chromatogram and DAD concentrations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--chrom-file",
        type=Path,
        help="Optional chromatogram CSV to analyze instead of defaults.",
    )
    parser.add_argument(
        "--chrom-metrics",
        type=str,
        help="Comma-separated chromatogram columns to test (default: conc_mg_ml,amount_mg_per_gDW).",
    )
    parser.add_argument(
        "--dad-file",
        type=Path,
        help="Optional DAD CSV to analyze instead of defaults.",
    )
    parser.add_argument(
        "--dad-metrics",
        type=str,
        help="Comma-separated DAD columns to test (default: predicted_* mg/mL and mg/gDW columns).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag to append to output filenames (e.g., 'delta').",
    )
    return parser.parse_args()


def load_repo_root(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)
    return config_path.resolve().parent


def infer_form(metric: str) -> str:
    for form in FORMS:
        if form in metric:
            return form
    return "n/a"


def infer_measurement_type(metric: str) -> str:
    if "mg_per_gDW" in metric:
        return "mg_per_gDW"
    if "mg_ml" in metric:
        return "mg_ml"
    return metric


def anova_and_kw(groups: Dict[float, np.ndarray]) -> Tuple[float, float, float, float, float]:
    """Return F, p, eta_sq, H, kw_p."""
    valid_groups = [g for g in groups.values() if len(g) > 1]
    if len(valid_groups) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    f_stat, p_value = stats.f_oneway(*valid_groups)

    all_values = np.concatenate(valid_groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in valid_groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in valid_groups)
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan

    try:
        h_stat, kw_p = stats.kruskal(*valid_groups)
    except ValueError:
        h_stat, kw_p = (np.nan, np.nan)

    return float(f_stat), float(p_value), float(eta_sq), float(h_stat), float(kw_p)


def collect_chrom_group_stats(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    records = []
    for form in FORMS:
        form_df = df[df["form"] == form]
        if form_df.empty:
            continue
        for metric in metrics:
            if metric not in form_df.columns:
                continue
            for dose, group in form_df.groupby("p_uva_mw_cm2"):
                values = group[metric].dropna().to_numpy()
                records.append(
                    {
                        "form": form,
                        "measurement": metric,
                        "p_uva_mw_cm2": dose,
                        "n": len(values),
                        "mean": np.mean(values) if len(values) else np.nan,
                        "median": np.median(values) if len(values) else np.nan,
                        "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                        "iqr": np.percentile(values, 75) - np.percentile(values, 25) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def analyze_chrom_dataset(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    records = []
    for form in FORMS:
        form_df = df[df["form"] == form]
        if form_df.empty:
            continue
        for metric in metrics:
            if metric not in form_df.columns:
                continue
            subset = form_df.dropna(subset=[metric, "p_uva_mw_cm2"])
            if subset.empty:
                continue
            groups = {
                dose: group[metric].dropna().to_numpy(dtype=float)
                for dose, group in subset.groupby("p_uva_mw_cm2")
            }
            f_stat, p_value, eta_sq, h_stat, kw_p = anova_and_kw(groups)
            records.append(
                {
                    "form": form,
                    "measurement": metric,
                    "anova_F": f_stat,
                    "anova_p": p_value,
                    "eta_squared": eta_sq,
                    "kruskal_H": h_stat,
                    "kruskal_p": kw_p,
                    "n_groups": sum(len(g) > 0 for g in groups.values()),
                }
            )
    return pd.DataFrame.from_records(records)


def analyze_dad_dataset(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    records = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        subset = df.dropna(subset=[metric, "p_uva_mw_cm2"])
        if subset.empty:
            continue
        groups = {
            dose: group[metric].dropna().to_numpy(dtype=float)
            for dose, group in subset.groupby("p_uva_mw_cm2")
        }
        f_stat, p_value, eta_sq, h_stat, kw_p = anova_and_kw(groups)
        records.append(
            {
                "form": infer_form(metric),
                "measurement": metric,
                "anova_F": f_stat,
                "anova_p": p_value,
                "eta_squared": eta_sq,
                "kruskal_H": h_stat,
                "kruskal_p": kw_p,
                "n_groups": sum(len(g) > 0 for g in groups.values()),
            }
        )
    return pd.DataFrame.from_records(records)


def collect_dad_group_stats(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    records = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for dose, group in df.groupby("p_uva_mw_cm2"):
            values = group[metric].dropna().to_numpy()
            records.append(
                {
                    "form": infer_form(metric),
                    "measurement": metric,
                    "p_uva_mw_cm2": dose,
                    "n": len(values),
                    "mean": np.mean(values) if len(values) else np.nan,
                    "median": np.median(values) if len(values) else np.nan,
                    "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "iqr": np.percentile(values, 75) - np.percentile(values, 25) if len(values) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def load_chrom_file(repo: Path, chrom_file: Optional[Path]) -> pd.DataFrame:
    if chrom_file is None:
        raw_path = repo / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv"
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        df = pd.read_csv(raw_path)
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        df = df.merge(truth_df, on="sample_id", how="left")
        return df
    path = chrom_file if chrom_file.is_absolute() else repo / chrom_file
    return pd.read_csv(path)


def run_chromatogram(
    repo: Path,
    chrom_file: Optional[Path],
    metrics: Optional[List[str]],
    tag: Optional[str],
) -> None:
    chrom_df = load_chrom_file(repo, chrom_file)
    if "p_uva_mw_cm2" not in chrom_df.columns:
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        chrom_df = chrom_df.merge(truth_df, on="sample_id", how="left")

    if metrics is None:
        metrics = [m for m in ["conc_mg_ml", "amount_mg_per_gDW"] if m in chrom_df.columns]

    stats_df = analyze_chrom_dataset(chrom_df, metrics)
    groups_df = collect_chrom_group_stats(chrom_df, metrics)

    suffix = f"_{tag}" if tag else ""
    out_dir = repo / "_concentrations_vs_dose_with_robust_mean"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / f"chromatogram_dose_stats{suffix}.csv", index=False)
    groups_df.to_csv(out_dir / f"chromatogram_dose_group_stats{suffix}.csv", index=False)


def load_dad_file(repo: Path, dad_file: Optional[Path]) -> pd.DataFrame:
    if dad_file is None:
        return pd.read_csv(repo / "DAD_derived_concentrations_corrected.csv")
    path = dad_file if dad_file.is_absolute() else repo / dad_file
    return pd.read_csv(path)


def default_dad_metrics(df: pd.DataFrame) -> List[str]:
    candidates = []
    for col in df.columns:
        if col.startswith("predicted_") and (col.endswith("_mg_ml") or col.endswith("_mg_per_gDW")) and all(suffix not in col for suffix in ["_delta", "_ratio", "_pct", "_zscore"]):
            candidates.append(col)
    return candidates


def run_dad(
    repo: Path,
    dad_file: Optional[Path],
    metrics: Optional[List[str]],
    tag: Optional[str],
) -> None:
    df = load_dad_file(repo, dad_file)
    if "p_uva_mw_cm2" not in df.columns or "p_uvb_mw_cm2" not in df.columns:
        truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
        truth_df = pd.read_csv(truth_path)[["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        df = df.merge(truth_df, on="sample_id", how="left")

    if metrics is None:
        metrics = default_dad_metrics(df)

    stats_df = analyze_dad_dataset(df, metrics)
    group_df = collect_dad_group_stats(df, metrics)

    suffix = f"_{tag}" if tag else ""
    out_dir = repo / "_concentrations_vs_dose_with_robust_mean"
    stats_df.to_csv(out_dir / f"dad_dose_stats{suffix}.csv", index=False)
    group_df.to_csv(out_dir / f"dad_dose_group_stats{suffix}.csv", index=False)


def main() -> None:
    args = parse_args()
    repo_root = load_repo_root(args.config)
    chrom_metrics = [m.strip() for m in args.chrom_metrics.split(",")] if args.chrom_metrics else None
    dad_metrics = [m.strip() for m in args.dad_metrics.split(",")] if args.dad_metrics else None
    run_chromatogram(repo_root, args.chrom_file, chrom_metrics, args.tag)
    run_dad(repo_root, args.dad_file, dad_metrics, args.tag)
    print("Dose-effect statistics written to _concentrations_vs_dose_with_robust_mean/")


if __name__ == "__main__":
    main()
