#!/usr/bin/env python3
"""
Compute robust dose-level summaries (median + IQR) for chromatogram concentrations and amounts,
and generate summary tables/plots per form.

Outputs:
    _concentrations_vs_dose_with_robust_mean/chromatogram_robust_summary_mg_mL.csv
    _concentrations_vs_dose_with_robust_mean/chromatogram_robust_summary_mg_per_gDW.csv
    (plots saved alongside CSVs)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
mpl_cache = REPO_ROOT / ".mplcache"
mpl_cache.mkdir(parents=True, exist_ok=True)
(mpl_cache / "tmp").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("TMPDIR", str(mpl_cache / "tmp"))

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None

FORMS = ["total", "oxidized", "reduced"]
DOSE_COLUMNS = ["p_uva_mw_cm2", "p_uvb_mw_cm2", "uva_uvb_ratio", "uva_times_uvb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chromatogram robust dose summaries.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as fh:
        yaml.safe_load(fh)
    return path.resolve().parent


def robust_summary(series: pd.Series) -> Tuple[float, float, float]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return median, q1, q3


def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    records = []
    for form in FORMS:
        sub = df[df["form"] == form]
        for _, group in sub.groupby(["p_uva_mw_cm2", "p_uvb_mw_cm2"], dropna=False):
            uva = group["p_uva_mw_cm2"].iloc[0]
            uvb = group["p_uvb_mw_cm2"].iloc[0]
            ratio = group["uva_uvb_ratio"].iloc[0]
            product = group["uva_times_uvb"].iloc[0]
            median, q1, q3 = robust_summary(group[value_col])
            records.append(
                {
                    "form": form,
                    "p_uva_mw_cm2": uva,
                    "p_uvb_mw_cm2": uvb,
                    "uva_uvb_ratio": ratio,
                    "uva_times_uvb": product,
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "n": int(group[value_col].count()),
                }
            )
    return pd.DataFrame.from_records(records)


FORM_COLORS = {
    "total": "#1f77b4",
    "oxidized": "#d62728",
    "reduced": "#2ca02c",
}


def plot_summary(summary_df: pd.DataFrame, dose_col: str, value_label: str, output_dir: Path, suffix: str) -> None:
    if plt is None:
        return
    for form in FORMS:
        form_df = summary_df[summary_df["form"] == form].copy()
        form_df = form_df.sort_values(dose_col)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = form_df[dose_col].to_numpy()
        y = form_df["median"].to_numpy()
        q1 = form_df["q1"].to_numpy()
        q3 = form_df["q3"].to_numpy()
        color = FORM_COLORS.get(form, "#333333")

        ax.scatter(x, y, color=color, zorder=3)
        ax.vlines(x, q1, q3, colors=color, linewidth=1.2, alpha=0.8, zorder=2)
        if len(x) > 1:
            span = max(x) - min(x)
            half_width = max(span * 0.015, 0.01)
        else:
            half_width = 0.01
        ax.hlines(q1, x - half_width, x + half_width, colors=color, linewidth=1.0, alpha=0.8, zorder=2)
        ax.hlines(q3, x - half_width, x + half_width, colors=color, linewidth=1.0, alpha=0.8, zorder=2)

        ax.set_xlabel(dose_col.replace("_", " "))
        ax.set_ylabel(value_label)
        ax.set_title(f"{form.capitalize()} {value_label} vs {dose_col}")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        fig.tight_layout()
        out_path = output_dir / f"{form}_{suffix}_{dose_col}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = load_config(args.config)

    raw_df = pd.read_csv(repo / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv")
    corrected_df = pd.read_csv(repo / "DAD_to_Concentration_AUC" / "treatments_corrected_amounts.csv")
    truth_df = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")[
        ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2", "uva_pct_mdv", "uvb_pct_mdv"]
    ]
    merged = raw_df.merge(
        corrected_df[["sample_id", "form", "amount_mg_per_gDW"]],
        on=["sample_id", "form"],
        how="left",
    ).merge(truth_df, on="sample_id", how="left")
    merged["p_uva_mw_cm2"] = merged["p_uva_mw_cm2"].fillna(0.0)
    merged["p_uvb_mw_cm2"] = merged["p_uvb_mw_cm2"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["uva_uvb_ratio"] = merged["p_uva_mw_cm2"] / merged["p_uvb_mw_cm2"]
    merged["uva_uvb_ratio"] = merged["uva_uvb_ratio"].replace([np.inf, -np.inf], np.nan)
    merged["uva_times_uvb"] = merged["p_uva_mw_cm2"] * merged["p_uvb_mw_cm2"]
    merged["uva_uvb_ratio"] = merged["uva_uvb_ratio"].fillna(0.0)

    output_dir = repo / "_concentrations_vs_dose_with_robust_mean"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_mg_ml = summarize(merged, "conc_mg_ml")
    summary_mg_ml.to_csv(output_dir / "chromatogram_robust_summary_mg_mL.csv", index=False)
    summary_mg_gdw = summarize(merged, "amount_mg_per_gDW")
    summary_mg_gdw.to_csv(output_dir / "chromatogram_robust_summary_mg_per_gDW.csv", index=False)

    for dose_col in ["p_uva_mw_cm2", "p_uvb_mw_cm2", "uva_uvb_ratio", "uva_times_uvb"]:
        plot_summary(summary_mg_ml, dose_col, "Concentration (mg/mL)", output_dir, "chromatogram_mg_mL")
        plot_summary(summary_mg_gdw, dose_col, "Amount (mg/gDW)", output_dir, "chromatogram_mg_per_gDW")

    print("Chromatogram robust summaries written to", output_dir)


if __name__ == "__main__":
    main()
