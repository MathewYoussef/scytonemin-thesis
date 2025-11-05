#!/usr/bin/env python3
"""Compile key statistical artifacts into markdown summaries.

All numbers are taken directly from previously generated CSV/JSON files in
``stats_for_paper/``. No new model fitting is performed; the script only groups
and formats existing results so they are easier to quote in the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = PROJECT_ROOT / "stats_for_paper"


def _write_markdown(name: str, sections: Iterable[str]) -> None:
    path = STATS_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n\n".join(section.strip() for section in sections if section))
        fh.write("\n")


def _df_to_markdown(df: pd.DataFrame, heading: str | None = None) -> str:
    if df.empty:
        return ""
    formatted = df.copy()
    for col in formatted.select_dtypes(include="number").columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    table = formatted.to_markdown(index=False)
    if heading:
        return f"## {heading}\n\n{table}"
    return table


def build_calibration_summary() -> None:
    sections: list[str] = ["# Calibration Summary"]

    # DAD calibrations (total, oxidized, reduced)
    rows = []
    for form in ["total", "oxidized", "reduced"]:
        json_path = STATS_DIR / f"calibration_{form}.json"
        if not json_path.exists():
            continue
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        rows.append(
            {
                "form": form,
                "slope": data.get("slope"),
                "intercept": data.get("intercept"),
                "slope_se": data.get("slope_se"),
                "intercept_se": data.get("intercept_se"),
                "r_squared": data.get("r_squared"),
                "max_abs_rel_residual": data.get("max_abs_rel_residual"),
                "df": data.get("degrees_of_freedom"),
            }
        )
    if rows:
        sections.append(
            _df_to_markdown(pd.DataFrame(rows), "DAD AUC calibrations (values taken from calibration_*.json)"),
        )

    # Chromatogram standards (raw areas vs known concentrations)
    chrom_tables = []
    for form in ["total", "oxidized", "reduced"]:
        csv_path = STATS_DIR / f"raw_{form}_scytonemin.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        standards = df[df["sample_category"] == "standard"][
            [
                "standard_level",
                "standard_concentration_mg_ml",
                "area",
            ]
        ].dropna()
        standards = standards.sort_values("standard_level")
        chrom_tables.append(
            _df_to_markdown(standards, f"Chromatogram standards — {form} form"),
        )
    sections.extend(chrom_tables)
    _write_markdown("calibration_summary.md", sections)


def build_single_factor_summary() -> None:
    sections: list[str] = ["# Single-factor Dose Regressions"]
    regression_path = STATS_DIR / "regression_summary.csv"
    if regression_path.exists():
        reg = pd.read_csv(regression_path)
        sections.append(
            _df_to_markdown(
                reg.rename(
                    columns={
                        "driver": "dose_metric",
                        "form": "metric",
                        "slope": "slope",
                        "intercept": "intercept",
                        "r_squared": "R^2",
                        "n": "n",
                    }
                ),
                "Chromatogram regressions (from regression_summary.csv)",
            )
        )

    for csv_name, label in [
        ("chromatogram_dose_correlations.csv", "Dose correlations (Pearson/Spearman)"),
        ("Chromatogram_derived_concentrations.csv", "Sample-level chromatogram concentrations (excerpt)"),
        ("DAD_derived_concentrations_corrected.csv", "Sample-level DAD concentrations (excerpt)"),
    ]:
        path = STATS_DIR / csv_name
        if path.exists():
            df = pd.read_csv(path)
            if len(df) > 12:
                df = df.head(12)
            sections.append(_df_to_markdown(df, label))

    _write_markdown("single_factor_summary.md", sections)


def _load_anova_summary() -> pd.DataFrame:
    records = []
    file_specs = [
        ("chromatogram_two_way_anova.csv", "raw"),
        ("chromatogram_two_way_anova_delta.csv", "delta"),
        ("chromatogram_two_way_anova_pct.csv", "percent"),
        ("chromatogram_two_way_anova_ratio.csv", "ratio"),
        ("chromatogram_two_way_anova_zscore.csv", "zscore"),
        ("dad_two_way_anova.csv", "raw"),
        ("dad_two_way_anova_delta.csv", "delta"),
        ("dad_two_way_anova_pct.csv", "percent"),
        ("dad_two_way_anova_ratio.csv", "ratio"),
        ("dad_two_way_anova_zscore.csv", "zscore"),
    ]
    term_map = {
        "C(p_uva_mw_cm2)": "UVA",
        "C(p_uvb_mw_cm2)": "UVB",
        "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)": "Interaction",
    }
    for filename, variant in file_specs:
        path = STATS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for (dataset, measurement), group in df.groupby(["dataset", "measurement"]):
            record: dict[str, object] = {
                "dataset": dataset,
                "variant": variant,
                "measurement": measurement,
            }
            for term, label in term_map.items():
                row = group[group["term"] == term]
                if row.empty:
                    continue
                rec = row.iloc[0]
                record[f"{label}_df"] = rec["df"]
                record[f"{label}_F"] = rec["F"]
                record[f"{label}_p"] = rec["PR(>F)"]
                if "eta_squared" in rec:
                    record[f"{label}_eta2"] = rec["eta_squared"]
            records.append(record)
    return pd.DataFrame(records)


def build_two_factor_summary() -> None:
    sections: list[str] = ["# Two-factor Modeling Summary"]

    anova_df = _load_anova_summary()
    if not anova_df.empty:
        chrom = anova_df[anova_df["dataset"] == "chromatogram"].copy()
        dad = anova_df[anova_df["dataset"] == "dad"].copy()
        if not chrom.empty:
            sections.append(
                _df_to_markdown(
                    chrom.drop(columns=["dataset"]).sort_values(["variant", "measurement"]),
                    "Classical ANOVA — Chromatogram",
                )
            )
        if not dad.empty:
            sections.append(
                _df_to_markdown(
                    dad.drop(columns=["dataset"]).sort_values(["variant", "measurement"]),
                    "Classical ANOVA — DAD",
                )
            )

    for fname, label in [
        ("chromatogram_two_way_anova_robust.csv", "HC3-robust ANOVA — Chromatogram"),
        ("chromatogram_two_way_anova_rank.csv", "Rank-transformed ANOVA — Chromatogram"),
        ("dad_two_way_anova_robust.csv", "HC3-robust ANOVA — DAD"),
        ("dad_two_way_anova_rank.csv", "Rank-transformed ANOVA — DAD"),
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            filtered = df[df["term"].isin([
                "C(p_uva_mw_cm2)",
                "C(p_uvb_mw_cm2)",
                "C(p_uva_mw_cm2):C(p_uvb_mw_cm2)",
            ])]
            sections.append(_df_to_markdown(filtered, label))

    coeff_tables = []
    for fname, label in [
        ("chromatogram_regression_coeffs.csv", "OLS coefficients — Chromatogram (raw metrics)"),
        ("chromatogram_regression_coeffs_delta.csv", "OLS coefficients — Chromatogram (delta)"),
        ("chromatogram_regression_coeffs_ratio.csv", "OLS coefficients — Chromatogram (ratio)"),
        ("chromatogram_regression_coeffs_pct.csv", "OLS coefficients — Chromatogram (percent)"),
        ("chromatogram_regression_coeffs_zscore.csv", "OLS coefficients — Chromatogram (z-score)"),
        ("dad_regression_coeffs.csv", "OLS coefficients — DAD (raw)"),
        ("dad_regression_coeffs_delta.csv", "OLS coefficients — DAD (delta)"),
        ("dad_regression_coeffs_ratio.csv", "OLS coefficients — DAD (ratio)"),
        ("dad_regression_coeffs_pct.csv", "OLS coefficients — DAD (percent)"),
        ("dad_regression_coeffs_zscore.csv", "OLS coefficients — DAD (z-score)"),
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            coeff_tables.append(_df_to_markdown(df, label))
    sections.extend(coeff_tables)

    # Ridge and PLS summaries
    for fname, label in [
        ("chromatogram_ridge_results.csv", "Ridge regression (chromatogram)") ,
        ("dad_ridge_results.csv", "Ridge regression (DAD)") ,
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            sections.append(_df_to_markdown(df, label))

    for fname, label in [
        ("chromatogram_pls_results.csv", "PLS regression (chromatogram)"),
        ("dad_pls_results.csv", "PLS regression (DAD)"),
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            sections.append(_df_to_markdown(df, label))

    if (STATS_DIR / "ridge_bootstrap_summary.csv").exists():
        boot = pd.read_csv(STATS_DIR / "ridge_bootstrap_summary.csv")
        sections.append(_df_to_markdown(boot, "Bootstrap ridge coefficient intervals"))

    # Permutation tests
    perm_sections = []
    for fname, label in [
        ("chromatogram_two_way_anova_permutation.csv", "Simple permutation ANOVA (chromatogram)") ,
        ("dad_two_way_anova_permutation.csv", "Simple permutation ANOVA (DAD)") ,
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            perm_sections.append(_df_to_markdown(df, label))
    if perm_sections:
        sections.extend(perm_sections)

    if (STATS_DIR / "freedman_lane_interaction.csv").exists():
        freedman = pd.read_csv(STATS_DIR / "freedman_lane_interaction.csv")
        sections.append(_df_to_markdown(freedman, "Freedman–Lane permutation test"))

    _write_markdown("two_factor_summary.md", sections)


def build_diagnostics_summary() -> None:
    sections: list[str] = ["# Diagnostics and Sensitivity"]

    for fname, label in [
        ("anova_residual_diagnostics.csv", "Residual diagnostics"),
        ("regression_vif.csv", "Variance inflation factors"),
        ("uva_uvb_correlations.csv", "UVA/UVB Pearson correlation"),
        ("outlier_sensitivity.csv", "Outlier removal scenarios"),
    ]:
        path = STATS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            sections.append(_df_to_markdown(df, label))

    _write_markdown("diagnostics_and_sensitivity_summary.md", sections)


def main() -> None:
    if not STATS_DIR.exists():
        raise SystemExit("stats_for_paper directory not found")

    build_calibration_summary()
    build_single_factor_summary()
    build_two_factor_summary()
    build_diagnostics_summary()


if __name__ == "__main__":
    main()
