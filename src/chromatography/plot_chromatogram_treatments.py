#!/usr/bin/env python3
"""
Create chromatogram plots overlaying standards and treatment predictions, grouped by UVA dose.

Outputs per form:
- Chromatogram_Calibration_Plots/{form}_treatment_overlay.svg   (fallback SVG if matplotlib unavailable)
- Chromatogram_Calibration_Plots/{form}_treatment_overlay.csv   (data used for plotting)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot chromatogram calibration curve with treatment predictions grouped by UVA dose."
    )
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


def load_standards(form: str, chrom_dir: Path, standards_lookup: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(chrom_dir / f"raw_{form}_scytonemin.csv")
    std = df[df["sample_category"] == "standard"].copy()
    std["standard_level"] = std["standard_level"].astype(float)
    std = std.dropna(subset=["area"])
    std["area"] = std["area"].astype(float)
    std["known_concentration_mg_ml"] = std["standard_concentration_mg_ml"]
    missing = std["known_concentration_mg_ml"].isna()
    if missing.any():
        lookup_dict = dict(
            zip(
                standards_lookup["standard_level"].astype(float),
                standards_lookup["concentration_mg_ml"].astype(float),
            )
        )
        std.loc[missing, "known_concentration_mg_ml"] = std.loc[missing, "standard_level"].map(lookup_dict)
    std = std.dropna(subset=["known_concentration_mg_ml"])
    std["sample_id"] = std["sample_name"]
    std["sample_type"] = "standard"
    std["uva_dose"] = np.nan
    std["uvb_dose"] = np.nan
    std["uva_uvb_ratio"] = np.nan
    return std[
        [
            "sample_id",
            "area",
            "known_concentration_mg_ml",
            "sample_type",
            "uva_dose",
            "uvb_dose",
            "uva_uvb_ratio",
        ]
    ]


def load_treatments(form: str, chrom_dir: Path, truth_df: pd.DataFrame, stage_b_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(chrom_dir / f"raw_{form}_scytonemin.csv")
    treat = df[df["sample_category"] == "sample"].copy()
    treat["sample_id"] = treat["sample_name"].apply(extract_sample_id)
    treat = treat.merge(
        stage_b_df[stage_b_df["form"] == form][["sample_id", "conc_mg_ml"]],
        how="left",
        on="sample_id",
    )
    treat = treat.merge(
        truth_df[["sample_id", "p_uva_mw_cm2"]],
        how="left",
        on="sample_id",
    )
    treat = treat.merge(
        truth_df[["sample_id", "p_uvb_mw_cm2"]],
        how="left",
        on="sample_id",
    )
    treat = treat.dropna(subset=["area"])
    treat["area"] = treat["area"].astype(float)
    treat["sample_type"] = "treatment"
    treat = treat.rename(
        columns={
            "conc_mg_ml": "predicted_concentration_mg_ml",
            "p_uva_mw_cm2": "uva_dose",
            "p_uvb_mw_cm2": "uvb_dose",
        }
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        treat["uva_uvb_ratio"] = treat["uva_dose"] / treat["uvb_dose"]
    return treat[
        [
            "sample_id",
            "area",
            "predicted_concentration_mg_ml",
            "sample_type",
            "uva_dose",
            "uvb_dose",
            "uva_uvb_ratio",
        ]
    ]


def extract_sample_id(name: str) -> str:
    if not isinstance(name, str):
        return ""
    tokens = name.strip().split()
    return tokens[-1] if tokens else name


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _svg_number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _generate_svg(path: Path, data: pd.DataFrame, form: str, dose_col: str, dose_label: str) -> None:
    width, height = 900, 420
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    min_area, max_area = data["area"].min(), data["area"].max()
    min_conc = min(data["concentration_mg_ml"].min(), 0)
    max_conc = data["concentration_mg_ml"].max()
    span_area = max_area - min_area if max_area > min_area else 1.0
    span_conc = max_conc - min_conc if max_conc > min_conc else 1.0

    def map_x(val: float) -> float:
        return margin + (val - min_area) / span_area * plot_width

    def map_y(val: float) -> float:
        return margin + plot_height - (val - min_conc) / span_conc * plot_height

    elements = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />']
    title = f"Chromatogram area vs concentration ({form})"
    elements.append(
        f'<text x="{width/2:.1f}" y="35" font-family="Helvetica" font-size="18" text-anchor="middle">{title}</text>'
    )

    # Axes
    x0, y0 = margin, margin + plot_height
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_width}" y2="{y0}" stroke="#444" stroke-width="1.5"/>')
    elements.append(f'<line x1="{x0}" y1="{margin}" x2="{x0}" y2="{y0}" stroke="#444" stroke-width="1.5"/>')
    elements.append(
        f'<text x="{margin + plot_width/2:.1f}" y="{height - 10:.1f}" font-family="Helvetica" font-size="14" text-anchor="middle">'
        "Peak area</text>"
    )
    elements.append(
        f'<text x="20" y="{margin + plot_height/2:.1f}" font-family="Helvetica" font-size="14" text-anchor="middle" transform="rotate(-90 20 {margin + plot_height/2:.1f})">'
        "Concentration (mg/mL)</text>"
    )

    # Tick marks (4 each)
    for i in range(4):
        area_val = min_area + span_area * i / 3
        x = map_x(area_val)
        elements.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 6}" stroke="#666"/>')
        elements.append(
            f'<text x="{x:.1f}" y="{y0 + 22:.1f}" font-family="Helvetica" font-size="12" text-anchor="middle">{_svg_number(area_val)}</text>'
        )
        conc_val = min_conc + span_conc * i / 3
        y = map_y(conc_val)
        elements.append(f'<line x1="{x0 - 6}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="#666"/>')
        elements.append(
            f'<text x="{x0 - 10}" y="{y + 4:.1f}" font-family="Helvetica" font-size="12" text-anchor="end">{_svg_number(conc_val)}</text>'
        )

    color_standard = "#1f77b4"
    palette = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    treatments = data[data["sample_type"] == "treatment"].copy()
    unique_doses = sorted(treatments[dose_col].dropna().unique())
    dose_colors = {
        dose: ("#ff69b4" if dose == 0 else palette[i % len(palette)])
        for i, dose in enumerate(unique_doses)
    }

    # Standards scatter
    standards = data[data["sample_type"] == "standard"]
    for _, row in standards.iterrows():
        elements.append(
            f'<circle cx="{map_x(row["area"]):.2f}" cy="{map_y(row["concentration_mg_ml"]):.2f}" r="5" fill="{color_standard}" />'
        )

    # Treatments scatter
    for _, row in treatments.iterrows():
        dose = row[dose_col]
        color = dose_colors.get(dose, "#7f7f7f")
        elements.append(
            f'<circle cx="{map_x(row["area"]):.2f}" cy="{map_y(row["concentration_mg_ml"]):.2f}" r="4" fill="{color}" opacity="0.9" />'
        )

    # Legend
    legend_x = margin + 20
    legend_y = margin + 20
    legend_height = 40 + 20 * len(unique_doses)
    elements.append(
        f'<rect x="{legend_x - 10}" y="{legend_y - 18}" width="280" height="{legend_height}" fill="#ffffff" stroke="#999"/>'
    )
    elements.append(
        f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color_standard}" />'
        f'<text x="{legend_x + 12}" y="{legend_y + 4}" font-family="Helvetica" font-size="12">Standards</text>'
    )
    for i, dose in enumerate(unique_doses):
        y = legend_y + 20 + i * 18
        elements.append(
            f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{dose_colors[dose]}" />'
            f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Helvetica" font-size="12">{dose_label} {dose:.3f} mW/cm²</text>'
        )

    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>"
    )
    path.write_text(svg_content, encoding="utf-8")


def plot_with_matplotlib(path: Path, data: pd.DataFrame, form: str, dose_col: str, dose_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    standards = data[data["sample_type"] == "standard"]
    treatments = data[data["sample_type"] == "treatment"]

    ax.scatter(
        standards["area"],
        standards["concentration_mg_ml"],
        color="#1f77b4",
        label="Standards",
        s=45,
        zorder=3,
    )

    unique_doses = sorted(treatments[dose_col].dropna().unique())
    palette = plt.get_cmap("tab10")
    for idx, dose in enumerate(unique_doses):
        mask = treatments[dose_col] == dose
        color = "#ff69b4" if dose == 0 else palette(idx % 10)
        ax.scatter(
            treatments.loc[mask, "area"],
            treatments.loc[mask, "concentration_mg_ml"],
            color=color,
            label=f"{dose_label} {dose:.3f} mW/cm²",
            s=36,
            alpha=0.85,
        )

    if treatments[dose_col].isna().any():
        ax.scatter(
            treatments.loc[treatments[dose_col].isna(), "area"],
            treatments.loc[treatments[dose_col].isna(), "concentration_mg_ml"],
            color="#7f7f7f",
            label=f"{dose_label} NA",
            s=36,
            alpha=0.85,
        )

    ax.set_title(f"Chromatogram area vs concentration ({form})")
    ax.set_xlabel("Peak area")
    ax.set_ylabel("Concentration (mg/mL)")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_outputs(
    combined: pd.DataFrame,
    plot_dir: Path,
    form: str,
    dose_col: str,
    suffix: str,
    dose_label: str,
) -> None:
    csv_path = plot_dir / f"{form}_treatment_overlay_{suffix}.csv"
    combined.to_csv(csv_path, index=False)

    plot_path = plot_dir / f"{form}_treatment_overlay_{suffix}.png"
    if plt is not None:
        plot_with_matplotlib(plot_path, combined, form, dose_col, dose_label)
    else:
        svg_path = plot_dir / f"{form}_treatment_overlay_{suffix}.svg"
        _generate_svg(svg_path, combined, form, dose_col, dose_label)
        print(f"[INFO] matplotlib not available; wrote SVG overlay to {svg_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo = args.config.resolve().parent

    outputs_cfg = config["paths"]["outputs"]
    chrom_dir = repo / config["paths"]["chrom_areas_dir"]
    calibrations_dir = repo / outputs_cfg["calibrations_dir"]
    plot_dir = repo / outputs_cfg["chromatogram_plots_dir"]
    ensure_directory(plot_dir)

    standards_lookup = pd.read_csv(repo / config["paths"]["standards_csv"])
    stage_b_df = pd.read_csv(calibrations_dir / "treatments_concentration_raw.csv")
    truth_df = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")

    for form in config["scytonemin"]["forms"]:
        std_df = load_standards(form, chrom_dir, standards_lookup)
        treat_df = load_treatments(form, chrom_dir, truth_df, stage_b_df)
        combined = std_df.rename(columns={"known_concentration_mg_ml": "concentration_mg_ml"}).copy()
        treat_df = treat_df.rename(columns={"predicted_concentration_mg_ml": "concentration_mg_ml"})
        combined = pd.concat([combined, treat_df], ignore_index=True)
        combined = combined.sort_values(["sample_type", "sample_id"]).reset_index(drop=True)

        write_outputs(combined, plot_dir, form, "uva_dose", "grouped_by_uva", "UVA")
        write_outputs(combined, plot_dir, form, "uvb_dose", "grouped_by_uvb", "UVB")
        write_outputs(combined, plot_dir, form, "uva_uvb_ratio", "grouped_by_uva_uvb_ratio", "UVA/UVB ratio")


if __name__ == "__main__":
    main()
