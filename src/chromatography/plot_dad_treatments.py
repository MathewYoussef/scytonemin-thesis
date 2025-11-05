#!/usr/bin/env python3
"""
Overlay diode-array calibration curves with treatment predictions grouped by UVA dose.

Produces per-form CSVs and SVG plots under Diode_Array_Derived_Calibration_Plots/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import json
import yaml

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None


FORMS = ["total", "oxidized", "reduced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diode-array treatment overlays.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--forms",
        nargs="*",
        help="Subset of forms to process (default: all).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_plot(
    output_base: Path,
    standards: pd.DataFrame,
    treatments: pd.DataFrame,
    slope: float,
    intercept: float,
    r_squared: float,
    max_rel_resid: float,
    form: str,
) -> None:
    title = f"{form.capitalize()} DAD calibration with treatments"
    if plt is not None:
        path = output_base.with_suffix(".png")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(
            standards["auc"],
            standards["conc"],
            color="#1f77b4",
            label="Standards",
            s=45,
        )
        palette = plt.get_cmap("viridis")
        if not treatments.empty:
            doses = np.sort(treatments["p_uva_mw_cm2"].unique())
            dose_to_color = {
                dose: ("#ff69b4" if dose == 0 else palette(i / max(len(doses) - 1, 1)))
                for i, dose in enumerate(doses)
            }
            for dose, subset in treatments.groupby("p_uva_mw_cm2"):
                ax.scatter(
                    subset["auc"],
                    subset["pred"],
                    color=dose_to_color[dose],
                    s=35,
                    label=f"UVA {dose:.3f}",
                    alpha=0.85,
                )
        xs = np.linspace(0, max(standards["auc"].max(), treatments["auc"].max() if not treatments.empty else 0) * 1.05, 100)
        ax.plot(xs, intercept + slope * xs, color="#d62728", label="Fit")
        y_min = min(standards["conc"].min(), intercept, (intercept + slope * xs).min())
        if not treatments.empty:
            y_min = min(y_min, treatments["pred"].min())
        y_max = max(standards["conc"].max(), (intercept + slope * xs).max())
        if not treatments.empty:
            y_max = max(y_max, treatments["pred"].max())
        ax.set_ylim(y_min - abs(y_max - y_min) * 0.05, y_max * 1.05)
        ax.set_xlabel("AUC (320–480 nm, corrected)")
        ax.set_ylabel("Concentration (mg/mL)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        stats_text = (
            f"slope = {slope:.3e}\n"
            f"intercept = {intercept:.3e} mg/mL\n"
            f"R² = {r_squared:.4f}\n"
            f"max |rel residual| = {max_rel_resid:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return

    # Fallback SVG
    path = output_base.with_suffix(".svg")
    width, height = 700, 420
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    x_max = max(standards["auc"].max(), treatments["auc"].max() if not treatments.empty else 0) * 1.05
    x_max = max(x_max, 1e-6)
    y_candidates = [standards["conc"].max(), intercept + slope * x_max]
    y_min_candidates = [standards["conc"].min(), intercept]
    if not treatments.empty:
        y_candidates.append(treatments["pred"].max())
        y_min_candidates.append(treatments["pred"].min())
    y_max = max(y_candidates)
    y_min = min(y_min_candidates)
    if y_max == y_min:
        y_max = y_min + 1e-6

    def map_x(val: float) -> float:
        return margin + (val / x_max) * plot_width

    def map_y(val: float) -> float:
        return margin + plot_height - ((val - y_min) / (y_max - y_min)) * plot_height

    xs = np.linspace(0, x_max, 120)
    points = " ".join(f"{map_x(xv):.2f},{map_y(intercept + slope * xv):.2f}" for xv in xs)

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="30" font-family="Helvetica" font-size="18" text-anchor="middle">{title}</text>',
        f'<line x1="{margin}" y1="{margin + plot_height}" x2="{margin + plot_width}" y2="{margin + plot_height}" stroke="#444" stroke-width="1.2"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_height}" stroke="#444" stroke-width="1.2"/>',
        f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="2"/>',
    ]

    for _, row in standards.iterrows():
        elements.append(f'<circle cx="{map_x(row["auc"]):.2f}" cy="{map_y(row["conc"]):.2f}" r="5" fill="#1f77b4"/>')

    palette = ["#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"]
    if not treatments.empty:
        doses = sorted(treatments["p_uva_mw_cm2"].unique())
        dose_colors: Dict[float, str] = {
            dose: ("#ff69b4" if dose == 0 else palette[i % len(palette)])
            for i, dose in enumerate(doses)
        }
        for _, row in treatments.iterrows():
            color = dose_colors[row["p_uva_mw_cm2"]]
            elements.append(
                f'<circle cx="{map_x(row["auc"]):.2f}" cy="{map_y(row["pred"]):.2f}" r="4" fill="{color}" opacity="0.85"/>'
            )
        legend_y = margin + 15
        elements.append(
            f'<rect x="{width - margin - 150}" y="{legend_y - 18}" width="140" height="{20 * (len(doses) + 1)}" fill="#ffffff" stroke="#bbbbbb"/>'
        )
        elements.append(
            f'<circle cx="{width - margin - 135}" cy="{legend_y}" r="5" fill="#1f77b4" />'
            f'<text x="{width - margin - 120}" y="{legend_y + 4}" font-family="Helvetica" font-size="11">Standards</text>'
        )
        for i, dose in enumerate(doses, start=1):
            y_pos = legend_y + i * 20
            color = dose_colors[dose]
            elements.append(
                f'<circle cx="{width - margin - 135}" cy="{y_pos}" r="5" fill="{color}" />'
                f'<text x="{width - margin - 120}" y="{y_pos + 4}" font-family="Helvetica" font-size="11">UVA {dose:.3f}</text>'
            )

    for tick in np.linspace(0, x_max, 5):
        x_pos = map_x(tick)
        elements.append(f'<line x1="{x_pos:.2f}" y1="{margin + plot_height:.2f}" x2="{x_pos:.2f}" y2="{margin + plot_height + 6:.2f}" stroke="#666" stroke-width="1"/>')
        elements.append(
            f'<text x="{x_pos:.2f}" y="{margin + plot_height + 22:.2f}" font-family="Helvetica" font-size="11" text-anchor="middle">{tick:.2e}</text>'
        )
    for tick in np.linspace(y_min, y_max, 5):
        y_pos = map_y(tick)
        elements.append(f'<line x1="{margin - 6:.2f}" y1="{y_pos:.2f}" x2="{margin:.2f}" y2="{y_pos:.2f}" stroke="#666" stroke-width="1"/>')
        elements.append(
            f'<text x="{margin - 10:.2f}" y="{y_pos + 4:.2f}" font-family="Helvetica" font-size="11" text-anchor="end">{tick:.2e}</text>'
        )

    elements.append(
        f'<text x="{margin + plot_width/2:.1f}" y="{margin + plot_height + 40:.1f}" font-family="Helvetica" font-size="13" text-anchor="middle">AUC (320–480 nm, corrected)</text>'
    )
    elements.append(
        f'<text x="{margin - 45:.1f}" y="{margin + plot_height/2:.1f}" font-family="Helvetica" font-size="13" text-anchor="middle" transform="rotate(-90 {margin - 45:.1f} {margin + plot_height/2:.1f})">Concentration (mg/mL)</text>'
    )

    info_text = (
        f"slope: {slope:.3e}"
        f"\nintercept: {intercept:.3e} mg/mL"
        f"\nR²: {r_squared:.4f}"
        f"\nmax |rel residual|: {max_rel_resid:.3f}"
    )
    elements.append(
        f'<text x="{margin + 10}" y="{margin + 20}" font-family="Helvetica" font-size="13" fill="#000000">'
        + info_text.replace("\n", "&#10;")
        + "</text>"
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo = args.config.resolve().parent

    auc_path = repo / config["paths"]["outputs"]["dad_auc_dir"] / "diode_array_auc.csv"
    auc_df = pd.read_csv(auc_path)

    standards_csv = repo / config["paths"]["standards_csv"]
    std_info = pd.read_csv(standards_csv)
    level_to_conc = dict(zip(std_info["standard_level"], std_info["concentration_mg_ml"]))

    cal_dir = repo / config["paths"]["outputs"]["dad_calibration_plots_dir"]
    forms = args.forms if args.forms else FORMS
    slopes: Dict[str, float] = {}
    intercepts: Dict[str, float] = {}
    r_squared_map: Dict[str, float] = {}
    max_resid_map: Dict[str, float] = {}
    for form in forms:
        with (cal_dir / f"calibration_{form}.json").open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        slopes[form] = data["slope"]
        intercepts[form] = data.get("intercept", 0.0)
        r_squared_map[form] = data.get("r_squared", float("nan"))
        max_resid_map[form] = data.get("max_abs_rel_residual", float("nan"))
    missing = [f for f in forms if f not in slopes]
    if missing:
        raise KeyError(f"Missing calibration data for forms: {', '.join(missing)}")

    truth_path = repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv"
    truth_df = pd.read_csv(truth_path)
    truth_df = truth_df[["sample_id", "p_uva_mw_cm2"]]

    out_dir = cal_dir
    ensure_dir(out_dir)

    for form in FORMS:
        std = auc_df[(auc_df["sample_category"] == "standard") & (auc_df["spectrum_state"] == form)].copy()
        std["standard_level"] = std["sample_id"].str.replace("Standard", "").astype(float)
        std["conc"] = std["standard_level"].map(level_to_conc)
    # Rename using reference to calibration CSV column order
        standards = std[["auc_corrected", "conc"]].rename(columns={"auc_corrected": "auc"})

        trt = auc_df[(auc_df["sample_category"] == "sample") & (auc_df["spectrum_state"] == form)].copy()
        predictions = intercepts[form] + slopes[form] * trt["auc_corrected"]
        trt = trt[["sample_id", "auc_corrected"]].rename(columns={"auc_corrected": "auc"})
        trt["pred"] = predictions
        trt = trt.merge(truth_df, how="left", on="sample_id")

        csv_path = out_dir / f"{form}_treatment_overlay_grouped_by_uva.csv"
        out_df = standards.assign(sample_type="standard").rename(columns={"conc": "concentration_mg_ml"})
        out_df = out_df[["auc", "concentration_mg_ml", "sample_type"]]
        trt_out = trt.assign(sample_type="treatment").rename(columns={"pred": "concentration_mg_ml"})
        trt_out = trt_out[["sample_id", "auc", "concentration_mg_ml", "p_uva_mw_cm2", "sample_type"]]
        combined = pd.concat([out_df, trt_out], ignore_index=True)
        combined.to_csv(csv_path, index=False)

        plot_base = out_dir / f"{form}_treatment_overlay_grouped_by_uva"
        make_plot(
            plot_base,
            standards,
            trt,
            slopes[form],
            intercepts[form],
            r_squared_map[form],
            max_resid_map[form],
            form,
        )

    print("Generated treatment overlay plots and CSVs for diode-array calibration.")


if __name__ == "__main__":
    main()
