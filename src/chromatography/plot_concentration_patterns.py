#!/usr/bin/env python3
"""
Generate chromatogram-derived concentration pattern plots and regression summaries.

Outputs under Chromatogram_derived_concentration_patterns_plots/:
  - regression_summary.csv
  - {driver}_concentration_data_{form}.csv for each driver/form
  - {driver}_vs_concentration.svg (fallback SVG with three subplots)

Drivers:
  * UVA dose (`p_uva_mw_cm2`)
  * UVB dose (`p_uvb_mw_cm2`)
  * UVA/UVB ratio (`uva_uvb_ratio`)
  * ΔUVA vs Δcorrected concentration (`delta_p_uva_from_zero` vs `delta_{form}_from_uva0`)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None


FORMS = ["oxidized", "reduced", "total"]


@dataclass
class RegressionResult:
    form: str
    driver: str
    slope: float
    intercept: float
    r_squared: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chromatogram-derived concentration patterns vs UV doses.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chromatogram_derived_concentrations.csv"),
        help="CSV containing derived concentrations (default: %(default)s)",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dropna_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def fit_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def render_svg(path: Path, driver_label: str, series: Dict[str, Dict[str, np.ndarray]]) -> None:
    width, height = 900, 320
    margin_x, margin_y = 60, 50
    plot_width = (width - 4 * margin_x) / 3
    plot_height = height - 2 * margin_y
    forms = ["oxidized", "reduced", "total"]
    form_titles = {"oxidized": "Oxidized", "reduced": "Reduced", "total": "Total"}

    def pad_range(min_val: float, max_val: float) -> Tuple[float, float]:
        if math.isnan(min_val) or math.isnan(max_val):
            return 0.0, 1.0
        if math.isclose(max_val, min_val):
            delta = abs(max_val) * 0.1 or 1.0
            return min_val - delta, max_val + delta
        span = max_val - min_val
        pad = span * 0.08
        return min_val - pad, max_val + pad

    elements = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />']
    elements.append(
        f'<text x="{width/2:.1f}" y="25" font-family="Helvetica" font-size="18" text-anchor="middle">'
        f'{driver_label} vs Corrected Concentration</text>'
    )

    for idx, form in enumerate(forms):
        data = series.get(form)
        if not data:
            continue
        x = data["x"]
        y = data["y"]
        slope = data["slope"]
        intercept = data["intercept"]
        r2 = data["r2"]
        n = data["n"]

        xmin, xmax = pad_range(float(np.nanmin(x)), float(np.nanmax(x)))
        ymin, ymax = pad_range(float(np.nanmin(y)), float(np.nanmax(y)))

        def map_x(value: float) -> float:
            if math.isclose(xmax, xmin):
                return margin_x + idx * (plot_width + margin_x) + plot_width / 2
            return margin_x + idx * (plot_width + margin_x) + (value - xmin) / (xmax - xmin) * plot_width

        def map_y(value: float) -> float:
            if math.isclose(ymax, ymin):
                return margin_y + plot_height / 2
            return margin_y + plot_height - (value - ymin) / (ymax - ymin) * plot_height

        origin_x = margin_x + idx * (plot_width + margin_x)
        origin_y = margin_y

        # Axes
        elements.append(
            f'<line x1="{origin_x:.1f}" y1="{origin_y + plot_height:.1f}" '
            f'x2="{origin_x + plot_width:.1f}" y2="{origin_y + plot_height:.1f}" stroke="#444" stroke-width="1.2"/>'
        )
        elements.append(
            f'<line x1="{origin_x:.1f}" y1="{origin_y:.1f}" x2="{origin_x:.1f}" y2="{origin_y + plot_height:.1f}" '
            f'stroke="#444" stroke-width="1.2"/>'
        )

        # Ticks (4 each)
        for i in range(4):
            tx = xmin + (xmax - xmin) * i / 3
            ty = ymin + (ymax - ymin) * i / 3
            x_pos = map_x(tx)
            y_pos = map_y(ty)
            elements.append(
                f'<line x1="{x_pos:.1f}" y1="{origin_y + plot_height:.1f}" x2="{x_pos:.1f}" y2="{origin_y + plot_height + 6:.1f}" '
                f'stroke="#666"/>'
            )
            elements.append(
                f'<text x="{x_pos:.1f}" y="{origin_y + plot_height + 22:.1f}" font-family="Helvetica" font-size="11" text-anchor="middle">'
                f'{tx:.2f}</text>'
            )
            elements.append(
                f'<line x1="{origin_x - 6:.1f}" y1="{y_pos:.1f}" x2="{origin_x:.1f}" y2="{y_pos:.1f}" stroke="#666"/>'
            )
            elements.append(
                f'<text x="{origin_x - 10:.1f}" y="{y_pos + 4:.1f}" font-family="Helvetica" font-size="11" text-anchor="end">'
                f'{ty:.2f}</text>'
            )

        # Scatter points
        for xi, yi in zip(x, y):
            elements.append(
                f'<circle cx="{map_x(float(xi)):.2f}" cy="{map_y(float(yi)):.2f}" r="4" fill="#1f77b4" opacity="0.85"/>'
            )

        # Regression line
        if not math.isnan(slope) and not math.isnan(intercept):
            xs = np.linspace(xmin, xmax, num=80)
            ys = slope * xs + intercept
            points = " ".join(f"{map_x(float(xx)):.2f},{map_y(float(yy)):.2f}" for xx, yy in zip(xs, ys))
            elements.append(f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="2"/>')

        # Titles and stats
        elements.append(
            f'<text x="{origin_x + plot_width / 2:.1f}" y="{origin_y - 10:.1f}" font-family="Helvetica" font-size="16" '
            f'text-anchor="middle">{form_titles[form]}</text>'
        )
        elements.append(
            f'<text x="{origin_x + plot_width / 2:.1f}" y="{origin_y + plot_height + 40:.1f}" font-family="Helvetica" '
            f'font-size="12" text-anchor="middle">n={n}, slope={slope:.3f}, R²={r2:.3f}</text>'
        )

    elements.append(
        f'<text x="{width/2:.1f}" y="{height - 10:.1f}" font-family="Helvetica" font-size="13" text-anchor="middle">'
        f'{driver_label}</text>'
    )
    elements.append(
        f'<text x="20" y="{height/2:.1f}" font-family="Helvetica" font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 20 {height/2:.1f})">Corrected concentration (mg/gDW)</text>'
    )

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">' \
          + "".join(elements) + "</svg>"
    path.write_text(svg, encoding="utf-8")


def plot_with_matplotlib(path: Path, driver_label: str, series: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    form_titles = {"oxidized": "Oxidized", "reduced": "Reduced", "total": "Total"}
    for ax, form in zip(axes, ["oxidized", "reduced", "total"]):
        data = series.get(form)
        if not data:
            ax.axis("off")
            continue
        x, y, slope, intercept, r2, n = (
            data["x"],
            data["y"],
            data["slope"],
            data["intercept"],
            data["r2"],
            data["n"],
        )
        ax.scatter(x, y, color="#1f77b4", alpha=0.85)
        if not math.isnan(slope) and not math.isnan(intercept):
            xs = np.linspace(np.nanmin(x), np.nanmax(x), num=100)
            ax.plot(xs, slope * xs + intercept, color="#d62728")
        ax.set_title(f"{form_titles[form]} (R²={r2:.3f}, n={n})")
        ax.set_xlabel(driver_label)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("Corrected concentration (mg/gDW)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = pd.Series(dtype=object)  # placeholder (not used but kept for symmetry)
    output_dir = Path("Chromatogram_derived_concentration_patterns_plots")
    ensure_dir(output_dir)

    df = pd.read_csv(args.input)
    df = df.sort_values("sample_id").reset_index(drop=True)

    drivers: list[tuple[str, str, Dict[str, str]]] = [
        ("p_uva_mw_cm2", "UVA dose (mW/cm²)", {form: f"{form}_mg_per_gDW" for form in FORMS}),
        ("p_uvb_mw_cm2", "UVB dose (mW/cm²)", {form: f"{form}_mg_per_gDW" for form in FORMS}),
        ("uva_uvb_ratio", "UVA/UVB dose ratio", {form: f"{form}_mg_per_gDW" for form in FORMS}),
        ("delta_p_uva_from_zero", "ΔUVA dose from baseline (mW/cm²)", {form: f"delta_{form}_from_uva0" for form in FORMS}),
    ]

    summaries: list[RegressionResult] = []

    for driver_col, driver_label, y_map in drivers:
        series_data: Dict[str, Dict[str, np.ndarray]] = {}
        for form in FORMS:
            y_col = y_map.get(form)
            if y_col not in df.columns:
                continue
            x = df[driver_col].to_numpy(dtype=float)
            y = df[y_col].to_numpy(dtype=float)
            x_clean, y_clean = dropna_pairs(x, y)
            slope, intercept, r2 = fit_regression(x_clean, y_clean)
            summaries.append(
                RegressionResult(
                    form=form,
                    driver=driver_col,
                    slope=slope,
                    intercept=intercept,
                    r_squared=r2,
                    n=len(x_clean),
                )
            )
            data_dict = {
                "x": x_clean,
                "y": y_clean,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "n": len(x_clean),
            }
            series_data[form] = data_dict

            csv_path = output_dir / f"{driver_col}_concentration_data_{form}.csv"
            pd.DataFrame({"x": x_clean, "y": y_clean}).to_csv(csv_path, index=False)

        plot_path = output_dir / f"{driver_col}_vs_concentration.png"
        if plt is not None:
            plot_with_matplotlib(plot_path, driver_label, series_data)
        else:
            svg_path = output_dir / f"{driver_col}_vs_concentration.svg"
            render_svg(svg_path, driver_label, series_data)
            print(f"[INFO] matplotlib not available; wrote SVG to {svg_path}")

    summary_df = pd.DataFrame(
        [
            {
                "driver": result.driver,
                "form": result.form,
                "slope": result.slope,
                "intercept": result.intercept,
                "r_squared": result.r_squared,
                "n": result.n,
            }
            for result in summaries
        ]
    )
    summary_df.to_csv(output_dir / "regression_summary.csv", index=False)


if __name__ == "__main__":
    main()
