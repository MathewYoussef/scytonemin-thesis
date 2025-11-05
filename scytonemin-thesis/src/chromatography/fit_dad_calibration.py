#!/usr/bin/env python3
"""
Fit diode-array calibration curves (forced origin) using standard AUC values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None


FORMS = ["total", "oxidized", "reduced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit diode-array calibration curves.")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_forced_zero_regression(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> dict:
    if weights is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = weights.astype(float)
    if np.any(w <= 0):
        raise ValueError("Weights must be positive.")

    sum_wx2 = np.sum(w * x * x)
    sum_wxy = np.sum(w * x * y)
    slope = sum_wxy / sum_wx2
    fitted = slope * x
    residuals = y - fitted
    rss = np.sum(w * residuals**2)
    dof = len(x) - 1
    sigma2 = rss / dof
    slope_se = math.sqrt(sigma2 / sum_wx2)

    tss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")
    return {
        "slope": slope,
        "slope_se": slope_se,
        "residuals": residuals,
        "fitted": fitted,
        "weights": w,
        "rss": rss,
        "sigma2": sigma2,
        "dof": dof,
        "r_squared": r_squared,
    }


def linear_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    r_squared: float,
    max_rel_resid: float,
    title: str,
) -> None:
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(x, y, color="#1f77b4", label="Standards / blanks")
        xs = np.linspace(0, x.max() * 1.05, num=100)
        ax.plot(xs, intercept + slope * xs, color="#d62728", label="Linear fit")
        ax.set_xlabel("AUC (320–480 nm, corrected)")
        ax.set_ylabel("Concentration (mg/mL)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        text = (
            f"slope = {slope:.3e}\n"
            f"intercept = {intercept:.3e} mg/mL\n"
            f"R² = {r_squared:.4f}\n"
            f"max |rel residual| = {max_rel_resid:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            text,
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
    width, height = 600, 400
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    x_max = max(x.max(), 1e-6)
    y_max = max(y.max(), slope * x_max, 1e-6)

    def map_x(val: float) -> float:
        return margin + (val / x_max) * plot_width

    def map_y(val: float) -> float:
        return margin + plot_height - (val / y_max) * plot_height

    xs = np.linspace(0, x_max * 1.05, num=100)
    points = " ".join(f"{map_x(val):.2f},{map_y(intercept + slope*val):.2f}" for val in xs)

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="30" font-family="Helvetica" font-size="18" text-anchor="middle">{title}</text>',
        f'<line x1="{margin}" y1="{margin + plot_height}" x2="{margin + plot_width}" y2="{margin + plot_height}" stroke="#444" stroke-width="1.2"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_height}" stroke="#444" stroke-width="1.2"/>',
        f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="2"/>',
    ]
    for xx, yy in zip(x, y):
        elements.append(f'<circle cx="{map_x(xx):.2f}" cy="{map_y(yy):.2f}" r="5" fill="#1f77b4"/>')

    x_ticks = np.linspace(0, x_max, 5)
    y_ticks = np.linspace(0, y_max, 5)
    for tick in x_ticks:
        x_pos = map_x(tick)
        elements.append(f'<line x1="{x_pos:.2f}" y1="{margin + plot_height:.2f}" x2="{x_pos:.2f}" y2="{margin + plot_height + 6:.2f}" stroke="#666" stroke-width="1"/>')
        elements.append(
            f'<text x="{x_pos:.2f}" y="{margin + plot_height + 22:.2f}" font-family="Helvetica" font-size="11" text-anchor="middle">{tick:.2e}</text>'
        )
    for tick in y_ticks:
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
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    auc_path = Path(config["paths"]["outputs"]["dad_auc_dir"]) / "diode_array_auc.csv"
    auc_df = pd.read_csv(auc_path)

    standards_csv = Path(config["paths"]["standards_csv"])
    std_df = pd.read_csv(standards_csv)
    level_to_conc = dict(zip(std_df["standard_level"].astype(int), std_df["concentration_mg_ml"].astype(float)))

    def sample_level(sample_id: str) -> int:
        token = sample_id.replace("Standard", "").strip()
        return int(float(token))

    calibration_dir = Path(config["paths"]["outputs"]["dad_calibration_plots_dir"])
    ensure_dir(calibration_dir)

    blanks = auc_df[auc_df["sample_category"] == "blank"]
    blank_std_auc = blanks["auc_raw"].std(ddof=1) if len(blanks) > 1 else float("nan")

    modeling_cfg = config.get("modeling", {})
    weighting = modeling_cfg.get("weighting", "none")

    results = []
    for form in FORMS:
        subset = auc_df[
            (auc_df["sample_category"] == "standard") & (auc_df["spectrum_state"] == form)
        ].copy()
        if subset.empty:
            continue
        subset["standard_level"] = subset["sample_id"].apply(sample_level)
        subset["known_concentration_mg_ml"] = subset["standard_level"].map(level_to_conc)
        subset = subset.dropna(subset=["auc_corrected", "known_concentration_mg_ml"])

        x = subset["auc_corrected"].to_numpy(dtype=float)
        y = subset["known_concentration_mg_ml"].to_numpy(dtype=float)

        if weighting == "1/x":
            w = 1.0 / x
        elif weighting in {"1/x2", "1/x^2"}:
            w = 1.0 / (x**2)
        else:
            w = None

        # Fit with intercept (weighted or unweighted)
        if w is None:
            X = np.vstack([np.ones_like(x), x]).T
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            intercept, slope = beta
            fitted = X @ beta
            residuals = y - fitted
            dof = len(x) - 2
            rss_weighted = np.sum(residuals**2)
            sigma2 = rss_weighted / dof
            XtX_inv = np.linalg.inv(X.T @ X)
            slope_se = math.sqrt(sigma2 * XtX_inv[1, 1])
            intercept_se = math.sqrt(sigma2 * XtX_inv[0, 0])
        else:
            sqrt_w = np.sqrt(w)
            X = np.vstack([np.ones_like(x), x]).T
            Xw = X * sqrt_w[:, None]
            yw = y * sqrt_w
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            intercept, slope = beta
            fitted = X @ beta
            residuals = y - fitted
            dof = len(x) - 2
            rss_weighted = np.sum((sqrt_w * residuals) ** 2)
            sigma2 = rss_weighted / dof
            XtX_inv = np.linalg.inv(Xw.T @ Xw)
            slope_se = math.sqrt(sigma2 * XtX_inv[1, 1])
            intercept_se = math.sqrt(sigma2 * XtX_inv[0, 0])

        rss_unweighted = np.sum(residuals**2)
        tss = np.sum((y - np.mean(y)) ** 2)
        if w is None:
            r_squared = 1.0 - rss_unweighted / tss if tss > 0 else float("nan")
        else:
            weighted_mean = np.average(y, weights=w)
            tss_weighted = np.sum(w * (y - weighted_mean) ** 2)
            r_squared = 1.0 - rss_weighted / tss_weighted if tss_weighted > 0 else float("nan")

        residuals = residuals
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_resid = np.where(y != 0, residuals / y, np.nan)
        max_rel_resid = float(np.nanmax(np.abs(rel_resid)))

        lod = (3.3 * blank_std_auc) / slope if slope != 0 and not math.isnan(blank_std_auc) else float("nan")
        loq = (10.0 * blank_std_auc) / slope if slope != 0 and not math.isnan(blank_std_auc) else float("nan")

        calib_json = {
            "form": form,
            "slope": slope,
            "slope_se": slope_se,
            "intercept": intercept,
            "intercept_se": intercept_se,
            "weighting": weighting,
            "r_squared": r_squared,
            "rss_unweighted": rss_unweighted,
            "rss_weighted": float(rss_weighted),
            "sigma2": sigma2,
            "degrees_of_freedom": dof,
            "max_abs_rel_residual": max_rel_resid,
            "blank_std_auc": blank_std_auc,
            "lod_mg_ml": lod,
            "loq_mg_ml": loq,
        }
        (calibration_dir / f"calibration_{form}.json").write_text(json.dumps(calib_json, indent=2), encoding="utf-8")

        fitted_df = subset[["sample_id", "standard_level", "known_concentration_mg_ml"]].copy()
        fitted_df["auc_corrected"] = x
        fitted_df["predicted_concentration_mg_ml"] = fitted
        fitted_df["residual_mg_ml"] = residuals
        fitted_df["relative_residual"] = rel_resid
        fitted_df.to_csv(calibration_dir / f"standards_fitted_{form}.csv", index=False)

        plot_title = f"{form.capitalize()} concentration vs DAD AUC"
        plot_path = calibration_dir / f"calibration_{form}.svg"
        linear_plot(
            plot_path,
            x,
            y,
            slope,
            intercept,
            r_squared,
            max_rel_resid,
            plot_title,
        )

        print(
            f"[INFO] {form}: slope={slope:.6e}, intercept={intercept:.3e}, "
            f"R^2={r_squared:.4f}, max|rel residual|={max_rel_resid:.3f}"
        )
        results.append((form, slope, r_squared))

    if not results:
        raise RuntimeError("No calibration results produced; check standard spectra.")


if __name__ == "__main__":
    main()
