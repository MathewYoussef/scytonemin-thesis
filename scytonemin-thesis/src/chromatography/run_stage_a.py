#!/usr/bin/env python3
"""
Stage A: Calibrate scytonemin chromatogram areas against known concentrations.

Reads configuration from analysis_config.yaml, fits weighted/unweighted linear models
for each scytonemin form, writes calibration JSON/CSV assets, and generates diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A calibration for scytonemin chromatogram areas.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--forms",
        nargs="*",
        help="Optional subset of forms to process (default: use config.sc ytonemin.forms)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_standard_table(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    if "standard_level" not in df or "concentration_mg_ml" not in df:
        raise ValueError(f"Standard table {path} must have columns standard_level, concentration_mg_ml")
    table = {}
    for row in df.itertuples(index=False):
        if pd.isna(row.standard_level) or pd.isna(row.concentration_mg_ml):
            continue
        table[int(row.standard_level)] = float(row.concentration_mg_ml)
    if not table:
        raise ValueError(f"No valid rows found in {path}")
    return table


# ---------------------------------------------------------------------------
# Weighted linear regression implementation
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    form: str
    weighting: str
    intercept_mode: str
    slope: float
    intercept: float
    slope_se: float
    intercept_se: Optional[float]
    cov_intercept_slope: Optional[float]
    r_squared: float
    r_squared_weighted: float
    rss: float
    tss: float
    aic: float
    bic: float
    dof: int
    residuals: np.ndarray
    fitted: np.ndarray
    weights: np.ndarray
    intercept_pvalue_normal: Optional[float]
    max_abs_rel_residual: float
    residual_poly_r2: float
    qc_pass: bool


def _normal_pvalue(z_score: float) -> float:
    """Two-sided p-value assuming normal distribution."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))


def _weighted_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    force_zero_intercept: bool,
) -> tuple[float, float, float, Optional[float], Optional[float], np.ndarray, np.ndarray, float, float, int]:
    if weights is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = weights.astype(float)
    if np.any(w <= 0):
        raise ValueError("Weights must be strictly positive.")

    n = len(x)
    if force_zero_intercept:
        # beta1 = sum(w x y)/sum(w x^2)
        sum_wx2 = np.sum(w * x * x)
        slope = np.sum(w * x * y) / sum_wx2
        intercept = 0.0
        fitted = slope * x
        dof = n - 1
        rss = np.sum(w * (y - fitted) ** 2)
        sigma2 = rss / dof
        slope_se = math.sqrt(sigma2 / sum_wx2)
        intercept_se = None
        cov_beta0_beta1 = None
    else:
        W = np.sum(w)
        wx = np.sum(w * x)
        wy = np.sum(w * y)
        wxx = np.sum(w * x * x)
        wxy = np.sum(w * x * y)
        denom = W * wxx - wx * wx
        if abs(denom) < 1e-12:
            raise ValueError("Singular matrix encountered during regression.")
        slope = (W * wxy - wx * wy) / denom
        intercept = (wy - slope * wx) / W
        fitted = intercept + slope * x
        dof = n - 2
        rss = np.sum(w * (y - fitted) ** 2)
        sigma2 = rss / dof
        slope_se = math.sqrt(sigma2 * W / denom)
        intercept_se = math.sqrt(sigma2 * wxx / denom)
        cov_beta0_beta1 = -sigma2 * wx / denom
    return slope, intercept, slope_se, intercept_se, cov_beta0_beta1, fitted, w, rss, sigma2, dof


def _compute_r2(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray, force_zero: bool) -> tuple[float, float, float]:
    y_bar = np.average(y, weights=weights) if len(y) else 0.0
    tss = np.sum(weights * (y - y_bar) ** 2)
    rss = np.sum(weights * (y - fitted) ** 2)
    wmean = np.sum(weights * y) / np.sum(weights)
    tss_weighted = np.sum(weights * (y - wmean) ** 2)
    if force_zero:
        # Provide both weighted/unweighted style R^2
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        r2_weighted = 1.0 - rss / tss_weighted if tss_weighted > 0 else float("nan")
    else:
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        r2_weighted = 1.0 - rss / tss_weighted if tss_weighted > 0 else float("nan")
    return r2, r2_weighted, rss


def _calc_poly_r2(x: np.ndarray, residuals: np.ndarray) -> float:
    """R^2 of quadratic fit residual ~ poly2(fitted)."""
    if len(x) < 3:
        return float("nan")
    coeffs = np.polyfit(x, residuals, deg=2)
    trend = np.polyval(coeffs, x)
    ss_res = np.sum((residuals - trend) ** 2)
    ss_tot = np.sum((residuals - residuals.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_weights(weighting: str, areas: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    if weighting == "none":
        return np.ones_like(concentrations, dtype=float)
    if weighting == "1/x":
        return 1.0 / concentrations
    if weighting in {"1/x2", "1/x^2"}:
        return 1.0 / (concentrations**2)
    raise ValueError(f"Unsupported weighting option: {weighting}")


def evaluate_fit(
    form: str,
    weighting: str,
    intercept_mode: str,
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    qc_thresholds: dict,
) -> FitResult:
    force_zero = intercept_mode == "zero"
    slope, intercept, slope_se, intercept_se, cov_b0_b1, fitted, w, rss, sigma2, dof = _weighted_regression(
        x, y, weights, force_zero
    )
    r2, r2_weighted, rss_val = _compute_r2(y, fitted, w, force_zero)
    resid = y - fitted
    n = len(y)
    k = 1 if force_zero else 2
    sigma_sq = rss_val / n if n else float("nan")
    aic = n * math.log(sigma_sq) + 2 * k if sigma_sq > 0 else float("nan")
    bic = n * math.log(sigma_sq) + k * math.log(n) if sigma_sq > 0 else float("nan")
    if intercept_se is not None and intercept_se > 0:
        pvalue = _normal_pvalue(intercept / intercept_se)
    else:
        pvalue = None
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_resid = np.where(y != 0, resid / y, np.nan)
    max_abs_rel_resid = np.nanmax(np.abs(rel_resid))
    residual_poly_r2 = _calc_poly_r2(fitted, resid)
    qc_pass = (
        (not math.isnan(r2) and r2 >= qc_thresholds["calib_r2_min"])
        and (not math.isnan(max_abs_rel_resid) and max_abs_rel_resid <= qc_thresholds["max_abs_rel_residual"])
    )
    return FitResult(
        form=form,
        weighting=weighting,
        intercept_mode=intercept_mode,
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        intercept_se=intercept_se,
        r_squared=r2,
        r_squared_weighted=r2_weighted,
        rss=rss_val,
        tss=np.sum(w * (y - np.average(y, weights=w)) ** 2),
        aic=aic,
        bic=bic,
        dof=dof,
        residuals=resid,
        fitted=fitted,
        weights=w,
        intercept_pvalue_normal=pvalue,
        max_abs_rel_residual=max_abs_rel_resid,
        residual_poly_r2=residual_poly_r2,
        qc_pass=qc_pass,
        cov_intercept_slope=cov_b0_b1,
    )


def select_best_fit(
    fits: Iterable[FitResult],
    weighting_preference: str,
) -> FitResult:
    candidates = list(fits)
    if not candidates:
        raise ValueError("No fit candidates generated.")

    # Group by weighting to apply intercept preference first
    best_by_weight: Dict[str, FitResult] = {}
    for candidate in candidates:
        key = candidate.weighting
        current = best_by_weight.get(key)
        if current is None:
            best_by_weight[key] = candidate
            continue
        # Prefer zero intercept when intercept is not significant and QC holds
        if (
            candidate.intercept_mode == "zero"
            and candidate.qc_pass
            and (current.intercept_mode != "zero")
            and (current.intercept_pvalue_normal is None or current.intercept_pvalue_normal > 0.05)
        ):
            best_by_weight[key] = candidate
            continue
        if candidate.aic < current.aic:
            best_by_weight[key] = candidate

    # Filter by QC pass when possible
    qc_candidates = [fit for fit in best_by_weight.values() if fit.qc_pass]
    pool = qc_candidates if qc_candidates else list(best_by_weight.values())

    def sort_key(fit: FitResult) -> tuple:
        weighting_rank = 0 if fit.weighting == weighting_preference else 1
        intercept_rank = 0 if fit.intercept_mode == "zero" and (fit.intercept_pvalue_normal is None or fit.intercept_pvalue_normal > 0.05) else 1
        return (fit.qc_pass is False, weighting_rank, fit.aic, intercept_rank)

    pool.sort(key=sort_key)
    return pool[0]


# ---------------------------------------------------------------------------
# Main calibration workflow
# ---------------------------------------------------------------------------


def load_chromatogram(form: str, base_dir: Path) -> pd.DataFrame:
    filename = f"raw_{form}_scytonemin.csv"
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Chromatogram file missing: {path}")
    df = pd.read_csv(path)
    df["type"] = df["sample_category"].map({"sample": "treatment", "standard": "standard", "blank": "blank"})
    return df


def prepare_standards(df: pd.DataFrame, standard_lookup: dict[int, float]) -> pd.DataFrame:
    subset = df[df["type"] == "standard"].copy()
    subset["standard_level"] = subset["standard_level"].astype(float)
    subset["known_concentration_mg_ml"] = subset["standard_concentration_mg_ml"]
    missing_mask = subset["known_concentration_mg_ml"].isna() & subset["standard_level"].notna()
    if missing_mask.any():
        subset.loc[missing_mask, "known_concentration_mg_ml"] = subset.loc[missing_mask, "standard_level"].map(standard_lookup)
    subset = subset.dropna(subset=["area", "known_concentration_mg_ml"])
    subset["area"] = subset["area"].astype(float)
    subset["known_concentration_mg_ml"] = subset["known_concentration_mg_ml"].astype(float)
    return subset


def compute_blank_stats(df: pd.DataFrame) -> dict:
    blanks = df[df["type"] == "blank"]["area"].dropna().astype(float)
    if blanks.empty:
        return {"n_blank": 0, "mean_blank_area": float("nan"), "std_blank_area": float("nan")}
    return {
        "n_blank": int(blanks.count()),
        "mean_blank_area": float(blanks.mean()),
        "std_blank_area": float(blanks.std(ddof=1)) if blanks.count() > 1 else 0.0,
    }


def compute_detection_limits(blank_stats: dict, slope: float, lod_multiplier: float, loq_multiplier: float) -> dict:
    sigma = blank_stats.get("std_blank_area", float("nan"))
    if math.isnan(sigma) or sigma == 0 or slope == 0:
        return {"LOD_mg_ml": float("nan"), "LOQ_mg_ml": float("nan")}
    lod = lod_multiplier * sigma / slope
    loq = loq_multiplier * sigma / slope
    return {"LOD_mg_ml": float(lod), "LOQ_mg_ml": float(loq)}


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _pad_range(min_val: float, max_val: float) -> tuple[float, float]:
    if math.isclose(max_val, min_val):
        delta = abs(max_val) * 0.1 or 1.0
        return min_val - delta, max_val + delta
    span = max_val - min_val
    pad = span * 0.08
    return min_val - pad, max_val + pad


def _svg_axis_ticks(min_val: float, max_val: float, count: int = 4) -> list[float]:
    if count <= 1 or math.isclose(max_val, min_val):
        return [min_val]
    return [min_val + (max_val - min_val) * i / (count - 1) for i in range(count)]


def _svg_number(val: float) -> str:
    if abs(val) >= 1000 or abs(val) < 0.001:
        return f"{val:.2e}"
    return f"{val:.3f}"


def _write_svg_calibration(path: Path, standards: pd.DataFrame, fit: FitResult) -> None:
    width, height = 900, 420
    margin = 60
    plot_width = (width - 3 * margin) / 2
    plot_height = height - 2 * margin

    scatter_x0 = margin
    scatter_y0 = margin
    resid_x0 = scatter_x0 + plot_width + margin
    resid_y0 = margin

    areas = standards["area"].to_numpy(dtype=float)
    conc = standards["known_concentration_mg_ml"].to_numpy(dtype=float)
    fitted = fit.fitted
    residuals = fit.residuals

    area_min, area_max = _pad_range(float(areas.min()), float(areas.max()))
    conc_min, conc_max = _pad_range(float(min(conc.min(), fitted.min())), float(max(conc.max(), fitted.max())))
    resid_min, resid_max = _pad_range(float(residuals.min()), float(residuals.max()))
    if resid_min > 0:
        resid_min = 0.0
    if resid_max < 0:
        resid_max = 0.0

    def map_x(value: float, min_val: float, max_val: float, origin_x: float) -> float:
        if math.isclose(max_val, min_val):
            return origin_x + plot_width / 2
        return origin_x + (value - min_val) / (max_val - min_val) * plot_width

    def map_y(value: float, min_val: float, max_val: float, origin_y: float) -> float:
        if math.isclose(max_val, min_val):
            return origin_y + plot_height / 2
        return origin_y + plot_height - (value - min_val) / (max_val - min_val) * plot_height

    elements: list[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')
    # Titles
    elements.append(
        f'<text x="{scatter_x0 + plot_width / 2:.1f}" y="{margin / 2:.1f}" '
        f'font-family="Helvetica" font-size="18" text-anchor="middle">Calibration ({fit.form})</text>'
    )

    # Scatter axes
    x_axis_y = scatter_y0 + plot_height
    elements.append(
        f'<line x1="{scatter_x0:.1f}" y1="{x_axis_y:.1f}" x2="{scatter_x0 + plot_width:.1f}" y2="{x_axis_y:.1f}" '
        f'stroke="#444" stroke-width="1.2" />'
    )
    elements.append(
        f'<line x1="{scatter_x0:.1f}" y1="{scatter_y0:.1f}" x2="{scatter_x0:.1f}" y2="{x_axis_y:.1f}" '
        f'stroke="#444" stroke-width="1.2" />'
    )

    # Residual axes
    resid_axis_y = resid_y0 + plot_height
    elements.append(
        f'<line x1="{resid_x0:.1f}" y1="{resid_axis_y:.1f}" x2="{resid_x0 + plot_width:.1f}" y2="{resid_axis_y:.1f}" '
        f'stroke="#444" stroke-width="1.2" />'
    )
    elements.append(
        f'<line x1="{resid_x0:.1f}" y1="{resid_y0:.1f}" x2="{resid_x0:.1f}" y2="{resid_axis_y:.1f}" '
        f'stroke="#444" stroke-width="1.2" />'
    )

    # Scatter ticks
    for val in _svg_axis_ticks(area_min, area_max):
        x = map_x(val, area_min, area_max, scatter_x0)
        elements.append(
            f'<line x1="{x:.1f}" y1="{x_axis_y:.1f}" x2="{x:.1f}" y2="{x_axis_y + 6:.1f}" stroke="#666" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{x_axis_y + 22:.1f}" font-family="Helvetica" font-size="12" text-anchor="middle">'
            f'{_svg_number(val)}</text>'
        )
    elements.append(
        f'<text x="{scatter_x0 + plot_width / 2:.1f}" y="{x_axis_y + 40:.1f}" font-family="Helvetica" font-size="14" '
        f'text-anchor="middle">Peak area</text>'
    )
    for val in _svg_axis_ticks(conc_min, conc_max):
        y = map_y(val, conc_min, conc_max, scatter_y0)
        elements.append(
            f'<line x1="{scatter_x0 - 6:.1f}" y1="{y:.1f}" x2="{scatter_x0:.1f}" y2="{y:.1f}" stroke="#666" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{scatter_x0 - 10:.1f}" y="{y + 4:.1f}" font-family="Helvetica" font-size="12" '
            f'text-anchor="end">{_svg_number(val)}</text>'
        )
    elements.append(
        f'<text x="{scatter_x0 - 45:.1f}" y="{scatter_y0 + plot_height / 2:.1f}" '
        f'font-family="Helvetica" font-size="14" text-anchor="middle" transform="rotate(-90 {scatter_x0 - 45:.1f} {scatter_y0 + plot_height / 2:.1f})">'
        f'Concentration (mg/mL)</text>'
    )

    # Residual ticks
    for val in _svg_axis_ticks(area_min, area_max):
        x = map_x(val, area_min, area_max, resid_x0)
        elements.append(
            f'<line x1="{x:.1f}" y1="{resid_axis_y:.1f}" x2="{x:.1f}" y2="{resid_axis_y + 6:.1f}" stroke="#666" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{resid_axis_y + 22:.1f}" font-family="Helvetica" font-size="12" text-anchor="middle">'
            f'{_svg_number(val)}</text>'
        )
    elements.append(
        f'<text x="{resid_x0 + plot_width / 2:.1f}" y="{resid_axis_y + 40:.1f}" font-family="Helvetica" font-size="14" '
        f'text-anchor="middle">Fitted concentration (mg/mL)</text>'
    )
    for val in _svg_axis_ticks(resid_min, resid_max):
        y = map_y(val, resid_min, resid_max, resid_y0)
        elements.append(
            f'<line x1="{resid_x0 - 6:.1f}" y1="{y:.1f}" x2="{resid_x0:.1f}" y2="{y:.1f}" stroke="#666" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{resid_x0 - 10:.1f}" y="{y + 4:.1f}" font-family="Helvetica" font-size="12" '
            f'text-anchor="end">{_svg_number(val)}</text>'
        )
    zero_y = map_y(0.0, resid_min, resid_max, resid_y0)
    elements.append(
        f'<line x1="{resid_x0:.1f}" y1="{zero_y:.1f}" x2="{resid_x0 + plot_width:.1f}" y2="{zero_y:.1f}" '
        f'stroke="#999" stroke-width="1" stroke-dasharray="6,4"/>'
    )
    elements.append(
        f'<text x="{resid_x0 - 45:.1f}" y="{resid_y0 + plot_height / 2:.1f}" '
        f'font-family="Helvetica" font-size="14" text-anchor="middle" transform="rotate(-90 {resid_x0 - 45:.1f} {resid_y0 + plot_height / 2:.1f})">'
        f'Residual (mg/mL)</text>'
    )

    # Data points scatter
    for area_val, conc_val in zip(areas, conc):
        x = map_x(area_val, area_min, area_max, scatter_x0)
        y = map_y(conc_val, conc_min, conc_max, scatter_y0)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f77b4" />')

    # Fit line
    xs = np.linspace(area_min, area_max, num=80)
    ys = fit.slope * xs if fit.intercept_mode == "zero" else fit.intercept + fit.slope * xs
    points = " ".join(
        f"{map_x(x, area_min, area_max, scatter_x0):.2f},{map_y(y, conc_min, conc_max, scatter_y0):.2f}"
        for x, y in zip(xs, ys)
    )
    elements.append(f'<polyline points="{points}" fill="none" stroke="#ff7f0e" stroke-width="2"/>')

    # Residual points
    for fit_val, resid_val in zip(fitted, residuals):
        x = map_x(fit_val, area_min, area_max, resid_x0)
        y = map_y(resid_val, resid_min, resid_max, resid_y0)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2ca02c" />')

    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(elements) + "</svg>"
    )
    path.write_text(svg_content, encoding="utf-8")


def save_calibration_json(path: Path, result: FitResult, blank_stats: dict, detection_limits: dict) -> None:
    data = {
        "form": result.form,
        "weighting": result.weighting,
        "intercept_mode": result.intercept_mode,
        "slope": result.slope,
        "intercept": result.intercept,
        "slope_se": result.slope_se,
        "intercept_se": result.intercept_se,
        "r_squared": result.r_squared,
        "r_squared_weighted": result.r_squared_weighted,
        "rss": result.rss,
        "aic": result.aic,
        "bic": result.bic,
        "degrees_of_freedom": result.dof,
        "intercept_pvalue_normal": result.intercept_pvalue_normal,
        "max_abs_rel_residual": result.max_abs_rel_residual,
        "residual_poly_r2": result.residual_poly_r2,
        "qc_pass": result.qc_pass,
        "cov_intercept_slope": result.cov_intercept_slope,
        "blank_stats": blank_stats,
        "detection_limits": detection_limits,
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def save_fitted_csv(path: Path, standards: pd.DataFrame, fit: FitResult) -> None:
    df = standards.copy()
    df["fitted_concentration_mg_ml"] = fit.fitted
    df["residual_mg_ml"] = fit.residuals
    with np.errstate(divide="ignore", invalid="ignore"):
        df["relative_residual"] = np.where(
            df["known_concentration_mg_ml"] != 0,
            df["residual_mg_ml"] / df["known_concentration_mg_ml"],
            np.nan,
        )
    df.to_csv(path, index=False)


def plot_calibration(path: Path, standards: pd.DataFrame, fit: FitResult) -> None:
    if plt is None:
        svg_path = path.with_suffix(".svg")
        _write_svg_calibration(svg_path, standards, fit)
        print(f"[INFO] matplotlib not available; wrote SVG plot to {svg_path}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter, ax_resid = axes

    ax_scatter.scatter(
        standards["area"],
        standards["known_concentration_mg_ml"],
        color="tab:blue",
        label="Standards",
    )
    x_vals = np.linspace(standards["area"].min(), standards["area"].max(), 100)
    if fit.intercept_mode == "zero":
        y_vals = fit.slope * x_vals
    else:
        y_vals = fit.intercept + fit.slope * x_vals
    ax_scatter.plot(x_vals, y_vals, color="tab:orange", label="Fit")
    ax_scatter.set_xlabel("Peak area")
    ax_scatter.set_ylabel("Concentration (mg/mL)")
    ax_scatter.set_title(f"{fit.form} calibration ({fit.weighting}, {fit.intercept_mode})")
    ax_scatter.legend()

    ax_resid.scatter(fit.fitted, fit.residuals, color="tab:green")
    ax_resid.axhline(0.0, color="black", linewidth=1)
    ax_resid.set_xlabel("Fitted concentration (mg/mL)")
    ax_resid.set_ylabel("Residual (mg/mL)")
    ax_resid.set_title("Residuals vs fitted")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_for_form(
    form: str,
    config: dict,
    repo_root: Path,
    standard_lookup: dict[int, float],
) -> None:
    chrom_dir = repo_root / config["paths"]["chrom_areas_dir"]
    outputs_cfg = config["paths"]["outputs"]
    standards_out_dir = repo_root / outputs_cfg["calibrations_dir"]
    plots_dir = repo_root / outputs_cfg["chromatogram_plots_dir"]

    ensure_directory(standards_out_dir)
    ensure_directory(plots_dir)

    df = load_chromatogram(form, chrom_dir)
    standards = prepare_standards(df, standard_lookup)
    if standards.empty:
        raise ValueError(f"No standards found for form {form}")

    blank_stats = compute_blank_stats(df)
    weighting_options = ["none"]
    preferred_weighting = config["modeling"].get("weighting", "none")
    if config["modeling"].get("use_weighted_regression", False):
        weighting_options = ["none", "1/x", "1/x2"]

    fits: list[FitResult] = []
    qc_thresholds = {
        "calib_r2_min": config["qc"]["calib_r2_min"],
        "max_abs_rel_residual": config["qc"]["max_abs_rel_residual"],
    }
    x = standards["area"].to_numpy(dtype=float)
    y = standards["known_concentration_mg_ml"].to_numpy(dtype=float)

    for weighting in weighting_options:
        weights = build_weights(weighting, x, y) if weighting != "none" else np.ones_like(y)
        for intercept_mode in ("free", "zero"):
            try:
                fits.append(evaluate_fit(form, weighting, intercept_mode, x, y, weights, qc_thresholds))
            except ValueError as exc:
                print(f"[WARN] Skipping {form} weighting={weighting} intercept={intercept_mode}: {exc}")

    if not fits:
        raise RuntimeError(f"Unable to fit any models for {form}")

    best_fit = select_best_fit(fits, preferred_weighting)
    detection_limits = compute_detection_limits(
        blank_stats,
        best_fit.slope,
        config["modeling"].get("lod_multiplier", 3.3),
        config["modeling"].get("loq_multiplier", 10.0),
    )

    calib_json_path = standards_out_dir / f"calibration_{form}.json"
    standards_csv_path = standards_out_dir / f"standards_fitted_{form}.csv"
    plot_path = plots_dir / f"calibration_{form}.png"

    save_calibration_json(calib_json_path, best_fit, blank_stats, detection_limits)
    save_fitted_csv(standards_csv_path, standards, best_fit)
    plot_calibration(plot_path, standards, best_fit)

    print(
        f"[INFO] {form}: selected weighting={best_fit.weighting}, intercept={best_fit.intercept_mode}, "
        f"slope={best_fit.slope:.6g}, intercept={best_fit.intercept:.6g}, R^2={best_fit.r_squared:.6f}, "
        f"QC={'PASS' if best_fit.qc_pass else 'FAIL'}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo_root = args.config.resolve().parent
    standard_lookup = read_standard_table(repo_root / config["paths"]["standards_csv"])

    forms = args.forms if args.forms else config["scytonemin"]["forms"]
    for form in forms:
        run_for_form(form, config, repo_root, standard_lookup)


if __name__ == "__main__":
    main()
def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
