#!/usr/bin/env python3
"""Utilities for producing raw vs. denoised ROI overlay plots."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.roi_metrics import compute_dip_metrics, _fit_polynomial_baseline


@dataclass(frozen=True)
class PlotSettings:
    """Configuration for ROI window visualisations."""

    roi: Sequence[float] = (320.0, 500.0)
    half_width_nm: float = 7.0
    baseline_guard_nm: float = 10.0
    poly_order: int = 2
    enforce_min_separation_nm: float = 3.0
    max_dips: int = 6
    dpi: int = 150


@dataclass(frozen=True)
class PlotMetadata:
    """Describes the spectrum used in the overlay."""

    relative_path: str
    treatment: str = ""
    sample: str = ""
    angle: str = ""


def _angle_label_to_degrees(angle_label: str) -> Optional[float]:
    """Best-effort conversion of angle labels such as `6Oclock` to degrees."""
    match = re.match(r"(\d+)", str(angle_label))
    if not match:
        return None
    hour = int(match.group(1)) % 12
    radians = 2.0 * math.pi * hour / 12.0
    cos_theta = math.cos(radians)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))


def _format_title(metadata: PlotMetadata) -> str:
    treatment_label = metadata.treatment or ""
    treatment_match = re.search(r"(\d+)", treatment_label)
    treatment_id = treatment_match.group(1) if treatment_match else ""

    sample_label_raw = metadata.sample or ""
    sample_match = re.search(r"sample[_-]?(.+)", sample_label_raw, re.IGNORECASE)
    sample_suffix = sample_match.group(1) if sample_match else sample_label_raw
    sample_suffix = sample_suffix.replace("_", "").upper()

    if treatment_id and sample_suffix:
        sample_label = f"{treatment_id}{sample_suffix}"
    elif sample_suffix:
        sample_label = sample_suffix
    else:
        sample_label = metadata.relative_path

    angle_label = metadata.angle or ""
    if angle_label:
        angle_match = re.match(r"(\d+)", angle_label)
        if angle_match:
            angle_label = f"{angle_match.group(1)} o'clock"
        else:
            angle_label = angle_label.replace("_", " ").lower()
        return f"Example denoising effect — sample {sample_label} angle {angle_label}".strip()
    return f"Example denoising effect — sample {sample_label}"


def _ensure_1d(spectrum: np.ndarray) -> np.ndarray:
    """Collapse spectra with leading singleton dimensions."""
    if spectrum.ndim == 1:
        return spectrum
    if spectrum.ndim == 2 and 1 in spectrum.shape[:-1]:
        return spectrum.reshape(-1)
    if spectrum.ndim >= 2 and 1 in spectrum.shape[:-1]:
        return spectrum.reshape(-1)
    raise ValueError(f"Unsupported spectrum shape {spectrum.shape!r}")


def plot_roi_windows(
    raw_spectrum: np.ndarray,
    denoised_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path,
    settings: PlotSettings,
    metadata: PlotMetadata,
) -> None:
    """Render ROI overlays for a single raw/denoised pair."""
    raw_1d = _ensure_1d(np.asarray(raw_spectrum, dtype=np.float32))
    denoised_1d = _ensure_1d(np.asarray(denoised_spectrum, dtype=np.float32))
    if raw_1d.shape != denoised_1d.shape:
        raise ValueError(
            f"Raw spectrum shape {raw_1d.shape} does not match denoised shape {denoised_1d.shape}"
        )

    lam = np.asarray(wavelengths, dtype=np.float32)
    if lam.ndim != 1:
        raise ValueError(f"Wavelength grid must be 1-D, received shape {lam.shape!r}")
    if lam.shape[0] != raw_1d.shape[0]:
        raise ValueError(
            f"Wavelength grid length {lam.shape[0]} does not match spectrum length {raw_1d.shape[0]}"
        )

    _, dips = compute_dip_metrics(
        denoised_1d,
        raw_1d,
        lam,
        roi=tuple(settings.roi),
        known_lines_nm=None,
        half_width_nm=settings.half_width_nm,
        baseline="local",
        baseline_guard_nm=settings.baseline_guard_nm,
        centroid_method="parabolic",
        max_dips=settings.max_dips,
        poly_order=settings.poly_order,
        min_separation_nm=settings.enforce_min_separation_nm,
    )

    max_panels = max(settings.max_dips, 1)
    cols = min(3, max_panels)
    rows = int(math.ceil(max_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), sharey=False)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for dip in dips:
        panel_idx = int(dip.get("dip_id", 0)) - 1
        if panel_idx < 0 or panel_idx >= len(axes):
            continue
        ax = axes[panel_idx]
        ax.axis("on")
        start = int(dip["window_start_index"])
        end = int(dip["window_end_index"])
        window_idx = np.arange(start, end + 1)
        lam_window = lam[window_idx]
        raw_window = raw_1d[window_idx]
        denoised_window = denoised_1d[window_idx]

        baseline_raw = _fit_polynomial_baseline(
            lam_window,
            raw_window,
            dip["center_nm"],
            settings.baseline_guard_nm,
            settings.poly_order,
        )
        baseline_denoised = _fit_polynomial_baseline(
            lam_window,
            denoised_window,
            dip["center_nm"],
            settings.baseline_guard_nm,
            settings.poly_order,
        )

        ax.plot(lam_window, raw_window, label="raw", color="#1f77b4", linewidth=1.2)
        ax.plot(lam_window, denoised_window, label="denoised", color="#d62728", linewidth=1.0)
        ax.plot(
            lam_window,
            baseline_raw,
            label="baseline",
            color="#2ca02c",
            linestyle="--",
            linewidth=1.0,
        )

        ax.fill_between(lam_window, raw_window, baseline_raw, color="#1f77b4", alpha=0.15)
        ax.fill_between(
            lam_window,
            denoised_window,
            baseline_denoised,
            color="#d62728",
            alpha=0.12,
        )

        ax.set_title(
            f"Dip {dip['dip_id']}: Δλ={dip['centroid_error_nm']:.3f} nm\nΔarea={dip['area_error_pct']:.1f}%",
            fontsize=9,
        )
        ax.set_xlabel("λ (nm)", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Reflectance", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            fontsize=8,
            ncol=3,
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
        )
    fig.suptitle(_format_title(metadata), fontsize=10, y=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(fig)


def make_output_name(base_dir: Path, relative_path: str, index: int) -> Path:
    """Generate a flattened PNG path for a given spectrum."""
    safe_stem = relative_path.replace("/", "__").replace(" ", "_")
    return base_dir / f"{index:05d}_{safe_stem}.png"
