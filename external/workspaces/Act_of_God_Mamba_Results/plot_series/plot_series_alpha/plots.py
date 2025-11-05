"""Plotting helpers for the plot series alpha figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import constants
from .metrics import robust_quantile_limits


def plot_variance_heatmaps(
    treatments: Iterable[str],
    raw_matrix: np.ndarray,
    denoised_matrix: np.ndarray,
    wavelengths: np.ndarray,
    *,
    output_path: Path,
    clip_limits: Tuple[float, float] | None = None,
) -> None:
    """Render side-by-side variance heatmaps pre/post denoising."""
    treatments = list(treatments)
    if clip_limits is None:
        clip_limits = robust_quantile_limits(raw_matrix, denoised_matrix)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, max(4, 0.5 * len(treatments))), sharey=True)
    data_and_titles = [
        (raw_matrix, "Raw variance"),
        (denoised_matrix, "Denoised variance"),
    ]

    for ax, (matrix, title) in zip(axes, data_and_titles):
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap=constants.HEATMAP_CMAP,
            extent=[wavelengths[0], wavelengths[-1], -0.5, len(treatments) - 0.5],
            vmin=clip_limits[0],
            vmax=clip_limits[1],
        )
        _draw_roi_band(ax)
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_yticks(range(len(treatments)))
        ax.set_yticklabels(treatments)

    axes[0].set_ylabel("Treatment")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Variance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.9, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_ratio(
    wavelengths: np.ndarray,
    ratio_data: dict,
    *,
    output_path: Path,
) -> None:
    """Plot the variance ratio with optional bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ratio = ratio_data["ratio"]
    ax.plot(wavelengths, ratio, color=constants.VARIANCE_RATIO_COLOR, label="Variance ratio (raw/denoised)")

    if "ci_lower" in ratio_data and "ci_upper" in ratio_data:
        ax.fill_between(
            wavelengths,
            ratio_data["ci_lower"],
            ratio_data["ci_upper"],
            color=constants.VARIANCE_RATIO_COLOR,
            alpha=0.2,
            linewidth=0,
            label="95% bootstrap CI",
        )

    _draw_roi_band(ax, alpha=0.1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Variance ratio")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.set_title("Variance ratio across wavelengths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sam_to_control(
    treatments: Iterable[str],
    sam_raw: Dict[str, float],
    sam_den: Dict[str, float],
    *,
    output_path: Path,
) -> None:
    """Plot SAM to control before/after denoising."""
    treatments = list(treatments)
    y_positions = np.arange(len(treatments))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(treatments))))
    raw_values = [sam_raw[t] for t in treatments]
    den_values = [sam_den[t] for t in treatments]

    ax.scatter(raw_values, y_positions, color=constants.RAW_COLOR, label="Raw")
    ax.scatter(den_values, y_positions, color=constants.DENOISED_COLOR, label="Denoised")
    for y, raw_val, den_val in zip(y_positions, raw_values, den_values):
        ax.plot([raw_val, den_val], [y, y], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("SAM to reference (degrees)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(treatments)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.set_title("Spectral angle to reference, pre/post denoise")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_sam(
    wavelengths: np.ndarray,
    treatments: Iterable[str],
    delta_map: Dict[str, np.ndarray],
    *,
    output_path: Path,
) -> None:
    """Plot ΔSAM = SAM_denoised - SAM_raw across wavelengths."""
    fig, ax = plt.subplots(figsize=(12, 5))
    treatment_list = list(treatments)
    for idx, treatment in enumerate(treatment_list):
        delta = delta_map[treatment]
        color = constants.DELTA_SAM_COLORS[idx % len(constants.DELTA_SAM_COLORS)]
        ax.plot(wavelengths, delta, label=treatment, color=color)

    _draw_roi_band(ax, alpha=0.1)
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Δ SAM (degrees)")
    ax.set_title("Sliding-window Δ SAM (denoised - raw)")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_preservation_indices(
    indices: Dict[str, Dict[str, float]],
    *,
    output_path: Path,
) -> None:
    """Scatter plot summarizing noise collapse vs pigment preservation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (treatment, metrics) in enumerate(indices.items()):
        color = constants.DELTA_SAM_COLORS[idx % len(constants.DELTA_SAM_COLORS)]
        ax.scatter(
            metrics["noise_collapse"],
            metrics["pigment_preservation"],
            color=color,
            label=treatment,
            s=80,
        )
        ax.annotate(
            treatment,
            (metrics["noise_collapse"], metrics["pigment_preservation"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Noise collapse (mean variance ratio outside ROI)")
    ax.set_ylabel("Pigment preservation (1 - |ΔSAM| in ROI)")
    ax.set_title("Preservation vs. collapse summary")
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roi_overlays(
    wavelengths: np.ndarray,
    treatments: Iterable[str],
    medians_raw: Dict[str, np.ndarray],
    iqrs_raw: Dict[str, Tuple[np.ndarray, np.ndarray]],
    medians_den: Dict[str, np.ndarray],
    iqrs_den: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path,
) -> None:
    """Create small multiples of ROI spectra with median ± IQR for raw and denoised data."""
    roi_mask = (wavelengths >= 350) & (wavelengths <= 400)
    roi_wavelengths = wavelengths[roi_mask]

    treatment_list = list(treatments)
    n_treatments = len(treatment_list)
    n_cols = 2
    n_rows = int(np.ceil(n_treatments / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for idx, treatment in enumerate(treatment_list):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        median_raw = medians_raw[treatment][roi_mask]
        q25_raw, q75_raw = iqrs_raw[treatment]
        median_den = medians_den[treatment][roi_mask]
        q25_den, q75_den = iqrs_den[treatment]

        ax.plot(roi_wavelengths, median_raw, color=constants.RAW_COLOR, label="Raw")
        ax.fill_between(roi_wavelengths, q25_raw[roi_mask], q75_raw[roi_mask], color=constants.RAW_COLOR, alpha=0.2)

        ax.plot(roi_wavelengths, median_den, color=constants.DENOISED_COLOR, label="Denoised")
        ax.fill_between(roi_wavelengths, q25_den[roi_mask], q75_den[roi_mask], color=constants.DENOISED_COLOR, alpha=0.2)

        ax.set_title(treatment)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.grid(True, linestyle=":", alpha=0.3)

    # Hide unused subplots if any.
    for idx in range(n_treatments, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("ROI spectra (median ± IQR), raw vs denoised", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_effect_sizes(
    treatments: Iterable[str],
    effect_sizes: Dict[str, Dict[str, float]],
    *,
    output_path: Path,
) -> None:
    """Plot Cohen's d effect sizes inside/outside the ROI."""
    treatments = list(treatments)
    y_positions = np.arange(len(treatments))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(treatments))))
    outside_values = [effect_sizes[t]["outside_roi"] for t in treatments]
    inside_values = [effect_sizes[t]["inside_roi"] for t in treatments]

    ax.scatter(outside_values, y_positions, color=constants.VARIANCE_RATIO_COLOR, label="Outside ROI")
    ax.scatter(inside_values, y_positions, color=constants.ROI_BAND_COLOR, label="Inside ROI")
    for y, out_val, in_val in zip(y_positions, outside_values, inside_values):
        ax.plot([in_val, out_val], [y, y], color="gray", linestyle="--", linewidth=1)

    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Cohen's d (denoised vs raw)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(treatments)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.set_title("Effect sizes inside vs outside ROI")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _draw_roi_band(ax: plt.Axes, alpha: float = 0.15) -> None:
    ax.axvspan(
        constants.ROI_MIN_NM,
        constants.ROI_MAX_NM,
        color=constants.ROI_BAND_COLOR,
        alpha=alpha,
        label="Scytonemin ROI",
    )
