"""
Matplotlib figure builders for the diagnostic plot series.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - runtime dependency shim
    from .metrics import (
        EPS,
        GroupSNRResult,
        GroupVarianceResult,
        SAMResult,
        VarianceRatioResult,
    )
except ImportError:  # pragma: no cover
    from metrics import (  # type: ignore
        EPS,
        GroupSNRResult,
        GroupVarianceResult,
        SAMResult,
        VarianceRatioResult,
    )


ROI_COLOR = "#ffdf80"
RAW_COLOR = "#1f77b4"
DENOISED_COLOR = "#ff7f0e"
DELTA_COLOR = "#2ca02c"

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _format_labels(labels: Sequence[str]) -> Sequence[str]:
    treatment_map = {
        "treatment_6": "dose 1",
        "treatment_5": "dose 2",
        "treatment_4": "dose 3",
        "treatment_3": "dose 4",
        "treatment_2": "dose 5",
        "treatment_1": "dose 6",
    }
    formatted: list[str] = []
    for label in labels:
        parts = label.split("::")
        if parts:
            parts[0] = treatment_map.get(parts[0], parts[0])
        formatted.append(" · ".join(parts))
    return formatted


def _quantile_limits(arr: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> Tuple[float, float]:
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return 0.0, 1.0
    return float(np.quantile(flat, lower)), float(np.quantile(flat, upper))


def plot_variance_heatmaps(
    result: GroupVarianceResult,
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
    reductions: Sequence[float] | None = None,
    order: Sequence[int] | None = None,
) -> plt.Figure:
    labels = np.array(_format_labels(result.labels))
    raw = result.raw.copy()
    den = result.denoised.copy()
    if order is not None and len(order):
        raw = raw[order]
        den = den[order]
        labels = labels[list(order)]
        if reductions is not None:
            reductions = [reductions[i] if i < len(reductions) else np.nan for i in order]
    log_raw = np.log10(np.clip(raw, EPS, None))
    log_den = np.log10(np.clip(den, EPS, None))
    percent_delta = 100.0 * (raw - den) / np.clip(raw, EPS, None)

    n_groups = len(labels)
    height = max(4, 0.5 * n_groups + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, height), sharey=True)
    matrices = [log_raw, log_den, percent_delta]
    titles = ["Raw variance (log₁₀)", "Denoised variance (log₁₀)", "Percent reduction"]
    cmaps = ["viridis", "viridis", "coolwarm"]

    extent = [result.wavelengths[0], result.wavelengths[-1], -0.5, n_groups - 0.5]
    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps, strict=True):
        if title == "Percent reduction":
            limit = np.nanmax(np.abs(mat[np.isfinite(mat)]))
            vmax = max(5.0, limit)
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                vmin=-vmax,
                vmax=vmax,
                cmap=cmap,
            )
            label = "Δ Var (%)"
        else:
            vmin, vmax = _quantile_limits(mat)
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            label = "log₁₀ variance"
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.axvspan(roi_nm[0], roi_nm[1], color=ROI_COLOR, alpha=0.25)
        fig.colorbar(im, ax=ax, pad=0.01, label=label)

    axes[0].set_yticks(range(n_groups))
    axes[0].set_yticklabels(labels)

    if reductions is not None:
        for idx, value in enumerate(reductions):
            if np.isnan(value):
                continue
            axes[2].text(
                result.wavelengths[-1] + 5,
                idx,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )
        axes[2].set_xlim(result.wavelengths[0], result.wavelengths[-1] + 30)
        axes[2].set_ylabel("Mean reduction outside ROI")

    fig.suptitle("Fig. C — Variance behaviour (log + Δ)", y=0.995, fontsize=14)
    fig.tight_layout()
    return fig


def plot_snr_heatmaps(
    result: GroupSNRResult,
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
    order: Sequence[int] | None = None,
    snr_gains: Sequence[float] | None = None,
    vmax_delta: Optional[float] = None,
) -> plt.Figure:
    labels = np.array(_format_labels(result.labels))
    snr_delta = result.snr_delta_db.copy()
    if order is not None and len(order):
        order = list(order)
        snr_delta = snr_delta[order]
        labels = labels[order]
        if snr_gains is not None:
            snr_gains = [snr_gains[i] if i < len(snr_gains) else np.nan for i in order]

    delta_finite = np.abs(snr_delta[np.isfinite(snr_delta)]) if snr_delta.size else np.array([])
    if delta_finite.size:
        delta_limit = float(np.nanmax(delta_finite))
    else:
        delta_limit = 1.0
    if vmax_delta is not None:
        delta_limit = float(max(0.1, vmax_delta))
    else:
        delta_limit = max(delta_limit, 0.5)

    n_groups = len(labels)
    height = max(4, 0.4 * n_groups + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    extent = [result.wavelengths[0], result.wavelengths[-1], -0.5, n_groups - 0.5]
    im = ax.imshow(
        snr_delta,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
        vmin=-delta_limit,
        vmax=delta_limit,
        cmap="coolwarm",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title("ΔSNR (denoised – raw) in dB")
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, pad=0.01, label="ΔSNR (dB)")

    if snr_gains is not None:
        for idx, value in enumerate(snr_gains):
            if np.isnan(value):
                continue
            ax.text(
                result.wavelengths[-1] + 5,
                idx,
                f"{value:.2f} dB",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )
        ax.set_xlim(result.wavelengths[0], result.wavelengths[-1] + 35)
        ax.set_ylabel("Mean ΔSNR inside ROI")

    fig.tight_layout()
    return fig


def plot_snr_summary(
    result: GroupSNRResult,
    *,
    order: Sequence[int] | None = None,
) -> plt.Figure:
    labels = np.array(_format_labels(result.labels))
    mean_out = result.mean_delta_outside.astype(float, copy=True)
    mean_in = result.mean_delta_inside.astype(float, copy=True)
    if order is not None and len(order):
        order = list(order)
        mean_out = mean_out[order]
        mean_in = mean_in[order]
        labels = labels[order]

    n = len(labels)
    height = max(4, 0.4 * n + 1.5)
    fig, ax = plt.subplots(figsize=(8, height))
    y = np.arange(n)
    for idx in range(n):
        ax.hlines(y[idx], mean_out[idx], mean_in[idx], color="grey", lw=1.0)
    ax.plot(mean_out, y, "o", color=RAW_COLOR, label="Outside ROI ΔSNR")
    ax.plot(mean_in, y, "o", color=DENOISED_COLOR, label="Inside ROI ΔSNR")
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    max_abs = np.nanmax(np.abs(np.concatenate([mean_out, mean_in]))) if n else 1.0
    if not np.isfinite(max_abs):
        max_abs = 1.0
    ax.set_xlim(-max(0.5, max_abs * 1.2), max(0.5, max_abs * 1.2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ΔSNR (dB)")
    ax.set_title("Fig. B — ΔSNR summary outside vs inside ROI")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_variance_ratio_ribbon(
    result: VarianceRatioResult,
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.wavelengths, result.ratio, color=RAW_COLOR, label="Variance ratio (raw/denoised)")
    if result.ci_low is not None and result.ci_high is not None:
        ax.fill_between(result.wavelengths, result.ci_low, result.ci_high, color=RAW_COLOR, alpha=0.2, label="95% CI")
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.axvspan(roi_nm[0], roi_nm[1], color=ROI_COLOR, alpha=0.25, lw=0, label="ROI 370–382 nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Variance ratio")
    ax.set_title("Fig. D — Variance ratio (raw/denoised) vs wavelength")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_sam_panels(
    sam_results: Sequence[SAMResult],
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
) -> plt.Figure:
    treatments = [res.treatment for res in sam_results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: SAM to control, raw vs denoised.
    y_positions = np.arange(len(treatments))
    raw_vals = [res.sam_raw for res in sam_results]
    den_vals = [res.sam_denoised for res in sam_results]
    axes[0].plot(raw_vals, y_positions, marker="o", color=RAW_COLOR, label="Raw")
    axes[0].plot(den_vals, y_positions, marker="o", color=DENOISED_COLOR, label="Denoised")
    for y, raw_val, den_val in zip(y_positions, raw_vals, den_vals, strict=True):
        axes[0].plot([raw_val, den_val], [y, y], color="grey", lw=0.8)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(treatments)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SAM angle (radians)")
    axes[0].set_title("SAM to control (raw vs denoised)")
    axes[0].legend()

    # Panel 2: ΔSAM windowed.
    for res in sam_results:
        axes[1].plot(res.window_centers, res.delta_windowed, label=res.treatment)
    axes[1].axhline(0.0, color="black", lw=1, ls="--")
    axes[1].axvspan(roi_nm[0], roi_nm[1], color=ROI_COLOR, alpha=0.25, lw=0)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("ΔSAM (denoised - raw)")
    axes[1].set_title("Windowed ΔSAM across wavelengths")
    axes[1].legend(fontsize=8)

    fig.suptitle("Fig. E — SAM to reference, before vs after", fontsize=14)
    fig.tight_layout()
    return fig


def plot_preservation_indices(
    table: "pd.DataFrame | Mapping[str, Mapping[str, float]] | np.ndarray",
) -> plt.Figure:
    import pandas as pd  # local import to keep top-level lean

    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    treatments = list(table["treatment"])
    y_pos = np.arange(len(treatments))

    for ax, column, title in zip(
        axes,
        ("noise_collapse", "pigment_preservation"),
        ("Noise-collapse outside ROI", "Pigment-preservation inside ROI"),
        strict=True,
    ):
        values = table[column].to_numpy()
        ax.hlines(y_pos, 0, values, color="grey", lw=1.0)
        ax.plot(values, y_pos, "o", color=DENOISED_COLOR)
        ax.set_title(title)
        ax.set_xlabel(column.replace("_", " ").title())
        ax.axvline(0, color="black", lw=0.8)
        if column == "pigment_preservation":
            ax.set_xlim(0, 1.05)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(treatments)
    axes[0].invert_yaxis()
    fig.suptitle("Fig. F — Preservation vs collapse indices", fontsize=14)
    fig.tight_layout()
    return fig


def plot_roi_micro_panels(
    stats: Mapping[str, Mapping[str, np.ndarray]],
) -> plt.Figure:
    treatments = list(stats.keys())
    n = len(treatments)
    cols = 3 if n >= 4 else 2
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, treatment in zip(axes, treatments, strict=False):
        if treatment not in stats:
            ax.axis("off")
            continue
        entry = stats[treatment]
        wl = entry["wavelengths"]
        ax.fill_between(wl, entry["raw_q1"], entry["raw_q3"], color=RAW_COLOR, alpha=0.2)
        ax.plot(wl, entry["raw_median"], color=RAW_COLOR, label="Raw median")
        ax.fill_between(wl, entry["den_q1"], entry["den_q3"], color=DENOISED_COLOR, alpha=0.2)
        ax.plot(wl, entry["den_median"], color=DENOISED_COLOR, label="Denoised median")
        ax.axvspan(370.0, 382.0, color=ROI_COLOR, alpha=0.2, lw=0)
        ax.set_title(treatment)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Fig. G — UV-A ROI overlays (median ± IQR)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_effect_sizes(
    effect_table: "pd.DataFrame | Mapping[str, Sequence]",
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
) -> plt.Figure:
    import pandas as pd  # local import

    if not isinstance(effect_table, pd.DataFrame):
        effect_table = pd.DataFrame(effect_table)
    pivot = effect_table.pivot(index="treatment", columns="region", values="effect_size")
    treatments = list(pivot.index)
    x_out = pivot["outside_roi"].to_numpy()
    x_in = pivot["inside_roi"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(treatments))
    ax.plot(x_out, y, "o-", color=RAW_COLOR, label="Outside ROI")
    ax.plot(x_in, y, "o-", color=DENOISED_COLOR, label="Inside ROI")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(treatments)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (raw vs denoised)")
    ax.set_title("Fig. H — Treatment-wise effect sizes")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
