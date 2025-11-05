"""
Numerical kernels for the six diagnostic panels.

Each function is pure (no filesystem IO) so callers can mix and match in scripts
or notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


EPS = 1e-12
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class GroupVarianceResult:
    labels: Sequence[str]
    raw: np.ndarray
    denoised: np.ndarray
    wavelengths: np.ndarray


@dataclass(frozen=True)
class VarianceRatioResult:
    ratio: np.ndarray
    raw_var: np.ndarray
    denoised_var: np.ndarray
    ci_low: Optional[np.ndarray]
    ci_high: Optional[np.ndarray]
    wavelengths: np.ndarray


@dataclass(frozen=True)
class SAMResult:
    treatment: str
    sam_raw: float
    sam_denoised: float
    sam_raw_windowed: np.ndarray
    sam_denoised_windowed: np.ndarray
    delta_windowed: np.ndarray
    window_centers: np.ndarray


@dataclass(frozen=True)
class GroupSNRResult:
    labels: Sequence[str]
    snr_raw_db: np.ndarray
    snr_denoised_db: np.ndarray
    snr_delta_db: np.ndarray
    wavelengths: np.ndarray
    mean_delta_outside: np.ndarray
    mean_delta_inside: np.ndarray


def compute_group_variances(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    group_col: str = "group_label",
) -> GroupVarianceResult:
    labels: list[str] = []
    raw_rows: list[np.ndarray] = []
    den_rows: list[np.ndarray] = []
    for label, group in frame.groupby(group_col):
        raw_stack = np.stack(group["raw"].to_numpy(), axis=0)
        den_stack = np.stack(group["denoised"].to_numpy(), axis=0)
        ddof = 1 if raw_stack.shape[0] > 1 else 0
        raw_var = raw_stack.var(axis=0, ddof=ddof)
        den_var = den_stack.var(axis=0, ddof=ddof)
        labels.append(label)
        raw_rows.append(raw_var)
        den_rows.append(den_var)
    raw_arr = np.vstack(raw_rows) if raw_rows else np.empty((0, wavelengths.size))
    den_arr = np.vstack(den_rows) if den_rows else np.empty((0, wavelengths.size))
    return GroupVarianceResult(labels=labels, raw=raw_arr, denoised=den_arr, wavelengths=wavelengths)


def _robust_snr_db(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 2:
        raise ValueError("Stack must be 2D [replicate, wavelength]")
    median = np.median(stack, axis=0)
    noise = np.median(np.abs(stack - median), axis=0) * MAD_TO_SIGMA
    noise = np.clip(noise, EPS, None)
    signal = np.clip(np.abs(median), EPS, None)
    snr_linear = signal / noise
    return 20.0 * np.log10(np.clip(snr_linear, EPS, None))


def compute_group_snr(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    group_col: str = "group_label",
    roi_nm: Tuple[float, float] = (320.0, 480.0),
) -> GroupSNRResult:
    labels: list[str] = []
    raw_rows: list[np.ndarray] = []
    den_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []
    mean_out: list[float] = []
    mean_in: list[float] = []
    lower, upper = roi_nm
    inside_mask = (wavelengths >= lower) & (wavelengths <= upper)
    outside_mask = ~inside_mask
    for label, group in frame.groupby(group_col):
        raw_stack = np.stack(group["raw"].to_numpy(), axis=0)
        den_stack = np.stack(group["denoised"].to_numpy(), axis=0)
        snr_raw = _robust_snr_db(raw_stack)
        snr_den = _robust_snr_db(den_stack)
        labels.append(label)
        raw_rows.append(snr_raw)
        den_rows.append(snr_den)
        delta = snr_den - snr_raw
        delta_rows.append(delta)
        mean_out.append(float(np.nanmean(delta[outside_mask])) if outside_mask.any() else np.nan)
        mean_in.append(float(np.nanmean(delta[inside_mask])) if inside_mask.any() else np.nan)
    raw_arr = np.vstack(raw_rows) if raw_rows else np.empty((0, wavelengths.size))
    den_arr = np.vstack(den_rows) if den_rows else np.empty((0, wavelengths.size))
    delta_arr = np.vstack(delta_rows) if delta_rows else np.empty((0, wavelengths.size))
    return GroupSNRResult(
        labels=labels,
        snr_raw_db=raw_arr,
        snr_denoised_db=den_arr,
        snr_delta_db=delta_arr,
        wavelengths=wavelengths,
        mean_delta_outside=np.asarray(mean_out, dtype=float),
        mean_delta_inside=np.asarray(mean_in, dtype=float),
    )


def compute_variance_ratio(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    *,
    bootstrap_samples: int = 500,
    random_seed: Optional[int] = None,
    clip_min: float = 1e-12,
) -> VarianceRatioResult:
    raw_stack = np.stack(frame["raw"].to_numpy(), axis=0)
    den_stack = np.stack(frame["denoised"].to_numpy(), axis=0)
    raw_var = raw_stack.var(axis=0, ddof=1)
    den_var = np.clip(den_stack.var(axis=0, ddof=1), clip_min, None)
    ratio = raw_var / den_var
    ci_low = ci_high = None
    if bootstrap_samples > 0:
        rng = np.random.default_rng(random_seed)
        boot_ratios = np.empty((bootstrap_samples, wavelengths.size), dtype=np.float64)
        n = raw_stack.shape[0]
        for i in range(bootstrap_samples):
            idx = rng.integers(0, n, size=n)
            r_var = raw_stack[idx].var(axis=0, ddof=1)
            d_var = np.clip(den_stack[idx].var(axis=0, ddof=1), clip_min, None)
            boot_ratios[i] = r_var / d_var
        ci_low = np.percentile(boot_ratios, 2.5, axis=0)
        ci_high = np.percentile(boot_ratios, 97.5, axis=0)
    return VarianceRatioResult(
        ratio=ratio,
        raw_var=raw_var,
        denoised_var=den_var,
        ci_low=ci_low,
        ci_high=ci_high,
        wavelengths=wavelengths,
    )


def _sam_angle(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < EPS or b_norm < EPS:
        return np.nan
    cos_theta = np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _window_centers(wavelengths: np.ndarray, window_size: int) -> np.ndarray:
    half = window_size // 2
    if window_size % 2 == 0:
        return wavelengths[half - 1 : -(half)]
    return wavelengths[half : -half or None]


def _sliding_sam(series: np.ndarray, reference: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if window_size <= 1 or window_size > series.size:
        window_size = min(series.size, max(1, window_size))
    if window_size % 2 == 0:
        window_size += 1
    windows = sliding_window_view(series, window_shape=window_size)
    ref_windows = sliding_window_view(reference, window_shape=window_size)
    sams = np.array([_sam_angle(w, r) for w, r in zip(windows, ref_windows)])
    centers = _window_centers(np.arange(series.size), window_size)
    return sams, centers


def compute_treatment_medians(frame: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    medians: Dict[str, Dict[str, np.ndarray]] = {}
    for treatment, group in frame.groupby("treatment"):
        raw_stack = np.stack(group["raw"].to_numpy(), axis=0)
        den_stack = np.stack(group["denoised"].to_numpy(), axis=0)
        medians[treatment] = {
            "raw": np.median(raw_stack, axis=0),
            "denoised": np.median(den_stack, axis=0),
        }
    return medians


def compute_sam_results(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    *,
    control_treatment: str,
    window_nm: float = 11.0,
) -> Sequence[SAMResult]:
    medians = compute_treatment_medians(frame)
    if control_treatment not in medians:
        raise KeyError(f"Control treatment '{control_treatment}' not present in data.")
    reference = medians[control_treatment]["denoised"]
    step = float(np.median(np.diff(wavelengths)))
    window_size = max(1, int(round(window_nm / step)))
    results: list[SAMResult] = []
    for treatment, spectra in medians.items():
        sam_raw = _sam_angle(spectra["raw"], reference)
        sam_denoised = _sam_angle(spectra["denoised"], reference)
        sam_raw_win, centers_idx = _sliding_sam(spectra["raw"], reference, window_size)
        sam_den_win, _ = _sliding_sam(spectra["denoised"], reference, window_size)
        centers_wl = wavelengths[centers_idx.astype(int)]
        results.append(
            SAMResult(
                treatment=treatment,
                sam_raw=sam_raw,
                sam_denoised=sam_denoised,
                sam_raw_windowed=sam_raw_win,
                sam_denoised_windowed=sam_den_win,
                delta_windowed=sam_den_win - sam_raw_win,
                window_centers=centers_wl,
            )
        )
    return results


def compute_preservation_indices(
    variance_ratio: VarianceRatioResult,
    sam_results: Sequence[SAMResult],
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
) -> pd.DataFrame:
    wl = variance_ratio.wavelengths
    lower, upper = roi_nm
    outside_mask = (wl < lower) | (wl > upper)
    rows = []
    for sam in sam_results:
        delta_wl = sam.window_centers
        inside_roi = (delta_wl >= lower) & (delta_wl <= upper)
        if inside_roi.any():
            pigment_preservation = 1.0 - np.median(np.abs(sam.delta_windowed[inside_roi]))
        else:
            pigment_preservation = np.nan
        noise_collapse = float(np.nanmean(variance_ratio.ratio[outside_mask])) if outside_mask.any() else np.nan
        rows.append(
            {
                "treatment": sam.treatment,
                "noise_collapse": noise_collapse,
                "pigment_preservation": pigment_preservation,
            }
        )
    return pd.DataFrame(rows)


def compute_roi_micro_panel_stats(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    *,
    window_nm: Tuple[float, float] = (350.0, 400.0),
) -> Dict[str, Dict[str, np.ndarray]]:
    lower, upper = window_nm
    roi_mask = (wavelengths >= lower) & (wavelengths <= upper)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for treatment, group in frame.groupby("treatment"):
        raw_stack = np.stack(group["raw"].to_numpy(), axis=0)[:, roi_mask]
        den_stack = np.stack(group["denoised"].to_numpy(), axis=0)[:, roi_mask]
        results[treatment] = {
            "wavelengths": wavelengths[roi_mask],
            "raw_median": np.median(raw_stack, axis=0),
            "raw_q1": np.quantile(raw_stack, 0.25, axis=0),
            "raw_q3": np.quantile(raw_stack, 0.75, axis=0),
            "den_median": np.median(den_stack, axis=0),
            "den_q1": np.quantile(den_stack, 0.25, axis=0),
            "den_q3": np.quantile(den_stack, 0.75, axis=0),
        }
    return results


def _flatten_region(stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty(0, dtype=np.float64)
    return stack[:, mask].astype(np.float64, copy=False).ravel()


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= EPS:
        return np.nan
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def compute_treatment_effect_sizes(
    frame: pd.DataFrame,
    wavelengths: np.ndarray,
    *,
    roi_nm: Tuple[float, float] = (370.0, 382.0),
) -> pd.DataFrame:
    wl = wavelengths
    inside_mask = (wl >= roi_nm[0]) & (wl <= roi_nm[1])
    outside_mask = ~inside_mask
    rows = []
    for treatment, group in frame.groupby("treatment"):
        raw_stack = np.stack(group["raw"].to_numpy(), axis=0)
        den_stack = np.stack(group["denoised"].to_numpy(), axis=0)
        raw_out = _flatten_region(raw_stack, outside_mask)
        den_out = _flatten_region(den_stack, outside_mask)
        raw_in = _flatten_region(raw_stack, inside_mask)
        den_in = _flatten_region(den_stack, inside_mask)
        rows.append(
            {
                "treatment": treatment,
                "region": "outside_roi",
                "effect_size": _cohens_d(raw_out, den_out),
            }
        )
        rows.append(
            {
                "treatment": treatment,
                "region": "inside_roi",
                "effect_size": _cohens_d(raw_in, den_in),
            }
        )
    return pd.DataFrame(rows)
