"""Metric helpers for spectra diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import constants


def compute_variance_matrices(
    raw_stacks: Dict[str, np.ndarray], denoised_stacks: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ordered treatments with corresponding variance matrices."""
    treatments = sorted(raw_stacks.keys())
    raw_matrix = np.stack([np.var(raw_stacks[t], axis=0) for t in treatments])
    denoised_matrix = np.stack([np.var(denoised_stacks[t], axis=0) for t in treatments])
    return np.array(treatments), raw_matrix, denoised_matrix


def compute_variance_ratio(
    raw_stacks: Dict[str, np.ndarray],
    denoised_stacks: Dict[str, np.ndarray],
    *,
    n_bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """Compute the variance ratio and optional bootstrap confidence bands."""
    raw_concat = _concatenate_stacks(raw_stacks)
    den_concat = _concatenate_stacks(denoised_stacks)
    var_raw = np.var(raw_concat, axis=0)
    var_den = np.var(den_concat, axis=0)

    ratio = np.divide(
        var_raw,
        var_den,
        out=np.full_like(var_raw, np.nan),
        where=var_den > 0,
    )

    result = {"ratio": ratio, "var_raw": var_raw, "var_denoised": var_den}
    if n_bootstrap and n_bootstrap > 0:
        if rng is None:
            rng = np.random.default_rng(seed=2024)
        lower, upper = _bootstrap_variance_ratio(raw_concat, den_concat, n_bootstrap, rng)
        result["ci_lower"] = lower
        result["ci_upper"] = upper
    return result


def robust_quantile_limits(*arrays: Iterable[np.ndarray], q_low: float = 0.05, q_high: float = 0.95) -> Tuple[float, float]:
    """Return robust limits across several arrays, ignoring NaNs."""
    concatenated = np.concatenate([np.ravel(a) for a in arrays])
    finite = concatenated[np.isfinite(concatenated)]
    return np.quantile(finite, q_low), np.quantile(finite, q_high)


def _bootstrap_variance_ratio(
    raw_concat: np.ndarray, den_concat: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = raw_concat.shape[0]
    boot_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_var_raw = np.var(raw_concat[idx], axis=0)
        boot_var_den = np.var(den_concat[idx], axis=0)
        boot_ratio = np.divide(
            boot_var_raw,
            boot_var_den,
            out=np.full_like(boot_var_raw, np.nan),
            where=boot_var_den > 0,
        )
        boot_samples.append(boot_ratio)
    boot_matrix = np.stack(boot_samples)
    return np.nanquantile(boot_matrix, 0.025, axis=0), np.nanquantile(boot_matrix, 0.975, axis=0)


def _concatenate_stacks(stacks: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(list(stacks.values()), axis=0)


def compute_medians_and_iqr(stacks: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Return per-treatment medians and (q25, q75) envelopes."""
    medians: Dict[str, np.ndarray] = {}
    iqrs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for treatment, spectra in stacks.items():
        medians[treatment] = np.median(spectra, axis=0)
        q25 = np.quantile(spectra, 0.25, axis=0)
        q75 = np.quantile(spectra, 0.75, axis=0)
        iqrs[treatment] = (q25, q75)
    return medians, iqrs


def window_size_from_nm(wavelengths: np.ndarray, window_nm: float) -> int:
    """Convert a wavelength span in nm to the closest odd window size in samples."""
    if wavelengths.size < 2:
        raise ValueError("Need at least two wavelengths to determine spacing.")
    spacing = np.diff(wavelengths[:2])[0]
    estimated = max(3, int(round(window_nm / spacing)))
    if estimated % 2 == 0:
        estimated += 1
    return estimated


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the spectral angle (SAM) in radians between vectors a and b."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    cos_theta = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def sam_to_reference(
    medians: Dict[str, np.ndarray], reference: np.ndarray
) -> Dict[str, float]:
    """Compute SAM (in degrees) between each treatment median and the reference."""
    return {t: np.degrees(spectral_angle(spec, reference)) for t, spec in medians.items()}


def delta_sam_sliding(
    medians_raw: Dict[str, np.ndarray],
    medians_den: Dict[str, np.ndarray],
    reference_raw: np.ndarray,
    reference_den: np.ndarray,
    wavelengths: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute sliding-window SAM for raw/denoised medians and their delta."""
    if window_size >= len(reference_raw):
        raise ValueError("Window size must be smaller than the spectrum length.")
    window_centers = _window_centers(window_size, wavelengths)
    sam_raw_map: Dict[str, np.ndarray] = {}
    sam_den_map: Dict[str, np.ndarray] = {}
    delta_map: Dict[str, np.ndarray] = {}

    ref_raw_windows = sliding_window_view(reference_raw, window_size)
    ref_den_windows = sliding_window_view(reference_den, window_size)

    for treatment in medians_raw.keys():
        tr_raw_windows = sliding_window_view(medians_raw[treatment], window_size)
        tr_den_windows = sliding_window_view(medians_den[treatment], window_size)

        sam_raw = _sam_for_windows(tr_raw_windows, ref_raw_windows)
        sam_den = _sam_for_windows(tr_den_windows, ref_den_windows)
        sam_raw_map[treatment] = sam_raw
        sam_den_map[treatment] = sam_den
        delta_map[treatment] = sam_den - sam_raw

    return window_centers, sam_raw_map, sam_den_map, delta_map


def _sam_for_windows(windows_a: np.ndarray, windows_b: np.ndarray) -> np.ndarray:
    dots = np.einsum("ij,ij->i", windows_a, windows_b)
    norms_a = np.linalg.norm(windows_a, axis=1)
    norms_b = np.linalg.norm(windows_b, axis=1)
    denom = norms_a * norms_b
    cos_theta = np.divide(dots, denom, out=np.full_like(dots, np.nan), where=denom > 0)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _window_centers(window_size: int, wavelengths: np.ndarray) -> np.ndarray:
    half = window_size // 2
    return wavelengths[half : len(wavelengths) - half]


def compute_preservation_indices(
    treatments: Iterable[str],
    wavelengths: np.ndarray,
    raw_variances: np.ndarray,
    denoised_variances: np.ndarray,
    delta_map: Dict[str, np.ndarray],
    delta_centers: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute noise-collapse and pigment-preservation indices per treatment."""
    ratio_matrix = np.divide(
        raw_variances,
        denoised_variances,
        out=np.full_like(raw_variances, np.nan),
        where=denoised_variances > 0,
    )

    outside_mask = (wavelengths < constants.ROI_MIN_NM) | (wavelengths > constants.ROI_MAX_NM)
    inside_mask_delta = (delta_centers >= constants.ROI_MIN_NM) & (delta_centers <= constants.ROI_MAX_NM)

    indices: Dict[str, Dict[str, float]] = {}
    for idx, treatment in enumerate(treatments):
        ratio = ratio_matrix[idx]
        noise_collapse = float(np.nanmean(ratio[outside_mask]))

        delta = delta_map[treatment]
        if np.any(inside_mask_delta):
            pigment_pres = 1.0 - float(np.nanmean(np.abs(delta[inside_mask_delta])))
        else:
            pigment_pres = np.nan

        indices[treatment] = {
            "noise_collapse": noise_collapse,
            "pigment_preservation": pigment_pres,
        }
    return indices


def compute_effect_sizes(
    raw_stacks: Dict[str, np.ndarray],
    denoised_stacks: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute Cohen's d effect sizes inside and outside the ROI."""
    outside_mask = (wavelengths < constants.ROI_MIN_NM) | (wavelengths > constants.ROI_MAX_NM)
    inside_mask = ~outside_mask

    effect_sizes: Dict[str, Dict[str, float]] = {}
    for treatment in raw_stacks.keys():
        raw = raw_stacks[treatment]
        den = denoised_stacks[treatment]

        raw_outside = raw[:, outside_mask].ravel()
        den_outside = den[:, outside_mask].ravel()
        raw_inside = raw[:, inside_mask].ravel()
        den_inside = den[:, inside_mask].ravel()

        effect_sizes[treatment] = {
            "outside_roi": cohens_d(raw_outside, den_outside),
            "inside_roi": cohens_d(raw_inside, den_inside),
        }
    return effect_sizes


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cohen's d with pooled standard deviation."""
    mean_a = np.nanmean(sample_a)
    mean_b = np.nanmean(sample_b)
    var_a = np.nanvar(sample_a, ddof=1)
    var_b = np.nanvar(sample_b, ddof=1)

    n_a = np.sum(np.isfinite(sample_a))
    n_b = np.sum(np.isfinite(sample_b))

    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    if pooled == 0:
        return np.nan
    return float((mean_b - mean_a) / pooled)
