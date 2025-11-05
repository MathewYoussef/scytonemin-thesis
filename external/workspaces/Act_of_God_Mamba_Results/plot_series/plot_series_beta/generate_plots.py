#!/usr/bin/env python3
"""
CLI entry point for generating the beta plot series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _import_modules():
    if __package__:
        from . import data_loader, metrics, paths, plotting  # type: ignore
        return data_loader, metrics, paths, plotting
    # Running as a script: fall back to relative imports.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))
    import data_loader  # type: ignore
    import metrics  # type: ignore
    import paths  # type: ignore
    import plotting  # type: ignore

    return data_loader, metrics, paths, plotting


data_loader, metrics, paths, plotting = _import_modules()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the denoising diagnostic plot series.")
    parser.add_argument("--raw-dir", type=Path, default=paths.RAW_STAGING, help="Path to raw spectra staging directory.")
    parser.add_argument(
        "--denoised-dir",
        type=Path,
        default=paths.DENOISED_STAGING,
        help="Path to denoised spectra staging directory.",
    )
    parser.add_argument(
        "--wavelength-grid",
        type=Path,
        default=paths.WAVELENGTH_GRID_PATH,
        help="Path to the wavelength grid (.npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.DEFAULT_OUTPUT_DIR,
        help="Where to write figure files.",
    )
    parser.add_argument(
        "--control-treatment",
        type=str,
        default="treatment_6",
        help="Treatment to use as the SAM control reference.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Bootstrap draws for the variance ratio ribbon (0 to disable).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap sampling.")
    parser.add_argument(
        "--roi",
        nargs=2,
        type=float,
        default=(370.0, 382.0),
        metavar=("LOWER", "UPPER"),
        help="Region of interest bounds in nm.",
    )
    parser.add_argument(
        "--sam-window",
        type=float,
        default=11.0,
        help="Sliding window width for ΔSAM (in nm).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=("treatment", "group"),
        default="treatment",
        help="Aggregation level for heatmaps (treatment collapses samples/angles, group keeps full detail).",
    )
    parser.add_argument(
        "--snr-delta-max",
        type=float,
        default=0.75,
        help="Clamp ΔSNR heatmap colors to ±value (dB).",
    )
    return parser.parse_args(argv)


def _compute_percent_reduction(raw: np.ndarray, den: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return float("nan")
    raw_slice = raw[mask]
    den_slice = den[mask]
    raw_mean = raw_slice.mean() if raw_slice.size else float("nan")
    den_mean = den_slice.mean() if den_slice.size else float("nan")
    if not np.isfinite(raw_mean) or raw_mean <= metrics.EPS:
        return float("nan")
    reduction = 100.0 * (1.0 - den_mean / np.maximum(raw_mean, metrics.EPS))
    return reduction


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roi_bounds: Tuple[float, float] = (float(args.roi[0]), float(args.roi[1]))
    snr_roi_bounds: Tuple[float, float] = (320.0, 480.0)

    bundle = data_loader.load_spectra(args.raw_dir, args.denoised_dir, args.wavelength_grid)
    frame = bundle.frame
    wavelengths = bundle.wavelengths
    inside_mask = (wavelengths >= roi_bounds[0]) & (wavelengths <= roi_bounds[1])
    outside_mask = ~inside_mask

    group_col = "treatment" if args.group_by == "treatment" else "group_label"

    group_snr = metrics.compute_group_snr(frame, wavelengths, group_col=group_col, roi_nm=snr_roi_bounds)
    snr_annotations = group_snr.mean_delta_inside.tolist()
    snr_order = np.asarray(group_snr.mean_delta_inside, dtype=float)
    if snr_order.size:
        order_indices = np.argsort(np.nan_to_num(snr_order, nan=-np.inf))[::-1]
    else:
        order_indices = np.arange(len(group_snr.labels))
    order_indices = order_indices.tolist()

    group_var = metrics.compute_group_variances(frame, wavelengths, group_col=group_col)
    reductions = [
        _compute_percent_reduction(raw_row, den_row, outside_mask)
        for raw_row, den_row in zip(group_var.raw, group_var.denoised, strict=True)
    ]

    variance_ratio = metrics.compute_variance_ratio(
        frame,
        wavelengths,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
    )

    sam_results = metrics.compute_sam_results(
        frame,
        wavelengths,
        control_treatment=args.control_treatment,
        window_nm=args.sam_window,
    )
    preservation_table = metrics.compute_preservation_indices(
        variance_ratio,
        sam_results,
        roi_nm=roi_bounds,
    )
    roi_stats = metrics.compute_roi_micro_panel_stats(frame, wavelengths, window_nm=(350.0, 400.0))
    effect_sizes = metrics.compute_treatment_effect_sizes(frame, wavelengths, roi_nm=roi_bounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "fig_A_snr_heatmap.png": plotting.plot_snr_heatmaps(
            group_snr,
            roi_nm=snr_roi_bounds,
            order=order_indices,
            snr_gains=snr_annotations,
            vmax_delta=args.snr_delta_max,
        ),
        "fig_B_snr_summary.png": plotting.plot_snr_summary(
            group_snr,
            order=order_indices,
        ),
        "fig_C_variance_heatmaps.png": plotting.plot_variance_heatmaps(
            group_var,
            roi_nm=roi_bounds,
            reductions=reductions,
            order=order_indices,
        ),
        "fig_D_variance_ratio.png": plotting.plot_variance_ratio_ribbon(
            variance_ratio,
            roi_nm=roi_bounds,
        ),
        "fig_E_sam_panels.png": plotting.plot_sam_panels(
            sorted(sam_results, key=lambda r: r.treatment),
            roi_nm=roi_bounds,
        ),
        "fig_F_preservation_indices.png": plotting.plot_preservation_indices(preservation_table),
        "fig_G_roi_micro_panels.png": plotting.plot_roi_micro_panels(roi_stats),
        "fig_H_effect_sizes.png": plotting.plot_effect_sizes(effect_sizes, roi_nm=roi_bounds),
    }

    for filename, fig in figures.items():
        output_path = args.output_dir / filename
        plotting.save_figure(fig, output_path)
        print(f"[saved] {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
