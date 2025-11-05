"""Entry-point script for the plot_series_alpha diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import constants
from .data_io import load_manifest, load_treatment_stacks, load_wavelength_grid
from .metrics import (
    compute_effect_sizes,
    compute_medians_and_iqr,
    compute_preservation_indices,
    compute_variance_matrices,
    compute_variance_ratio,
    delta_sam_sliding,
    sam_to_reference,
    window_size_from_nm,
)
from .plots import (
    plot_delta_sam,
    plot_effect_sizes,
    plot_preservation_indices,
    plot_roi_overlays,
    plot_sam_to_control,
    plot_variance_heatmaps,
    plot_variance_ratio,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for raw vs denoised spectra.")
    parser.add_argument("--raw-dir", type=Path, default=constants.RAW_DATA_ROOT, help="Root directory with raw spectra.")
    parser.add_argument(
        "--denoised-dir",
        type=Path,
        default=constants.DENOISED_DATA_ROOT,
        help="Root directory with denoised spectra.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=constants.MANIFEST_PATH,
        help="Manifest CSV describing raw spectra.",
    )
    parser.add_argument(
        "--wavelength-grid",
        type=Path,
        default=constants.WAVELENGTH_GRID_PATH,
        help="Shared wavelength grid .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap samples for the variance ratio confidence interval (0 disables bootstrap).",
    )
    parser.add_argument(
        "--reference-treatment",
        type=str,
        default="treatment_1",
        help="Treatment to use as the reference spectrum for SAM calculations.",
    )
    parser.add_argument(
        "--window-nm",
        type=float,
        default=10.0,
        help="Approximate sliding window width in nanometers for ΔSAM computations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    wavelengths = load_wavelength_grid(args.wavelength_grid)

    raw_stacks = load_treatment_stacks(manifest, args.raw_dir)
    denoised_stacks = load_treatment_stacks(manifest, args.denoised_dir, suffix="_denoised")

    treatments, raw_matrix, denoised_matrix = compute_variance_matrices(raw_stacks, denoised_stacks)
    treatments_list = list(treatments)

    heatmap_path = args.output_dir / "variance_heatmaps.png"
    plot_variance_heatmaps(treatments, raw_matrix, denoised_matrix, wavelengths, output_path=heatmap_path)

    bootstrap_samples = max(0, args.bootstrap_samples)
    ratio_data = compute_variance_ratio(raw_stacks, denoised_stacks, n_bootstrap=bootstrap_samples)
    ratio_path = args.output_dir / "variance_ratio.png"
    plot_variance_ratio(wavelengths, ratio_data, output_path=ratio_path)

    medians_raw, iqrs_raw = compute_medians_and_iqr(raw_stacks)
    medians_den, iqrs_den = compute_medians_and_iqr(denoised_stacks)

    reference_treatment = args.reference_treatment
    if reference_treatment not in medians_raw:
        raise ValueError(f"Reference treatment '{reference_treatment}' not found in data.")

    reference_raw = medians_raw[reference_treatment]
    reference_den = medians_den[reference_treatment]

    sam_raw = sam_to_reference(medians_raw, reference_raw)
    sam_den = sam_to_reference(medians_den, reference_den)

    window_size = window_size_from_nm(wavelengths, args.window_nm)
    delta_wavelengths, _, _, delta_map = delta_sam_sliding(
        medians_raw,
        medians_den,
        reference_raw,
        reference_den,
        wavelengths,
        window_size,
    )

    sam_plot_path = args.output_dir / "sam_to_reference.png"
    plot_sam_to_control(treatments_list, sam_raw, sam_den, output_path=sam_plot_path)

    delta_plot_path = args.output_dir / "delta_sam.png"
    plot_delta_sam(delta_wavelengths, treatments_list, delta_map, output_path=delta_plot_path)

    preservation_indices = compute_preservation_indices(
        treatments,
        wavelengths,
        raw_matrix,
        denoised_matrix,
        delta_map,
        delta_wavelengths,
    )
    preservation_path = args.output_dir / "preservation_vs_collapse.png"
    plot_preservation_indices(preservation_indices, output_path=preservation_path)

    roi_plot_path = args.output_dir / "roi_overlays.png"
    plot_roi_overlays(
        wavelengths,
        treatments_list,
        medians_raw,
        iqrs_raw,
        medians_den,
        iqrs_den,
        output_path=roi_plot_path,
    )

    effect_sizes = compute_effect_sizes(raw_stacks, denoised_stacks, wavelengths)
    effect_plot_path = args.output_dir / "effect_sizes.png"
    plot_effect_sizes(treatments_list, effect_sizes, output_path=effect_plot_path)

    print(f"Saved variance heatmaps to {heatmap_path}")
    print(f"Saved variance ratio plot to {ratio_path}")
    print(f"Saved SAM comparison plot to {sam_plot_path}")
    print(f"Saved ΔSAM plot to {delta_plot_path}")
    print(f"Saved preservation index plot to {preservation_path}")
    print(f"Saved ROI overlays to {roi_plot_path}")
    print(f"Saved effect size plot to {effect_plot_path}")


if __name__ == "__main__":
    main()
