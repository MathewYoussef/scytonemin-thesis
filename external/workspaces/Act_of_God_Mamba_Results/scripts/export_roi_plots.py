#!/usr/bin/env python3
"""Batch export raw vs. denoised ROI overlays using a manifest."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from local_plot_creation import (  # noqa: E402
    PlotMetadata,
    PlotSettings,
    make_output_name,
    plot_roi_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV manifest with relative_path column")
    parser.add_argument("--raw-root", required=True, help="Root directory containing raw spectra")
    parser.add_argument("--denoised-root", required=True, help="Root directory with *_denoised.npy files")
    parser.add_argument("--wavelength-grid", required=True, help="Path to wavelength_grid.npy")
    parser.add_argument("--output-dir", required=True, help="Directory where plots will be written")
    parser.add_argument("--limit", type=int, default=60, help="Maximum number of spectra to plot")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling spectra")
    parser.add_argument("--roi", type=float, nargs=2, default=(320.0, 500.0), help="ROI bounds in nm")
    parser.add_argument("--half-width-nm", type=float, default=7.0, help="Half-width around each dip")
    parser.add_argument("--baseline-guard-nm", type=float, default=10.0, help="Guard band for baseline fit")
    parser.add_argument("--poly-order", type=int, default=2, help="Polynomial order for baseline fitting")
    parser.add_argument(
        "--enforce-min-separation-nm",
        type=float,
        default=3.0,
        help="Minimum separation between detected dips",
    )
    parser.add_argument("--max-dips", type=int, default=6, help="Maximum number of dips to display")
    parser.add_argument("--dpi", type=int, default=150, help="Image resolution for saved plots")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return rows


def sample_indices(total: int, limit: int, seed: int) -> List[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if limit <= 0 or limit >= total:
        return indices
    return indices[:limit]


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw_root = Path(args.raw_root).resolve()
    denoised_root = Path(args.denoised_root).resolve()
    wavelength_grid_path = Path(args.wavelength_grid).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found")
    if not wavelength_grid_path.exists():
        raise FileNotFoundError(f"Wavelength grid {wavelength_grid_path} not found")

    rows = load_manifest_rows(manifest_path)
    if not rows:
        raise ValueError(f"No entries found in manifest {manifest_path}")

    indices = sample_indices(len(rows), args.limit, args.seed)
    wavelengths = np.load(wavelength_grid_path).astype(np.float32)

    settings = PlotSettings(
        roi=tuple(args.roi),
        half_width_nm=float(args.half_width_nm),
        baseline_guard_nm=float(args.baseline_guard_nm),
        poly_order=int(args.poly_order),
        enforce_min_separation_nm=float(args.enforce_min_separation_nm),
        max_dips=int(args.max_dips),
        dpi=int(args.dpi),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_missing = 0

    for plot_index, row_idx in enumerate(indices):
        row = rows[row_idx]
        relative_path = str(row.get("relative_path", "")).strip()
        if not relative_path:
            skipped_missing += 1
            continue

        raw_path = raw_root / relative_path
        denoised_rel = Path(relative_path).with_name(Path(relative_path).stem + "_denoised.npy")
        denoised_path = denoised_root / denoised_rel

        if not raw_path.exists() or not denoised_path.exists():
            skipped_missing += 1
            continue

        raw_spectrum = np.load(raw_path)
        denoised_spectrum = np.load(denoised_path)

        metadata = PlotMetadata(
            relative_path=relative_path,
            treatment=str(row.get("treatment", "")).strip(),
            sample=str(row.get("sample", "")).strip(),
            angle=str(row.get("angle", "")).strip(),
        )

        outfile = make_output_name(output_dir, relative_path, plot_index)
        plot_roi_windows(raw_spectrum, denoised_spectrum, wavelengths, outfile, settings, metadata)
        saved += 1

    print(f"Saved {saved} plot(s) to {output_dir}")
    if skipped_missing:
        print(f"Skipped {skipped_missing} entries due to missing spectra")


if __name__ == "__main__":
    main()
