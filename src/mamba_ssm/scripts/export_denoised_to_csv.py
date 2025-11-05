#!/usr/bin/env python3
"""
Export denoised spectra into a single analysis-ready CSV.

For every entry in the provided manifest the script:
- Loads the denoised `.npy` spectrum (suffix `_denoised.npy` by default).
- Merges manifest metadata (treatment, sample, angle, group_id, etc.).
- Optionally joins per-spectrum dose/irradiance features.
- Expands the spectrum into wide columns keyed by wavelength.

Example:
    python scripts/export_denoised_to_csv.py \
        --manifest data/_staging/manifest.csv \
        --denoised-root data/denoised_full_run_staging \
        --output-csv final_analytics/denoised_full_run.csv \
        --dose-features data/metadata/dose_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_WAVELENGTH_GRID = "data/spectra_for_fold/wavelength_grid.npy"
DEFAULT_SUFFIX = "_denoised.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest containing a `relative_path` column (e.g., data/_staging/manifest.csv).",
    )
    parser.add_argument(
        "--denoised-root",
        required=True,
        help="Root directory that mirrors manifest layout and stores *_denoised.npy files.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the aggregated CSV.",
    )
    parser.add_argument(
        "--wavelength-grid",
        default=DEFAULT_WAVELENGTH_GRID,
        help=f"Path to wavelength grid .npy (default: {DEFAULT_WAVELENGTH_GRID}).",
    )
    parser.add_argument(
        "--dose-features",
        default=None,
        help="Optional CSV of dose features keyed by `relative_path`.",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"Suffix applied to each relative_path to locate denoised spectra (default: {DEFAULT_SUFFIX}).",
    )
    return parser.parse_args()


def format_wavelength_columns(wavelengths: np.ndarray) -> List[str]:
    cols: List[str] = []
    for wl in wavelengths:
        label = format(float(wl), ".3f").rstrip("0").rstrip(".")
        cols.append(f"reflectance_nm_{label}")
    return cols


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest {manifest_path} missing columns: {sorted(missing)}")
    return df


def load_dose_features(path: Path) -> pd.DataFrame:
    dose_df = pd.read_csv(path)
    if "relative_path" not in dose_df.columns:
        raise KeyError(f"Dose feature CSV {path} must contain `relative_path`.")

    drop_cols = [
        "treatment",
        "treatment_number",
        "sample",
        "angle",
        "group_id",
        "column_label",
    ]
    dose_df = dose_df.drop(columns=[c for c in drop_cols if c in dose_df.columns])
    return dose_df


def iter_paths(
    manifest_paths: Iterable[str],
    root: Path,
    suffix: str,
) -> Iterable[Tuple[str, Path]]:
    for rel in manifest_paths:
        rel_path = Path(rel)
        den_path = root / rel_path.with_name(rel_path.stem + suffix)
        yield rel, den_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    den_root = Path(args.denoised_root).resolve()
    out_csv = Path(args.output_csv).resolve()
    wavelengths = np.load(Path(args.wavelength_grid).resolve()).astype(np.float32)
    if wavelengths.ndim != 1:
        raise ValueError("Wavelength grid must be 1-D.")

    manifest_df = load_manifest(manifest_path)
    if args.dose_features:
        dose_df = load_dose_features(Path(args.dose_features).resolve())
        manifest_df = manifest_df.merge(
            dose_df,
            on="relative_path",
            how="left",
            validate="one_to_one",
        )

    spectral_cols = format_wavelength_columns(wavelengths)
    spectra = np.empty((len(manifest_df), len(wavelengths)), dtype=np.float32)

    missing_paths: List[str] = []
    for idx, (rel, den_path) in enumerate(iter_paths(manifest_df["relative_path"], den_root, args.suffix)):
        if not den_path.exists():
            missing_paths.append(str(den_path))
            spectra[idx] = np.nan
            continue
        spectrum = np.load(den_path).astype(np.float32)
        if spectrum.ndim != 1 or spectrum.shape[0] != wavelengths.shape[0]:
            raise ValueError(
                f"Spectra mismatch for {den_path}: expected {wavelengths.shape[0]} values, "
                f"got {spectrum.shape}"
            )
        spectra[idx] = spectrum

    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} denoised spectra. Examples: {missing_paths[:5]}"
        )

    spectra_df = pd.DataFrame(spectra, columns=spectral_cols)
    combined_df = pd.concat([manifest_df.reset_index(drop=True), spectra_df], axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(combined_df)} rows x {combined_df.shape[1]} columns to {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
