"""
Routines for pulling paired raw + denoised spectra into tidy data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectraDataBundle:
    """Container holding all spectra and supporting metadata."""

    frame: pd.DataFrame
    wavelengths: np.ndarray

    @property
    def treatments(self) -> Iterable[str]:
        return tuple(sorted(self.frame["treatment"].unique()))


def load_manifest(raw_root: Path) -> pd.DataFrame:
    manifest_path = raw_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    expected_columns = {"relative_path", "treatment", "sample", "angle", "group_id"}
    missing = expected_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return manifest


def load_wavelength_grid(wavelength_path: Path) -> np.ndarray:
    if not wavelength_path.exists():
        raise FileNotFoundError(f"Missing wavelength grid npy at {wavelength_path}")
    wavelengths = np.load(wavelength_path)
    if wavelengths.ndim != 1:
        raise ValueError("Wavelength grid must be a 1D array")
    return wavelengths


def _load_pair(raw_path: Path, denoised_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw spectrum not found: {raw_path}")
    if not denoised_path.exists():
        raise FileNotFoundError(f"Denoised spectrum not found: {denoised_path}")
    raw = np.load(raw_path).astype(np.float64, copy=False)
    denoised = np.load(denoised_path).astype(np.float64, copy=False)
    if raw.shape != denoised.shape:
        raise ValueError(f"Shape mismatch raw vs denoised for {raw_path}")
    return raw, denoised


def load_spectra(
    raw_root: Path,
    denoised_root: Path,
    wavelength_path: Path,
) -> SpectraDataBundle:
    """
    Load all spectra into a tidy dataframe with paired raw and denoised arrays.

    Columns:
        - treatment / sample / angle / replicate / group_id metadata
        - raw: np.ndarray
        - denoised: np.ndarray
    """
    manifest = load_manifest(raw_root)
    wavelengths = load_wavelength_grid(wavelength_path)
    records = []
    for entry in manifest.itertuples():
        relative_path = Path(entry.relative_path)
        raw_path = raw_root / relative_path
        den_path = denoised_root / relative_path.parent / f"{relative_path.stem}_denoised.npy"
        raw, den = _load_pair(raw_path, den_path)
        records.append(
            {
                "treatment": entry.treatment,
                "sample": entry.sample,
                "angle": entry.angle,
                "group_id": entry.group_id,
                "replicate": relative_path.stem,
                "raw": raw,
                "denoised": den,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["group_label"] = (
        frame["treatment"].astype(str)
        + "::"
        + frame["sample"].astype(str)
        + "::"
        + frame["angle"].astype(str)
    )
    return SpectraDataBundle(frame=frame, wavelengths=wavelengths)

