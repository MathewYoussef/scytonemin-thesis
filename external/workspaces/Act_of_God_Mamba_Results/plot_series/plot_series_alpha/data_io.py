"""Utilities for loading raw and denoised spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectrumKey:
    """Metadata describing a single spectrum."""

    treatment: str
    sample: str
    angle: str
    replicate: int
    kind: str  # "raw" or "denoised"


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the manifest and sort by logical acquisition order."""
    df = pd.read_csv(manifest_path)
    df["replicate"] = df["relative_path"].map(_infer_replicate_id)
    df = df.sort_values(["treatment", "sample", "angle", "replicate"]).reset_index(drop=True)
    return df


def load_wavelength_grid(path: Path) -> np.ndarray:
    """Load the shared wavelength grid."""
    wavelengths = np.load(path)
    if wavelengths.ndim != 1:
        raise ValueError(f"Wavelength grid at {path} is expected to be 1-D, found shape {wavelengths.shape}.")
    return wavelengths


def load_treatment_stacks(
    manifest_df: pd.DataFrame, base_dir: Path, *, suffix: str = ""
) -> Dict[str, np.ndarray]:
    """Return a dictionary mapping treatment -> stacked spectra matrix."""
    stacks: Dict[str, np.ndarray] = {}
    for treatment, group in manifest_df.groupby("treatment"):
        spectra = [_load_spectrum(base_dir, rel_path, suffix) for rel_path in group["relative_path"]]
        if not spectra:
            continue
        stacks[treatment] = np.stack(spectra, axis=0)
    return stacks


def iter_spectra(
    manifest_df: pd.DataFrame, base_dir: Path, *, suffix: str = "", kind: str
) -> Iterable[Tuple[SpectrumKey, np.ndarray]]:
    """Yield (metadata, spectrum) pairs in manifest order."""
    for _, row in manifest_df.iterrows():
        rel_path = row["relative_path"]
        spectrum = _load_spectrum(base_dir, rel_path, suffix)
        yield SpectrumKey(
            treatment=row["treatment"],
            sample=row["sample"],
            angle=row["angle"],
            replicate=row["replicate"],
            kind=kind,
        ), spectrum


def _load_spectrum(base_dir: Path, relative_path: str, suffix: str) -> np.ndarray:
    spectrum_path = _resolve_relative_path(base_dir, relative_path, suffix)
    array = np.load(spectrum_path)
    if array.ndim != 1:
        raise ValueError(f"Spectrum at {spectrum_path} is expected to be 1-D, found shape {array.shape}.")
    return array


def _resolve_relative_path(base_dir: Path, relative_path: str, suffix: str) -> Path:
    path = (base_dir / relative_path).resolve()
    if suffix:
        stem = path.stem
        path = path.with_name(f"{stem}{suffix}{path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Expected spectrum missing: {path}")
    return path


def _infer_replicate_id(relative_path: str) -> int:
    stem = Path(relative_path).stem
    pieces = stem.split("_")
    for token in reversed(pieces):
        if token.isdigit():
            return int(token)
    raise ValueError(f"Unable to infer replicate id from {relative_path}")

