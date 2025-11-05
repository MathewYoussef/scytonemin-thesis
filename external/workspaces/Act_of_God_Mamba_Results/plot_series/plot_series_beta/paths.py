from __future__ import annotations

from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = REPO_ROOT / "data"
RAW_STAGING: Path = DATA_ROOT / "_staging"
DENOISED_STAGING: Path = DATA_ROOT / "denoised_full_run_staging"
WAVELENGTH_GRID_PATH: Path = RAW_STAGING / "wavelength_grid.npy"
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent / "outputs"

