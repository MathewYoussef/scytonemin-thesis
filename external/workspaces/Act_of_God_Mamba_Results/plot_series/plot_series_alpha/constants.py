"""Constants shared across the plot_series_alpha scripts."""

from pathlib import Path

RAW_DATA_ROOT = Path("data/_staging")
DENOISED_DATA_ROOT = Path("data/denoised_full_run_staging")
MANIFEST_PATH = RAW_DATA_ROOT / "manifest.csv"
WAVELENGTH_GRID_PATH = RAW_DATA_ROOT / "wavelength_grid.npy"

ROI_MIN_NM = 370.0
ROI_MAX_NM = 382.0

DEFAULT_OUTPUT_DIR = Path("plot_series/plot_series_alpha/output")

HEATMAP_CMAP = "magma"
VARIANCE_RATIO_COLOR = "#1f77b4"
ROI_BAND_COLOR = "#ff7f0e"
RAW_COLOR = "#d62728"
DENOISED_COLOR = "#2ca02c"

DELTA_SAM_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
