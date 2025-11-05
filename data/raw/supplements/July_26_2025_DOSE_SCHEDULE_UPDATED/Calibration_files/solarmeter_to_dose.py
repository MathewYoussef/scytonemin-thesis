"""
Calibrate UVA & UVB Solarmeter readings and compute first-hour UVB dose.

Assumes the following files live in:
    /Users/mathewyoussef/Desktop/extremophile_detection/Data/July_4_2025/Calibration_files

    ├── uva_readings_raw.csv                 # 2 readings per grid
    ├── uvb_readings_raw.csv                 # 2 readings × (t = 0, 30, 60 min) per grid
    └── SolarMeter_5.0_Spectral_response.csv # wavelength (nm) vs response (%)

Outputs:
    uva_calibrated.csv
    uvb_calibrated.csv
in the same folder.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# 0 ── Paths
# ---------------------------------------------------------------------
DATA_DIR = Path(
    "/Users/mathewyoussef/Desktop/extremophile_detection/Data/July_4_2025/Calibration_files"
)

f_uva_raw   = DATA_DIR / "UVA_Readings__Raw_.csv"
f_uvb_raw   = DATA_DIR / "UVB_Readings__Raw_.csv"
f_response  = DATA_DIR / "SolarMeter_5.0_Spectral_response.csv"
f_uva_out   = DATA_DIR / "uva_calibrated.csv"
f_uvb_out   = DATA_DIR / "uvb_calibrated.csv"

# ---------------------------------------------------------------------
# 1 ── Load raw data
# ---------------------------------------------------------------------
uva_df = pd.read_csv(f_uva_raw)
uvb_df = pd.read_csv(f_uvb_raw)
resp_df = pd.read_csv(
    f_response,
    comment="#",
    header=None,
    names=["wavelength_nm", "response_pct"],
)

# ---------------------------------------------------------------------
# 2 ── Response-curve helper
# ---------------------------------------------------------------------
def interp_response(wavelength_nm: float) -> float:
    """Linear-interpolate Solarmeter %-response at a wavelength (nm)."""
    return np.interp(wavelength_nm, resp_df.wavelength_nm, resp_df.response_pct)

resp_310 = interp_response(310.0)
resp_370 = interp_response(370.0)

factor_uva = 100.0 / resp_370           # ≈ 1.0  (Solarmeter fully sensitive at 370 nm)
factor_uvb = 100.0 / resp_310           # boosts UVB readings to match UVA scale

# ---------------------------------------------------------------------
# 3 ── Geometry helpers
# ---------------------------------------------------------------------
UVA_HEIGHT_MM = 150.0
UVB_HEIGHT_MM = 215.0

def add_geometry(df: pd.DataFrame, lamp_height_mm: float) -> pd.DataFrame:
    r = df["distance_mm"].astype(float)
    h = lamp_height_mm
    out = df.copy()
    out["lamp_height_mm"] = h
    out["theta_deg"] = np.degrees(np.arctan(r / h))
    out["hyp_mm"] = np.sqrt(h**2 + r**2)
    return out

def calibrate(df, corr_factor, lamp_height_mm, band):
    df = add_geometry(df, lamp_height_mm)
    df["irr_corr_mW_cm2"] = df["irradiance_mW_cm2"] * corr_factor
    df["lower_mW_cm2"]    = df["irr_corr_mW_cm2"] * 0.95
    df["upper_mW_cm2"]    = df["irr_corr_mW_cm2"] * 1.05
    df["waveband"]        = band
    return df

uva = calibrate(uva_df, factor_uva, UVA_HEIGHT_MM, "UVA")
uvb = calibrate(uvb_df, factor_uvb, UVB_HEIGHT_MM, "UVB")

# ---------------------------------------------------------------------
# 4 ── Average duplicate readings per grid
# ---------------------------------------------------------------------
uva_cal = (
    uva.groupby("grid", as_index=False)
        .agg({
            "distance_mm":       "mean",
            "theta_deg":         "mean",
            "hyp_mm":            "mean",
            "irr_corr_mW_cm2":   "mean",
            "lower_mW_cm2":      "mean",
            "upper_mW_cm2":      "mean",
        })
)

uvb_cal = (
    uvb.groupby(["grid", "time_min"], as_index=False)
        .agg({
            "distance_mm":       "mean",
            "theta_deg":         "mean",
            "hyp_mm":            "mean",
            "irr_corr_mW_cm2":   "mean",
            "lower_mW_cm2":      "mean",
            "upper_mW_cm2":      "mean",
        })
)

# ---------------------------------------------------------------------
# 5 ── Trapezoid dose for UVB (0–60 min)
# ---------------------------------------------------------------------
dose_rows = []
for grid, sub in uvb_cal.groupby("grid"):
    x_sec = sub["time_min"].values.astype(float) * 60          # convert to seconds
    y     = sub["irr_corr_mW_cm2"].values
    order = np.argsort(x_sec)
    dose_mJ = np.trapz(y[order], x_sec[order])                 # mW · s → mJ
    dose_rows.append({
        "grid": grid,
        "dose_mJ_cm2_t0_60": dose_mJ,
        "avg_mW_cm2_t0_60":  dose_mJ / 3600.0,
    })

dose_df = pd.DataFrame(dose_rows)

# attach summary to uvb_cal
uvb_cal = uvb_cal.merge(dose_df, on="grid", how="left")

# add a summary "ALL" row per grid
# ---------------------------------------------------------------------
# Append a summary 'ALL' row per grid in UVB calibrated
# ---------------------------------------------------------------------
summary_rows = []
for _, r in dose_df.iterrows():          # iterate over the DataFrame, cleaner
    summary_rows.append({
        "grid":           int(r["grid"]),
        "time_min":       "ALL",
        "distance_mm":    np.nan,
        "theta_deg":      np.nan,
        "hyp_mm":         np.nan,
        "irr_corr_mW_cm2": np.nan,
        "lower_mW_cm2":    np.nan,
        "upper_mW_cm2":    np.nan,
        "dose_mJ_cm2_t0_60": r["dose_mJ_cm2_t0_60"],
        "avg_mW_cm2_t0_60":  r["avg_mW_cm2_t0_60"],
    })

uvb_cal = pd.concat([uvb_cal, pd.DataFrame(summary_rows)], ignore_index=True)

# ---------------------------------------------------------------------
# 6 ── Save outputs
# ---------------------------------------------------------------------
uva_cal.to_csv(f_uva_out, index=False)
uvb_cal.to_csv(f_uvb_out, index=False)

print("Calibration complete ✔")
print(f"  • UVA → {f_uva_out}")
print(f"  • UVB → {f_uvb_out}")