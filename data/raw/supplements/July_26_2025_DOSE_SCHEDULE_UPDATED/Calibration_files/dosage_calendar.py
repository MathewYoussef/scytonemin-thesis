#!/usr/bin/env python3
"""
Create a daily UV-lamp schedule table with explicit units in every column.

• UVA lamp time  = UVA_HOURS (default 24 h).
• UVB lamp time  = UVB_HOURS_OVERRIDE
                   – OR – auto-computed so GRID 3 delivers the MDV 24-h UVB dose.

Output
------
chamber_dose_schedule.csv   columns & units:
    grid (#)
    UVA_hours (h)
    UVB_hours (h)
    P_UVA (mW cm^-2)
    P_UVB (mW cm^-2)
    MDV_UVA_24h_power (mW cm^-2)
    MDV_UVB_24h_power (mW cm^-2)
    %MDV_UVA_dose (%)
    %MDV_UVB_dose (%)
"""

from pathlib import Path
import pandas as pd

# ───────── USER SETTINGS ───────────────────────────────────────────────
UVA_HOURS          = 24.0     # LED run-time in hours
UVB_HOURS_OVERRIDE = None     # float to force Hg-lamp hours, or None for auto
REFERENCE_GRID     = 3        # grid used to match UVB dose
# ───────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent
field_path = DATA_DIR / "MDV_24h_average_power.csv"
grid_path  = DATA_DIR / "chamber_vs_MDV_24h.csv"

# 1 ── Load & validate -----------------------------------------------------------
for p in (field_path, grid_path):
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

field = pd.read_csv(field_path).iloc[0]
MDV_UVA = float(field["UVA_mW_cm2"])
MDV_UVB = float(field["UVB_mW_cm2"])

ch = pd.read_csv(grid_path)

# harmonise possible column names
rename = {}
if "P_chamber_UVA" in ch.columns: rename["P_chamber_UVA"] = "P_UVA"
if "P_chamber_UVB" in ch.columns: rename["P_chamber_UVB"] = "P_UVB"
ch = ch.rename(columns=rename)

for col in ("grid", "P_UVA", "P_UVB"):
    if col not in ch.columns:
        raise KeyError(f"Column '{col}' missing in {grid_path.name}")

# drop duplicate grid rows
ch = ch.drop_duplicates("grid", keep="first")

# 2 ── Reference grid & UVB lamp time -------------------------------------------
ref_row = ch.loc[ch["grid"] == REFERENCE_GRID]
if ref_row.empty:
    raise ValueError(f"Reference grid {REFERENCE_GRID} not found.")
P_ref_UVB = float(ref_row.iloc[0]["P_UVB"])

if UVB_HOURS_OVERRIDE is None:
    UVB_hours = 24.0 * MDV_UVB / P_ref_UVB      # exact, no rounding
else:
    UVB_hours = float(UVB_HOURS_OVERRIDE)

UVA_hours = float(UVA_HOURS)

# 3 ── Dose calculations ---------------------------------------------------------
SEC_PER_H = 3600.0
MDV_UVA_day_mJ = MDV_UVA * 24 * SEC_PER_H
MDV_UVB_day_mJ = MDV_UVB * 24 * SEC_PER_H

ch["UVA_hours (h)"] = UVA_hours
ch["UVB_hours (h)"] = UVB_hours
ch["P_UVA (mW cm^-2)"] = ch["P_UVA"]
ch["P_UVB (mW cm^-2)"] = ch["P_UVB"]
ch["dose_UVA_mJ_cm2"] = ch["P_UVA"] * SEC_PER_H * UVA_hours
ch["dose_UVB_mJ_cm2"] = ch["P_UVB"] * SEC_PER_H * UVB_hours
ch["%MDV_UVA_dose (%)"] = 100.0 * ch["dose_UVA_mJ_cm2"] / MDV_UVA_day_mJ
ch["%MDV_UVB_dose (%)"] = 100.0 * ch["dose_UVB_mJ_cm2"] / MDV_UVB_day_mJ
ch["MDV_UVA_24h_power (mW cm^-2)"] = MDV_UVA
ch["MDV_UVB_24h_power (mW cm^-2)"] = MDV_UVB
ch.rename(columns={"grid": "grid (#)"}, inplace=True)

out_cols = [
    "grid (#)",
    "UVA_hours (h)", "UVB_hours (h)",
    "P_UVA (mW cm^-2)", "P_UVB (mW cm^-2)",
    "MDV_UVA_24h_power (mW cm^-2)", "MDV_UVB_24h_power (mW cm^-2)",
    "%MDV_UVA_dose (%)", "%MDV_UVB_dose (%)",
]

out_path = DATA_DIR / "chamber_dose_schedule.csv"
ch[out_cols].sort_values("grid (#)").to_csv(out_path, index=False)

# 4 ── Console summary -----------------------------------------------------------
print("✔ chamber_dose_schedule.csv written →", out_path)
print(f"  LED (UVA)     : {UVA_hours} h per lab day")
print(f"  Hg-lamp (UVB) : {UVB_hours} h per lab day")
print(f"  Reference grid = {REFERENCE_GRID}")