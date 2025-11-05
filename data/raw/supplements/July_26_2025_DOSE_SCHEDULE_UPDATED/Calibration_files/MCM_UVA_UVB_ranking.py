#!/usr/bin/env python3
"""
Rank McMurdo’s ten strongest summer UV-B days (15 Dec → 15 Jan, 2023-24),
compute both peak-hour and 24-hour-average irradiance, and compare them
to your chamber grids.

Outputs
-------
1. MDV_top10_day_stats.csv      – daily UVA/UVB dose + daylight hours
2. MDV_grand_profile.csv        – 24-row hourly UVA/UVB power (grand average)
3. MDV_24h_average_power.csv    – single-row mean power over 24 h
4. chamber_vs_MDV.csv           – factors based on *peak* hour (as before)
5. chamber_vs_MDV_24h.csv       – factors based on 24-h average power
"""

import pandas as pd
from pathlib import Path

# ── 1. configuration ────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
DB3_FILE = DATA_DIR / "MCM_v33.2_DB3_meas_new2.csv"
SUMMER_A, SUMMER_B = "12-15", "01-15"          # mm-dd window
TOP_N = 10                                     # brightest days to keep

# ── 2. load DB3 & convert to mW cm⁻² ────────────────────────────────────
use = ["Time start scan", "E290-315", "E315-400", "Flags"]
df = pd.read_csv(DB3_FILE, header=0, skiprows=[1], usecols=use,
                 dtype={"Time start scan": "float64",
                        "E290-315": "float64",
                        "E315-400": "float64",
                        "Flags": "object"},
                 comment="#", na_values=["", " "], low_memory=False)

df = df[df["Flags"].isna()]                         # good scans only
epoch = pd.Timestamp("1900-01-01", tz="UTC")
df["ts"] = epoch + pd.to_timedelta(df["Time start scan"] * 24, unit="h")
df = df.set_index("ts").sort_index()
df = df[["E290-315", "E315-400"]] / 1000.0          # µW → mW
df.columns = ["UVB_mW_cm2", "UVA_mW_cm2"]

# ── 3. keep peak-summer window ──────────────────────────────────────────
mask = (df.index.strftime("%m-%d") >= SUMMER_A) | \
       (df.index.strftime("%m-%d") <= SUMMER_B)
summer_hours = df.loc[mask]

# ── 4. hourly means & day-level stats ───────────────────────────────────
hourly = summer_hours.resample("60min").mean()

daily = (hourly
         .resample("1D")
         .agg(UVB_dose_mJ_cm2 = ("UVB_mW_cm2", lambda x: (x*3600).sum()),
              UVA_dose_mJ_cm2 = ("UVA_mW_cm2", lambda x: (x*3600).sum()),
              daylight_hours  = ("UVA_mW_cm2", "count")))      # 0-24

top10 = daily.nlargest(TOP_N, "UVB_dose_mJ_cm2")
top10.to_csv(DATA_DIR / "MDV_top10_day_stats.csv", float_format="%.2f")

# ── 5. grand-average hourly profile (10 days) ───────────────────────────
hours_10 = hourly[hourly.index.normalize().isin(top10.index)]

grand = (hours_10
         .groupby(hours_10.index.hour)
         .mean()
         .rename_axis("UTC_hour"))
grand.to_csv(DATA_DIR / "MDV_grand_profile.csv", float_format="%.4f")

# 5a. 24-h mean power (new CSV) -----------------------------------------
avg24 = grand.mean()
avg24.to_frame(name="mW_cm2").transpose()\
      .to_csv(DATA_DIR / "MDV_24h_average_power.csv", float_format="%.4f")

# peak-hour powers for legacy factors
peak_hour = grand["UVB_mW_cm2"].idxmax()
noon_uvb  = grand.loc[peak_hour, "UVB_mW_cm2"]
noon_uva  = grand.loc[peak_hour, "UVA_mW_cm2"]

# 24-h average powers
avg_uva = avg24["UVA_mW_cm2"]
avg_uvb = avg24["UVB_mW_cm2"]

# ── 6. load chamber grid powers ─────────────────────────────────────────
u_ch = pd.read_csv(DATA_DIR / "uva_calibrated.csv")[["grid", "irr_corr_mW_cm2"]]
v_ch = pd.read_csv(DATA_DIR / "uvb_calibrated.csv")[["grid", "avg_mW_cm2_t0_60"]]

ch = (u_ch.merge(v_ch, on="grid")
          .rename(columns={"irr_corr_mW_cm2":  "P_chamber_UVA",
                           "avg_mW_cm2_t0_60": "P_chamber_UVB"}))

# ── 7. peak-hour factors (unchanged) ────────────────────────────────────
peak_tbl = ch.copy()
peak_tbl["P_MDV_UVA"]  = noon_uva
peak_tbl["P_MDV_UVB"]  = noon_uvb
peak_tbl["UVA_factor"] = noon_uva / peak_tbl["P_chamber_UVA"]
peak_tbl["UVB_factor"] = noon_uvb / peak_tbl["P_chamber_UVB"]

peak_cols = ["grid", "P_MDV_UVA","P_MDV_UVB",
             "P_chamber_UVA","P_chamber_UVB",
             "UVA_factor","UVB_factor"]

peak_tbl[peak_cols].sort_values("grid")\
        .to_csv(DATA_DIR / "chamber_vs_MDV.csv",
                index=False, float_format="%.3f")

# ── 8. 24-h-average factors (new) ───────────────────────────────────────
avg_tbl = ch.copy()
avg_tbl["P_MDV_UVA_24h"] = avg_uva
avg_tbl["P_MDV_UVB_24h"] = avg_uvb
avg_tbl["UVA_factor_24h"] = avg_uva / avg_tbl["P_chamber_UVA"]
avg_tbl["UVB_factor_24h"] = avg_uvb / avg_tbl["P_chamber_UVB"]

avg_cols = ["grid",
            "P_MDV_UVA_24h","P_MDV_UVB_24h",
            "P_chamber_UVA","P_chamber_UVB",
            "UVA_factor_24h","UVB_factor_24h"]

avg_tbl[avg_cols].sort_values("grid")\
        .to_csv(DATA_DIR / "chamber_vs_MDV_24h.csv",
                index=False, float_format="%.3f")

# ── 9. console summary ─────────────────────────────────────────────────
print("✔  Outputs written to", DATA_DIR)
print("• MDV_top10_day_stats.csv")
print("• MDV_grand_profile.csv")
print("• MDV_24h_average_power.csv")
print("• chamber_vs_MDV.csv  (peak-hour factors)")
print("• chamber_vs_MDV_24h.csv  (24-h average factors)\n")

print(f"Peak (max-UVB) hour UTC: {peak_hour:02d}:00  "
      f"UVB {noon_uvb:.3f} mW cm⁻²,  UVA {noon_uva:.3f} mW cm⁻²")
print(f"24-h mean power:          UVB {avg_uvb:.3f} mW cm⁻²,  "
      f"UVA {avg_uva:.3f} mW cm⁻²")
print("NOAA SUV-100 entrance optic height: 1.5 m above surface at Arrival Heights")