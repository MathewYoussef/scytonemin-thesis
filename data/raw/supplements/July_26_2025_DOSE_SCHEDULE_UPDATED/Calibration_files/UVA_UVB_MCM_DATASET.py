import pandas as pd
from pathlib import Path

DATA_DIR  = Path("/Users/mathewyoussef/Desktop/extremophile_detection/Data/July_4_2025/Calibration_files")
DB3_FILE  = DATA_DIR / "MCM_v33.2_DB3_meas_new2.csv"
HOURLY_OUT = DATA_DIR / "MCM_v33.2_DB3_hourly_dose.csv"

use = ["Time start scan", "E290-315", "E315-400", "Flags"]
dtypes = {
    "Time start scan": "float64",
    "E290-315":        "float64",
    "E315-400":        "float64",
    "Flags":           "object",
}

# ── read, skip the units line ───────────────────────────────────────────
df = pd.read_csv(
    DB3_FILE,
    comment="#",
    header=0,          # first line = column names
    skiprows=[1],      # second line = units → drop
    usecols=use,
    dtype=dtypes,
    na_values=["", " "],
    low_memory=False,
)

# ── keep only good scans ────────────────────────────────────────────────
df = df[df["Flags"].isna()].copy()

# ── build timestamp (days since 1900-01-01) ─────────────────────────────
epoch = pd.Timestamp("1900-01-01", tz="UTC")
df["timestamp"] = epoch + pd.to_timedelta(df["Time start scan"] * 24, unit="h")
df = df.set_index("timestamp").sort_index()

# ── µW → mW, then hourly power & dose ───────────────────────────────────
df["UVB_mW_cm2"] = df["E290-315"] / 1000.0
df["UVA_mW_cm2"] = df["E315-400"] / 1000.0

hourly_power = df[["UVB_mW_cm2", "UVA_mW_cm2"]].resample("60min").mean()
hourly_dose  = hourly_power * 3600.0

out = hourly_power.add_suffix("_avg").join(hourly_dose.add_suffix("_dose"))
out.to_csv(HOURLY_OUT, float_format="%.4f")

print("✅ saved →", HOURLY_OUT)
print(out.head())

# show two December clear-sky hours
dec15 = out.loc["2022-12-15 00:00":"2022-12-15 23:59"]

print(dec15.between_time("10:00", "14:00").head())

# grab a midsummer day that *is* in vol 33 (Antarctic summer = Dec 2023)
summer = out.loc["2023-12-15 00:00":"2023-12-16 00:00"]

print(summer.dropna().head())