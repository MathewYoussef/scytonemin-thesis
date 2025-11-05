#!/usr/bin/env python3
# ======================================================================
#  SMCF Pipeline for Side-Mounted UV Lamps
#
#  WHAT IT DOES  -------------------------------------------------------
#  1. Loads the Solarmeter 5.0 spectral responsivity curve that you
#     digitised with WebPlotDigitizer.
#  2. Scans a directory for lamp-spectrum CSV files (each captured with
#     your spectroradiometer at one geometry: distance, x-y grid node,
#     or warm-up timestamp).
#  3. For every spectrum it calculates a Spectral-Mismatch Correction
#     Factor (SMCF):
#
#          SMCF =  ∫ Eλ dλ           (true broadband UV)
#                ------------------  ----------------------------------
#                 ∫ Eλ · Sλ dλ       (what the Solarmeter “sees”)
#
#     where
#        Eλ = lamp spectral irradiance   [W m⁻² nm⁻¹   or counts]
#        Sλ = Solarmeter relative response (peak = 1.0)
#
#  4. Saves a lookup table  smcf_lookup.csv  (one row per spectrum).
#  5. *Optional, later:*   If a meter_readings.csv file exists
#     (distance_cm, raw_mW_cm2) it multiplies each reading by the
#     matching SMCF →  meter_corrected.csv
#
#  HOW TO USE   --------------------------------------------------------
#  $ python smcf_pipeline.py \
#        --resp   data/solarmeter5_resp.csv \
#        --specs  data/spectra/ \
#        --meter  data/meter_readings.csv   # <- optional, can omit
#
#  FILE NAMING CONVENTION (edit to taste) ------------------------------
#      lamp_<TYPE>_d<DIST>cm_x<X>_y<Y>.csv      # spectrum files
#      meter_readings.csv                       # Solarmeter file
#
#  Each spectrum CSV must have two columns, *without* headers:
#      wavelength_nm ,  irradiance
#
#  All calculations default to the biologically relevant 280–400 nm
#  window.  Edit WAVEL_MIN / WAVEL_MAX below if you need UV-C or PAR.
#
#  Author:  <your name>, 2025-07-05
# ======================================================================

import argparse, re, pathlib, sys
from math import isnan
import numpy  as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate   import trapz

# ----------------------------------------------------------------------
# User-tweakable constants
# ----------------------------------------------------------------------
WAVEL_MIN = 280      # nm  (lower limit of interest)
WAVEL_MAX = 400      # nm  (upper limit of interest)
RESP_PEAK_NORMALISE = True   # True → divide Sλ by its max so peak = 1

# Regex pattern to pull metadata out of a spectrum filename
#   lamp_UVA_d25cm_x200_y050.csv  → type=UVA , distance=25 , x=200 , y=50
PAT_SPECTRUM = re.compile(
    r"lamp_(?P<type>[A-Za-z0-9]+)"
    r"_d(?P<dist>\d+)cm"
    r"(?:_x(?P<x>\d+)_y(?P<y>\d+))?"
    r".csv$"
)

# ----------------------------------------------------------------------
def load_responsivity(path: pathlib.Path) -> pd.DataFrame:
    """Load Solarmeter responsivity CSV -> DataFrame(wl_nm, Sλ)."""
    df = pd.read_csv(path, names=["wl_nm", "S"])
    if RESP_PEAK_NORMALISE:
        df["S"] = df["S"] / df["S"].max()
    return df

# ----------------------------------------------------------------------
def smcf_for_spectrum(spec_df: pd.DataFrame,
                      resp_interp: interp1d) -> float:
    """
    spec_df : DataFrame(wl_nm, Eλ)
    resp_interp : interpolation object giving Sλ at any λ
    returns : SMCF (unitless)
    """
    wl  = spec_df.iloc[:,0].to_numpy()
    E   = spec_df.iloc[:,1].to_numpy()
    S   = resp_interp(wl)            # match Solarmeter response to spec grid

    mask = (wl >= WAVEL_MIN) & (wl <= WAVEL_MAX)
    wl, E, S = wl[mask], E[mask], S[mask]

    numerator   = trapz(E,      wl)        # ∫Eλ dλ
    denominator = trapz(E * S, wl)        # ∫Eλ Sλ dλ
    return numerator / denominator if denominator else np.nan

# ----------------------------------------------------------------------
def process_directory(spec_dir: pathlib.Path,
                      resp_df : pd.DataFrame) -> pd.DataFrame:
    """Loop over all spectra, compute SMCF, return tidy lookup table."""
    resp_interp = interp1d(resp_df.wl_nm, resp_df.S,
                           bounds_error=False, fill_value=0.0)

    rows = []
    for csv_path in sorted(spec_dir.glob("*.csv")):
        spec_df = pd.read_csv(csv_path, header=None)
        smcf    = smcf_for_spectrum(spec_df, resp_interp)

        meta = PAT_SPECTRUM.search(csv_path.name)
        if meta:
            rows.append({
                "file"        : csv_path.name,
                "lamp_type"   : meta["type"],
                "distance_cm" : int(meta["dist"]),
                "x_mm"        : int(meta["x"]) if meta["x"] else np.nan,
                "y_mm"        : int(meta["y"]) if meta["y"] else np.nan,
                "smcf"        : smcf
            })
        else:
            print(f"[WARN] filename not recognised: {csv_path.name}",
                  file=sys.stderr)

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
def apply_to_meter(meter_path: pathlib.Path,
                   smcf_df   : pd.DataFrame,
                   out_path  : pathlib.Path):
    """Merge Solarmeter readings with SMCF and write corrected CSV."""
    meter = pd.read_csv(meter_path)   # expects: distance_cm, raw_mW_cm2
    merged = meter.merge(
        smcf_df[["distance_cm", "smcf"]].drop_duplicates("distance_cm"),
        on="distance_cm", how="left"
    )
    merged["corr_mW_cm2"] = merged.raw_mW_cm2 * merged.smcf
    merged.to_csv(out_path, index=False)
    print(f"[OK] Corrected meter file  →  {out_path}")

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute Spectral-Mismatch Correction Factors "
                    "for Solarmeter-5.0 and (optionally) apply them "
                    "to raw handheld readings."
    )
    ap.add_argument("--resp",   required=True,
        help="CSV of Solarmeter responsivity (wl_nm,Sλ)")
    ap.add_argument("--specs",  required=True,
        help="Directory of lamp spectrum CSV files")
    ap.add_argument("--meter",  required=False,
        help="(Optional) CSV of raw Solarmeter readings")
    args = ap.parse_args()

    resp_df  = load_responsivity(pathlib.Path(args.resp))
    smcf_df  = process_directory(pathlib.Path(args.specs), resp_df)
    smcf_df.to_csv("smcf_lookup.csv", index=False)
    print(f"[OK] SMCF lookup table     →  smcf_lookup.csv")

    if args.meter:
        apply_to_meter(pathlib.Path(args.meter),
                       smcf_df,
                       pathlib.Path("meter_corrected.csv"))
    else:
        print("[INFO] No meter file supplied yet; "
              "run again with --meter once you have it.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()