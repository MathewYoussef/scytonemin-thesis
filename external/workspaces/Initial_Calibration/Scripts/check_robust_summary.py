#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    for filename in [
        "chromatogram_robust_summary_mg_mL.csv",
        "chromatogram_robust_summary_mg_per_gDW.csv",
        "dad_robust_summary_mg_mL.csv",
        "dad_robust_summary_mg_per_gDW.csv",
    ]:
        path = repo / "_concentrations_vs_dose_with_robust_mean" / filename
        df = pd.read_csv(path)
        counts = df.groupby("form").size()
        print(filename, counts.to_dict())


if __name__ == "__main__":
    main()
