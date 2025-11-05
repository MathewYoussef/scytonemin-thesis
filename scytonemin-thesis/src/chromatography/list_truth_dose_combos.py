#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    truth = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")[
        ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    ]
    combos = (
        truth.dropna(subset=["p_uva_mw_cm2", "p_uvb_mw_cm2"])
        .drop_duplicates()
        .sort_values(["p_uva_mw_cm2", "p_uvb_mw_cm2"])
    )
    print("Unique truth combos:", len(combos))
    print(combos.to_string(index=False))


if __name__ == "__main__":
    main()
