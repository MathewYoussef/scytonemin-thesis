#!/usr/bin/env python3
#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    truth = pd.read_csv(repo / "Compiled_DAD_DATA" / "Scytonemin" / "sample_id_truth.csv")[
        ["sample_id", "p_uva_mw_cm2", "p_uvb_mw_cm2"]
    ]
    chrom = pd.read_csv(repo / "DAD_to_Concentration_AUC" / "treatments_concentration_raw.csv")
    merged = chrom.merge(truth, on="sample_id", how="left")
    combos = (
        merged[["p_uva_mw_cm2", "p_uvb_mw_cm2"]]
        .drop_duplicates()
        .sort_values(["p_uva_mw_cm2", "p_uvb_mw_cm2"])
    )
    print("Unique UVA/UVB combos:", len(combos))
    print(combos.to_string(index=False))


if __name__ == "__main__":
    main()
