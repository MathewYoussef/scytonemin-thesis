#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

FORMS = ["total", "oxidized", "reduced"]


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    base = Path("Diode_Array_Derived_Calibration_Plots")
    alt = Path("Diode_Array_Derived_Calibration_Plots_no_blank")
    print("form,scenario,slope,intercept,r_squared,max_abs_rel_residual")
    for form in FORMS:
        for label, directory in (("blank_sub", base), ("raw", alt)):
            data = load(directory / f"calibration_{form}.json")
            slope = data.get("slope", float("nan"))
            intercept = data.get("intercept", float("nan"))
            r2 = data.get("r_squared", float("nan"))
            max_resid = data.get("max_abs_rel_residual", float("nan"))
            print(f"{form},{label},{slope:.6e},{intercept:.6e},{r2:.4f},{max_resid:.3f}")


if __name__ == "__main__":
    main()
