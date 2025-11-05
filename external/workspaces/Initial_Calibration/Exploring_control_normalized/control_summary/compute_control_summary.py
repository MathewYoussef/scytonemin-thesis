#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Scripts.control_normalization_utils import (
    load_chromatogram,
    load_dad,
    compute_control_baselines,
    save_baselines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute control baseline statistics for chromatogram and DAD datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis_config.yaml"),
        help="Path to analysis_config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def baselines_to_frames(baselines: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chrom_records = []
    for form, metrics in baselines["chromatogram"].items():
        for metric, stats in metrics.items():
            chrom_records.append(
                {
                    "form": form,
                    "metric": metric,
                    **stats,
                }
            )
    chrom_df = pd.DataFrame.from_records(chrom_records)

    dad_records = []
    for col, stats in baselines["dad"].items():
        dad_records.append({"column": col, **stats})
    dad_df = pd.DataFrame.from_records(dad_records)
    return chrom_df, dad_df


def main() -> None:
    args = parse_args()
    repo_root = args.config.resolve().parent

    chrom_df = load_chromatogram(repo_root)
    dad_df = load_dad(repo_root)
    baselines = compute_control_baselines(chrom_df, dad_df)

    out_dir = repo_root / "Exploring_control_normalized" / "control_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_json = out_dir / "control_baselines.json"
    save_baselines(baselines, baseline_json)

    chrom_summary, dad_summary = baselines_to_frames(baselines)
    chrom_summary.to_csv(out_dir / "chromatogram_control_summary.csv", index=False)
    dad_summary.to_csv(out_dir / "dad_control_summary.csv", index=False)

    print(f"Control baselines saved to {baseline_json}")
    print("Chromatogram summary rows:", len(chrom_summary))
    print("DAD summary rows:", len(dad_summary))


if __name__ == "__main__":
    main()
