"""Generate lightweight samples from the reflectance canonical dataset."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "Reflectance" / "reflectance" / "canonical_dataset"
OUT_DIR = Path(__file__).resolve().parent

SAMPLES = {
    "dose_summary.csv": {"rows": 3},
    "precision_weighted_concentrations.csv": {"rows": 12},
    "precision_weighted_concentrations_treatment.csv": {"rows": 6},
    "reflectance_trimmed_stats.csv": {"rows": 40},
}


def downsample_csv(filename: str, rows: int) -> None:
    src = SOURCE_DIR / filename
    dst = OUT_DIR / f"{Path(filename).stem}_sample.csv"
    with src.open("r", newline="") as fh_in, dst.open("w", newline="") as fh_out:
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx >= rows:
                break
            writer.writerow(row)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, cfg in SAMPLES.items():
        downsample_csv(name, cfg["rows"])
    print(f"Wrote samples to {OUT_DIR}")
