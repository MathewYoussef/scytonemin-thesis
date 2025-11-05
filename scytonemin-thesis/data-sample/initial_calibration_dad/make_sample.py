"""Downsample Initial Calibration DAD exports for quick validation."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = {
    'DAD_derived_concentrations_corrected.csv': 20,
    'DAD_derived_concentrations.csv': 20,
    'diode_array_auc.csv': 20,
    'treatments_corrected_amounts.csv': 20,
    'sample_id_truth.csv': 20,
}

SOURCE_DIR = ROOT / 'Initial_Calibration'
OUT_DIR = Path(__file__).resolve().parent

def downsample_csv(filename: str, rows: int) -> None:
    src = SOURCE_DIR / filename
    dst = OUT_DIR / f"{Path(filename).stem}_sample.csv"
    with src.open('r', newline='') as fh_in, dst.open('w', newline='') as fh_out:
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx >= rows:
                break
            writer.writerow(row)


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    for filename, rows in SOURCE_FILES.items():
        downsample_csv(filename, rows)
    print(f"Sample CSVs written to {OUT_DIR}")
