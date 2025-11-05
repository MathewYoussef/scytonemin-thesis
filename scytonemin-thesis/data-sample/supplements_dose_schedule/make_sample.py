"""Produce trimmed calibration samples for supplements dose schedule."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'Supplements' / 'July_26_2025_DOSE_SCHEDULE_UPDATED' / 'Calibration_files'
TARGETS = {
    'chamber_dose_schedule.csv': 48,
    'MDV_24h_average_power.csv': 48,
    'uva_calibrated.csv': 48,
    'uvb_calibrated.csv': 48,
}
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
    for filename, rows in TARGETS.items():
        downsample_csv(filename, rows)
    print(f"Supplement samples written to {OUT_DIR}")
