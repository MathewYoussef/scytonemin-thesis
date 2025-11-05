"""Create a lightweight sample of the validation panel."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / 'Act_of_God_Mamba_Results' / 'data' / 'validation_panel.csv'
OUT_DIR = Path(__file__).resolve().parent

SAMPLE_ROWS = 32


def downsample() -> None:
    dst = OUT_DIR / 'validation_panel_sample.csv'
    with SOURCE_FILE.open('r', newline='') as fh_in, dst.open('w', newline='') as fh_out:
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx >= SAMPLE_ROWS:
                break
            writer.writerow(row)


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    downsample()
    print(f"Sample panel written to {OUT_DIR}")
