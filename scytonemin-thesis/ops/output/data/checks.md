# Data Health Checks — Agent 3

## Processed Tables
- Checksums and dimensions logged in `ops/output/data/processed_table_checks.csv` for all processed outputs and sample cuts.
- Reflectance dose summary rows: 6 doses × 5 columns; DAD/Chrom ratio spans **1.784–2.097**.
- Initial Calibration dose response aggregates cover 4 UVB levels with total means **0.664–1.822 mg/g DW**.
- Mamba validation panel counts per treatment/angle range **150–300 spectra** across 2 measured angles.
- Supplements UV dose schedule yields integrated doses between **0.59–275.18 mJ/cm²** for UVA/UVB.

## Schema Spot Checks
- Compared dataset schemas (`data/*/schema.json`) against live files: all primary keys present and column counts unchanged.
- Sample generators under `data-sample/**/make_sample.py` were executed; resulting CSVs verified (<10 MB each).

## Next Actions
- Run `dvc add` on processed tables once remote storage is provisioned.
- Integrate automated CI check to recompute `processed_table_checks.csv` and flag checksum drift.
