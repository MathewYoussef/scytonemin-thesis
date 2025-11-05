# Supplements Block

Supplementary materials host UV chamber calibration data, procurement records, instrument certificates, and photographic documentation that support claims in the thesis. Everything is pulled from the structured directories below:

- Raw inputs — `data/raw/supplements/`
- Curated references — `data/reference/supplements/`
- Figures/tables/notebooks — `scaffold/supplements/{figures,tables,notebooks}`
- Media assets — `scaffold/supplements/assets/images/`

## Key Assets

- Chamber dose schedule (`data/raw/supplements/July_26_2025_DOSE_SCHEDULE_UPDATED/Calibration_files/`)
  - Raw SolarMeter reads, conversion scripts, and NOAA benchmark comparisons.
  - Use `solarmeter_to_dose.py` and `dosage_calendar.py` to regenerate processed dose tables.
- Instrument & procurement records (`data/reference/supplements/*.pdf`, `*.xlsx`)
  - Diffuser plate certification, HPLC method excerpt, UV power audit spreadsheets, planning notes.
- Photo documentation (`scaffold/supplements/assets/images/`)
  - UV bench setups, Nostoc handling, instrument close-ups.

## Reproduction Checklist

1. Run `notebooks/01_dosimetry_mdv_benchmark.ipynb` to convert raw Solarmeter readings into chamber doses and verify %MDV scaling.
2. Update `data/reference/supplements/CHECKSUMS.sha256` whenever documentation changes.
3. Coordinate with Agent 4 to transform the raw photo set into the public media gallery (`docs/supplements/media.md`).
4. Log any new supplementary assets in `ops/logs/` with provenance hashes.

## Thesis Integration

Map supplementary summaries to these `thesis.docx.md` sections:
- `## Materials and Methods` — chamber layout, UVA/UVB duty cycles.
- `## Discussion` — environmental benchmarking caveats and instrument geometry.
- `## Discussion` — replication considerations informed by photo evidence and lab notes.

## Next Steps

- [ ] Publish processed dose tables and analytical notebooks to `scaffold/supplements/tables/` and `scaffold/supplements/notebooks/`.
- [ ] Draft `docs/supplements/methods.md` and `docs/supplements/media.md` once Agent 4 curates captions.
- [ ] Add a `src/supplements/` module wrapping the calibration scripts for reproducible runs.
