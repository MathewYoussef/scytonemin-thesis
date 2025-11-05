# Supplements — Raw Assets

Canonical source files supporting the thesis supplements. Each file below retains the original filename from the lab handoff; checksums are listed in `CHECKSUMS.sha256`.

## UV Chamber & Meter Calibration
- `July_26_2025_DOSE_SCHEDULE_UPDATED/Calibration_files/` — raw UVA/UVB meter logs, NOAA benchmark pulls, lambda ranking scripts, and conversion helpers:
  - `UVA_Readings__Raw_.csv`, `UVB_Readings__Raw_.csv` — raw SolarMeter readings.
  - `solarmeter_to_dose.py`, `dosage_calendar.py` — scripts that turn raw logs into daily dose schedules.
  - `chamber_dose_schedule.csv`, `MDV_24h_average_power.csv`, `chamber_vs_MDV.csv` — exports used to cross-check the light columns.
  - `MCM_v33.2_DB3_*.csv`, `MCM_UVA_UVB_ranking.py` — NOAA Arrival Heights reference data + ranking utility.
- `July_26_2025_DOSE_SCHEDULE_UPDATED/scripts/UV_energy_with_correction_factor.py` — exploratory notebook/script for energy corrections.

## Procurement & Instrument Documentation
- `Diffuser_plate_6_percent_thesis_2025.pdf` — certificate for the 6 % diffuse reflectance standard.
- `500x500mm, Wavelength 250-800nm Reflectivity 6%--Test Data(2025-07-10 02_33_18).xlsx` — factory test data for the diffuser plate.
- `HPLC Section.pdf` — method excerpt describing chromatographic configuration.
- `Par schedule-4.pdf` — planning sheet for chamber positioning.
- `UV_dose_Calcs-2.pdf` — hand calculations backing the UV duty cycle selected in the experiment.
- `sorurce of truth for methods I did notes.pdf` — researcher notes summarising deviations and procedural decisions.
- `Appendix_M_Extended_Material.textClipping` — clipboard text dump with references for supplementary write-up (retain for provenance).

## Media
- Raw JPEGs were migrated to `scaffold/supplements/assets/images/` so they can be curated into the MkDocs site. The original filenames are preserved.

Use these inputs with the forthcoming `src/supplements` pipeline or bespoke notebooks to regenerate derived tables in `data/reference/supplements/` and figures in `scaffold/supplements/`.
