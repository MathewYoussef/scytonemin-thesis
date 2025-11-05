# Raw Spectrometer Counts (Subset)

Subset of raw Ocean Insight spectrometer exports used to demonstrate the conversion to relative reflectance in the notebooks.

- `Treatment_5/sample_A/Angle_12Oclock/` — first 60 scans for sample A, treatment 5, 12 o’clock orientation.
- `Treatment_5/sample_A/Angle_6Oclock/` — first 60 scans for the same sample at 6 o’clock.
- `Treatment_5/sample_A/WhiteRef_Calibration_1/` — matching white-tile scans (certified 6 % standard) captured immediately before sample measurements.
- `Dark_reference/` — dark-frame scans acquired at the start of the session.

Each `.txt` file is the direct spectrometer output used in the relative reflectance calculation: the notebook reads these counts, subtracts the dark, divides by the averaged white reference, and emits per-scan relative reflectance for downstream denoising.

The full dataset remains under the original acquisition directory (`Desktop/extremophile_detection/Data/Final_Experiment_August_18_2025/…`).
