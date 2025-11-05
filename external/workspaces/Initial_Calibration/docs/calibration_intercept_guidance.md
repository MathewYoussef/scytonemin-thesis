# Calibration Intercept Strategy

## Short Answer
Keep the intercept unless your blanks and low-end standards prove it is statistically indistinguishable from zero and the through-origin fit demonstrably improves accuracy across the calibration range. Regulatory guidance (EPA SW-846 8000D, ICH Q2(R2), ICH M10, ISO 8466/11095, Eurachem) all emphasise reporting slope *and* intercept and selecting the regression model on the basis of data.

## Why We Do Not Force Zero
- **Background signal exists.** Even after blank subtraction, 320–480 nm diode-array spectra include solvent/mobile-phase absorbance, flow-cell film, lamp drift, and reference-channel offsets. These contributions are captured by the intercept. Agilent DAD documentation and peak-purity literature discuss managing, not eliminating, such baselines.
- **Matrix co-absorbers.** Pigments and other chromophores contribute absorbance shoulders near 384 nm; zero scytonemin does not guarantee zero AUC.
- **Carryover/integration bias.** Low-level chromatographic integrations rarely land exactly on zero; the intercept acts as the calculated blank (ISO 8466 terminology).

## Evidence from Our Data
- **Chromatogram calibration (Stage A)**: Free-intercept fits give R² ≈ 0.99 and max |relative residual| ≤ 0.13. Forcing zero drops R² to 0.97–0.98 and inflates residuals above 0.6. JSON outputs (`DAD_to_Concentration_AUC/calibration_{form}.json`) explicitly record `intercept_mode: "free"` with intercepts around −0.016 mg/mL.
- **Diode-array calibration**: With blank subtraction and free intercept we obtain slopes (×10⁻⁸) of 4.526/3.007/4.711 for total/oxidized/reduced, intercepts near −0.02 to −0.03 mg/mL, R² ≈ 1.0, and residuals within 0.47. Forced-origin fits showed much larger residuals (≥0.74) and poorer R².
- **Blanks**: Three diode-array blanks exhibit substantial scatter (σ ≈ 4.66×10⁴ AU·nm); the intercept absorbs this residual background. Subtracting the minimum blank before fitting reduces—but does not eliminate—the offset.

## Decision Rule Implemented
1. Fit two models: free intercept and forced origin (weighted 1/x or 1/x² as configured).
2. Compare back-calculated accuracy (%RE), residual plots, and QC thresholds (R², max |relative residual|).
3. Keep the intercept unless the origin-constrained model clearly improves all QC statistics and blanks confirm zero signal within uncertainty.
4. Document slope, intercept, SEs, and LOD/LOQ derived from blanks. (Matches EPA 8000D §11, ICH Q2(R2) linearity guidance, ISO 8466 calculated blank discussion.)

## References (non-exhaustive)
- **EPA SW-846 8000D** — warns against forcing calibration through the origin without evidence; discusses weighting and blank verification.
- **ICH Q2(R2) (2023)** and **ICH M10 (2022)** — require slope/intercept reporting, weighted regression rationale, residual analysis across the range.
- **ISO 8466 / ISO 11095** — linear calibration with intercept as calculated blank.
- **Eurachem Guides (Fitness for Purpose, Blanks supplement, 2025)** — practical treatment of blanks and regression selection.
- **Dolan (LCGC) column series** — chromatography practice for intercept and weighting decisions.

Adhering to this strategy keeps the calibration statistically defensible and consistent with regulatory expectations for LC/UPLC detectors, including diode-array systems.
