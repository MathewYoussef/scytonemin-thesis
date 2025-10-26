# Initial Calibration Block

This block documents the chromatogram → DAD → concentration → dry weight pipeline that underpins the thesis calibration narrative. Legacy assets reside in `Initial_Calibration/` and include raw diode-array exports, derived concentration tables, and exploratory notebooks.

## Narrative Scope

- Describe the wet-lab schedule, calibration fits, and correction factors applied to convert instrument readings into concentrations.
- Summarise validation steps (blank removal, occupancy models) and how they propagate to the reflectance analyses.
- Provide links to supplementary diagnostics (e.g., raw chromatograms, QC plots).

## Integration Points

- **Figures:** `scaffold/initial_calibration/figures/` for chromatogram thumbnails, residual plots, and calibration curves.
- **Tables:** `scaffold/initial_calibration/tables/` for coefficient summaries, uncertainty budgets, and intermediate datasets.
- **Notebooks:** `scaffold/initial_calibration/notebooks/` to hold executed notebooks or scripts producing the above artifacts.
- **Media:** Cross-link to Agent 4 derivatives once placed in `docs/media/`.

## Thesis Integration Points

Reference these sections in `thesis.docx.md` when populating this page:

- `## Materials and Methods` → `### Pigment extraction for UPLC`
- `## Materials and Methods` → `### UPLC–DAD–MS`
- `## Materials and Methods` → `### Robust Mean Concentration and Dose`
- `## Results` → `### Chromatographic quantification and matrix effects (HPLC‑DAD/PDA)`
- `## Discussion` → `### Chlorophyll and β‑carotene were not quantified (implications for ratios)`
- `## Discussion` → `### Reconciling reflectance and concentration: the “inversion”`

## Thesis Excerpts (draft slots)

> _Source: `thesis.docx.md`; cite line ranges for audit traceability._

### Pigment extraction for UPLC (lines 222–244)

> “Post-exposure thalli were frozen, lyophilised, and stored at −80 °C before extraction. Native scytonemin was recovered with a 1:1 methanol:ethyl acetate solvent (0.1 % acetic acid), executed in three low-temperature sonication + rocking passes to maximize yield while avoiding acetone-driven scytonemin-imine conversion.”

### UPLC–DAD–MS (lines 246–276)

> “Analyses used a Waters ACQUITY UPLC with PDA and Xevo TQ-MS, operating a methanol/acetonitrile gradient at 0.35 mL min⁻¹ (35 °C column, 4 °C autosampler). PDA data were captured from 190–500 nm, while targeted SIM/MRM channels monitored reduced and oxidized scytonemin alongside MV-chlorophyll a transitions.”

### Robust mean concentration & dose (lines 310–325)

> “Robust (trimmed) means summarized each dose with 95 % CIs. Chromatogram totals peaked near 0.99 mg·gDW⁻¹ at dose₄, while reduced pools rose to 1.79 mg·gDW⁻¹ at dose₅; DAD totals mirrored the dose₄ crest, reinforcing dose-dependent induction despite small n.”

### Chromatographic matrix effects (lines 422–430)

> “HPLC-DAD quantitation in the UVA–blue region is vulnerable to matrix overlap from MAAs, carotenoids, and pheophytins. Remedies include matrix-matched calibration, scytonemin-null blanks, multi-wavelength quantitation, and chemometric deconvolution to maintain fidelity near the lower range.”

### Chlorophyll & β-carotene gaps (lines 431–434)

> “The scytonemin-focused extraction and gradient precluded reliable chlorophyll a or β-carotene quantitation; without pigment-specific gradients and standards, values would be biased. Future iterations should run a parallel pigment pipeline to recover those ratios.”

### Reflectance vs concentration inversion (lines 445–454)

> “Σ bowl-occupancy dropped from 0.0703 (dose₂) to 0.0257 (dose₄) while Chrom_total climbed to 0.986 mg·gDW⁻¹, illustrating an inversion between spectral depth and analyte abundance. Heterogeneous scattering, moisture shifts, and co-absorbers can flatten band depth even as scytonemin rises, so reflectance must be treated as a multicomponent fingerprint.”

## Pending Inputs

- Agent 3 to register raw → processed dataset lineage in `ops/output/data/catalog.csv`.
- Agent 2 to import thesis excerpts covering calibration methodology from `thesis.docx.md`.
- Agent 4 to embed setup photography (e.g., chromatograph images) once derivatives are ready.
