# Reflectance Block

This block curates the denoising, spectral analysis, and downstream comparisons for the field reflectance campaign. Source code currently lives under `Reflectance/`, with NumPy staging data and canonical tables identified by Agent 1. Migration into the scaffold will follow that proposal.

## Narrative Scope

- Summarise acquisition setup, calibration frames, and denoising pipeline (Mamba SSM handoff occurs in its own section).
- Present per-treatment spectra, SNR improvements, and quality gates referenced in the thesis discussion.
- Link to supplementary orientation scans that live outside the mainline stats.

## Integration Points

- **Figures:** `scaffold/reflectance/figures/` for spectra panels, SNR heatmaps, and QC plots.
- **Tables:** `scaffold/reflectance/tables/` for numeric summaries (e.g., ΔSNR by treatment).
- **Notebooks:** `scaffold/reflectance/notebooks/` receives executed notebooks exporting the above.
- **Media:** Agent 4 will drop photos/videos into `scaffold/reflectance/assets/` before promoting to `docs/media/`.

## Thesis Integration Points

Link the following thesis sections (from `thesis.docx.md`) into this page once excerpts are prepared:

- `## Materials and Methods` → `### Reflectance spectroscopy`
- `## Materials and Methods` → `### Computation, normalization, & denoising`
- `## Results` → `### Reflectance Spectra (Dose and Concentration)`
- `## Discussion` → `### Instrument‑specific influences (sensor, angle, and geometry)`
- `## Discussion` → `### Noise in the data, temperature control, and light source`
- `## Discussion` → `### Redox forms and spectral fingerprints`

## Thesis Excerpts (draft slots)

> _Source: `thesis.docx.md` (line references noted for traceability). Replace or trim as narrative solidifies._

### Reflectance spectroscopy (lines 131–144)

> “Spectra were collected in a darkened laboratory using an Ocean Insight ST-UV spectrometer (nominal 190–650 nm, 25 µm slit) with the fiber 4 cm from the 5 × 5 cm panel, clamped 30° off nadir to a xenon arc lamp (ABET LS-150-Xe). Because the unit was uncalibrated for absolute irradiance, we report relative reflectance factors and focus on spectral shape, magnitude, and curvature rather than absolute radiance.”

### Acquisition cadence (lines 141–146)

> “Integration time was fixed at 665 370 µs for all scans. Each thallus was sampled in four orientations with 150 spectra at 1 s cadence; 12 o’clock and 6 o’clock views anchor the main analysis, while the remaining orientations are retained as supplementary.”

### Computation, normalization, & denoising (lines 147–167)

> “After ratio-to-reference and deterministic normalization, spectra behave as a slowly varying continuum plus narrow pigment-linked dips corrupted by high-frequency sensor noise. The denoiser must suppress high-frequency fluctuations without distorting band morphology in the 320–480 nm ROI where pigment features reside.”

### Reflectance spectra: dose & concentration (lines 342–356)

> “Mamba-SSM denoised spectra were compared across two wavelength windows (320–480 nm; 360–410 nm). Continuum removal standardized shoulders across doses, allowing absorption valleys to be expressed as positive ‘bowl occupancy’ features and making dose trends directly comparable.”

### Instrument-specific influences (lines 412–417)

> “Reflectance depends on illumination and viewing geometry; variation in fiber angle, foreoptic height, or microtopography injects variance even at fixed pigment content. Off-nadir capture with textured thalli confounded geometry and specular reflections, motivating future nadir-stabilized setups.”

### Noise, temperature, and light source (lines 418–421)

> “Baseline noise and drift rise when detector or flow-cell temperatures fluctuate—common in broadband lamps versus UV LEDs. LED-driven UVA/UVB with radiometric protocols offer more stable, line-centered fluence and lower spectral uncertainty.”

### Redox forms & fingerprints (lines 435–438)

> “Oxidized and reduced scytonemin exhibit distinct spectral shapes and HPLC behavior; the oxidized form tracked a clean quadratic dose trend, reinforcing the need to report both redox states and treat reflectance as a multi-wavelength fingerprint rather than a single-band proxy.”

## Pending Inputs

- Agent 3 to catalogue datasets (`data/reflectance/**`) and expose sample subsets for `make quickstart`.
- Agent 2 to backfill prose from `thesis.docx.md` once available.
- Agent 4 to embed media gallery after the data harvest establishes canonical filenames.
