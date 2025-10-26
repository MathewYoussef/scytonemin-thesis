# Supplements Block

Supplemental materials capture calibrations, UV dose calculations, instrumentation certificates, and visual documentation that support the core thesis narrative. Source artifacts currently live in `Supplements/` and include spreadsheets, PDFs, and media directories.

## Narrative Scope

- Provide context for each supplementary asset (e.g., diffuser plate certificate, UV mapping schedules).
- Link to processed summaries produced by Agents 3 & 4 (tables, galleries, and downloadable archives).
- Highlight how supplemental data ties back to specific claims in the thesis.

## Integration Points

- **Figures:** `scaffold/supplements/figures/` for condensed charts derived from supplementary calculations.
- **Tables:** `scaffold/supplements/tables/` for tabular exports (dose schedules, instrument metadata).
- **Notebooks:** `scaffold/supplements/notebooks/` to host scripts that transform the raw supplemental assets.
- **Assets:** `scaffold/supplements/assets/images/` and `/videos/` stage media before Agent 4 publishes derivatives to `docs/media/`.

## Thesis Integration Points

Use the following sections from `thesis.docx.md` to anchor supplemental prose and datasets:

- `## Materials and Methods` → `### Experimental design`
- `## Discussion` → `### Environmental dose benchmarking and site caveats`
- `## Discussion` → `### Instrument‑specific influences (sensor, angle, and geometry)`
- `## Discussion` → `### Replication and statistical power`
- `## Acknowledgements` — cite facility, instrument, and personnel credits.

## Thesis Excerpts (draft slots)

> _Quick pulls from `thesis.docx.md`; tailor and shorten as the supplements narrative forms._

### Experimental design & dosimetry (lines 41–83)

> “Variation in UV dose among treatments was achieved by positioning thalli within pre-mapped irradiance columns sourced from NOAA Arrival Heights benchmarks. UVA ran 24 h day⁻¹, UVB was pulsed 1.06 h day⁻¹, and daily dose was computed as D₍band₎ = P₍band₎ × t₍band₎ × 3600, yielding Grid 3 ≈ 100 % MDV UV-B and 110 % MDV UV-A.”

### Environmental benchmarking caveats (lines 439–444)

> “Arrival Heights measurements provide energy benchmarks, not strict equivalence: we did not correct for vertical gradients, horizon shading, or local sky-view factors. Hemispherical photography or SVF estimation would scale benchmarks to the actual optical environment of the sample plane.”

### Geometry & instrumentation (lines 412–417)

> “Off-nadir fiber geometry and textured thalli confounded BRDF effects, with scytonemin’s sheath localization amplifying surface heterogeneity. Future supplementary documentation should note spacers, nadir capture, or cosine correctors to standardize sampling.”

### Replication guidance (lines 455–458)

> “Large replicate variability indicates future power analyses should target ≥15–20 biological replicates per treatment with paired reflectance and HPLC. Distinguishing technical from biological variance, and leveraging culture-grown material with scytonemin-null controls, will better capture fingerprint dynamics.”

### Acknowledgements (lines 469–475)

> “Summaries should reference the thesis acknowledgements section to credit instrumentation support, facility access, and collaborators who enabled the supplemental datasets.”

## Pending Inputs

- Agent 3 to checksum supplemental data and generate reviewer-friendly summaries.
- Agent 4 to prepare compressed thumbnails/posters for the media gallery.
- Agent 2 to weave in cross-references using the manuscript sections listed above.
