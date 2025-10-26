# Act of God — Mamba-SSM Block

The Act-of-God (AoG) run promotes the best-performing fold-02 checkpoint into an all-data model, safeguarded by a fixed validation panel. Source material is in `Act_of_God_Mamba_Results/`, with manifests, configs, and evaluator scripts ready for cataloguing.

## Narrative Scope

- Document the warm-start rationale, validation gates, and proxy evaluation metrics referenced in the thesis results chapter.
- Surface training/validation manifests and checkpoint hashes so reviewers can reproduce the AoG run.
- Summarise comparisons against baseline denoisers and downstream analytics.

## Integration Points

- **Figures:** `scaffold/mamba_ssm/figures/` for loss curves, panel performance charts, and downstream proxy summaries.
- **Tables:** `scaffold/mamba_ssm/tables/` for per-gate metrics, manifest counts, and readiness checklists.
- **Notebooks:** `scaffold/mamba_ssm/notebooks/` where evaluation notebooks deposit their outputs.
- **Media:** Videos or animations explaining model behaviour funnel through Agent 4 before linking here.

## Thesis Integration Points

Tie this page back to `thesis.docx.md` excerpts once they are ready:

- `## Materials and Methods` → `### Computation, normalization, & denoising`
- `## Results` → `### Reflectance Spectra (Dose and Concentration)` (for comparative baselines)
- `## Discussion` → `### Noise in the data, temperature control, and light source`
- `## Discussion` → `### Hypothesis appraisal and Conclusion`

## Thesis Excerpts (draft slots)

> _Source references point into `thesis.docx.md` for downstream verification._

### Computation, normalization, & denoising (lines 147–170)

> “To suppress high-frequency sensor noise without blurring pigment features, we applied a one-dimensional Mamba state-space model along the wavelength axis. Selective gates and FiLM-conditioned metadata let the backbone respond differently to dose and angle regimes while preserving narrow band morphology.”

### Reflectance spectra + Mamba outputs (lines 342–353)

> “Mamba-SSM denoised spectra were compared in 320–480 nm and 360–410 nm windows. Continuum removal aligned shoulders across doses, turning absorption valleys into positive bowl-occupancy features that track dose responses in figures 4–5.”

### Noise, temperature, and lamp stability (lines 418–421)

> “Baseline drift increases when detector temperatures fluctuate, especially under broadband lamps. LED-driven UVA/UVB deliver more stable, line-centered fluence with lower spectral uncertainty—informing future AoG runs.”

### Hypothesis appraisal & conclusion (lines 459–468)

> “Reflectance bore a reproducible scytonemin-region fingerprint but failed to remain strictly monotonic with pigment abundance, owing to geometry, moisture, and co-absorbers. Next iterations call for nadir-locked capture, matrix-matched calibration, and joint scytonemin + pigmentomics pipelines to tighten the AoG checks.”

## Pending Inputs

- Agent 3 to hook DVC/LFS pointers for checkpoints under `models/` and drop processed evaluation tables here.
- Agent 2 to incorporate thesis prose describing the AoG experiment once excerpts are lifted from `thesis.docx.md`.
- Agent 4 to embed storyboarded media highlighting instrumentation or inference demos.
