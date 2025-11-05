# Initial Calibration Scaffold — Purpose & Write API

**Purpose:** Collect chromatogram and DAD calibration artifacts that feed the concentration and dry-weight derivations.

## Writers
- **Agent 3:** Publishes calibration plots, coefficient tables, and notebooks that transform raw diode-array exports.
- **Agent 4:** Supplies setup imagery or lab walkthroughs prior to embedding them in the docs site.

## Expected Outputs
- `figures/` — Chromatogram traces, residual plots, calibration curves.
- `tables/` — Parameter tables, uncertainty budgets, mapping manifests.
- `notebooks/` — Re-runnable notebooks/scripts producing calibration deliverables.
- `assets/images/`, `assets/videos/` — Staging for supplementary media clips and photos.

## Conventions
- Keep filenames aligned with thesis figure/table numbering when available.
- Provide context in notebook metadata (input paths, config versions) to aid reproducibility.
- Large raw exports should stay in `data/` under DVC/LFS; only publish reviewer-sized derivatives here.

