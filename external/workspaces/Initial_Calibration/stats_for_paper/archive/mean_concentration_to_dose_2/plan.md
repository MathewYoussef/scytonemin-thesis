## Dose-Level Robust Mean Analysis Plan

### 1. Assemble Dose-Level Summaries
- Pull UVA/UVB doses and chromatogram & DAD concentrations from `Combined_Scytonemin_Concentrations.csv`.
- For each dose cell (UVAᵢ, UVBᵢ), compute robust location for total/oxidized/reduced:
  - start with a 20 % trimmed mean; fall back to Huber M-estimate if bootstrap shows strong asymmetry.
  - record trimmed SD (or MAD × 1.4826) as the within-dose spread.
- Bootstrap each cell (200 draws, resampling replicates) to obtain 95 % CIs for the robust mean.
- Store results in a tidy table with one row per dose and columns:
  `dose_id`, `uva_mw_cm2`, `uvb_mw_cm2`, `dose_rank` (monotone UVA rank), then robust means/CIs and spreads for both assays.

### 2. Summarise Mean-Level Dose Relationships
- Treat the robust-mean table (n = 6) as the “dose-level” dataset.
- Define a primary dose axis using `dose_rank` (monotone in UVA; note UVB peaks at the penultimate rank).
- For each metric (chrom_total, chrom_oxidized, …, dad_reduced, latent totals):
  - compute Pearson/Spearman/Kendall vs. `dose_rank`.
  - run weighted least squares: weights = 1 / bootstrap_variance. Report slope, SE, 95 % CI, R².
- Flag in notes that UVB is redundant except for the final swap (UVA_max pairs with UVB=0.707 < 0.768).

### 3. Maintain Replicate-Level Context
- Keep existing replicate-level models (ANOVA, permutation tests, ridge/PLS) untouched.
- Prepare a short comparison table summarising:
  - replicate-level inference (n = 30): “no robust dose effect”.
  - dose-level robust means (n = 6): slopes, CIs, and monotonicity checks.
- Plan a figure overlaying replicate stripplots with robust mean ± bootstrap CI to visualise variance collapse.

### 4. Chromatogram vs. DAD Alignment at Dose Level
- Using the robust means per dose, evaluate agreement between chromatogram and DAD outputs:
  - compute Pearson/Spearman correlations.
  - run Deming or orthogonal regression (use trimmed SD ratio to set error ratio). Report slope/intercept/CI.
- Document that both assays preserve the same dose ranking despite scale offsets.

### 5. Package Outputs for Reflectance Work
- Save the tidy summary to `mean_concentration_to_dose_2/dose_level_summary.csv` (later step).
- Include metadata (`mean_concentration_to_dose_2/README.md`) describing dose IDs, ranks, robust estimator choice, bootstrap settings.
- Ensure downstream scripts reference this table as the canonical concentration benchmark for reflectance modelling.

### 6. Narrative Integration
- Update Results text to present dual perspectives:
  1. Replicate-level variance obscures consistent dose effects (supported by existing statistics).
  2. Collapsing to robust dose means reveals monotone increases (effect sizes + CIs, descriptive emphasis).
- Highlight why both views matter: biological variability vs. latent dose-response, and note the UVA/UVB non-monotone pairing at the top dose.

### Open Questions / Validation
- Confirm trimmed vs. Huber choice yields stable means across all forms.
- Verify bootstrap CI coverage with sensitivity runs (e.g., 500 resamples) for edge doses.
- Decide whether latent precision-weighted means are stored alongside assay-specific metrics or in a companion table.
