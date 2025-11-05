# Plots Test 1 — Figure Production Checklist

This folder scaffolds the figure set described in the latest specification. Each subdirectory (or filename) mirrors the planned plots so we can drop scripts/outputs in an organized way.

## Core Figures

1. **fig01_calibration_{assay}_{analyte}.ipynb / .py**
   - Chromatogram (total/oxidized/reduced)
   - DAD (total/oxidized/reduced)
   - Includes residual overlays and annotation payload (slope/intercept/SE/R²/max |rel resid|/df).

2. **fig02_dose_structure.***
   - UVA vs UVB scatter with Pearson r and VIF call-out.
   - UVA and UVB marginal histograms/densities.

3. **fig03_single_factor_chrom_{analyte}_{dose}.***
   - Total/oxidized/reduced vs UVB and vs UVA (mg·gDW⁻¹).

4. **fig04_single_factor_dad_total_{dose}.***
   - UVB and UVA single-factor fits on the corrected DAD scale; annotate slopes ± SE and R².

5. **fig05_emm_{endpoint}.***
   - Estimated marginal means for chromatogram z and raw amount endpoints.

6. **fig06_pvals_methods_{endpoint}.***
   - Classical vs HC3 vs rank interaction p-values (−log₁₀ scale).

7. **fig07_freedman_lane_{endpoint}.***
   - Freedman–Lane permutation histograms for Δ / z chromatogram endpoints plus DAD total raw.

8. **fig08_cv_performance.***
   - Ridge & PLS cross-validated R² summaries and actual-vs-predicted scatter.

9. **fig09_bootstrap_coeffs.***
   - Ridge bootstrap coefficient distributions (β_UVB, β_interaction).

10. **fig10_outlier_path.***
    - Interaction F/p trajectory under outlier removal.

## Supplemental Figures (S-series)

- **figS01_diagnostics_{endpoint}.***: Residual/Q–Q/leverage diagnostics.
- **figS07_cross_assay_alignment.***: Chromatogram vs DAD sanity plot.

## Notes

- Use the repository CSVs/JSONs cited in the Results section as authoritative sources.
- Keep axis limits consistent across comparable panels.
- Save generated figures as both `.png` (publication ready) and `.pdf` (vector backup) once produced.
- Each plotting script should write metadata (data file SHA, timestamp, script version) to a sidecar `.yml` for provenance.

