# Supplemental Figures Roadmap

- **figS01_diagnostics/**  
  Generate residual vs fitted, Q–Q, and leverage/Cook’s panels for representative chromatogram and DAD endpoints (n = 30).  
  Annotate thresholds (studentised residual ±3, leverage 2p/n, Cook’s distance 4/(n − p)).  
  Reiterate that no point breached influence criteria.

- **figS07_cross_assay_alignment/**  
  Convert chromatogram mg·gDW⁻¹ to mg·mL⁻¹ via `conc_mg_ml = conc_mg_per_gDW * (dry_mass_g / extract_volume_mL)` for paired samples, plot against DAD total mg·mL⁻¹, include OLS fit and 45° identity line.  
  Report Pearson r and note assays are corroborative, not identical.

Use the provenance template in the parent directory to log data sources and script hashes once the plots are generated.
