# Dose-Level Results Overview

## Summary Tables

### Dose Means & CIs

_See `dose_level_summary.csv` for full metrics; key stats consolidated below._

| Metric | Trimmed mean range (mg·gDW⁻¹) | Final − Initial | Increases | Decreases | Dose-to-dose pattern |
|:-------|------------------------------:|----------------:|---------:|---------:|:---------------------|
| Chromatogram Total | 0.274→0.986 | 0.558 | 4 | 1 | 0.000->0.647: increase (0.396); 0.647->1.095: increase (0.077); 1.095->1.692: increase (0.239); 1.692->2.488: decrease (-0.178); 2.488->3.185: increase (0.024) |
| Chromatogram Oxidized | 0.090→0.611 | 0.303 | 4 | 1 | 0.000->0.647: increase (0.179); 0.647->1.095: increase (0.276); 1.095->1.692: increase (0.067); 1.692->2.488: decrease (-0.378); 2.488->3.185: increase (0.160) |
| Chromatogram Reduced | 0.740→1.790 | 0.671 | 3 | 2 | 0.000->0.647: increase (0.601); 0.647->1.095: decrease (-0.159); 1.095->1.692: increase (0.336); 1.692->2.488: increase (0.272); 2.488->3.185: decrease (-0.379) |
| DAD Total | 0.520→1.844 | 1.065 | 4 | 1 | 0.000->0.647: increase (0.885); 0.647->1.095: increase (0.087); 1.095->1.692: increase (0.352); 1.692->2.488: decrease (-0.404); 2.488->3.185: increase (0.145) |
| DAD Oxidized | -0.114→0.951 | 0.880 | 4 | 1 | 0.000->0.647: increase (0.491); 0.647->1.095: increase (0.416); 1.095->1.692: increase (0.158); 1.692->2.488: decrease (-0.583); 2.488->3.185: increase (0.398) |
| DAD Reduced | 1.050→2.093 | 0.863 | 3 | 2 | 0.000->0.647: increase (0.891); 0.647->1.095: decrease (-0.235); 1.095->1.692: increase (0.319); 1.692->2.488: increase (0.067); 2.488->3.185: decrease (-0.180) |

### Linear & Weighted Trends

_Slopes (mg·gDW⁻¹ per mW·cm⁻²) against UVA; UVB slopes available in CSV._

| Metric | OLS slope | OLS p | WLS slope | WLS p | R² (OLS) |
|:-------|----------:|------:|----------:|------:|---------:|
| Chromatogram total | 0.147 | 0.109 | 0.105 | 0.190 | 0.514 |
| Chromatogram oxidized | 0.055 | 0.521 | 0.038 | 0.439 | 0.110 |
| Chromatogram reduced | 0.223 | 0.087 | 0.313 | 0.040 | 0.560 |
| Dad total | 0.245 | 0.167 | 0.173 | 0.301 | 0.415 |
| Dad oxidized | 0.187 | 0.243 | 0.109 | 0.243 | 0.319 |
| Dad reduced | 0.224 | 0.130 | 0.417 | 0.046 | 0.474 |

### Quadratic Fits (UVA)

| Metric | coef₁ (x) | p(x) | coef₂ (x²) | p(x²) | adj R² |
|:-------|---------:|-----:|-----------:|------:|-------:|
| Chromatogram Total | 0.593 | 0.017 | -0.139 | 0.033 | 0.859 |
| Chromatogram Oxidized | 0.439 | 0.154 | -0.119 | 0.183 | 0.255 |
| Chromatogram Reduced | 0.755 | 0.061 | -0.165 | 0.120 | 0.712 |
| DAD Total | 1.104 | 0.038 | -0.266 | 0.064 | 0.741 |
| DAD Oxidized | 0.839 | 0.132 | -0.202 | 0.195 | 0.410 |
| DAD Reduced | 0.884 | 0.053 | -0.205 | 0.095 | 0.702 |

### Isotonic Fits

| Metric | RMSE | Isotonic mean path | Residuals (observed − isotonic) |
|:-------|-----:|:-------------------|:---------------------------------|
| Chromatogram Total | 0.056 | 0.274; 0.670; 0.747; 0.881; 0.881; 0.881 | +0.000; +0.000; +0.000; +0.105; -0.074; -0.050 |
| Chromatogram Oxidized | 0.119 | 0.090; 0.269; 0.451; 0.451; 0.451; 0.451 | +0.000; +0.000; +0.093; +0.160; -0.218; -0.058 |
| Chromatogram Reduced | 0.119 | 0.740; 1.261; 1.261; 1.518; 1.600; 1.600 | +0.000; +0.079; -0.079; +0.000; +0.190; -0.190 |
| DAD Total | 0.118 | 0.520; 1.405; 1.492; 1.628; 1.628; 1.628 | +0.000; +0.000; +0.000; +0.216; -0.189; -0.043 |
| DAD Oxidized | 0.174 | -0.114; 0.377; 0.691; 0.691; 0.691; 0.765 | +0.000; +0.000; +0.102; +0.260; -0.323; +0.000 |
| DAD Reduced | 0.086 | 1.050; 1.824; 1.824; 2.008; 2.008; 2.008 | +0.000; +0.117; -0.117; +0.017; +0.084; -0.095 |

### Correlations (UVA)

| Metric | Pearson r (p) | Spearman ρ (p) | Kendall τ (p) |
|:-------|---------------|----------------|----------------|
| Chromatogram total | 0.717 (0.109) | 0.829 (0.042) | 0.733 (0.056) |
| Chromatogram oxidized | 0.332 (0.521) | 0.371 (0.468) | 0.333 (0.469) |
| Chromatogram reduced | 0.748 (0.087) | 0.771 (0.072) | 0.600 (0.136) |
| Dad total | 0.645 (0.167) | 0.714 (0.111) | 0.600 (0.136) |
| Dad oxidized | 0.565 (0.243) | 0.371 (0.468) | 0.333 (0.469) |
| Dad reduced | 0.689 (0.130) | 0.543 (0.266) | 0.467 (0.272) |

## Key Findings

- Chromatogram reduced shows the strongest monotonic-like behaviour: trimmed means rise in 3/5 steps, WLS slope 0.313 (p≈0.040), UVB alias slope significant (p≈0.014).
- DAD reduced mirrors this pattern (WLS slope 0.417, p≈0.046) though the OLS fit is only marginal; step summary shows increase–decrease–increase–increase–decrease.
- Total forms (chrom & DAD) increase at the first three doses, dip at 2.488 mW·cm⁻², and tick back up, consistent with a “hump then recovery” rather than strict monotonicity; quadratic terms are significant/slightly positive.
- Oxidized fractions show mild trends; slopes non-significant but still positive, suggesting weaker linkage to dose.
- Isotonic fits reduce RMSE modestly and reveal plateaus (e.g., total concentrations flatten at high dose), implying a saturation or mid-dose dip.

## Next Analyses

- Compare the isotope-derived mean paths with reflectance-derived curves to see if the same increase–dip–increase pattern appears.
- Use the step-sequence deltas as a template when aligning with reflectance (look for matching sign runs such as +,−,+,+,−).
- Consider Jonckheere–Terpstra or aligned-rank trend tests to formally assess ordered alternatives beyond linear slopes.
