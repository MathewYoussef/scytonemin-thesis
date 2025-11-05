# Dose-Level Summary Statistics (mg/gDW)

All regressions use dose-rank (UVA order) as the predictor with weights = 1 / bootstrap variance of the trimmed mean. Values rounded to three decimals.

| Metric | WLS slope (95% CI) | p (slope) | Spearman r (p) | Interpretation |
|--------|-------------------|-----------|-----------------|----------------|
| Chromatogram total | 0.071 (−0.034, 0.175) | 0.134 | 0.829 (0.041) | Mean trend increases with dose, but CI spans zero once weights applied. |
| Chromatogram oxidized | 0.026 (−0.047, 0.099) | 0.377 | 0.371 (0.468) | Weak monotone evidence; oxidized component remains noisy after aggregation. |
| Chromatogram reduced | 0.186 (0.015, 0.356) | 0.039 | 0.771 (0.072) | Positive monotone relationship; reduced fraction drives total trend. |
| DAD total | 0.125 (−0.119, 0.369) | 0.228 | 0.714 (0.111) | Similar incline to chromatogram total, but CI crosses zero. |
| DAD oxidized | 0.079 (−0.052, 0.210) | 0.169 | 0.371 (0.468) | Oxidized DAD signal mirrors chromatogram variability; monotonic but not significant. |
| DAD reduced | 0.250 (0.028, 0.473) | 0.035 | 0.543 (0.266) | Reduced component shows the clearest mean-level increase with dose. |

Deming regression (chromatogram vs. DAD dose means):

| Pair | Error ratio λ | Slope (95% CI) | Intercept (95% CI) | Notes |
|------|---------------|----------------|--------------------|-------|
| Total | 0.229 | 1.885 (1.119, 2.186) | 0.026 (−0.255, 0.646) | Near-linear scaling; DAD totals ≈ 1.9× chromatogram totals. |
| Oxidized | 0.426 | 2.052 (1.430, 2.917) | −0.209 (−0.376, 0.058) | Oxidized component shows higher scaling yet consistent dose ranking. |
| Reduced | 1.148 | 1.090 (0.375, 1.408) | 0.338 (0.021, 1.430) | Reduced concentrations align closely across assays. |
