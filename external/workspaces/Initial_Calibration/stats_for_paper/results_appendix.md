# Results Appendix

This appendix consolidates calibration, regression, ANOVA, robustness, and predictive modeling outputs cited in the main Results section. Source files are noted for traceability.

## Calibration

### DAD AUC Calibrations

| form     | slope (mg·mL⁻¹·AUC⁻¹) | intercept (mg·mL⁻¹) | slope SE | intercept SE | R²    | max |relative residual| | df |
|:---------|----------------------:|---------------------:|---------:|-------------:|:------|---------------------:|---:|
| total    | 4.526e-08             | -0.02402             | 6.447e-09| 0.02446      | 0.9079| 0.3073               | 5  |
| oxidized | 3.007e-08             | -0.03419             | 3.655e-09| 0.02283      | 0.9312| 0.2658               | 5  |
| reduced  | 4.711e-08             | -0.02169             | 3.069e-09| 0.01110      | 0.9792| 0.4688               | 5  |

Source: `calibration_summary.md:5-9`.

### Chromatogram Calibrations

| form     | slope (mg·mL⁻¹·AUC⁻¹) | intercept (mg·mL⁻¹) | slope SE | intercept SE | R²    | max |relative residual| | df |
|:---------|----------------------:|---------------------:|---------:|-------------:|:------|---------------------:|---:|
| total    | 7.211e-06             | -0.01599             | 2.632e-07| 0.005092     | 0.9934| 0.1274               | 5  |
| oxidized | 1.104e-05             | -0.01612             | 6.145e-07| 0.007752     | 0.9847| 0.1417               | 5  |
| reduced  | 2.056e-05             | -0.01473             | 3.18e-07 | 0.002148     | 0.9988| 0.1201               | 5  |

Source: `standards_fitted_total.csv`, `standards_fitted_oxidized.csv`, `standards_fitted_reduced.csv`.

### Orientation and Detection Limits

Calibrations are parameterized as concentration (mg·mL⁻¹) versus blank-corrected response (`calibration_summary.md:5-9`; `DAD_to_Concentration_AUC/calibration_total.json`). Instrument blanks were unavailable or unstable; consequently `LOD_mg_ml` and `LOQ_mg_ml` remain `NaN` in all calibration JSON files.

## Dose Design

- UVA levels (mW·cm⁻²): 0, 0.647, 1.095, 1.692, 2.488, 3.185.  
- UVB levels (mW·cm⁻²): 0, 0.246, 0.338, 0.584, 0.707, 0.768.  
- Five replicates per UVA–UVB combination, yielding n=30 for single-factor models; removing UVB=0 rows leaves n=25 for ratios.

UVA × UVB replicate counts (`Chromatogram_derived_concentrations.csv`; identical for the DAD dataset):

| UVA \\ UVB (mW·cm⁻²) | 0.000 | 0.246 | 0.338 | 0.584 | 0.707 | 0.768 |
|:---------------------|------:|------:|------:|------:|------:|------:|
| 0.000                | 5     | 0     | 0     | 0     | 0     | 0     |
| 0.647                | 0     | 5     | 0     | 0     | 0     | 0     |
| 1.095                | 0     | 0     | 5     | 0     | 0     | 0     |
| 1.692                | 0     | 0     | 0     | 5     | 0     | 0     |
| 2.488                | 0     | 0     | 0     | 0     | 0     | 5     |
| 3.185                | 0     | 0     | 0     | 0     | 5     | 0     |

## Single-Factor Dose Relationships

### Chromatogram OLS Regressions

| dose_metric           | metric   |     slope |   intercept |       R² |   n |
|:----------------------|:---------|----------:|------------:|---------:|----:|
| p_uva_mw_cm2          | oxidized |  0.02531  | 0.3572      | 0.006287 |  30 |
| p_uva_mw_cm2          | reduced  |  0.2475   | 1.006       | 0.1514   |  30 |
| p_uva_mw_cm2          | total    |  0.115    | 0.5552      | 0.163    |  30 |
| p_uvb_mw_cm2          | oxidized |  0.1378   | 0.3349      | 0.01177  |  30 |
| p_uvb_mw_cm2          | reduced  |  1.19     | 0.8573      | 0.2208   |  30 |
| p_uvb_mw_cm2          | total    |  0.5607   | 0.4827      | 0.2445   |  30 |
| uva_uvb_ratio         | oxidized | -0.005497 | 0.443       | 0.000115 |  25 |
| uva_uvb_ratio         | reduced  | -0.0193   | 1.579       | 0.000349 |  25 |
| uva_uvb_ratio         | total    | -0.0100   | 0.8377      | 0.000751 |  25 |
| delta_p_uva_from_zero | oxidized |  0.02531  | 0.1075      | 0.006287 |  30 |
| delta_p_uva_from_zero | reduced  |  0.2475   | 0.2951      | 0.1514   |  30 |
| delta_p_uva_from_zero | total    |  0.115    | 0.2003      | 0.163    |  30 |

Source: `single_factor_summary.md:5-18`.

### DAD OLS Regressions

| dose metric     | response (unit)       | slope | slope SE | intercept | intercept SE | R²    | n  |
|:----------------|:----------------------|------:|---------:|----------:|-------------:|:------|---:|
| UVA             | total mg·mL⁻¹         | 0.0194| 0.00733  | 0.0724    | 0.0137       | 0.2008| 30 |
| UVA             | total mg·gDW⁻¹        | 0.221 | 0.0999   | 1.096     | 0.186        | 0.1490| 30 |
| UVA             | oxidized mg·mL⁻¹      | 0.0109| 0.00678  | 0.0245    | 0.0126       | 0.0846| 30 |
| UVA             | oxidized mg·gDW⁻¹     | 0.137 | 0.0964   | 0.357     | 0.179        | 0.0674| 30 |
| UVA             | reduced mg·mL⁻¹       | 0.0297| 0.0114   | 0.0915    | 0.0212       | 0.1952| 30 |
| UVA             | reduced mg·gDW⁻¹      | 0.332 | 0.1498   | 1.421     | 0.279        | 0.1488| 30 |
| UVB             | total mg·mL⁻¹         | 0.0829| 0.0286   | 0.0654    | 0.0148       | 0.2303| 30 |
| UVB             | total mg·gDW⁻¹        | 1.034 | 0.384    | 0.976     | 0.199        | 0.2055| 30 |
| UVB             | oxidized mg·mL⁻¹      | 0.0432| 0.0270   | 0.0221    | 0.0140       | 0.0837| 30 |
| UVB             | oxidized mg·gDW⁻¹     | 0.586 | 0.381    | 0.307     | 0.197        | 0.0776| 30 |
| UVB             | reduced mg·mL⁻¹       | 0.1218| 0.0450   | 0.0829    | 0.0233       | 0.2073| 30 |
| UVB             | reduced mg·gDW⁻¹      | 1.466 | 0.584    | 1.279     | 0.302        | 0.1836| 30 |
| UVA/UVB ratio   | total mg·mL⁻¹         | 0.0102| 0.0118   | 0.0793    | 0.0396       | 0.0313| 25 |
| UVA/UVB ratio   | total mg·gDW⁻¹        | 0.0088| 0.150    | 1.556     | 0.503        | 0.0002| 25 |
| UVA/UVB ratio   | oxidized mg·mL⁻¹      | 0.0109| 0.0110   | 0.0111    | 0.0372       | 0.0410| 25 |
| UVA/UVB ratio   | oxidized mg·gDW⁻¹     | 0.0905| 0.153    | 0.358     | 0.514        | 0.0150| 25 |
| UVA/UVB ratio   | reduced mg·mL⁻¹       | 0.0197| 0.0226   | 0.0851    | 0.0761       | 0.0320| 25 |
| UVA/UVB ratio   | reduced mg·gDW⁻¹      | 0.0910| 0.290    | 1.806     | 0.976        | 0.0043| 25 |

Source: `DAD_derived_concentrations_corrected.csv`.

### Dose Correlations

| form     | measurement   | dose_metric   |   pearson_r |   spearman_r |   n |
|:---------|:--------------|:--------------|------------:|-------------:|----:|
| total    | mg_mL         | UVA_mW_cm2    |     0.4802  |      0.4160  |  30 |
| total    | mg_mL         | UVB_mW_cm2    |     0.5359  |      0.4138  |  30 |
| total    | mg_mL         | UVA_div_UVB   |     0.1471  |      0.1490  |  25 |
| total    | mg_mL         | UVA_times_UVB |     0.4283  |      0.4160  |  30 |
| total    | mg_per_gDW    | UVA_mW_cm2    |     0.4037  |      0.3642  |  30 |
| total    | mg_per_gDW    | UVB_mW_cm2    |     0.4945  |      0.3800  |  30 |
| total    | mg_per_gDW    | UVA_div_UVB   |    -0.0274  |      0.0157  |  25 |
| total    | mg_per_gDW    | UVA_times_UVB |     0.3357  |      0.3642  |  30 |
| oxidized | mg_mL         | UVA_mW_cm2    |     0.1247  |      0.1974  |  30 |
| oxidized | mg_mL         | UVB_mW_cm2    |     0.1343  |      0.1545  |  30 |
| oxidized | mg_mL         | UVA_div_UVB   |     0.0681  |      0.1098  |  25 |
| oxidized | mg_mL         | UVA_times_UVB |     0.0714  |      0.1974  |  30 |

Source: `single_factor_summary.md:20-35`.

## Two-Factor Models

### Classical ANOVA — Chromatogram

| variant   | measurement             |   UVA_df |   UVA_F |   UVA_p |   UVA_eta² |   UVB_df |   UVB_F |   UVB_p |   UVB_eta² |   Interaction_df |   Interaction_F |   Interaction_p |   Interaction_eta² |
|:----------|:------------------------|---------:|--------:|--------:|-----------:|---------:|--------:|--------:|-----------:|-----------------:|----------------:|----------------:|-------------------:|
| delta     | delta_amount_mg_per_gDW | 5 | 2.622 | 0.1091 | 0.06771 | 5 | 2.632 | 0.1085 | 0.06797 | 25 | 3.334 | 0.00852 | 0.4305 |
| delta     | delta_conc_mg_ml        | 5 | 3.582 | 0.06185| 0.09113 | 5 | 3.487 | 0.06533| 0.08872 | 25 | 3.087 | 0.01316 | 0.3927 |
| percent   | pct_amount_mg_per_gDW   | 5 | 0.9634| 0.3307 | 0.04357 | 5 | 1.099 | 0.2991 | 0.04973 | 25 | 1.849 | 0.1187 | 0.4182 |
| percent   | pct_conc_mg_ml          | 5 | 2.941 | 0.09207| 0.1071  | 5 | 2.941 | 0.09207| 0.1071  | 25 | 2.157 | 0.07245| 0.3927 |
| ratio     | ratio_amount_mg_per_gDW | 5 | 1.550 | 0.2185 | 0.06871 | 5 | 0.9634| 0.3307 | 0.04270 | 25 | 1.849 | 0.1187 | 0.4099 |
| ratio     | ratio_conc_mg_ml        | 5 | 2.730 | 0.1043 | 0.1119  | 5 | 0.08367|0.7735 | 0.00343 | 25 | 2.157 | 0.07245| 0.4420 |
| raw       | amount_mg_per_gDW       | 5 | 1.107 | 0.2957 | 0.03921 | 5 | 1.101 | 0.2971 | 0.03898 | 25 | 1.847 | 0.1125 | 0.3270 |
| raw       | conc_mg_ml              | 5 | 2.334 | 0.1303 | 0.07521 | 5 | 2.301 | 0.1330 | 0.07415 | 25 | 1.920 | 0.09945| 0.3094 |
| zscore    | z_amount_mg_per_gDW     | 5 | 1.840 | 0.1786 | 0.05613 | 5 | 1.860 | 0.1762 | 0.05674 | 25 | 2.457 | 0.03965| 0.3747 |
| zscore    | z_conc_mg_ml            | 5 | 4.946 | 0.02884| 0.1303  | 5 | 3.470 | 0.06597| 0.09141 | 25 | 2.550 | 0.03374| 0.3358 |

Source: `two_factor_summary.md:5-16`.

### Classical ANOVA — DAD

| variant   | measurement                          |   UVA_df |   UVA_F |   UVA_p |   UVA_eta² |   UVB_df |   UVB_F |   UVB_p |   UVB_eta² |   Interaction_df |   Interaction_F |   Interaction_p |   Interaction_eta² |
|:----------|:-------------------------------------|---------:|--------:|--------:|-----------:|---------:|--------:|--------:|-----------:|-----------------:|----------------:|----------------:|-------------------:|
| delta     | predicted_oxidized_mg_ml_delta       | 5 | 0.2572 | 0.6166 | 0.02137 | 5 | 0.2577 | 0.6164 | 0.02140 | 25 | 1.345 | 0.2797 | 0.5585 |
| delta     | predicted_oxidized_mg_per_gDW_delta  | 5 | 0.3661 | 0.5508 | 0.02644 | 5 | 0.9411 | 0.3417 | 0.06797 | 25 | 1.548 | 0.2128 | 0.5589 |
| delta     | predicted_reduced_mg_ml_delta        | 5 | 0.2027 | 0.6566 | 0.01440 | 5 | 0.9330 | 0.3437 | 0.06630 | 25 | 1.627 | 0.1910 | 0.5782 |
| delta     | predicted_reduced_mg_per_gDW_delta   | 5 | 2.066  | 0.1635 | 0.1400  | 5 | 0.5124 | 0.4810 | 0.03473 | 25 | 1.475 | 0.2348 | 0.4999 |
| delta     | predicted_total_mg_ml_delta          | 5 | 2.118  | 0.1586 | 0.1234  | 5 | 0.02471|0.8764 | 0.00144 | 25 | 2.043 | 0.1084 | 0.5954 |
| delta     | predicted_total_mg_per_gDW_delta     | 5 | 0.1067 | 0.7468 | 0.00605 | 5 | 0.1717 | 0.6823 | 0.00973 | 25 | 2.515 | 0.05754| 0.7123 |
| percent   | predicted_oxidized_mg_ml_pct         | 5 | 0.2210 | 0.6426 | 0.01834 | 5 | 0.3041 | 0.5864 | 0.02524 | 25 | 1.345 | 0.2797 | 0.5581 |
| percent   | predicted_oxidized_mg_per_gDW_pct    | 5 | 0.9299 | 0.3445 | 0.06453 | 5 | 0.9417 | 0.3415 | 0.06535 | 25 | 1.548 | 0.2128 | 0.5370 |
| percent   | predicted_reduced_mg_ml_pct          | 5 | 1.170  | 0.2901 | 0.07285 | 5 | 1.958  | 0.1745 | 0.1219  | 25 | 1.627 | 0.1910 | 0.5065 |
| percent   | predicted_reduced_mg_per_gDW_pct     | 5 | 0.8278 | 0.3720 | 0.05996 | 5 | 0.8034 | 0.3790 | 0.05820 | 25 | 1.475 | 0.2348 | 0.5127 |

Source: `two_factor_summary.md:18-33`.

### OLS Coefficients — Chromatogram (Δ Metrics)

| measurement             | term                      |    coef |   pvalue |   R²    | adj R² |
|:------------------------|:--------------------------|--------:|---------:|--------:|-------:|
| delta_conc_mg_ml        | Intercept                 | 0.01516 | 0.0818   | 0.1499 | 0.1202 |
| delta_conc_mg_ml        | p_uva_mw_cm2              | 0.00681 | 0.7445   |        |        |
| delta_conc_mg_ml        | p_uvb_mw_cm2              | 0.07346 | 0.09395  |        |        |
| delta_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.01649| 0.4388   |        |        |
| delta_amount_mg_per_gDW | Intercept                 | 0.1323  | 0.2455   | 0.1623 | 0.1331 |
| delta_amount_mg_per_gDW | p_uva_mw_cm2              | 0.09696 | 0.7240   |        |        |
| delta_amount_mg_per_gDW | p_uvb_mw_cm2              | 1.353   | 0.01982  |        |        |
| delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.3703 | 0.1872   |        |        |

Source: `two_factor_summary.md:134-145`.

### Permutation and Freedman–Lane Tests

| dataset      | variant | measurement                | F_obs | permuted p | permutations |
|:-------------|:--------|:---------------------------|------:|-----------:|-------------:|
| chromatogram | delta   | delta_conc_mg_ml           | 3.087 | 0.01199    | 2000         |
| chromatogram | delta   | delta_amount_mg_per_gDW    | 3.334 | 0.00950    | 2000         |
| chromatogram | zscore  | z_conc_mg_ml               | 2.550 | 0.03148    | 2000         |
| chromatogram | zscore  | z_amount_mg_per_gDW        | 2.457 | 0.03848    | 2000         |
| DAD          | raw     | predicted_total_mg_per_gDW | 2.515 | 0.06247    | 2000         |

| dataset      | variant | measurement                | F_obs | Freedman–Lane p | permutations |
|:-------------|:--------|:---------------------------|------:|----------------:|-------------:|
| chromatogram | delta   | delta_conc_mg_ml           | 3.087 | 0.7271          | 2000         |
| chromatogram | delta   | delta_amount_mg_per_gDW    | 3.334 | 0.7136          | 2000         |
| chromatogram | zscore  | z_conc_mg_ml               | 2.550 | 0.7406          | 2000         |
| chromatogram | zscore  | z_amount_mg_per_gDW        | 2.457 | 0.7366          | 2000         |
| DAD          | raw     | predicted_total_mg_per_gDW | 2.515 | 0.7871          | 2000         |

Source: `two_factor_summary.md:460-493`.

## Robustness and Diagnostics

### Residual Diagnostics

| dataset      | variant | metric                     | n_obs | Shapiro p | Jarque–Bera | JB p    | max |studentized resid| | max leverage | leverage threshold | n high leverage | max Cook's D |
|:-------------|:--------|:---------------------------|------:|----------:|------------:|--------:|------------------:|-------------:|-------------------:|----------------:|-------------:|
| chromatogram | raw     | conc_mg_ml                 | 90 | 3.702e-05 | 61.7  | 4.0e-14 | 4.261 | 0.06667 | 0.1333 | 0 | 0.03603 |
| chromatogram | raw     | amount_mg_per_gDW          | 90 | 3.244e-05 | 40.2  | 1.86e-09 | 3.743 | 0.06667 | 0.1333 | 0 | 0.02780 |
| chromatogram | delta   | delta_conc_mg_ml           | 90 | 2.708e-06 |118.6 | 1.79e-26 | 4.682 | 0.06667 | 0.1333 | 0 | 0.04349 |
| chromatogram | delta   | delta_amount_mg_per_gDW    | 90 | 3.636e-05 | 45.95| 1.05e-10 | 3.981 | 0.06667 | 0.1333 | 0 | 0.03144 |
| chromatogram | zscore  | z_conc_mg_ml               | 90 | 1.272e-07 |178.2 | 2.05e-39 | 4.826 | 0.06667 | 0.1333 | 0 | 0.04621 |
| chromatogram | zscore  | z_amount_mg_per_gDW        | 90 | 3.740e-08 |117.6 | 2.96e-26 | 4.250 | 0.06667 | 0.1333 | 0 | 0.03583 |
| DAD          | raw     | predicted_total_mg_per_gDW | 30 | 0.5902    | 1.184 | 0.5533  | 2.284 | 0.2000  | 0.4000 | 0 | 0.03624 |
| DAD          | delta   | predicted_total_mg_per_gDW_delta | 30 | 0.5902| 1.184 | 0.5533  | 2.284 | 0.2000  | 0.4000 | 0 | 0.03624 |

Source: `diagnostics_and_sensitivity_summary.md:5-14`.

### Variance Inflation Factors

| dataset      | variant | metric                     | term                      | VIF  |
|:-------------|:--------|:---------------------------|:--------------------------|-----:|
| chromatogram | raw     | conc_mg_ml                 | p_uva_mw_cm2              | 37.85|
| chromatogram | raw     | conc_mg_ml                 | p_uvb_mw_cm2              | 10.35|
| chromatogram | raw     | conc_mg_ml                 | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13|
| chromatogram | delta   | delta_conc_mg_ml           | p_uva_mw_cm2              | 37.85|
| chromatogram | delta   | delta_conc_mg_ml           | p_uvb_mw_cm2              | 10.35|
| chromatogram | delta   | delta_conc_mg_ml           | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13|
| chromatogram | zscore  | z_conc_mg_ml               | p_uva_mw_cm2              | 37.85|
| chromatogram | zscore  | z_conc_mg_ml               | p_uvb_mw_cm2              | 10.35|
| chromatogram | zscore  | z_conc_mg_ml               | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13|
| DAD          | raw     | predicted_total_mg_per_gDW | p_uva_mw_cm2              | 37.85|
| DAD          | raw     | predicted_total_mg_per_gDW | p_uvb_mw_cm2              | 10.35|
| DAD          | raw     | predicted_total_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13|

Source: `diagnostics_and_sensitivity_summary.md:18-43`.

### UVA–UVB Correlation

| dataset      | variant | Pearson r | n_obs |
|:-------------|:--------|----------:|------:|
| chromatogram | raw     | 0.9498    | 90    |
| chromatogram | delta   | 0.9498    | 90    |
| chromatogram | zscore  | 0.9498    | 90    |
| DAD          | raw     | 0.9498    | 30    |
| DAD          | delta   | 0.9498    | 30    |

Source: `diagnostics_and_sensitivity_summary.md:45-54`.

### Outlier Sensitivity

| variant | measurement             | scenario         | removed | n_removed | F    | p-value |
|:--------|:------------------------|:-----------------|:-------:|----------:|-----:|--------:|
| delta   | delta_conc_mg_ml        | baseline         | —       | 0         | 3.087| 0.01316|
| delta   | delta_conc_mg_ml        | drop_single      | 2B      | 1         | 3.222| 0.01043|
| delta   | delta_conc_mg_ml        | drop_single      | 1J      | 1         | 3.213| 0.01060|
| delta   | delta_conc_mg_ml        | drop_flagged_set | 2B;1J   | 2         | 3.323| 0.00878|
| delta   | delta_amount_mg_per_gDW | baseline         | —       | 0         | 3.334| 0.00852|
| delta   | delta_amount_mg_per_gDW | drop_single      | 2B      | 1         | 3.592| 0.00545|
| zscore  | z_conc_mg_ml            | baseline         | —       | 0         | 2.550| 0.03374|
| zscore  | z_conc_mg_ml            | drop_flagged_set | 2B;1J   | 2         | 2.701| 0.02608|

Source: `diagnostics_and_sensitivity_summary.md:57-69`.

## Penalized and Predictive Models

### Ridge Regression Summaries

| dataset      | variant | measurement                | R²    |
|:-------------|:--------|:---------------------------|------:|
| chromatogram | raw     | conc_mg_ml                 | 0.0917|
| chromatogram | raw     | amount_mg_per_gDW          | 0.0820|
| chromatogram | delta   | delta_conc_mg_ml           | 0.1388|
| chromatogram | delta   | delta_amount_mg_per_gDW    | 0.1609|
| chromatogram | zscore  | z_conc_mg_ml               | 0.1171|
| chromatogram | zscore  | z_amount_mg_per_gDW        | 0.1065|
| DAD          | raw     | predicted_total_mg_per_gDW | 0.2995|
| DAD          | delta   | predicted_total_mg_per_gDW_delta | 0.2995|

Sources: `chromatogram_ridge_results.csv`, `dad_ridge_results.csv`.

### Ridge Bootstrap Intervals (selected terms)

| dataset      | measurement             | term                      | mean coef | 95% CI (p2.5, p97.5) |
|:-------------|:------------------------|:--------------------------|----------:|--------------------:|
| chromatogram | delta_amount_mg_per_gDW | p_uvb_mw_cm2              | 1.3471    | [0.0549, 2.6231]    |
| chromatogram | delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.3613   | [-0.8480, 0.1987]   |
| chromatogram | delta_conc_mg_ml        | p_uvb_mw_cm2              | 0.0683    | [-0.0082, 0.1467]   |
| DAD          | predicted_total_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.0307 | [-1.9627, 0.0281]   |

Source: `ridge_bootstrap_summary.csv`.

### PLS Models

| dataset      | variant | measurement                   | components | R²    | CV R² |
|:-------------|:--------|:------------------------------|-----------:|------:|------:|
| chromatogram | raw     | conc_mg_ml                    | 2          | 0.0985| 0.0716|
| chromatogram | raw     | amount_mg_per_gDW             | 2          | 0.0961| 0.0635|
| chromatogram | delta   | delta_conc_mg_ml              | 2          | 0.1491| 0.1118|
| chromatogram | delta   | delta_amount_mg_per_gDW       | 2          | 0.1606| 0.1013|
| chromatogram | zscore  | z_conc_mg_ml                  | 2          | 0.1252| 0.0956|
| chromatogram | zscore  | z_amount_mg_per_gDW           | 2          | 0.1237| 0.0846|
| DAD          | raw     | predicted_total_mg_per_gDW    | 2          | 0.3011| 0.0195|
| DAD          | delta   | predicted_total_mg_per_gDW_delta | 2       | 0.3011| 0.0195|

Sources: `chromatogram_pls_results.csv`, `dad_pls_results.csv`; cross-validation scheme defined in `analysis_config.yaml:39-43`.
