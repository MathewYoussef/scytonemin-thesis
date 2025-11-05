# Pending Inserts for Results Section

## Chromatogram Calibration Coefficients
Validated free-intercept fits for chromatogram standards (QC pass, max |relative residual| ≤0.142) are summarized below.

| form     | slope (mg·mL⁻¹·AUC⁻¹) | intercept (mg·mL⁻¹) | slope SE | intercept SE | R²    | max |relative residual| | df |
|:---------|----------------------:|---------------------:|---------:|-------------:|:------|---------------------:|---:|
| total    | 7.211e-06             | -0.01599             | 2.632e-07| 0.005092     | 0.9934| 0.1274               | 5  |
| oxidized | 1.104e-05             | -0.01612             | 6.145e-07| 0.007752     | 0.9847| 0.1417               | 5  |
| reduced  | 2.056e-05             | -0.01473             | 3.18e-07 | 0.002148     | 0.9988| 0.1201               | 5  |

Source files: `standards_fitted_total.csv`, `standards_fitted_oxidized.csv`, `standards_fitted_reduced.csv`.

## DAD Single-Factor Regressions
Ordinary least-squares fits between UVA/UVB metrics and DAD-derived concentrations (n=30 for raw doses; n=25 for ratios).

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

Source file: `DAD_derived_concentrations_corrected.csv`.

## PLS Components and Cross-Validation
Two-component PLS models were evaluated with 5-fold cross-validation (`analysis_config.yaml:39-43`).

| dataset       | variant | measurement                   | components | R²     | CV R² |
|:--------------|:--------|:------------------------------|-----------:|-------:|------:|
| chromatogram  | raw     | conc_mg_ml                    | 2          | 0.0985 | 0.0716|
| chromatogram  | raw     | amount_mg_per_gDW             | 2          | 0.0961 | 0.0635|
| chromatogram  | delta   | delta_conc_mg_ml              | 2          | 0.1491 | 0.1118|
| chromatogram  | delta   | delta_amount_mg_per_gDW       | 2          | 0.1606 | 0.1013|
| chromatogram  | zscore  | z_conc_mg_ml                  | 2          | 0.1252 | 0.0956|
| chromatogram  | zscore  | z_amount_mg_per_gDW           | 2          | 0.1237 | 0.0846|
| DAD           | raw     | predicted_total_mg_per_gDW    | 2          | 0.3011 | 0.0195|
| DAD           | delta   | predicted_total_mg_per_gDW_delta | 2       | 0.3011 | 0.0195|

Sources: `chromatogram_pls_results.csv`, `dad_pls_results.csv`.

## Dose Design Counts
Unique dose levels and replicate counts used for chromatogram and DAD analyses.

- UVA levels (mW·cm⁻²): 0, 0.647, 1.095, 1.692, 2.488, 3.185.
- UVB levels (mW·cm⁻²): 0, 0.246, 0.338, 0.584, 0.707, 0.768.
- Single-factor regressions use n=30 samples (all dose cells); ratio metrics omit UVB=0 and yield n=25.

UVA × UVB cell counts (replicates per combination):

| UVA \\ UVB (mW·cm⁻²) | 0.000 | 0.246 | 0.338 | 0.584 | 0.707 | 0.768 |
|:---------------------|------:|------:|------:|------:|------:|------:|
| 0.000                | 5     | 0     | 0     | 0     | 0     | 0     |
| 0.647                | 0     | 5     | 0     | 0     | 0     | 0     |
| 1.095                | 0     | 0     | 5     | 0     | 0     | 0     |
| 1.692                | 0     | 0     | 0     | 5     | 0     | 0     |
| 2.488                | 0     | 0     | 0     | 0     | 0     | 5     |
| 3.185                | 0     | 0     | 0     | 0     | 5     | 0     |

Sources: `Chromatogram_derived_concentrations.csv`, `DAD_derived_concentrations_corrected.csv`.

## Calibration Orientation Note
All calibrations express concentration (mg·mL⁻¹) as a function of blank-corrected response/AUC (see `calibration_summary.md:5-9` and `DAD_to_Concentration_AUC/calibration_total.json`). Earlier inverted mappings should be annotated accordingly in the Methods/Appendix.

## LOD/LOQ Statement
Chromatogram blanks were not measured in the retained dataset, and DAD blank AUCs were unstable (mean ≈4.7×10⁴ AUC units) so instrument detection limits remain undefined. JSON calibration files (`DAD_to_Concentration_AUC/calibration_*.json`) report `LOD_mg_ml` and `LOQ_mg_ml` as `NaN`; note this explicitly in the Methods or Appendix.
