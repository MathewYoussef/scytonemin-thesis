# Diagnostics and Sensitivity

## Residual diagnostics

| dataset      | variant   | metric                           |   n_obs |   shapiro_p |   jarque_bera |   jarque_bera_p |   max_abs_studentized_resid |   max_leverage |   leverage_threshold |   n_high_leverage |   max_cooks_distance |
|:-------------|:----------|:---------------------------------|--------:|------------:|--------------:|----------------:|----------------------------:|---------------:|---------------------:|------------------:|---------------------:|
| chromatogram | raw       | conc_mg_ml                       |      90 |   3.702e-05 |        61.7   |       4e-14     |                       4.261 |        0.06667 |               0.1333 |                 0 |              0.03603 |
| chromatogram | raw       | amount_mg_per_gDW                |      90 |   3.244e-05 |        40.2   |       1.863e-09 |                       3.743 |        0.06667 |               0.1333 |                 0 |              0.0278  |
| chromatogram | delta     | delta_conc_mg_ml                 |      90 |   2.708e-06 |       118.6   |       1.79e-26  |                       4.682 |        0.06667 |               0.1333 |                 0 |              0.04349 |
| chromatogram | delta     | delta_amount_mg_per_gDW          |      90 |   3.636e-05 |        45.95  |       1.053e-10 |                       3.981 |        0.06667 |               0.1333 |                 0 |              0.03144 |
| chromatogram | zscore    | z_conc_mg_ml                     |      90 |   1.272e-07 |       178.2   |       2.048e-39 |                       4.826 |        0.06667 |               0.1333 |                 0 |              0.04621 |
| chromatogram | zscore    | z_amount_mg_per_gDW              |      90 |   3.74e-08  |       117.6   |       2.962e-26 |                       4.25  |        0.06667 |               0.1333 |                 0 |              0.03583 |
| dad          | raw       | predicted_total_mg_per_gDW       |      30 |   0.5902    |         1.184 |       0.5533    |                       2.284 |        0.2     |               0.4    |                 0 |              0.03624 |
| dad          | delta     | predicted_total_mg_per_gDW_delta |      30 |   0.5902    |         1.184 |       0.5533    |                       2.284 |        0.2     |               0.4    |                 0 |              0.03624 |

## Variance inflation factors

| dataset      | variant   | metric                           | term                      |   vif |
|:-------------|:----------|:---------------------------------|:--------------------------|------:|
| chromatogram | raw       | conc_mg_ml                       | p_uva_mw_cm2              | 37.85 |
| chromatogram | raw       | conc_mg_ml                       | p_uvb_mw_cm2              | 10.35 |
| chromatogram | raw       | conc_mg_ml                       | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| chromatogram | raw       | amount_mg_per_gDW                | p_uva_mw_cm2              | 37.85 |
| chromatogram | raw       | amount_mg_per_gDW                | p_uvb_mw_cm2              | 10.35 |
| chromatogram | raw       | amount_mg_per_gDW                | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| chromatogram | delta     | delta_conc_mg_ml                 | p_uva_mw_cm2              | 37.85 |
| chromatogram | delta     | delta_conc_mg_ml                 | p_uvb_mw_cm2              | 10.35 |
| chromatogram | delta     | delta_conc_mg_ml                 | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| chromatogram | delta     | delta_amount_mg_per_gDW          | p_uva_mw_cm2              | 37.85 |
| chromatogram | delta     | delta_amount_mg_per_gDW          | p_uvb_mw_cm2              | 10.35 |
| chromatogram | delta     | delta_amount_mg_per_gDW          | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| chromatogram | zscore    | z_conc_mg_ml                     | p_uva_mw_cm2              | 37.85 |
| chromatogram | zscore    | z_conc_mg_ml                     | p_uvb_mw_cm2              | 10.35 |
| chromatogram | zscore    | z_conc_mg_ml                     | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| chromatogram | zscore    | z_amount_mg_per_gDW              | p_uva_mw_cm2              | 37.85 |
| chromatogram | zscore    | z_amount_mg_per_gDW              | p_uvb_mw_cm2              | 10.35 |
| chromatogram | zscore    | z_amount_mg_per_gDW              | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| dad          | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2              | 37.85 |
| dad          | raw       | predicted_total_mg_per_gDW       | p_uvb_mw_cm2              | 10.35 |
| dad          | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |
| dad          | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2              | 37.85 |
| dad          | delta     | predicted_total_mg_per_gDW_delta | p_uvb_mw_cm2              | 10.35 |
| dad          | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2:p_uvb_mw_cm2 | 25.13 |

## UVA/UVB Pearson correlation

| dataset      | variant   |   corr_uva_uvb |   n_obs |
|:-------------|:----------|---------------:|--------:|
| chromatogram | raw       |         0.9498 |      90 |
| chromatogram | delta     |         0.9498 |      90 |
| chromatogram | zscore    |         0.9498 |      90 |
| dad          | raw       |         0.9498 |      30 |
| dad          | delta     |         0.9498 |      30 |

## Outlier removal scenarios

| variant   | measurement             | scenario         | removed   |   n_removed |     F |   p_value |
|:----------|:------------------------|:-----------------|:----------|------------:|------:|----------:|
| delta     | delta_conc_mg_ml        | baseline         | nan       |           0 | 3.087 |  0.01316  |
| delta     | delta_conc_mg_ml        | drop_single      | 2B        |           1 | 3.222 |  0.01043  |
| delta     | delta_conc_mg_ml        | drop_single      | 1J        |           1 | 3.213 |  0.0106   |
| delta     | delta_conc_mg_ml        | drop_flagged_set | 2B;1J     |           2 | 3.323 |  0.008782 |
| delta     | delta_amount_mg_per_gDW | baseline         | nan       |           0 | 3.334 |  0.00852  |
| delta     | delta_amount_mg_per_gDW | drop_single      | 2B        |           1 | 3.592 |  0.005452 |
| delta     | delta_amount_mg_per_gDW | drop_flagged_set | 2B        |           1 | 3.592 |  0.005452 |
| zscore    | z_conc_mg_ml            | baseline         | nan       |           0 | 2.55  |  0.03374  |
| zscore    | z_conc_mg_ml            | drop_single      | 2B        |           1 | 2.597 |  0.03115  |
| zscore    | z_conc_mg_ml            | drop_single      | 1J        |           1 | 2.695 |  0.02629  |
| zscore    | z_conc_mg_ml            | drop_flagged_set | 2B;1J     |           2 | 2.701 |  0.02608  |
| zscore    | z_amount_mg_per_gDW     | baseline         | nan       |           0 | 2.457 |  0.03965  |
| zscore    | z_amount_mg_per_gDW     | drop_single      | 2B        |           1 | 2.624 |  0.02974  |
| zscore    | z_amount_mg_per_gDW     | drop_single      | 2J        |           1 | 2.408 |  0.0433   |
| zscore    | z_amount_mg_per_gDW     | drop_flagged_set | 2B;2J     |           2 | 3.254 |  0.009913 |
