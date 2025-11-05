# Two-factor Modeling Summary

## Classical ANOVA — Chromatogram

| variant   | measurement             |   UVA_df |   UVA_F |   UVA_p |   UVA_eta2 |   UVB_df |   UVB_F |   UVB_p |   UVB_eta2 |   Interaction_df |   Interaction_F |   Interaction_p |   Interaction_eta2 |
|:----------|:------------------------|---------:|--------:|--------:|-----------:|---------:|--------:|--------:|-----------:|-----------------:|----------------:|----------------:|-------------------:|
| delta     | delta_amount_mg_per_gDW |        5 |  2.622  | 0.1091  |    0.06771 |        5 | 2.632   | 0.1085  |   0.06797  |               25 |           3.334 |         0.00852 |             0.4305 |
| delta     | delta_conc_mg_ml        |        5 |  3.582  | 0.06185 |    0.09113 |        5 | 3.487   | 0.06533 |   0.08872  |               25 |           3.087 |         0.01316 |             0.3927 |
| percent   | pct_amount_mg_per_gDW   |        5 |  0.9634 | 0.3307  |    0.04357 |        5 | 1.099   | 0.2991  |   0.04973  |               25 |           1.849 |         0.1187  |             0.4182 |
| percent   | pct_conc_mg_ml          |        5 |  2.941  | 0.09207 |    0.1071  |        5 | 2.941   | 0.09207 |   0.1071   |               25 |           2.157 |         0.07245 |             0.3927 |
| ratio     | ratio_amount_mg_per_gDW |        5 |  1.55   | 0.2185  |    0.06871 |        5 | 0.9634  | 0.3307  |   0.0427   |               25 |           1.849 |         0.1187  |             0.4099 |
| ratio     | ratio_conc_mg_ml        |        5 |  2.73   | 0.1043  |    0.1119  |        5 | 0.08367 | 0.7735  |   0.003429 |               25 |           2.157 |         0.07245 |             0.442  |
| raw       | amount_mg_per_gDW       |        5 |  1.107  | 0.2957  |    0.03921 |        5 | 1.101   | 0.2971  |   0.03898  |               25 |           1.847 |         0.1125  |             0.327  |
| raw       | conc_mg_ml              |        5 |  2.334  | 0.1303  |    0.07521 |        5 | 2.301   | 0.133   |   0.07415  |               25 |           1.92  |         0.09945 |             0.3094 |
| zscore    | z_amount_mg_per_gDW     |        5 |  1.84   | 0.1786  |    0.05613 |        5 | 1.86    | 0.1762  |   0.05674  |               25 |           2.457 |         0.03965 |             0.3747 |
| zscore    | z_conc_mg_ml            |        5 |  4.946  | 0.02884 |    0.1303  |        5 | 3.47    | 0.06597 |   0.09141  |               25 |           2.55  |         0.03374 |             0.3358 |

## Classical ANOVA — DAD

| variant   | measurement                          |   UVA_df |   UVA_F |   UVA_p |   UVA_eta2 |   UVB_df |   UVB_F |   UVB_p |   UVB_eta2 |   Interaction_df |   Interaction_F |   Interaction_p |   Interaction_eta2 |
|:----------|:-------------------------------------|---------:|--------:|--------:|-----------:|---------:|--------:|--------:|-----------:|-----------------:|----------------:|----------------:|-------------------:|
| delta     | predicted_oxidized_mg_ml_delta       |        5 | 0.2572  |  0.6166 |   0.02137  |        5 | 0.2577  |  0.6164 |   0.0214   |               25 |           1.345 |         0.2797  |             0.5585 |
| delta     | predicted_oxidized_mg_per_gDW_delta  |        5 | 0.3661  |  0.5508 |   0.02644  |        5 | 0.9411  |  0.3417 |   0.06797  |               25 |           1.548 |         0.2128  |             0.5589 |
| delta     | predicted_reduced_mg_ml_delta        |        5 | 0.2027  |  0.6566 |   0.0144   |        5 | 0.933   |  0.3437 |   0.0663   |               25 |           1.627 |         0.191   |             0.5782 |
| delta     | predicted_reduced_mg_per_gDW_delta   |        5 | 2.066   |  0.1635 |   0.14     |        5 | 0.5124  |  0.481  |   0.03473  |               25 |           1.475 |         0.2348  |             0.4999 |
| delta     | predicted_total_mg_ml_delta          |        5 | 2.118   |  0.1586 |   0.1234   |        5 | 0.02471 |  0.8764 |   0.00144  |               25 |           2.043 |         0.1084  |             0.5954 |
| delta     | predicted_total_mg_per_gDW_delta     |        5 | 0.1067  |  0.7468 |   0.006045 |        5 | 0.1717  |  0.6823 |   0.009725 |               25 |           2.515 |         0.05754 |             0.7123 |
| percent   | predicted_oxidized_mg_ml_pct         |        5 | 0.221   |  0.6426 |   0.01834  |        5 | 0.3041  |  0.5864 |   0.02524  |               25 |           1.345 |         0.2797  |             0.5581 |
| percent   | predicted_oxidized_mg_per_gDW_pct    |        5 | 0.9299  |  0.3445 |   0.06453  |        5 | 0.9417  |  0.3415 |   0.06535  |               25 |           1.548 |         0.2128  |             0.537  |
| percent   | predicted_reduced_mg_ml_pct          |        5 | 1.17    |  0.2901 |   0.07285  |        5 | 1.958   |  0.1745 |   0.1219   |               25 |           1.627 |         0.191   |             0.5065 |
| percent   | predicted_reduced_mg_per_gDW_pct     |        5 | 0.8278  |  0.372  |   0.05996  |        5 | 0.8034  |  0.379  |   0.0582   |               25 |           1.475 |         0.2348  |             0.5341 |
| percent   | predicted_total_mg_ml_pct            |        5 | 0.6712  |  0.4207 |   0.04176  |        5 | 0.3863  |  0.5401 |   0.02403  |               25 |           2.043 |         0.1084  |             0.6356 |
| percent   | predicted_total_mg_per_gDW_pct       |        5 | 0.06346 |  0.8033 |   0.00359  |        5 | 0.2407  |  0.6282 |   0.01362  |               25 |           2.515 |         0.05754 |             0.7113 |
| ratio     | predicted_oxidized_mg_ml_ratio       |        5 | 0.9215  |  0.3467 |   0.07053  |        5 | 0.62    |  0.4388 |   0.04745  |               25 |           1.345 |         0.2797  |             0.5146 |
| ratio     | predicted_oxidized_mg_per_gDW_ratio  |        5 | 0.8714  |  0.3599 |   0.06152  |        5 | 0.7547  |  0.3936 |   0.05329  |               25 |           1.548 |         0.2128  |             0.5463 |
| ratio     | predicted_reduced_mg_ml_ratio        |        5 | 2.793   |  0.1077 |   0.1734   |        5 | 0.3792  |  0.5438 |   0.02354  |               25 |           1.627 |         0.191   |             0.5051 |
| ratio     | predicted_reduced_mg_per_gDW_ratio   |        5 | 1.068   |  0.3117 |   0.07516  |        5 | 0.9656  |  0.3356 |   0.06797  |               25 |           1.475 |         0.2348  |             0.519  |
| ratio     | predicted_total_mg_ml_ratio          |        5 | 0.1297  |  0.7219 |   0.008414 |        5 | 0.2631  |  0.6127 |   0.01708  |               25 |           2.043 |         0.1084  |             0.663  |
| ratio     | predicted_total_mg_per_gDW_ratio     |        5 | 0.04205 |  0.8393 |   0.002391 |        5 | 0.1724  |  0.6817 |   0.009804 |               25 |           2.515 |         0.05754 |             0.7149 |
| raw       | predicted_oxidized_mg_ml             |        5 | 0.8544  |  0.3645 |   0.06725  |        5 | 0.3265  |  0.573  |   0.0257   |               25 |           1.345 |         0.2797  |             0.5292 |
| raw       | predicted_oxidized_mg_per_gDW        |        5 | 0.3661  |  0.5508 |   0.02644  |        5 | 0.9411  |  0.3417 |   0.06797  |               25 |           1.548 |         0.2128  |             0.5589 |
| raw       | predicted_reduced_mg_ml              |        5 | 2.226   |  0.1487 |   0.1454   |        5 | 0.1508  |  0.7012 |   0.009848 |               25 |           1.627 |         0.191   |             0.5313 |
| raw       | predicted_reduced_mg_per_gDW         |        5 | 0.9656  |  0.3356 |   0.06983  |        5 | 0.6888  |  0.4148 |   0.04981  |               25 |           1.475 |         0.2348  |             0.5333 |
| raw       | predicted_total_mg_ml                |        5 | 2.118   |  0.1586 |   0.1234   |        5 | 0.02471 |  0.8764 |   0.00144  |               25 |           2.043 |         0.1084  |             0.5954 |
| raw       | predicted_total_mg_per_gDW           |        5 | 0.1067  |  0.7468 |   0.006045 |        5 | 0.1717  |  0.6823 |   0.009725 |               25 |           2.515 |         0.05754 |             0.7123 |
| zscore    | predicted_oxidized_mg_ml_zscore      |        5 | 0.2188  |  0.6442 |   0.01829  |        5 | 0.2204  |  0.6429 |   0.01843  |               25 |           1.345 |         0.2797  |             0.5621 |
| zscore    | predicted_oxidized_mg_per_gDW_zscore |        5 | 0.6288  |  0.4356 |   0.04558  |        5 | 0.6288  |  0.4356 |   0.04558  |               25 |           1.548 |         0.2128  |             0.5609 |
| zscore    | predicted_reduced_mg_ml_zscore       |        5 | 1.925   |  0.1781 |   0.1145   |        5 | 1.946   |  0.1758 |   0.1158   |               25 |           1.627 |         0.191   |             0.4841 |
| zscore    | predicted_reduced_mg_per_gDW_zscore  |        5 | 1.213   |  0.2817 |   0.08308  |        5 | 1.211   |  0.2821 |   0.08293  |               25 |           1.475 |         0.2348  |             0.5052 |
| zscore    | predicted_total_mg_ml_zscore         |        5 | 0.7007  |  0.4108 |   0.04404  |        5 | 0.1916  |  0.6655 |   0.01204  |               25 |           2.043 |         0.1084  |             0.6422 |
| zscore    | predicted_total_mg_per_gDW_zscore    |        5 | 0.1722  |  0.6819 |   0.009763 |        5 | 0.09073 |  0.7658 |   0.005145 |               25 |           2.515 |         0.05754 |             0.7129 |

## HC3-robust ANOVA — Chromatogram

| dataset      | variant   | measurement             | term                            |     sum_sq |   df |       F |   PR(>F) |
|:-------------|:----------|:------------------------|:--------------------------------|-----------:|-----:|--------:|---------:|
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2)                 |   0.01629  |    5 | 1.656   | 0.2016   |
| chromatogram | raw       | conc_mg_ml              | C(p_uvb_mw_cm2)                 |   0.01633  |    5 | 1.661   | 0.201    |
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |   0.1362   |   25 | 2.769   | 0.02299  |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2)                 |   1.885    |    5 | 0.9866  | 0.3234   |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uvb_mw_cm2)                 |   1.566    |    5 | 0.8194  | 0.368    |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  26.72     |   25 | 2.797   | 0.0219   |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2)                 |   0.0106   |    5 | 1.733   | 0.1915   |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uvb_mw_cm2)                 |   0.004176 |    5 | 0.6828  | 0.411    |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |   0.1137   |   25 | 3.717   | 0.004349 |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2)                 |   1.216    |    5 | 1.149   | 0.2869   |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uvb_mw_cm2)                 |   1.216    |    5 | 1.149   | 0.2868   |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  20.22     |   25 | 3.82    | 0.003626 |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2)                 |   0.652    |    5 | 0.05642 | 0.8128   |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uvb_mw_cm2)                 |  26.16     |    5 | 2.264   | 0.1361   |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 225        |   25 | 3.893   | 0.003189 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2)                 |  16.8      |    5 | 1.386   | 0.2423   |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uvb_mw_cm2)                 |  16.86     |    5 | 1.392   | 0.2414   |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 252.1      |   25 | 4.163   | 0.00199  |

## Rank-transformed ANOVA — Chromatogram

| dataset      | variant   | measurement             | term                            |   sum_sq |   df |       F |   PR(>F) |
|:-------------|:----------|:------------------------|:--------------------------------|---------:|-----:|--------:|---------:|
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2)                 |    764.9 |    5 | 0.2371  | 0.6276   |
| chromatogram | raw       | conc_mg_ml              | C(p_uvb_mw_cm2)                 |   1131   |    5 | 0.3506  | 0.5553   |
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  32670   |   25 | 2.025   | 0.08335  |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2)                 |    148.1 |    5 | 0.04672 | 0.8294   |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uvb_mw_cm2)                 |    148   |    5 | 0.04669 | 0.8295   |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  37430   |   25 | 2.362   | 0.04676  |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2)                 |   1943   |    5 | 0.6504  | 0.4222   |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uvb_mw_cm2)                 |   2836   |    5 | 0.9495  | 0.3326   |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  52790   |   25 | 3.535   | 0.00598  |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2)                 |   1463   |    5 | 0.4996  | 0.4816   |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uvb_mw_cm2)                 |    780.8 |    5 | 0.2667  | 0.6069   |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  57740   |   25 | 3.945   | 0.002915 |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2)                 |   3029   |    5 | 0.9977  | 0.3207   |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uvb_mw_cm2)                 |   1144   |    5 | 0.377   | 0.5409   |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  48710   |   25 | 3.209   | 0.01062  |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2)                 |    334.8 |    5 | 0.114   | 0.7365   |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uvb_mw_cm2)                 |    321.1 |    5 | 0.1093  | 0.7418   |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |  56880   |   25 | 3.872   | 0.003313 |

## HC3-robust ANOVA — DAD

| dataset   | variant   | measurement                      | term                            |     sum_sq |   df |         F |   PR(>F) |
|:----------|:----------|:---------------------------------|:--------------------------------|-----------:|-----:|----------:|---------:|
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2)                 |  0.02495   |    5 | 0.01591   |   0.9007 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uvb_mw_cm2)                 |  3.066e-05 |    5 | 1.955e-05 |   0.9965 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 11.25      |   25 | 1.435     |   0.248  |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2)                 |  0.008955  |    5 | 0.005709  |   0.9404 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uvb_mw_cm2)                 |  0.005631  |    5 | 0.00359   |   0.9527 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 11.25      |   25 | 1.435     |   0.248  |

## Rank-transformed ANOVA — DAD

| dataset   | variant   | measurement                      | term                            |   sum_sq |   df |      F |   PR(>F) |
|:----------|:----------|:---------------------------------|:--------------------------------|---------:|-----:|-------:|---------:|
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2)                 |    108.2 |    5 | 0.3052 |   0.5858 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uvb_mw_cm2)                 |    105.1 |    5 | 0.2964 |   0.5912 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |   2730   |   25 | 1.54   |   0.215  |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2)                 |    108.2 |    5 | 0.3052 |   0.5858 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uvb_mw_cm2)                 |    105.1 |    5 | 0.2964 |   0.5912 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) |   2730   |   25 | 1.54   |   0.215  |

## OLS coefficients — Chromatogram (raw metrics)

| dataset      | measurement       | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:-------------|:------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| chromatogram | conc_mg_ml        | Intercept                 |  0.032    | 0.004275 |     0.09903 |         0.0676  |
| chromatogram | conc_mg_ml        | p_uva_mw_cm2              |  0.006811 | 0.7969   |     0.09903 |         0.0676  |
| chromatogram | conc_mg_ml        | p_uvb_mw_cm2              |  0.07346  | 0.1846   |     0.09903 |         0.0676  |
| chromatogram | conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.01649  | 0.5407   |     0.09903 |         0.0676  |
| chromatogram | amount_mg_per_gDW | Intercept                 |  0.4431   | 0.004493 |     0.09709 |         0.06559 |
| chromatogram | amount_mg_per_gDW | p_uva_mw_cm2              |  0.09696  | 0.7925   |     0.09709 |         0.06559 |
| chromatogram | amount_mg_per_gDW | p_uvb_mw_cm2              |  1.353    | 0.08053  |     0.09709 |         0.06559 |
| chromatogram | amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.3703   | 0.3248   |     0.09709 |         0.06559 |

## OLS coefficients — Chromatogram (delta)

| dataset      | measurement             | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:-------------|:------------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| chromatogram | delta_conc_mg_ml        | Intercept                 |  0.01516  |  0.0818  |      0.1499 |          0.1202 |
| chromatogram | delta_conc_mg_ml        | p_uva_mw_cm2              |  0.006811 |  0.7445  |      0.1499 |          0.1202 |
| chromatogram | delta_conc_mg_ml        | p_uvb_mw_cm2              |  0.07346  |  0.09395 |      0.1499 |          0.1202 |
| chromatogram | delta_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.01649  |  0.4388  |      0.1499 |          0.1202 |
| chromatogram | delta_amount_mg_per_gDW | Intercept                 |  0.1323   |  0.2455  |      0.1623 |          0.1331 |
| chromatogram | delta_amount_mg_per_gDW | p_uva_mw_cm2              |  0.09696  |  0.724   |      0.1623 |          0.1331 |
| chromatogram | delta_amount_mg_per_gDW | p_uvb_mw_cm2              |  1.353    |  0.01982 |      0.1623 |          0.1331 |
| chromatogram | delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.3703   |  0.1872  |      0.1623 |          0.1331 |

## OLS coefficients — Chromatogram (ratio)

| dataset      | measurement             | term                      |    coef |   pvalue |   r_squared |   adj_r_squared |
|:-------------|:------------------------|:--------------------------|--------:|---------:|------------:|----------------:|
| chromatogram | ratio_conc_mg_ml        | Intercept                 |  2.058  | 0.006081 |      0.1595 |         0.1144  |
| chromatogram | ratio_conc_mg_ml        | p_uva_mw_cm2              |  0.4726 | 0.7877   |      0.1595 |         0.1144  |
| chromatogram | ratio_conc_mg_ml        | p_uvb_mw_cm2              |  5.191  | 0.159    |      0.1595 |         0.1144  |
| chromatogram | ratio_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.163  | 0.5155   |      0.1595 |         0.1144  |
| chromatogram | ratio_amount_mg_per_gDW | Intercept                 |  1.628  | 0.01059  |      0.1433 |         0.09738 |
| chromatogram | ratio_amount_mg_per_gDW | p_uva_mw_cm2              |  0.4403 | 0.7686   |      0.1433 |         0.09738 |
| chromatogram | ratio_amount_mg_per_gDW | p_uvb_mw_cm2              |  5.45   | 0.08424  |      0.1433 |         0.09738 |
| chromatogram | ratio_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.545  | 0.3124   |      0.1433 |         0.09738 |

## OLS coefficients — Chromatogram (percent)

| dataset      | measurement           | term                      |    coef |   pvalue |   r_squared |   adj_r_squared |
|:-------------|:----------------------|:--------------------------|--------:|---------:|------------:|----------------:|
| chromatogram | pct_conc_mg_ml        | Intercept                 |  105.8  |  0.1482  |      0.1595 |         0.1144  |
| chromatogram | pct_conc_mg_ml        | p_uva_mw_cm2              |   47.26 |  0.7877  |      0.1595 |         0.1144  |
| chromatogram | pct_conc_mg_ml        | p_uvb_mw_cm2              |  519.1  |  0.159   |      0.1595 |         0.1144  |
| chromatogram | pct_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -116.3  |  0.5155  |      0.1595 |         0.1144  |
| chromatogram | pct_amount_mg_per_gDW | Intercept                 |   62.76 |  0.3122  |      0.1433 |         0.09738 |
| chromatogram | pct_amount_mg_per_gDW | p_uva_mw_cm2              |   44.03 |  0.7686  |      0.1433 |         0.09738 |
| chromatogram | pct_amount_mg_per_gDW | p_uvb_mw_cm2              |  545    |  0.08424 |      0.1433 |         0.09738 |
| chromatogram | pct_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -154.5  |  0.3124  |      0.1433 |         0.09738 |

## OLS coefficients — Chromatogram (z-score)

| dataset      | measurement         | term                      |     coef |   pvalue |   r_squared |   adj_r_squared |
|:-------------|:--------------------|:--------------------------|---------:|---------:|------------:|----------------:|
| chromatogram | z_conc_mg_ml        | Intercept                 |  0.08328 |  0.8245  |      0.1255 |         0.09499 |
| chromatogram | z_conc_mg_ml        | p_uva_mw_cm2              |  0.1664  |  0.8547  |      0.1255 |         0.09499 |
| chromatogram | z_conc_mg_ml        | p_uvb_mw_cm2              |  2.94    |  0.1227  |      0.1255 |         0.09499 |
| chromatogram | z_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.5328  |  0.5648  |      0.1255 |         0.09499 |
| chromatogram | z_amount_mg_per_gDW | Intercept                 |  0.03063 |  0.9364  |      0.1243 |         0.09379 |
| chromatogram | z_amount_mg_per_gDW | p_uva_mw_cm2              |  0.1584  |  0.8646  |      0.1243 |         0.09379 |
| chromatogram | z_amount_mg_per_gDW | p_uvb_mw_cm2              |  3.987   |  0.04159 |      0.1243 |         0.09379 |
| chromatogram | z_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.9283  |  0.3272  |      0.1243 |         0.09379 |

## OLS coefficients — DAD (raw)

| dataset   | measurement                   | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:----------|:------------------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| dad       | predicted_total_mg_ml         | Intercept                 |  0.04977  | 0.0111   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml         | p_uva_mw_cm2              |  0.05286  | 0.2409   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml         | p_uvb_mw_cm2              |  0.08044  | 0.3884   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml         | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.06707  | 0.1466   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_per_gDW    | Intercept                 |  0.6698   | 0.007945 |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW    | p_uva_mw_cm2              |  0.7349   | 0.2037   |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW    | p_uvb_mw_cm2              |  1.746    | 0.1489   |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW    | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.186    | 0.04878  |      0.335  |         0.2583  |
| dad       | predicted_oxidized_mg_ml      | Intercept                 |  0.0058   | 0.733    |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml      | p_uva_mw_cm2              |  0.07     | 0.09734  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml      | p_uvb_mw_cm2              |  0.002391 | 0.9777   |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml      | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.07605  | 0.07779  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_per_gDW | Intercept                 |  0.03454  | 0.884    |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW | p_uva_mw_cm2              |  0.9713   | 0.0987   |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW | p_uvb_mw_cm2              |  0.4144   | 0.7285   |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.189    | 0.04944  |      0.2074 |         0.1159  |
| dad       | predicted_reduced_mg_ml       | Intercept                 |  0.07799  | 0.01444  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml       | p_uva_mw_cm2              |  0.02813  | 0.6991   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml       | p_uvb_mw_cm2              |  0.09151  | 0.5468   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml       | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.02584  | 0.7271   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_per_gDW  | Intercept                 |  1.107    | 0.007649 |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW  | p_uva_mw_cm2              |  0.3513   | 0.7075   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW  | p_uvb_mw_cm2              |  2.022    | 0.3039   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW  | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.6403   | 0.5029   |      0.2023 |         0.1103  |

## OLS coefficients — DAD (delta)

| dataset   | measurement                         | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:----------|:------------------------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| dad       | predicted_total_mg_ml_delta         | Intercept                 |  0.03617  |  0.05753 |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_delta         | p_uva_mw_cm2              |  0.05286  |  0.2409  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_delta         | p_uvb_mw_cm2              |  0.08044  |  0.3884  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_delta         | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.06707  |  0.1466  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_per_gDW_delta    | Intercept                 |  0.4932   |  0.04395 |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_delta    | p_uva_mw_cm2              |  0.7349   |  0.2037  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_delta    | p_uvb_mw_cm2              |  1.746    |  0.1489  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_delta    | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.186    |  0.04878 |      0.335  |         0.2583  |
| dad       | predicted_oxidized_mg_ml_delta      | Intercept                 |  0.02924  |  0.09393 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_delta      | p_uva_mw_cm2              |  0.07     |  0.09734 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_delta      | p_uvb_mw_cm2              |  0.002391 |  0.9777  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_delta      | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.07605  |  0.07779 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_per_gDW_delta | Intercept                 |  0.339    |  0.16    |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_delta | p_uva_mw_cm2              |  0.9713   |  0.0987  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_delta | p_uvb_mw_cm2              |  0.4144   |  0.7285  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_delta | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.189    |  0.04944 |      0.2074 |         0.1159  |
| dad       | predicted_reduced_mg_ml_delta       | Intercept                 |  0.01265  |  0.6741  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_delta       | p_uva_mw_cm2              |  0.02813  |  0.6991  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_delta       | p_uvb_mw_cm2              |  0.09151  |  0.5468  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_delta       | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.02584  |  0.7271  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_per_gDW_delta  | Intercept                 |  0.002445 |  0.995   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_delta  | p_uva_mw_cm2              |  0.3513   |  0.7075  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_delta  | p_uvb_mw_cm2              |  2.022    |  0.3039  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_delta  | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.6403   |  0.5029  |      0.2023 |         0.1103  |

## OLS coefficients — DAD (ratio)

| dataset   | measurement                         | term                      |    coef |   pvalue |   r_squared |   adj_r_squared |
|:----------|:------------------------------------|:--------------------------|--------:|---------:|------------:|----------------:|
| dad       | predicted_total_mg_ml_ratio         | Intercept                 |  3.66   | 0.0111   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_ratio         | p_uva_mw_cm2              |  3.888  | 0.2409   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_ratio         | p_uvb_mw_cm2              |  5.916  | 0.3884   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_ratio         | p_uva_mw_cm2:p_uvb_mw_cm2 | -4.933  | 0.1466   |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_per_gDW_ratio    | Intercept                 |  3.793  | 0.007945 |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_ratio    | p_uva_mw_cm2              |  4.162  | 0.2037   |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_ratio    | p_uvb_mw_cm2              |  9.886  | 0.1489   |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_ratio    | p_uva_mw_cm2:p_uvb_mw_cm2 | -6.716  | 0.04878  |      0.335  |         0.2583  |
| dad       | predicted_oxidized_mg_ml_ratio      | Intercept                 | -0.2474 | 0.733    |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_ratio      | p_uva_mw_cm2              | -2.986  | 0.09734  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_ratio      | p_uvb_mw_cm2              | -0.102  | 0.9777   |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_ratio      | p_uva_mw_cm2:p_uvb_mw_cm2 |  3.244  | 0.07779  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_per_gDW_ratio | Intercept                 | -0.1134 | 0.884    |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_ratio | p_uva_mw_cm2              | -3.19   | 0.0987   |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_ratio | p_uvb_mw_cm2              | -1.361  | 0.7285   |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_ratio | p_uva_mw_cm2:p_uvb_mw_cm2 |  3.907  | 0.04944  |      0.2074 |         0.1159  |
| dad       | predicted_reduced_mg_ml_ratio       | Intercept                 |  1.194  | 0.01444  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_ratio       | p_uva_mw_cm2              |  0.4306 | 0.6991   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_ratio       | p_uvb_mw_cm2              |  1.401  | 0.5468   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_ratio       | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.3955 | 0.7271   |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_per_gDW_ratio  | Intercept                 |  1.002  | 0.007649 |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_ratio  | p_uva_mw_cm2              |  0.3181 | 0.7075   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_ratio  | p_uvb_mw_cm2              |  1.832  | 0.3039   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_ratio  | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.5799 | 0.5029   |      0.2023 |         0.1103  |

## OLS coefficients — DAD (percent)

| dataset   | measurement                       | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:----------|:----------------------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| dad       | predicted_total_mg_ml_pct         | Intercept                 |  266      |  0.05753 |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_pct         | p_uva_mw_cm2              |  388.8    |  0.2409  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_pct         | p_uvb_mw_cm2              |  591.6    |  0.3884  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_pct         | p_uva_mw_cm2:p_uvb_mw_cm2 | -493.3    |  0.1466  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_per_gDW_pct    | Intercept                 |  279.3    |  0.04395 |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_pct    | p_uva_mw_cm2              |  416.2    |  0.2037  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_pct    | p_uvb_mw_cm2              |  988.6    |  0.1489  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_pct    | p_uva_mw_cm2:p_uvb_mw_cm2 | -671.6    |  0.04878 |      0.335  |         0.2583  |
| dad       | predicted_oxidized_mg_ml_pct      | Intercept                 | -124.7    |  0.09393 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_pct      | p_uva_mw_cm2              | -298.6    |  0.09734 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_pct      | p_uvb_mw_cm2              |  -10.2    |  0.9777  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_pct      | p_uva_mw_cm2:p_uvb_mw_cm2 |  324.4    |  0.07779 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_per_gDW_pct | Intercept                 | -111.3    |  0.16    |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_pct | p_uva_mw_cm2              | -319      |  0.0987  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_pct | p_uvb_mw_cm2              | -136.1    |  0.7285  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_pct | p_uva_mw_cm2:p_uvb_mw_cm2 |  390.7    |  0.04944 |      0.2074 |         0.1159  |
| dad       | predicted_reduced_mg_ml_pct       | Intercept                 |   19.37   |  0.6741  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_pct       | p_uva_mw_cm2              |   43.06   |  0.6991  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_pct       | p_uvb_mw_cm2              |  140.1    |  0.5468  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_pct       | p_uva_mw_cm2:p_uvb_mw_cm2 |  -39.55   |  0.7271  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_per_gDW_pct  | Intercept                 |    0.2214 |  0.995   |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_pct  | p_uva_mw_cm2              |   31.81   |  0.7075  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_pct  | p_uvb_mw_cm2              |  183.2    |  0.3039  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_pct  | p_uva_mw_cm2:p_uvb_mw_cm2 |  -57.99   |  0.5029  |      0.2023 |         0.1103  |

## OLS coefficients — DAD (z-score)

| dataset   | measurement                          | term                      |      coef |   pvalue |   r_squared |   adj_r_squared |
|:----------|:-------------------------------------|:--------------------------|----------:|---------:|------------:|----------------:|
| dad       | predicted_total_mg_ml_zscore         | Intercept                 |  0.04354  |  0.8897  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_zscore         | p_uva_mw_cm2              |  0.9031   |  0.2409  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_zscore         | p_uvb_mw_cm2              |  1.374    |  0.3884  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_ml_zscore         | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.146    |  0.1466  |      0.2919 |         0.2102  |
| dad       | predicted_total_mg_per_gDW_zscore    | Intercept                 |  0.007769 |  0.9787  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_zscore    | p_uva_mw_cm2              |  0.9096   |  0.2037  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_zscore    | p_uvb_mw_cm2              |  2.161    |  0.1489  |      0.335  |         0.2583  |
| dad       | predicted_total_mg_per_gDW_zscore    | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.468    |  0.04878 |      0.335  |         0.2583  |
| dad       | predicted_oxidized_mg_ml_zscore      | Intercept                 | -0.07836  |  0.7909  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_zscore      | p_uva_mw_cm2              |  1.217    |  0.09734 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_zscore      | p_uvb_mw_cm2              |  0.04158  |  0.9777  |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_ml_zscore      | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.323    |  0.07779 |      0.1912 |         0.09787 |
| dad       | predicted_oxidized_mg_per_gDW_zscore | Intercept                 | -0.08523  |  0.762   |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_zscore | p_uva_mw_cm2              |  1.154    |  0.0987  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_zscore | p_uvb_mw_cm2              |  0.4924   |  0.7285  |      0.2074 |         0.1159  |
| dad       | predicted_oxidized_mg_per_gDW_zscore | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.413    |  0.04944 |      0.2074 |         0.1159  |
| dad       | predicted_reduced_mg_ml_zscore       | Intercept                 |  0.3178   |  0.7425  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_zscore       | p_uva_mw_cm2              |  0.9051   |  0.6991  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_zscore       | p_uvb_mw_cm2              |  2.944    |  0.5468  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_ml_zscore       | p_uva_mw_cm2:p_uvb_mw_cm2 | -0.8314   |  0.7271  |      0.212  |         0.121   |
| dad       | predicted_reduced_mg_per_gDW_zscore  | Intercept                 |  0.225    |  0.8079  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_zscore  | p_uva_mw_cm2              |  0.8408   |  0.7075  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_zscore  | p_uvb_mw_cm2              |  4.841    |  0.3039  |      0.2023 |         0.1103  |
| dad       | predicted_reduced_mg_per_gDW_zscore  | p_uva_mw_cm2:p_uvb_mw_cm2 | -1.533    |  0.5029  |      0.2023 |         0.1103  |

## Ridge regression (chromatogram)

| dataset      | variant   | measurement             | term                      |   alpha |   coef_scaled |   coef_orig |   intercept_orig |   r_squared |
|:-------------|:----------|:------------------------|:--------------------------|--------:|--------------:|------------:|-----------------:|------------:|
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2              |      10 |      0.003122 |    0.002892 |          0.03845 |     0.09172 |
| chromatogram | raw       | conc_mg_ml              | p_uvb_mw_cm2              |      10 |      0.01132  |    0.04175  |          0.03845 |     0.09172 |
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2:p_uvb_mw_cm2 |      10 |     -0.001952 |   -0.002258 |          0.03845 |     0.09172 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2              |      10 |      0.02611  |    0.02418  |          0.5666  |     0.08196 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uvb_mw_cm2              |      10 |      0.1933   |    0.7127   |          0.5666  |     0.08196 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |      10 |     -0.07455  |   -0.08624  |          0.5666  |     0.08196 |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2              |      10 |      0.003122 |    0.002892 |          0.0216  |     0.1388  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uvb_mw_cm2              |      10 |      0.01132  |    0.04175  |          0.0216  |     0.1388  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 |      10 |     -0.001952 |   -0.002258 |          0.0216  |     0.1388  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2              |       1 |      0.05445  |    0.05044  |          0.1627  |     0.1609  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uvb_mw_cm2              |       1 |      0.3397   |    1.253    |          0.1627  |     0.1609  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 |       1 |     -0.2431   |   -0.2813   |          0.1627  |     0.1609  |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2              |      10 |      0.1109   |    0.1027   |          0.3162  |     0.1171  |
| chromatogram | zscore    | z_conc_mg_ml            | p_uvb_mw_cm2              |      10 |      0.4429   |    1.633    |          0.3162  |     0.1171  |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2:p_uvb_mw_cm2 |      10 |     -0.05943  |   -0.06875  |          0.3162  |     0.1171  |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2              |      10 |      0.06299  |    0.05835  |          0.3635  |     0.1065  |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uvb_mw_cm2              |      10 |      0.5588   |    2.06     |          0.3635  |     0.1065  |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2:p_uvb_mw_cm2 |      10 |     -0.1927   |   -0.223    |          0.3635  |     0.1065  |

## Ridge regression (DAD)

| dataset   | variant   | measurement                      | term                      |   alpha |   coef_scaled |   coef_orig |   intercept_orig |   r_squared |
|:----------|:----------|:---------------------------------|:--------------------------|--------:|--------------:|------------:|-----------------:|------------:|
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2              |       1 |        0.2512 |      0.2327 |           0.8515 |      0.2995 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uvb_mw_cm2              |       1 |        0.4446 |      1.64   |           0.8515 |      0.2995 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |       1 |       -0.4524 |     -0.5233 |           0.8515 |      0.2995 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2              |       1 |        0.2512 |      0.2327 |           0.6749 |      0.2995 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uvb_mw_cm2              |       1 |        0.4446 |      1.64   |           0.6749 |      0.2995 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2:p_uvb_mw_cm2 |       1 |       -0.4524 |     -0.5233 |           0.6749 |      0.2995 |

## PLS regression (chromatogram)

| dataset      | variant   | measurement             | term                      |   component |   n_components |   x_weight |   x_loading |   y_loading |   r_squared |   cv_r_squared |
|:-------------|:----------|:------------------------|:--------------------------|------------:|---------------:|-----------:|------------:|------------:|------------:|---------------:|
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2              |           1 |              2 |     0.5684 |      0.5843 |      0.1661 |     0.09851 |        0.07159 |
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2              |           2 |              2 |    -0.1946 |     -0.2543 |      0.4824 |     0.09851 |        0.07159 |
| chromatogram | raw       | conc_mg_ml              | p_uvb_mw_cm2              |           1 |              2 |     0.6345 |      0.5753 |      0.1661 |     0.09851 |        0.07159 |
| chromatogram | raw       | conc_mg_ml              | p_uvb_mw_cm2              |           2 |              2 |     0.7222 |      0.7428 |      0.4824 |     0.09851 |        0.07159 |
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.5238 |      0.5782 |      0.1661 |     0.09851 |        0.07159 |
| chromatogram | raw       | conc_mg_ml              | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    -0.6637 |     -0.6238 |      0.4824 |     0.09851 |        0.07159 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2              |           1 |              2 |     0.5546 |      0.588  |      0.1407 |     0.09608 |        0.06354 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2              |           2 |              2 |     0.2325 |      0.2895 |     -0.6942 |     0.09608 |        0.06354 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uvb_mw_cm2              |           1 |              2 |     0.6784 |      0.5806 |      0.1407 |     0.09608 |        0.06354 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uvb_mw_cm2              |           2 |              2 |    -0.6823 |     -0.7017 |     -0.6942 |     0.09608 |        0.06354 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.4818 |      0.5811 |      0.1407 |     0.09608 |        0.06354 |
| chromatogram | raw       | amount_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |     0.6932 |      0.6549 |     -0.6942 |     0.09608 |        0.06354 |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2              |           1 |              2 |     0.5684 |      0.5843 |      0.2044 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2              |           2 |              2 |    -0.1946 |     -0.2543 |      0.5935 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uvb_mw_cm2              |           1 |              2 |     0.6345 |      0.5753 |      0.2044 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uvb_mw_cm2              |           2 |              2 |     0.7222 |      0.7428 |      0.5935 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.5238 |      0.5782 |      0.2044 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_conc_mg_ml        | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    -0.6637 |     -0.6238 |      0.5935 |     0.1491  |        0.1118  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2              |           1 |              2 |     0.5546 |      0.588  |      0.182  |     0.1606  |        0.1013  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2              |           2 |              2 |     0.2325 |      0.2895 |     -0.8976 |     0.1606  |        0.1013  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uvb_mw_cm2              |           1 |              2 |     0.6784 |      0.5806 |      0.182  |     0.1606  |        0.1013  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uvb_mw_cm2              |           2 |              2 |    -0.6823 |     -0.7017 |     -0.8976 |     0.1606  |        0.1013  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.4818 |      0.5811 |      0.182  |     0.1606  |        0.1013  |
| chromatogram | delta     | delta_amount_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |     0.6932 |      0.6549 |     -0.8976 |     0.1606  |        0.1013  |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2              |           1 |              2 |     0.5668 |      0.5842 |      0.1885 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2              |           2 |              2 |    -0.2193 |     -0.2577 |      0.5277 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_conc_mg_ml            | p_uvb_mw_cm2              |           1 |              2 |     0.6332 |      0.5751 |      0.1885 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_conc_mg_ml            | p_uvb_mw_cm2              |           2 |              2 |     0.7326 |      0.7447 |      0.5277 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.527  |      0.5781 |      0.1885 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_conc_mg_ml            | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    -0.6443 |     -0.6175 |      0.5277 |     0.1252  |        0.09556 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2              |           1 |              2 |     0.5531 |      0.5875 |      0.1625 |     0.1237  |        0.08456 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2              |           2 |              2 |    -0.2505 |     -0.2901 |      0.7664 |     0.1237  |        0.08456 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uvb_mw_cm2              |           1 |              2 |     0.6751 |      0.58   |      0.1625 |     0.1237  |        0.08456 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uvb_mw_cm2              |           2 |              2 |     0.6936 |      0.7061 |      0.7664 |     0.1237  |        0.08456 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |     0.4882 |      0.5808 |      0.1625 |     0.1237  |        0.08456 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    -0.6754 |     -0.6478 |      0.7664 |     0.1237  |        0.08456 |

## PLS regression (DAD)

| dataset   | variant   | measurement                      | term                      |   component |   n_components |   x_weight |   x_loading |   y_loading |   r_squared |   cv_r_squared |
|:----------|:----------|:---------------------------------|:--------------------------|------------:|---------------:|-----------:|------------:|------------:|------------:|---------------:|
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2              |           1 |              2 |    0.5767  |      0.5893 |      0.2366 |      0.3011 |        0.01946 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2              |           2 |              2 |    0.07999 |      0.2685 |     -1.353  |      0.3011 |        0.01946 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uvb_mw_cm2              |           1 |              2 |    0.6771  |      0.582  |      0.2366 |      0.3011 |        0.01946 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uvb_mw_cm2              |           2 |              2 |   -0.6036  |     -0.7011 |     -1.353  |      0.3011 |        0.01946 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |    0.4571  |      0.5821 |      0.2366 |      0.3011 |        0.01946 |
| dad       | raw       | predicted_total_mg_per_gDW       | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    0.7933  |      0.7    |     -1.353  |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2              |           1 |              2 |    0.5767  |      0.5893 |      0.2366 |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2              |           2 |              2 |    0.07999 |      0.2685 |     -1.353  |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uvb_mw_cm2              |           1 |              2 |    0.6771  |      0.582  |      0.2366 |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uvb_mw_cm2              |           2 |              2 |   -0.6036  |     -0.7011 |     -1.353  |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2:p_uvb_mw_cm2 |           1 |              2 |    0.4571  |      0.5821 |      0.2366 |      0.3011 |        0.01946 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | p_uva_mw_cm2:p_uvb_mw_cm2 |           2 |              2 |    0.7933  |      0.7    |     -1.353  |      0.3011 |        0.01946 |

## Bootstrap ridge coefficient intervals

| dataset      | variant   | measurement                | term                      |   coef_median |   coef_mean |   coef_std |   coef_p2_5 |   coef_p97_5 |   alpha |
|:-------------|:----------|:---------------------------|:--------------------------|--------------:|------------:|-----------:|------------:|-------------:|--------:|
| chromatogram | delta     | delta_conc_mg_ml           | p_uva_mw_cm2              |      0.005117 |     0.0043  |    0.01533 |   -0.02773  |      0.03246 |     1   |
| chromatogram | delta     | delta_conc_mg_ml           | p_uva_mw_cm2:p_uvb_mw_cm2 |     -0.0121   |    -0.01176 |    0.01323 |   -0.03516  |      0.01597 |     1   |
| chromatogram | delta     | delta_conc_mg_ml           | p_uvb_mw_cm2              |      0.06805  |     0.06826 |    0.04054 |   -0.008158 |      0.1467  |     1   |
| chromatogram | delta     | delta_amount_mg_per_gDW    | p_uva_mw_cm2              |      0.1027   |     0.09251 |    0.2987  |   -0.5118   |      0.6319  |     0.1 |
| chromatogram | delta     | delta_amount_mg_per_gDW    | p_uva_mw_cm2:p_uvb_mw_cm2 |     -0.3754   |    -0.3613  |    0.2665  |   -0.848    |      0.1987  |     0.1 |
| chromatogram | delta     | delta_amount_mg_per_gDW    | p_uvb_mw_cm2              |      1.372    |     1.347   |    0.6481  |    0.05489  |      2.623   |     0.1 |
| chromatogram | zscore    | z_conc_mg_ml               | p_uva_mw_cm2              |      0.1356   |     0.1252  |    0.6596  |   -1.257    |      1.35    |     1   |
| chromatogram | zscore    | z_conc_mg_ml               | p_uva_mw_cm2:p_uvb_mw_cm2 |     -0.4194   |    -0.394   |    0.5555  |   -1.419    |      0.7641  |     1   |
| chromatogram | zscore    | z_conc_mg_ml               | p_uvb_mw_cm2              |      2.663    |     2.63    |    1.717   |   -0.7736   |      6.008   |     1   |
| chromatogram | zscore    | z_amount_mg_per_gDW        | p_uva_mw_cm2              |      0.1043   |     0.07283 |    0.6581  |   -1.257    |      1.309   |     1   |
| chromatogram | zscore    | z_amount_mg_per_gDW        | p_uva_mw_cm2:p_uvb_mw_cm2 |     -0.7302   |    -0.7102  |    0.5745  |   -1.748    |      0.4845  |     1   |
| chromatogram | zscore    | z_amount_mg_per_gDW        | p_uvb_mw_cm2              |      3.604    |     3.62    |    1.768   |    0.1742   |      7.144   |     1   |
| dad          | raw       | predicted_total_mg_per_gDW | p_uva_mw_cm2              |      0.607    |     0.6     |    0.5045  |   -0.3865   |      1.561   |     0.1 |
| dad          | raw       | predicted_total_mg_per_gDW | p_uva_mw_cm2:p_uvb_mw_cm2 |     -1.042    |    -1.031   |    0.5065  |   -1.963    |      0.02814 |     0.1 |
| dad          | raw       | predicted_total_mg_per_gDW | p_uvb_mw_cm2              |      1.773    |     1.794   |    1.228   |   -0.6461   |      4.179   |     0.1 |

## Simple permutation ANOVA (chromatogram)

| dataset      | variant   | measurement             | term                            | F_obs   |    p_perm |   n_permutations |
|:-------------|:----------|:------------------------|:--------------------------------|:--------|----------:|-----------------:|
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2)                 | 2.334   | 0.1274    |             2000 |
| chromatogram | raw       | conc_mg_ml              | C(p_uvb_mw_cm2)                 | 2.301   | 0.1329    |             2000 |
| chromatogram | raw       | conc_mg_ml              | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 1.92    | 0.09895   |             2000 |
| chromatogram | raw       | conc_mg_ml              | Residual                        |         | 0.0004998 |             2000 |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2)                 | 1.107   | 0.3033    |             2000 |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uvb_mw_cm2)                 | 1.101   | 0.3048    |             2000 |
| chromatogram | raw       | amount_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 1.847   | 0.1194    |             2000 |
| chromatogram | raw       | amount_mg_per_gDW       | Residual                        |         | 0.0004998 |             2000 |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2)                 | 3.582   | 0.05647   |             2000 |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uvb_mw_cm2)                 | 3.487   | 0.05647   |             2000 |
| chromatogram | delta     | delta_conc_mg_ml        | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 3.087   | 0.01199   |             2000 |
| chromatogram | delta     | delta_conc_mg_ml        | Residual                        |         | 0.0004998 |             2000 |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2)                 | 2.622   | 0.1074    |             2000 |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uvb_mw_cm2)                 | 2.632   | 0.1059    |             2000 |
| chromatogram | delta     | delta_amount_mg_per_gDW | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 3.334   | 0.009495  |             2000 |
| chromatogram | delta     | delta_amount_mg_per_gDW | Residual                        |         | 0.0004998 |             2000 |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2)                 | 4.946   | 0.02899   |             2000 |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uvb_mw_cm2)                 | 3.47    | 0.06347   |             2000 |
| chromatogram | zscore    | z_conc_mg_ml            | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 2.55    | 0.03148   |             2000 |
| chromatogram | zscore    | z_conc_mg_ml            | Residual                        |         | 0.0004998 |             2000 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2)                 | 1.84    | 0.1734    |             2000 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uvb_mw_cm2)                 | 1.86    | 0.1699    |             2000 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 2.457   | 0.03848   |             2000 |
| chromatogram | zscore    | z_amount_mg_per_gDW     | Residual                        |         | 0.0004998 |             2000 |

## Simple permutation ANOVA (DAD)

| dataset   | variant   | measurement                      | term                            | F_obs   |    p_perm |   n_permutations |
|:----------|:----------|:---------------------------------|:--------------------------------|:--------|----------:|-----------------:|
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2)                 | 0.1067  | 0.7521    |             2000 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uvb_mw_cm2)                 | 0.1717  | 0.6952    |             2000 |
| dad       | raw       | predicted_total_mg_per_gDW       | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 2.515   | 0.06247   |             2000 |
| dad       | raw       | predicted_total_mg_per_gDW       | Residual                        |         | 0.0004998 |             2000 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2)                 | 0.1067  | 0.7496    |             2000 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uvb_mw_cm2)                 | 0.1717  | 0.6962    |             2000 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | C(p_uva_mw_cm2):C(p_uvb_mw_cm2) | 2.515   | 0.06247   |             2000 |
| dad       | delta     | predicted_total_mg_per_gDW_delta | Residual                        |         | 0.0004998 |             2000 |

## Freedman–Lane permutation test

| dataset      | variant   | measurement                |   F_obs |   p_freedman_lane |   n_permutations |
|:-------------|:----------|:---------------------------|--------:|------------------:|-----------------:|
| chromatogram | delta     | delta_conc_mg_ml           |   3.087 |            0.7271 |             2000 |
| chromatogram | delta     | delta_amount_mg_per_gDW    |   3.334 |            0.7136 |             2000 |
| chromatogram | zscore    | z_conc_mg_ml               |   2.55  |            0.7406 |             2000 |
| chromatogram | zscore    | z_amount_mg_per_gDW        |   2.457 |            0.7366 |             2000 |
| dad          | raw       | predicted_total_mg_per_gDW |   2.515 |            0.7871 |             2000 |
