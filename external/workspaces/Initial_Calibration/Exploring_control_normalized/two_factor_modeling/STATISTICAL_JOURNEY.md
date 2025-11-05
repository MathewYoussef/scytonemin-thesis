# Two-Factor Modeling Journey

*Purpose:* Provide an auditor-ready chronology of the UVA/UVB modeling work, the statistical procedures applied, why each was chosen, the primary results (with table references), and the evolving interpretation.

---

## 1. Baseline Two-Factor Models

| Item | Details |
| --- | --- |
| **Question** | Does modeling UVA and UVB together reveal dose–response patterns in chromatogram and DAD datasets? |
| **Datasets** | `Chromatogram` (raw, delta, ratio, percent, z-score) and `DAD` (raw + normalized variants). |
| **Methods** | **Two-way ANOVA** (`chromatogram_two_way_anova*.csv`, `dad_two_way_anova*.csv`): treats UVA and UVB doses as categorical factors to test main effects and the interaction via F-tests (assumes normal, homoscedastic residuals).<br>**OLS regression with interaction** (`chromatogram_regression_coeffs*.csv`, `dad_regression_coeffs*.csv`): uses continuous UVA/UVB plus interaction to inspect coefficients, p-values, and R². |
| **Key findings** | Raw chromatogram metrics showed no significant main/interaction effects (all p ≥ 0.13). Normalized chromatogram metrics produced marginal interaction p-values (0.013–0.09) and moderate η² (0.33–0.43). DAD interactions were borderline (p≈0.058, η²≈0.71). Continuous models had low R² (≤0.16) despite positive UVB coefficients for amount metrics. |
| **Diagnostic red flags** | `anova_residual_diagnostics.csv:1-8` indicated non-normal residuals (Shapiro p ≤ 4e-5) and high studentized residuals (|r| up to 4.8). `regression_vif.csv:1-24` and `uva_uvb_correlations.csv:1-5` revealed severe collinearity (ρ≈0.95; VIFs≈38/10/25). |
| **Why we proceeded** | Violated assumptions meant classical ANOVA/OLS results were fragile; further robustness checks were required. |

---

## 2. Robustness Checks (HC3 & Rank ANOVA)

| Item | Details |
| --- | --- |
| **Goal** | Determine if the UVA×UVB signal survives when relaxing normality and equal-variance assumptions. |
| **Methods** | **HC3-robust ANOVA** (`chromatogram_two_way_anova_robust.csv`, `dad_two_way_anova_robust.csv`): same factorial model but with heteroskedasticity-consistent (HC3) standard errors to guard against unequal variances.<br>**Rank-transformed ANOVA** (`chromatogram_two_way_anova_rank.csv`, `dad_two_way_anova_rank.csv`): converts responses to ranks (aligned-rank approach) before ANOVA; provides a non-parametric alternative less sensitive to non-normal errors. |
| **Results** | Chromatogram UV interactions remained significant across normalized variants (p≈0.002–0.011); DAD interactions lost significance (p≈0.25). |
| **Interpretation** | Chromatogram interaction hints persisted beyond Gaussian assumptions, but the effect remained inconsistent and dataset-specific. |

---

## 3. Complementary Continuous Modeling (Ridge, PLS, Simple Permutations)

| Item | Details |
| --- | --- |
| **Goal** | Address multicollinearity and test the interaction using alternative model paradigms. |
| **Methods** | **Ridge regression** (`chromatogram_ridge_results.csv`, `dad_ridge_results.csv`): penalized regression with cross-validated α to stabilize coefficients when predictors are highly correlated.<br>**Partial Least Squares (PLS)** (`chromatogram_pls_results.csv`, `dad_pls_results.csv`): projects UVA/UVB into latent components maximizing covariance with the response, handling collinearity with dimensionality reduction.<br>**Simple permutation ANOVA** (`chromatogram_two_way_anova_permutation.csv`, `dad_two_way_anova_permutation.csv`): permutes the raw response vector to obtain empirical p-values without relying on theoretical F-distributions. |
| **Results** | Ridge kept UVB coefficients positive (e.g., delta amount coefficient ≈1.25) but R² remained modest (≤0.16 chromatogram, ≈0.30 DAD). PLS cross-validated R² were low (≤0.12 chromatogram, ≈0.02 DAD). Simple permutation ANOVA echoed classical results: normalized chromatogram interactions significant (p_perm ≈0.009–0.038); DAD interactions marginal. |
| **Interpretation** | The UVA×UVB hint persisted under multiple modeling angles, but low predictive power and collinearity limited confidence. |

---

## 4. Resampling Sensitivity (Freedman–Lane & Bootstrap Ridge)

| Item | Details |
| --- | --- |
| **Goal** | Stress-test the interaction effect with methods that respect the factorial structure and quantify coefficient stability. |
| **Methods** | **Freedman–Lane permutation test** (`freedman_lane_interaction.csv`): shuffles residuals from the reduced model (without interaction), preserving UVA/UVB main effects when estimating the null distribution for the interaction F-statistic.<br>**Bootstrap ridge (2 000 replicates)** (`ridge_bootstrap_summary.csv`): resamples observations with replacement, refits ridge regression (α chosen via RidgeCV), and records coefficient medians/95 % CI to gauge stability. |
| **Results** | Freedman–Lane p-values were large (p≥0.71 for all chromatogram metrics), suggesting the interaction effect is highly sensitive to assumption-respecting permutations. Bootstrap intervals showed UVB coefficients positive for amount metrics (delta/z-score 95 % CI excludes zero) but interaction coefficients spanned zero across all endpoints. |
| **Interpretation** | Once reduced-model structure is preserved, the interaction evidence weakens substantially; coefficient intervals highlight UVB as the most consistent effect, not the interaction. |

---

## 5. Outlier & Leverage Sensitivity

| Item | Details |
| --- | --- |
| **Goal** | Confirm that high-leverage observations are not solely responsible for the interaction signal. |
| **Method** | `run_outlier_sensitivity.py` identifies observations with |studentized residual| ≥3 (from original OLS) and recomputes ANOVA after dropping each and the combined set. Output: `outlier_sensitivity.csv`. |
| **Results** | Dropping flagged points left interaction p-values below 0.05 and often strengthened them (e.g., z_amount_mg_per_gDW p from 0.040 to 0.010; `outlier_sensitivity.csv:11-14`). |
| **Interpretation** | No single outlier drives the interaction—effects are distributed across multiple observations. |

---

## Current Interpretation (August 2025)

1. **Interaction remains tentative.** Normalized chromatogram metrics show UVA×UVB hints under classical, HC3, rank, and simple permutation tests, yet Freedman–Lane permutations (which preserve UVA/UVB structure) no longer reject the null (p≥0.71). The signal is therefore highly sensitive to analysis choice.
2. **UVB main effect more reliable.** Bootstrap ridge intervals keep UVB coefficients positive for normalized amount metrics; interaction intervals span zero, indicating UVB may drive any observed effect.
3. **DAD evidence weak.** DAD metrics stay marginal or null in every method.
4. **Diagnostics unresolved.** Severe UVA/UVB collinearity (ρ≈0.95; VIF>10) and heavy-tailed chromatogram residuals remain; predictive R² stays low (<0.20). These limit causal interpretation even where p-values are small.

---

## Recommended Next Analytical Steps

- **Dose-grid diagnostics:** Tabulate UVA/UVB cell counts to understand why Freedman–Lane diverges from simple permutation/ANOVA (likely sparsity in the factorial grid).  
- **Variance-stabilizing or hierarchical models:** Apply transformations or mixed-effects/GLM variants that can handle heavy-tailed residuals while retaining UVA/UVB structure.  
- **Deferred (requires new data):** Re-design UV dose grids or collect additional replicates to address collinearity—currently infeasible given lab constraints.

All scripts live in `Exploring_control_normalized/two_factor_modeling/`. Execute them sequentially to reproduce the results referenced above. The companion README summarizes the latest findings; this document provides the historical context and rationale for each method.***
