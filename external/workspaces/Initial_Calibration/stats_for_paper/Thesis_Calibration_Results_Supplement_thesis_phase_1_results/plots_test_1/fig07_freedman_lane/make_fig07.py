import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

sns.set(style='whitegrid')
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]
np.random.seed(42)
N = 2000

cases = [
    ('Chrom Δ concentration', project_root / 'chromatogram_delta.csv', 'delta_conc_mg_ml'),
    ('Chrom Δ amount', project_root / 'chromatogram_delta.csv', 'delta_amount_mg_per_gDW'),
    ('Chrom z concentration', project_root / 'chromatogram_zscores.csv', 'z_conc_mg_ml'),
    ('Chrom z amount', project_root / 'chromatogram_zscores.csv', 'z_amount_mg_per_gDW'),
    ('DAD total mg·gDW⁻¹', project_root / 'DAD_derived_concentrations_corrected.csv', 'predicted_total_mg_per_gDW')
]

fig, axes = plt.subplots(len(cases), 1, figsize=(10, 15), sharex=False)
axes = np.atleast_1d(axes)

for ax, (label, path, response) in zip(axes, cases):
    df = pd.read_csv(path)[['p_uva_mw_cm2', 'p_uvb_mw_cm2', response]].dropna()
    df['p_uva_mw_cm2'] = df['p_uva_mw_cm2'].astype('category')
    df['p_uvb_mw_cm2'] = df['p_uvb_mw_cm2'].astype('category')

    formula_reduced = f"{response} ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2)"
    formula_full = f"{response} ~ C(p_uva_mw_cm2) * C(p_uvb_mw_cm2)"
    reduced = smf.ols(formula_reduced, data=df).fit()
    full = smf.ols(formula_full, data=df).fit()
    F_obs = anova_lm(full, typ=2).loc['C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', 'F']

    fit_red = reduced.fittedvalues
    resid = reduced.resid.values
    F_perm = []
    for _ in range(N):
        perm_res = np.random.permutation(resid)
        df['_y_star'] = fit_red + perm_res
        full_perm = smf.ols("_y_star ~ C(p_uva_mw_cm2) * C(p_uvb_mw_cm2)", data=df).fit()
        F_perm.append(anova_lm(full_perm, typ=2).loc['C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', 'F'])
    F_perm = np.array(F_perm)
    p_value = (np.sum(F_perm >= F_obs) + 1) / (N + 1)

    ax.hist(F_perm, bins=30, color='#1f77b4', alpha=0.7)
    ax.axvline(F_obs, color='red', linestyle='--', linewidth=2, label=f'Observed F = {F_obs:.3f}')
    ax.set_title(f"{label} — p_FL ≈ {p_value:.3f}")
    ax.set_xlabel('F-statistic')
    ax.set_ylabel('Frequency')
    ax.legend()

fig.suptitle('Fig. 7 — Freedman–Lane permutation tests (2000 permutations)', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_dir / 'fig07_freedman_lane.png', dpi=300)
fig.savefig(out_dir / 'fig07_freedman_lane.pdf')
plt.close(fig)
