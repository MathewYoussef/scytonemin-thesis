import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

sns.set(style='whitegrid')
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]

df = pd.read_csv(project_root / 'Chromatogram_derived_concentrations.csv')

metrics = {
    'Total': 'total_mg_per_gDW',
    'Oxidized': 'oxidized_mg_per_gDW',
    'Reduced': 'reduced_mg_per_gDW'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

for col_idx, (label, col) in enumerate(metrics.items()):
    ax = axes[0, col_idx]
    X = sm.add_constant(df['p_uvb_mw_cm2'])
    model = sm.OLS(df[col], X).fit()
    x_vals = pd.Series(sorted(df['p_uvb_mw_cm2']))
    pred = model.get_prediction(sm.add_constant(x_vals))
    ax.scatter(df['p_uvb_mw_cm2'], df[col], s=50, alpha=0.8, color='#1f77b4')
    ax.plot(x_vals, pred.predicted_mean, color='#1f77b4')
    ax.fill_between(x_vals, pred.conf_int()[:, 0], pred.conf_int()[:, 1],
                    color='#1f77b4', alpha=0.2)
    ax.set_xlabel('UVB (mW·cm$^{-2}$)')
    ax.set_ylabel(f'{label} (mg·gDW$^{-1}$)')
    ax.set_title(f'Fig. 3{chr(65 + col_idx)} — {label} vs UVB')
    slope = model.params['p_uvb_mw_cm2']
    slope_se = model.bse['p_uvb_mw_cm2']
    r2 = model.rsquared
    annotation = "Slope = {0:.4f} ± {1:.4f}\n$R^2$ = {2:.4f}\nn = {3}".format(slope, slope_se, r2, len(df))
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for col_idx, (label, col) in enumerate(metrics.items()):
    ax = axes[1, col_idx]
    X = sm.add_constant(df['p_uva_mw_cm2'])
    model = sm.OLS(df[col], X).fit()
    x_vals = pd.Series(sorted(df['p_uva_mw_cm2']))
    pred = model.get_prediction(sm.add_constant(x_vals))
    ax.scatter(df['p_uva_mw_cm2'], df[col], s=50, alpha=0.8, color='#ff7f0e')
    ax.plot(x_vals, pred.predicted_mean, color='#ff7f0e')
    ax.fill_between(x_vals, pred.conf_int()[:, 0], pred.conf_int()[:, 1],
                    color='#ff7f0e', alpha=0.2)
    ax.set_xlabel('UVA (mW·cm$^{-2}$)')
    ax.set_ylabel(f'{label} (mg·gDW$^{-1}$)')
    ax.set_title(f'Fig. 3{chr(68 + col_idx)} — {label} vs UVA')
    slope = model.params['p_uva_mw_cm2']
    slope_se = model.bse['p_uva_mw_cm2']
    r2 = model.rsquared
    annotation = "Slope = {0:.4f} ± {1:.4f}\n$R^2$ = {2:.4f}\nn = {3}".format(slope, slope_se, r2, len(df))
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Fig. 3 — Chromatogram single-factor trends', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig03_single_factor_chrom.png', dpi=300)
fig.savefig(out_dir / 'fig03_single_factor_chrom.pdf')
plt.close(fig)
