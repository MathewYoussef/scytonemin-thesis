import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]
dad_root = project_root.parent / 'Diode_Array_Derived_Calibration_Plots'

chrom_files = {
    'Total': project_root / 'standards_fitted_total.csv',
    'Oxidized': project_root / 'standards_fitted_oxidized.csv',
    'Reduced': project_root / 'standards_fitted_reduced.csv',
}

dad_files = {
    'Total': dad_root / 'standards_fitted_total.csv',
    'Oxidized': dad_root / 'standards_fitted_oxidized.csv',
    'Reduced': dad_root / 'standards_fitted_reduced.csv',
}

calibration_stats = {
    ('Chromatogram','Total'): dict(slope=7.210567878414983e-06, intercept=-0.015988813752569535,
                                   slope_se=2.632e-07, intercept_se=0.005092, r2=0.9934,
                                   max_rel=0.1274, df=5),
    ('Chromatogram','Oxidized'): dict(slope=1.104e-05, intercept=-0.01612,
                                      slope_se=6.145e-07, intercept_se=0.007752, r2=0.9847,
                                      max_rel=0.1417, df=5),
    ('Chromatogram','Reduced'): dict(slope=2.056e-05, intercept=-0.01473,
                                     slope_se=3.18e-07, intercept_se=0.002148, r2=0.9988,
                                     max_rel=0.1201, df=5),
    ('DAD','Total'): dict(slope=4.526e-08, intercept=-0.02402,
                          slope_se=6.447e-09, intercept_se=0.02446, r2=0.9079,
                          max_rel=0.3073, df=5),
    ('DAD','Oxidized'): dict(slope=3.007e-08, intercept=-0.03419,
                             slope_se=3.655e-09, intercept_se=0.02283, r2=0.9312,
                             max_rel=0.2658, df=5),
    ('DAD','Reduced'): dict(slope=4.711e-08, intercept=-0.02169,
                            slope_se=3.069e-09, intercept_se=0.01110, r2=0.9792,
                            max_rel=0.4688, df=5),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

for idx, (analyte, path) in enumerate(chrom_files.items()):
    ax = axes[0, idx]
    df = pd.read_csv(path)
    auc = df['response'] if 'response' in df.columns else df['area']
    conc = df['known_concentration_mg_ml']
    ax.scatter(auc, conc, color='#1f77b4', s=50, alpha=0.8)
    stats = calibration_stats[('Chromatogram', analyte)]
    x_vals = pd.Series(sorted(auc))
    y_vals = stats['slope'] * x_vals + stats['intercept']
    ax.plot(x_vals, y_vals, color='#1f77b4', linewidth=2)
    ax.set_title(f'Chromatogram {analyte}')
    ax.set_xlabel('AUC (blank-corrected)')
    ax.set_ylabel('Concentration (mg·mL$^{-1}$)')
    inset = ax.inset_axes([0.05, 0.55, 0.42, 0.4])
    fitted = stats['slope'] * auc + stats['intercept']
    rel_resid = (conc - fitted) / conc
    inset.axhline(0.2, linestyle='--', color='grey', linewidth=1)
    inset.axhline(-0.2, linestyle='--', color='grey', linewidth=1)
    inset.axhline(0.1, linestyle=':', color='grey', linewidth=1)
    inset.axhline(-0.1, linestyle=':', color='grey', linewidth=1)
    inset.scatter(fitted, rel_resid, color='#1f77b4', s=30)
    inset.set_xlabel('Fitted (mg·mL$^{-1}$)', fontsize=8)
    inset.set_ylabel('Relative residual', fontsize=8)
    inset.tick_params(axis='both', labelsize=8)
    ax.text(0.05, 0.05,
            (f"β₀ = {stats['intercept']:.3g}\n"
             f"β₁ = {stats['slope']:.3g}\n"
             f"SE(β₀) = {stats['intercept_se']:.2g}\n"
             f"SE(β₁) = {stats['slope_se']:.2g}\n"
             f"$R^2$ = {stats['r2']:.4f}\n"
             f"max |rel resid| = {stats['max_rel']:.3f}\n"
             f"df = {stats['df']}"),
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for idx, (analyte, path) in enumerate(dad_files.items()):
    ax = axes[1, idx]
    df = pd.read_csv(path)
    auc = df['auc_corrected']
    conc = df['known_concentration_mg_ml']
    ax.scatter(auc, conc, color='#d62728', s=50, alpha=0.8)
    stats = calibration_stats[('DAD', analyte)]
    x_vals = pd.Series(sorted(auc))
    y_vals = stats['slope'] * x_vals + stats['intercept']
    ax.plot(x_vals, y_vals, color='#d62728', linewidth=2)
    ax.set_title(f'DAD {analyte}')
    ax.set_xlabel('AUC (blank-corrected)')
    ax.set_ylabel('Concentration (mg·mL$^{-1}$)')
    inset = ax.inset_axes([0.05, 0.55, 0.42, 0.4])
    fitted = stats['slope'] * auc + stats['intercept']
    rel_resid = (conc - fitted) / conc
    inset.axhline(0.2, linestyle='--', color='grey', linewidth=1)
    inset.axhline(-0.2, linestyle='--', color='grey', linewidth=1)
    inset.axhline(0.1, linestyle=':', color='grey', linewidth=1)
    inset.axhline(-0.1, linestyle=':', color='grey', linewidth=1)
    inset.scatter(fitted, rel_resid, color='#d62728', s=30)
    inset.set_xlabel('Fitted (mg·mL$^{-1}$)', fontsize=8)
    inset.set_ylabel('Relative residual', fontsize=8)
    inset.tick_params(axis='both', labelsize=8)
    ax.text(0.05, 0.05,
            (f"β₀ = {stats['intercept']:.3g}\n"
             f"β₁ = {stats['slope']:.3g}\n"
             f"SE(β₀) = {stats['intercept_se']:.2g}\n"
             f"SE(β₁) = {stats['slope_se']:.2g}\n"
             f"$R^2$ = {stats['r2']:.4f}\n"
             f"max |rel resid| = {stats['max_rel']:.3f}\n"
             f"df = {stats['df']}"),
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Fig. 1 — Calibration integrity', fontsize=20)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(out_dir / 'fig01_calibration.png', dpi=300)
fig.savefig(out_dir / 'fig01_calibration.pdf')
plt.close(fig)
