import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]

df = pd.read_csv(project_root / 'dad_delta.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, dose_col, label, title in [
    (axes[0], 'p_uvb_mw_cm2', 'UVB', 'Fig. 4A — DAD total mg·gDW$^{-1}$ vs UVB'),
    (axes[1], 'p_uva_mw_cm2', 'UVA', 'Fig. 4B — DAD total mg·gDW$^{-1}$ vs UVA')
]:
    X = sm.add_constant(df[dose_col])
    model = sm.OLS(df['predicted_total_mg_per_gDW'], X).fit()
    x_vals = pd.Series(sorted(df[dose_col]))
    pred = model.get_prediction(sm.add_constant(x_vals))
    ax.scatter(df[dose_col], df['predicted_total_mg_per_gDW'], s=60, alpha=0.8, color='#d62728')
    ax.plot(x_vals, pred.predicted_mean, color='#d62728')
    ax.fill_between(x_vals, pred.conf_int()[:,0], pred.conf_int()[:,1], color='#d62728', alpha=0.2)
    slope = model.params[dose_col]
    slope_se = model.bse[dose_col]
    r2 = model.rsquared
    ax.text(0.05, 0.95,
            f"Slope = {slope:.4f} ± {slope_se:.4f}\n$R^2$ = {r2:.4f}\nn = {len(df)}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel(f"{label} (mW·cm$^{-2}$)")
    ax.set_ylabel('DAD total (mg·gDW$^{-1}$)')
    ax.set_title(title)

fig.suptitle('Fig. 4 — DAD single-factor trends (biomass-normalized)', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig04_single_factor_dad.png', dpi=300)
fig.savefig(out_dir / 'fig04_single_factor_dad.pdf')
plt.close(fig)
