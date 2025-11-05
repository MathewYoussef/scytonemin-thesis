import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]

z_df = pd.read_csv(project_root / "chromatogram_zscores.csv")
raw_df = pd.read_csv(project_root / "Chromatogram_derived_concentrations.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)

def plot_emm(ax, df, value_col, group_col, label):
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'count', 'std']).reset_index()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    ax.errorbar(grouped[group_col], grouped['mean'], yerr=grouped['sem'], fmt='-o', capsize=4)
    ax.set_xlabel(f"{label} (mW·cm$^{-2}$)")
    ax.set_ylabel(value_col)
    ax.set_xticks(grouped[group_col])
    ax.set_xticklabels([f"{x:.3g}" for x in grouped[group_col]])

plot_emm(axes[0,0], z_df, 'z_conc_mg_ml', 'p_uva_mw_cm2', 'UVA')
axes[0,0].set_title('Fig. 5A — z-score concentration vs UVA')
plot_emm(axes[0,1], z_df, 'z_conc_mg_ml', 'p_uvb_mw_cm2', 'UVB')
axes[0,1].set_title('Fig. 5B — z-score concentration vs UVB')
plot_emm(axes[1,0], raw_df, 'total_mg_per_gDW', 'p_uva_mw_cm2', 'UVA')
axes[1,0].set_title('Fig. 5C — raw amount vs UVA')
plot_emm(axes[1,1], raw_df, 'total_mg_per_gDW', 'p_uvb_mw_cm2', 'UVB')
axes[1,1].set_title('Fig. 5D — raw amount vs UVB')

fig.suptitle('Fig. 5 — Estimated marginal means (Chromatogram)', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_dir / 'fig05_emm.png', dpi=300)
fig.savefig(out_dir / 'fig05_emm.pdf')
plt.close(fig)
