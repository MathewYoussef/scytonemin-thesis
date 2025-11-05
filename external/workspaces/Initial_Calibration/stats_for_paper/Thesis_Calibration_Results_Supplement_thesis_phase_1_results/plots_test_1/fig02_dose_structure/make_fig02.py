import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')
out_dir = Path(__file__).resolve().parent
project_root = out_dir.parents[1]

df = pd.read_csv(project_root / 'Chromatogram_derived_concentrations.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(df['p_uva_mw_cm2'], df['p_uvb_mw_cm2'], s=60, alpha=0.7, color='#1f77b4')
ax.set_xlabel('UVA (mW·cm$^{-2}$)')
ax.set_ylabel('UVB (mW·cm$^{-2}$)')
ax.set_title('Fig. 2A — UVA vs UVB scatter (n=30)')
ax.set_xlim(df['p_uva_mw_cm2'].min()-0.2, df['p_uva_mw_cm2'].max()+0.2)
ax.set_ylim(df['p_uvb_mw_cm2'].min()-0.1, df['p_uvb_mw_cm2'].max()+0.1)
r = df[['p_uva_mw_cm2','p_uvb_mw_cm2']].corr().iloc[0,1]
ax.text(0.05, 0.9, f"Pearson r = {r:.4f}", transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[1]
ax.hist(df['p_uva_mw_cm2'], bins=6, color='#ff7f0e', alpha=0.8, edgecolor='black')
ax.set_xlabel('UVA (mW·cm$^{-2}$)')
ax.set_ylabel('Count')
ax.set_title('Fig. 2B — UVA distribution')
ax.text(0.05, 0.85, 'VIF$_{UVA}$ ≈ 37.9', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[2]
ax.hist(df['p_uvb_mw_cm2'], bins=6, color='#2ca02c', alpha=0.8, edgecolor='black')
ax.set_xlabel('UVB (mW·cm$^{-2}$)')
ax.set_ylabel('Count')
ax.set_title('Fig. 2C — UVB distribution')
ax.text(0.05, 0.85, 'VIF$_{UVB}$ ≈ 10.4\nVIF$_{UVA×UVB}$ ≈ 25.1', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Fig. 2 — Dose structure and collinearity', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig02_dose_structure.png', dpi=300)
fig.savefig(out_dir / 'fig02_dose_structure.pdf')
plt.close(fig)
