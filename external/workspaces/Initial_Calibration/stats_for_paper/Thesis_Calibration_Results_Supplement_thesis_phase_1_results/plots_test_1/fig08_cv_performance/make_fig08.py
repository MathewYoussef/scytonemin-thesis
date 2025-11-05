import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

sns.set(style="whitegrid")
out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)

cases = [
    {
        'label': 'Chrom Δ amount',
        'data': pd.read_csv(Path('../../chromatogram_delta.csv')),
        'response': 'delta_amount_mg_per_gDW'
    },
    {
        'label': 'Chrom z amount',
        'data': pd.read_csv(Path('../../chromatogram_zscores.csv')),
        'response': 'z_amount_mg_per_gDW'
    },
    {
        'label': 'DAD total mg·gDW$^{-1}$',
        'data': pd.read_csv(Path('../../dad_delta.csv')),
        'response': 'predicted_total_mg_per_gDW'
    }
]

ridge_scores = []
pls_scores = []
scatter_data = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
alpha = 1.0

for case in cases:
    df = case['data'][['p_uva_mw_cm2', 'p_uvb_mw_cm2', case['response']]].dropna()
    X = df[['p_uva_mw_cm2', 'p_uvb_mw_cm2']].copy()
    X['interaction'] = X['p_uva_mw_cm2'] * X['p_uvb_mw_cm2']
    y = df[case['response']].values

    ridge = Ridge(alpha=alpha)
    ridge_cv = cross_val_score(ridge, X, y, cv=kf, scoring='r2')
    ridge_scores.append({'endpoint': case['label'], 'R2': ridge_cv.mean()})
    y_pred_ridge = cross_val_predict(ridge, X, y, cv=kf)
    scatter_data.append({'endpoint': case['label'] + ' (Ridge)', 'observed': y, 'predicted': y_pred_ridge})

    pls = PLSRegression(n_components=2)
    pls_cv = cross_val_score(pls, X, y, cv=kf, scoring='r2')
    pls_scores.append({'endpoint': case['label'], 'R2': pls_cv.mean()})
    y_pred_pls = cross_val_predict(pls, X, y, cv=kf)
    scatter_data.append({'endpoint': case['label'] + ' (PLS)', 'observed': y, 'predicted': y_pred_pls})

ridge_df = pd.DataFrame(ridge_scores)
pls_df = pd.DataFrame(pls_scores)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(data=ridge_df, x='endpoint', y='R2', ax=axes[0,0], color='#1f77b4')
axes[0,0].axhline(0, color='black', linewidth=1)
axes[0,0].set_title('Fig. 8A — Ridge CV $R^2$')
axes[0,0].set_xlabel('Endpoint')
axes[0,0].set_ylabel('CV $R^2$')
axes[0,0].tick_params(axis='x', rotation=15)

sns.barplot(data=pls_df, x='endpoint', y='R2', ax=axes[0,1], color='#ff7f0e')
axes[0,1].axhline(0, color='black', linewidth=1)
axes[0,1].set_title('Fig. 8B — PLS CV $R^2$')
axes[0,1].set_xlabel('Endpoint')
axes[0,1].set_ylabel('CV $R^2$')
axes[0,1].tick_params(axis='x', rotation=15)

axes[1,0].scatter(scatter_data[0]['observed'], scatter_data[0]['predicted'], alpha=0.7)
lims = [min(scatter_data[0]['observed'].min(), scatter_data[0]['predicted'].min()),
        max(scatter_data[0]['observed'].max(), scatter_data[0]['predicted'].max())]
axes[1,0].plot(lims, lims, 'r--')
axes[1,0].set_xlabel('Observed')
axes[1,0].set_ylabel('Predicted (CV)')
axes[1,0].set_title(scatter_data[0]['endpoint'])

axes[1,1].scatter(scatter_data[1]['observed'], scatter_data[1]['predicted'], alpha=0.7)
lims = [min(scatter_data[1]['observed'].min(), scatter_data[1]['predicted'].min()),
        max(scatter_data[1]['observed'].max(), scatter_data[1]['predicted'].max())]
axes[1,1].plot(lims, lims, 'r--')
axes[1,1].set_xlabel('Observed')
axes[1,1].set_ylabel('Predicted (CV)')
axes[1,1].set_title(scatter_data[1]['endpoint'])

fig.suptitle('Fig. 8 — Cross-validated predictive performance', fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out_dir / 'fig08_cv_performance.png', dpi=300)
fig.savefig(out_dir / 'fig08_cv_performance.pdf')
plt.close(fig)
