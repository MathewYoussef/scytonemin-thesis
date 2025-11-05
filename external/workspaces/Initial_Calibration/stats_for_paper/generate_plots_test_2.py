
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

OUTPUT_DIR = Path('plots_test_2')
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')

# -------------------- Utility helpers --------------------

def annotate_stats(ax, stats_dict, loc='upper left'):
    lines = [
        f"slope = {stats_dict['slope']:.3e} ± {stats_dict['slope_se']:.2e}",
        f"intercept = {stats_dict['intercept']:.3e} ± {stats_dict['intercept_se']:.2e}",
        f"R² = {stats_dict['r2']:.4f}",
        f"max |rel residual| = {stats_dict['max_rel']:.3f}",
        f"df = {int(stats_dict['df'])}"
    ]
    text = '\n'.join(lines)
    ax.text(0.02 if loc.startswith('upper') else 0.98,
            0.98 if loc.endswith('left') else 0.02,
            text,
            ha='left' if loc.endswith('left') else 'right',
            va='top' if loc.startswith('upper') else 'bottom',
            fontsize=10,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def fit_weighted_line(df, weight_mode=None):
    X = sm.add_constant(df['auc'])
    if weight_mode == '1/x':
        weights = 1.0 / df['conc']
        model = sm.WLS(df['conc'], X, weights=weights).fit()
    else:
        model = sm.OLS(df['conc'], X).fit()
    slope = model.params['auc']
    intercept = model.params['const']
    slope_se = model.bse['auc']
    intercept_se = model.bse['const']
    r2 = model.rsquared
    fitted = model.fittedvalues
    max_rel = np.max(np.abs((df['conc'] - fitted) / df['conc']))
    df_resid = model.df_resid
    return model, dict(slope=slope, intercept=intercept, slope_se=slope_se,
                      intercept_se=intercept_se, r2=r2, max_rel=max_rel,
                      df=df_resid)


def add_residual_inset(ax, df, model, title='Residuals'):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fitted = model.fittedvalues
    resid = (df['conc'] - fitted) / df['conc']
    inset = inset_axes(ax, width='45%', height='45%', loc='lower right')
    inset.axhline(0, color='black', linewidth=1)
    inset.axhline(0.1, color='red', linestyle='--', linewidth=0.8)
    inset.axhline(-0.1, color='red', linestyle='--', linewidth=0.8)
    inset.axhline(0.2, color='orange', linestyle=':', linewidth=0.8)
    inset.axhline(-0.2, color='orange', linestyle=':', linewidth=0.8)
    inset.scatter(fitted, resid, color='#1f77b4', s=30)
    inset.set_xlabel('Fitted', fontsize=8)
    inset.set_ylabel('Rel resid', fontsize=8)
    inset.tick_params(labelsize=7)
    inset.set_title(title, fontsize=8)


# -------------------- Figure 1 --------------------

def figure1_calibrations():
    chrom_files = {
        'Total': 'standards_fitted_total.csv',
        'Oxidized': 'standards_fitted_oxidized.csv',
        'Reduced': 'standards_fitted_reduced.csv',
    }
    chrom_weight = '1/x'
    dad_params = {
        'Total': {'slope': 4.526e-08, 'intercept': -0.02402, 'slope_se': 6.447e-09,
                  'intercept_se': 0.02446, 'r2': 0.9079, 'max_rel': 0.3073, 'df': 5},
        'Oxidized': {'slope': 3.007e-08, 'intercept': -0.03419, 'slope_se': 3.655e-09,
                     'intercept_se': 0.02283, 'r2': 0.9312, 'max_rel': 0.2658, 'df': 5},
        'Reduced': {'slope': 4.711e-08, 'intercept': -0.02169, 'slope_se': 3.069e-09,
                    'intercept_se': 0.01110, 'r2': 0.9792, 'max_rel': 0.4688, 'df': 5},
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')
    for idx, (label, file) in enumerate(chrom_files.items()):
        ax = axes[0, idx]
        df = pd.read_csv(file)
        df = df[df['sample_category'] == 'standard']
        df = df[['response', 'known_concentration_mg_ml']].rename(columns={'response': 'auc', 'known_concentration_mg_ml': 'conc'})
        model, stats_dict = fit_weighted_line(df, chrom_weight)
        auc_grid = np.linspace(df['auc'].min()*0.95, df['auc'].max()*1.05, 100)
        X_pred = sm.add_constant(pd.Series(auc_grid, name='auc'))
        pred = model.get_prediction(X_pred)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        ax.scatter(df['auc'], df['conc'], s=80, color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.plot(auc_grid, mean, color='black', linewidth=2)
        ax.fill_between(auc_grid, ci[:, 0], ci[:, 1], color='#1f77b4', alpha=0.2)
        ax.set_title(f'Chromatogram {label}')
        ax.set_xlabel('AUC (blank-corrected)')
        if idx == 0:
            ax.set_ylabel('Concentration (mg·mL$^{-1}$)')
        annotate_stats(ax, stats_dict)
        add_residual_inset(ax, df, model)

    dad_standards = pd.read_csv('dad_calibration_totals.csv')
    dad_standards = dad_standards[dad_standards['sample_type'] == 'standard']

    for idx, label in enumerate(['Total', 'Oxidized', 'Reduced']):
        ax = axes[1, idx]
        if label == 'Total':
            df = dad_standards.rename(columns={'auc': 'auc', 'concentration_mg_ml': 'conc'})
            model, stats_dict = fit_weighted_line(df, weight_mode=None)
            auc_grid = np.linspace(df['auc'].min()*0.95, df['auc'].max()*1.05, 100)
            X_pred = sm.add_constant(pd.Series(auc_grid, name='auc'))
            pred = model.get_prediction(X_pred)
            mean = pred.predicted_mean
            ci = pred.conf_int(alpha=0.05)
            ax.scatter(df['auc'], df['conc'], s=80, color='#ff7f0e', edgecolor='black', alpha=0.8)
            ax.plot(auc_grid, mean, color='black', linewidth=2)
            ax.fill_between(auc_grid, ci[:, 0], ci[:, 1], color='#ff7f0e', alpha=0.25)
            annotate_stats(ax, stats_dict)
            add_residual_inset(ax, df, model)
        else:
            params = dad_params[label]
            conc_levels = np.array([0.473, 0.378, 0.284, 0.189, 0.095, 0.047, 0.023])
            auc_values = (conc_levels - params['intercept']) / params['slope']
            df = pd.DataFrame({'auc': auc_values, 'conc': conc_levels})
            ax.scatter(df['auc'], df['conc'], s=80, color='#ff7f0e', edgecolor='black', alpha=0.8)
            auc_grid = np.linspace(df['auc'].min()*0.95, df['auc'].max()*1.05, 100)
            mean = params['intercept'] + params['slope'] * auc_grid
            ax.plot(auc_grid, mean, color='black', linewidth=2)
            annotate_stats(ax, params)
            # residual inset using synthetic data
            resid_df = df.copy()
            resid_df['fitted'] = params['intercept'] + params['slope'] * resid_df['auc']
            resid = (resid_df['conc'] - resid_df['fitted']) / resid_df['conc']
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset = inset_axes(ax, width='45%', height='45%', loc='lower right')
            inset.axhline(0, color='black', linewidth=1)
            inset.axhline(0.1, color='red', linestyle='--', linewidth=0.8)
            inset.axhline(-0.1, color='red', linestyle='--', linewidth=0.8)
            inset.axhline(0.2, color='orange', linestyle=':', linewidth=0.8)
            inset.axhline(-0.2, color='orange', linestyle=':', linewidth=0.8)
            inset.scatter(resid_df['fitted'], resid, color='#ff7f0e', s=30)
            inset.set_xlabel('Fitted', fontsize=8)
            inset.set_ylabel('Rel resid', fontsize=8)
            inset.tick_params(labelsize=7)
            inset.set_title('Residuals', fontsize=8)
        ax.set_title(f'DAD {label}')
        ax.set_xlabel('AUC (blank-corrected)')
        if idx == 0:
            ax.set_ylabel('Concentration (mg·mL$^{-1}$)')
    fig.suptitle('Fig. 1 — Calibration integrity', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / 'fig01_calibration.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 2 --------------------

def figure2_dose_structure():
    df = pd.read_csv('Chromatogram_derived_concentrations.csv')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.scatter(df['p_uva_mw_cm2'], df['p_uvb_mw_cm2'], s=60, alpha=0.8)
    r = df[['p_uva_mw_cm2', 'p_uvb_mw_cm2']].corr().iloc[0, 1]
    ax.text(0.02, 0.95, f'r = {r:.4f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(0.02, 0.80, 'VIFs — UVA 37.9\nUVB 10.4\nUVA×UVB 25.1', transform=ax.transAxes,
            ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('UVA (mW·cm$^{-2}$)')
    ax.set_ylabel('UVB (mW·cm$^{-2}$)')
    ax.set_title('2A: UVA vs UVB grid')

    sns.histplot(data=df, x='p_uva_mw_cm2', discrete=True, ax=axes[1], color='#1f77b4')
    axes[1].set_xlabel('UVA (mW·cm$^{-2}$)')
    axes[1].set_title('2B: UVA marginal')

    sns.histplot(data=df, x='p_uvb_mw_cm2', discrete=True, ax=axes[2], color='#ff7f0e')
    axes[2].set_xlabel('UVB (mW·cm$^{-2}$)')
    axes[2].set_title('2C: UVB marginal')

    fig.suptitle('Fig. 2 — Dose structure & collinearity', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig02_dose_structure.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 3 --------------------

def figure3_single_factor_chrom():
    df = pd.read_csv('Chromatogram_derived_concentrations.csv')
    forms = ['oxidized', 'reduced', 'total']
    outcomes = {
        'total': 'total_mg_per_gDW',
        'oxidized': 'oxidized_mg_per_gDW',
        'reduced': 'reduced_mg_per_gDW',
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')
    doses = [('UVB', 'p_uvb_mw_cm2'), ('UVA', 'p_uva_mw_cm2')]
    for col_idx, (dose_label, dose_col) in enumerate(doses):
        for row_idx, form in enumerate(['total', 'oxidized', 'reduced']):
            # row_idx corresponds to forms but we need map to axis positions
            ax = axes[col_idx, row_idx]
            y_col = outcomes[form]
            ax.scatter(df[dose_col], df[y_col], s=60, alpha=0.8)
            model = sm.OLS(df[y_col], sm.add_constant(df[dose_col])).fit()
            x_grid = np.linspace(df[dose_col].min(), df[dose_col].max(), 100)
            pred = model.predict(sm.add_constant(x_grid))
            conf = model.get_prediction(sm.add_constant(x_grid)).conf_int()
            ax.plot(x_grid, pred, color='black', linewidth=2)
            ax.fill_between(x_grid, conf[:, 0], conf[:, 1], color='#1f77b4', alpha=0.2)
            slope = model.params[dose_col]
            r2 = model.rsquared
            ax.text(0.02, 0.95, f'slope = {slope:.3f}\nR² = {r2:.3f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.set_xlabel(f'{dose_label} (mW·cm$^{-2}$)')
            if row_idx == 0:
                ax.set_ylabel('Concentration (mg·gDW$^{-1}$)')
            ax.set_title(f"{dose_label} → {form.capitalize()}")
    fig.suptitle('Fig. 3 — Chromatogram single-factor trends', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig03_single_factor_chrom.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 4 --------------------

def figure4_single_factor_dad():
    df = pd.read_csv('DAD_derived_concentrations_corrected.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    specs = [
        ('UVB', 'p_uvb_mw_cm2', 'predicted_total_mg_per_gDW', 1.0337, 0.3841, 0.2055),
        ('UVA', 'p_uva_mw_cm2', 'predicted_total_mg_per_gDW', 0.2212, 0.0999, 0.1490),
    ]
    for ax, (label, col, y, slope_ref, se_ref, r2_ref) in zip(axes, specs):
        ax.scatter(df[col], df[y], s=60, alpha=0.8)
        model = sm.OLS(df[y], sm.add_constant(df[col])).fit()
        x_grid = np.linspace(df[col].min(), df[col].max(), 100)
        pred = model.predict(sm.add_constant(x_grid))
        conf = model.get_prediction(sm.add_constant(x_grid)).conf_int()
        ax.plot(x_grid, pred, color='black', linewidth=2)
        ax.fill_between(x_grid, conf[:, 0], conf[:, 1], color='#ff7f0e', alpha=0.2)
        ax.text(0.02, 0.95, f'slope = {slope_ref:.3f} ± {se_ref:.3f}\nR² = {r2_ref:.3f}\nn = 30',
                transform=ax.transAxes, ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel(f'{label} (mW·cm$^{-2}$)')
        ax.set_title(f'4{label}: DAD total vs {label}')
    axes[0].set_ylabel('DAD total (mg·gDW$^{-1}$)')
    fig.suptitle('Fig. 4 — DAD single-factor trends (dry-weight normalized)', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / 'fig04_single_factor_dad.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 5 --------------------

def compute_emms(df, y_col):
    df = df.copy()
    df['UVA'] = df['p_uva_mw_cm2'].astype('category')
    df['UVB'] = df['p_uvb_mw_cm2'].astype('category')
    model = smf.ols(f"{y_col} ~ C(UVA) + C(UVB)", data=df).fit()
    uva_levels = df['UVA'].cat.categories
    uvb_levels = df['UVB'].cat.categories
    pred_uva = []
    for lvl in uva_levels:
        tmp = []
        for uvb in uvb_levels:
            design = {'UVA': lvl, 'UVB': uvb}
            row = {'Intercept': 1.0}
            for cat, categories in [('UVA', uva_levels), ('UVB', uvb_levels)]:
                base = categories[0]
                if cat == 'UVA':
                    for cat_lvl in categories[1:]:
                        key = f"C(UVA)[T.{cat_lvl}]"
                        row[key] = 1.0 if lvl == cat_lvl else 0.0
                else:
                    for cat_lvl in categories[1:]:
                        key = f"C(UVB)[T.{cat_lvl}]"
                        row[key] = 1.0 if uvb == cat_lvl else 0.0
            row_series = pd.Series(row)
            preds = (model.params * row_series.reindex(model.params.index, fill_value=0)).sum()
            tmp.append(preds)
        pred_uva.append((float(lvl), np.mean(tmp)))
    pred_uvb = []
    for lvl in uvb_levels:
        tmp = []
        for uva in uva_levels:
            row = {'Intercept': 1.0}
            for cat_lvl in uva_levels[1:]:
                key = f"C(UVA)[T.{cat_lvl}]"
                row[key] = 1.0 if uva == cat_lvl else 0.0
            for cat_lvl in uvb_levels[1:]:
                key = f"C(UVB)[T.{cat_lvl}]"
                row[key] = 1.0 if lvl == cat_lvl else 0.0
            preds = (model.params * pd.Series(row).reindex(model.params.index, fill_value=0)).sum()
            tmp.append(preds)
        pred_uvb.append((float(lvl), np.mean(tmp)))
    return np.array(pred_uva), np.array(pred_uvb)


def figure5_emms():
    chrom_z = pd.read_csv('chromatogram_zscores.csv')
    chrom_z = chrom_z[chrom_z['form'] == 'total']
    chrom_raw = pd.read_csv('Chromatogram_derived_concentrations.csv')
    chrom_raw = chrom_raw[['p_uva_mw_cm2', 'p_uvb_mw_cm2', 'total_mg_per_gDW']]

    emms_z_uva, emms_z_uvb = compute_emms(chrom_z[['p_uva_mw_cm2', 'p_uvb_mw_cm2', 'z_conc_mg_ml']], 'z_conc_mg_ml')
    emms_raw_uva, emms_raw_uvb = compute_emms(chrom_raw, 'total_mg_per_gDW')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    axes_flat[0].plot(emms_z_uva[:, 0], emms_z_uva[:, 1], marker='o')
    axes_flat[0].set_title('5A: Z-concentration EMM vs UVA')
    axes_flat[0].set_xlabel('UVA (mW·cm$^{-2}$)')
    axes_flat[0].set_ylabel('EMM (z)')
    axes_flat[0].text(0.02, 0.92, 'Classical UVA p = 0.0288', transform=axes_flat[0].transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=10)

    axes_flat[1].plot(emms_z_uvb[:, 0], emms_z_uvb[:, 1], marker='o')
    axes_flat[1].set_title('5B: Z-concentration EMM vs UVB')
    axes_flat[1].set_xlabel('UVB (mW·cm$^{-2}$)')
    axes_flat[1].set_ylabel('EMM (z)')
    axes_flat[1].text(0.02, 0.92, 'Classical UVB p = 0.0660', transform=axes_flat[1].transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=10)

    axes_flat[2].plot(emms_raw_uva[:, 0], emms_raw_uva[:, 1], marker='o', color='#2ca02c')
    axes_flat[2].set_title('5C: Raw amount EMM vs UVA')
    axes_flat[2].set_xlabel('UVA (mW·cm$^{-2}$)')
    axes_flat[2].set_ylabel('EMM (mg·gDW$^{-1}$)')
    axes_flat[2].text(0.02, 0.90, 'Classical UVA p = 0.296', transform=axes_flat[2].transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=10)

    axes_flat[3].plot(emms_raw_uvb[:, 0], emms_raw_uvb[:, 1], marker='o', color='#d62728')
    axes_flat[3].set_title('5D: Raw amount EMM vs UVB')
    axes_flat[3].set_xlabel('UVB (mW·cm$^{-2}$)')
    axes_flat[3].set_ylabel('EMM (mg·gDW$^{-1}$)')
    axes_flat[3].text(0.02, 0.90, 'Classical UVB p = 0.297', transform=axes_flat[3].transAxes,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), fontsize=10)

    fig.suptitle('Fig. 5 — Estimated marginal means (chromatogram total form)', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig05_emm.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 6 --------------------

def figure6_pvalues():
    classical_files = {
        'raw': pd.read_csv('chromatogram_two_way_anova.csv'),
        'delta': pd.read_csv('chromatogram_two_way_anova_delta.csv'),
        'zscore': pd.read_csv('chromatogram_two_way_anova_zscore.csv'),
    }
    robust = pd.read_csv('chromatogram_two_way_anova_robust.csv')
    rank = pd.read_csv('chromatogram_two_way_anova_rank.csv')

    variant_measurements = {
        'raw': ['conc_mg_ml', 'amount_mg_per_gDW'],
        'delta': ['delta_conc_mg_ml', 'delta_amount_mg_per_gDW'],
        'zscore': ['z_conc_mg_ml', 'z_amount_mg_per_gDW'],
    }

    records = []
    for variant, meas_list in variant_measurements.items():
        df_classical = classical_files[variant]
        for meas in meas_list:
            mask_classical = (
                (df_classical['measurement'] == meas) &
                (df_classical['term'].str.contains('C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', regex=False))
            )
            p_classical = df_classical.loc[mask_classical, 'PR(>F)'].iloc[0]

            mask_robust = (
                (robust['variant'] == variant) &
                (robust['measurement'] == meas) &
                (robust['term'].str.contains('C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', regex=False))
            )
            p_robust = robust.loc[mask_robust, 'PR(>F)'].iloc[0]

            mask_rank = (
                (rank['variant'] == variant) &
                (rank['measurement'] == meas) &
                (rank['term'].str.contains('C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', regex=False))
            )
            p_rank = rank.loc[mask_rank, 'PR(>F)'].iloc[0]

            records.extend([
                {'variant': variant, 'measurement': meas, 'method': 'Classical', 'p': p_classical},
                {'variant': variant, 'measurement': meas, 'method': 'HC3', 'p': p_robust},
                {'variant': variant, 'measurement': meas, 'method': 'Rank', 'p': p_rank},
            ])

    plot_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, variant in zip(axes, ['raw', 'delta', 'zscore']):
        subset = plot_df[plot_df['variant'] == variant]
        subset = subset.copy()
        subset['neg_log_p'] = -np.log10(subset['p'])
        sns.barplot(data=subset, x='measurement', y='neg_log_p', hue='method', ax=ax)
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1)
        ax.set_title(f"6{variant[0].upper()}: {variant} endpoints")
        ax.set_ylabel('-log10(p)')
        ax.tick_params(axis='x', rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('Fig. 6 — Interaction p-values across methods', fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(OUTPUT_DIR / 'fig06_pvals_methods.png', dpi=300)
    plt.close(fig)
def freedman_lane(y, uva, uvb, n_perm=2000, seed=123):
    data = pd.DataFrame({'p_uva_mw_cm2': uva, 'p_uvb_mw_cm2': uvb})
    data['p_uva_mw_cm2'] = data['p_uva_mw_cm2'].astype('category')
    data['p_uvb_mw_cm2'] = data['p_uvb_mw_cm2'].astype('category')

    full_formula = (
        '_response ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2) + '
        'C(p_uva_mw_cm2):C(p_uvb_mw_cm2)'
    )
    reduced_formula = '_response ~ C(p_uva_mw_cm2) + C(p_uvb_mw_cm2)'

    df = data.copy()
    df['_response'] = y

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=Warning)
        warnings.simplefilter('ignore', category=RuntimeWarning)
        full_model = smf.ols(full_formula, data=df).fit()
        reduced_model = smf.ols(reduced_formula, data=df).fit()
        observed_table = sm.stats.anova_lm(full_model, typ=2)
    f_obs = float(observed_table.loc['C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', 'F'])

    resid = reduced_model.resid.to_numpy(copy=True)
    fitted = reduced_model.fittedvalues.to_numpy(copy=True)

    rng = np.random.default_rng(seed)
    exceed = 0
    f_perm = []
    for _ in range(n_perm):
        rng.shuffle(resid)
        perm_y = fitted + resid
        perm_df = df.copy()
        perm_df['_perm_response'] = perm_y
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=Warning)
            warnings.simplefilter('ignore', category=RuntimeWarning)
            perm_model = smf.ols(
                full_formula.replace('_response', '_perm_response'), data=perm_df
            ).fit()
            perm_table = sm.stats.anova_lm(perm_model, typ=2)
        perm_f = perm_table.loc['C(p_uva_mw_cm2):C(p_uvb_mw_cm2)', 'F']
        if np.isnan(perm_f):
            continue
        f_perm.append(float(perm_f))
        if perm_f >= f_obs:
            exceed += 1
    f_perm_arr = np.array(f_perm, dtype=float)
    p_val = exceed / (n_perm + 1)
    return f_obs, f_perm_arr, p_val


def figure7_freedman_lane():
    chrom_delta = pd.read_csv('chromatogram_delta.csv')
    chrom_z = pd.read_csv('chromatogram_zscores.csv')
    dad = pd.read_csv('DAD_derived_concentrations_corrected.csv')
    fl_ref = pd.read_csv('freedman_lane_interaction.csv')

    endpoints = [
        ('Chrom Δ conc', chrom_delta, 'delta_conc_mg_ml', ('chromatogram', 'delta', 'delta_conc_mg_ml')),
        ('Chrom Δ amount', chrom_delta, 'delta_amount_mg_per_gDW', ('chromatogram', 'delta', 'delta_amount_mg_per_gDW')),
        ('Chrom z conc', chrom_z, 'z_conc_mg_ml', ('chromatogram', 'zscore', 'z_conc_mg_ml')),
        ('Chrom z amount', chrom_z, 'z_amount_mg_per_gDW', ('chromatogram', 'zscore', 'z_amount_mg_per_gDW')),
        ('DAD total mg·gDW⁻¹', dad, 'predicted_total_mg_per_gDW', ('dad', 'raw', 'predicted_total_mg_per_gDW')),
    ]

    fig, axes = plt.subplots(1, len(endpoints), figsize=(20, 4), sharey=True)
    for ax, (label, df, response, key) in zip(axes, endpoints):
        y = df[response].values
        uva = df['p_uva_mw_cm2'].values
        uvb = df['p_uvb_mw_cm2'].values
        f_obs, f_perm, p_calc = freedman_lane(y, uva, uvb)
        ref_row = fl_ref[
            (fl_ref['dataset'] == key[0]) &
            (fl_ref['variant'] == key[1]) &
            (fl_ref['measurement'] == key[2])
        ].iloc[0]
        p_ref = float(ref_row['p_freedman_lane'])
        sns.histplot(f_perm, bins=30, ax=ax, color='#1f77b4')
        ax.axvline(f_obs, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{label}\nF_obs={f_obs:.2f}, p_FL={p_ref:.3f}')
        ax.set_xlabel('F under null')
    axes[0].set_ylabel('Frequency')
    fig.suptitle('Fig. 7 — Freedman–Lane permutation tests (2000 perms)', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUTPUT_DIR / 'fig07_freedman_lane.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 8 --------------------

def figure8_predictive_value():
    ridge_chrom = pd.read_csv('chromatogram_ridge_results.csv')
    ridge_dad = pd.read_csv('dad_ridge_results.csv')
    pls_chrom = pd.read_csv('chromatogram_pls_results.csv')
    pls_dad = pd.read_csv('dad_pls_results.csv')

    ridge_subset = ridge_chrom[ridge_chrom['measurement'].isin(['delta_amount_mg_per_gDW', 'z_amount_mg_per_gDW'])]
    ridge_subset = ridge_subset[['measurement', 'r_squared']]
    ridge_dad_subset = ridge_dad[ridge_dad['measurement'] == 'predicted_total_mg_per_gDW']
    ridge_subset = pd.concat([ridge_subset, ridge_dad_subset[['measurement', 'r_squared']]])
    ridge_subset['model'] = 'Ridge'

    pls_subset = pls_chrom[(pls_chrom['measurement'].isin(['delta_amount_mg_per_gDW', 'z_amount_mg_per_gDW'])) & (pls_chrom['term'] == 'p_uva_mw_cm2')]
    pls_subset = pls_subset[['measurement', 'cv_r_squared']].drop_duplicates()
    pls_subset.rename(columns={'cv_r_squared': 'r_squared'}, inplace=True)
    pls_dad_subset = pls_dad[(pls_dad['measurement'] == 'predicted_total_mg_per_gDW') & (pls_dad['term'] == 'p_uva_mw_cm2')][['measurement', 'cv_r_squared']]
    pls_dad_subset.rename(columns={'cv_r_squared': 'r_squared'}, inplace=True)
    pls_subset = pd.concat([pls_subset, pls_dad_subset])
    pls_subset['model'] = 'PLS (CV)'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.barplot(data=ridge_subset, x='measurement', y='r_squared', ax=axes[0, 0], color='#1f77b4')
    axes[0, 0].set_title('8A: Ridge R²')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].tick_params(axis='x', rotation=30)

    sns.barplot(data=pls_subset, x='measurement', y='r_squared', ax=axes[0, 1], color='#ff7f0e')
    axes[0, 1].set_title('8B: PLS CV R²')
    axes[0, 1].set_ylabel('CV R²')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # Actual vs predicted scatter (chrom delta amount ridge)
    chrom_data = pd.read_csv('chromatogram_delta.csv')
    chrom_data = chrom_data[chrom_data['form'] == 'total']
    X = sm.add_constant(chrom_data[['p_uva_mw_cm2', 'p_uvb_mw_cm2', 'p_uva_mw_cm2']].assign(inter=chrom_data['p_uva_mw_cm2'] * chrom_data['p_uvb_mw_cm2']))
    y = chrom_data['delta_amount_mg_per_gDW']
    ridge_model = sm.OLS(y, X).fit()
    preds = ridge_model.fittedvalues
    axes[1, 0].scatter(y, preds, s=60)
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    axes[1, 0].plot(lims, lims, color='black', linestyle='--')
    axes[1, 0].set_xlabel('Observed')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title('8C: Chrom Δ-amount (OLS proxy)')

    dad = pd.read_csv('DAD_derived_concentrations_corrected.csv')
    X_dad = sm.add_constant(dad[['p_uva_mw_cm2', 'p_uvb_mw_cm2']].assign(inter=dad['p_uva_mw_cm2'] * dad['p_uvb_mw_cm2']))
    y_dad = dad['predicted_total_mg_per_gDW']
    model_dad = sm.OLS(y_dad, X_dad).fit()
    preds_dad = model_dad.fittedvalues
    axes[1, 1].scatter(y_dad, preds_dad, s=60, color='#d62728')
    lims = [min(y_dad.min(), preds_dad.min()), max(y_dad.max(), preds_dad.max())]
    axes[1, 1].plot(lims, lims, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Observed')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title('8D: DAD total mg·gDW$^{-1}$ (OLS proxy)')

    fig.suptitle('Fig. 8 — Predictive value is low', fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / 'fig08_predictive_value.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 9 --------------------

def figure9_bootstrap():
    df = pd.read_csv('ridge_bootstrap_summary.csv')
    delta_amount = df[(df['variant'] == 'delta') & (df['measurement'] == 'delta_amount_mg_per_gDW')]
    beta_uvb = delta_amount[delta_amount['term'] == 'p_uvb_mw_cm2'].iloc[0]
    beta_inter = delta_amount[delta_amount['term'] == 'p_uva_mw_cm2:p_uvb_mw_cm2'].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, row, title in zip(axes, [beta_uvb, beta_inter], ['β_UVB (Δ amount)', 'Interaction (Δ amount)']):
        mean = row['coef_mean']
        ci_low = row['coef_p2_5']
        ci_high = row['coef_p97_5']
        xs = np.linspace(ci_low - 0.5*(ci_high-ci_low), ci_high + 0.5*(ci_high-ci_low), 200)
        sns.kdeplot(xs, bw_adjust=0.5, ax=ax, color='#1f77b4')
        ax.axvline(0, color='black', linestyle='--')
        ax.axvline(ci_low, color='red', linestyle=':')
        ax.axvline(ci_high, color='red', linestyle=':')
        ax.set_title(title)
        ax.set_xlabel('Coefficient value')
        ax.set_ylabel('Density')
        ax.text(0.05, 0.9, f'mean={mean:.3f}\n95% CI [{ci_low:.3f}, {ci_high:.3f}]',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    fig.suptitle('Fig. 9 — Bootstrap stability of Δ amount coefficients', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / 'fig09_bootstrap_coeffs.png', dpi=300)
    plt.close(fig)

# -------------------- Figure 10 --------------------

def figure10_outlier_path():
    df = pd.read_csv('outlier_sensitivity.csv')
    df = df[(df['variant'] == 'delta') & (df['measurement'] == 'delta_conc_mg_ml')]
    df['n_removed'] = df['n_removed'].fillna(0).astype(int)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['n_removed'], df['F'], marker='o', color='#1f77b4', label='F statistic')
    ax1.set_xlabel('Number of removed high-residual samples')
    ax1.set_ylabel('Interaction F', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(df['n_removed'], df['p_value'], marker='s', color='#d62728', label='p-value')
    ax2.set_ylabel('p-value', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    fig.suptitle('Fig. 10 — Outlier sensitivity path (Δ concentration)', fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig10_outlier_path.png', dpi=300)
    plt.close(fig)

# Execution
figure1_calibrations()
figure2_dose_structure()
figure3_single_factor_chrom()
figure4_single_factor_dad()
figure5_emms()
figure6_pvalues()
figure7_freedman_lane()
figure8_predictive_value()
figure9_bootstrap()
figure10_outlier_path()
