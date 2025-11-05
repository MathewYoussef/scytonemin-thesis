import csv
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = BASE_DIR / "_mplconfig_cache"
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("TMPDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N_BOOT = 500
SEED = 20240907

METRICS_GDW = [
    "chrom_total_mg_per_gDW",
    "chrom_oxidized_mg_per_gDW",
    "chrom_reduced_mg_per_gDW",
    "dad_total_mg_per_gDW",
    "dad_oxidized_mg_per_gDW",
    "dad_reduced_mg_per_gDW",
]

METRICS_ML = [
    "chrom_total_mg_ml",
    "dad_total_mg_ml",
]

ALL_METRICS = METRICS_GDW + METRICS_ML


def mad(values: list[float]) -> float:
    median = statistics.median(values)
    return 1.4826 * statistics.median([abs(x - median) for x in values])


def huber(values: list[float], c: float = 1.345, tol: float = 1e-6, max_iter: int = 50) -> float:
    """Iteratively reweighted Huber location estimate using MAD for scale."""
    if not values:
        return float("nan")
    mu = statistics.mean(values)
    scale = mad(values)
    if scale == 0.0:
        spread = max(abs(v - mu) for v in values)
        scale = max(spread, 1e-6)
    for _ in range(max_iter):
        weights = []
        for v in values:
            diff = v - mu
            if abs(diff) <= c * scale:
                weights.append(1.0)
            else:
                weights.append((c * scale) / abs(diff))
        new_mu = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        if abs(new_mu - mu) < tol:
            mu = new_mu
            break
        mu = new_mu
        abs_res = [abs(v - mu) for v in values]
        new_scale = 1.4826 * statistics.median(abs_res)
        if new_scale > 0:
            scale = new_scale
    return mu


def bootstrap_huber(values: list[float], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return [float("nan")] * N_BOOT
    res = []
    for _ in range(N_BOOT):
        sample = rng.choice(values, size=n, replace=True)
        res.append(huber(sample.tolist()))
    return res


def load_replicates(path: Path) -> dict[float, dict]:
    by_dose = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            uva = float(row["p_uva_mw_cm2"])
            uvb = float(row["p_uvb_mw_cm2"])
            ratio = row["uva_uvb_ratio"]
            ratio_val = float(ratio) if ratio else None
            slot = by_dose.setdefault(
                uva,
                {"uvb": uvb, "ratio": ratio_val, "rows": []},
            )
            slot["rows"].append({metric: float(row[metric]) for metric in ALL_METRICS})
    return dict(sorted(by_dose.items()))


def compute_summary(by_dose: dict[float, dict]):
    summary_rows = []
    bootstrap_map = defaultdict(list)  # (metric, dose_index) -> list
    for idx, (uva, info) in enumerate(by_dose.items()):
        uvb = info["uvb"]
        ratio = info["ratio"]
        rows = info["rows"]
        n_reps = len(rows)
        metrics_stats = {}
        for metric in ALL_METRICS:
            values = [r[metric] for r in rows]
            location = huber(values)
            dispersion = mad(values)
            boots = bootstrap_huber(values, SEED + idx * 37 + hash(metric) % 7919)
            boot_mean = statistics.mean(boots)
            boot_sd = statistics.stdev(boots) if len(set(boots)) > 1 else 0.0
            boot_var = boot_sd ** 2
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
            metrics_stats[metric] = {
                "huber": location,
                "mad": dispersion,
                "boot_sd": boot_sd,
                "boot_var": boot_var if boot_var > 0 else 1e-12,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
            bootstrap_map[(metric, idx)] = boots

        ratio_value = "" if uvb == 0 else (ratio if ratio is not None else uva / uvb)
        summary_rows.append(
            {
                "dose_id": idx + 1,
                "p_uva_mw_cm2": uva,
                "p_uvb_mw_cm2": uvb,
                "uva_uvb_ratio": ratio_value,
                "n_reps": n_reps,
                **{
                    f"{metric}_mean_huber": metrics_stats[metric]["huber"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_mad": metrics_stats[metric]["mad"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_boot_sd": metrics_stats[metric]["boot_sd"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_boot_var": metrics_stats[metric]["boot_var"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_ci_low": metrics_stats[metric]["ci_low"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_ci_high": metrics_stats[metric]["ci_high"]
                    for metric in METRICS_GDW
                },
                **{
                    f"{metric}_mean_huber": metrics_stats[metric]["huber"]
                    for metric in METRICS_ML
                },
                **{
                    f"{metric}_mad": metrics_stats[metric]["mad"]
                    for metric in METRICS_ML
                },
                **{
                    f"{metric}_boot_sd": metrics_stats[metric]["boot_sd"]
                    for metric in METRICS_ML
                },
                **{
                    f"{metric}_boot_var": metrics_stats[metric]["boot_var"]
                    for metric in METRICS_ML
                },
                **{
                    f"{metric}_ci_low": metrics_stats[metric]["ci_low"]
                    for metric in METRICS_ML
                },
                **{
                    f"{metric}_ci_high": metrics_stats[metric]["ci_high"]
                    for metric in METRICS_ML
                },
            }
        )
    return summary_rows, bootstrap_map


def write_dose_summary(path: Path, summary_rows):
    fieldnames = [
        "dose_id",
        "p_uva_mw_cm2",
        "p_uvb_mw_cm2",
        "uva_uvb_ratio",
        "n_reps",
    ]
    for metric in METRICS_GDW:
        fieldnames.extend(
            [
                f"{metric}_mean_huber",
                f"{metric}_mad",
                f"{metric}_boot_sd",
                f"{metric}_boot_var",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
            ]
        )
    for metric in METRICS_ML:
        fieldnames.extend(
            [
                f"{metric}_mean_huber",
                f"{metric}_mad",
                f"{metric}_boot_sd",
                f"{metric}_boot_var",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
            ]
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def compute_correlations(summary_rows, output_path: Path):
    fieldnames = [
        "metric",
        "axis",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "n_doses",
    ]
    uva_values = [row["p_uva_mw_cm2"] for row in summary_rows]
    uvb_values = [row["p_uvb_mw_cm2"] for row in summary_rows]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRICS_GDW + METRICS_ML:
            values = [row[f"{metric}_mean_huber"] for row in summary_rows]
            for axis_name, axis_values in [("UVA", uva_values), ("UVB", uvb_values)]:
                r, p = stats.pearsonr(axis_values, values)
                rs, ps = stats.spearmanr(axis_values, values)
                tau, ptau = stats.kendalltau(axis_values, values)
                writer.writerow(
                    {
                        "metric": metric,
                        "axis": axis_name,
                        "pearson_r": r,
                        "pearson_p": p,
                        "spearman_r": rs,
                        "spearman_p": ps,
                        "kendall_tau": tau,
                        "kendall_p": ptau,
                        "n_doses": len(axis_values),
                    }
                )


def wls(axis_values, metric_values, weights):
    x = np.asarray(axis_values)
    y = np.asarray(metric_values)
    w = np.asarray(weights)
    X = np.column_stack([np.ones_like(x), x])
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta
    df = len(x) - X.shape[1]
    rss = float(np.sum(w * resid ** 2))
    sigma2 = rss / df
    cov_beta = sigma2 * np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(cov_beta))
    slope = beta[1]
    intercept = beta[0]
    slope_se = se[1]
    intercept_se = se[0]
    t_val = slope / slope_se
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
    y_bar = np.sum(w * y) / np.sum(w)
    tss = float(np.sum(w * (y - y_bar) ** 2))
    r_squared = 1 - rss / tss if tss else 0.0
    tcrit = stats.t.ppf(0.975, df)
    slope_ci_low = slope - tcrit * slope_se
    slope_ci_high = slope + tcrit * slope_se
    return intercept, slope, intercept_se, slope_se, slope_ci_low, slope_ci_high, r_squared, p_val


def compute_regressions(summary_rows, output_path: Path):
    fieldnames = [
        "metric",
        "axis",
        "slope",
        "slope_se",
        "slope_ci_low",
        "slope_ci_high",
        "intercept",
        "intercept_se",
        "r_squared",
        "p_value",
        "n_doses",
    ]
    uva_values = [row["p_uva_mw_cm2"] for row in summary_rows]
    uvb_values = [row["p_uvb_mw_cm2"] for row in summary_rows]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRICS_GDW + METRICS_ML:
            values = [row[f"{metric}_mean_huber"] for row in summary_rows]
            vars_ = [row[f"{metric}_boot_var"] for row in summary_rows]
            weights = [1.0 / var if var > 0 else 1e6 for var in vars_]
            for axis_name, axis_values in [("UVA", uva_values), ("UVB", uvb_values)]:
                intercept, slope, intercept_se, slope_se, ci_low, ci_high, r_sq, p_val = wls(
                    axis_values, values, weights
                )
                writer.writerow(
                    {
                        "metric": metric,
                        "axis": axis_name,
                        "slope": slope,
                        "slope_se": slope_se,
                        "slope_ci_low": ci_low,
                        "slope_ci_high": ci_high,
                        "intercept": intercept,
                        "intercept_se": intercept_se,
                        "r_squared": r_sq,
                        "p_value": p_val,
                        "n_doses": len(axis_values),
                    }
                )


def compute_poly(summary_rows, output_path: Path):
    fieldnames = [
        "metric",
        "coef_intercept",
        "coef_linear",
        "coef_quadratic",
        "p_linear",
        "p_quadratic",
        "r_squared",
        "n_doses",
    ]
    uva_values = np.asarray([row["p_uva_mw_cm2"] for row in summary_rows])
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRICS_GDW:
            y = np.asarray([row[f"{metric}_mean_huber"] for row in summary_rows])
            vars_ = np.asarray([row[f"{metric}_boot_var"] for row in summary_rows])
            weights = np.asarray([1.0 / var if var > 0 else 1e6 for var in vars_])
            X = np.column_stack([np.ones_like(uva_values), uva_values, uva_values ** 2])
            XtWX = X.T @ (weights[:, None] * X)
            XtWy = X.T @ (weights * y)
            beta = np.linalg.solve(XtWX, XtWy)
            resid = y - X @ beta
            df = len(uva_values) - X.shape[1]
            rss = float(np.sum(weights * resid ** 2))
            sigma2 = rss / df
            cov_beta = sigma2 * np.linalg.inv(XtWX)
            se = np.sqrt(np.diag(cov_beta))
            t_vals = beta / se
            p_vals = [2 * (1 - stats.t.cdf(abs(t), df)) for t in t_vals]
            y_bar = np.sum(weights * y) / np.sum(weights)
            tss = float(np.sum(weights * (y - y_bar) ** 2))
            r_squared = 1 - rss / tss if tss else 0.0
            writer.writerow(
                {
                    "metric": metric,
                    "coef_intercept": beta[0],
                    "coef_linear": beta[1],
                    "coef_quadratic": beta[2],
                    "p_linear": p_vals[1],
                    "p_quadratic": p_vals[2],
                    "r_squared": r_squared,
                    "n_doses": len(uva_values),
                }
            )


def compute_step_differences(summary_rows, bootstrap_map, output_path: Path):
    fieldnames = [
        "metric",
        "step",
        "from_uva",
        "from_uvb",
        "to_uva",
        "to_uvb",
        "diff_mean",
        "diff_ci_low",
        "diff_ci_high",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRICS_GDW:
            for i in range(len(summary_rows) - 1):
                mean_diff = summary_rows[i + 1][f"{metric}_mean_huber"] - summary_rows[i][
                    f"{metric}_mean_huber"
                ]
                boots_from = bootstrap_map[(metric, i)]
                boots_to = bootstrap_map[(metric, i + 1)]
                diffs = [b_to - b_from for b_from, b_to in zip(boots_from, boots_to)]
                ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
                writer.writerow(
                    {
                        "metric": metric,
                        "step": i + 1,
                        "from_uva": summary_rows[i]["p_uva_mw_cm2"],
                        "from_uvb": summary_rows[i]["p_uvb_mw_cm2"],
                        "to_uva": summary_rows[i + 1]["p_uva_mw_cm2"],
                        "to_uvb": summary_rows[i + 1]["p_uvb_mw_cm2"],
                        "diff_mean": mean_diff,
                        "diff_ci_low": float(ci_low),
                        "diff_ci_high": float(ci_high),
                    }
                )


def deming(x, y, lam=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    s_x = np.sum((x - x_bar) ** 2) / len(x)
    s_y = np.sum((y - y_bar) ** 2) / len(y)
    s_xy = np.sum((x - x_bar) * (y - y_bar)) / len(x)
    term = s_y - lam * s_x
    slope = (term + math.sqrt(term ** 2 + 4 * lam * s_xy ** 2)) / (2 * s_xy)
    intercept = y_bar - slope * x_bar
    return slope, intercept


def compute_alignment(summary_rows, bootstrap_map, output_path: Path):
    fieldnames = [
        "chrom_metric",
        "dad_metric",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "deming_lambda",
        "deming_slope",
        "deming_intercept",
        "deming_slope_ci_low",
        "deming_slope_ci_high",
        "deming_intercept_ci_low",
        "deming_intercept_ci_high",
        "n_doses",
    ]
    metric_pairs = [
        ("chrom_total_mg_per_gDW", "dad_total_mg_per_gDW"),
        ("chrom_oxidized_mg_per_gDW", "dad_oxidized_mg_per_gDW"),
        ("chrom_reduced_mg_per_gDW", "dad_reduced_mg_per_gDW"),
    ]
    doses = len(summary_rows)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chrom_metric, dad_metric in metric_pairs:
            chrom_values = np.asarray([row[f"{chrom_metric}_mean_huber"] for row in summary_rows])
            dad_values = np.asarray([row[f"{dad_metric}_mean_huber"] for row in summary_rows])
            pearson_r, pearson_p = stats.pearsonr(chrom_values, dad_values)
            spearman_r, spearman_p = stats.spearmanr(chrom_values, dad_values)
            kendall_tau, kendall_p = stats.kendalltau(chrom_values, dad_values)

            chrom_vars = np.asarray([row[f"{chrom_metric}_boot_var"] for row in summary_rows])
            dad_vars = np.asarray([row[f"{dad_metric}_boot_var"] for row in summary_rows])
            lam = (
                float(np.mean(chrom_vars) / np.mean(dad_vars))
                if np.mean(dad_vars) > 0
                else 1.0
            )
            slope, intercept = deming(chrom_values, dad_values, lam=lam)

            slope_boot = []
            intercept_boot = []
            for i in range(N_BOOT):
                boot_chrom = np.array([bootstrap_map[(chrom_metric, idx)][i] for idx in range(doses)])
                boot_dad = np.array([bootstrap_map[(dad_metric, idx)][i] for idx in range(doses)])
                s, b = deming(boot_chrom, boot_dad, lam=lam)
                slope_boot.append(s)
                intercept_boot.append(b)
            slope_ci_low, slope_ci_high = np.percentile(slope_boot, [2.5, 97.5])
            intercept_ci_low, intercept_ci_high = np.percentile(intercept_boot, [2.5, 97.5])

            writer.writerow(
                {
                    "chrom_metric": chrom_metric,
                    "dad_metric": dad_metric,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "kendall_tau": kendall_tau,
                    "kendall_p": kendall_p,
                    "deming_lambda": lam,
                    "deming_slope": slope,
                    "deming_intercept": intercept,
                    "deming_slope_ci_low": slope_ci_low,
                    "deming_slope_ci_high": slope_ci_high,
                    "deming_intercept_ci_low": intercept_ci_low,
                    "deming_intercept_ci_high": intercept_ci_high,
                    "n_doses": doses,
                }
            )


def plot_strip(summary_rows, by_dose, output_path: Path):
    dose_ids = list(range(1, len(summary_rows) + 1))
    uva_labels = [f"{row['p_uva_mw_cm2']:.3f}" for row in summary_rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()

    metric_panels = [
        ("chrom_total_mg_per_gDW", "Chromatogram Total"),
        ("chrom_oxidized_mg_per_gDW", "Chromatogram Oxidized"),
        ("chrom_reduced_mg_per_gDW", "Chromatogram Reduced"),
        ("dad_total_mg_per_gDW", "DAD Total"),
        ("dad_oxidized_mg_per_gDW", "DAD Oxidized"),
        ("dad_reduced_mg_per_gDW", "DAD Reduced"),
    ]

    for ax, (metric, title) in zip(axes, metric_panels):
        for idx, (uva, info) in enumerate(by_dose.items()):
            replicates = [row[metric] for row in info["rows"]]
            jitter = np.linspace(-0.15, 0.15, len(replicates))
            ax.scatter(
                np.full(len(replicates), idx + 1) + jitter,
                replicates,
                color="#555555",
                alpha=0.7,
                s=30,
            )
            row = summary_rows[idx]
            mean_val = row[f"{metric}_mean_huber"]
            ci_low = row[f"{metric}_ci_low"]
            ci_high = row[f"{metric}_ci_high"]
            ax.errorbar(
                idx + 1,
                mean_val,
                yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                fmt="o",
                color="#1f77b4",
                capsize=4,
                linewidth=2,
            )
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xlim(0.5, len(summary_rows) + 0.5)

    for ax in axes[-3:]:
        ax.set_xlabel("UVA dose (mW·cm⁻²)")
        ax.set_xticks(dose_ids)
        ax.set_xticklabels(uva_labels, rotation=45, ha="right")

    for ax in axes[::3]:
        ax.set_ylabel("mg·gDW⁻¹")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    base = BASE_DIR
    replicate_path = base / "Combined_Scytonemin_Concentrations.csv"
    by_dose = load_replicates(replicate_path)

    summary_rows, bootstrap_map = compute_summary(by_dose)
    write_dose_summary(base / "dose_summary.csv", summary_rows)

    compute_correlations(summary_rows, base / "dose_level_correlations.csv")
    compute_regressions(summary_rows, base / "dose_level_regressions.csv")
    compute_poly(summary_rows, base / "dose_level_poly_regressions.csv")
    compute_step_differences(summary_rows, bootstrap_map, base / "dose_level_step_differences.csv")
    compute_alignment(summary_rows, bootstrap_map, base / "chrom_dad_alignment.csv")

    plot_strip(summary_rows, by_dose, base / "fig_dose_means.png")
    # Duplicate export to maintain legacy filename
    plot_strip(summary_rows, by_dose, base / "fig_dose_means_faceted.png")


if __name__ == "__main__":
    main()
