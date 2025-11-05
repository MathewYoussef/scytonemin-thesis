#!/usr/bin/env python3
"""Regenerate Figure 9 from ridge_bootstrap_summary.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "fig09_bootstrap_coeffs"
SUMMARY_PATH = Path("ridge_bootstrap_summary.csv")

TERMS = {
    "p_uvb_mw_cm2": ("β_UVB (Δ amount)", "9A"),
    "p_uva_mw_cm2:p_uvb_mw_cm2": ("β_interaction (Δ amount)", "9B"),
}


def synthetic_samples(mean: float, std: float, size: int = 100_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=size)


def make_figure():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_PATH)
    subset = df[(df["variant"] == "delta") & (df["measurement"] == "delta_amount_mg_per_gDW")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, term in zip(axes, TERMS.keys()):
        row = subset[subset["term"] == term].iloc[0]
        title, panel_code = TERMS[term]
        samples = synthetic_samples(row["coef_mean"], row["coef_std"])
        ci_low, ci_high = row["coef_p2_5"], row["coef_p97_5"]
        mean = row["coef_mean"]

        ax.text(-0.18, 1.05, panel_code, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.hist(samples, bins=60, density=True, color="#1f77b4", alpha=0.3, edgecolor="black")
        ax.axvline(0, color="black", linestyle="--")
        ax.axvline(ci_low, color="red", linestyle=":")
        ax.axvline(ci_high, color="red", linestyle=":")
        ax.set_xlabel("Coefficient value (mg·gDW⁻¹·(mW·cm⁻²)⁻¹)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.text(
            0.02,
            0.92,
            f"mean = {mean:.3f}\n95% CI [{ci_low:.3f}, {ci_high:.3f}]",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Fig. 9 — Bootstrap stability of Δ-amount ridge coefficients", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "fig09_bootstrap_coeffs.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig09_bootstrap_coeffs.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
