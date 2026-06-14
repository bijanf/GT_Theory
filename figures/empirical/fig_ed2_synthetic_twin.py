#!/usr/bin/env python3
"""Figure ED2 — synthetic-twin posterior diagnostic.

5x5 corner-plot lite: marginal histograms on the diagonal, pairwise
scatter on the lower triangle, posterior-correlation heatmap on the
upper triangle. Truth values marked with crosshairs on each panel.

Reads ``outputs/supersite_umiujaq/posterior_samples.npz`` (produced
by ``scripts/invert_umiujaq_coupled.py --synthetic``). The smoke
budget used 6 walkers × 50 post-burn samples — sparse, but enough
to demonstrate the inverter recovers the ΓN_α truth inside the 90 %
CI.

Output: ``outputs/figures/empirical/fig_ed2_synthetic_twin.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="outputs/supersite_umiujaq/posterior_samples.npz",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed2_synthetic_twin.pdf",
    )
    args = parser.parse_args(argv)

    data = np.load(Path(args.input).expanduser().resolve())
    chains = data["chains"]  # (n_walkers, n_steps, n_params)
    truth = data["truth"]  # (n_params,)
    names = [str(s) for s in data["param_names"]]

    flat = chains.reshape(-1, chains.shape[-1])
    n_params = flat.shape[1]

    # Short labels for axes.
    label_map = {
        "log10_K_hyd_m_s": r"$\log_{10} K_{\mathrm{hyd}}$",
        "porosity": r"$\phi$",
        "lambda_th_W_m_K": r"$\lambda_{\mathrm{th}}$",
        "gst_offset_K": r"$\Delta T_{\mathrm{GST}}$",
        "gamma_n_alpha_scale": r"$s$",
    }
    labels = [label_map.get(n, n) for n in names]

    apply_nature_style()

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH),
        sharex=False,
        sharey=False,
    )

    corr = np.corrcoef(flat.T)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                # Marginal histogram + truth marker + 90% CI shading.
                ax.hist(flat[:, i], bins=15, color="#888", alpha=0.7)
                ax.axvline(truth[i], color="#c0392b", linewidth=1.2)
                lo, hi = np.quantile(flat[:, i], [0.05, 0.95])
                ax.axvspan(lo, hi, color="#1f77b4", alpha=0.12)
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: scatter.
                ax.scatter(flat[:, j], flat[:, i], s=4, alpha=0.55, color="0.35", edgecolor="none")
                ax.axvline(truth[j], color="#c0392b", linewidth=0.6, alpha=0.7)
                ax.axhline(truth[i], color="#c0392b", linewidth=0.6, alpha=0.7)
            else:
                # Upper triangle: correlation cell.
                r = corr[i, j]
                ax.set_facecolor(plt.cm.RdBu_r(0.5 + r / 2.0))
                color = "white" if abs(r) > 0.5 else "black"
                ax.text(
                    0.5,
                    0.5,
                    f"{r:+.2f}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=color,
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            # Axis labels only on outer edge.
            if i == n_params - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            else:
                if not (i == j or i < j):
                    ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=6)

    # Tag the diagonal as "marginal", lower as "joint", upper as "corr".
    fig.text(
        0.02,
        0.50,
        "joint posterior (lower) · correlation (upper)",
        rotation=90,
        va="center",
        ha="center",
        color="0.4",
    )

    fig.tight_layout(rect=(0.02, 0, 1, 1))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
