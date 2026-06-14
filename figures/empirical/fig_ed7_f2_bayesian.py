#!/usr/bin/env python3
"""Figure ED7 — R17 W3: hierarchical Bayesian EIV posterior for the
F2 SAT-GST coupling slope, contrasted with OLS and Deming.

Output: ``outputs/figures/empirical/fig_ed7_f2_bayesian.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


BAND_COLORS = {
    "tropical": "#d62728",
    "mid": "#1f77b4",
    "boreal": "#2ca02c",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--draws",
        default="outputs/global/f2_bayesian_eiv.parquet",
    )
    parser.add_argument(
        "--pairs",
        default="outputs/global/f2_bayesian_eiv_pairs.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed7_f2_bayesian.pdf",
    )
    args = parser.parse_args(argv)

    draws = pd.read_parquet(args.draws)
    pairs = pd.read_parquet(args.pairs)

    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42),
    )

    # Panel a: per-band posterior on beta_g + global hyperparameter.
    ax_a = axes[0]
    band_labels = ("tropical", "mid", "boreal")
    bins = np.linspace(0.5, 3.5, 60)
    for label in band_labels:
        ax_a.hist(
            draws[f"beta_{label}"],
            bins=bins,
            color=BAND_COLORS[label],
            alpha=0.55,
            edgecolor="0.25",
            linewidth=0.3,
            label=f"$\\beta_{{\\rm {label}}}$",
        )
    ax_a.axvline(
        np.median(draws["beta_global"]),
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=rf"$\beta_{{\rm global}}$ median = {np.median(draws['beta_global']):.2f}",
    )
    ax_a.set_xlabel(r"posterior slope  $\beta$")
    ax_a.set_ylabel("posterior density (counts)")
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_a.text(0.03, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel b: comparison of OLS / Deming / EIV.
    ax_b = axes[1]
    ols = float(draws.attrs.get("ols_slope", np.nan))
    deming_l1 = float(draws.attrs.get("deming_lambda_1", np.nan))
    deming_l4 = float(draws.attrs.get("deming_lambda_4", np.nan))
    eiv_med = float(np.median(draws["beta_global"]))
    eiv_lo = float(np.quantile(draws["beta_global"], 0.05))
    eiv_hi = float(np.quantile(draws["beta_global"], 0.95))

    # Clip Deming values to a finite axis range so the figure still
    # communicates "Deming exploded".
    clip_lo, clip_hi = -4.0, 4.0
    methods = [
        ("OLS", ols, None, "#7f7f7f"),
        (r"Deming $\lambda=1$", deming_l1, None, "#bcbd22"),
        (r"Deming $\lambda=4$", deming_l4, None, "#ff7f0e"),
        ("Bayes EIV", eiv_med, (eiv_lo, eiv_hi), "#1f77b4"),
    ]
    for k, (name, val, ci, col) in enumerate(methods):
        plot_val = np.clip(val, clip_lo, clip_hi)
        ax_b.scatter(plot_val, k, s=60, color=col, edgecolor="0.2", linewidth=0.5, zorder=4)
        if ci is not None:
            ax_b.plot([ci[0], ci[1]], [k, k], color=col, linewidth=2, zorder=3)
            ax_b.text(
                ci[1] + 0.07,
                k,
                f"  90% CI [{ci[0]:.2f}, {ci[1]:.2f}]",
                va="center",
                color=col,
            )
        else:
            label = f"  {val:.2f}" if abs(val) < 100 else f"  {val:.0f}"
            ax_b.text(
                plot_val + 0.07,
                k,
                label,
                va="center",
                color="0.25",
            )
            if abs(val) > clip_hi:
                ax_b.annotate(
                    "(clipped)",
                    (plot_val, k),
                    xytext=(plot_val + 0.07, k - 0.25),
                    color="0.4",
                    fontsize=6,
                )
    ax_b.axvline(0.0, color="0.7", linewidth=0.4, linestyle=":")
    ax_b.set_yticks(range(len(methods)))
    ax_b.set_yticklabels([name for name, *_ in methods])
    ax_b.set_xlim(clip_lo, clip_hi)
    ax_b.set_xlabel(r"slope estimate  $\widehat{\beta}$")
    ax_b.invert_yaxis()
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.6, axis="x")
    ax_b.text(0.03, 0.96, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print(
        f"  EIV beta_global posterior median = {eiv_med:.3f} (90% CI [{eiv_lo:.3f}, {eiv_hi:.3f}])"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
