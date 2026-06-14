#!/usr/bin/env python3
"""Figure ED8 — R17 W4: F5 diffusive-lag distribution at 5 m and
20 m, unfiltered vs Butterworth-bandpassed (3-30 yr) cross-correlation.

Output: ``outputs/figures/empirical/fig_ed8_f5_lag.pdf``.
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
from gt_theory.stats.signal import theoretical_diffusive_lag_months


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in",
        dest="in_path",
        default="outputs/global/f5_bandpass.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed8_f5_lag.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.in_path).expanduser().resolve())

    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42),
    )

    kappa_med = float(df["kappa_m2_per_s"].median())
    theory_5m = theoretical_diffusive_lag_months(
        depth_m=5.0,
        kappa_m2_per_s=kappa_med,
    )
    theory_20m = theoretical_diffusive_lag_months(
        depth_m=20.0,
        kappa_m2_per_s=kappa_med,
    )

    # Panel a: 5 m lag distribution.
    ax_a = axes[0]
    bins_5m = np.arange(-12, 121, 3)
    ax_a.hist(
        df["lag_5m_unfiltered_months"],
        bins=bins_5m,
        alpha=0.55,
        color="#7f7f7f",
        edgecolor="0.3",
        linewidth=0.3,
        label=f"unfiltered (median {df['lag_5m_unfiltered_months'].median():.0f} mo)",
    )
    ax_a.hist(
        df["lag_5m_filtered_months"],
        bins=bins_5m,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="0.3",
        linewidth=0.3,
        label=f"bandpass (median {df['lag_5m_filtered_months'].median():.0f} mo)",
    )
    ax_a.axvline(
        theory_5m,
        color="#c0392b",
        linewidth=1.0,
        linestyle="--",
        label=rf"$\tau_{{\rm theory}}={theory_5m:.0f}$ mo",
    )
    ax_a.set_xlabel("F5 lag at $z = 5$ m  (months)")
    ax_a.set_ylabel("number of sites")
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_a.text(0.03, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel b: 20 m lag distribution.
    ax_b = axes[1]
    bins_20m = np.arange(-12, 121, 4)
    ax_b.hist(
        df["lag_20m_unfiltered_months"],
        bins=bins_20m,
        alpha=0.55,
        color="#7f7f7f",
        edgecolor="0.3",
        linewidth=0.3,
        label=f"unfiltered (median {df['lag_20m_unfiltered_months'].median():.0f} mo)",
    )
    ax_b.hist(
        df["lag_20m_filtered_months"],
        bins=bins_20m,
        alpha=0.55,
        color="#2ca02c",
        edgecolor="0.3",
        linewidth=0.3,
        label=f"bandpass (median {df['lag_20m_filtered_months'].median():.0f} mo)",
    )
    ax_b.axvline(
        theory_20m,
        color="#c0392b",
        linewidth=1.0,
        linestyle="--",
        label=rf"$\tau_{{\rm theory}}={theory_20m:.0f}$ mo",
    )
    ax_b.set_xlabel("F5 lag at $z = 20$ m  (months)")
    ax_b.set_ylabel("number of sites")
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}  ({len(df)} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
