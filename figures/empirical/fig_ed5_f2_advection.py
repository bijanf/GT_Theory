#!/usr/bin/env python3
"""Figure ED5 — R17 W1: per-site advection-induced shift in
ground-heat-content vs. the conduction-only baseline.

Output: ``outputs/figures/empirical/fig_ed5_f2_advection.pdf``.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in",
        dest="in_path",
        default="outputs/global/f2_advection_on.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed5_f2_advection.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.in_path).expanduser().resolve())

    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42),
    )

    # Panel a: per-site RMS T(z) profile difference, advection on - off.
    ax_a = axes[0]
    rms_K = df["rms_T_diff_K"].values
    ax_a.hist(
        rms_K,
        bins=40,
        color="#1f77b4",
        edgecolor="0.25",
        linewidth=0.4,
    )
    med = float(np.nanmedian(rms_K))
    p95 = float(np.nanquantile(rms_K, 0.95))
    ax_a.axvline(med, color="#c0392b", linewidth=0.9, linestyle="--", label=f"median = {med:.4f} K")
    ax_a.axvline(p95, color="#c0392b", linewidth=0.9, linestyle=":", label=f"p95 = {p95:.4f} K")
    ax_a.set_xlabel(r"per-site RMS  $T_{s=1} - T_{s=0}$  (K)")
    ax_a.set_ylabel("number of sites")
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_a.text(0.03, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel b: column-integrated GHC shift, %.
    ax_b = axes[1]
    pct = 100.0 * df["ghc_shift_fraction"].values
    ax_b.hist(
        pct,
        bins=40,
        color="#2ca02c",
        edgecolor="0.25",
        linewidth=0.4,
    )
    med_pct = float(np.nanmedian(pct))
    p95_pct = float(np.nanquantile(np.abs(pct), 0.95))
    ax_b.axvline(
        med_pct, color="#c0392b", linewidth=0.9, linestyle="--", label=f"median = {med_pct:.2f} %"
    )
    ax_b.axvline(
        p95_pct, color="#c0392b", linewidth=0.9, linestyle=":", label=f"|p95| = {p95_pct:.2f} %"
    )
    ax_b.axvline(-p95_pct, color="#c0392b", linewidth=0.9, linestyle=":")
    ax_b.set_xlabel("GHC shift advection-on vs off  (%)")
    ax_b.set_ylabel("number of sites")
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print(f"  N sites: {len(df)}")
    print(f"  N with GGMN match: {int(df['has_ggmn_match'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
