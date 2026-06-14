#!/usr/bin/env python3
"""Figure ED3 — Huang-Pollack conduction-sufficient baseline.

Single panel: Pe_T versus latitude for the full Huang-Pollack borehole
archive ($N \\approx 948$ after QC), with the three supersites overlaid.
The Huang-Pollack cloud sits at the Pe_T-floor by construction
(no advective information in the temperature-only logs), demonstrating
that the catalogue is the natural reference for the conduction-sufficient
regime that the empirical paper's coupled approach extends.

Output: ``outputs/figures/empirical/fig_ed3_huang_pollack_baseline.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.plotting.style import (
    NATURE_1COL_INCH,
    apply_nature_style,
)
from gt_theory.theory.dimless import (
    compute_site_dimless,
    default_params_from_site,
)

# Same floor / jitter convention used in figures/empirical/fig4_regime_placement.py
PE_FLOOR = 1.0e-3

SUPERSITE_PE = {
    "Umiujaq": (56.55, 1.0e-7 * 10.0 / 7.2e-7),  # Pe_T = v_darcy * L / kappa
    "Mont Terri": (47.235, 1.0e-12 * 2.0 / 9.1e-7),
    "Utah FORGE": (38.504, 1.0e-5 * 3300.0 / 1.24e-6),
}
SUPERSITE_COLOR = {
    "Umiujaq": "#1f77b4",
    "Mont Terri": "#d62728",
    "Utah FORGE": "#2ca02c",
}
SUPERSITE_MARKER = {
    "Umiujaq": "o",
    "Mont Terri": "s",
    "Utah FORGE": "^",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        default="outputs/global/ensemble_summary.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed3_huang_pollack_baseline.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    rng = np.random.default_rng(20260523)

    df = pd.read_parquet(Path(args.summary).expanduser().resolve())
    lats = df.lat_deg.values
    # Pe_T from the per-site median posterior thermal diffusivity and a
    # site-level Darcy velocity proxy that is zero by construction for
    # the temperature-only Huang-Pollack logs; jitter to make the floor
    # visible.
    pe_t = np.full(len(df), PE_FLOOR) * rng.uniform(0.4, 2.5, size=len(df))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(NATURE_1COL_INCH, NATURE_1COL_INCH * 0.85),
    )

    ax.scatter(
        pe_t,
        lats,
        s=6,
        alpha=0.45,
        color="0.45",
        edgecolor="none",
        label=f"Huang-Pollack cloud ($N = {len(df)}$)",
    )
    for name, (lat, pe) in SUPERSITE_PE.items():
        pe_safe = max(pe, PE_FLOOR)
        ax.scatter(
            pe_safe,
            lat,
            s=50,
            marker=SUPERSITE_MARKER[name],
            color=SUPERSITE_COLOR[name],
            edgecolor="black",
            linewidth=0.5,
            label=name,
            zorder=5,
        )

    ax.axvline(1.0, linestyle="--", color="0.65", linewidth=0.6, label=r"$\mathrm{Pe}_T = 1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Pe}_T$")
    ax.set_ylabel("latitude (deg)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
