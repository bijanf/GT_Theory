#!/usr/bin/env python3
"""Figure 1 — Schematic of the integrated interface dynamics framework.

Three horizontal interfaces (LSAI, SWRI, CMI) drawn as labelled bands
with arrows for the dominant mass / energy fluxes and the dimensionless
numbers that govern each band.  The figure also serves as the
schematic that the accompanying paper references.

Geometry / palette are kept close to Nature-family conventions
(sans-serif, vector PDF, 180 mm width).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = "#0066cc") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
        )
    )


def build_figure(out_path: Path) -> None:
    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()

    fig, ax = plt.subplots(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.62))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Atmosphere region (above LSAI).
    ax.add_patch(mpatches.Rectangle((0, 8.5), 10, 1.5, facecolor="#eef5ff", edgecolor="none"))
    ax.text(0.2, 9.55, "atmosphere", fontsize=6, color="#3366aa")

    # LSAI band.
    ax.add_patch(mpatches.Rectangle((0, 7.6), 10, 0.9, facecolor="#fff5e6", edgecolor="#cc7700"))
    ax.text(0.2, 8.3, "LSAI", fontsize=7, weight="bold", color="#aa5500")
    ax.text(0.2, 7.85, "Land Surface-Atmosphere Interface", fontsize=5.5, color="#aa5500")
    ax.text(8.0, 8.3, r"$N_Q,\ \mathcal{Q}$", fontsize=7, color="#aa5500")

    # SWRI band (deeper, wider in altitude).
    ax.add_patch(mpatches.Rectangle((0, 3.0), 10, 4.4, facecolor="#f5f0ff", edgecolor="#553399"))
    ax.text(0.2, 7.05, "SWRI", fontsize=7, weight="bold", color="#553399")
    ax.text(0.2, 6.65, "Sediment-Water-Rock Interface", fontsize=5.5, color="#553399")
    ax.text(8.0, 6.95, r"$\mathcal{L},\ N_s,\ \Gamma N_\alpha$", fontsize=7, color="#553399")

    # CMI band.
    ax.add_patch(mpatches.Rectangle((0, 1.5), 10, 1.4, facecolor="#ffe6e6", edgecolor="#aa3333"))
    ax.text(0.2, 2.2, "CMI", fontsize=7, weight="bold", color="#aa3333")
    ax.text(0.2, 1.75, "Crust-Mantle Interface", fontsize=5.5, color="#aa3333")
    ax.text(8.0, 2.2, r"$Pe_T \ll 1,\ Fo \ll 1$", fontsize=7, color="#aa3333")

    # Mantle region (below CMI).
    ax.add_patch(
        mpatches.Rectangle((0, 0), 10, 1.4, facecolor="#aa2222", edgecolor="none", alpha=0.4)
    )
    ax.text(0.2, 0.5, "mantle (deep geothermal reservoir)", fontsize=6, color="#ffffff")

    # Dominant fluxes.
    # SAT -> GST (top-down).
    _arrow(ax, 2.0, 9.4, 2.0, 8.5, color="#cc7700")
    ax.text(2.1, 9.0, "SAT", fontsize=5.5, color="#cc7700")
    # GST -> deep T (advection + conduction).
    _arrow(ax, 2.0, 7.55, 2.0, 5.0, color="#553399")
    ax.text(2.1, 6.3, "GST signal", fontsize=5.5, color="#553399")
    # Geothermal flux up from mantle.
    _arrow(ax, 5.0, 0.6, 5.0, 1.5, color="#aa3333")
    _arrow(ax, 5.0, 1.45, 5.0, 3.0, color="#aa3333")
    ax.text(5.1, 1.0, r"$q_\mathrm{bot}$", fontsize=6, color="#aa3333")

    # Latent-heat layer near 0 deg C (drawn as a dashed horizontal line
    # to symbolise the freezing zone in permafrost sites).
    ax.plot([0.5, 9.5], [5.7, 5.7], color="#003366", linestyle="--", linewidth=0.4)
    ax.text(0.5, 5.85, "freezing front", fontsize=5, color="#003366")

    # Darcy advection: down-arrow in SWRI.
    _arrow(ax, 7.5, 7.4, 7.5, 5.5, color="#0066cc")
    ax.text(7.55, 6.6, r"$v_\mathrm{darcy}$", fontsize=5.5, color="#0066cc")

    # Title bar.
    ax.text(
        5.0,
        9.85,
        "Integrated interface dynamics: nine dimensionless numbers across three interfaces",
        fontsize=6,
        ha="center",
        weight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
