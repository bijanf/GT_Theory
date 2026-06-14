#!/usr/bin/env python3
"""Figure 1 (hero schematic) -- the coupled column.

A three-zone orientation figure: the coupled physical system (a) is
discretised by the cell-centred finite-volume scheme with a sequential
Picard loop (b), and each simulated column lands in a regime of the
dimensionless-number taxonomy (c).  Pure schematic (no data file), drawn
to Nature-family spec (vector, sans-serif, <=7 pt, RGB, 180 mm).

Output: ``outputs/figures/numerical/fig01_schematic.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from gt_theory.plotting.style import NATURE_2COL_INCH, apply_nature_style

CASE_COLOURS = {
    "permafrost": "#1f77b4",
    "geothermal": "#d62728",
    "arid": "#7f7f7f",
    "thermo": "#2ca02c",
}
FROZEN = "#cfe3f2"
UNFROZEN = "#efe2cf"
ICE = "#9ecae1"
CELL_HI = "#fde9d0"
INK = "#222222"


def _arrow(ax, xy0, xy1, color=INK, lw=1.0, style="-|>", mut=7, alpha=1.0):
    ax.add_patch(
        FancyArrowPatch(
            xy0,
            xy1,
            arrowstyle=style,
            mutation_scale=mut,
            lw=lw,
            color=color,
            alpha=alpha,
            zorder=5,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _panel_a_column(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")
    cx0, cx1 = 0.30, 0.74  # column x-extent
    top, bot = 0.24, 0.95  # column y-extent (0 = surface row)
    front = 0.42  # freezing front depth (fraction)

    # Column body: frozen (upper) over unfrozen (lower).
    ax.add_patch(
        Rectangle(
            (cx0, top), cx1 - cx0, front - top, facecolor=FROZEN, edgecolor=INK, lw=1.0, zorder=2
        )
    )
    ax.add_patch(
        Rectangle(
            (cx0, front),
            cx1 - cx0,
            bot - front,
            facecolor=UNFROZEN,
            edgecolor=INK,
            lw=1.0,
            zorder=2,
        )
    )
    # Freezing front.
    ax.plot([cx0, cx1], [front, front], color="#3b7dd8", lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax.text(
        cx1 + 0.015,
        front,
        "freezing front\n" r"$S_i(T)$, Eq.",
        fontsize=5.6,
        color="#3b7dd8",
        va="center",
        ha="left",
    )
    # Ice crystals (active layer).
    rng_x = np.linspace(cx0 + 0.04, cx1 - 0.04, 6)
    rng_y = np.linspace(top + 0.04, front - 0.03, 3)
    for yy in rng_y:
        for xx in rng_x:
            ax.plot(xx, yy, marker="*", ms=3.0, color=ICE, zorder=3)

    # Surface forcing (down arrows), spread to keep labels apart.
    for xx, lab in [(0.40, r"$T_{\rm surf}(t)$"), (0.64, r"$p_{\rm top}(t)$")]:
        _arrow(ax, (xx, top - 0.115), (xx, top - 0.008), lw=1.1)
        ax.text(xx, top - 0.135, lab, fontsize=5.8, ha="center", va="bottom")

    # Darcy through-flow (upward arrow inside the column).
    _arrow(ax, (0.46, bot - 0.05), (0.46, front + 0.06), color="#1f6fb0", lw=1.6, mut=9)
    ax.text(
        0.485,
        (front + bot) / 2,
        r"$v_{\rm Darcy}$" "\nDarcy flux",
        fontsize=5.8,
        color="#1f6fb0",
        va="center",
        ha="left",
    )

    # Geothermal flux at base (up arrows).
    for xx in (0.40, 0.54, 0.66):
        _arrow(ax, (xx, bot + 0.085), (xx, bot + 0.01), color="#b03030", lw=1.0)
    ax.text(
        (cx0 + cx1) / 2,
        bot + 0.10,
        r"$q_{\rm geo}$  (geothermal flux)",
        fontsize=5.8,
        color="#b03030",
        ha="center",
        va="top",
    )

    # Pore-scale callout (water + ice).
    ax.text(
        cx0 - 0.02,
        (top + front) / 2,
        "frozen\n(water + ice)",
        fontsize=5.4,
        color="0.35",
        ha="right",
        va="center",
    )
    ax.text(
        cx0 - 0.02,
        (front + bot) / 2,
        "unfrozen\n(saturated)",
        fontsize=5.4,
        color="0.35",
        ha="right",
        va="center",
    )

    ax.text(0.02, 0.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.set_title("physical system", fontsize=7, pad=4)


def _panel_b_fvm(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("finite-volume discretisation", fontsize=7, pad=2)

    # Cell stack (4 cells), middle one highlighted.  The stack is kept in
    # the upper half of the panel so the Picard flow-diagram below has room.
    cx0, cx1 = 0.10, 0.50
    ys = np.linspace(0.15, 0.56, 5)  # 4 cells -> 5 edges
    for k in range(4):
        y0, y1 = ys[k], ys[k + 1]
        fc = CELL_HI if k == 2 else "white"
        ax.add_patch(
            Rectangle((cx0, y0), cx1 - cx0, y1 - y0, facecolor=fc, edgecolor=INK, lw=1.0, zorder=2)
        )
    yc = 0.5 * (ys[2] + ys[3])  # centre of highlighted cell i
    xf = (cx0 + cx1) / 2
    # Cell-centre unknowns: dot + label, both kept inside cell i and to the
    # left so they never cross the advection arrow at the right.
    ax.plot(cx0 + 0.10, yc, "o", ms=4, color=INK, zorder=4)
    ax.text(cx0 + 0.135, yc, r"$T_i,\,p_i,\,S_i$", fontsize=5.6, va="center", ha="left", zorder=4)
    # Neighbour indices, placed inside the neighbouring cells (left side).
    ax.text(
        cx0 + 0.025,
        0.5 * (ys[1] + ys[2]),
        r"$i-1$",
        fontsize=5.6,
        va="center",
        ha="left",
        color="0.4",
    )
    ax.text(
        cx0 + 0.025,
        0.5 * (ys[3] + ys[4]),
        r"$i+1$",
        fontsize=5.6,
        va="center",
        ha="left",
        color="0.4",
    )

    # Diffusive (harmonic-mean) face flux: blue arrow on the left face of
    # cell i, with the face transport coefficients labelled to the far left.
    _arrow(ax, (cx0 + 0.04, ys[2]), (cx0 + 0.04, ys[3]), color="#1f6fb0", lw=1.2, mut=7)
    ax.text(
        cx0 - 0.015,
        ys[2],
        r"$\lambda_{i-1/2},\,M_{i-1/2}$",
        fontsize=5.0,
        ha="right",
        va="center",
        color="#1f6fb0",
    )
    ax.text(
        cx0 - 0.015,
        ys[3],
        r"$\lambda_{i+1/2},\,M_{i+1/2}$",
        fontsize=5.0,
        ha="right",
        va="center",
        color="#1f6fb0",
    )
    # Upwind advective flux: red arrow just to the right of the stack,
    # spanning the column, with its label clear of the cells.
    xadv = cx1 + 0.05
    _arrow(ax, (xadv, ys[4] - 0.005), (xadv, ys[1] + 0.005), color="#b03030", lw=1.2, mut=8)
    ax.text(
        xadv + 0.03,
        yc,
        r"$J^{\rm adv}_{i\pm1/2}$" "\n(upwind)",
        fontsize=5.0,
        va="center",
        ha="left",
        color="#b03030",
    )
    ax.text(
        xf,
        ys[0] - 0.02,
        "harmonic-mean faces",
        fontsize=5.4,
        color="0.4",
        ha="center",
        va="bottom",
        style="italic",
    )

    # Sequential Picard loop (flow diagram) below the stack.
    def _box(cxc, cyc, w, h, text, fc="white"):
        ax.add_patch(
            FancyBboxPatch(
                (cxc - w / 2, cyc - h / 2),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=fc,
                edgecolor=INK,
                lw=0.8,
                zorder=3,
            )
        )
        ax.text(cxc, cyc, text, fontsize=5.3, ha="center", va="center", zorder=4)

    yb = 0.72
    bh = 0.13
    _box(0.19, yb, 0.28, bh, "mass eq.\n" r"$\to p^{n+1}$", fc="#eaf2fb")
    _box(0.52, yb, 0.26, bh, "face fluxes\n" r"$q_{i+1/2}$")
    _box(0.84, yb, 0.26, bh, "energy eq.\n" r"$\to T^{n+1}$", fc="#fdecea")
    _arrow(ax, (0.33, yb), (0.39, yb), lw=0.9)
    _arrow(ax, (0.65, yb), (0.71, yb), lw=0.9)
    # Loop-back arrow, routed in a clear lane BELOW the boxes (no overlap).
    ax.add_patch(
        FancyArrowPatch(
            (0.84, yb + bh / 2 + 0.015),
            (0.19, yb + bh / 2 + 0.015),
            connectionstyle="arc3,rad=-0.30",
            arrowstyle="-|>",
            mutation_scale=7,
            lw=0.9,
            color="0.45",
            zorder=3,
        )
    )
    ax.text(
        0.515,
        0.975,
        "sequential Picard "
        r"(two tridiagonal solves), backward Euler",
        fontsize=5.2,
        color="0.45",
        ha="center",
        va="center",
    )
    ax.text(0.02, 0.06, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _panel_c_regime(ax):
    ax.set_xlim(-0.08, 1.04)
    ax.set_ylim(-0.08, 1.10)
    ax.axis("off")
    ax.set_title("regime taxonomy", fontsize=7, pad=2)
    # Plane box with four quadrants.
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, lw=1.0))
    ax.axvline(0.5, ymin=0.0, ymax=1.0 / 1.18, color="0.4", ls="--", lw=0.7)
    ax.axhline(0.5, xmin=0.0, xmax=1.0 / 1.12, color="0.4", ls=":", lw=0.7)
    ax.text(0.5, -0.06, r"$\mathrm{Pe}_T$  (advection)", fontsize=6, ha="center", va="top")
    ax.text(
        -0.06,
        0.5,
        r"$\mathcal{L}$  (latent heat)",
        fontsize=6,
        rotation=90,
        ha="right",
        va="center",
    )
    # Quadrant tints.
    ax.add_patch(Rectangle((0, 0), 0.5, 1, facecolor="#f0f4f8", zorder=0))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 1, facecolor="#fbf2ec", zorder=0))
    # Four case dots (schematic positions matching the atlas).
    pts = {
        "permafrost": (0.12, 0.80, "permafrost"),
        "thermo": (0.20, 0.34, "thermo-poro"),
        "arid": (0.34, 0.30, "arid"),
        "geothermal": (0.83, 0.18, "geothermal"),
    }
    for key, (x, y, lab) in pts.items():
        ax.plot(
            x, y, "o", ms=5, color=CASE_COLOURS[key], markeredgecolor="black", mew=0.5, zorder=4
        )
        ax.text(
            x + 0.03, y + 0.04, lab, fontsize=5.4, color=CASE_COLOURS[key], ha="left", va="bottom"
        )
    qkw = dict(fontsize=5.2, color="0.45", style="italic", ha="center")
    ax.text(0.25, 0.95, "latent", **qkw)
    ax.text(0.75, 0.95, "latent+adv.", **qkw)
    ax.text(0.75, 0.05, "advection", **qkw)
    ax.text(0.02, 1.0, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig01_schematic.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 0.85], wspace=0.12)
    _panel_a_column(fig.add_subplot(gs[0, 0]))
    _panel_b_fvm(fig.add_subplot(gs[0, 1]))
    _panel_c_regime(fig.add_subplot(gs[0, 2]))
    # Stage arrows between panels.
    fig.text(0.355, 0.52, r"$\Rightarrow$", fontsize=13, color="0.5", ha="center", va="center")
    fig.text(0.665, 0.52, r"$\Rightarrow$", fontsize=13, color="0.5", ha="center", va="center")
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
