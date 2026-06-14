#!/usr/bin/env python3
"""Figure 8 (hero) -- the Regime Atlas.

A single finite-volume solver traverses the whole regime plane: the
four canonical case studies are placed at their solver-computed
coordinates on the heat-transport plane (Pe_T, L), and each is fused
to a framed inset thumbnail of its own defining field, so that
parameter-space position is welded to representative behaviour.  A
companion strip carries the mass-transport plane (N_p, Gamma N_alpha).

Panel a (hero): (Pe_T, L) heat-transport plane with signature-field
  insets -- permafrost ice-saturation active layer; geothermal
  advective temperature plume; arid-basin conduction; thermo-poro
  thermal overpressure pulse.
Panel b: (N_p, Gamma N_alpha) mass-transport plane.

Output: ``outputs/figures/numerical/fig08_regime_placement.pdf``
(same filename as the figure it supersedes, so the manuscript include
is unchanged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)

YEAR_S = 365.25 * 86400.0

CASE_COLOURS = {
    "permafrost": "#1f77b4",  # blue
    "geothermal": "#d62728",  # red
    "arid_basin": "#7f7f7f",  # grey
    "thermo_poro": "#2ca02c",  # green
}
CASE_LABELS = {
    "permafrost": "permafrost",
    "geothermal": "geothermal",
    "arid_basin": "arid basin",
    "thermo_poro": "thermo-poro",
}

PE_FLOOR = 1.4e-5  # display floor for Pe_T -> 0 cases on the log axis

# Per-case signature field for the inset thumbnail: (nc var, cmap, short label).
SIGNATURE = {
    "permafrost": ("S_i", "Blues", r"$S_i$"),
    "geothermal": ("T_degC", "inferno", r"$T$"),
    "arid_basin": ("T_degC", "inferno", r"$T$"),
    "thermo_poro": ("p_Pa", "viridis", r"$p$"),
}

# Inset rectangles in axes-fraction coords [x0, y0, w, h], one per case,
# placed in the quadrant whose physics it shows (and clear of its marker).
INSET_BOX = {
    "permafrost": [0.050, 0.650, 0.265, 0.300],  # upper-left (latent)
    "thermo_poro": [0.050, 0.080, 0.265, 0.300],  # lower-left (coupling)
    "arid_basin": [0.388, 0.080, 0.265, 0.300],  # lower-mid (conduction)
    "geothermal": [0.705, 0.520, 0.265, 0.300],  # upper-right, above its marker
}
# Per-case label offset (points) and horizontal anchor, to keep text clear
# of frames and the plot edge.
LABEL_OFFSET = {
    "permafrost": ((7, 6), "left"),
    "thermo_poro": ((7, 6), "left"),
    "arid_basin": ((7, 6), "left"),
    "geothermal": ((-7, 6), "right"),
}


def _load_field(case: str, nc_dir: Path):
    var, cmap, lab = SIGNATURE[case]
    ds = xr.open_dataset(nc_dir / f"{case}.nc")
    field = ds[var].values  # (time, depth)
    t_yr = ds["time"].values / YEAR_S
    depth = ds["depth_m"].values
    return t_yr, depth, field, cmap, lab


def _add_inset(ax, case: str, marker_xy, nc_dir: Path) -> None:
    t_yr, depth, field, cmap, lab = _load_field(case, nc_dir)
    box = INSET_BOX[case]
    axin = ax.inset_axes(box)
    axin.pcolormesh(t_yr, depth, field.T, cmap=cmap, shading="auto", rasterized=True)
    axin.invert_yaxis()
    axin.set_xticks([])
    axin.set_yticks([])
    for sp in axin.spines.values():
        sp.set_edgecolor(CASE_COLOURS[case])
        sp.set_linewidth(1.1)
    # Tiny field-type tag inside the inset.
    axin.text(
        0.06,
        0.10,
        lab,
        transform=axin.transAxes,
        fontsize=6,
        color="white",
        fontweight="bold",
        path_effects=None,
        ha="left",
        va="bottom",
    )
    # Thin leader from the marker (data coords) to the inset centre.
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    ax.annotate(
        "",
        xy=marker_xy,
        xycoords="data",
        xytext=(cx, cy),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-", color=CASE_COLOURS[case], lw=0.5, alpha=0.65, shrinkA=2, shrinkB=2
        ),
        zorder=2,
    )


def _panel_atlas(ax, df: pd.DataFrame, nc_dir: Path) -> None:
    # Soft regime-quadrant tints.
    ax.axvspan(1e-5, 1.0, ymin=0, ymax=1, color="#f0f4f8", zorder=0)
    ax.fill_between([1.0, 1e3], 1e-3, 1e2, color="#fbf2ec", zorder=0)
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.6, zorder=1)
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=0.6, zorder=1)

    for _, row in df.iterrows():
        c = row["case"]
        pe = max(float(row["Pe_T"]), PE_FLOOR)
        L = float(row["L_calL"])
        _add_inset(ax, c, (pe, L), nc_dir)
        ax.scatter(pe, L, s=95, color=CASE_COLOURS[c], edgecolor="black", linewidth=0.7, zorder=5)
        (dx, dy), ha = LABEL_OFFSET[c]
        ax.annotate(
            CASE_LABELS[c],
            (pe, L),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            ha=ha,
            fontweight="bold",
            color=CASE_COLOURS[c],
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1e3)
    ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel(
        r"$\mathrm{Pe}_T = v_{\rm Darcy}\,L / \kappa$"
        "   (advection / conduction)"
    )
    ax.set_ylabel(
        r"$\mathcal{L} = \rho_i L_f \phi / [(\rho c)_{\rm eff}\,\Delta T]$"
        "   (latent / sensible heat)"
    )
    qkw = dict(fontsize=6.5, color="0.45", ha="center", va="center", style="italic", zorder=1)
    ax.text(4e-2, 35, "latent-heat\ndominated", **qkw)
    ax.text(1.5e2, 4e-3, "advection-\ndominated", **qkw)
    # The conduction-dominated quadrant (lower-left, blue tint) holds three
    # of the four cases, so it is left unlabelled to avoid overprinting; the
    # tint and the x-axis "(advection / conduction)" label carry it.
    ax.text(0.02, 0.975, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.grid(True, which="major", linestyle=":", linewidth=0.3, alpha=0.4)


def _panel_mass(ax, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        c = row["case"]
        ax.scatter(
            row["N_p"],
            row["Gamma_N_alpha"],
            s=70,
            color=CASE_COLOURS[c],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            CASE_LABELS[c],
            (row["N_p"], row["Gamma_N_alpha"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6,
            color=CASE_COLOURS[c],
        )
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 1e-2)
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlabel(r"$N_p = \phi \beta_w \Delta p$")
    ax.set_ylabel(r"$\Gamma N_\alpha = \alpha_w \Delta T / (\beta_w \Delta p)$")
    ax.text(
        2e-7,
        200,
        "coupling\ndominated",
        fontsize=6,
        color="0.45",
        style="italic",
        ha="left",
        va="center",
    )
    ax.text(
        2e-7,
        0.3,
        "thermal feedback\nnegligible",
        fontsize=6,
        color="0.45",
        style="italic",
        ha="left",
        va="center",
    )
    ax.text(0.06, 0.975, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.grid(True, which="major", linestyle=":", linewidth=0.3, alpha=0.4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="outputs/cases/dimless_summary.csv")
    parser.add_argument("--nc-dir", default="outputs/cases")
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig08_regime_placement.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(Path(args.summary).expanduser().resolve())
    nc_dir = Path(args.nc_dir).expanduser().resolve()

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.83))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 0.28], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_atlas(ax_a, df, nc_dir)
    _panel_mass(ax_b, df)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
