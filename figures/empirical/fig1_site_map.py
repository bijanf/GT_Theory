#!/usr/bin/env python3
r"""Figure 1 — the three-supersite world map.

Northern-hemisphere-biased global view (Plate Carrée centred on
-40 deg E) with three dots: Umiujaq (talik, primary), Mont Terri
HE-D (indurated clay, secondary), Utah FORGE 16A (granite EGS,
secondary).

Each dot is colour-coded by its dominant transport regime
(advection vs conduction) from the framework's Pe_T classification.

Output: ``outputs/figures/empirical/fig1_site_map.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


SITES = [
    {
        "label": "Umiujaq",
        "lat": 56.55,
        "lon": -76.55,
        "color": "#1f77b4",
        "marker": "o",
        "regime": "advection-dominated",
    },
    {
        "label": "Mont Terri HE-D",
        "lat": 47.235,
        "lon": 7.155,
        "color": "#d62728",
        "marker": "s",
        "regime": "pressure-dominated",
    },
    {
        "label": "Utah FORGE 16A",
        "lat": 38.504,
        "lon": -112.896,
        "color": "#2ca02c",
        "marker": "^",
        "regime": "high-Pe$_T$ injection",
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig1_site_map.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()

    proj = ccrs.PlateCarree(central_longitude=-40.0)
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.set_extent([-130.0, 30.0, 25.0, 75.0], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f4f1ec", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#dfe9f0", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.5")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.7")

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linestyle=":",
        linewidth=0.3,
        color="0.6",
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6}
    gl.ylabel_style = {"size": 6}

    for s in SITES:
        ax.scatter(
            s["lon"],
            s["lat"],
            s=70,
            marker=s["marker"],
            color=s["color"],
            edgecolor="black",
            linewidth=0.6,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        # Annotate with a leader line offset.
        ax.annotate(
            f"{s['label']}\n[{s['regime']}]",
            xy=(s["lon"], s["lat"]),
            xytext=(s["lon"] + 6.0, s["lat"] + 4.5),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            color="0.15",
            ha="left",
            arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.4),
        )

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
