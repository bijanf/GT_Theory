#!/usr/bin/env python3
"""Figure 0 (hero) — global 948-site Huang-Pollack ensemble + three
co-located T+p supersites.

Three panels, Nature 2-col layout:

a. Robinson world map: 948 H-P posterior sites coloured by
   posterior-median recent (0-25 yr) GST anomaly. Three supersites
   overplotted as crossed circles (Umiujaq, Mont Terri, Utah FORGE).
b. Posterior recent GST anomaly vs absolute latitude with the
   5th-95th-percentile envelope. Confirms the expected tropical
   amplification.
c. Recent GST anomaly vs depth-averaged steady-state thermal
   gradient -- the proxy for advective heat input. Sites in the
   upper-right corner are the high-Pe_T candidates where the
   framework predicts ΓN_α should matter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


SUPERSITES = [
    {"label": "Umiujaq", "lat": 56.55, "lon": -76.55, "color": "#1f77b4"},
    {"label": "Mont Terri", "lat": 47.235, "lon": 7.155, "color": "#d62728"},
    {"label": "Utah FORGE", "lat": 38.504, "lon": -112.896, "color": "#2ca02c"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        default="outputs/global/ensemble_summary.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig0_global_map.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.summary).expanduser().resolve())

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.62))

    # ---- panel a: world map ----
    proj = ccrs.Robinson(central_longitude=0.0)
    ax_a = fig.add_subplot(2, 2, (1, 2), projection=proj)
    ax_a.set_global()
    ax_a.add_feature(cfeature.LAND, facecolor="#f4f1ec", edgecolor="none")
    ax_a.add_feature(cfeature.OCEAN, facecolor="#dfe9f0", edgecolor="none")
    ax_a.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="0.55")

    vmin, vmax = -3.0, 3.0
    sc = ax_a.scatter(
        df.lon_deg.values,
        df.lat_deg.values,
        c=df.gst_median_recent_K.values,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=8,
        edgecolor="none",
        alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    for s in SUPERSITES:
        ax_a.scatter(
            s["lon"],
            s["lat"],
            s=110,
            marker="o",
            color="white",
            edgecolor=s["color"],
            linewidth=1.3,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax_a.scatter(
            s["lon"],
            s["lat"],
            s=110,
            marker="+",
            color=s["color"],
            linewidth=1.3,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    cb = fig.colorbar(sc, ax=ax_a, orientation="horizontal", shrink=0.55, pad=0.05, aspect=40)
    cb.set_label(r"posterior $\Delta T_{\rm GST}^{0-25\,\rm yr}$  (K)")
    # Force a vector-not-raster colourbar gradient: replace the
    # default rasterised solid with a many-stop discrete fill.
    cb.solids.set_rasterized(False)

    ax_a.text(0.01, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Headline annotation: lead the eye to the +0.78 K median.
    median_recent = float(np.nanmedian(df.gst_median_recent_K.values))
    ax_a.text(
        0.99,
        0.03,
        f"global median $\\Delta T_{{\\rm GST}}^{{0-25\\,\\rm yr}} = "
        f"{median_recent:+.2f}$ K  ($N = {len(df)}$)",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", linewidth=0.4, alpha=0.92
        ),
    )

    # ---- panel b: GST anomaly vs |lat| with 5-95 envelope ----
    ax_b = fig.add_subplot(2, 2, 3)
    abs_lat = np.abs(df.lat_deg.values)
    gst = df.gst_median_recent_K.values
    # Latitude bands of 10 deg.
    edges = np.arange(0, 91, 10)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(centers.size, np.nan)
    lo = np.full(centers.size, np.nan)
    hi = np.full(centers.size, np.nan)
    for k in range(centers.size):
        m = (abs_lat >= edges[k]) & (abs_lat < edges[k + 1])
        if m.sum() >= 3:
            med[k] = np.median(gst[m])
            lo[k] = np.quantile(gst[m], 0.05)
            hi[k] = np.quantile(gst[m], 0.95)
    ax_b.scatter(abs_lat, gst, s=4, alpha=0.35, color="0.45", edgecolor="none")
    ax_b.fill_between(centers, lo, hi, color="#c0392b", alpha=0.18, label="5-95 % band")
    ax_b.plot(centers, med, color="#c0392b", linewidth=1.1, label="median")
    ax_b.axhline(0.0, color="0.65", linewidth=0.5, linestyle=":")
    ax_b.set_xlabel("|latitude| (deg)")
    ax_b.set_ylabel(r"$\Delta T_{\rm GST}^{0-25\,\rm yr}$  (K)")
    ax_b.set_xlim(0.0, 80.0)
    ax_b.set_ylim(-4.0, 6.5)
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_b.legend(loc="upper left", frameon=False)
    ax_b.text(0.04, 0.96, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    # ---- panel c: GST anomaly vs geothermal gradient ----
    ax_c = fig.add_subplot(2, 2, 4)
    grad = df.geo_gradient_K_per_km.values
    # Bulk cloud: small grey circles (the conduction-dominated majority).
    ax_c.scatter(
        grad,
        gst,
        s=4,
        alpha=0.4,
        color="0.45",
        edgecolor="none",
        marker="o",
        label=r"$dT/dz \leq 50$ K km$^{-1}$ ($N = "
        f"{int((grad <= 50.0).sum())}$)",
    )
    # High-gradient sites: distinct *shape* (filled triangle) so the
    # distinction survives greyscale print, plus the red colour for
    # colour readers.
    hi_mask = grad > 50.0
    ax_c.scatter(
        grad[hi_mask],
        gst[hi_mask],
        s=22,
        color="#c0392b",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        marker="^",
        label=r"$dT/dz > 50$ K km$^{-1}$ ($N = "
        f"{int(hi_mask.sum())}$)",
    )
    ax_c.axhline(0.0, color="0.65", linewidth=0.5, linestyle=":")
    ax_c.axvline(50.0, color="0.65", linewidth=0.5, linestyle=":")
    ax_c.set_xlabel(r"steady-state $dT/dz$  (K km$^{-1}$)")
    ax_c.set_ylabel(r"$\Delta T_{\rm GST}^{0-25\,\rm yr}$  (K)")
    ax_c.set_xlim(0.0, 130.0)
    ax_c.set_ylim(-4.0, 6.5)
    ax_c.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_c.legend(loc="upper right", frameon=False)
    ax_c.text(0.04, 0.96, "c", transform=ax_c.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print(f"  N sites plotted: {len(df)}")
    print(f"  N with dT/dz > 50 K/km: {int(hi_mask.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
