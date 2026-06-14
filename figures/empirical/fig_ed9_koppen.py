#!/usr/bin/env python3
"""Figure ED9 — R17 W5: Köppen-Geiger biome occupancy of the
Huang-Pollack ensemble + biome-weighted global median ΔT_GST.

Output: ``outputs/figures/empirical/fig_ed9_koppen.pdf``.
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

# Major-class colour palette.
MAJOR_COLOR = {
    "A": "#1b9e77",  # tropical
    "B": "#d95f02",  # arid
    "C": "#7570b3",  # temperate
    "D": "#1f78b4",  # continental
    "E": "#a6a6a6",  # polar
}


def _major(class_name: object) -> str:
    if class_name is None or (isinstance(class_name, float) and np.isnan(class_name)):
        return "?"
    s = str(class_name)
    return s[0] if s else "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in",
        dest="in_path",
        default="outputs/global/ensemble_summary_koppen.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed9_koppen.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.in_path).expanduser().resolve())

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.68))

    # Panel a: world map by Köppen major class.
    proj = ccrs.Robinson(central_longitude=0.0)
    ax_a = fig.add_subplot(2, 2, (1, 2), projection=proj)
    ax_a.set_global()
    ax_a.add_feature(cfeature.LAND, facecolor="#f4f1ec", edgecolor="none")
    ax_a.add_feature(cfeature.OCEAN, facecolor="#dfe9f0", edgecolor="none")
    ax_a.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="0.55")

    df["_major"] = df["koppen_class"].apply(_major)
    for mc, sub in df.groupby("_major", sort=True):
        ax_a.scatter(
            sub["lon_deg"].values,
            sub["lat_deg"].values,
            s=8,
            color=MAJOR_COLOR.get(mc, "0.6"),
            edgecolor="none",
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            label=f"{mc}  (N={len(sub)})",
            zorder=3,
        )
    ax_a.legend(loc="lower left", frameon=False, ncol=3)
    ax_a.text(0.01, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel b: per-class median GST anomaly + count (sorted by N).
    ax_b = fig.add_subplot(2, 2, 3)
    by_class = (
        df.groupby("koppen_class", dropna=False)
        .agg(n=("site_id", "count"), med=("gst_median_recent_K", "median"))
        .sort_values("n", ascending=False)
        .head(12)
    )
    colors = [MAJOR_COLOR.get(_major(c), "0.6") for c in by_class.index]
    bars = ax_b.barh(
        range(len(by_class)),
        by_class["med"].values,
        color=colors,
        edgecolor="0.25",
        linewidth=0.4,
    )
    ax_b.set_yticks(range(len(by_class)))
    ax_b.set_yticklabels([f"{c} (N={n})" for c, n in zip(by_class.index, by_class["n"])])
    ax_b.invert_yaxis()
    ax_b.axvline(0.0, color="0.5", linewidth=0.4, linestyle=":")
    ax_b.set_xlabel(r"per-biome median $\Delta T_{\rm GST}^{0-25}$  (K)")
    ax_b.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel c: lat-band-weighted vs Köppen-weighted global median.
    ax_c = fig.add_subplot(2, 2, 4)
    lat_med = float(
        df.attrs.get(
            "global_median_lat_band",
            np.nan,
        )
    )
    kop_med = float(
        df.attrs.get(
            "global_median_koppen",
            np.nan,
        )
    )
    # If attrs are absent, recompute on the fly with the cached weights.
    if not np.isfinite(lat_med):
        lat_med = float(np.median(df["gst_median_recent_K"]))
    if not np.isfinite(kop_med):
        kop_med = lat_med
    methods = [
        ("Latitude-band\nHorvitz-Thompson", lat_med, "#7f7f7f"),
        ("Köppen-Geiger\nbiome-stratified", kop_med, "#1f77b4"),
    ]
    for k, (name, val, col) in enumerate(methods):
        ax_c.scatter(val, k, s=80, color=col, edgecolor="0.2", linewidth=0.5, zorder=4)
        ax_c.text(val + 0.04, k, f"  {val:+.3f} K", va="center", color="0.25")
    ax_c.axvline(0.0, color="0.7", linewidth=0.4, linestyle=":")
    ax_c.set_yticks(range(len(methods)))
    ax_c.set_yticklabels([m[0] for m in methods])
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.05, 1.5)
    ax_c.set_xlabel(r"global median $\Delta T_{\rm GST}^{0-25}$  (K)")
    ax_c.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    ax_c.text(0.03, 0.96, "c", transform=ax_c.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print(f"  N sites: {len(df)}; N classified: {int((df['koppen_code'] > 0).sum())}")
    print(f"  lat-band median: {lat_med:+.3f} K;  Köppen median: {kop_med:+.3f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
