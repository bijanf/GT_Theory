#!/usr/bin/env python3
"""Figure 2 — Global borehole map + QC tiers + latitude coverage.

Three-panel Nature-spec figure built around the 951-site Huang-Pollack
catalog at ``catalogs/all_sites.csv``:

  (a) world map with all sites coloured by QC tier (depth >= 300 m = tier 1,
      200-300 m = tier 2, < 200 m = tier 3) and the curated smoke-10
      subset highlighted in a distinct symbol;
  (b) latitude histogram in 10-deg bands with continental land-area
      Horvitz-Thompson weights overlaid;
  (c) per-country sample counts (top 12 contributors).

Usage::

    python figures/fig2_borehole_map.py --out outputs/figures/fig2_borehole_map.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Repository-relative defaults.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_CSV = REPO_ROOT / "catalogs" / "all_sites.csv"
DEFAULT_CURATED_YAML = REPO_ROOT / "catalogs" / "boreholes.yaml"

# Hard-coded HT-weight reference: continental land-area shares from the
# UN Statistics Division (rounded). Used only for the inset density
# overlay; replaced by a higher-fidelity grid when F4 lands in #23.
_LAND_AREA_KM2_PER_LAT_BAND = {
    # 10-deg lat band centre -> approximate global land area in km^2.
    -65: 12e6,
    -55: 2.5e6,
    -45: 4.5e6,
    -35: 12e6,
    -25: 18e6,
    -15: 19e6,
    -5: 17e6,
    5: 16e6,
    15: 22e6,
    25: 22e6,
    35: 20e6,
    45: 17e6,
    55: 13e6,
    65: 11e6,
    75: 5e6,
    85: 2e6,
}


def _qc_tier(max_depth_m: float) -> int:
    if max_depth_m >= 300.0:
        return 1
    if max_depth_m >= 200.0:
        return 2
    return 3


def _load_smoke_ids(curated_yaml: Path) -> set[str]:
    with curated_yaml.open() as fh:
        raw = yaml.safe_load(fh) or {}
    subsets = raw.get("subsets") or {}
    return set(subsets.get("smoke-10", []))


def _ht_weights(lats: np.ndarray, sites_per_band: pd.Series) -> pd.Series:
    """Per-site Horvitz-Thompson weight: continental land area / sample
    count in the same 10-deg lat band."""
    band_centres = ((lats // 10.0) * 10.0 + 5.0).astype(int)
    weights = []
    for c, n in zip(
        band_centres,
        sites_per_band.reindex(band_centres).fillna(0).values,
        strict=True,
    ):
        land = _LAND_AREA_KM2_PER_LAT_BAND.get(int(c), 0.0)
        weights.append(0.0 if n == 0 else land / float(n))
    return pd.Series(weights, index=lats.index)


def build_figure(catalog_csv: Path, curated_yaml: Path, out_path: Path) -> None:
    df = pd.read_csv(catalog_csv)
    df["qc_tier"] = df["max_depth_m"].apply(_qc_tier)
    df["country"] = df["country"].fillna("Unknown")

    smoke_ids = _load_smoke_ids(curated_yaml)
    smoke_mask = df["site_id"].isin(smoke_ids)

    # Apply Nature style locally without polluting global rcParams.
    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()

    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.55))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.4, 1.0),
        height_ratios=(1.0, 0.9),
        wspace=0.18,
        hspace=0.35,
    )

    # (a) world map ------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#ffffff", edgecolor="none")
    ax_map.coastlines(linewidth=0.4, color="#888888")
    ax_map.gridlines(linewidth=0.2, color="#cccccc", alpha=0.6)

    tier_colours = {1: "#0066cc", 2: "#cc7700", 3: "#aa0033"}
    for tier in (3, 2, 1):  # back-to-front so tier-1 dominates visually
        sub = df[(df["qc_tier"] == tier) & ~smoke_mask]
        ax_map.scatter(
            sub["lon_deg"],
            sub["lat_deg"],
            s=4.0,
            c=tier_colours[tier],
            edgecolors="none",
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            label=f"tier {tier} (n={(df['qc_tier'] == tier).sum()})",
        )

    smoke_df = df[smoke_mask]
    ax_map.scatter(
        smoke_df["lon_deg"],
        smoke_df["lat_deg"],
        s=20.0,
        marker="*",
        c="#222222",
        edgecolors="white",
        linewidths=0.4,
        transform=ccrs.PlateCarree(),
        label=f"smoke-10 (n={len(smoke_df)})",
        zorder=5,
    )

    ax_map.legend(loc="lower left", frameon=False, ncol=2, fontsize=5.5)
    ax_map.set_title("a   Global Huang-Pollack borehole archive", loc="left", weight="bold")

    # (b) latitude histogram with HT weights -----------------------------
    ax_lat = fig.add_subplot(gs[1, 0])
    bins = np.arange(-90, 91, 10)
    counts, _, _ = ax_lat.hist(
        df["lat_deg"],
        bins=bins,
        color="#0066cc",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.4,
    )
    band_centres = (bins[:-1] + bins[1:]) / 2.0
    land = np.array([_LAND_AREA_KM2_PER_LAT_BAND.get(int(c), 0.0) for c in band_centres])
    # Plot HT density overlay on a second y-axis.
    ax_ht = ax_lat.twinx()
    with np.errstate(divide="ignore", invalid="ignore"):
        ht_per_site = np.where(counts > 0, land / counts, np.nan)
    ax_ht.plot(band_centres, ht_per_site / 1e6, color="#cc7700", linewidth=1.0)
    ax_ht.set_ylabel(r"HT weight (10$^6$ km$^2$ site$^{-1}$)", color="#cc7700")
    ax_ht.tick_params(axis="y", colors="#cc7700")
    ax_lat.set_xlabel("Latitude (deg)")
    ax_lat.set_ylabel("Borehole count")
    ax_lat.set_xlim(-90, 90)
    ax_lat.set_title("b   Latitude coverage", loc="left", weight="bold")

    # (c) top contributors ----------------------------------------------
    ax_top = fig.add_subplot(gs[1, 1])
    top = df["country"].value_counts().head(12).iloc[::-1]
    ax_top.barh(top.index, top.values, color="#0066cc", alpha=0.8)
    ax_top.set_xlabel("count")
    ax_top.set_title("c   Top contributors", loc="left", weight="bold")
    ax_top.tick_params(axis="y", labelsize=5.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_CSV)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED_YAML)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.catalog, args.curated, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
