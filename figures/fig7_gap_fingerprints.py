#!/usr/bin/env python3
"""Figure 7 — F6 latent-heat dominance + F7 GRACE-proxy placeholder.

Three panels, Nature 180 mm:
  (a) per-site freezing-interval occupancy vs |lat|, with the
      detection threshold drawn as a horizontal line;
  (b) fraction of sites flagged as latent-heat dominant in 3 broad
      latitude bands (tropical / mid-lat / polar), with binomial
      confidence intervals;
  (c) world map of all sites coloured by latent-dominant flag,
      Robinson projection.

F7 (GRACE proxy) is documented as a planned panel and currently shows
the F6 latent-dominant map at panel (c) — the global-GRACE-TWS
download required to populate F7 properly is a follow-up.
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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _binomial_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def build_figure(occupancy_parquet: Path, out_path: Path) -> None:
    df = pd.read_parquet(occupancy_parquet)

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.1, 1.0, 1.6), wspace=0.4)

    # (a) Occupancy vs |lat|.
    ax_a = fig.add_subplot(gs[0, 0])
    abslat = df["lat_deg"].abs().to_numpy()
    occ = df["occupancy"].to_numpy()
    flag = df["latent_dominant"].to_numpy()
    ax_a.scatter(
        abslat[~flag],
        occ[~flag],
        s=4,
        c="#888888",
        alpha=0.5,
        edgecolor="none",
        label="not flagged",
    )
    ax_a.scatter(
        abslat[flag],
        occ[flag],
        s=8,
        c="#0066cc",
        edgecolor="white",
        linewidths=0.3,
        label="latent dominant",
    )
    ax_a.axhline(0.05, color="#cc7700", linewidth=0.5, linestyle="--", label="threshold")
    ax_a.set_xlabel("|Latitude| (deg)")
    ax_a.set_ylabel("freezing-interval occupancy")
    ax_a.set_yscale("symlog", linthresh=0.001)
    ax_a.set_title(f"a   per-site occupancy (n={len(df)})", loc="left", weight="bold")
    ax_a.legend(loc="upper left", frameon=False, fontsize=5.5)
    ax_a.grid(alpha=0.25, linewidth=0.3)

    # (b) Fraction flagged per lat band.
    ax_b = fig.add_subplot(gs[0, 1])
    df["band"] = pd.cut(
        df["lat_deg"].abs(),
        bins=[0, 23.5, 50, 90],
        labels=["tropical", "midlat", "polar"],
    )
    counts = df.groupby("band", observed=True).agg(
        n=("latent_dominant", "count"),
        n_dom=("latent_dominant", "sum"),
    )
    counts["frac"] = counts["n_dom"] / counts["n"]
    bands = list(counts.index)
    fracs = counts["frac"].to_numpy()
    cis = [_binomial_ci(int(counts.loc[b, "n_dom"]), int(counts.loc[b, "n"])) for b in bands]
    err_lo = np.array([f - lo for f, (lo, _hi) in zip(fracs, cis, strict=True)])
    err_hi = np.array([hi - f for f, (_lo, hi) in zip(fracs, cis, strict=True)])
    x = np.arange(len(bands))
    ax_b.bar(
        x,
        fracs,
        yerr=[err_lo, err_hi],
        capsize=4,
        color=["#cc7700", "#888888", "#0066cc"],
        edgecolor="white",
        alpha=0.85,
    )
    for i, b in enumerate(bands):
        ax_b.text(
            i,
            fracs[i] + max(0.02, err_hi[i] + 0.02),
            f"{int(counts.loc[b, 'n_dom'])}/{int(counts.loc[b, 'n'])}",
            ha="center",
            fontsize=5.5,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(bands)
    ax_b.set_ylabel("fraction latent dominant")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title("b   per-band detection rate", loc="left", weight="bold")
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.3)

    # (c) Global map of latent-dominant sites.
    ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    ax_c.set_global()
    ax_c.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax_c.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none")
    ax_c.coastlines(linewidth=0.4, color="#888888")
    ax_c.scatter(
        df.loc[~df["latent_dominant"], "lon_deg"] if "lon_deg" in df.columns else [],
        df.loc[~df["latent_dominant"], "lat_deg"] if "lon_deg" in df.columns else [],
        s=3,
        c="#cccccc",
        edgecolor="none",
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )
    if "lon_deg" in df.columns:
        ax_c.scatter(
            df.loc[df["latent_dominant"], "lon_deg"],
            df.loc[df["latent_dominant"], "lat_deg"],
            s=6,
            c="#0066cc",
            edgecolor="white",
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
        )
    ax_c.set_title(
        f"c   latent-dominant sites ({int(flag.sum())} of {len(df)})",
        loc="left",
        weight="bold",
    )

    # F7 (GRACE proxy) will populate a fourth panel once the global
    # TWS grids are downloaded (follow-up #33).  No suptitle on the
    # published PDF.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path}  n={len(df)}, dominant={int(flag.sum())} ({100 * flag.mean():.1f}%)",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=REPO_ROOT / "outputs" / "full" / "f6_occupancy.parquet",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.occupancy)
    if "lon_deg" not in df.columns:
        # Attach longitudes from the global catalog so the map panel can plot.
        all_sites = pd.read_csv(REPO_ROOT / "catalogs" / "all_sites.csv")
        df = df.merge(all_sites[["site_id", "lon_deg"]], on="site_id", how="left")
        df.to_parquet(args.occupancy, index=False)

    build_figure(args.occupancy, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
