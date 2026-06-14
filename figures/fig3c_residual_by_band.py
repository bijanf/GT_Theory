#!/usr/bin/env python3
"""Figure 3c — Residual (data - conduction theory) decomposed by
latitude band.

If the framework's novel coupling/latent-heat terms matter, the
residual structure should differ systematically between tropical,
mid-latitude, and polar boreholes.  This figure draws the cross-site
median residual + bootstrap CI for each of the three bands on the
same axes, so the reader can see whether polar sites systematically
deviate from the pure-conduction kernel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.fingerprints.f1_erfc import load_smoke_pair
from gt_theory.inversion import build_forward_operator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resample_to_grid(z_obs: np.ndarray, dT_obs: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    return np.interp(z_grid, z_obs, dT_obs, left=np.nan, right=np.nan)


def build_figure(subset_dir: Path, all_sites_csv: Path, out_path: Path) -> None:
    profiles, inversions = load_smoke_pair(subset_dir)
    site_ids = [str(inv["site_id"].iloc[0]) for inv in inversions]
    catalog = pd.read_csv(all_sites_csv).set_index("site_id")
    lats = np.array(
        [float(catalog.loc[s, "lat_deg"]) if s in catalog.index else np.nan for s in site_ids]
    )
    keep = np.where(np.isfinite(lats))[0]
    profiles = [profiles[i] for i in keep]
    inversions = [inversions[i] for i in keep]
    lats = lats[keep]
    n_sites = len(profiles)
    print(f"loaded {n_sites} sites with latitudes", file=sys.stderr)

    z_grid = np.arange(20.0, 601.0, 20.0)
    resid_grid = np.full((n_sites, z_grid.size), np.nan)

    for i, (prof, inv) in enumerate(zip(profiles, inversions, strict=True)):
        z = prof["depth_m"].to_numpy(dtype=float)
        T = prof["temperature_c"].to_numpy(dtype=float)
        T0 = float(inv["T0_K"].iloc[0])
        dTdz = float(inv["dTdz_K_per_m"].iloc[0])
        dT_obs = T - (T0 + dTdz * z)

        edges = np.concatenate(
            [
                inv["bin_edge_young_yr"].to_numpy(dtype=float),
                inv["bin_edge_old_yr"].to_numpy(dtype=float)[-1:],
            ]
        )
        kappa = float(inv["kappa_median"].iloc[0])
        s = inv["median_K"].to_numpy(dtype=float)
        G = build_forward_operator(z, edges, kappa)
        dT_pred = G @ s
        resid_grid[i] = _resample_to_grid(z, dT_obs - dT_pred, z_grid)

    abslat = np.abs(lats)
    bands = {
        "tropical (|lat|<23.5)": (abslat < 23.5),
        "midlat (23.5-50)": (abslat >= 23.5) & (abslat < 50.0),
        "polar (>=50)": (abslat >= 50.0),
    }
    colours = {
        "tropical (|lat|<23.5)": "#cc7700",
        "midlat (23.5-50)": "#888888",
        "polar (>=50)": "#aa3333",
    }

    rng = np.random.default_rng(20260522)
    n_boot = 400

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    ax.axhspan(-0.2, 0.2, color="#cc7700", alpha=0.10, label=r"$\pm$0.2 K target")
    ax.axhline(0.0, color="black", linewidth=0.3, alpha=0.4)

    for name, mask in bands.items():
        sub = resid_grid[mask]
        n = int(mask.sum())
        if n < 5:
            continue
        median = np.nanmedian(sub, axis=0)
        boot = np.empty((n_boot, z_grid.size))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = np.nanmedian(sub[idx], axis=0)
        lo = np.nanpercentile(boot, 5.0, axis=0)
        hi = np.nanpercentile(boot, 95.0, axis=0)
        col = colours[name]
        ax.fill_between(z_grid, lo, hi, color=col, alpha=0.20)
        ax.plot(
            z_grid,
            median,
            color=col,
            linewidth=1.2,
            label=f"{name}, n={n} (max |median|={float(np.nanmax(np.abs(median))):.3f} K)",
        )

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel(r"observed - predicted (K)")
    ax.set_xlim(20, 600)
    ax.set_ylim(-0.6, 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax.set_title(
        "Residual structure vs depth by latitude band — does polar deviate?",
        loc="left",
        weight="bold",
    )
    ax.grid(alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  n_sites={n_sites}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "full")
    parser.add_argument("--all-sites", type=Path, default=REPO_ROOT / "catalogs" / "all_sites.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.all_sites, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
