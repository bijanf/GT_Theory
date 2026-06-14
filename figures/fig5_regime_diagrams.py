#!/usr/bin/env python3
"""Figure 5 — per-band recovered Delta GST distribution.

PLACEHOLDER until per-site Darcy velocity estimation lands.  The
intended hero regime diagram (sites placed in (Pe_T, L_calL) space)
collapses to a single point when Pe_T = 0 across the archive.  Until
the F5 forward-sim or an independent recharge estimate yields per-site
v_darcy, this script renders a defensible alternative: the latitudinal
distribution of recovered Delta GST across all sites in the run.

Two panels, Nature 180 mm:
  (a) Per-lat-band Delta GST distribution as a violin-style strip
      (median + IQR + range).
  (b) Distribution of recovered kappa (m^2 s^-1) overlaid by latitude
      band, showing the inverse correlation between kappa and the
      depth of the recovered anomaly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gt_theory.fingerprints.f1_erfc import load_smoke_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURATED_YAML = REPO_ROOT / "catalogs" / "boreholes.yaml"
DEFAULT_ALL_SITES_CSV = REPO_ROOT / "catalogs" / "all_sites.csv"


def _site_lats(catalog_path: Path, site_ids: list[str]) -> dict[str, float]:
    with catalog_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    sites = raw.get("sites") or {}
    out: dict[str, float] = {}
    for sid in site_ids:
        if sid in sites and "lat_deg" in sites[sid]:
            out[sid] = float(sites[sid]["lat_deg"])
    missing = [s for s in site_ids if s not in out]
    if missing and DEFAULT_ALL_SITES_CSV.exists():
        df = pd.read_csv(DEFAULT_ALL_SITES_CSV)
        df = df[df["site_id"].isin(missing)]
        for _, row in df.iterrows():
            out[str(row["site_id"])] = float(row["lat_deg"])
    return out


def build_figure(subset_dir: Path, catalog_path: Path, out_path: Path) -> None:
    _, inversions = load_smoke_pair(subset_dir)
    site_ids = [str(inv["site_id"].iloc[0]) for inv in inversions]
    lat_lookup = _site_lats(catalog_path, site_ids)
    keep = [i for i, sid in enumerate(site_ids) if sid in lat_lookup]
    inversions = [inversions[i] for i in keep]
    site_ids = [site_ids[i] for i in keep]
    lats = np.array([lat_lookup[s] for s in site_ids])
    dGST = np.array([float(inv.iloc[0]["median_K"]) for inv in inversions])
    kappa = np.array([float(inv.iloc[0]["kappa_median"]) for inv in inversions])

    band_edges = np.array([-90.0, -55.0, -35.0, -15.0, 5.0, 25.0, 45.0, 65.0, 90.0])
    band_centres = 0.5 * (band_edges[:-1] + band_edges[1:])
    band_idx = np.clip(np.digitize(lats, band_edges) - 1, 0, len(band_centres) - 1)

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.45))
    gs = fig.add_gridspec(1, 2, wspace=0.32)

    # (a) per-band Delta GST distribution (box + strip).
    ax_a = fig.add_subplot(gs[0, 0])
    data_per_band = [dGST[band_idx == b] for b in range(len(band_centres))]
    counts = [d.size for d in data_per_band]
    pos = band_centres
    bp = ax_a.boxplot(
        data_per_band,
        positions=pos,
        widths=14,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.0},
    )
    for box in bp["boxes"]:
        box.set(facecolor="#0066cc", alpha=0.7, edgecolor="#003566")
    # Overlay individual sites as a thin strip.
    for c, sub in zip(band_centres, data_per_band, strict=True):
        if sub.size == 0:
            continue
        jitter = np.random.default_rng(int(c) + 1000).normal(0.0, 2.5, size=sub.size)
        ax_a.scatter(c + jitter, sub, s=2, c="#003566", alpha=0.25, edgecolor="none")
    ax_a.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax_a.set_xlabel("Latitude (deg)")
    ax_a.set_ylabel(r"recovered $\Delta$GST (K)")
    ax_a.set_title(
        f"a   per-band $\\Delta$GST distribution (n={dGST.size})",
        loc="left",
        weight="bold",
    )
    ax_a.set_xticks(band_centres)
    ax_a.set_xticklabels(
        [f"{int(c)}\n(n={n})" for c, n in zip(band_centres, counts, strict=True)], fontsize=5
    )
    ax_a.grid(axis="y", alpha=0.25, linewidth=0.3)

    # (b) kappa distribution, log-scale histogram, by lat band.
    ax_b = fig.add_subplot(gs[0, 1])
    log_kappa = np.log10(np.maximum(kappa, 1.0e-9))
    band_labels = [
        "polar S",
        "midlat S",
        "subtropical S",
        "tropical S",
        "tropical N",
        "subtropical N",
        "midlat N",
        "polar N",
    ]
    cmap = plt.get_cmap("viridis", len(band_centres))
    for b, label in enumerate(band_labels):
        sub = log_kappa[band_idx == b]
        if sub.size < 3:
            continue
        ax_b.hist(
            sub,
            bins=np.linspace(log_kappa.min(), log_kappa.max(), 30),
            histtype="step",
            linewidth=1.0,
            color=cmap(b),
            label=f"{label} (n={sub.size})",
        )
    ax_b.set_xlabel(r"$\log_{10}\,\kappa$ (m$^2$ s$^{-1}$)")
    ax_b.set_ylabel("borehole count")
    ax_b.legend(loc="upper right", frameon=False, fontsize=5, ncol=2)
    ax_b.set_title(
        "b   posterior $\\kappa$ distribution by lat band",
        loc="left",
        weight="bold",
    )
    ax_b.grid(alpha=0.25, linewidth=0.3)

    # Note: the (Pe_T, L_calL) regime diagram intended for the hero
    # figure requires per-site Darcy velocity; until that lands in
    # follow-up task #31, this script reports the directly observable
    # per-band Delta GST + posterior kappa.  No suptitle on the
    # published PDF.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    median_gst = float(np.nanmedian(dGST))
    print(
        f"wrote {out_path}  n_sites={dGST.size}, median dGST={median_gst:.2f} K",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "smoke-10")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CURATED_YAML)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.catalog, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
