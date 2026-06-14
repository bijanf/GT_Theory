#!/usr/bin/env python3
"""Figure 4 — F2 (SAT-GST coupling) and F3 (latitudinal amplification).

Three panels, Nature 180 mm:
  (a) Latitude-band-averaged Delta SAT and Delta GST with bootstrap
      90% error bars.  Shows the structural relationship instead of
      drowning it in 951 scatter dots.
  (b) Per-band coupling ratio eta_band = mean(Delta GST) / mean(Delta SAT)
      with bootstrap CI.
  (c) Boreal/equator amplification ratios for SAT and GST with
      bootstrap 5-95 error bars and the theoretical reference values
      (~0.94 GST vs ~1.49 SAT from the accompanying paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gt_theory.fingerprints import compute_f2, compute_f3
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


def _bootstrap_band_means(
    values: np.ndarray,
    band_idx: np.ndarray,
    n_bands: int,
    n_bootstrap: int = 1000,
    seed: int = 20260522,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-band mean + bootstrap 5/95 CI.  Returns (mean, ci_lo, ci_hi)
    arrays of length n_bands; NaN where the band has fewer than 2 sites."""
    rng = np.random.default_rng(seed)
    mean = np.full(n_bands, np.nan)
    lo = np.full(n_bands, np.nan)
    hi = np.full(n_bands, np.nan)
    for b in range(n_bands):
        sub = values[(band_idx == b) & np.isfinite(values)]
        if sub.size < 2:
            continue
        mean[b] = float(np.mean(sub))
        boot = np.array(
            [np.mean(rng.choice(sub, size=sub.size, replace=True)) for _ in range(n_bootstrap)]
        )
        lo[b] = float(np.percentile(boot, 5.0))
        hi[b] = float(np.percentile(boot, 95.0))
    return mean, lo, hi


def build_figure(
    subset_dir: Path,
    catalog_path: Path,
    cru_sat_path: Path,
    out_path: Path,
    *,
    target_sat_ratio: float = 1.49,
    target_gst_ratio: float = 0.94,
) -> None:
    _, inversions = load_smoke_pair(subset_dir)
    site_ids = [str(inv["site_id"].iloc[0]) for inv in inversions]
    lat_lookup = _site_lats(catalog_path, site_ids)
    keep = [i for i, sid in enumerate(site_ids) if sid in lat_lookup]
    inversions = [inversions[i] for i in keep]
    site_ids = [site_ids[i] for i in keep]
    lats = np.array([lat_lookup[s] for s in site_ids])

    cru = pd.read_parquet(cru_sat_path)
    f2 = compute_f2(inversions=inversions, site_ids=site_ids, cru_sat=cru, n_bootstrap=400)
    f3 = compute_f3(
        inversions=inversions,
        site_ids=site_ids,
        lats_deg=lats,
        cru_sat=cru,
        n_bootstrap=400,
    )

    # 8 latitude bands centred on -65, -45, -25, -5, +15, +35, +55, +75.
    band_edges = np.array([-90.0, -55.0, -35.0, -15.0, 5.0, 25.0, 45.0, 65.0, 90.0])
    band_centres = 0.5 * (band_edges[:-1] + band_edges[1:])
    band_idx = np.clip(np.digitize(lats, band_edges) - 1, 0, len(band_centres) - 1)

    sat_mean, sat_lo, sat_hi = _bootstrap_band_means(f2.delta_sat_K, band_idx, len(band_centres))
    gst_mean, gst_lo, gst_hi = _bootstrap_band_means(f2.delta_gst_K, band_idx, len(band_centres))
    eta_mean, eta_lo, eta_hi = _bootstrap_band_means(f2.eta_per_site, band_idx, len(band_centres))

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.4, 1.0, 1.0), wspace=0.4)

    # (a) lat-band-averaged Delta SAT and Delta GST.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(
        band_centres,
        sat_mean,
        yerr=[sat_mean - sat_lo, sat_hi - sat_mean],
        fmt="o-",
        color="#cc7700",
        capsize=3,
        linewidth=1.0,
        markersize=4,
        label=r"$\Delta$SAT (CRU)",
    )
    ax_a.errorbar(
        band_centres,
        gst_mean,
        yerr=[gst_mean - gst_lo, gst_hi - gst_mean],
        fmt="s-",
        color="#0066cc",
        capsize=3,
        linewidth=1.0,
        markersize=4,
        label=r"$\Delta$GST (recovered)",
    )
    ax_a.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax_a.set_xlabel("Latitude (deg)")
    ax_a.set_ylabel("Warming 2000-2024 minus 1901-1960 (K)")
    ax_a.legend(loc="upper left", frameon=False, fontsize=6)
    ax_a.set_title(f"a   lat-band warming (n_sites={lats.size})", loc="left", weight="bold")
    ax_a.grid(alpha=0.25, linewidth=0.3)

    # (b) per-band coupling ratio.
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.errorbar(
        band_centres,
        eta_mean,
        yerr=[eta_mean - eta_lo, eta_hi - eta_mean],
        fmt="o-",
        color="#0066cc",
        capsize=3,
        linewidth=1.0,
        markersize=4,
    )
    ax_b.axhline(1.0, color="black", linewidth=0.4, alpha=0.4, label="full coupling")
    ax_b.set_xlabel("Latitude (deg)")
    ax_b.set_ylabel(r"$\bar{\eta} = \Delta$GST / $\Delta$SAT")
    ax_b.legend(loc="upper right", frameon=False, fontsize=6)
    ax_b.set_title("b   per-band coupling ratio", loc="left", weight="bold")
    ax_b.grid(alpha=0.25, linewidth=0.3)

    # (c) boreal/equator amplification ratios + theory reference.
    ax_c = fig.add_subplot(gs[0, 2])
    centres = np.array([0, 1])
    vals = np.array([f3.sat_ratio, f3.gst_ratio])
    err_lo = np.array([f3.sat_ratio - f3.sat_ratio_ci[0], f3.gst_ratio - f3.gst_ratio_ci[0]])
    err_hi = np.array([f3.sat_ratio_ci[1] - f3.sat_ratio, f3.gst_ratio_ci[1] - f3.gst_ratio])
    ax_c.bar(
        centres,
        vals,
        yerr=[err_lo, err_hi],
        capsize=4,
        color=["#cc7700", "#0066cc"],
        edgecolor="white",
        alpha=0.8,
    )
    ax_c.scatter(
        [0, 1],
        [target_sat_ratio, target_gst_ratio],
        s=24,
        marker="D",
        c="black",
        zorder=5,
        label="theory ref.",
    )
    ax_c.set_xticks(centres)
    ax_c.set_xticklabels([f"SAT\n(n_b={f3.n_boreal}, n_e={f3.n_equator})", "GST"])
    ax_c.set_ylabel("boreal / equator ratio")
    ax_c.legend(loc="upper left", frameon=False, fontsize=5.5)
    ax_c.set_title("c   amplification ratio", loc="left", weight="bold")
    ax_c.grid(axis="y", alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path}  (lat-bands={len(band_centres)}, n_sites={lats.size}; "
        f"SAT ratio={f3.sat_ratio:.2f}, GST ratio={f3.gst_ratio:.2f})",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "smoke-10")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CURATED_YAML)
    parser.add_argument(
        "--cru-sat-parquet",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke-10" / "cru_sat.parquet",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.catalog, args.cru_sat_parquet, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
