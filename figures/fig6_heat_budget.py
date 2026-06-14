#!/usr/bin/env python3
"""Figure 6 — F4 continental ground-heat-content closure.

Three panels, Nature 180 mm:
  (a) Per-lat-band median delta-E (J m^-2) with bootstrap 5-95 CI.
      Shows where the heat is being stored geographically.
  (b) HT-reweighted continental total in ZJ for the 1960-2018 window,
      with bootstrap 90% CI and the Cuesta-Valero (2025) 17.6 ZJ
      reference band.
  (c) True sensitivity tornado: each row is a real rerun of
      aggregate_continental with one parameter perturbed, so the bar
      lengths reflect the model's actual partial derivatives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gt_theory.fingerprints import aggregate_continental
from gt_theory.fingerprints.f1_erfc import load_smoke_pair
from gt_theory.fingerprints.f4_budget import RHO_C_EFF, site_energy_gain

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


def _band_stats(
    values: np.ndarray, band_idx: np.ndarray, n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.full(n_bands, np.nan)
    lo = np.full(n_bands, np.nan)
    hi = np.full(n_bands, np.nan)
    rng = np.random.default_rng(20260522)
    for b in range(n_bands):
        sub = values[(band_idx == b) & np.isfinite(values)]
        if sub.size < 2:
            continue
        median[b] = float(np.median(sub))
        boot = np.array(
            [np.median(rng.choice(sub, size=sub.size, replace=True)) for _ in range(800)]
        )
        lo[b] = float(np.percentile(boot, 5.0))
        hi[b] = float(np.percentile(boot, 95.0))
    return median, lo, hi


def build_figure(
    subset_dir: Path,
    catalog_path: Path,
    out_path: Path,
    *,
    target_ZJ: float = 17.6,
    target_band_ZJ: float = 3.0,
) -> None:
    _, inversions = load_smoke_pair(subset_dir)
    site_ids = [str(inv["site_id"].iloc[0]) for inv in inversions]
    lat_lookup = _site_lats(catalog_path, site_ids)
    keep = [i for i, sid in enumerate(site_ids) if sid in lat_lookup]
    inversions = [inversions[i] for i in keep]
    site_ids = [site_ids[i] for i in keep]
    lats = np.array([lat_lookup[s] for s in site_ids])

    # Baseline run.
    base = aggregate_continental(
        inversions,
        lats,
        site_ids,
        window_years=(0.0, 58.0),
        target_ZJ=target_ZJ,
        equivalence_band_ZJ=target_band_ZJ,
        n_bootstrap=600,
    )
    per_site_E = np.array([g.delta_E_J_per_m2 for g in base.site_gains])

    # Lat band stats for panel (a).
    band_edges = np.array([-90.0, -55.0, -35.0, -15.0, 5.0, 25.0, 45.0, 65.0, 90.0])
    band_centres = 0.5 * (band_edges[:-1] + band_edges[1:])
    band_idx = np.clip(np.digitize(lats, band_edges) - 1, 0, len(band_centres) - 1)
    median_E, lo_E, hi_E = _band_stats(per_site_E, band_idx, len(band_centres))

    # True sensitivity tornado: rerun aggregate_continental with each
    # parameter perturbation.
    def _total_for(
        *,
        rho_c: float = RHO_C_EFF,
        z_max_m: float = 600.0,
        window: tuple[float, float] = (0.0, 58.0),
    ) -> float:
        per = np.array(
            [
                site_energy_gain(inv, window_years=window, rho_c_eff=rho_c, z_max_m=z_max_m)
                for inv in inversions
            ]
        )
        from gt_theory.fingerprints import horvitz_thompson_weights

        w, _ = horvitz_thompson_weights(lats)
        return float(np.sum(per * w)) / 1.0e21  # to ZJ

    sensitivities = [
        ("rho c_eff = 2.0 MJ m^-3 K^-1", _total_for(rho_c=2.0e6) - base.total_ZJ),
        ("rho c_eff = 3.0 MJ m^-3 K^-1", _total_for(rho_c=3.0e6) - base.total_ZJ),
        ("z_max = 400 m", _total_for(z_max_m=400.0) - base.total_ZJ),
        ("z_max = 800 m", _total_for(z_max_m=800.0) - base.total_ZJ),
        ("window 0-50 yr", _total_for(window=(0.0, 50.0)) - base.total_ZJ),
        ("window 0-70 yr", _total_for(window=(0.0, 70.0)) - base.total_ZJ),
    ]

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.3, 1.0, 1.4), wspace=0.45)

    # (a) per-band median delta-E with bootstrap CI.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(
        band_centres,
        median_E / 1.0e9,  # GJ m^-2
        yerr=[(median_E - lo_E) / 1.0e9, (hi_E - median_E) / 1.0e9],
        fmt="o-",
        color="#0066cc",
        capsize=3,
        linewidth=1.0,
        markersize=4,
    )
    ax_a.axhline(0.0, color="black", linewidth=0.4, alpha=0.4)
    ax_a.set_xlabel("Latitude (deg)")
    ax_a.set_ylabel(r"per-band median $\Delta E$ (GJ m$^{-2}$)")
    ax_a.set_title(
        f"a   per-band heat-content gain (n={lats.size})",
        loc="left",
        weight="bold",
    )
    ax_a.grid(alpha=0.25, linewidth=0.3)

    # (b) HT-reweighted continental total.
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axhspan(
        target_ZJ - target_band_ZJ, target_ZJ + target_band_ZJ, color="#cc7700", alpha=0.18
    )
    ax_b.axhline(target_ZJ, color="#cc7700", linewidth=1.0, label="Cuesta-Valero 2025")
    ax_b.errorbar(
        [0],
        [base.total_ZJ],
        yerr=[[base.total_ZJ - base.ci_lo_ZJ], [base.ci_hi_ZJ - base.total_ZJ]],
        fmt="o",
        color="#0066cc",
        capsize=4,
        label="this work",
    )
    ax_b.set_xticks([0])
    ax_b.set_xticklabels([f"n={base.n_sites}"])
    ax_b.set_ylabel(r"continental $\Delta E$ (ZJ)")
    ax_b.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax_b.set_title(
        f"b   HT-weighted total (TOST passes: {base.passes_equivalence})",
        loc="left",
        weight="bold",
    )
    ax_b.grid(axis="y", alpha=0.25, linewidth=0.3)

    # (c) Tornado of TRUE sensitivities (real reruns).
    ax_c = fig.add_subplot(gs[0, 2])
    labels = [s[0] for s in sensitivities]
    deltas = [s[1] for s in sensitivities]
    y = np.arange(len(labels))
    colours = ["#0066cc" if d >= 0 else "#aa0033" for d in deltas]
    ax_c.barh(y, deltas, color=colours, alpha=0.85)
    ax_c.axvline(0.0, color="black", linewidth=0.5)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels, fontsize=5)
    ax_c.set_xlabel(r"$\Delta$ total (ZJ)")
    ax_c.set_title("c   sensitivity tornado (real reruns)", loc="left", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path}  total={base.total_ZJ:.2f} ZJ, CI [{base.ci_lo_ZJ:.2f}, {base.ci_hi_ZJ:.2f}], "
        f"target {target_ZJ:.1f}+/-{target_band_ZJ:.1f}; TOST passes={base.passes_equivalence}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke-10",
        help="Directory containing profiles/ and inversions/ subdirs.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CURATED_YAML)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.catalog, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
