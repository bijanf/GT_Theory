#!/usr/bin/env python3
"""Figure 3 — Theory-data overlay for the erfc-attenuation envelope (F1).

This is the canonical physics figure: the theoretical pattern is drawn
as a curve, the data is overlaid on the same axes, and the reader can
see immediately whether the conduction kernel of Section 7 captures
the global borehole archive.  Three panels, Nature 180 mm:

  (a) Per-site detrended Delta T(z) as thin grey lines for all 951
      sites + cross-site median (blue, with bootstrap CI band) +
      theoretical erfc envelope (black) at the median recovered
      (kappa, tau, Delta GST).
  (b) Predicted-vs-observed scatter: for every (site, observed depth)
      pair, push the recovered GST history through the analytic
      forward operator and compare to the observed anomaly.  1:1
      line drawn for reference.  Hexbin to handle the ~50k points.
  (c) Cross-site median residual (observed - predicted) vs depth,
      with the +/- 0.2 K theory-acceptance band shaded.  Bootstrap CI
      on the median.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from gt_theory.fingerprints.f1_erfc import load_smoke_pair
from gt_theory.inversion import build_forward_operator

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR_S: float = 365.25 * 86400.0


def _resample_to_grid(z_obs: np.ndarray, dT_obs: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    return np.interp(z_grid, z_obs, dT_obs, left=np.nan, right=np.nan)


def build_figure(subset_dir: Path, out_path: Path, *, envelope_band_K: float = 0.2) -> None:
    profiles, inversions = load_smoke_pair(subset_dir)
    n_sites = len(profiles)
    print(f"loaded {n_sites} site pairs", file=sys.stderr)

    z_grid = np.arange(20.0, 601.0, 20.0)
    obs_grid = np.full((n_sites, z_grid.size), np.nan)
    pred_grid = np.full((n_sites, z_grid.size), np.nan)
    obs_raw_list: list[np.ndarray] = []
    pred_raw_list: list[np.ndarray] = []
    kappa_list: list[float] = []
    gst_recent_list: list[float] = []

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

        obs_grid[i] = _resample_to_grid(z, dT_obs, z_grid)
        pred_grid[i] = _resample_to_grid(z, dT_pred, z_grid)
        # Also keep the per-(site, observed depth) raw values for the
        # scatter in panel (b).
        obs_raw_list.append(dT_obs)
        pred_raw_list.append(dT_pred)
        kappa_list.append(kappa)
        gst_recent_list.append(float(s[0]))

    obs_raw = np.concatenate(obs_raw_list)
    pred_raw = np.concatenate(pred_raw_list)
    median_obs = np.nanmedian(obs_grid, axis=0)
    median_resid = np.nanmedian(obs_grid - pred_grid, axis=0)

    rng = np.random.default_rng(20260522)
    n_boot = 400
    boot_obs = np.empty((n_boot, z_grid.size))
    boot_resid = np.empty((n_boot, z_grid.size))
    for b in range(n_boot):
        idx = rng.integers(0, n_sites, size=n_sites)
        boot_obs[b] = np.nanmedian(obs_grid[idx], axis=0)
        boot_resid[b] = np.nanmedian(obs_grid[idx] - pred_grid[idx], axis=0)
    obs_lo = np.nanpercentile(boot_obs, 5.0, axis=0)
    obs_hi = np.nanpercentile(boot_obs, 95.0, axis=0)
    resid_lo = np.nanpercentile(boot_resid, 5.0, axis=0)
    resid_hi = np.nanpercentile(boot_resid, 95.0, axis=0)

    # Theoretical erfc envelope at the median recovered parameters.
    # Use the median kappa and a characteristic tau ~ 100 yr at which
    # the erfc curve has support across the full 20-600 m window.
    kappa_med = float(np.median(kappa_list))
    gst_med = float(np.median(gst_recent_list))
    tau_yr_choice = 100.0
    erfc_curve = gst_med * erfc(z_grid / (2.0 * np.sqrt(kappa_med * tau_yr_choice * YEAR_S)))

    from matplotlib.colors import LogNorm

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    # Trim to 0-400 m where the diffusive signal lives; below 400 m the
    # erfc envelope is essentially flat at zero and only crowds the
    # plot.
    z_mask = z_grid <= 400.0
    z_plot = z_grid[z_mask]
    obs_plot = obs_grid[:, z_mask]
    med_obs_plot = median_obs[z_mask]
    obs_lo_plot = obs_lo[z_mask]
    obs_hi_plot = obs_hi[z_mask]
    erfc_plot = erfc_curve[z_mask]
    med_resid_plot = median_resid[z_mask]
    resid_lo_plot = resid_lo[z_mask]
    resid_hi_plot = resid_hi[z_mask]

    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.05, 1.0), wspace=0.55)

    # (a) erfc envelope vs data, 0-400 m window.
    ax_a = fig.add_subplot(gs[0, 0])
    sample = rng.choice(n_sites, size=min(200, n_sites), replace=False)
    for i in sample:
        ax_a.plot(obs_plot[i], z_plot, color="#aaaaaa", linewidth=0.2, alpha=0.5)
    ax_a.fill_betweenx(
        z_plot, obs_lo_plot, obs_hi_plot, color="#0066cc", alpha=0.25, label="data 90% CI"
    )
    ax_a.plot(med_obs_plot, z_plot, color="#003366", linewidth=1.4, label="data median")
    ax_a.plot(
        erfc_plot,
        z_plot,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=r"erfc, $\tau$=100 yr",
    )
    ax_a.axvline(0.0, color="black", linewidth=0.3, alpha=0.4)
    ax_a.set_xlabel(r"$\Delta T(z)$ (K)")
    ax_a.set_ylabel("Depth (m)")
    ax_a.set_xlim(-1.5, 2.2)
    ax_a.set_ylim(400, 0)  # depth increases downward
    ax_a.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.4)
    ax_a.set_title(f"a   erfc envelope (n={n_sites})", loc="left", weight="bold")
    ax_a.grid(alpha=0.25, linewidth=0.3)

    # (b) Predicted vs observed hexbin density, log colorbar, clipped at
    # the 99th percentile so a few outlier points don't crush the range.
    ax_b = fig.add_subplot(gs[0, 1])
    mask = np.isfinite(obs_raw) & np.isfinite(pred_raw)
    pred_v = pred_raw[mask]
    obs_v = obs_raw[mask]
    rng_max = float(np.nanpercentile(np.abs(np.concatenate([pred_v, obs_v])), 99.0))
    keep = (np.abs(pred_v) <= rng_max) & (np.abs(obs_v) <= rng_max)
    hb = ax_b.hexbin(
        pred_v[keep],
        obs_v[keep],
        gridsize=60,
        cmap="Blues",
        mincnt=1,
        linewidths=0.0,
        norm=LogNorm(),
        extent=(-rng_max, rng_max, -rng_max, rng_max),
    )
    ax_b.plot(
        [-rng_max, rng_max],
        [-rng_max, rng_max],
        color="black",
        linewidth=0.8,
        linestyle="--",
        label="1:1",
    )
    ax_b.set_xlim(-rng_max, rng_max)
    ax_b.set_ylim(-rng_max, rng_max)
    cb = fig.colorbar(hb, ax=ax_b, shrink=0.7, pad=0.02)
    cb.set_label("count (log)", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    ax_b.set_xlabel(r"predicted $\Delta T$ (K)")
    ax_b.set_ylabel(r"observed $\Delta T$ (K)")
    ax_b.legend(loc="upper left", frameon=False, fontsize=5, handlelength=1.4)
    r = float(np.corrcoef(pred_v[keep], obs_v[keep])[0, 1])
    ax_b.set_title(f"b   pred vs obs (r={r:.3f}, n={int(keep.sum()):,})", loc="left", weight="bold")

    # (c) Residual vs depth + +/- 0.2 K band, 0-400 m window.
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axhspan(
        -envelope_band_K,
        envelope_band_K,
        color="#cc7700",
        alpha=0.18,
        label=rf"$\pm${envelope_band_K} K",
    )
    ax_c.fill_between(
        z_plot, resid_lo_plot, resid_hi_plot, color="#0066cc", alpha=0.3, label="90% CI"
    )
    ax_c.plot(z_plot, med_resid_plot, color="#003366", linewidth=1.2, label="median")
    ax_c.axhline(0.0, color="black", linewidth=0.3, alpha=0.4)
    ax_c.set_xlabel("Depth (m)")
    ax_c.set_ylabel(r"obs - pred (K)")
    ax_c.set_xlim(20, 400)
    ax_c.set_ylim(-envelope_band_K * 1.5, envelope_band_K * 1.5)
    max_abs = float(np.nanmax(np.abs(med_resid_plot)))
    ax_c.legend(loc="upper right", frameon=False, fontsize=5, handlelength=1.4)
    ax_c.set_title(f"c   residual (max |median|={max_abs:.3f} K)", loc="left", weight="bold")
    ax_c.grid(alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path}  n_sites={n_sites}  median_kappa={kappa_med:.2e}  "
        f"median ΔGST={gst_med:.2f} K  pred-obs r={r:.3f}  "
        f"max |median residual| = {max_abs:.3f} K",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envelope-band-K", type=float, default=0.2)
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.out, envelope_band_K=args.envelope_band_K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
