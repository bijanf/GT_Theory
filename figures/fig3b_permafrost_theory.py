#!/usr/bin/env python3
"""Figure 3b — Where does latent-heat physics actually matter? (Revised.)

The previous version of this figure compared the conduction solver and
the enthalpy solver under a contrived +0.7 K step forcing on an
initially -1 deg C column, and found that pure conduction matched the
permafrost-site median better than the enthalpy solver -- an
artefact of the step-forcing scenario, not a verdict on the physics
(per reviewer feedback).

This revision drives BOTH solvers with the actual CRU TS monthly SAT
time series at each of the 188 latent-dominant sites flagged by F6,
then compares their predicted present-day depth profiles against the
observed (detrended) anomaly.  Three curves are overlaid on the
cross-site median:

  1.  Pure conduction (column_1d) under the realistic monthly forcing.
  2.  Enthalpy solver (column_enthalpy) with apparent-heat-capacity
      freeze-thaw on the same forcing.  At permafrost sites the
      surface trajectory repeatedly crosses the freezing interval
      [T_f - dTc, T_f], so the latent-heat spike actually contributes.
  3.  Observed (detrended) median across the 188 sites.

Diagnostic: maximum |median residual| against each solver on the
20--400 m window.  The enthalpy solver should track the data better
than pure conduction at these sites, validating the F6 latent-heat
fingerprint at the depth-profile level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.fingerprints.f1_erfc import load_smoke_pair
from gt_theory.solvers import run_column_1d, run_column_enthalpy

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR_S: float = 365.25 * 86400.0


def _resample_to_grid(z_obs: np.ndarray, dT_obs: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    return np.interp(z_grid, z_obs, dT_obs, left=np.nan, right=np.nan)


def _run_pair(
    sat_monthly_c: np.ndarray,
    *,
    duration_s: float,
    dt_s: float,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run both solvers on the same zero-mean monthly forcing.  Returns
    (cond_dT_z, enth_dT_z) sampled onto z_grid."""
    mean_sat = float(np.nanmean(sat_monthly_c))
    sat_anom = sat_monthly_c - mean_sat
    depth_max = max(z_grid.max() + 20.0, 60.0)
    dz_m = 4.0

    cond = run_column_1d(
        depth_max_m=depth_max,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=1.0e-6,
        sat=sat_anom,
        n_winter=1.0,
        T_init=0.0,
    )
    enth = run_column_enthalpy(
        depth_max_m=depth_max,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        lambda_thermal=2.5,
        rho_c_solid=2.5e6,
        porosity=0.30,
        dTc=1.0,
        # Anchor the enthalpy run on the actual mean so its freezing
        # interval is exercised at the realistic temperature -- a
        # mean of -3 deg C means most of the column sits sub-freezing
        # and the seasonal cycle crosses through 0 deg C.
        sat=sat_anom + mean_sat,
        T_init=mean_sat,
    )

    cond_z = _resample_to_grid(cond.z, cond.T[-1], z_grid)
    enth_anom = enth.T[-1] - mean_sat
    enth_z = _resample_to_grid(enth.z, enth_anom, z_grid)
    return cond_z, enth_z


def build_figure(
    subset_dir: Path,
    occupancy_path: Path,
    cru_parquet: Path,
    out_path: Path,
    *,
    max_sites: int | None = None,
) -> None:
    profiles, inversions = load_smoke_pair(subset_dir)
    site_ids_all = [str(inv["site_id"].iloc[0]) for inv in inversions]

    occ = pd.read_parquet(occupancy_path)
    latent_set = set(occ.loc[occ["latent_dominant"], "site_id"].astype(str))
    cru = pd.read_parquet(cru_parquet)
    sat_panel = {
        sid: g.sort_values("time")["sat_c"].to_numpy(dtype=float)
        for sid, g in cru.groupby("site_id")
    }

    site_indices = [
        i for i, sid in enumerate(site_ids_all) if sid in latent_set and sid in sat_panel
    ]
    if max_sites is not None:
        site_indices = site_indices[:max_sites]
    n_sites = len(site_indices)
    if n_sites < 5:
        raise RuntimeError(f"too few latent-dominant sites with CRU data ({n_sites})")
    print(f"loaded {n_sites} latent-dominant sites with CRU forcing", file=sys.stderr)

    # Inspect one SAT series to learn the timestep.
    sample_sat = sat_panel[site_ids_all[site_indices[0]]]
    nt = sample_sat.size
    dt_s = YEAR_S / 12.0
    duration_s = (nt - 1) * dt_s

    z_grid = np.arange(20.0, 401.0, 10.0)
    obs_grid = np.full((n_sites, z_grid.size), np.nan)
    cond_grid = np.full((n_sites, z_grid.size), np.nan)
    enth_grid = np.full((n_sites, z_grid.size), np.nan)

    for k, i in enumerate(site_indices):
        prof = profiles[i]
        inv = inversions[i]
        sid = site_ids_all[i]

        z_obs = prof["depth_m"].to_numpy(dtype=float)
        T_obs = prof["temperature_c"].to_numpy(dtype=float)
        T0 = float(inv["T0_K"].iloc[0])
        dTdz = float(inv["dTdz_K_per_m"].iloc[0])
        obs_grid[k] = _resample_to_grid(z_obs, T_obs - (T0 + dTdz * z_obs), z_grid)

        sat = sat_panel[sid]
        cond_z, enth_z = _run_pair(sat, duration_s=duration_s, dt_s=dt_s, z_grid=z_grid)
        cond_grid[k] = cond_z
        enth_grid[k] = enth_z

        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{n_sites} sites done", file=sys.stderr)

    median_obs = np.nanmedian(obs_grid, axis=0)
    median_cond = np.nanmedian(cond_grid, axis=0)
    median_enth = np.nanmedian(enth_grid, axis=0)
    rng = np.random.default_rng(20260522)
    n_boot = 400
    boot_obs = np.empty((n_boot, z_grid.size))
    for b in range(n_boot):
        idx = rng.integers(0, n_sites, size=n_sites)
        boot_obs[b] = np.nanmedian(obs_grid[idx], axis=0)
    lo = np.nanpercentile(boot_obs, 5.0, axis=0)
    hi = np.nanpercentile(boot_obs, 95.0, axis=0)

    resid_cond = median_obs - median_cond
    resid_enth = median_obs - median_enth
    max_cond = float(np.nanmax(np.abs(resid_cond)))
    max_enth = float(np.nanmax(np.abs(resid_enth)))

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.45))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.3, 1.0), wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.fill_betweenx(z_grid, lo, hi, color="#0066cc", alpha=0.25, label="data 90% CI")
    ax_a.plot(
        median_obs, z_grid, color="#003366", linewidth=1.4, label=f"data median (n={n_sites})"
    )
    ax_a.plot(
        median_cond,
        z_grid,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="conduction (CRU forcing)",
    )
    ax_a.plot(median_enth, z_grid, color="#aa3333", linewidth=1.2, label="enthalpy (CRU forcing)")
    ax_a.axvline(0.0, color="black", linewidth=0.3, alpha=0.4)
    ax_a.set_xlabel(r"$\Delta T(z)$ (K)")
    ax_a.set_ylabel("Depth (m)")
    ax_a.set_ylim(400, 20)
    ax_a.set_xlim(-1.5, 1.5)
    ax_a.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.4)
    ax_a.set_title(
        "a   permafrost sites, realistic monthly CRU forcing",
        loc="left",
        weight="bold",
    )
    ax_a.grid(alpha=0.25, linewidth=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axhspan(-0.2, 0.2, color="#cc7700", alpha=0.15, label=r"$\pm$0.2 K")
    ax_b.axhline(0.0, color="black", linewidth=0.3, alpha=0.4)
    ax_b.plot(
        z_grid,
        resid_cond,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=f"conduction (max |.|={max_cond:.3f} K)",
    )
    ax_b.plot(
        z_grid,
        resid_enth,
        color="#aa3333",
        linewidth=1.2,
        label=f"enthalpy   (max |.|={max_enth:.3f} K)",
    )
    ax_b.set_xlabel("Depth (m)")
    ax_b.set_ylabel(r"data median - theory (K)")
    ax_b.set_xlim(20, 400)
    ax_b.legend(loc="upper right", frameon=False, fontsize=5, handlelength=1.4)
    winner = "enthalpy" if max_enth < max_cond else "conduction"
    ax_b.set_title(
        f"b   residual: {winner} wins on these sites",
        loc="left",
        weight="bold",
    )
    ax_b.grid(alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path}  n_permafrost={n_sites}  "
        f"max |resid| cond={max_cond:.3f} K vs enth={max_enth:.3f} K",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "full")
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=REPO_ROOT / "outputs" / "full" / "f6_occupancy.parquet",
    )
    parser.add_argument(
        "--cru-parquet",
        type=Path,
        default=REPO_ROOT / "outputs" / "full" / "cru_sat.parquet",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-sites", type=int, default=None, help="Limit to N sites (for fast development)."
    )
    args = parser.parse_args(argv)
    build_figure(
        args.subset_dir, args.occupancy, args.cru_parquet, args.out, max_sites=args.max_sites
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
