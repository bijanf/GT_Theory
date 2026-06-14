#!/usr/bin/env python3
"""R18-C: re-invert the Umiujaq supersite ``s = gamma_n_alpha_scale``
coupling parameter with the merged T+p+S_i solver.

We use a profile-likelihood approach instead of a full Bayesian MCMC:
the merged solver is ~5x slower per call than ``run_column_coupled``,
and a full adaptive-MH ensemble would take hours.  Instead we sweep
``s`` on a 1-D grid, holding the other reduced-vector parameters
(K_zz, porosity, lambda_r) fixed at the values implied by the
forward-validation in R18-B, and report:

* the MAP estimate s_MAP that minimises the T-only misfit to the
  Borealis VDTBS thermistor record;
* a 90% CI from the curvature of the profile log-likelihood at the
  MAP (Wilks 1938).

The published headline from the synthetic-twin test (R8 / P2b) was
``s = 0.97 (90% CI [0.58, 1.53])`` -- a recovery test against a
known truth of s=1.0 with the *pure-sensible* column_coupled solver.

The merged-solver headline is the load-bearing scientific update:
it is the first time the coupling parameter has been inferred at
Umiujaq using a solver that carries the latent-heat physics that
dominates the local ℒ regime.

Output
------
``outputs/supersite_umiujaq/posterior_merged.parquet`` -- one row per
``s`` value with: s, total_rms_K (merged solver), total_rms_K_cc
(column_coupled at the same s), n_finite_pairs, log_likelihood.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gt_theory.diagnostics.forward_misfit import rms_misfit_on_common_grid
from gt_theory.solvers.column_coupled import run_column_coupled
from gt_theory.solvers.column_thermo_freeze_coupled import (
    run_column_thermo_freeze_coupled,
)


def _build_forcing(
    obs: xr.Dataset,
    surface_depth_m: float = 0.17,
    target_borehole: str = "TA1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bh_idx = list(obs["borehole"].values).index(target_borehole)
    depth_arr = obs["depth_m"].values
    idx_surf = int(np.argmin(np.abs(depth_arr - surface_depth_m)))
    T_at_surf = obs["T_degC"].values[:, idx_surf, bh_idx]
    times = obs["time"].values
    s = pd.Series(T_at_surf, index=pd.to_datetime(times))
    monthly = s.resample("MS").mean().dropna()
    months = monthly.index.to_numpy()
    T_surf = monthly.to_numpy(dtype=float)
    obs_times = pd.to_datetime(obs["time"].values)
    initial_mask = obs_times <= pd.Timestamp("2002-12-31")
    T_init_obs = np.nanmean(
        obs["T_degC"].values[initial_mask, :, :],
        axis=(0, 2),
    )
    return months, T_surf, T_init_obs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--obs",
        default="data/supersite_umiujaq/processed/vdtbs_thermistors.nc",
    )
    parser.add_argument(
        "--out",
        default="outputs/supersite_umiujaq/posterior_merged.parquet",
    )
    parser.add_argument(
        "--s_grid",
        default="0.0,0.1,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0",
    )
    parser.add_argument("--depth_max_m", type=float, default=20.0)
    parser.add_argument("--dz_m", type=float, default=0.5)
    parser.add_argument("--porosity", type=float, default=0.30)
    parser.add_argument("--lambda_r", type=float, default=2.5)
    parser.add_argument("--K_zz", type=float, default=1.0e-12)
    parser.add_argument("--T_f", type=float, default=0.0)
    parser.add_argument("--dTc", type=float, default=1.0)
    parser.add_argument("--sigma_T", type=float, default=1.0)
    args = parser.parse_args(argv)

    obs = xr.open_dataset(Path(args.obs).expanduser().resolve())
    months, T_surf, T_init_obs = _build_forcing(obs)
    nt = months.size
    t_seconds = ((months - months[0]) / np.timedelta64(1, "s")).astype(float)
    dt_s = float(np.median(np.diff(t_seconds)))
    duration_s = (nt - 1) * dt_s

    z_fwd = np.arange(0.0, args.depth_max_m + 0.5 * args.dz_m, args.dz_m)
    bulk = float(np.nanmean(obs["T_degC"].values))
    T_init_obs_filled = np.where(np.isfinite(T_init_obs), T_init_obs, bulk)
    sort_idx = np.argsort(obs["depth_m"].values)
    T_init_forward = np.interp(
        z_fwd,
        obs["depth_m"].values[sort_idx],
        T_init_obs_filled[sort_idx],
    )

    s_grid = np.array([float(s) for s in args.s_grid.split(",")])

    rows = []
    for s in s_grid:
        print(f"profile s = {s:.2f}")
        common = dict(
            depth_max_m=args.depth_max_m,
            dz_m=args.dz_m,
            duration_s=duration_s,
            dt_s=dt_s,
            porosity=args.porosity,
            K_zz=args.K_zz,
            gamma_n_alpha_scale=float(s),
            sat=T_surf,
            p_top=0.0,
            T_init=T_init_forward,
            p_init=0.0,
            q_bot=0.05,
        )
        res_m = run_column_thermo_freeze_coupled(
            lambda_r=args.lambda_r,
            T_f=args.T_f,
            dTc=args.dTc,
            picard_max_iter=15,
            picard_tol_K=1.0e-3,
            picard_omega=0.7,
            **common,
        )
        res_c = run_column_coupled(
            lambda_thermal=args.lambda_r,
            rho_c_eff=2.5e6,
            **common,
        )
        mf_m = rms_misfit_on_common_grid(
            forward_T_K=res_m.T,
            forward_t_s=t_seconds,
            forward_z_m=res_m.z,
            forward_t0=months[0],
            observed_ds=obs,
            depth_min_m=0.5,
            depth_max_m=args.depth_max_m,
        )
        mf_c = rms_misfit_on_common_grid(
            forward_T_K=res_c.T,
            forward_t_s=t_seconds,
            forward_z_m=res_c.z,
            forward_t0=months[0],
            observed_ds=obs,
            depth_min_m=0.5,
            depth_max_m=args.depth_max_m,
        )
        # Gaussian log-likelihood at sigma_T noise.
        ll_m = -0.5 * (mf_m.rms_total_K / args.sigma_T) ** 2 * mf_m.n_cells
        ll_c = -0.5 * (mf_c.rms_total_K / args.sigma_T) ** 2 * mf_c.n_cells
        rows.append(
            dict(
                s=float(s),
                rms_K_merged=float(mf_m.rms_total_K),
                rms_K_cc=float(mf_c.rms_total_K),
                n_finite_pairs=int(mf_m.n_cells),
                log_lik_merged=float(ll_m),
                log_lik_cc=float(ll_c),
            )
        )
        print(f"  merged RMS={mf_m.rms_total_K:.3f}K, cc RMS={mf_c.rms_total_K:.3f}K")

    df = pd.DataFrame(rows)
    # Drop rows where Picard failed to converge (RMS exceeds physical scale).
    df = df[df["rms_K_merged"] < 100.0].reset_index(drop=True).copy()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    # Headline statistics: MAP s for each solver and a quick parabolic-fit
    # 90% CI from the local curvature of the log-likelihood profile.
    def _map_and_ci(s_arr, ll_arr):
        i_max = int(np.argmax(ll_arr))
        s_map = float(s_arr[i_max])
        # Parabolic fit to ll(s); CI at ll_max - 1.353 (chi^2_1 90%/2).
        if 0 < i_max < len(s_arr) - 1:
            xs = s_arr[i_max - 1 : i_max + 2]
            ys = ll_arr[i_max - 1 : i_max + 2]
            A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
            a, b, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            if a < 0:
                # Fisher info I = -2a (negative of second derivative).
                sigma_s = float(np.sqrt(-1.0 / (2.0 * a)))
                ci_lo = s_map - 1.645 * sigma_s
                ci_hi = s_map + 1.645 * sigma_s
                return s_map, ci_lo, ci_hi
        return s_map, float("nan"), float("nan")

    s_map_m, lo_m, hi_m = _map_and_ci(df["s"].values, df["log_lik_merged"].values)
    s_map_c, lo_c, hi_c = _map_and_ci(df["s"].values, df["log_lik_cc"].values)

    print(f"\nwrote {out_path}")
    print("\nProfile-likelihood headline:")
    print(f"  merged   s_MAP = {s_map_m:.3f}  90% CI [{lo_m:.3f}, {hi_m:.3f}]")
    print(f"  cc       s_MAP = {s_map_c:.3f}  90% CI [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"\nFor comparison, the published synthetic-twin headline was")
    print(f"  s = 0.97 (90% CI [0.58, 1.53])  -- pure-sensible solver, recovery test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
