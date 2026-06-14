#!/usr/bin/env python3
"""R18-B: forward-validate the merged T+p+S_i solver against the
Borealis VDTBS thermistor record at Umiujaq.

The strategy is to use the 0-m thermistor record from the deepest
borehole (TA1, 2001-2024) as the surface-temperature forcing, and
forward-predict the temperature evolution at the deeper thermistors.
We then compare predictions at depth against the observed records and
compute the RMS misfit -- both for the merged T+p+S_i solver (with
latent heat) and for ``column_coupled`` (without latent heat), and
report the misfit reduction.

Output
------
``outputs/supersite_umiujaq/forward_runs_merged.nc`` with variables
``T(time, depth_m)``, ``p(time, depth_m)``, ``S_i(time, depth_m)``
and attributes recording the misfit summary.
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


YEAR_S = 365.25 * 86400.0


def _monthly_surface_from_vdtbs(
    obs: xr.Dataset,
    target_borehole: str = "TA1",
    surface_depth_m: float = 0.17,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a monthly mean surface temperature series from VDTBS.

    Returns (months, T_surface_degC) on a calendar-monthly grid.  The
    0 m thermistor is too noisy (snow / surface artefacts), so we
    use the shallowest sub-surface thermistor available -- typically
    0.17 m at TA1.
    """
    bh_idx = list(obs["borehole"].values).index(target_borehole)
    depth_arr = obs["depth_m"].values
    idx = int(np.argmin(np.abs(depth_arr - surface_depth_m)))
    T_at_surf = obs["T_degC"].values[:, idx, bh_idx]
    times = obs["time"].values
    s = pd.Series(T_at_surf, index=pd.to_datetime(times))
    monthly = s.resample("MS").mean().dropna()
    return monthly.index.to_numpy(), monthly.to_numpy(dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--obs",
        default="data/supersite_umiujaq/processed/vdtbs_thermistors.nc",
    )
    parser.add_argument(
        "--out",
        default="outputs/supersite_umiujaq/forward_runs_merged.nc",
    )
    parser.add_argument("--depth_max_m", type=float, default=20.0)
    parser.add_argument("--dz_m", type=float, default=0.5)
    parser.add_argument("--porosity", type=float, default=0.30)
    parser.add_argument("--lambda_r", type=float, default=2.5)
    parser.add_argument("--K_zz", type=float, default=1.0e-12)
    parser.add_argument("--T_f", type=float, default=0.0)
    parser.add_argument("--dTc", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args(argv)

    obs_path = Path(args.obs).expanduser().resolve()
    if not obs_path.exists():
        raise SystemExit(
            f"Observed VDTBS NetCDF not found at {obs_path}; "
            f"run scripts/ingest_borealis_vdtbs.py first."
        )
    obs = xr.open_dataset(obs_path)

    months, T_surf_monthly = _monthly_surface_from_vdtbs(obs)
    if months.size < 24:
        raise SystemExit("Not enough monthly surface samples")

    # Build a uniform monthly grid for the forward solver.
    t_seconds = ((months - months[0]) / np.timedelta64(1, "s")).astype(float)
    # Ensure strictly increasing and uniform monthly spacing.
    dt_s = float(np.median(np.diff(t_seconds)))
    if dt_s <= 0:
        raise SystemExit("monthly spacing zero or negative")
    nt = months.size
    duration_s = (nt - 1) * dt_s

    # Initial conditions: a smoothed mean profile from VDTBS over the
    # first ~18 months of borehole records.
    depth_grid = obs["depth_m"].values
    obs_times = pd.to_datetime(obs["time"].values)
    initial_mask_obs = obs_times <= pd.Timestamp("2002-12-31")
    T_init_from_obs = np.nanmean(obs["T_degC"].values[initial_mask_obs, :, :], axis=(0, 2))
    # Replace any remaining NaNs with the bulk mean.
    bulk_mean = float(np.nanmean(obs["T_degC"].values))
    T_init_from_obs = np.where(
        np.isfinite(T_init_from_obs),
        T_init_from_obs,
        bulk_mean,
    )
    # Map onto the forward depth grid.
    z_fwd = np.arange(0.0, args.depth_max_m + 0.5 * args.dz_m, args.dz_m)
    # Sort obs depths to interpolate cleanly.
    sort_idx = np.argsort(depth_grid)
    T_init_forward = np.interp(
        z_fwd,
        depth_grid[sort_idx],
        T_init_from_obs[sort_idx],
    )

    common = dict(
        depth_max_m=args.depth_max_m,
        dz_m=args.dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        porosity=args.porosity,
        K_zz=args.K_zz,
        gamma_n_alpha_scale=args.gamma,
        sat=T_surf_monthly,
        p_top=0.0,
        T_init=T_init_forward,
        p_init=0.0,
        q_bot=0.05,  # ~50 mW m-2 boreal Shield heat flux
    )

    res_merged = run_column_thermo_freeze_coupled(
        lambda_r=args.lambda_r,
        T_f=args.T_f,
        dTc=args.dTc,
        picard_max_iter=15,
        picard_tol_K=1.0e-3,
        picard_omega=0.7,
        **common,
    )
    res_cc = run_column_coupled(
        lambda_thermal=args.lambda_r,
        rho_c_eff=2.5e6,
        **common,
    )

    fwd_t0 = months[0]
    misfit_merged = rms_misfit_on_common_grid(
        forward_T_K=res_merged.T,
        forward_t_s=t_seconds,
        forward_z_m=res_merged.z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        depth_min_m=0.5,
        depth_max_m=args.depth_max_m,
    )
    misfit_cc = rms_misfit_on_common_grid(
        forward_T_K=res_cc.T,
        forward_t_s=t_seconds,
        forward_z_m=res_cc.z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        depth_min_m=0.5,
        depth_max_m=args.depth_max_m,
    )
    reduction_pct = (
        100.0 * (misfit_cc.rms_total_K - misfit_merged.rms_total_K) / misfit_cc.rms_total_K
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            "T_merged": (("time", "depth_m"), res_merged.T),
            "S_i_merged": (("time", "depth_m"), res_merged.S_i),
            "p_merged": (("time", "depth_m"), res_merged.p),
            "T_cc": (("time", "depth_m"), res_cc.T),
            "picard_iters": (("time_step",), np.concatenate([[0], res_merged.picard_iters])),
            "T_surf_forcing": (("time",), T_surf_monthly),
        },
        coords={
            "time": months,
            "depth_m": res_merged.z,
            "time_step": np.arange(nt),
        },
        attrs={
            "merged_solver_rms_K": float(misfit_merged.rms_total_K),
            "column_coupled_solver_rms_K": float(misfit_cc.rms_total_K),
            "rms_reduction_pct": float(reduction_pct),
            "depth_window_min_m": 0.5,
            "depth_window_max_m": float(args.depth_max_m),
            "n_finite_pairs_merged": int(misfit_merged.n_cells),
            "porosity": float(args.porosity),
            "K_zz": float(args.K_zz),
            "T_f": float(args.T_f),
            "dTc": float(args.dTc),
            "gamma_n_alpha_scale": float(args.gamma),
            "obs_source": str(obs_path),
            "ingest_doi": "10.5683/SP3/QSRW0I",
        },
    )
    ds_out.to_netcdf(out_path)
    print(f"wrote {out_path}")
    print(f"  merged solver RMS:        {misfit_merged.rms_total_K:.3f} K")
    print(f"  column_coupled RMS:       {misfit_cc.rms_total_K:.3f} K")
    print(f"  RMS reduction (merged):   {reduction_pct:.1f} %")
    print(f"  per-depth merged RMS:     {misfit_merged.rms_per_depth_K}")
    print(f"  per-depth merged bias:    {misfit_merged.bias_per_depth_K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
