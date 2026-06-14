#!/usr/bin/env python3
"""InSAR-vs-solver cross-check at the three supersites.

Smoke-mode driver: if no real EGMS / ASF DAAC CSV is on disk under
``data/insar/raw/``, synthesise a noisy reference series from the
coupled solver itself, run the cross-check, and report the residual
reduction R = 1 - RMS_on / RMS_off.

For real-data mode: drop the operator-downloaded CSVs into
``data/insar/raw/<site>.csv`` and re-run; the script will pick them
up automatically.

Output: stderr summary + ``outputs/global/insar_check.parquet``
(one row per supersite).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from gt_theory.analysis import (
    compare_insar_solver,
    predict_surface_displacement,
)
from gt_theory.io.egms import load_egms_csv, nearest_point
from gt_theory.solvers import run_column_coupled


YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


SUPERSITES = [
    {
        "name": "umiujaq",
        "lat": 56.55,
        "lon": -76.55,
        "raw_csv": "data/insar/raw/umiujaq.csv",
        "porosity": 0.40,
        "K_zz_m_per_s": 5.0e-6,
        "lambda_th": 1.80,
        "depth_max_m": 10.0,
        "dz_m": 0.5,
        "duration_yr": 2.0,
        "dt_d": 5.0,
        "sat_C_seasonal_amp": 10.0,
        "sat_C_mean": -3.0,
    },
    {
        "name": "mont_terri",
        "lat": 47.235,
        "lon": 7.155,
        "raw_csv": "data/insar/raw/mont_terri.csv",
        "porosity": 0.15,
        "K_zz_m_per_s": 2.0e-13,
        "lambda_th": 2.10,
        "depth_max_m": 20.0,
        "dz_m": 0.5,
        "duration_yr": 1.3,
        "dt_d": 1.0,
        "sat_C_seasonal_amp": 1.0,
        "sat_C_mean": 15.0,
    },
    {
        "name": "utah_forge",
        "lat": 38.504,
        "lon": -112.896,
        "raw_csv": "data/insar/raw/utah_forge.csv",
        "porosity": 0.02,
        "K_zz_m_per_s": 5.0e-7,
        "lambda_th": 3.10,
        "depth_max_m": 3300.0,
        "dz_m": 25.0,
        "duration_yr": 0.12,
        "dt_d": 0.1,
        "sat_C_seasonal_amp": 4.0,
        "sat_C_mean": 15.0,
    },
]


def _solver_pair(site: dict, *, gamma: float):
    nt = int(round(site["duration_yr"] * YEAR_S / (site["dt_d"] * DAY_S))) + 1
    t = np.arange(nt) * site["dt_d"] * DAY_S
    sat = site["sat_C_mean"] + site["sat_C_seasonal_amp"] * np.sin(2.0 * np.pi * t / YEAR_S)
    rho_w = 1000.0
    mu = 1.3e-3
    g = 9.81
    K_zz = site["K_zz_m_per_s"]
    k_intrinsic = K_zz * mu / (rho_w * g)
    return run_column_coupled(
        depth_max_m=site["depth_max_m"],
        dz_m=site["dz_m"],
        duration_s=site["duration_yr"] * YEAR_S,
        dt_s=site["dt_d"] * DAY_S,
        lambda_thermal=site["lambda_th"],
        rho_c_eff=2.5e6,
        K_zz=k_intrinsic,
        mu=mu,
        porosity=site["porosity"],
        alpha_w=2.1e-4,
        beta_w=4.5e-10,
        rho_w=rho_w,
        g=g,
        gamma_n_alpha_scale=gamma,
        sat=sat,
        p_top=0.0,
    )


def _synthetic_insar(
    u_truth_mm: np.ndarray,
    solver_t_s: np.ndarray,
    *,
    reference_time: pd.Timestamp,
    noise_mm: float = 1.0,
    n_obs: int = 30,
    rng: np.random.Generator,
) -> xr.Dataset:
    """Sample the s=1 solver displacement at a sparse cadence + noise."""
    obs_idx = np.linspace(0, solver_t_s.size - 1, n_obs).astype(int)
    obs_t = solver_t_s[obs_idx]
    u_obs = u_truth_mm[obs_idx] + rng.normal(0.0, noise_mm, size=n_obs)
    obs_time = reference_time + pd.to_timedelta(obs_t, unit="s")
    return xr.Dataset(
        data_vars={
            "lat": ((), 0.0),
            "lon": ((), 0.0),
            "velocity_mm_yr": (
                (),
                float((u_truth_mm[-1] - u_truth_mm[0]) / (solver_t_s[-1] - solver_t_s[0]) * YEAR_S),
            ),
            "velocity_std_mm_yr": ((), float(noise_mm)),
            "displacement_mm": (("time",), u_obs),
        },
        coords={"point": "synthetic", "time": obs_time.values},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/global/insar_check.parquet",
    )
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args(argv)

    rng_master = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for site in SUPERSITES:
        res_on = _solver_pair(site, gamma=1.0)
        res_off = _solver_pair(site, gamma=0.0)
        u_on = predict_surface_displacement(res_on, porosity=site["porosity"])
        u_off = predict_surface_displacement(res_off, porosity=site["porosity"])
        ref_time = pd.Timestamp("2020-01-01")

        # Load real EGMS CSV if present, else synthesise from s=1 + noise.
        raw_csv = Path(site["raw_csv"]).expanduser().resolve()
        if raw_csv.exists():
            full_ds = load_egms_csv(raw_csv)
            point_ds = nearest_point(full_ds, lat_deg=site["lat"], lon_deg=site["lon"])
            source = "egms"
        else:
            point_ds = _synthetic_insar(
                u_on,
                res_on.t,
                reference_time=ref_time,
                noise_mm=2.0,
                rng=np.random.default_rng(rng_master.integers(2**32)),
            )
            source = "synthetic"

        cmp = compare_insar_solver(
            point_ds,
            solver_t_s=res_on.t,
            u_solver_on_mm=u_on,
            u_solver_off_mm=u_off,
            reference_time=ref_time,
        )

        rows.append(
            {
                "site": site["name"],
                "lat_deg": site["lat"],
                "lon_deg": site["lon"],
                "source": source,
                "u_obs_range_mm": float(np.ptp(point_ds["displacement_mm"].values)),
                "u_solver_on_range_mm": float(np.ptp(u_on)),
                "u_solver_off_range_mm": float(np.ptp(u_off)),
                "rms_on_mm": cmp.rms_on_mm,
                "rms_off_mm": cmp.rms_off_mm,
                "residual_reduction": cmp.residual_reduction,
            }
        )
        print(
            f"  {site['name']:12s} [{source}]  "
            f"u_obs ptp={np.ptp(point_ds['displacement_mm'].values):.2f} mm  "
            f"RMS on/off={cmp.rms_on_mm:.3f}/{cmp.rms_off_mm:.3f} mm  "
            f"R={cmp.residual_reduction:+.3f}",
            file=sys.stderr,
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
