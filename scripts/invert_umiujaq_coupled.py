#!/usr/bin/env python3
"""Joint T+p Bayesian inversion driver for the Umiujaq supersite.

Reads operator-prepared observation netCDFs out of
``data/supersite_umiujaq/processed/`` (or a synthetic-twin
companion if --synthetic is set) and runs the adaptive-MH ensemble
implemented in :mod:`gt_theory.inversion.bayes_coupled`.

The synthetic-twin path is the canonical smoke-test until the
operator has finished extracting the Nordicana D / Borealis zips;
it generates noisy observations from a chosen ground-truth vector
via the same forward operator, then verifies recovery.

Outputs:

- ``outputs/supersite_umiujaq/posterior_samples.npz`` (chains, log
  posterior, accept rate, truth-if-known, param-names)
- A stderr summary table: per-parameter median + 90% CI; flag
  identifiability (|posterior corr| > 0.9 between any pair).

Usage::

    # synthetic-twin smoke run (default truth: gamma=1.0)
    python scripts/invert_umiujaq_coupled.py --synthetic

    # real-data run (assumes operator has prepared processed/*.nc)
    python scripts/invert_umiujaq_coupled.py \\
        --T-obs data/supersite_umiujaq/processed/vdtbs_T_profile.nc \\
        --p-obs data/supersite_umiujaq/processed/immatsiak1_head.nc \\
        --config data/supersite_umiujaq/site_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from gt_theory.inversion import (
    CoupledPosteriorResult,
    coupled_forward,
    invert_coupled_posterior,
)

YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


def _synthetic_setup(
    *,
    duration_s: float,
    dt_s: float,
    sigma_T: float,
    sigma_p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    nt = int(round(duration_s / dt_s)) + 1
    t_solver = np.arange(nt) * dt_s
    sat = -3.0 + 10.0 * np.sin(2.0 * np.pi * t_solver / YEAR_S)

    t_obs_s = np.linspace(0.0, duration_s, 12)
    z_obs_m = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    forward_kwargs = {
        "depth_max_m": 6.0,
        "dz_m": 0.5,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "sat": sat,
        "t_obs_s": t_obs_s,
        "z_obs_m": z_obs_m,
        "z_piezo_m": 5.0,
    }

    truth = np.array([-5.0, 0.40, 1.80, 0.3, 1.0])
    T_clean, p_clean = coupled_forward(truth, **forward_kwargs)
    T_obs = T_clean + rng.normal(0.0, sigma_T, size=T_clean.shape)
    p_obs = p_clean + rng.normal(0.0, sigma_p, size=p_clean.shape)
    return T_obs, p_obs, forward_kwargs, truth


def _real_setup(
    *,
    T_obs_path: Path,
    p_obs_path: Path,
    sat_path: Path,
    duration_s: float,
    dt_s: float,
    z_piezo_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    T_ds = xr.open_dataset(T_obs_path)
    p_ds = xr.open_dataset(p_obs_path)
    sat_ds = xr.open_dataset(sat_path)

    T_obs = T_ds["T_degC"].values
    p_obs = p_ds["head_m"].values * 9.81 * 1000.0  # m -> Pa (rho g)
    z_obs_m = T_ds["depth_m"].values

    nt = int(round(duration_s / dt_s)) + 1
    t_obs_s = np.linspace(0.0, duration_s, T_obs.shape[0])

    sat_var = [v for v in sat_ds.data_vars if "T" in v.upper()][0]
    sat_raw = sat_ds[sat_var].values
    sat = np.interp(
        np.arange(nt) * dt_s,
        np.linspace(0.0, duration_s, sat_raw.size),
        sat_raw,
    )

    forward_kwargs = {
        "depth_max_m": float(z_obs_m.max()) + 1.0,
        "dz_m": 0.25,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "sat": sat,
        "t_obs_s": t_obs_s,
        "z_obs_m": z_obs_m,
        "z_piezo_m": z_piezo_m,
    }
    return T_obs, p_obs, forward_kwargs


def _summarise(result: CoupledPosteriorResult) -> str:
    samples = result.flat_samples
    lines = ["Umiujaq joint T+p adaptive-MH posterior summary"]
    lines.append(f"  walkers={result.chains.shape[0]}  steps_post_burn={result.chains.shape[1]}")
    lines.append(f"  mean accept rate: {float(result.accept_rate.mean()):.2f}")
    for i, name in enumerate(result.param_names):
        med = float(np.median(samples[:, i]))
        lo, hi = result.credible_interval(i, level=0.90)
        truth_str = f"  truth={result.truth[i]:+.3f}" if result.truth is not None else ""
        lines.append(f"  {name:<24s} median={med:+.3f}  90% CI=[{lo:+.3f}, {hi:+.3f}]{truth_str}")

    # Identifiability flag: highest pairwise correlation.
    corr = np.corrcoef(samples.T)
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    abs_pairs = sorted(
        [
            (abs(corr[i, j]), result.param_names[i], result.param_names[j], corr[i, j])
            for i, j in zip(iu, ju)
        ],
        reverse=True,
    )
    lines.append("  posterior correlations (|r| descending):")
    for abs_r, n_i, n_j, r in abs_pairs[:5]:
        flag = "  *** identifiability concern ***" if abs_r > 0.9 else ""
        lines.append(f"    {n_i} ↔ {n_j}: r={r:+.3f}{flag}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run against a synthetic-twin observation (default if no --T-obs/--p-obs)",
    )
    parser.add_argument("--T-obs", default=None, help="Observed T-profile netCDF")
    parser.add_argument("--p-obs", default=None, help="Observed piezometer head netCDF")
    parser.add_argument(
        "--sat", default="data/supersite_umiujaq/processed/d9_VDTSYBU_Tsurf_daily.nc"
    )
    parser.add_argument("--z-piezo", type=float, default=5.0)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--dt-days", type=float, default=5.0)
    parser.add_argument("--n-walkers", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--n-burn", type=int, default=120)
    parser.add_argument("--sigma-T", type=float, default=0.10)
    parser.add_argument("--sigma-p", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=2026_05_22)
    parser.add_argument("--out", default="outputs/supersite_umiujaq/posterior_samples.npz")
    args = parser.parse_args(argv)

    duration_s = args.years * YEAR_S
    dt_s = args.dt_days * DAY_S
    rng = np.random.default_rng(args.seed)

    if args.synthetic or (args.T_obs is None and args.p_obs is None):
        T_obs, p_obs, forward_kwargs, truth = _synthetic_setup(
            duration_s=duration_s,
            dt_s=dt_s,
            sigma_T=args.sigma_T,
            sigma_p=args.sigma_p,
            rng=rng,
        )
    else:
        if args.T_obs is None or args.p_obs is None:
            parser.error("--T-obs and --p-obs are both required for a real-data run")
        T_obs, p_obs, forward_kwargs = _real_setup(
            T_obs_path=Path(args.T_obs),
            p_obs_path=Path(args.p_obs),
            sat_path=Path(args.sat),
            duration_s=duration_s,
            dt_s=dt_s,
            z_piezo_m=args.z_piezo,
        )
        truth = None

    result = invert_coupled_posterior(
        T_obs=T_obs,
        p_obs=p_obs,
        sigma_T=args.sigma_T,
        sigma_p=args.sigma_p,
        forward_kwargs=forward_kwargs,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        rng=rng,
        truth=truth,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        chains=result.chains,
        log_post=result.log_post,
        accept_rate=result.accept_rate,
        truth=np.asarray(result.truth if result.truth is not None else np.full(5, np.nan)),
        param_names=np.asarray(result.param_names),
    )
    print(_summarise(result), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
