#!/usr/bin/env python3
"""Forward-simulate a synthetic borehole temperature profile from a
deterministic GST history derived from the site_id and a base seed.

Output is a parquet file with two columns:
  depth_m  : metres, positive downward
  temperature_c : present-day temperature (deg C), including the imposed
                  linear geothermal background.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importable when the project is installed (``pip install -e .``) or via
# PYTHONPATH=src on developer machines.
from gt_theory.solvers import run_column_1d

YEAR_S = 365.25 * 86400.0


def _seed_from(site_id: str, seed_base: int) -> int:
    h = hashlib.sha1(site_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ seed_base) & 0xFFFFFFFF


def _synthetic_gst_history(site_id: str, seed_base: int, nt: int) -> np.ndarray:
    """Build a deterministic monthly GST anomaly time series of length
    ``nt``.  Combines a slow centennial trend with a seasonal cycle and
    a small bit of jitter so each synthetic site has a distinct shape."""
    rng = np.random.default_rng(_seed_from(site_id, seed_base))
    months = np.arange(nt)
    # Slow ramp: 0 -> ~+1.0 K over the integration window.
    trend = np.linspace(0.0, rng.uniform(0.4, 1.5), nt)
    # Seasonal cycle (1-2 K peak-to-peak), phase randomised.
    seasonal_amp = rng.uniform(0.5, 1.0)
    seasonal_phase = rng.uniform(0.0, 2.0 * np.pi)
    seasonal = seasonal_amp * np.cos(2.0 * np.pi * months / 12.0 + seasonal_phase)
    return trend + seasonal


def _site_geothermal_background(site_id: str, seed_base: int, z: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(_seed_from(site_id + "_bkg", seed_base))
    T0 = rng.uniform(2.0, 18.0)
    dTdz = rng.uniform(0.015, 0.035)  # K m^-1
    return T0 + dTdz * z


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", required=True, type=str)
    parser.add_argument("--kappa", required=True, type=float)
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--n-years", type=int, default=120, help="Integration window in years.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    dt_s = YEAR_S / 12.0
    duration_s = args.n_years * YEAR_S
    nt = int(round(duration_s / dt_s)) + 1
    sat = _synthetic_gst_history(args.site_id, args.seed_base, nt)

    result = run_column_1d(
        depth_max_m=600.0,
        dz_m=5.0,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=args.kappa,
        sat=sat,
        n_winter=1.0,
    )

    background = _site_geothermal_background(args.site_id, args.seed_base, result.z)
    T_present_c = result.T[-1] + background

    df = pd.DataFrame(
        {
            "depth_m": result.z.astype(np.float64),
            "temperature_c": T_present_c.astype(np.float64),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
