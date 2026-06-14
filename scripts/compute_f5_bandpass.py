#!/usr/bin/env python3
"""R17 W4 — compute the F5 diffusive-lag fingerprint on a
latitude-stratified subset of the 948 Huang-Pollack sites with
proper Butterworth-bandpass filtering of the SAT and forward-
simulated GST series.

Closes editor item 4: F5 was never implemented in the codebase
before this revision (no F5 module, no test, no script). The
"117 months at 20 m" figure quoted in the accompanying paper was
not produced by any running test; it appears to be a back-of-
envelope cross-correlation of the secular-warming-dominated raw
SAT/GST series.

This script does the actual computation: for each of the
selected sites, it loads the monthly CRU TS SAT at the (lat, lon),
forward-simulates the monthly GST response at 5 m and 20 m via
the existing 1-D heat solver, and computes the cross-correlation
lag against the SAT forcing (a) unfiltered and (b) bandpass-
filtered to the (3-yr, 30-yr) decadal band where the diffusive
prediction is valid.

Output: ``outputs/global/f5_bandpass.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gt_theory.fingerprints.f5_diffusive_lag import compute_f5_at_site
from gt_theory.io.cru_ts import extract_sat_at_point, load_cru_ts


DEFAULT_SUMMARY = "outputs/global/ensemble_summary.parquet"
DEFAULT_CRU = Path("data/raw/cru_ts/cru_ts4.09.1901.2024.tmp.dat.nc")
DEFAULT_OUT = "outputs/global/f5_bandpass.parquet"


def _latitude_stratified_subset(
    df: pd.DataFrame,
    *,
    n_per_band: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Take ``n_per_band`` random sites from each of three latitude
    bands so the F5 distribution is geographically balanced."""
    rng = np.random.default_rng(seed)
    abs_lat = np.abs(df["lat_deg"].values)
    masks = {
        "tropical": abs_lat < 30.0,
        "mid": (abs_lat >= 30.0) & (abs_lat < 60.0),
        "boreal": abs_lat >= 60.0,
    }
    picks: list[pd.DataFrame] = []
    for label, m in masks.items():
        candidates = df[m]
        n_pick = min(n_per_band, len(candidates))
        if n_pick == 0:
            continue
        idx = rng.choice(len(candidates), size=n_pick, replace=False)
        picks.append(candidates.iloc[idx])
    return pd.concat(picks, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cru", default=str(DEFAULT_CRU))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n-per-band", type=int, default=50)
    parser.add_argument("--low-yr", type=float, default=30.0)
    parser.add_argument("--high-yr", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    summary = pd.read_parquet(Path(args.summary).expanduser().resolve())
    n_per = 5 if args.smoke else args.n_per_band
    subset = _latitude_stratified_subset(summary, n_per_band=n_per)
    print(f"selected {len(subset)} sites (n_per_band={n_per})", flush=True)

    print(f"loading CRU TS {args.cru}", flush=True)
    cru = load_cru_ts(args.cru)

    rows: list[dict[str, Any]] = []
    n_total = len(subset)
    for i, (_, site) in enumerate(subset.iterrows()):
        lat = float(site["lat_deg"])
        lon = float(site["lon_deg"])
        kappa = float(site["kappa_median"])
        try:
            sat_series = extract_sat_at_point(
                cru,
                lat_deg=lat,
                lon_deg=lon,
                tolerance_deg=1.0,
            )
        except Exception:
            continue
        sat_arr = sat_series.values.astype(float)
        if not np.all(np.isfinite(sat_arr)) or sat_arr.size < 360:
            continue
        # Convert to anomaly relative to long-term mean.
        sat_anom = sat_arr - float(np.nanmean(sat_arr))
        try:
            result = compute_f5_at_site(
                site_id=str(site["site_id"]),
                lat_deg=lat,
                lon_deg=lon,
                sat_monthly_K=sat_anom,
                kappa_m2_per_s=kappa,
                low_yr=args.low_yr,
                high_yr=args.high_yr,
            )
        except Exception as e:
            print(f"  site {site['site_id']} failed: {e}", flush=True)
            continue
        rows.append(
            {
                "site_id": result.site_id,
                "lat_deg": result.lat_deg,
                "lon_deg": result.lon_deg,
                "kappa_m2_per_s": result.kappa_m2_per_s,
                "lag_5m_unfiltered_months": result.lag_5m_unfiltered_months,
                "lag_5m_filtered_months": result.lag_5m_filtered_months,
                "lag_20m_unfiltered_months": result.lag_20m_unfiltered_months,
                "lag_20m_filtered_months": result.lag_20m_filtered_months,
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == n_total:
            print(f"  {i + 1}/{n_total} sites processed", flush=True)

    df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\nwrote {out_path} ({len(df)} sites)\n")

    # Headline summary.
    print("=== R17 W4 headline ===")
    print(f"  median lag 5m unfiltered:  {df['lag_5m_unfiltered_months'].median():.0f} mo")
    print(f"  median lag 5m filtered:    {df['lag_5m_filtered_months'].median():.0f} mo")
    print(f"  median lag 20m unfiltered: {df['lag_20m_unfiltered_months'].median():.0f} mo")
    print(f"  median lag 20m filtered:   {df['lag_20m_filtered_months'].median():.0f} mo")
    print(
        "Theoretical lag at 20 m, kappa=1e-6: "
        f"{(20.0**2 / (4 * 1.0e-6)) / (30.44 * 86400.0):.1f} months"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
