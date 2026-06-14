#!/usr/bin/env python3
"""R17 W2 — F3 boreal/equator subsurface warming ratio under
three Zhang 2005 winter n-factor scenarios.

Closes editor item 2: the F3 retreat from Arctic amplification
($1.49 \\to 0.94$) was framed as a "snow insulation / permafrost
decoupling" hand-wave in the accompanying paper, with the GTN-P
permafrost network named as the natural data source for modelling
the mechanism. The GTN-P parquet on disk is empty, so we instead
use *forward-modeling* through the existing 1-D heat solver: drive
``column_1d`` at every Huang-Pollack site with a Zhang $n_w$-modulated
CRU TS SAT history, and report the boreal/equator ratio of the
subsurface response under each scenario.

Output: ``outputs/global/f3_nfactor_corrected.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gt_theory.io.cru_ts import extract_sat_at_point, load_cru_ts
from gt_theory.solvers.column_1d import run_column_1d
from gt_theory.theory.zhang_nfactor import (
    SCENARIOS,
    ScenarioName,
    apply_nfactor,
)


YEAR_S = 365.25 * 86400.0
MEAN_MONTH_S = 30.44 * 86400.0
DEFAULT_SUMMARY = "outputs/global/ensemble_summary.parquet"
DEFAULT_CRU = Path("data/raw/cru_ts/cru_ts4.09.1901.2024.tmp.dat.nc")
DEFAULT_OUT = "outputs/global/f3_nfactor_corrected.parquet"

BASELINE = (1901, 1960)
RECENT = (2000, 2024)


def _site_response_under_scenario(
    *,
    cru_da,
    lat_deg: float,
    lon_deg: float,
    kappa: float,
    scenario: ScenarioName | str,
    probe_depth_m: float = 10.0,
) -> float | None:
    """Forward-simulate the column under the n-factor-modulated SAT
    and return the recent-window minus baseline-window mean
    *subsurface* anomaly at ``probe_depth_m``.

    Returns None if data are missing.
    """
    try:
        sat_series = extract_sat_at_point(
            cru_da,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tolerance_deg=1.0,
        )
    except Exception:
        return None
    times = pd.to_datetime(sat_series["time"].values)
    sat = sat_series.values.astype(float)
    if not np.all(np.isfinite(sat)) or sat.size < 360:
        return None

    years = times.year.to_numpy()
    months = times.month.to_numpy()
    sat_climatology = float(np.nanmean(sat))
    sat_anom = sat - sat_climatology

    if scenario == "raw":
        sat_drive_anom = sat_anom
    else:
        sat_drive_anom = apply_nfactor(
            sat_anom,
            months=months,
            years=years,
            scenario=scenario,
            lat_deg=lat_deg,
        )

    sat_drive = sat_climatology + sat_drive_anom
    # Forward through column_1d for n_months steps; need n_months+1 points.
    n_months = sat_drive.size
    duration_s = (n_months - 1) * MEAN_MONTH_S
    res = run_column_1d(
        depth_max_m=30.0,
        dz_m=1.0,
        duration_s=duration_s,
        dt_s=MEAN_MONTH_S,
        kappa=kappa,
        sat=sat_drive,
    )
    iz = int(np.argmin(np.abs(res.z - probe_depth_m)))
    gst_monthly = res.T[:, iz]
    baseline_mask = (years >= BASELINE[0]) & (years <= BASELINE[1])
    recent_mask = (years >= RECENT[0]) & (years <= RECENT[1])
    if baseline_mask.sum() < 50 or recent_mask.sum() < 24:
        return None
    return float(np.nanmean(gst_monthly[recent_mask]) - np.nanmean(gst_monthly[baseline_mask]))


def _band_ratio(values: np.ndarray, lats: np.ndarray) -> float:
    """boreal (|lat| >= 50) / equator (|lat| <= 20) mean ratio."""
    abs_lat = np.abs(lats)
    boreal = (abs_lat >= 50.0) & np.isfinite(values)
    equator = (abs_lat <= 20.0) & np.isfinite(values)
    if boreal.sum() == 0 or equator.sum() == 0:
        return float("nan")
    b = float(np.mean(values[boreal]))
    e = float(np.mean(values[equator]))
    return b / e if abs(e) > 1.0e-6 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cru", default=str(DEFAULT_CRU))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--max-sites",
        type=int,
        default=400,
        help="Cap sites processed to keep runtime manageable.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    summary = pd.read_parquet(Path(args.summary).expanduser().resolve())
    if args.smoke:
        # Latitude-stratified small subset for smoke.
        boreal = summary[np.abs(summary["lat_deg"]) >= 50.0].head(15)
        equator = summary[np.abs(summary["lat_deg"]) <= 20.0].head(15)
        summary = pd.concat([boreal, equator], ignore_index=True)
    elif args.max_sites < len(summary):
        # Stratified by latitude band for representativeness.
        boreal = summary[np.abs(summary["lat_deg"]) >= 50.0]
        equator = summary[np.abs(summary["lat_deg"]) <= 20.0]
        midlat = summary[(np.abs(summary["lat_deg"]) > 20.0) & (np.abs(summary["lat_deg"]) < 50.0)]
        per_band = args.max_sites // 3
        rng = np.random.default_rng(20260523)

        def _pick(d):
            n = min(per_band, len(d))
            if n == 0:
                return d.iloc[:0]
            idx = rng.choice(len(d), size=n, replace=False)
            return d.iloc[idx]

        summary = pd.concat([_pick(boreal), _pick(midlat), _pick(equator)], ignore_index=True)

    print(f"processing {len(summary)} sites", flush=True)
    print(f"loading CRU TS {args.cru}", flush=True)
    cru = load_cru_ts(args.cru)

    rows: list[dict[str, Any]] = []
    for i, (_, site) in enumerate(summary.iterrows()):
        lat = float(site["lat_deg"])
        lon = float(site["lon_deg"])
        kappa = float(site["kappa_median"])
        row: dict[str, Any] = {
            "site_id": site["site_id"],
            "lat_deg": lat,
            "lon_deg": lon,
            "kappa_median": kappa,
        }
        for scen in ("raw",) + SCENARIOS:
            val = _site_response_under_scenario(
                cru_da=cru,
                lat_deg=lat,
                lon_deg=lon,
                kappa=kappa,
                scenario=scen,
            )
            row[f"delta_gst_{scen}"] = val if val is not None else float("nan")
        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(summary):
            print(f"  {i + 1}/{len(summary)}", flush=True)

    df = pd.DataFrame(rows)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\nwrote {out_path} ({len(df)} sites)")

    lats = df["lat_deg"].to_numpy()
    print("\n=== R17 W2 headline (boreal/equator ratio) ===")
    print(f"  target (SAT surface ratio):       1.49")
    print(f"  observed (accompanying paper):    0.94")
    for scen in ("raw",) + SCENARIOS:
        ratio = _band_ratio(df[f"delta_gst_{scen}"].to_numpy(), lats)
        print(f"  {scen:30s}: {ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
