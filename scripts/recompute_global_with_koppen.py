#!/usr/bin/env python3
"""R17 W5 — recompute the global +0.78 K headline under
Köppen-Geiger biome-stratified weighting.

Closes editor item 5: spatial reweighting is plain 10-degree
latitude bands in the manuscript today; the reviewer asks for
Köppen-stratified weighting.

The script:
1. Loads ``outputs/global/ensemble_summary.parquet`` (948 sites).
2. Computes a 1980-2010 CRU TS T + P climatology.
3. Classifies each site into one of 30 Köppen classes via the
   Peel--Finlayson--McMahon 2007 decision tree (matching Beck 2018).
4. Computes biome-weighted vs latitude-band-weighted global median
   recent (0-25 yr) GST anomaly.
5. Emits ``outputs/global/ensemble_summary_koppen.parquet`` with
   the per-site Köppen class added and the weighted aggregates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.fingerprints.f4_budget import (
    horvitz_thompson_weights,
    koppen_weights,
)
from gt_theory.io.cru_ts import load_cru_ts
from gt_theory.io.koppen import (
    cru_climatology,
    koppen_code,
    koppen_for_sites,
)


DEFAULT_SUMMARY = "outputs/global/ensemble_summary.parquet"
DEFAULT_CRU_TMP = Path("data/raw/cru_ts/cru_ts4.09.1901.2024.tmp.dat.nc")
DEFAULT_CRU_PRE = Path.home() / "Documents/climate_shift/data/cru_ts4.09.1901.2024.pre.dat.nc"
DEFAULT_OUT = "outputs/global/ensemble_summary_koppen.parquet"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    v = values[finite]
    w = weights[finite]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    half = 0.5 * cw[-1]
    idx = int(np.searchsorted(cw, half))
    idx = min(idx, v.size - 1)
    return float(v[idx])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cru-tmp", default=str(DEFAULT_CRU_TMP))
    parser.add_argument("--cru-pre", default=str(DEFAULT_CRU_PRE))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--baseline-start",
        type=int,
        default=1980,
    )
    parser.add_argument(
        "--baseline-end",
        type=int,
        default=2010,
    )
    args = parser.parse_args(argv)

    summary = pd.read_parquet(Path(args.summary).expanduser().resolve())
    print(f"loaded {len(summary)} sites from {args.summary}", flush=True)

    print(f"loading CRU TS T  {args.cru_tmp}", flush=True)
    cru_tmp = load_cru_ts(args.cru_tmp, variable="tmp")
    print(f"loading CRU TS P  {args.cru_pre}", flush=True)
    cru_pre = load_cru_ts(args.cru_pre, variable="pre")

    print(
        f"computing 1980-{args.baseline_end} T + P climatology...",
        flush=True,
    )
    clim = cru_climatology(
        cru_tmp=cru_tmp,
        cru_pre=cru_pre,
        baseline=(args.baseline_start, args.baseline_end),
    )

    print("classifying sites...", flush=True)
    classes = koppen_for_sites(
        clim,
        lats_deg=summary["lat_deg"].tolist(),
        lons_deg=summary["lon_deg"].tolist(),
    )
    codes = np.array([koppen_code(c) for c in classes], dtype=int)
    print(f"  classified: {int((codes > 0).sum())}/{len(codes)} sites", flush=True)

    # Per-site weights under both schemes.
    lat_w, lat_band_w = horvitz_thompson_weights(summary["lat_deg"].values)
    kop_w, kop_per_code = koppen_weights(codes)

    gst = summary["gst_median_recent_K"].to_numpy(dtype=float)
    lat_band_median = _weighted_median(gst, lat_w)
    kop_median = _weighted_median(gst, kop_w)

    print("\n=== R17 W5 headline ===")
    print(f"sites classified: {int((codes > 0).sum())}/{len(codes)}")
    print(f"lat-band weighted global median ΔT_GST:  {lat_band_median:.3f} K")
    print(f"Köppen weighted global median ΔT_GST:    {kop_median:.3f} K")
    print(f"shift Köppen - lat-band:                  {kop_median - lat_band_median:+.3f} K")

    # Per-biome stats.
    summary_out = summary.copy()
    summary_out["koppen_class"] = classes
    summary_out["koppen_code"] = codes
    summary_out["lat_band_weight_m2"] = lat_w
    summary_out["koppen_weight_m2"] = kop_w
    summary_out.attrs["global_median_lat_band"] = float(lat_band_median)
    summary_out.attrs["global_median_koppen"] = float(kop_median)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out.to_parquet(out_path)
    print(f"\nwrote {out_path}  ({len(summary_out)} sites)")

    # Print biome occupancy table.
    print("\nper-biome occupancy + median ΔT_GST:")
    by_class = (
        summary_out.groupby("koppen_class", dropna=False)
        .agg(
            n=("site_id", "count"),
            median_gst=("gst_median_recent_K", "median"),
        )
        .sort_values("n", ascending=False)
    )
    print(by_class.head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
