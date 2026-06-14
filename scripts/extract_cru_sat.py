#!/usr/bin/env python3
"""Extract a monthly SAT time series at each borehole site from CRU TS v4.

Reads the curated catalog or an all-sites CSV, opens the CRU TS netCDF
lazily (it is multi-GB), samples the nearest-neighbour grid cell at each
site's (lat, lon), and writes a single parquet with columns ``site_id``,
``time``, ``sat_c``.

Usage::

    python scripts/extract_cru_sat.py \\
        --catalog-csv catalogs/all_sites.csv \\
        --cru-nc data/raw/cru_ts/cru_ts4.09.1901.2024.tmp.dat.nc \\
        --out outputs/cru_sat_per_site.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-csv", required=True, type=Path)
    parser.add_argument("--cru-nc", required=True, type=Path)
    parser.add_argument("--variable", default="tmp", help="CRU TS variable name (default: tmp)")
    parser.add_argument("--site-id-col", default="site_id", type=str, help="Site ID column name")
    parser.add_argument("--lat-col", default="lat_deg", type=str)
    parser.add_argument("--lon-col", default="lon_deg", type=str)
    parser.add_argument(
        "--subset",
        nargs="*",
        type=str,
        default=None,
        help="Optional list of site IDs to restrict the extraction to.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    cat = pd.read_csv(args.catalog_csv)
    if args.subset:
        cat = cat[cat[args.site_id_col].isin(set(args.subset))]
    if cat.empty:
        print("ERROR: no sites after subset filter", file=sys.stderr)
        return 2

    ds = xr.open_dataset(args.cru_nc, decode_times=True)
    if args.variable not in ds.data_vars:
        print(f"ERROR: variable {args.variable!r} not in {args.cru_nc.name}", file=sys.stderr)
        return 3
    da = ds[args.variable]

    frames = []
    for _, row in cat.iterrows():
        sid = str(row[args.site_id_col])
        lat = float(row[args.lat_col])
        lon = float(row[args.lon_col])
        try:
            sample = da.sel(lat=lat, lon=lon, method="nearest")
        except Exception as exc:
            print(f"WARN: skipping {sid}: {exc}", file=sys.stderr)
            continue
        # Drop any all-NaN cells (CRU TS masks out oceans / Antarctica).
        # ``.values`` is the right call here despite the PD011 hint — these
        # are xarray DataArrays, not pandas, and ``.to_numpy()`` is not
        # available on the time coordinate.
        values = sample.values  # noqa: PD011
        if np.all(np.isnan(values)):
            print(f"WARN: {sid} (lat={lat}, lon={lon}) is in a CRU-masked cell", file=sys.stderr)
            continue
        df = pd.DataFrame(
            {
                "site_id": sid,
                "time": sample["time"].values,  # noqa: PD011
                "sat_c": values.astype(np.float32),
            }
        )
        frames.append(df)

    if not frames:
        print("ERROR: no sites produced a valid time series", file=sys.stderr)
        return 4

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_sites = out["site_id"].nunique()
    n_months = out.groupby("site_id").size().iloc[0]
    print(
        f"wrote {len(out)} rows ({n_sites} sites, {n_months} months/site) -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
