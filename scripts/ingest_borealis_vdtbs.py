#!/usr/bin/env python3
"""R18-B: ingest the Borealis VDTBS lithalsa thermistor bundle and
write a single processed NetCDF on a unified ``(time, depth_m,
borehole)`` grid.

Input
-----
``data/supersite_umiujaq/raw/borealis_vdtbs/data_ta{1..5}_{0114|1424|0121}_post_cleaning.tab``
(post-cleaning files only; pre-cleaning files are duplicates with
spike artefacts).

Output
------
``data/supersite_umiujaq/processed/vdtbs_thermistors.nc`` with
variables:

* ``T_degC(time, depth_m, borehole)`` -- thermistor temperature in
  degrees Celsius.

The Borealis archive packages each of the 5 boreholes (TA1..TA5) into
either one or two segments depending on instrument changes; this
ingest concatenates the segments per borehole and merges across
boreholes into a single (time, depth, borehole) grid.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_DIR = Path("data/supersite_umiujaq/raw/borealis_vdtbs")
OUT_PATH = Path("data/supersite_umiujaq/processed/vdtbs_thermistors.nc")

DEPTH_RE = re.compile(r"^(?P<depth>[0-9]+(?:\.[0-9]+)?)m$")
EXCEL_ORIGIN = "1899-12-30"


def _load_tab(path: Path) -> xr.Dataset:
    """Load one .tab file into an xarray Dataset (time, depth_m)."""
    df = pd.read_csv(path, sep="\t")
    if "DateTime" not in df.columns:
        raise KeyError(f"DateTime column missing in {path.name}")
    # Excel serial date → real datetime.  Some rows may be NaN at the
    # tail of an instrument record; drop them.
    df = df.dropna(subset=["DateTime"])
    times = pd.to_datetime(df["DateTime"].to_numpy(), origin=EXCEL_ORIGIN, unit="D")

    depth_cols: list[tuple[float, str]] = []
    for col in df.columns:
        if col == "DateTime":
            continue
        m = DEPTH_RE.match(str(col))
        if m is None:
            continue
        depth_cols.append((float(m.group("depth")), col))
    if not depth_cols:
        raise ValueError(f"No depth columns matched in {path.name}: {list(df.columns)}")
    depth_cols.sort()

    depths = np.array([d for d, _ in depth_cols], dtype=float)
    vals = np.stack([df[col].to_numpy(dtype=float) for _, col in depth_cols], axis=1)
    return xr.Dataset(
        data_vars={"T_degC": (("time", "depth_m"), vals)},
        coords={"time": times, "depth_m": depths},
    )


def _segments_for_borehole(borehole_id: str) -> list[Path]:
    """All post_cleaning .tab files belonging to one borehole, sorted
    by name (which sorts chronologically because the Borealis filename
    convention encodes the year range)."""
    files = sorted(RAW_DIR.glob(f"data_{borehole_id}_*_post_cleaning.tab"))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    boreholes = ["ta1", "ta2", "ta3", "ta4", "ta5"]
    per_bh: dict[str, xr.Dataset] = {}
    for bh in boreholes:
        files = _segments_for_borehole(bh)
        if not files:
            print(f"warning: no post_cleaning .tab found for {bh}; skipping")
            continue
        # Concatenate segments along time, take the *post-cleaning* set
        # only.  Some segments overlap (e.g. ta1_0114 and ta1_1424
        # share part of 2014); we keep the first segment's values for
        # the overlap, which is the production convention in the
        # Borealis README.
        segs = [_load_tab(f) for f in files]
        ds = xr.concat(segs, dim="time")
        # Drop exact-duplicate timestamps.
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
        per_bh[bh.upper()] = ds
        print(f"loaded {bh.upper()}: nt = {ds.sizes['time']}, depths = {ds['depth_m'].values}")

    # Build the union of depth axes across boreholes and reindex.
    all_depths = np.unique(np.concatenate([ds["depth_m"].values for ds in per_bh.values()]))
    # Build the union of time axes.  Different boreholes have different
    # sample spacings (mostly daily, some sub-daily); we use the union
    # of all unique timestamps without resampling so the operator can
    # downstream resample to a common frequency as needed.
    all_times = np.unique(np.concatenate([ds["time"].values for ds in per_bh.values()]))

    nt = all_times.size
    nz = all_depths.size
    nbh = len(per_bh)
    out_T = np.full((nt, nz, nbh), np.nan, dtype=float)
    bh_labels = []
    for ib, (bh, ds) in enumerate(per_bh.items()):
        bh_labels.append(bh)
        t_idx = np.searchsorted(all_times, ds["time"].values)
        d_idx = np.searchsorted(all_depths, ds["depth_m"].values)
        for k, (it, _t) in enumerate(zip(t_idx, ds["time"].values)):
            out_T[it, d_idx, ib] = ds["T_degC"].values[k]

    out = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m", "borehole"), out_T),
        },
        coords={
            "time": all_times,
            "depth_m": all_depths,
            "borehole": np.array(bh_labels),
        },
        attrs={
            "source_doi": "10.5683/SP3/QSRW0I",
            "product": "Tasiapik VDTBS lithalsa thermistors (post-cleaning)",
            "citation": (
                "Fortier, P., Fortier, R., Allard, M., Lemieux, J.-M., "
                "Sarrazin, D. (2025). Subsurface ground temperature data "
                "from an instrumented permafrost mound, Tasiapik Valley, "
                "Umiujaq, Nunavik, Quebec, Canada (2001-2024). Borealis V1."
            ),
        },
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    # Headline numbers.
    T = out["T_degC"].values
    finite = np.isfinite(T)
    print(f"wrote {out_path}")
    print(f"  shape: time x depth x borehole = {T.shape}")
    print(
        f"  time range: {pd.Timestamp(all_times[0]).date()} -> {pd.Timestamp(all_times[-1]).date()}"
    )
    print(f"  depths (m): {all_depths.tolist()}")
    print(f"  boreholes: {bh_labels}")
    print(f"  finite-fraction: {finite.mean():.3f}")
    print(f"  T range: [{np.nanmin(T):.2f}, {np.nanmax(T):.2f}] degC")
    print(f"  fraction below 0 degC: {(T[finite] < 0.0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
