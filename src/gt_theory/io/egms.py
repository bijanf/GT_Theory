"""Reader for European Ground Motion Service (EGMS) PS-InSAR products.

EGMS Level 3 ortho-vertical and ortho-horizontal CSVs distribute
per-point time series of ground deformation (in mm) plus a fitted
linear velocity (in mm/yr) for every Persistent Scatterer or
Distributed Scatterer survived by the GNSS-anchored Sentinel-1
processing chain. Coverage is the Copernicus pan-European domain
on a 100 m grid, 2015-2024 (with rolling updates). License is
CC-BY 4.0 via the Copernicus Land Monitoring Service.

This module exposes one entry point :func:`load_egms_csv` that
returns an :class:`xarray.Dataset` indexed by ``(point, time)``,
suitable for the InSAR-vs-solver cross-check at the empirical-paper
supersites + high-Pe_T H-P candidates.

For US / non-EU supersites that fall outside EGMS coverage (e.g.
Utah FORGE), the ASF DAAC distributes Sentinel-1 SLCs that you can
run through ISCE/HyP3 to produce an analogous time series; that
pipeline is *not* implemented here. Document the ASF fallback in
``data/insar/README.md``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# A time-series column in an EGMS CSV is an 8-digit ISO-date string
# YYYYMMDD. Metadata columns are everything else.
_TIME_COL_RE = re.compile(r"^(?P<date>20\d{6})$")


def _split_columns(columns: Iterable[str]) -> tuple[list[str], list[str], list[pd.Timestamp]]:
    """Partition columns into (metadata, time-series, parsed-timestamps)."""
    meta: list[str] = []
    ts_cols: list[str] = []
    ts_dates: list[pd.Timestamp] = []
    for col in columns:
        m = _TIME_COL_RE.match(str(col))
        if m is None:
            meta.append(str(col))
        else:
            ts_cols.append(str(col))
            ts_dates.append(pd.to_datetime(m.group("date"), format="%Y%m%d"))
    # Sort time-series columns by date so the output axis is monotone.
    order = np.argsort(ts_dates)
    return meta, [ts_cols[i] for i in order], [ts_dates[i] for i in order]


def load_egms_csv(
    path: str | Path,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    velocity_col: str = "velocity",
    velocity_std_col: str = "velocity_std",
    pid_col: str = "pid",
    sep: str = ",",
) -> xr.Dataset:
    """Load one EGMS Level-3 ortho-vertical or ortho-horizontal CSV.

    Parameters
    ----------
    path
        Path to a single CSV from the EGMS portal.
    lat_col, lon_col
        Column names of point latitude / longitude in decimal degrees.
    velocity_col
        Column name of the fitted linear velocity (mm/yr).
    velocity_std_col
        Column name of the velocity standard deviation (mm/yr); if
        absent in the file, the returned Dataset gets ``velocity_std`` =
        NaN.
    pid_col
        Column name of the per-point persistent identifier; if absent,
        synthesised as ``f"pt_{i:06d}"``.
    sep
        CSV delimiter; EGMS uses comma by default.

    Returns
    -------
    xarray.Dataset
        Coords: ``point`` (str), ``time`` (datetime64).
        Variables: ``lat`` (deg), ``lon`` (deg), ``velocity_mm_yr``,
        ``velocity_std_mm_yr``, ``displacement_mm(point, time)``.
        Attributes record source DOI placeholder + license.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"EGMS CSV not found: {p}\n"
            f"Download recipe in data/insar/README.md (EGMS portal, "
            f"Copernicus CC-BY)."
        )

    df = pd.read_csv(p, sep=sep, engine="python")
    meta_cols, ts_cols, ts_dates = _split_columns(df.columns)

    for required in (lat_col, lon_col, velocity_col):
        if required not in meta_cols:
            raise KeyError(f"Expected column {required!r} in {p.name}; have: {meta_cols}")

    if pid_col in meta_cols:
        pids = df[pid_col].astype(str).to_numpy()
    else:
        pids = np.array([f"pt_{i:06d}" for i in range(len(df))])

    if velocity_std_col in meta_cols:
        velocity_std = df[velocity_std_col].to_numpy(dtype=float)
    else:
        velocity_std = np.full(len(df), np.nan)

    if ts_cols:
        displacement = df[ts_cols].to_numpy(dtype=float)  # (n_points, n_t)
    else:
        displacement = np.empty((len(df), 0), dtype=float)

    ds = xr.Dataset(
        data_vars={
            "lat": (("point",), df[lat_col].to_numpy(dtype=float)),
            "lon": (("point",), df[lon_col].to_numpy(dtype=float)),
            "velocity_mm_yr": (("point",), df[velocity_col].to_numpy(dtype=float)),
            "velocity_std_mm_yr": (("point",), velocity_std),
            "displacement_mm": (("point", "time"), displacement),
        },
        coords={
            "point": pids,
            "time": np.asarray(ts_dates, dtype="datetime64[ns]"),
        },
        attrs={
            "source_file": str(p),
            "product": "European Ground Motion Service (EGMS)",
            "license": "CC-BY 4.0 via Copernicus Land Monitoring Service",
            "url": "https://land.copernicus.eu/en/products/european-ground-motion-service",
            "lat_col": lat_col,
            "lon_col": lon_col,
        },
    )
    ds["velocity_mm_yr"].attrs["units"] = "mm yr-1"
    ds["velocity_std_mm_yr"].attrs["units"] = "mm yr-1"
    ds["displacement_mm"].attrs["units"] = "mm"
    return ds


def points_in_bbox(
    ds: xr.Dataset,
    *,
    lat_lo: float,
    lat_hi: float,
    lon_lo: float,
    lon_hi: float,
) -> xr.Dataset:
    """Subset an EGMS Dataset to a latitude-longitude bounding box."""
    mask = (ds.lat >= lat_lo) & (ds.lat <= lat_hi) & (ds.lon >= lon_lo) & (ds.lon <= lon_hi)
    return ds.where(mask, drop=True)


def nearest_point(ds: xr.Dataset, *, lat_deg: float, lon_deg: float) -> xr.Dataset:
    """Return the single closest point to a target lat/lon as a
    1-point Dataset slice."""
    d2 = (ds.lat - lat_deg) ** 2 + (ds.lon - lon_deg) ** 2
    idx = int(np.argmin(d2.values))
    return ds.isel(point=idx)
