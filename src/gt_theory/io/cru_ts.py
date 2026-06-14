"""Minimal reader for CRU TS v4 gridded monthly surface-air-temperature.

CRU TS files are netCDF with dimensions ``(time, lat, lon)`` and a 0.5-deg
global land grid.  This module exposes a thin :func:`load_cru_ts` wrapper
returning an :class:`xarray.DataArray` plus :func:`extract_sat_at_point`
for nearest-pixel sampling at a borehole site.

Anything heavier (regridding, masking, anomaly construction) is left to
the fingerprint modules where it belongs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def load_cru_ts(
    path: str | Path,
    *,
    variable: str = "tmp",
) -> xr.DataArray:
    """Open a CRU TS netCDF and return the named variable as a DataArray.

    Parameters
    ----------
    path
        Path to a CRU TS netCDF file (e.g. ``cru_ts4.07.1901.2022.tmp.dat.nc``).
    variable
        Variable name to extract.  Defaults to ``"tmp"`` (monthly mean SAT
        in deg C).

    Returns
    -------
    xarray.DataArray
        Indexed by ``(time, lat, lon)``.  No regridding is applied.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"CRU TS file not found: {p}")
    ds = xr.open_dataset(p, decode_times=True)
    if variable not in ds.data_vars:
        raise KeyError(f"Variable {variable!r} not in {p.name} (have: {list(ds.data_vars)})")
    return ds[variable]


def extract_sat_at_point(
    da: xr.DataArray,
    *,
    lat_deg: float,
    lon_deg: float,
    method: str = "nearest",
    tolerance_deg: float | None = 0.5,
) -> xr.DataArray:
    """Sample a CRU TS DataArray at a borehole site.

    Parameters
    ----------
    da
        DataArray with ``lat`` and ``lon`` coordinates.
    lat_deg, lon_deg
        Site coordinates in decimal degrees.
    method
        Selection method passed to ``xarray.DataArray.sel``.  Default
        ``"nearest"``.
    tolerance_deg
        Tolerance (deg) on the nearest match.  Pass None to disable.

    Returns
    -------
    xarray.DataArray
        1-D series along the time axis.
    """
    lon_name = _find_lon_name(da)
    lat_name = _find_lat_name(da)
    sel_kwargs: dict[str, Any] = {lat_name: lat_deg, lon_name: lon_deg, "method": method}
    if tolerance_deg is not None:
        sel_kwargs["tolerance"] = tolerance_deg
    return da.sel(**sel_kwargs)


def monthly_anomaly(da: xr.DataArray, *, baseline: slice | None = None) -> xr.DataArray:
    """Compute month-of-year anomaly relative to a baseline window.

    Parameters
    ----------
    da
        DataArray indexed by a time dimension.
    baseline
        ``slice("YYYY", "YYYY")``; default is the entire record.
    """
    base = da if baseline is None else da.sel(time=baseline)
    clim = base.groupby("time.month").mean("time")
    return da.groupby("time.month") - clim


def _find_lat_name(da: xr.DataArray) -> str:
    for name in ("lat", "latitude", "y"):
        if name in da.coords:
            return name
    raise KeyError("No latitude coordinate found (expected one of: lat, latitude, y).")


def _find_lon_name(da: xr.DataArray) -> str:
    for name in ("lon", "longitude", "x"):
        if name in da.coords:
            return name
    raise KeyError("No longitude coordinate found (expected one of: lon, longitude, x).")


def normalise_longitudes(da: xr.DataArray, *, to_signed: bool = True) -> xr.DataArray:
    """Reorder a global longitude axis from 0-360 to -180-+180 (or back).

    Useful when a CRU file stores longitudes 0..360 but boreholes are
    catalogued in -180..+180.
    """
    lon_name = _find_lon_name(da)
    lons = da[lon_name].values
    if to_signed:
        new = np.where(lons > 180.0, lons - 360.0, lons)
    else:
        new = np.where(lons < 0.0, lons + 360.0, lons)
    order = np.argsort(new)
    return da.assign_coords({lon_name: new}).isel({lon_name: order})
