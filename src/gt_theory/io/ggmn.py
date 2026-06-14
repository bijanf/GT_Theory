"""Reader for the Global Groundwater Monitoring Network (GGMN)
annual-aggregate CSVs.

The cached files at
``~/Documents/MR_gwasser_cluster_cache/public_ggmn/`` ship two
products:

- ``AnnualDepthToGroundwater.csv`` --- per-station annual mean
  depth-to-water (metres below ground), columns
  ``StnID, Lat, Lon, IntegerYear, DepthToWater_m``.
- ``ggmn_wells.csv`` --- per-station metadata (aquifer name,
  surface elevation, country, etc.).

The GGMN is a UNESCO-IGRAC product (CC-BY 4.0). It is the closest
open-data substitute for the Benz et al. 2024 global groundwater
dataset called for in the R17 editorial verdict; GGMN coverage is
strongest in Europe + parts of North America and sparse in the
tropics and high latitudes, so a per-site nearest-station lookup
must accept potentially large distances.

Public API:

- :func:`load_ggmn` -- read the annual time-series for all
  stations into an :class:`xarray.Dataset`.
- :func:`load_ggmn_wells` -- read the metadata table into a
  :class:`pandas.DataFrame`.
- :func:`nearest_station` -- per-(lat, lon) lookup with great-circle
  distance.
- :func:`head_trend_per_station` -- per-station :math:`dh/dt` in
  metres per year from the annual time-series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REQUIRED_LEVEL_COLS: tuple[str, ...] = (
    "StnID",
    "Lat",
    "Lon",
    "IntegerYear",
    "DepthToWater_m",
)


def load_ggmn(csv_path: str | Path) -> xr.Dataset:
    """Load the GGMN annual depth-to-water CSV into an xarray Dataset
    indexed by (station, year).

    Parameters
    ----------
    csv_path
        Path to ``AnnualDepthToGroundwater.csv``.

    Returns
    -------
    xarray.Dataset
        Variables ``depth_to_water_m(station, year)``,
        ``lat(station)``, ``lon(station)``.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    ValueError
        If any required column is missing.
    """
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"GGMN CSV not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_LEVEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"GGMN CSV missing required columns: {missing}; got {list(df.columns)}")
    # Aggregate to one row per (station, year). The cached file
    # already contains an integer year; we group to be robust against
    # duplicate rows (occasional in the upstream CSV).
    agg = df.groupby(["StnID", "IntegerYear"], as_index=False).agg(
        DepthToWater_m=("DepthToWater_m", "mean"),
        Lat=("Lat", "first"),
        Lon=("Lon", "first"),
    )
    pivot = agg.pivot(  # noqa: PD010  (rows pre-aggregated; a strict reshape, no aggregation)
        index="StnID", columns="IntegerYear", values="DepthToWater_m"
    ).sort_index()
    coords_lat = agg.groupby("StnID")["Lat"].first().reindex(pivot.index)
    coords_lon = agg.groupby("StnID")["Lon"].first().reindex(pivot.index)
    return xr.Dataset(
        data_vars={
            "depth_to_water_m": (
                ("station", "year"),
                pivot.to_numpy(dtype=float),
            ),
            "lat": (("station",), coords_lat.to_numpy(dtype=float)),
            "lon": (("station",), coords_lon.to_numpy(dtype=float)),
        },
        coords={
            "station": pivot.index.to_numpy(dtype=int),
            "year": pivot.columns.to_numpy(dtype=int),
        },
    )


def load_ggmn_wells(csv_path: str | Path) -> pd.DataFrame:
    """Read the GGMN well-metadata table.

    The ``ground_surface_elevation`` and ``aquifer_name`` columns
    are the ones used downstream by
    :mod:`gt_theory.io.groundwater_v_darcy`.
    """
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"GGMN wells CSV not found: {p}")
    return pd.read_csv(p, low_memory=False)


@dataclass(frozen=True)
class NearestStation:
    station_id: int
    distance_km: float
    lat_deg: float
    lon_deg: float


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Great-circle distance in kilometres; broadcasts the first
    argument over the second."""
    r_earth_km = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r_earth_km * np.arcsin(np.sqrt(a))


def nearest_station(
    ds: xr.Dataset,
    *,
    lat_deg: float,
    lon_deg: float,
    max_distance_km: float | None = None,
) -> NearestStation | None:
    """Return the GGMN station nearest to a given (lat, lon).

    Returns None if ``max_distance_km`` is set and no station is
    within range.
    """
    lats = ds["lat"].to_numpy()
    lons = ds["lon"].to_numpy()
    finite = np.isfinite(lats) & np.isfinite(lons)
    if not finite.any():
        return None
    distances = _haversine_km(lats[finite], lons[finite], lat_deg, lon_deg)
    idx_local = int(np.argmin(distances))
    idx_global = int(np.flatnonzero(finite)[idx_local])
    dist_km = float(distances[idx_local])
    if max_distance_km is not None and dist_km > max_distance_km:
        return None
    return NearestStation(
        station_id=int(ds["station"].to_numpy()[idx_global]),
        distance_km=dist_km,
        lat_deg=float(lats[idx_global]),
        lon_deg=float(lons[idx_global]),
    )


def head_trend_per_station(
    ds: xr.Dataset,
    *,
    min_years: int = 5,
) -> xr.DataArray:
    """Linear :math:`dh/dt` (in metres per year) at each station.

    Sign convention: positive values mean the water table is
    *rising* (i.e. depth-to-water is decreasing); negative values
    mean the water table is falling. The conversion comes from
    ``h = elevation - depth_to_water``, so
    :math:`dh/dt = -d(\\text{depth})/dt`.

    Stations with fewer than ``min_years`` finite observations are
    returned as NaN.
    """
    depth = ds["depth_to_water_m"].to_numpy()  # (station, year)
    years = ds["year"].to_numpy()
    n_st = depth.shape[0]
    slope = np.full(n_st, np.nan)
    for i in range(n_st):
        d = depth[i, :]
        m = np.isfinite(d)
        if int(m.sum()) < min_years:
            continue
        yy = years[m].astype(float)
        dd = d[m].astype(float)
        # Least-squares slope of depth-vs-year; flip sign so
        # output is dh/dt (positive = rising water table).
        slope_depth = float(np.polyfit(yy, dd, 1)[0])
        slope[i] = -slope_depth
    return xr.DataArray(
        slope,
        coords={"station": ds["station"].to_numpy()},
        dims=("station",),
        name="head_trend_m_per_yr",
    )


__all__ = [
    "NearestStation",
    "head_trend_per_station",
    "load_ggmn",
    "load_ggmn_wells",
    "nearest_station",
]
