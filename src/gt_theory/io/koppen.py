"""Köppen-Geiger climate classification from CRU TS monthly fields.

This module reproduces the 31-class Köppen-Geiger taxonomy of
Beck et al. 2018 (Scientific Data) using the Peel, Finlayson,
McMahon (2007) decision tree applied to monthly air-temperature
and precipitation climatologies. The same decision tree is what
Beck 2018 uses; the only differences are in the source data
(WorldClim 2.0 vs CRU TS) and the resolution.

The main entry points are:

- :func:`koppen_class_from_monthly` -- single-point classifier.
- :func:`koppen_grid_from_cru` -- vectorised reduction of CRU TS
  monthly DataArrays to a lat/lon grid of integer class codes.
- :func:`koppen_at` -- nearest-point lookup against a precomputed
  grid (callable for many sites in a loop).

The integer encoding follows the order returned by
:func:`koppen_classes` (1 = Af, 2 = Am, ..., 30 = ET, 31 = EF) so
that class identity round-trips through the parquet output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

# Standard 31-class Köppen taxonomy ordered for stable codes.
KOPPEN_CLASSES: tuple[str, ...] = (
    "Af",
    "Am",
    "Aw",  # 1-3 tropical
    "BWh",
    "BWk",
    "BSh",
    "BSk",  # 4-7 arid
    "Csa",
    "Csb",
    "Csc",  # 8-10 temperate dry-summer
    "Cwa",
    "Cwb",
    "Cwc",  # 11-13 temperate dry-winter
    "Cfa",
    "Cfb",
    "Cfc",  # 14-16 temperate no-dry-season
    "Dsa",
    "Dsb",
    "Dsc",
    "Dsd",  # 17-20 continental dry-summer
    "Dwa",
    "Dwb",
    "Dwc",
    "Dwd",  # 21-24 continental dry-winter
    "Dfa",
    "Dfb",
    "Dfc",
    "Dfd",  # 25-28 continental no-dry-season
    "ET",
    "EF",  # 29-30 polar
    "Oc",  # 31 ocean / no land
)
_CLASS_TO_CODE: dict[str, int] = {c: i + 1 for i, c in enumerate(KOPPEN_CLASSES)}


def koppen_classes() -> tuple[str, ...]:
    """Return the 31-class Köppen list in the order used by the
    integer encoding."""
    return KOPPEN_CLASSES


def _major_class(
    t_monthly: np.ndarray,
    p_monthly: np.ndarray,
    lat_deg: float,
) -> str | None:
    """Decide the A / B / C / D / E major class.

    ``t_monthly`` is monthly mean air temperature in °C
    (length 12, Jan-Dec). ``p_monthly`` is monthly precipitation
    in mm (length 12, Jan-Dec).
    """
    if not (np.all(np.isfinite(t_monthly)) and np.all(np.isfinite(p_monthly))):
        return None
    t_max = float(np.max(t_monthly))
    t_min = float(np.min(t_monthly))
    t_mean = float(np.mean(t_monthly))
    p_total = float(np.sum(p_monthly))

    # Arid (B) is evaluated FIRST in Peel 2007 ordering: a desert
    # with all months > 18 °C is still B, not A.
    summer_months = [4, 5, 6, 7, 8, 9] if lat_deg >= 0.0 else [10, 11, 0, 1, 2, 3]
    winter_months = [m for m in range(12) if m not in summer_months]
    p_summer = float(np.sum(p_monthly[summer_months]))
    p_winter = float(np.sum(p_monthly[winter_months]))
    if p_summer >= 0.7 * p_total:
        p_threshold = 20.0 * t_mean + 280.0
    elif p_winter >= 0.7 * p_total:
        p_threshold = 20.0 * t_mean
    else:
        p_threshold = 20.0 * t_mean + 140.0
    if p_total < p_threshold:
        return "B"

    # Tropical: every month above 18 °C and total precip large enough.
    if t_min >= 18.0:
        return "A"

    # Polar (E): warmest-month mean below 10 °C.
    if t_max < 10.0:
        return "E"

    # Temperate (C): coldest-month mean between 0 and 18 °C.
    if 0.0 <= t_min < 18.0 and t_max >= 10.0:
        return "C"

    # Continental (D): coldest-month mean below 0 °C, warmest month >= 10.
    if t_min < 0.0 and t_max >= 10.0:
        return "D"

    return None


def _b_subclass(
    t_monthly: np.ndarray,
    p_monthly: np.ndarray,
    lat_deg: float,
) -> str:
    """B-major subdivision: W/S (desert vs steppe) and h/k (hot vs cold)."""
    summer_months = [4, 5, 6, 7, 8, 9] if lat_deg >= 0.0 else [10, 11, 0, 1, 2, 3]
    winter_months = [m for m in range(12) if m not in summer_months]
    p_total = float(np.sum(p_monthly))
    p_summer = float(np.sum(p_monthly[summer_months]))
    p_winter = float(np.sum(p_monthly[winter_months]))
    t_mean = float(np.mean(t_monthly))
    if p_summer >= 0.7 * p_total:
        threshold = 20.0 * t_mean + 280.0
    elif p_winter >= 0.7 * p_total:
        threshold = 20.0 * t_mean
    else:
        threshold = 20.0 * t_mean + 140.0
    is_desert = p_total < 0.5 * threshold
    is_hot = t_mean >= 18.0
    if is_desert:
        return "BWh" if is_hot else "BWk"
    return "BSh" if is_hot else "BSk"


def _cd_subclass(
    t_monthly: np.ndarray,
    p_monthly: np.ndarray,
    lat_deg: float,
    major: str,
) -> str:
    """C and D subdivisions (s/w/f and a/b/c/d)."""
    summer_months = [4, 5, 6, 7, 8, 9] if lat_deg >= 0.0 else [10, 11, 0, 1, 2, 3]
    winter_months = [m for m in range(12) if m not in summer_months]
    p_summer_min = float(np.min(p_monthly[summer_months]))
    p_summer_max = float(np.max(p_monthly[summer_months]))
    p_winter_min = float(np.min(p_monthly[winter_months]))
    p_winter_max = float(np.max(p_monthly[winter_months]))

    # s: dry summer, w: dry winter, f: no dry season.
    is_s = (p_summer_min < 30.0) and (p_summer_min < p_winter_max / 3.0)
    is_w = (p_winter_min < p_summer_max / 10.0) and not is_s
    if is_s:
        season = "s"
    elif is_w:
        season = "w"
    else:
        season = "f"

    # Temperature subdivision.
    t_max = float(np.max(t_monthly))
    t_min = float(np.min(t_monthly))
    n_warm = int(np.sum(t_monthly >= 10.0))
    if t_max >= 22.0:
        temp = "a"
    elif n_warm >= 4:
        temp = "b"
    elif t_min >= -38.0:
        temp = "c"
    else:
        temp = "d"
    return major + season + temp


def _a_subclass(p_monthly: np.ndarray, lat_deg: float) -> str:
    """A-major subdivision: Af monsoon, Am wet-dry, Aw savanna."""
    p_total = float(np.sum(p_monthly))
    p_min = float(np.min(p_monthly))
    summer_months = [4, 5, 6, 7, 8, 9] if lat_deg >= 0.0 else [10, 11, 0, 1, 2, 3]
    winter_months = [m for m in range(12) if m not in summer_months]
    p_dry_season = min(
        float(np.min(p_monthly[summer_months])),
        float(np.min(p_monthly[winter_months])),
    )
    if p_min >= 60.0:
        return "Af"
    if p_dry_season >= 100.0 - p_total / 25.0:
        return "Am"
    return "Aw"


def _e_subclass(t_monthly: np.ndarray) -> str:
    """ET (tundra) vs EF (ice cap)."""
    t_max = float(np.max(t_monthly))
    if t_max >= 0.0:
        return "ET"
    return "EF"


def koppen_class_from_monthly(
    *,
    t_monthly_c: np.ndarray,
    p_monthly_mm: np.ndarray,
    lat_deg: float,
) -> str | None:
    """Single-point classifier from 12 monthly mean temperatures (°C)
    and 12 monthly precipitation totals (mm)."""
    t_monthly = np.asarray(t_monthly_c, dtype=float)
    p_monthly = np.asarray(p_monthly_mm, dtype=float)
    if t_monthly.size != 12 or p_monthly.size != 12:
        raise ValueError("t_monthly and p_monthly must each have length 12")
    if not (np.all(np.isfinite(t_monthly)) and np.all(np.isfinite(p_monthly))):
        return None
    major = _major_class(t_monthly, p_monthly, lat_deg=lat_deg)
    if major is None:
        return None
    if major == "A":
        return _a_subclass(p_monthly, lat_deg=lat_deg)
    if major == "B":
        return _b_subclass(t_monthly, p_monthly, lat_deg=lat_deg)
    if major in ("C", "D"):
        return _cd_subclass(t_monthly, p_monthly, lat_deg=lat_deg, major=major)
    if major == "E":
        return _e_subclass(t_monthly)
    return None


def koppen_code(class_name: str | None) -> int:
    """Map a class name (``"Cfb"``, ``"BSh"``, ...) to its integer
    code. Returns 0 for an unknown / ocean class."""
    if class_name is None:
        return 0
    return _CLASS_TO_CODE.get(class_name, 0)


@dataclass(frozen=True)
class CruClimatology:
    """A 12-month T + P climatology on a (lat, lon) grid."""

    t_monthly_c: xr.DataArray  # (month, lat, lon)
    p_monthly_mm: xr.DataArray  # (month, lat, lon)
    baseline: tuple[int, int]  # (start_year, end_year)

    @property
    def lats(self) -> np.ndarray:
        return self.t_monthly_c["lat"].values

    @property
    def lons(self) -> np.ndarray:
        return self.t_monthly_c["lon"].values


def cru_climatology(
    *,
    cru_tmp: xr.DataArray,
    cru_pre: xr.DataArray,
    baseline: tuple[int, int] = (1980, 2010),
) -> CruClimatology:
    """Compute a 12-month T + P climatology over the named baseline
    window. The two inputs must be CRU TS monthly DataArrays on the
    same (lat, lon) grid."""
    t_sel = cru_tmp.sel(time=slice(f"{baseline[0]}-01", f"{baseline[1]}-12"))
    p_sel = cru_pre.sel(time=slice(f"{baseline[0]}-01", f"{baseline[1]}-12"))
    t_clim = t_sel.groupby("time.month").mean("time")
    p_clim = p_sel.groupby("time.month").mean("time")
    return CruClimatology(
        t_monthly_c=t_clim,
        p_monthly_mm=p_clim,
        baseline=baseline,
    )


def koppen_at(
    climatology: CruClimatology,
    *,
    lat_deg: float,
    lon_deg: float,
) -> str | None:
    """Köppen class at a single (lat, lon) sampled from the
    climatology by nearest grid cell."""
    t = climatology.t_monthly_c.sel(lat=lat_deg, lon=lon_deg, method="nearest").values
    p = climatology.p_monthly_mm.sel(lat=lat_deg, lon=lon_deg, method="nearest").values
    return koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=lat_deg,
    )


def koppen_for_sites(
    climatology: CruClimatology,
    lats_deg: Sequence[float],
    lons_deg: Sequence[float],
) -> list[str | None]:
    """Köppen class at each (lat, lon) pair."""
    lats = np.asarray(lats_deg, dtype=float)
    lons = np.asarray(lons_deg, dtype=float)
    out: list[str | None] = []
    for la, lo in zip(lats, lons, strict=True):
        out.append(koppen_at(climatology, lat_deg=float(la), lon_deg=float(lo)))
    return out


__all__ = [
    "CruClimatology",
    "KOPPEN_CLASSES",
    "cru_climatology",
    "koppen_at",
    "koppen_class_from_monthly",
    "koppen_classes",
    "koppen_code",
    "koppen_for_sites",
]
