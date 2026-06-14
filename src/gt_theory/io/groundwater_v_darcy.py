"""Per-site vertical Darcy velocity proxy from GGMN annual head
trends.

The R17 editorial verdict requires the forward Crank-Nicolson
solver to actually exercise its ``v_darcy`` parameter rather than
default it to zero across all F1-F7 fingerprint sweeps. The
closest open-data substitute for the Benz et al. 2024 dataset is
the GGMN annual head-trend record (`gt_theory.io.ggmn`). This
module converts the per-station :math:`dh/dt` into a
column-mean :math:`v_{\\mathrm{Darcy}}` proxy via Darcy's law:

.. math::
   v_{\\mathrm{Darcy}}(\\mathrm{lat}, \\mathrm{lon})
     = K_{\\mathrm{lith}} \\cdot \\frac{dh}{dt}
       \\cdot \\frac{\\Delta t_{\\mathrm{yr}}}{L_z},

where :math:`K_{\\mathrm{lith}}` is the lithology-class hydraulic
conductivity, :math:`dh/dt` is the GGMN head trend in m/yr at the
nearest station, :math:`\\Delta t_{\\mathrm{yr}}` is one year in
seconds, and :math:`L_z` is the characteristic vertical scale of
the borehole (typically the borehole depth itself).

The proxy is deliberately first-order. It is meant to give the
F2 spatial-variability recompute (`scripts/recompute_f2_with_advection.py`)
a non-zero, geographically-varying ``v_darcy`` value at each
Huang-Pollack site so the conduction-only floor is no longer the
only operating point of the framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from gt_theory.io.ggmn import head_trend_per_station, nearest_station

# Lithology-class hydraulic-conductivity defaults (m/s). Values are
# logarithmic-mean estimates from the Bear & Bachmat 1990 monograph;
# the within-class range is at least 2 orders of magnitude either
# way, so these are deliberately rough.
K_HYD_LITH_MS: dict[str, float] = {
    "sandstone": 1.0e-5,
    "limestone": 1.0e-6,
    "shale": 1.0e-9,
    "granite": 1.0e-10,
    "alluvium": 1.0e-4,
    "till": 1.0e-7,
    "unknown": 1.0e-6,  # fallback (used most of the time)
}

SECONDS_PER_YEAR: float = 365.25 * 86400.0


@dataclass(frozen=True)
class VDarcyEstimate:
    """A per-site Darcy-velocity estimate with provenance."""

    v_darcy_m_per_s: float
    nearest_station_id: int | None
    nearest_distance_km: float
    head_trend_m_per_yr: float
    lithology: str
    K_hyd_m_per_s: float


def site_v_darcy(
    ggmn_ds: xr.Dataset,
    *,
    lat_deg: float,
    lon_deg: float,
    depth_m: float = 300.0,
    lithology: str = "unknown",
    max_distance_km: float = 1500.0,
    min_years: int = 5,
    precomputed_trend: xr.DataArray | None = None,
) -> VDarcyEstimate:
    """Estimate :math:`v_{\\mathrm{Darcy}}` at a given borehole
    location from the nearest GGMN station's annual head trend.

    Parameters
    ----------
    ggmn_ds
        Output of :func:`gt_theory.io.ggmn.load_ggmn`.
    lat_deg, lon_deg
        Borehole location (decimal degrees).
    depth_m
        Characteristic vertical scale :math:`L_z` over which the
        Darcy velocity is averaged. Defaults to 300 m, the median
        Huang-Pollack borehole length.
    lithology
        Lithology class key from :data:`K_HYD_LITH_MS`. Defaults to
        ``"unknown"`` (uses the catalogue fallback).
    max_distance_km
        If the nearest GGMN station is farther than this, fall back
        to ``v_darcy = 0`` and record the no-station status.
    min_years
        Minimum number of finite GGMN observations required at the
        nearest station before its head trend is used.

    Returns
    -------
    VDarcyEstimate
        Per-site velocity estimate with provenance metadata.
    """
    K_hyd = K_HYD_LITH_MS.get(lithology, K_HYD_LITH_MS["unknown"])
    near = nearest_station(
        ggmn_ds,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        max_distance_km=max_distance_km,
    )
    if near is None:
        return VDarcyEstimate(
            v_darcy_m_per_s=0.0,
            nearest_station_id=None,
            nearest_distance_km=float("nan"),
            head_trend_m_per_yr=float("nan"),
            lithology=lithology,
            K_hyd_m_per_s=K_hyd,
        )
    trend_da = (
        precomputed_trend
        if precomputed_trend is not None
        else head_trend_per_station(ggmn_ds, min_years=min_years)
    )
    try:
        dh_dt = float(trend_da.sel(station=near.station_id).item())
    except KeyError:
        dh_dt = float("nan")
    if not np.isfinite(dh_dt):
        return VDarcyEstimate(
            v_darcy_m_per_s=0.0,
            nearest_station_id=near.station_id,
            nearest_distance_km=near.distance_km,
            head_trend_m_per_yr=float("nan"),
            lithology=lithology,
            K_hyd_m_per_s=K_hyd,
        )
    # v_darcy [m/s] = K_hyd [m/s] * (dh/dt [m/yr]) * (1 yr / sec)
    #                                 -------- normalised by L_z
    # The depth normalisation gives a column-mean estimate.
    v_darcy = K_hyd * (dh_dt / SECONDS_PER_YEAR) / max(depth_m, 1.0e-6)
    return VDarcyEstimate(
        v_darcy_m_per_s=float(v_darcy),
        nearest_station_id=near.station_id,
        nearest_distance_km=near.distance_km,
        head_trend_m_per_yr=dh_dt,
        lithology=lithology,
        K_hyd_m_per_s=K_hyd,
    )


__all__ = ["K_HYD_LITH_MS", "VDarcyEstimate", "site_v_darcy"]
