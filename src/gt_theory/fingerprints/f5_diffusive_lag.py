"""F5 -- Thermal diffusive lag at depth.

The theory paper's F5 fingerprint predicts that the
ground-temperature response at depth :math:`z` lags the
surface-air-temperature signal by
:math:`\\tau_{\\mathrm{lag}} = z^2 / (4 \\kappa)`. At
:math:`z = 5`\\,m, :math:`\\kappa = 10^{-6}`\\,m\\,s$^{-1}$ this is
:math:`\\sim 2.4`\\,months; at :math:`z = 20`\\,m, about
:math:`38`\\,months.

The previous-revision quoted observation of :math:`117` months at
20\\,m was a raw cross-correlation of monthly SAT against monthly
GST that the secular warming trend dominates. This module computes
the F5 lag on monthly series after Butterworth-bandpass filtering
to the decadal band where the diffusive prediction is valid.

Public API:

- :class:`F5Result` -- dataclass with per-site lag at 5\\,m + 20\\,m,
  unfiltered + filtered.
- :func:`compute_f5_at_site` -- single-site driver from a CRU TS
  monthly SAT series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gt_theory.solvers.column_1d import run_column_1d
from gt_theory.stats.signal import bandpass_lag_months, cross_correlation_lag

YEAR_S = 365.25 * 86400.0
MEAN_MONTH_S = 30.44 * 86400.0


@dataclass(frozen=True)
class F5Result:
    """Per-site F5 diffusive-lag estimates."""

    site_id: str
    lat_deg: float
    lon_deg: float
    kappa_m2_per_s: float
    lag_5m_unfiltered_months: int
    lag_5m_filtered_months: int
    lag_20m_unfiltered_months: int
    lag_20m_filtered_months: int


def _simulate_gst_monthly(
    *,
    sat_monthly_K: np.ndarray,
    kappa: float,
    depth_max_m: float = 50.0,
    dz_m: float = 1.0,
    probe_depths_m: tuple[float, ...] = (5.0, 20.0),
) -> dict[float, np.ndarray]:
    """Forward-simulate monthly ground temperature at the requested
    depths under the existing 1-D heat solver.

    Returns a dict keyed by ``depth_m`` with the monthly GST anomaly
    series at that depth.
    """
    n_months = sat_monthly_K.size
    dt_s = MEAN_MONTH_S
    # column_1d wants nt = duration/dt + 1 surface values; pad by
    # repeating the first SAT value as the initial-condition point.
    duration_s = (n_months - 1) * dt_s
    sat_padded = np.concatenate([[sat_monthly_K[0]], sat_monthly_K[1:]])
    res = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa,
        sat=sat_padded,
    )
    # res.T has shape (nt, nz); pick the nearest cell to each probe.
    out: dict[float, np.ndarray] = {}
    for z_target in probe_depths_m:
        iz = int(np.argmin(np.abs(res.z - z_target)))
        gst_series = res.T[:n_months, iz] - res.T[0, iz]
        out[z_target] = gst_series
    return out


def compute_f5_at_site(
    *,
    site_id: str,
    lat_deg: float,
    lon_deg: float,
    sat_monthly_K: np.ndarray,
    kappa_m2_per_s: float = 1.0e-6,
    low_yr: float = 30.0,
    high_yr: float = 3.0,
    max_lag_months: int = 240,
) -> F5Result:
    """Compute F5 lag at 5\\,m and 20\\,m for a single site.

    Parameters
    ----------
    site_id, lat_deg, lon_deg
        Site identifiers (propagated to the result).
    sat_monthly_K
        Monthly SAT *anomaly* series (K) at the site; should span
        $\\gtrsim 30$ years for stable cross-correlation.
    kappa_m2_per_s
        Bulk thermal diffusivity used in the forward simulation.
        Default ``1e-6`` matches the global ensemble's median.
    low_yr, high_yr
        Bandpass endpoints in years (defaults: 30-yr / 3-yr passband).
    max_lag_months
        Maximum absolute lag considered.
    """
    if sat_monthly_K.size < 360:  # 30 yr
        raise ValueError(
            f"sat_monthly_K too short ({sat_monthly_K.size} months); need >= 360 (30 yr)"
        )
    gst = _simulate_gst_monthly(
        sat_monthly_K=sat_monthly_K,
        kappa=kappa_m2_per_s,
        probe_depths_m=(5.0, 20.0),
    )
    sat_anom = sat_monthly_K - float(np.nanmean(sat_monthly_K))

    # Unfiltered cross-correlation.
    lag_5m_u, _ = cross_correlation_lag(
        sat_anom,
        gst[5.0],
        max_lag=max_lag_months,
    )
    lag_20m_u, _ = cross_correlation_lag(
        sat_anom,
        gst[20.0],
        max_lag=max_lag_months,
    )
    # Bandpass-filtered.
    lag_5m_f = bandpass_lag_months(
        sat_anom,
        gst[5.0],
        low_yr=low_yr,
        high_yr=high_yr,
        max_lag_months=max_lag_months,
    )
    lag_20m_f = bandpass_lag_months(
        sat_anom,
        gst[20.0],
        low_yr=low_yr,
        high_yr=high_yr,
        max_lag_months=max_lag_months,
    )

    return F5Result(
        site_id=site_id,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        kappa_m2_per_s=kappa_m2_per_s,
        lag_5m_unfiltered_months=int(lag_5m_u),
        lag_5m_filtered_months=int(lag_5m_f),
        lag_20m_unfiltered_months=int(lag_20m_u),
        lag_20m_filtered_months=int(lag_20m_f),
    )


__all__ = ["F5Result", "compute_f5_at_site"]
