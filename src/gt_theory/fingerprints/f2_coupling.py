"""F2 — SAT-GST coupling.

For each borehole site, compare the recovered recent-bin GST anomaly
against the surface-air-temperature anomaly observed at the same
location from CRU TS v4.  The coupling is summarised as:

* per-site eta = Delta GST / Delta SAT,
* an OLS slope and Deming (lambda = 4) slope of Delta GST on Delta SAT
  across the population of sites,
* bootstrap 5-95 CI on both slopes.

The Deming variance ratio lambda = 4 is the value singled out by
the accompanying paper (Section 3, F2).  It models the situation
where the GST uncertainty (sigma_GST) is twice the SAT uncertainty
(sigma_SAT): lambda = sigma_SAT^2 / sigma_GST^2 = 1/4 — but the
accompanying paper follows the
common geosciences convention where lambda denotes the inverse ratio
(Beltrami et al. 2006), so we adopt lambda = 4.

Reference windows
-----------------
recent  : 2000-01 -- end of CRU record (typically 2024-12)
baseline: 1901-01 -- 1960-12

These match the integration window of F4 (1960-2018) reasonably well
and bracket the bulk of the modern warming signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class F2Result:
    site_ids: list[str]
    delta_sat_K: np.ndarray
    delta_gst_K: np.ndarray
    eta_per_site: np.ndarray
    ols_slope: float
    ols_ci: tuple[float, float]
    deming_slope: float
    deming_ci: tuple[float, float]


def _window_mean(sat_series: pd.DataFrame, start_year: int, end_year: int) -> float:
    times = pd.to_datetime(sat_series["time"]).dt.year
    mask = (times >= start_year) & (times <= end_year)
    sub = sat_series.loc[mask, "sat_c"]
    return float(np.nanmean(sub)) if not sub.empty else float("nan")


def _deming_slope(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Deming (orthogonal-error) regression slope.

    Parameters
    ----------
    x, y
        Same-length 1-D arrays.
    lam
        Variance ratio sigma_x^2 / sigma_y^2.  Per the convention in the accompanying paper.
    """
    n = x.size
    if n < 3:
        return float("nan")
    x_bar = x.mean()
    y_bar = y.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    syy = float(np.sum((y - y_bar) ** 2))
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    discriminant = (syy - lam * sxx) ** 2 + 4.0 * lam * sxy * sxy
    if discriminant < 0 or sxy == 0:
        return float("nan")
    return float(((syy - lam * sxx) + np.sqrt(discriminant)) / (2.0 * sxy))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_bar = x.mean()
    y_bar = y.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx == 0:
        return float("nan")
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    return sxy / sxx


def compute_f2(
    *,
    inversions: list[pd.DataFrame],
    site_ids: list[str],
    cru_sat: pd.DataFrame,
    baseline_window: tuple[int, int] = (1901, 1960),
    recent_window: tuple[int, int] = (2000, 2024),
    lam_deming: float = 4.0,
    n_bootstrap: int = 500,
    seed: int = 20260522,
) -> F2Result:
    """Compute F2 from per-site inversions + a tidy CRU SAT table.

    Parameters
    ----------
    inversions
        One DataFrame per site, output of ``scripts/invert_profile.py``.
    site_ids
        IDs in the same order as ``inversions``.  Used to filter ``cru_sat``.
    cru_sat
        Long-form DataFrame with columns ``site_id``, ``time``, ``sat_c``.
    baseline_window, recent_window
        Year ranges (inclusive) used to compute Delta SAT.
    lam_deming
        Deming variance ratio (see module docstring).
    n_bootstrap
        Site-resample bootstrap draws.
    seed
        RNG seed.

    Returns
    -------
    F2Result
    """
    if len(inversions) != len(site_ids):
        raise ValueError("inversions and site_ids length mismatch")
    if not {"site_id", "time", "sat_c"}.issubset(cru_sat.columns):
        raise ValueError("cru_sat must have columns: site_id, time, sat_c")

    delta_sat = np.full(len(site_ids), np.nan)
    delta_gst = np.full(len(site_ids), np.nan)
    for i, sid in enumerate(site_ids):
        sat = cru_sat[cru_sat["site_id"] == sid]
        if sat.empty:
            continue
        baseline = _window_mean(sat, *baseline_window)
        recent = _window_mean(sat, *recent_window)
        delta_sat[i] = recent - baseline
        # Use the inversion's recent-bin median as Delta GST.
        delta_gst[i] = float(inversions[i].iloc[0]["median_K"])

    mask = np.isfinite(delta_sat) & np.isfinite(delta_gst)
    x = delta_sat[mask]
    y = delta_gst[mask]
    eta = np.where(np.isfinite(delta_sat / delta_gst), delta_gst / delta_sat, np.nan)

    ols = _ols_slope(x, y)
    dem = _deming_slope(x, y, lam_deming)

    rng = np.random.default_rng(seed)
    n = x.size
    boot_ols = np.empty(n_bootstrap)
    boot_dem = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_ols[b] = _ols_slope(x[idx], y[idx])
        boot_dem[b] = _deming_slope(x[idx], y[idx], lam_deming)
    ols_ci = (float(np.nanpercentile(boot_ols, 5.0)), float(np.nanpercentile(boot_ols, 95.0)))
    dem_ci = (float(np.nanpercentile(boot_dem, 5.0)), float(np.nanpercentile(boot_dem, 95.0)))

    return F2Result(
        site_ids=list(site_ids),
        delta_sat_K=delta_sat,
        delta_gst_K=delta_gst,
        eta_per_site=eta,
        ols_slope=ols,
        ols_ci=ols_ci,
        deming_slope=dem,
        deming_ci=dem_ci,
    )
