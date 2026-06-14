"""Zhang 2005 winter n-factor scenarios for the F3 boreal/equator
ratio diagnostic.

The Zhang (2005, *Rev. Geophys.*) winter n-factor

    n_w(t) = T_GST^winter(t) / T_SAT^winter(t)

quantifies the seasonal-mean decoupling between
ground-surface temperature and surface-air temperature in regions
with persistent snow cover. Heavy snow insulates the ground from
extreme winter cold (low n_w); snow loss progressively re-couples
ground and air (n_w -> 1). The same effect can equivalently be
applied to *anomalies*: a winter n-factor below 1 damps the
amplitude of the surface signal that reaches the subsurface.

The R17 W2 question is whether any plausible $n_w(t)$ trajectory
recovers the boreal/equator subsurface warming ratio of 1.49
(predicted by the surface-amplification signal) from the observed
0.94. We define three scenarios:

- ``constant_heavy``: $n_w = 0.5$ at all times (saturated snow
  insulation, no trend);
- ``declining_snow_insulation``: $n_w$ ramps linearly from 0.6 in
  1900 to 0.9 in 2024 (the snow-loss-driven decoupling collapse
  the framework's prose retracts to);
- ``no_insulation``: $n_w \\equiv 1$ at all times (the
  zero-snow limit, low-latitude / vegetation-controlled sites).

The functions below build the per-month n-factor coefficient given
a year vector and apply it to a winter mask. The convention is
that the n-factor multiplies the *anomaly* relative to the
long-term mean -- not the absolute temperature -- so the
operationalisation is

    SAT_modulated(t) = SAT_climatology(t) + n_w_eff(t) * SAT_anomaly(t)

with ``n_w_eff(t) = n_w(t)`` in winter and 1 otherwise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SCENARIOS = ("constant_heavy", "declining_snow_insulation", "no_insulation")
ScenarioName = Literal["constant_heavy", "declining_snow_insulation", "no_insulation"]


def nfactor_series(
    *,
    scenario: ScenarioName,
    years: np.ndarray,
) -> np.ndarray:
    """Return the winter n-factor :math:`n_w(t)` evaluated at each
    year in ``years``.

    Parameters
    ----------
    scenario
        One of ``"constant_heavy"``, ``"declining_snow_insulation"``,
        ``"no_insulation"``.
    years
        Calendar-year array (any length, monotonically increasing
        or unordered).
    """
    yr = np.asarray(years, dtype=float)
    if scenario == "constant_heavy":
        return np.full_like(yr, 0.5)
    if scenario == "no_insulation":
        return np.full_like(yr, 1.0)
    if scenario == "declining_snow_insulation":
        # Linear ramp from 0.6 in 1900 to 0.9 in 2024; clipped outside.
        slope = (0.9 - 0.6) / (2024.0 - 1900.0)
        return np.clip(0.6 + slope * (yr - 1900.0), 0.6, 0.9)
    raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")


def apply_nfactor(
    sat_monthly_anom_K: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    *,
    scenario: ScenarioName,
    lat_deg: float,
) -> np.ndarray:
    """Apply the winter n-factor scenario to a monthly SAT anomaly
    series.

    Parameters
    ----------
    sat_monthly_anom_K
        Length-N monthly SAT anomaly (anomalies, not absolute T).
    months
        Length-N 1-12 calendar months matching ``sat_monthly_anom_K``.
    years
        Length-N calendar year per sample.
    scenario
        Zhang n-factor scenario; see :data:`SCENARIOS`.
    lat_deg
        Site latitude in degrees. Determines which months are
        "winter": DJF in the northern hemisphere, JJA in the
        southern. Sites with $|lat| < 15$ have no operative winter
        and receive no modulation.

    Returns
    -------
    ndarray
        Modulated anomaly: winter months scaled by $n_w(t)$, all
        other months unchanged.
    """
    sat_anom = np.asarray(sat_monthly_anom_K, dtype=float)
    months = np.asarray(months, dtype=int)
    years = np.asarray(years, dtype=int)
    if not (sat_anom.shape == months.shape == years.shape):
        raise ValueError("sat, months, years must have the same shape")

    if abs(lat_deg) < 15.0:
        # No winter -- tropical site, no snow insulation possible.
        return sat_anom.copy()

    nw_per_year = nfactor_series(scenario=scenario, years=years)
    winter_mask = _winter_mask(months=months, lat_deg=lat_deg)
    multiplier = np.where(winter_mask, nw_per_year, 1.0)
    return multiplier * sat_anom


def _winter_mask(*, months: np.ndarray, lat_deg: float) -> np.ndarray:
    if lat_deg >= 0.0:
        # Northern hemisphere winter: DJF (months 12, 1, 2).
        return np.isin(months, (12, 1, 2))
    # Southern hemisphere winter: JJA.
    return np.isin(months, (6, 7, 8))


__all__ = [
    "SCENARIOS",
    "ScenarioName",
    "apply_nfactor",
    "nfactor_series",
]
