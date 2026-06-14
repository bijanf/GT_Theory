"""Bandpass + cross-correlation utilities for the F5 diffusive-lag
fingerprint.

The theory paper's F5 fingerprint predicts a thermal diffusive lag
$\\tau_{\\mathrm{lag}}(z) = z^2 / (4\\kappa)$ between the
surface-air-temperature (SAT) signal and the ground-temperature
response at depth $z$. At $\\kappa = 10^{-6}$\\,m$^2$\\,s$^{-1}$
and $z = 20$\\,m, that is $\\sim 38$--$50$\\,months.

The previous-revision quoted observed value of 117 months at 20\\,m
was a *raw cross-correlation* of monthly SAT against monthly GST
without high-pass filtering, which is dominated by the
multi-decadal warming trend rather than the diffusive seasonal-band
response that the theoretical prediction describes.

The bandpass functions here strip the secular trend (low frequency)
and the sub-decadal noise (high frequency) so the cross-correlation
isolates the decadal band where the diffusive lag prediction lives.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def butter_bandpass(
    series: np.ndarray,
    *,
    low_yr: float,
    high_yr: float,
    fs_per_year: float = 12.0,
    order: int = 4,
) -> np.ndarray:
    """4-th-order Butterworth bandpass with zero-phase ``filtfilt``.

    Parameters
    ----------
    series
        1-D array of equispaced samples.
    low_yr, high_yr
        Passband endpoints in years (``low_yr`` is the long-period
        cut-off; ``high_yr`` is the short-period cut-off).
    fs_per_year
        Samples per year. Default 12 for monthly data.
    order
        Butterworth order. Default 4 (a standard, mild-rolloff choice).

    Returns
    -------
    ndarray
        Filtered series, same shape as input.
    """
    if low_yr <= high_yr:
        raise ValueError(
            f"low_yr ({low_yr}) must be > high_yr ({high_yr}): "
            "low_yr is the long-period cut-off, high_yr the short-period"
        )
    nyquist_yr = 2.0 / fs_per_year  # in years, Nyquist period
    if high_yr <= nyquist_yr:
        raise ValueError(
            f"high_yr ({high_yr}) must exceed Nyquist period "
            f"({nyquist_yr}) for fs_per_year={fs_per_year}"
        )
    # Cycles per sample at the two cut-offs.
    f_low_cps = 1.0 / (low_yr * fs_per_year)
    f_high_cps = 1.0 / (high_yr * fs_per_year)
    # scipy expects normalised frequencies in (0, 1) where 1 = Nyquist.
    wn = [f_low_cps * 2.0, f_high_cps * 2.0]
    sos = sps.butter(order, wn, btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, series)


def cross_correlation_lag(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[int, np.ndarray]:
    """Lag (in samples) at which the cross-correlation of ``x``
    against ``y`` peaks.

    Sign convention: positive lag means ``y`` lags ``x``
    (i.e. ``y[t]`` correlates with ``x[t - lag]``). This is the
    intuitive ``ground lags surface`` convention.

    Parameters
    ----------
    x, y
        Same-length 1-D arrays.
    max_lag
        If set, restrict the search to ``|lag| <= max_lag``.

    Returns
    -------
    (lag_samples, full_correlation)
        ``lag_samples`` is the integer lag of the peak;
        ``full_correlation`` is the full cross-correlation array
        (centred so that index 0 corresponds to zero lag).
    """
    if x.shape != y.shape:
        raise ValueError(f"x.shape ({x.shape}) != y.shape ({y.shape})")
    x_d = x - x.mean()
    y_d = y - y.mean()
    norm = float(np.sqrt(np.sum(x_d**2) * np.sum(y_d**2)))
    if norm == 0:
        raise ValueError("zero variance in x or y — cannot correlate")
    full = sps.correlate(y_d, x_d, mode="full") / norm
    lags = np.arange(-x.size + 1, x.size)
    if max_lag is not None:
        m = np.abs(lags) <= max_lag
        full = full[m]
        lags = lags[m]
    peak_idx = int(np.argmax(full))
    lag = int(lags[peak_idx])
    return lag, full


def bandpass_lag_months(
    sat_monthly: np.ndarray,
    gst_monthly: np.ndarray,
    *,
    low_yr: float = 30.0,
    high_yr: float = 3.0,
    max_lag_months: int = 240,
) -> int:
    """End-to-end: bandpass-filter both series, then return the
    cross-correlation lag in months.

    A positive return value means the ground-temperature series
    lags the surface-air-temperature series.

    Parameters
    ----------
    sat_monthly, gst_monthly
        Monthly equispaced series.
    low_yr, high_yr
        Bandpass endpoints. Defaults to 30-year long cut-off and
        3-year short cut-off, isolating the decadal band where the
        diffusive lag prediction is valid.
    max_lag_months
        Maximum |lag| to consider; defaults to 240 (20 years).
    """
    sat = np.asarray(sat_monthly, dtype=float)
    gst = np.asarray(gst_monthly, dtype=float)
    if sat.shape != gst.shape:
        raise ValueError("sat_monthly and gst_monthly must be same length")
    sat_f = butter_bandpass(sat, low_yr=low_yr, high_yr=high_yr)
    gst_f = butter_bandpass(gst, low_yr=low_yr, high_yr=high_yr)
    lag, _ = cross_correlation_lag(
        sat_f,
        gst_f,
        max_lag=max_lag_months,
    )
    return lag


def theoretical_diffusive_lag_months(
    *,
    depth_m: float,
    kappa_m2_per_s: float,
) -> float:
    """The Carslaw-Jaeger diffusive lag $z^2 / (4 \\kappa)$ in months."""
    seconds = depth_m * depth_m / (4.0 * kappa_m2_per_s)
    return float(seconds / (30.44 * 86400.0))  # mean month length


__all__ = [
    "bandpass_lag_months",
    "butter_bandpass",
    "cross_correlation_lag",
    "theoretical_diffusive_lag_months",
]
