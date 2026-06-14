"""Tests for the F5 bandpass + cross-correlation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.stats.signal import (
    bandpass_lag_months,
    butter_bandpass,
    cross_correlation_lag,
    theoretical_diffusive_lag_months,
)


def test_bandpass_passes_decadal_signal() -> None:
    """A 10-year sine should pass cleanly through a (30, 3) bandpass."""
    n_months = 12 * 60
    t = np.arange(n_months) / 12.0
    signal = np.sin(2 * np.pi * t / 10.0)
    filtered = butter_bandpass(signal, low_yr=30.0, high_yr=3.0)
    # Filter passes the signal; allow ~10% amplitude loss from rolloff
    # + filtfilt edge effects.
    inner = filtered[60:-60]
    inner_signal = signal[60:-60]
    err_rms = float(np.sqrt(np.mean((inner - inner_signal) ** 2)))
    assert err_rms < 0.15


def test_bandpass_rejects_secular_trend() -> None:
    """A linear trend should be largely removed by a (30, 3) bandpass."""
    n_months = 12 * 100
    t = np.arange(n_months) / 12.0
    trend = 0.5 * t  # 0.5 K per year ramp
    filtered = butter_bandpass(trend, low_yr=30.0, high_yr=3.0)
    inner = filtered[200:-200]
    assert np.max(np.abs(inner)) < 0.05 * np.max(np.abs(trend[200:-200]))


def test_bandpass_rejects_subannual_noise() -> None:
    """A high-frequency (~3-month) sine should be heavily attenuated.
    A (30, 1)-year passband has a 1/yr cutoff, so a 6-cycles-per-year
    sine is firmly in the rejection band."""
    n_months = 12 * 50
    t = np.arange(n_months) / 12.0
    fast = np.sin(2 * np.pi * t * 6.0)  # 6 cycles per year (2-mo period)
    filtered = butter_bandpass(fast, low_yr=30.0, high_yr=1.0)
    assert np.max(np.abs(filtered[100:-100])) < 0.1


def test_cross_correlation_recovers_known_lag() -> None:
    """Construct y = x[t - 24] + noise; the peak lag must be +24."""
    rng = np.random.default_rng(101)
    n = 600
    x = rng.normal(0.0, 1.0, size=n)
    lag_true = 24
    y = np.concatenate([np.zeros(lag_true), x])[:n]
    y += rng.normal(0.0, 0.05, size=n)
    lag, _ = cross_correlation_lag(x, y, max_lag=120)
    assert lag == pytest.approx(lag_true, abs=2)


def test_bandpass_lag_recovers_lag_after_trend() -> None:
    """A delayed sine + secular trend: unfiltered cross-correlation is
    biased by the trend; bandpass-filtered recovers the true lag."""
    rng = np.random.default_rng(7)
    n_months = 12 * 80
    t = np.arange(n_months) / 12.0
    base = np.sin(2 * np.pi * t / 10.0)
    sat = base + 0.05 * t  # secular warming on top
    gst = np.concatenate([np.zeros(24), base])[:n_months] + 0.05 * t
    sat += rng.normal(0.0, 0.05, size=n_months)
    gst += rng.normal(0.0, 0.05, size=n_months)
    lag_filtered = bandpass_lag_months(sat, gst, low_yr=30.0, high_yr=3.0)
    # The known true lag is 24 months; bandpass must come close.
    assert abs(lag_filtered - 24) < 6


def test_bandpass_requires_low_above_high() -> None:
    with pytest.raises(ValueError, match="low_yr"):
        butter_bandpass(
            np.zeros(120),
            low_yr=3.0,
            high_yr=30.0,  # wrong order
        )


def test_cross_correlation_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        cross_correlation_lag(np.zeros(10), np.arange(10))


def test_theoretical_lag_at_known_depth_and_kappa() -> None:
    # z = 20 m, kappa = 1e-6 m^2/s -> tau = 400 / 4e-6 = 1e8 s ~ 38 mo.
    lag_mo = theoretical_diffusive_lag_months(
        depth_m=20.0,
        kappa_m2_per_s=1.0e-6,
    )
    assert 35.0 < lag_mo < 45.0
