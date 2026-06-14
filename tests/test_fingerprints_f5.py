"""Tests for the F5 diffusive-lag fingerprint."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.fingerprints.f5_diffusive_lag import (
    compute_f5_at_site,
)
from gt_theory.stats.signal import theoretical_diffusive_lag_months


def test_f5_recovers_positive_lag_at_5m() -> None:
    """A long monthly SAT anomaly should give a 5 m forward-simulated
    GST whose bandpass-filtered lag is positive (ground lags surface)
    and well below the 20 m lag (consistent with the
    $\\sqrt{z^2/\\kappa}$ phase-delay scaling for a periodic signal)."""
    rng = np.random.default_rng(11)
    n_months = 12 * 100
    t_yr = np.arange(n_months) / 12.0
    # SAT: decadal oscillation + small noise; no secular trend.
    sat = 2.0 * np.sin(2 * np.pi * t_yr / 12.0) + rng.normal(0.0, 0.2, size=n_months)
    kappa = 1.0e-6
    result = compute_f5_at_site(
        site_id="test_site",
        lat_deg=45.0,
        lon_deg=0.0,
        sat_monthly_K=sat,
        kappa_m2_per_s=kappa,
        low_yr=30.0,
        high_yr=3.0,
    )
    # Lag at 5 m is positive (ground lags surface) and smaller than
    # the 20 m lag.
    assert result.lag_5m_filtered_months > 0
    assert result.lag_5m_filtered_months < result.lag_20m_filtered_months
    # 20 m lag is within ±15 mo of the diffusive theory
    # (allow phase-vs-step-response framing slack).
    theory_20m = theoretical_diffusive_lag_months(
        depth_m=20.0,
        kappa_m2_per_s=kappa,
    )
    assert abs(result.lag_20m_filtered_months - theory_20m) < 15


def test_f5_bandpass_reduces_secular_trend_inflation() -> None:
    """Add a secular trend to the SAT; the unfiltered lag at 20 m
    inflates beyond the theoretical value, but the filtered lag stays
    closer to the theory prediction."""
    rng = np.random.default_rng(31)
    n_months = 12 * 120
    t_yr = np.arange(n_months) / 12.0
    # Decadal oscillation + secular warming
    sat = (
        1.5 * np.sin(2 * np.pi * t_yr / 8.0)
        + 0.02 * t_yr  # 2 K/century warming
        + rng.normal(0.0, 0.2, size=n_months)
    )
    kappa = 1.0e-6
    result = compute_f5_at_site(
        site_id="trend_site",
        lat_deg=50.0,
        lon_deg=0.0,
        sat_monthly_K=sat,
        kappa_m2_per_s=kappa,
    )
    theory_20m = theoretical_diffusive_lag_months(
        depth_m=20.0,
        kappa_m2_per_s=kappa,
    )
    # Filtered lag stays closer to theory than the unfiltered lag.
    err_filtered = abs(result.lag_20m_filtered_months - theory_20m)
    err_unfiltered = abs(result.lag_20m_unfiltered_months - theory_20m)
    assert err_filtered <= err_unfiltered + 1  # allow integer ties


def test_f5_rejects_short_input() -> None:
    sat_short = np.zeros(12 * 5)  # 5 yr -- too short
    with pytest.raises(ValueError, match="too short"):
        compute_f5_at_site(
            site_id="short",
            lat_deg=0.0,
            lon_deg=0.0,
            sat_monthly_K=sat_short,
        )
