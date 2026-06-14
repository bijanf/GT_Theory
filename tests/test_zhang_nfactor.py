"""Tests for the Zhang 2005 winter n-factor scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.theory.zhang_nfactor import (
    SCENARIOS,
    apply_nfactor,
    nfactor_series,
)


def test_constant_heavy_scenario_is_flat() -> None:
    years = np.arange(1900, 2025)
    nw = nfactor_series(scenario="constant_heavy", years=years)
    assert np.allclose(nw, 0.5)


def test_no_insulation_scenario_is_one() -> None:
    years = np.arange(1900, 2025)
    nw = nfactor_series(scenario="no_insulation", years=years)
    assert np.allclose(nw, 1.0)


def test_declining_scenario_ramps() -> None:
    years = np.array([1900, 1962, 2024])
    nw = nfactor_series(scenario="declining_snow_insulation", years=years)
    assert nw[0] == pytest.approx(0.6, abs=1e-9)
    assert nw[-1] == pytest.approx(0.9, abs=1e-9)
    # Mid-point monotonic.
    assert nw[0] < nw[1] < nw[2]


def test_apply_nfactor_damps_winter_anomaly_in_NH() -> None:
    months = np.array([12, 1, 2, 3, 6, 7], dtype=int)
    years = np.array([2020] * 6, dtype=int)
    sat = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # uniform +2 K anomaly
    out = apply_nfactor(
        sat,
        months,
        years,
        scenario="constant_heavy",
        lat_deg=55.0,
    )
    # Winter months (12, 1, 2) damped by 0.5; March and summer unchanged.
    expected = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_apply_nfactor_tropical_is_passthrough() -> None:
    months = np.array([12, 1, 6, 7], dtype=int)
    years = np.array([2020] * 4, dtype=int)
    sat = np.array([3.0, 3.0, 3.0, 3.0])
    out = apply_nfactor(
        sat,
        months,
        years,
        scenario="constant_heavy",
        lat_deg=5.0,
    )
    np.testing.assert_allclose(out, sat)


def test_apply_nfactor_no_insulation_is_identity() -> None:
    months = np.tile(np.arange(1, 13), 5)
    years = np.repeat(np.arange(2020, 2025), 12)
    sat = np.linspace(-1.0, 3.0, 60)
    out = apply_nfactor(
        sat,
        months,
        years,
        scenario="no_insulation",
        lat_deg=60.0,
    )
    np.testing.assert_allclose(out, sat)


def test_apply_nfactor_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        apply_nfactor(
            np.zeros(5),
            np.zeros(4, dtype=int),
            np.zeros(5, dtype=int),
            scenario="constant_heavy",
            lat_deg=50.0,
        )


def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        nfactor_series(scenario="mystery", years=np.arange(2000, 2010))  # type: ignore[arg-type]


def test_scenarios_constant_exposed() -> None:
    assert "constant_heavy" in SCENARIOS
    assert "declining_snow_insulation" in SCENARIOS
    assert "no_insulation" in SCENARIOS
