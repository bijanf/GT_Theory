"""Tests for gt_theory.theory.freezing_curve."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.theory.freezing_curve import (
    FreezingCurveParams,
    apparent_volumetric_capacity,
    ice_saturation,
    ice_saturation_derivative,
)


def test_ice_saturation_piecewise_segments() -> None:
    params = FreezingCurveParams(T_f=0.0, dTc=1.0, S_w_residual=0.0)
    T = np.array([-2.0, -1.0, -0.5, 0.0, 1.0])
    S_i = ice_saturation(T, params)
    # T = -2: fully frozen, S_i = 1
    assert S_i[0] == pytest.approx(1.0)
    # T = -1: at the boundary, fully frozen (T <= T_f - dTc)
    assert S_i[1] == pytest.approx(1.0)
    # T = -0.5: half-way through the interval
    assert S_i[2] == pytest.approx(0.5)
    # T = 0: at the upper boundary, fully liquid
    assert S_i[3] == pytest.approx(0.0)
    # T = 1: fully liquid
    assert S_i[4] == pytest.approx(0.0)


def test_ice_saturation_residual_unfrozen() -> None:
    params = FreezingCurveParams(T_f=0.0, dTc=1.0, S_w_residual=0.1)
    # Fully frozen state should be 1 - 0.1 = 0.9
    S_i = ice_saturation(np.array([-5.0]), params)
    assert S_i[0] == pytest.approx(0.9)
    # Inside interval at half-way: 0.5 * 0.9 = 0.45
    S_i_mid = ice_saturation(np.array([-0.5]), params)
    assert S_i_mid[0] == pytest.approx(0.45)


def test_ice_saturation_derivative_segments() -> None:
    params = FreezingCurveParams(T_f=0.0, dTc=0.5, S_w_residual=0.0)
    T = np.array([-1.0, -0.25, 0.5])
    dSi_dT = ice_saturation_derivative(T, params)
    assert dSi_dT[0] == pytest.approx(0.0)  # fully frozen plateau
    assert dSi_dT[1] == pytest.approx(-1.0 / 0.5)  # inside interval
    assert dSi_dT[2] == pytest.approx(0.0)  # fully liquid plateau


def test_freezing_curve_invariants_at_boundaries() -> None:
    """S_i is continuous and monotone non-increasing in T."""
    params = FreezingCurveParams(T_f=0.0, dTc=1.0)
    T = np.linspace(-3.0, 3.0, 121)
    S_i = ice_saturation(T, params)
    diffs = np.diff(S_i)
    assert np.all(diffs <= 1e-12)  # monotone non-increasing
    assert np.all(S_i >= 0.0)
    assert np.all(S_i <= 1.0)


def test_apparent_capacity_spike_magnitude() -> None:
    """Inside the freezing interval the apparent capacity is
    base + rho_w L_f phi (1-S_r)/dTc, matching the analytical limit."""
    params = FreezingCurveParams(T_f=0.0, dTc=0.5, S_w_residual=0.0)
    rho_c_dry = 2.5e6
    phi = 0.3
    L_f = 3.34e5
    rho_w = 1000.0
    c_app = apparent_volumetric_capacity(
        np.array([-0.25]),
        rho_c_dry=rho_c_dry,
        porosity=phi,
        L_f=L_f,
        rho_w=rho_w,
        params=params,
    )
    expected_spike = rho_w * L_f * phi * 1.0 / 0.5
    assert c_app[0] == pytest.approx(rho_c_dry + expected_spike)


def test_apparent_capacity_outside_interval() -> None:
    params = FreezingCurveParams(T_f=0.0, dTc=0.5)
    rho_c_dry = 2.5e6
    c_app = apparent_volumetric_capacity(
        np.array([-5.0, 5.0]),
        rho_c_dry=rho_c_dry,
        porosity=0.3,
        L_f=3.34e5,
        rho_w=1000.0,
        params=params,
    )
    # Both outside interval -> base capacity only.
    assert c_app[0] == pytest.approx(rho_c_dry)
    assert c_app[1] == pytest.approx(rho_c_dry)


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        FreezingCurveParams(dTc=0.0)
    with pytest.raises(ValueError):
        FreezingCurveParams(dTc=-1.0)
    with pytest.raises(ValueError):
        FreezingCurveParams(S_w_residual=1.5)
