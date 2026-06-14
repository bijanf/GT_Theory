"""Tests for gt_theory.benchmarks.undrained_ratio."""

from __future__ import annotations

import pytest

from gt_theory.benchmarks.undrained_ratio import expected_dp_dT
from gt_theory.solvers.column_coupled import run_column_coupled

YEAR_S = 365.25 * 86400.0


def test_expected_ratio_value() -> None:
    """alpha_w / beta_w with typical water values is O(10^5) Pa K^-1."""
    r = expected_dp_dT(alpha_w=2.1e-4, beta_w=4.5e-10)
    assert 1e5 < r < 1e6


def test_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        expected_dp_dT(alpha_w=-1.0, beta_w=4.5e-10)
    with pytest.raises(ValueError, match="positive"):
        expected_dp_dT(alpha_w=2.1e-4, beta_w=0.0)


def test_undrained_pressure_response_matches_ratio() -> None:
    """In the undrained limit (K_zz -> 0), the pressure response to a
    surface-temperature warming should satisfy dp/dT = alpha_w / beta_w
    at depth, where surface drainage has not yet reached."""
    alpha_w = 2.1e-4
    beta_w = 4.5e-10
    rho_w = 1000.0
    res = run_column_coupled(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        lambda_thermal=2.5,
        rho_c_eff=2.5e6,
        # K_zz small enough that drainage (at the corrected hydraulic
        # diffusivity c_v = K/(mu phi beta) ~ 1.5e-10 m^2/s here) stays
        # far above the 60 m probe over 100 yr -- the true undrained limit.
        K_zz=1.0e-23,
        mu=1.0e-3,
        porosity=0.15,
        alpha_w=alpha_w,
        beta_w=beta_w,
        rho_w=rho_w,
        g=0.0,
        gamma_n_alpha_scale=1.0,
        sat=1.0,
        p_top=0.0,
    )
    i_mid = 30
    T_mid = res.T[-1, i_mid]
    p_mid = res.p[-1, i_mid]
    expected = expected_dp_dT(alpha_w=alpha_w, beta_w=beta_w)
    assert T_mid > 0.05
    rel_err = abs(p_mid / T_mid - expected) / expected
    assert rel_err < 0.01, f"rel err = {rel_err:.3e}"
