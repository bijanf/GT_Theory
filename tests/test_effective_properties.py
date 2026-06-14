"""Tests for gt_theory.theory.effective_properties."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.theory.dimless import RHO_ICE, RHO_WATER
from gt_theory.theory.effective_properties import (
    C_ICE,
    C_ROCK_DEFAULT,
    C_WATER,
    LAMBDA_ICE,
    LAMBDA_ROCK_DEFAULT,
    LAMBDA_WATER,
    RHO_ROCK_DEFAULT,
    brooks_corey_k_rel,
    lambda_eff_geometric_mean,
    rho_c_eff_two_phase,
)


def test_lambda_eff_no_ice_limit() -> None:
    """At S_i = 0 the geometric mean reduces to lambda_r^(1-phi) lambda_w^phi."""
    phi = 0.3
    lam = lambda_eff_geometric_mean(np.array([0.0]), porosity=phi)
    expected = LAMBDA_ROCK_DEFAULT ** (1.0 - phi) * LAMBDA_WATER**phi
    assert lam[0] == pytest.approx(expected)


def test_lambda_eff_full_ice_limit() -> None:
    """At S_i = 1 the geometric mean reduces to lambda_r^(1-phi) lambda_i^phi."""
    phi = 0.3
    lam = lambda_eff_geometric_mean(np.array([1.0]), porosity=phi)
    expected = LAMBDA_ROCK_DEFAULT ** (1.0 - phi) * LAMBDA_ICE**phi
    assert lam[0] == pytest.approx(expected)


def test_lambda_eff_monotone_in_S_i() -> None:
    """For lambda_i > lambda_w (true for water below 0 C), lambda_eff
    increases monotonically as S_i goes from 0 to 1."""
    phi = 0.3
    S_i = np.linspace(0.0, 1.0, 21)
    lam = lambda_eff_geometric_mean(S_i, porosity=phi)
    assert LAMBDA_ICE > LAMBDA_WATER  # sanity
    assert np.all(np.diff(lam) >= -1e-12)


def test_brooks_corey_no_ice_limit() -> None:
    """k_rel(S_w=1) = 1 by construction."""
    k = brooks_corey_k_rel(np.array([0.0]))
    assert k[0] == pytest.approx(1.0)


def test_brooks_corey_full_ice_limit_zero_residual() -> None:
    """k_rel(S_w=0) = 0 when no residual unfrozen water."""
    k = brooks_corey_k_rel(np.array([1.0]))
    assert k[0] == pytest.approx(0.0)


def test_brooks_corey_eta_3_specific_value() -> None:
    """At S_w = 0.5, k_rel = 0.5^3 = 0.125."""
    k = brooks_corey_k_rel(np.array([0.5]), eta=3.0)
    assert k[0] == pytest.approx(0.125)


def test_rho_c_eff_no_ice() -> None:
    phi = 0.3
    rce = rho_c_eff_two_phase(np.array([0.0]), porosity=phi)
    expected = (1.0 - phi) * RHO_ROCK_DEFAULT * C_ROCK_DEFAULT + phi * RHO_WATER * C_WATER
    assert rce[0] == pytest.approx(expected)


def test_rho_c_eff_full_ice() -> None:
    phi = 0.3
    rce = rho_c_eff_two_phase(np.array([1.0]), porosity=phi)
    expected = (1.0 - phi) * RHO_ROCK_DEFAULT * C_ROCK_DEFAULT + phi * RHO_ICE * C_ICE
    assert rce[0] == pytest.approx(expected)


def test_rho_c_eff_water_to_ice_decreases() -> None:
    """For fixed phi, replacing water with ice in the pores decreases
    the sensible heat capacity (since rho_i c_i < rho_w c_w)."""
    phi = 0.3
    rce_water = rho_c_eff_two_phase(np.array([0.0]), porosity=phi)
    rce_ice = rho_c_eff_two_phase(np.array([1.0]), porosity=phi)
    assert rce_ice[0] < rce_water[0]
