"""Tests for gt_theory.benchmarks.theis (1-D pressure-diffusion analog)."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.theis import (
    hydraulic_diffusivity,
    step_pressure_response,
)
from gt_theory.solvers.column_coupled import run_column_coupled

YEAR_S = 365.25 * 86400.0


def test_hydraulic_diffusivity_value() -> None:
    """c_v = K_zz / (mu phi beta_w) -- the textbook value (rho_w cancels)."""
    c_v = hydraulic_diffusivity(
        K_zz=1.0e-13,
        mu=1.0e-3,
        porosity=0.15,
        beta_w=4.5e-10,
    )
    expected = 1e-13 / (1e-3 * 0.15 * 4.5e-10)
    assert c_v == pytest.approx(expected, rel=1e-12)


def test_step_pressure_surface_equals_delta_p() -> None:
    p = step_pressure_response(
        np.array([0.0]),
        np.array([1.0 * YEAR_S]),
        delta_p=1.0e5,
        c_v=1e-4,
    )
    assert p[0, 0] == pytest.approx(1e5)


def test_step_pressure_deep_zero() -> None:
    p = step_pressure_response(
        np.array([5000.0]),
        np.array([0.01 * YEAR_S]),
        delta_p=1e5,
        c_v=1e-8,
    )
    assert abs(p[0, 0]) < 1e-3


def test_solver_matches_step_pressure_response_uncoupled() -> None:
    """With gamma_n_alpha = 0 and no thermal forcing, applying a step
    surface pressure to the column should match the analytical erfc
    solution at the chosen diffusivity."""
    beta_w = 4.5e-10
    mu = 1.0e-3
    phi = 0.15
    K_zz = 1.0e-13
    rho_w = 1000.0
    c_v = hydraulic_diffusivity(
        K_zz=K_zz,
        mu=mu,
        porosity=phi,
        beta_w=beta_w,
    )
    L = 200.0
    dp = 1.0e5
    duration = 0.05 * L * L / c_v
    res = run_column_coupled(
        depth_max_m=L,
        dz_m=2.0,
        duration_s=duration,
        dt_s=duration / 500.0,
        K_zz=K_zz,
        mu=mu,
        porosity=phi,
        beta_w=beta_w,
        rho_w=rho_w,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        sat=0.0,
        p_top=dp,
        T_init=0.0,
        p_init=0.0,
    )
    p_an = step_pressure_response(
        res.z,
        np.array([res.t[-1]]),
        delta_p=dp,
        c_v=c_v,
    )[0]
    # Compare only at depths well within the diffusion front to avoid
    # the bottom-boundary artefact.
    mask = res.z < 0.4 * L
    err = float(np.max(np.abs(res.p[-1, mask] - p_an[mask])) / dp)
    assert err < 0.05, f"max rel err vs Theis = {err:.3e}"


def test_c_v_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        hydraulic_diffusivity(
            K_zz=-1.0,
            mu=1.0e-3,
            porosity=0.15,
            beta_w=4.5e-10,
        )
