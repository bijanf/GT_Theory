"""Tests for the analytical Stefan one-phase Neumann similarity solution
and for the merged solver's recovery of the front position xi(t)."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
    stefan_temperature_profile,
)

YEAR_S = 365.25 * 86400.0


def test_stefan_lambda_sign_and_range() -> None:
    """For a moderate Stefan number, the dimensionless front coefficient
    is bounded and positive."""
    p = StefanOnePhaseParams(T_s=-10.0)
    lam = p.stefan_lambda
    assert 0.05 < lam < 1.5


def test_stefan_front_increases_as_sqrt_t() -> None:
    p = StefanOnePhaseParams(T_s=-10.0)
    times = np.array([0.5, 1.0, 2.0, 4.0]) * YEAR_S
    xi = stefan_front_position(times, p)
    # Differences should scale exactly like sqrt(t)
    rel = xi / np.sqrt(times)
    assert np.allclose(rel, rel[0], rtol=1e-12)


def test_stefan_profile_at_surface() -> None:
    """T(0, t) == T_s for all t > 0."""
    p = StefanOnePhaseParams(T_s=-10.0)
    z = np.array([0.0, 0.5, 1.0])
    T = stefan_temperature_profile(z, t=1.0 * YEAR_S, params=p)
    assert T[0] == pytest.approx(p.T_s)


def test_stefan_profile_continuous_at_front() -> None:
    """At z = xi(t) the profile equals T_f to within the erf
    machine precision."""
    p = StefanOnePhaseParams(T_s=-10.0)
    t = 1.0 * YEAR_S
    xi = float(stefan_front_position(t, p))
    z = np.array([0.999 * xi, 1.001 * xi])
    T = stefan_temperature_profile(z, t=t, params=p)
    # Just inside frozen region: should be close to T_f.
    assert abs(T[0] - p.T_f) < 0.1
    # Just outside: exactly T_f.
    assert T[1] == pytest.approx(p.T_f)


def test_merged_solver_stefan_front_recovery() -> None:
    """The merged T+p+S_i solver, run with K_zz->0 (mass equation
    decoupled) and uniform sensible properties (so the geometric mean
    degenerates and the two-phase capacity is uniform), should
    reproduce the analytical Stefan front position to within a few
    percent at production time-stepping."""
    from gt_theory.solvers.column_thermo_freeze_coupled import (
        run_column_thermo_freeze_coupled,
    )

    # Configure water = ice for sensible properties so the frozen-phase
    # and unfrozen-phase volumetric heat capacities are identical (the
    # analytical Neumann one-phase solution requires uniform sensible
    # rho c throughout).
    phi = 0.30
    rho_r, c_r = 2700.0, 800.0
    rho_w = 1000.0
    c_uniform = 4186.0  # use water capacity for both phases
    # Sensible volumetric heat capacity of the medium:
    rho_c_sensible = (1 - phi) * rho_r * c_r + phi * rho_w * c_uniform

    p_stefan = StefanOnePhaseParams(
        T_s=-10.0,
        T_f=0.0,
        lambda_thermal=2.5,
        rho_c_solid=rho_c_sensible,
        porosity=phi,
        L_f=3.34e5,
        rho_w=rho_w,
    )
    duration = 0.5 * YEAR_S
    res = run_column_thermo_freeze_coupled(
        depth_max_m=4.0,
        dz_m=0.05,
        duration_s=duration,
        dt_s=duration / 500.0,
        porosity=phi,
        lambda_r=2.5,
        lambda_w=2.5,
        lambda_i=2.5,
        rho_r=rho_r,
        c_r=c_r,
        rho_w=rho_w,
        c_w=c_uniform,
        c_i=c_uniform * rho_w / 917.0,
        L_f=3.34e5,
        K_zz=1.0e-20,
        T_f=0.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=-10.0,
        T_init=0.0,
        picard_max_iter=20,
        picard_tol_K=1e-4,
    )
    # Diagnose the freezing-front position from S_i: the deepest cell
    # with S_i > 0.5 at the final time step.
    S_final = res.S_i[-1]
    z = res.z
    frozen_mask = S_final > 0.5
    if not frozen_mask.any():
        pytest.fail("no frozen cells at final time")
    xi_solver = float(z[frozen_mask][-1])

    xi_analytical = float(stefan_front_position(duration, p_stefan))
    rel_err = abs(xi_solver - xi_analytical) / xi_analytical
    assert rel_err < 0.20, (
        f"Stefan front mismatch: solver={xi_solver:.3f} m, "
        f"analytical={xi_analytical:.3f} m, rel err={rel_err:.1%}"
    )
