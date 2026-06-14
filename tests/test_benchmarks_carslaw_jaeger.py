"""Tests for gt_theory.benchmarks.carslaw_jaeger."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.carslaw_jaeger import (
    harmonic_temperature_response,
    step_temperature_response,
)
from gt_theory.solvers import run_column_thermo_freeze_coupled

YEAR_S = 365.25 * 86400.0


def test_step_response_surface_equals_dT() -> None:
    """At z = 0, the step response equals the imposed delta_T."""
    T = step_temperature_response(
        np.array([0.0]),
        np.array([1.0 * YEAR_S]),
        delta_T=2.0,
        kappa=1e-6,
    )
    assert T[0, 0] == pytest.approx(2.0)


def test_step_response_far_from_surface_zero() -> None:
    """Deep enough that the diffusion front hasn't reached, T -> 0."""
    T = step_temperature_response(
        np.array([200.0]),
        np.array([0.1 * YEAR_S]),
        delta_T=10.0,
        kappa=1e-7,
    )
    assert abs(T[0, 0]) < 1e-6


def test_solver_matches_step_response_uncoupled() -> None:
    """With ice off + coupling off + uniform sensible properties, the
    merged solver reproduces the Carslaw-Jaeger step response."""
    kappa = 1.0e-6
    phi = 0.10
    rho_r, c_r = 2700.0, 800.0
    rho_w, c_w = 1000.0, 4186.0
    rho_c_eff = (1 - phi) * rho_r * c_r + phi * rho_w * c_w
    lam = kappa * rho_c_eff

    res = run_column_thermo_freeze_coupled(
        depth_max_m=400.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        porosity=phi,
        lambda_r=lam,
        lambda_w=lam,
        lambda_i=lam,
        rho_r=rho_r,
        c_r=c_r,
        rho_w=rho_w,
        c_w=c_w,
        c_i=c_w * rho_w / 917.0,
        L_f=3.34e5,
        K_zz=1.0e-20,
        T_f=-50.0,
        dTc=0.1,  # push freezing out of the simulated range
        gamma_n_alpha_scale=0.0,
        sat=1.0,
        T_init=0.0,
    )
    T_an = step_temperature_response(
        res.z,
        np.array([res.t[-1]]),
        delta_T=1.0,
        kappa=kappa,
    )[0]
    err = float(np.max(np.abs(res.T[-1] - T_an)))
    assert err < 0.05, f"max |solver - analytical| = {err:.4f} K"


def test_harmonic_skin_depth_decay() -> None:
    """Peak amplitude at depth d (the skin depth) is A / e."""
    A = 5.0
    period = YEAR_S
    kappa = 1e-6
    d = float(np.sqrt(2.0 * kappa / (2.0 * np.pi / period)))
    # Sample over one full period and take the envelope amplitude.
    t = np.linspace(0.0, period, 200)
    T = harmonic_temperature_response(
        np.array([0.0, d]),
        t,
        amplitude_K=A,
        period_s=period,
        kappa=kappa,
    )
    assert float(np.max(np.abs(T[:, 0]))) == pytest.approx(A, rel=1e-3)
    assert float(np.max(np.abs(T[:, 1]))) == pytest.approx(A / np.e, rel=1e-3)


def test_step_response_singular_at_zero_time() -> None:
    with pytest.raises(ValueError):
        step_temperature_response(np.array([1.0]), np.array([0.0]), delta_T=1.0, kappa=1e-6)
