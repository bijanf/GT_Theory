"""Tests for the InSAR-vs-solver cross-check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gt_theory.analysis import (
    compare_insar_solver,
    predict_surface_displacement,
    residual_reduction,
)
from gt_theory.solvers import CoupledResult, run_column_coupled

YEAR_S = 365.25 * 86400.0


def _example_coupled_result() -> CoupledResult:
    """Run a small coupled-solver case that will produce a non-zero
    vertical displacement at the surface."""
    return run_column_coupled(
        depth_max_m=10.0,
        dz_m=0.5,
        duration_s=1.0 * YEAR_S,
        dt_s=5.0 * 86400.0,
        lambda_thermal=1.8,
        rho_c_eff=2.5e6,
        K_zz=1.0e-13,
        mu=1.0e-3,
        porosity=0.20,
        alpha_w=2.1e-4,
        beta_w=4.5e-10,
        gamma_n_alpha_scale=1.0,
        sat=np.linspace(-3.0, 5.0, int(round((1.0 * YEAR_S) / (5.0 * 86400.0))) + 1),
        p_top=0.0,
    )


def test_predict_surface_displacement_increases_with_warming() -> None:
    res = _example_coupled_result()
    u = predict_surface_displacement(res, porosity=0.20)
    assert u.shape == (res.t.size,)
    # Surface starts at u=0 (no anomaly at t=0) and grows.
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[-1] > u[0]
    # Magnitude should be sub-metre (mm-scale, given alpha_eff ~ 4e-5 and ΔT < 10 K).
    assert abs(u[-1]) < 1000.0  # not blown up


def test_predict_zero_displacement_when_T_and_p_unchanged() -> None:
    """If T and p never deviate from their initial state, u_z must
    be identically zero (numerical noise aside)."""
    res = run_column_coupled(
        depth_max_m=5.0,
        dz_m=0.5,
        duration_s=0.5 * YEAR_S,
        dt_s=5.0 * 86400.0,
        lambda_thermal=1.8,
        K_zz=1.0e-15,
        mu=1.0e-3,
        porosity=0.15,
        alpha_w=2.1e-4,
        beta_w=4.5e-10,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        sat=0.0,
        p_top=0.0,
        T_init=0.0,
        p_init=0.0,
    )
    u = predict_surface_displacement(res)
    assert float(np.max(np.abs(u))) < 1.0e-8


def _make_synthetic_insar(
    time_s: np.ndarray, displacement_mm: np.ndarray, reference_time: pd.Timestamp
) -> xr.Dataset:
    obs_times = reference_time + pd.to_timedelta(time_s, unit="s")
    return xr.Dataset(
        data_vars={
            "lat": ((), 47.235),
            "lon": ((), 7.155),
            "velocity_mm_yr": ((), -1.0),
            "velocity_std_mm_yr": ((), 0.2),
            "displacement_mm": (("time",), displacement_mm),
        },
        coords={"point": "test", "time": obs_times.values},
    )


def test_compare_insar_solver_reports_residual_reduction() -> None:
    """When the s=1 solver matches the observed series exactly and
    s=0 differs from it, the residual reduction must be near 1.0."""
    ref = pd.Timestamp("2020-01-01")
    t_solver = np.linspace(0.0, YEAR_S, 25)
    u_on = 0.5 * t_solver / YEAR_S  # mm
    u_off = np.zeros_like(t_solver)
    obs_time = np.linspace(0.05, 0.95, 10) * YEAR_S
    obs_disp = 0.5 * obs_time / YEAR_S
    ds = _make_synthetic_insar(obs_time, obs_disp, ref)

    cmp = compare_insar_solver(
        ds,
        solver_t_s=t_solver,
        u_solver_on_mm=u_on,
        u_solver_off_mm=u_off,
        reference_time=ref,
    )
    assert cmp.rms_on_mm < 1.0e-6
    assert cmp.rms_off_mm > 0.1
    assert cmp.residual_reduction > 0.95


def test_compare_insar_solver_zero_reduction_when_solver_does_nothing() -> None:
    """If s=1 and s=0 are identical solver outputs, residual reduction
    must be exactly 0 (no improvement from coupling)."""
    ref = pd.Timestamp("2020-01-01")
    t_solver = np.linspace(0.0, YEAR_S, 25)
    u_curve = 0.2 * t_solver / YEAR_S
    obs_time = np.linspace(0.05, 0.95, 10) * YEAR_S
    obs_disp = 0.5 * obs_time / YEAR_S
    ds = _make_synthetic_insar(obs_time, obs_disp, ref)

    red = residual_reduction(
        ds,
        solver_t_s=t_solver,
        u_solver_on_mm=u_curve,
        u_solver_off_mm=u_curve.copy(),
        reference_time=ref,
    )
    assert red == pytest.approx(0.0, abs=1e-9)


def test_compare_insar_solver_raises_on_disjoint_time_window() -> None:
    ref = pd.Timestamp("2030-01-01")  # solver window 2030
    t_solver = np.linspace(0.0, YEAR_S, 25)
    u_on = u_off = np.zeros_like(t_solver)
    obs_time = np.linspace(0.05, 0.95, 5) * YEAR_S
    obs_disp = np.zeros_like(obs_time)
    ds = _make_synthetic_insar(
        obs_time,
        obs_disp,
        pd.Timestamp("2010-01-01"),  # InSAR in 2010
    )
    with pytest.raises(ValueError, match="overlap"):
        compare_insar_solver(
            ds,
            solver_t_s=t_solver,
            u_solver_on_mm=u_on,
            u_solver_off_mm=u_off,
            reference_time=ref,
        )
