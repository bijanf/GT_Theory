"""Carslaw-Jaeger synthetic-twin test for the 1-D Crank-Nicolson solver.

Step surface forcing on a deep column with no advection and an insulated
bottom: the numerical solution must track the analytic erfc envelope.
"""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.solvers import (
    SolverResult,
    carslaw_jaeger_step_analytic,
    run_column_1d,
)

YEAR_S: float = 365.25 * 86400.0


@pytest.fixture(scope="module")
def step_run() -> SolverResult:
    """Run a 100-year step-surface problem on a 1000 m column."""
    return run_column_1d(
        depth_max_m=1000.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 12.0,  # monthly time step
        kappa=1.0e-6,
        q_bot=0.0,
        sat=1.0,  # constant +1 K step at the surface
        n_winter=1.0,
        v_darcy=0.0,
        T_init=0.0,
    )


def test_solver_outputs_have_consistent_shapes(step_run: SolverResult) -> None:
    assert step_run.T.shape == (step_run.t.size, step_run.z.size)
    assert step_run.gst.shape == (step_run.t.size,)
    assert step_run.z[0] == 0.0
    assert np.isclose(step_run.z[-1], 1000.0, atol=2.0)


def test_solver_surface_tracks_gst(step_run: SolverResult) -> None:
    """The Dirichlet top BC must be enforced exactly at every time step."""
    np.testing.assert_allclose(step_run.T[:, 0], step_run.gst, atol=1e-12)


def test_solver_initial_condition_zero(step_run: SolverResult) -> None:
    """At t = 0 the column is at the prescribed zero initial state below
    the surface node."""
    np.testing.assert_allclose(step_run.T[0, 1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("z_test_m", [20.0, 50.0, 100.0, 200.0, 400.0])
def test_carslaw_jaeger_recovery_per_depth(step_run: SolverResult, z_test_m: float) -> None:
    """At each test depth, the numerical solution should track the analytic
    erfc envelope with RMS < 5e-2 K over the second half of the run.

    The second half avoids start-up bias from the discrete time grid.
    """
    iz = int(np.argmin(np.abs(step_run.z - z_test_m)))
    half = step_run.t.size // 2

    t_pos = step_run.t[half:]
    T_num = step_run.T[half:, iz]
    T_ana = carslaw_jaeger_step_analytic(
        z=np.array([step_run.z[iz]]),
        t=t_pos,
        delta_T=1.0,
        kappa=1.0e-6,
    ).ravel()

    rms = float(np.sqrt(np.mean((T_num - T_ana) ** 2)))
    assert rms < 5.0e-2, f"RMS error at z={step_run.z[iz]:.1f} m is {rms:.4f} K"


def test_carslaw_jaeger_recovery_global(step_run: SolverResult) -> None:
    """Same comparison over the whole 20-400 m / second-half window."""
    z_window = (step_run.z >= 20.0) & (step_run.z <= 400.0)
    half = step_run.t.size // 2

    Z, T = np.meshgrid(step_run.z[z_window], step_run.t[half:], indexing="xy")
    T_num = step_run.T[half:, :][:, z_window]
    T_ana = carslaw_jaeger_step_analytic(
        z=step_run.z[z_window],
        t=step_run.t[half:],
        delta_T=1.0,
        kappa=1.0e-6,
    )
    rms_global = float(np.sqrt(np.mean((T_num - T_ana) ** 2)))
    assert rms_global < 3.0e-2, f"Global RMS = {rms_global:.4f} K"


def test_zhang_n_factor_attenuates_winter_only() -> None:
    """A periodic surface forcing with 50% sub-zero half should be damped
    only on the negative excursion when n_winter < 1."""
    nt = 12 * 50  # 50 yr at monthly cadence
    months = np.arange(nt)
    sat = np.cos(2.0 * np.pi * months / 12.0)  # ±1 K seasonal cycle
    duration_s = nt * (YEAR_S / 12.0)

    with_snow = run_column_1d(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=duration_s - YEAR_S / 12.0,
        dt_s=YEAR_S / 12.0,
        kappa=1.0e-6,
        sat=sat,
        n_winter=0.4,
    )
    no_snow = run_column_1d(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=duration_s - YEAR_S / 12.0,
        dt_s=YEAR_S / 12.0,
        kappa=1.0e-6,
        sat=sat,
        n_winter=1.0,
    )
    # Snow case must be warmer on average than the no-snow case (winter
    # cooling is attenuated, summer warming is not).
    assert with_snow.T[:, 1:].mean() > no_snow.T[:, 1:].mean()


def test_solver_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        run_column_1d(depth_max_m=10.0, dz_m=-1.0, duration_s=1.0, dt_s=0.1, kappa=1e-6)
    with pytest.raises(ValueError, match="n_winter"):
        run_column_1d(
            depth_max_m=10.0, dz_m=1.0, duration_s=1.0, dt_s=0.1, kappa=1e-6, n_winter=1.5
        )
    with pytest.raises(ValueError, match="requires t > 0"):
        carslaw_jaeger_step_analytic(np.array([1.0]), np.array([0.0]), 1.0, 1e-6)
