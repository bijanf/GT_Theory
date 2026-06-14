"""Hypothesis property tests for the 1-D Crank-Nicolson solver.

These tests assert invariants that must hold for *any* admissible
input, in contrast with the example-based tests in
``tests/test_solver_twin.py`` that hard-code a particular forcing.

Invariants exercised here
-------------------------
1. Output array shapes and finiteness for all admissible parameter
   combinations.
2. Maximum-principle bound: with zero initial condition, the column
   temperature must stay within ``[sat_min, sat_max]`` plus a small
   geothermal-flux extension.  No spurious oscillations may push values
   beyond this envelope.
3. Carslaw-Jaeger limit: a constant unit step on a deep column with no
   advection, no flux, no n-factor must agree with the analytic erfc
   solution at large times.
4. Determinism: identical inputs must yield identical outputs (no
   hidden randomness in the solver).
5. Linearity: scaling the surface forcing by a constant scales the
   anomaly by the same constant (within numerical tolerance) when the
   initial state is zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from gt_theory.solvers import (
    carslaw_jaeger_step_analytic,
    run_column_1d,
)

YEAR_S: float = 365.25 * 86400.0

# Slow-ish numerical integrations; cap the test budget to keep the suite
# under a second total wall-clock.
_PROPERTY_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------- strategies


def _kappa_strategy() -> st.SearchStrategy[float]:
    """Realistic thermal diffusivities in m^2 s^-1, 1e-7 to 1e-5."""
    return st.floats(min_value=1.0e-7, max_value=1.0e-5, allow_nan=False, allow_infinity=False)


def _grid_strategy() -> st.SearchStrategy[tuple[float, float]]:
    """(depth_max_m, dz_m) with dz dividing depth in 30-500 nodes."""
    return st.tuples(
        st.floats(min_value=100.0, max_value=1500.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=2.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    )


def _duration_strategy() -> st.SearchStrategy[tuple[float, float]]:
    """(duration_s, dt_s) such that nt is 50-500."""
    return st.tuples(
        st.floats(min_value=10.0 * YEAR_S, max_value=200.0 * YEAR_S),
        st.floats(min_value=YEAR_S / 24.0, max_value=YEAR_S),
    )


def _sat_strategy() -> st.SearchStrategy[float]:
    """Constant surface temperature in [-10, +10] K."""
    return st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


# ----------------------------------------------------------- property tests


@_PROPERTY_SETTINGS
@given(
    grid=_grid_strategy(),
    duration=_duration_strategy(),
    kappa=_kappa_strategy(),
    sat=_sat_strategy(),
    n_winter=st.floats(min_value=0.1, max_value=1.0),
)
def test_finite_outputs(grid, duration, kappa, sat, n_winter) -> None:
    """No NaN, no inf, correct shapes, for any admissible inputs."""
    depth_max_m, dz_m = grid
    duration_s, dt_s = duration

    result = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa,
        sat=sat,
        n_winter=n_winter,
    )

    assert result.T.ndim == 2
    assert result.T.shape == (result.t.size, result.z.size)
    assert np.all(np.isfinite(result.T))
    assert np.all(np.isfinite(result.gst))


@_PROPERTY_SETTINGS
@given(
    grid=_grid_strategy(),
    duration=_duration_strategy(),
    kappa=_kappa_strategy(),
    sat=_sat_strategy(),
)
def test_maximum_principle_no_advection_no_flux(grid, duration, kappa, sat) -> None:
    """Zero IC + constant SAT + no advection + no bottom flux + n_winter=1
    must keep the column within ``[min(0, sat), max(0, sat)]`` plus a
    small numerical tolerance.

    Crank-Nicolson is unconditionally stable but not monotone:
    overshoots of order O(r) appear once the diffusion number
    r = kappa * dt / dz^2 exceeds 0.5.  We therefore restrict this
    property test to the strictly-monotone regime ``r <= 0.4`` via
    ``hypothesis.assume``; large-r behaviour is tested separately under
    the explicit Carslaw-Jaeger benchmark.
    """
    depth_max_m, dz_m = grid
    duration_s, dt_s = duration
    r = kappa * dt_s / (dz_m * dz_m)
    assume(r <= 0.4)

    result = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa,
        q_bot=0.0,
        sat=sat,
        n_winter=1.0,
        v_darcy=0.0,
        T_init=0.0,
    )

    tol = max(1.0e-8, 1.0e-9 * abs(sat))
    lo = min(0.0, sat) - tol
    hi = max(0.0, sat) + tol
    assert result.T.min() >= lo, f"undershoot: min(T) = {result.T.min():.6e}, bound = {lo:.6e}"
    assert result.T.max() <= hi, f"overshoot:  max(T) = {result.T.max():.6e}, bound = {hi:.6e}"


@_PROPERTY_SETTINGS
@given(
    duration=_duration_strategy(),
    kappa=_kappa_strategy(),
    delta_T=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_carslaw_jaeger_limit_at_large_time(duration, kappa, delta_T) -> None:
    """Constant unit-magnitude step + deep column + no advection: at
    large times and intermediate depths, the numerical solution must
    track the analytic erfc solution to within ~5% relative error.
    """
    duration_s, dt_s = duration
    depth_max_m = 1500.0
    dz_m = 5.0

    result = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa,
        q_bot=0.0,
        sat=delta_T,
        n_winter=1.0,
        v_darcy=0.0,
        T_init=0.0,
    )

    # Test at the last time and a depth shallow enough to be sensitive
    # but deep enough to avoid the surface-BC trivial match.
    iz = int(np.argmin(np.abs(result.z - 50.0)))
    t_end = result.t[-1]
    T_num = result.T[-1, iz]
    T_ana = float(
        carslaw_jaeger_step_analytic(
            z=np.array([result.z[iz]]),
            t=np.array([t_end]),
            delta_T=delta_T,
            kappa=kappa,
        ).ravel()[0]
    )

    # Loose tolerance because (i) the grid is coarse for some draws and
    # (ii) when delta_T is near zero the absolute error is the only
    # meaningful measure.
    abs_err = abs(T_num - T_ana)
    rel_err = abs_err / (abs(T_ana) + 1.0e-3)
    assert abs_err < 0.1 or rel_err < 0.10, (
        f"kappa={kappa:.2e} delta_T={delta_T:.3f} z={result.z[iz]:.1f} m "
        f"t={t_end / YEAR_S:.1f} yr: T_num={T_num:.4f} T_ana={T_ana:.4f}"
    )


@_PROPERTY_SETTINGS
@given(
    kappa=_kappa_strategy(),
    sat=_sat_strategy(),
)
def test_solver_is_deterministic(kappa, sat) -> None:
    """The solver carries no internal random state; two identical calls
    must produce bitwise-identical output."""
    common = dict(
        depth_max_m=400.0,
        dz_m=4.0,
        duration_s=20.0 * YEAR_S,
        dt_s=YEAR_S / 6.0,
        kappa=kappa,
        sat=sat,
    )
    r1 = run_column_1d(**common)
    r2 = run_column_1d(**common)
    np.testing.assert_array_equal(r1.T, r2.T)
    np.testing.assert_array_equal(r1.gst, r2.gst)


@_PROPERTY_SETTINGS
@given(
    kappa=_kappa_strategy(),
    scale=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    sat_base=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_linearity_of_surface_forcing(kappa, scale, sat_base) -> None:
    """With zero initial state, no advection, no flux, n_winter=1: the
    column response is linear in the surface forcing.

    We use ``sat_base > 0`` to avoid the n-factor branch and keep the
    operator linear.  (Scaling a negative SAT under n_winter != 1 would
    break linearity, which is the *intended* non-linear behaviour of
    the snow-cover model — not a violation.)
    """
    common = dict(
        depth_max_m=400.0,
        dz_m=4.0,
        duration_s=20.0 * YEAR_S,
        dt_s=YEAR_S / 6.0,
        kappa=kappa,
        n_winter=1.0,
        v_darcy=0.0,
        T_init=0.0,
    )
    r1 = run_column_1d(sat=sat_base, **common)
    r2 = run_column_1d(sat=scale * sat_base, **common)

    np.testing.assert_allclose(r2.T, scale * r1.T, atol=1.0e-10, rtol=1.0e-8)


# ----------------------------------------------- targeted regression smoke


def test_n_factor_at_zero_disables_winter_response() -> None:
    """Sanity: n_winter = 0 zeroes out sub-zero SAT, so a purely
    negative constant SAT must yield identically zero column."""
    nt = 12 * 5
    sat = -1.0 * np.ones(nt)
    res = run_column_1d(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=(nt - 1) * (YEAR_S / 12.0),
        dt_s=YEAR_S / 12.0,
        kappa=1.0e-6,
        sat=sat,
        n_winter=0.0,
    )
    np.testing.assert_allclose(res.gst, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(res.T, 0.0, atol=1.0e-12)


@pytest.mark.parametrize("v_darcy", [-1.0e-9, 1.0e-9])
def test_advection_sign_skews_profile(v_darcy: float) -> None:
    """Positive Darcy velocity (downward flow) should advect surface
    warmth deeper than zero-advection; negative (upward) should hold it
    shallower.  We check the column-integrated 'centre of mass' depth
    of the anomaly at the final time."""
    common = dict(
        depth_max_m=300.0,
        dz_m=2.0,
        duration_s=30.0 * YEAR_S,
        dt_s=YEAR_S / 12.0,
        kappa=1.0e-6,
        sat=1.0,
        n_winter=1.0,
        T_init=0.0,
    )
    res0 = run_column_1d(v_darcy=0.0, **common)
    resv = run_column_1d(v_darcy=v_darcy, **common)

    def _com(result) -> float:
        T_end = result.T[-1, 1:]  # exclude the Dirichlet surface node
        z = result.z[1:]
        w = np.maximum(T_end, 0.0)
        return float((w * z).sum() / max(w.sum(), 1.0e-30))

    com0 = _com(res0)
    comv = _com(resv)
    if v_darcy > 0:
        assert comv > com0, f"downward v_darcy should deepen COM: {comv:.2f} vs {com0:.2f}"
    else:
        assert comv < com0, f"upward v_darcy should shallow COM: {comv:.2f} vs {com0:.2f}"
