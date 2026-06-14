"""Stefan-problem benchmark for the apparent-heat-capacity enthalpy
solver.

Constructs the classical 1-D semi-infinite freezing problem (warm
initial column at +T_hot, surface dropped to -T_cold) and checks that
the freezing-front position at 1 yr matches the Neumann analytic
solution within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.solvers import (
    EnthalpyResult,
    neumann_stefan_lambda,
    run_column_enthalpy,
)
from gt_theory.solvers.column_enthalpy import _apparent_capacity, _ice_saturation

YEAR_S: float = 365.25 * 86400.0


def _front_depth_from_T_profile(z: np.ndarray, T: np.ndarray) -> float | None:
    """Linear-interpolated depth of the 0 deg C isotherm, or None."""
    cross = np.where(np.diff(np.sign(T)))[0]
    if cross.size == 0:
        return None
    i = int(cross[0])
    denom = T[i + 1] - T[i]
    if denom == 0:
        return float(z[i])
    frac = -T[i] / denom
    return float(z[i] + frac * (z[i + 1] - z[i]))


@pytest.fixture(scope="module")
def stefan_setup() -> dict:
    c_solid = 2.5e6
    phi = 0.30
    L_f = 3.34e5
    rho_w = 1000.0
    lam_th = 2.5
    dT_hot_cold = 5.0
    alpha = lam_th / c_solid
    St = c_solid * dT_hot_cold / (L_f * rho_w * phi)
    lam = neumann_stefan_lambda(St)
    x_theory = 2.0 * lam * np.sqrt(alpha * YEAR_S)
    return dict(
        c_solid=c_solid,
        phi=phi,
        L_f=L_f,
        rho_w=rho_w,
        lam_th=lam_th,
        dT_hot_cold=dT_hot_cold,
        St=St,
        lam=lam,
        x_theory=x_theory,
    )


def test_neumann_root_solver_recovers_lambda() -> None:
    """Spot-check the bisection against a tabulated value: Stefan = 1
    has lambda ~ 0.6204 (Bonacina 1973)."""
    lam = neumann_stefan_lambda(1.0)
    assert lam == pytest.approx(0.6204, abs=2e-3)


def test_ice_saturation_piecewise_linear() -> None:
    T = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    Si = _ice_saturation(T, T_f=0.0, dTc=1.0)
    np.testing.assert_allclose(Si, [1.0, 0.5, 0.0, 0.0, 0.0])


def test_apparent_capacity_spike_inside_freezing_interval() -> None:
    T = np.array([-3.0, -0.5, 0.5, 3.0])
    c = _apparent_capacity(
        T,
        T_f=0.0,
        dTc=1.0,
        rho_c_solid=2.5e6,
        L_f=3.34e5,
        rho_w=1000.0,
        phi=0.30,
    )
    # Outside the interval (T = -3, +0.5, +3) -> just c_solid.
    assert c[0] == pytest.approx(2.5e6)
    assert c[2] == pytest.approx(2.5e6)
    assert c[3] == pytest.approx(2.5e6)
    # Inside the interval (T = -0.5) -> c_solid + L_f rho_w phi / dTc.
    expected = 2.5e6 + 3.34e5 * 1000.0 * 0.30 / 1.0
    assert c[1] == pytest.approx(expected, rel=1e-6)


def test_enthalpy_solver_matches_stefan_at_dtc_1K(stefan_setup: dict) -> None:
    """For dTc = 1 K (broad freezing interval typical of fine soils),
    the freezing front at 1 yr should match the Neumann analytic to
    within 10%."""
    s = stefan_setup
    res = run_column_enthalpy(
        depth_max_m=15.0,
        dz_m=0.25,
        duration_s=2.0 * YEAR_S,
        dt_s=YEAR_S / 300.0,
        lambda_thermal=s["lam_th"],
        rho_c_solid=s["c_solid"],
        porosity=s["phi"],
        dTc=1.0,
        sat=-s["dT_hot_cold"],
        T_init=+s["dT_hot_cold"],
    )
    assert isinstance(res, EnthalpyResult)
    t_idx = res.t.size // 2  # ~ 1 yr
    x_num = _front_depth_from_T_profile(res.z, res.T[t_idx])
    assert x_num is not None
    err_pct = 100.0 * abs(x_num - s["x_theory"]) / s["x_theory"]
    assert err_pct < 10.0, (
        f"x_num={x_num:.3f} m vs theory {s['x_theory']:.3f} m  err={err_pct:.1f}%"
    )


def test_enthalpy_solver_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="grid/time"):
        run_column_enthalpy(depth_max_m=-1.0, dz_m=0.5, duration_s=YEAR_S, dt_s=YEAR_S / 100)
    with pytest.raises(ValueError, match="porosity"):
        run_column_enthalpy(
            depth_max_m=10.0, dz_m=0.5, duration_s=YEAR_S, dt_s=YEAR_S / 100, porosity=1.5
        )
    with pytest.raises(ValueError, match="dTc"):
        run_column_enthalpy(
            depth_max_m=10.0, dz_m=0.5, duration_s=YEAR_S, dt_s=YEAR_S / 100, dTc=0.0
        )
