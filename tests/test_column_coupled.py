"""Benchmarks for the coupled T-p solver (``column_coupled``).

The four reference benchmarks from the accompanying paper:

1. Uncoupled limit: ``gamma_n_alpha_scale = 0`` recovers the Carslaw-Jaeger
   step solution for pure conduction.
2. Terzaghi 1-D consolidation: ``alpha_w`` effectively decoupled by
   ``s = 0``; checked against the standard series solution.  The roadmap
   names the Mandel-Cryer benchmark; we substitute Terzaghi here because
   the genuine Mandel-Cryer overshoot is a 2-D effect (it requires
   mechanical equilibrium of the solid skeleton, not just 1-D pressure
   diffusion).  This is documented in ``column_coupled.py``'s limitations
   block.
3. Thermo-poroelastic coupling: in the undrained limit (vanishing
   ``K_zz``), the induced pressure tracks temperature at the
   linearised ratio ``alpha_w / beta_w``.
4. Coupling switch: ``gamma_n_alpha_scale = 0`` leaves the pressure
   field exactly equal to its initial / boundary state regardless of
   T evolution; this is the controlled null hypothesis that
   ``figA_anomalous_sites.py`` compares against.
"""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.solvers import (
    CoupledResult,
    carslaw_jaeger_step_analytic,
    run_column_coupled,
)

YEAR_S: float = 365.25 * 86400.0


def test_uncoupled_limit_matches_carslaw_jaeger() -> None:
    """With ``s = 0`` and a step in surface T, the solver must reproduce
    the half-space Carslaw-Jaeger erfc kernel to within 1e-3 K."""
    kappa = 1.0e-6
    rho_c_eff = 2.5e6
    lam_th = kappa * rho_c_eff
    duration_s = 100.0 * YEAR_S

    res = run_column_coupled(
        depth_max_m=500.0,
        dz_m=2.0,
        duration_s=duration_s,
        dt_s=YEAR_S / 4.0,
        lambda_thermal=lam_th,
        rho_c_eff=rho_c_eff,
        gamma_n_alpha_scale=0.0,
        sat=1.0,
        p_top=0.0,
    )
    assert isinstance(res, CoupledResult)
    T_theory = carslaw_jaeger_step_analytic(res.z, np.array([duration_s]), 1.0, kappa)[0]
    err = np.abs(res.T[-1] - T_theory).max()
    assert err < 1.0e-3, f"max |T_num - T_theory| = {err:.3e} K"


def test_terzaghi_1d_consolidation() -> None:
    """1-D pressure diffusion in a column of length L with a top
    Dirichlet drainage boundary (``p_top = 0``) and bottom hydrostatic-
    Neumann (zero net flux when ``g = 0``).  Compare against the series

        p(z, t)/p_0 = sum_{m=0}^infinity  (4/((2m+1) pi))
                       * sin((2m+1) pi z / (2 L))
                       * exp(-(2m+1)^2 pi^2 c_v t / (4 L^2))

    with the hydraulic diffusivity ``c_v = K_zz / (mu rho_w phi beta_w)``
    that the solver's mass equation defines.  Setting ``g = 0`` cleanly
    isolates the consolidation problem from the hydrostatic body force.
    """
    beta_w = 4.5e-10
    mu = 1.0e-3
    phi = 0.15
    K_zz = 1.0e-13
    rho_w = 1000.0
    L = 100.0
    p0 = 1.0e5

    c_v = K_zz / (mu * phi * beta_w)
    char_t = L * L / c_v
    dt_s = char_t / 1000.0
    duration_s = 0.3 * char_t

    res = run_column_coupled(
        depth_max_m=L,
        dz_m=2.0,
        duration_s=duration_s,
        dt_s=dt_s,
        K_zz=K_zz,
        mu=mu,
        porosity=phi,
        beta_w=beta_w,
        rho_w=rho_w,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        sat=0.0,
        p_top=0.0,
        p_init=p0,
    )

    T_dim_target = 0.2
    n_idx = int(round(T_dim_target * char_t / dt_s))
    T_dim = c_v * res.t[n_idx] / (L * L)
    z = res.z

    p_an = np.zeros_like(z)
    for m in range(200):
        n = 2 * m + 1
        p_an += (
            (4.0 / (n * np.pi))
            * np.sin(n * np.pi * z / (2.0 * L))
            * np.exp(-((n * np.pi / 2.0) ** 2) * T_dim)
        )
    p_an *= p0

    err = np.abs(res.p[n_idx] - p_an) / p0
    assert err.max() < 5.0e-4, f"max rel err vs Terzaghi = {err.max():.3e}"


def test_thermo_poroelastic_undrained_ratio() -> None:
    """Undrained-limit consistency: with ``K_zz`` so small that pressure
    barely diffuses on the T-diffusion timescale, the linearised
    relation ``dp/dT = alpha_w / beta_w`` must hold in the interior.

    Reference: Detournay & Cheng (1993), Eq. 5.10 (linearised
    thermo-poroelastic constitutive equations) in the undrained limit.
    """
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
        K_zz=1.0e-23,  # essentially impermeable at the corrected c_v=K/(mu phi beta)
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
    # Probe a deep node where surface drainage to p_top hasn't reached yet.
    i_mid = 30  # 60 m depth
    T_mid = res.T[-1, i_mid]
    p_mid = res.p[-1, i_mid]
    expected_ratio = alpha_w / beta_w
    assert T_mid > 0.05, "T at probe node should have warmed measurably"
    rel_err = abs(p_mid / T_mid - expected_ratio) / expected_ratio
    assert rel_err < 5.0e-3, (
        f"p/T = {p_mid / T_mid:.0f} vs alpha/beta = {expected_ratio:.0f} (rel err {rel_err:.3e})"
    )


def test_coupling_switch_isolates_pressure_response() -> None:
    """With ``gamma_n_alpha_scale = 0``, applying a surface temperature
    step must leave the pressure field at its initial / BC state
    everywhere -- this is the null model that the coupled curve in
    ``figA_anomalous_sites.py`` is compared against."""
    res = run_column_coupled(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        K_zz=1.0e-20,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        sat=1.0,
        p_top=0.0,
        p_init=0.0,
    )
    # Pressure should remain at 0 everywhere despite T = 1 K applied at top.
    assert np.max(np.abs(res.p)) < 1.0e-9, (
        f"p drift with s=0 was {np.max(np.abs(res.p)):.3e} Pa, expected ~0"
    )


def test_coupled_solver_rejects_bad_inputs() -> None:
    base_kwargs = dict(
        depth_max_m=10.0,
        dz_m=0.5,
        duration_s=YEAR_S,
        dt_s=YEAR_S / 100.0,
    )
    with pytest.raises(ValueError, match="grid/time"):
        run_column_coupled(**{**base_kwargs, "depth_max_m": -1.0})
    with pytest.raises(ValueError, match="porosity"):
        run_column_coupled(**base_kwargs, porosity=1.5)
    with pytest.raises(ValueError, match="K_zz"):
        run_column_coupled(**base_kwargs, K_zz=0.0)
    with pytest.raises(ValueError, match="beta_w"):
        run_column_coupled(**base_kwargs, beta_w=0.0)
    with pytest.raises(ValueError, match="bot_p_bc"):
        run_column_coupled(**base_kwargs, bot_p_bc="banana")
