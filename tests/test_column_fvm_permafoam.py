"""Tests for the finite-volume PermaFoam-style coupled solver
(``column_fvm_permafoam``).

These cover the four verification steps that the accompanying paper
(Section 9) prescribes for the scheme -- (1) pure conduction vs.
Carslaw-Jaeger, (2) one-phase Stefan freezing front, (3) pressure
diffusion vs. Theis, (4) undrained thermo-poroelastic ratio -- plus a
cross-verification against the independent Crank-Nicolson solver and
the usual robustness/finiteness smoke checks.

Note on the hydraulic diffusivity.  This solver implements the
accompanying paper's
dimensionally consistent *mass* form (mass storage ``rho_w phi beta_w``
paired with mass flux ``rho_w k_rel K/mu``), so its hydraulic
diffusivity is ``c_v = K / (mu phi beta_w)`` -- the textbook value with
``rho_w`` cancelling.  The repo helper
``benchmarks.theis.hydraulic_diffusivity`` returns
``K / (mu rho_w phi beta_w)`` (an extra ``rho_w``), which is what the
Crank-Nicolson solver uses; the two differ by a factor ``rho_w``.  The
Theis test below therefore builds the *physical* diffusivity directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.carslaw_jaeger import step_temperature_response
from gt_theory.benchmarks.ogata_banks import ogata_banks_response
from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
)
from gt_theory.benchmarks.theis import step_pressure_response
from gt_theory.solvers.column_fvm_permafoam import run_column_fvm_permafoam

YEAR_S = 365.25 * 86400.0


# ----------------------------------------------------------------------
# Accompanying paper Section 9 step 1 -- pure conduction
# ----------------------------------------------------------------------
def test_conduction_vs_carslaw_jaeger() -> None:
    """No flow, no phase change: the FVM solver must reproduce the
    Carslaw-Jaeger erfc step response to the truncation order of the
    scheme."""
    phi = 0.15
    lam = 2.5
    rho_r = 2700.0
    # sensible volumetric capacity tuned to a round 2.5e6 J m^-3 K^-1
    c_r = (2.5e6 - phi * 1000.0 * 4186.0) / ((1.0 - phi) * rho_r)
    kappa = lam / 2.5e6
    dur = 5.0 * YEAR_S

    res = run_column_fvm_permafoam(
        depth_max_m=80.0,
        dz_m=0.5,
        duration_s=dur,
        dt_s=dur / 2000.0,
        porosity=phi,
        lambda_r=lam,
        lambda_w=lam,
        lambda_i=lam,
        rho_r=rho_r,
        c_r=c_r,
        rho_w=1000.0,
        c_w=4186.0,
        K_zz=1.0e-22,
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=10.0,
        T_init=0.0,
        p_top=0.0,
        q_bot=0.0,
        g=0.0,
        picard_max_iter=20,
    )
    T_ana = step_temperature_response(res.z, dur, delta_T=10.0, kappa=kappa)
    err = float(np.max(np.abs(res.T[-1] - T_ana)))
    assert err < 0.1, f"max |T_num - T_ana| = {err:.4f} K exceeds 0.1 K"
    assert np.all(np.isfinite(res.T))


# ----------------------------------------------------------------------
# Accompanying paper Section 9 step 2 -- one-phase Stefan freezing front
# ----------------------------------------------------------------------
def test_stefan_front_recovery() -> None:
    """With uniform sensible properties and the mass equation decoupled
    (K_zz -> 0), the apparent-heat-capacity FVM solver reproduces the
    analytical Neumann front position xi(t)."""
    phi = 0.30
    rho_r, c_r = 2700.0, 800.0
    rho_w = 1000.0
    c_uniform = 4186.0
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
    dur = 0.5 * YEAR_S
    res = run_column_fvm_permafoam(
        depth_max_m=4.0,
        dz_m=0.05,
        duration_s=dur,
        dt_s=dur / 500.0,
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
        g=0.0,
        picard_max_iter=20,
    )
    S_final = res.S_i[-1]
    frozen = S_final > 0.5
    assert frozen.any(), "no frozen cells at final time"
    xi_solver = float(res.z[frozen][-1])
    xi_ana = float(stefan_front_position(dur, p_stefan))
    rel_err = abs(xi_solver - xi_ana) / xi_ana
    assert rel_err < 0.10, (
        f"Stefan front mismatch: solver={xi_solver:.3f} m, "
        f"analytical={xi_ana:.3f} m, rel err={rel_err:.1%}"
    )


# ----------------------------------------------------------------------
# Accompanying paper Section 9 step 3 -- pressure diffusion vs Theis erfc
# ----------------------------------------------------------------------
def test_pressure_diffusion_vs_theis() -> None:
    """Mass-equation operator in isolation (no coupling, no gravity):
    a surface pressure step diffuses as the erfc similarity solution
    with the dimensionally consistent diffusivity c_v = K/(mu phi beta)."""
    K, mu, rho_w, phi, beta = 1.0e-13, 1.0e-3, 1000.0, 0.15, 4.5e-10
    c_v = K / (mu * phi * beta)  # physical (rho_w cancels)
    # keep the front well inside a 1000 m column: sqrt(c_v t) ~ 150 m
    dur = 150.0**2 / c_v
    res = run_column_fvm_permafoam(
        depth_max_m=1000.0,
        dz_m=5.0,
        duration_s=dur,
        dt_s=dur / 4000.0,
        porosity=phi,
        K_zz=K,
        mu=mu,
        beta_w=beta,
        rho_w=rho_w,
        g=0.0,
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=0.0,
        T_init=0.0,
        p_top=1.0e5,
        p_init=0.0,
        bot_p_bc="neumann_zero_gradient",
        picard_max_iter=20,
    )
    p_ana = step_pressure_response(res.z, dur, delta_p=1.0e5, c_v=c_v)
    err = float(np.max(np.abs(res.p[-1] - p_ana))) / 1.0e5
    assert err < 0.02, f"pressure profile RMS-max error {err:.4f} exceeds 2%"
    assert np.all(np.isfinite(res.p))


# ----------------------------------------------------------------------
# Accompanying paper Section 9 step 4 -- undrained thermo-poroelastic ratio
# ----------------------------------------------------------------------
def test_undrained_thermo_poroelastic_ratio() -> None:
    """In the impermeable limit the interior pressure response to a
    surface warming obeys dp/dT = +alpha_w/beta_w (heating pressurises
    a sealed column; Detournay & Cheng 1993)."""
    alpha_w, beta_w, rho_w = 2.1e-4, 4.5e-10, 1000.0
    res = run_column_fvm_permafoam(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        porosity=0.15,
        K_zz=1.0e-23,
        mu=1.0e-3,  # truly undrained for c_v=K/(mu phi beta)
        alpha_w=alpha_w,
        beta_w=beta_w,
        rho_w=rho_w,
        lambda_r=2.5,
        lambda_w=2.5,
        lambda_i=2.5,
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=1.0,
        sat=1.0,
        T_init=0.0,
        p_top=0.0,
        p_init=0.0,
        g=0.0,
        picard_max_iter=20,
    )
    i_mid = 30  # ~61 m depth
    T_mid = res.T[-1, i_mid]
    p_mid = res.p[-1, i_mid]
    expected = alpha_w / beta_w
    assert T_mid > 0.05, "probe should have warmed measurably"
    rel_err = abs(p_mid / T_mid - expected) / expected
    assert rel_err < 5.0e-3, (
        f"p/T = {p_mid / T_mid:.0f} vs alpha/beta = {expected:.0f} (rel err {rel_err:.3e})"
    )


def test_coupling_switch_isolates_pressure_response() -> None:
    """With gamma_n_alpha_scale = 0 a surface temperature step must leave
    the pressure field at its initial / BC state (the null model)."""
    res = run_column_fvm_permafoam(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        porosity=0.15,
        K_zz=1.0e-23,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        T_f=-100.0,
        dTc=0.5,
        sat=1.0,
        T_init=0.0,
        p_top=0.0,
        p_init=0.0,
        picard_max_iter=20,
    )
    assert np.max(np.abs(res.p)) < 1.0e-6, (
        f"p drift with gamma=0 was {np.max(np.abs(res.p)):.3e} Pa, expected ~0"
    )


# ----------------------------------------------------------------------
# Cross-verification: two independent schemes must agree
# ----------------------------------------------------------------------
def test_fvm_vs_cn_conduction_agreement() -> None:
    """The finite-volume backward-Euler solver and the finite-difference
    Crank-Nicolson solver, run on an identical pure-conduction
    configuration, must agree to well under 0.1 K despite using entirely
    different discretisations -- the gold-standard correctness check."""
    from gt_theory.solvers.column_thermo_freeze_coupled import (
        run_column_thermo_freeze_coupled,
    )

    phi = 0.15
    rho_r = 2700.0
    c_r = (2.5e6 - phi * 1000.0 * 4186.0) / ((1.0 - phi) * rho_r)
    dur = 5.0 * YEAR_S
    common = dict(
        depth_max_m=60.0,
        dz_m=0.5,
        duration_s=dur,
        dt_s=dur / 2000.0,
        porosity=phi,
        lambda_r=2.5,
        lambda_w=2.5,
        lambda_i=2.5,
        rho_r=rho_r,
        c_r=c_r,
        rho_w=1000.0,
        c_w=4186.0,
        K_zz=1.0e-22,
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=10.0,
        T_init=0.0,
        p_top=0.0,
        g=0.0,
        q_bot=0.0,
    )
    r_fvm = run_column_fvm_permafoam(**common)
    r_cn = run_column_thermo_freeze_coupled(**common)
    # CN is node-centred, FVM cell-centred: interpolate onto FVM centres.
    T_cn = np.interp(r_fvm.z, r_cn.z, r_cn.T[-1])
    max_diff = float(np.max(np.abs(r_fvm.T[-1] - T_cn)))
    assert max_diff < 0.1, f"FVM vs CN conduction disagree by {max_diff:.4f} K"


# ----------------------------------------------------------------------
# Robustness / finiteness
# ----------------------------------------------------------------------
def test_seasonal_freeze_thaw_stays_stable() -> None:
    """A boreal column driven through repeated freeze-thaw cycles must
    stay *stable*: T finite and physically bounded, p finite, S_i in
    [0, 1].  The backward-Euler + under-relaxation scheme is
    unconditionally stable, so the solution cannot blow up even when the
    Picard loop does not reach the tight (1e-4 K) tolerance within the
    cap -- which happens on steps where a node crosses the entire
    freezing interval in one step (the classic apparent-heat-capacity
    stagnation; the solution remains bounded and usable)."""
    nt = 240
    t = np.arange(nt + 1) * (YEAR_S / 24.0)
    sat = -2.0 + 12.0 * np.sin(2.0 * np.pi * t / YEAR_S)  # seasonal SAT
    res = run_column_fvm_permafoam(
        depth_max_m=20.0,
        dz_m=0.25,
        duration_s=nt * (YEAR_S / 24.0),
        dt_s=YEAR_S / 24.0,
        porosity=0.30,
        K_zz=1.0e-14,
        T_f=0.0,
        dTc=0.5,
        S_w_residual=0.05,
        gamma_n_alpha_scale=1.0,
        sat=sat,
        T_init=-2.0,
        p_top=0.0,
        q_bot=0.06,
        picard_max_iter=20,
        picard_omega=0.7,
    )
    assert np.all(np.isfinite(res.T))
    assert np.all(np.isfinite(res.p))
    assert np.all((res.S_i >= 0.0) & (res.S_i <= 1.0))
    # T cannot exceed the forcing envelope by more than a few K.
    assert res.T.min() > sat.min() - 5.0
    assert res.T.max() < sat.max() + 10.0
    assert res.picard_iters.max() <= 20


def test_dirichlet_bottom_pressure_is_fixed_level() -> None:
    """bot_p_bc='dirichlet' must hold the bottom pressure at the
    user-set constant ``p_bot`` (not the evolving last-step value).  With
    both boundaries pinned to the same level and no gravity, the steady
    state is a uniform pressure equal to that level."""
    pb = 5.0e5
    res = run_column_fvm_permafoam(
        depth_max_m=100.0,
        dz_m=2.0,
        duration_s=50.0 * YEAR_S,
        dt_s=YEAR_S,
        porosity=0.15,
        K_zz=1.0e-13,
        g=0.0,
        gamma_n_alpha_scale=0.0,
        T_f=-100.0,
        dTc=0.5,
        sat=0.0,
        T_init=0.0,
        p_top=pb,
        p_init=0.0,
        bot_p_bc="dirichlet",
        p_bot=pb,
        picard_max_iter=20,
    )
    assert np.allclose(res.p[-1], pb, rtol=2.0e-3), (
        f"bottom Dirichlet did not hold p={pb:.0f} Pa: "
        f"p[-1] range [{res.p[-1].min():.0f}, {res.p[-1].max():.0f}]"
    )


def test_fvm_advection_converges_to_ogata_banks() -> None:
    """Advection-dominated verification (Fig. 11).  A surface temperature
    step advected into a column at the geothermal regime's cell Peclet
    number must converge, under grid refinement, to the analytical
    Ogata-Banks advection-dispersion solution -- the benchmark the
    conduction tests above cannot reach.  The upwind finite-volume scheme
    is first-order accurate, so the L-infinity error must decrease
    monotonically at roughly first order, the imposed sustained Darcy
    velocity must be uniform at its target, and the field must stay
    monotone (no overshoot) at every resolution.
    """
    phi, lam = 0.02, 3.0
    rho_r, c_r = 2700.0, 1000.0
    rho_w, c_w, mu = 1000.0, 4186.0, 1.0e-3
    K_zz, v, dT = 1.0e-13, 1.0e-7, 10.0
    L, dur = 2000.0, 30.0 * YEAR_S
    rho_c = (1.0 - phi) * rho_r * c_r + phi * rho_w * c_w
    kappa = lam / rho_c
    v_T = rho_w * c_w * v / rho_c  # thermal front velocity
    dp = v * mu / K_zz * L  # head that drives v (g = 0)

    def run(dz: float):
        nz = int(round(L / dz))
        z = (np.arange(nz) + 0.5) * dz
        return run_column_fvm_permafoam(
            depth_max_m=L,
            dz_m=dz,
            duration_s=dur,
            dt_s=dur / 360.0,
            porosity=phi,
            lambda_r=lam,
            lambda_w=lam,
            lambda_i=lam,
            rho_r=rho_r,
            c_r=c_r,
            rho_w=rho_w,
            c_w=c_w,
            mu=mu,
            K_zz=K_zz,
            g=0.0,
            T_f=-100.0,
            dTc=0.5,
            gamma_n_alpha_scale=0.0,
            sat=dT,
            T_init=0.0,
            p_top=dp,
            p_init=dp * (1.0 - z / L),
            bot_p_bc="dirichlet",
            p_bot=0.0,
            q_bot=0.0,
            picard_max_iter=20,
        )

    errs = {}
    for dz in (40.0, 20.0, 10.0, 5.0):
        res = run(dz)
        assert np.all(np.isfinite(res.T)), f"non-finite T at dz={dz}"
        # the imposed sustained Darcy flux is uniform at its target
        assert abs(np.median(res.v_darcy[-1]) - v) / v < 0.05
        # upwind stays monotone -- no over/undershoot of the [0, dT] range
        assert res.T[-1].max() <= dT + 1.0e-6
        assert res.T[-1].min() >= -1.0e-6
        ob = ogata_banks_response(res.z, dur, delta_T=dT, v_T=v_T, kappa=kappa)
        interior = res.z < 0.5 * L
        errs[dz] = float(np.max(np.abs(res.T[-1][interior] - ob[interior])))

    # monotone error reduction under refinement
    assert errs[40.0] > errs[20.0] > errs[10.0] > errs[5.0], errs
    # ~first-order convergence on the finest resolved pair
    rate = np.log(errs[10.0] / errs[5.0]) / np.log(2.0)
    assert 0.5 < rate < 1.5, f"convergence rate {rate:.2f} outside [0.5, 1.5]"


def test_rejects_bad_inputs() -> None:
    base = dict(depth_max_m=10.0, dz_m=1.0, duration_s=YEAR_S, dt_s=YEAR_S / 10.0)
    with pytest.raises(ValueError, match="porosity"):
        run_column_fvm_permafoam(**{**base, "porosity": 1.5})
    with pytest.raises(ValueError, match="K_zz"):
        run_column_fvm_permafoam(**{**base, "K_zz": -1.0})
    with pytest.raises(ValueError, match="bot_p_bc"):
        run_column_fvm_permafoam(**{**base, "bot_p_bc": "nonsense"})
    with pytest.raises(ValueError, match="picard_omega"):
        run_column_fvm_permafoam(**{**base, "picard_omega": 1.5})
