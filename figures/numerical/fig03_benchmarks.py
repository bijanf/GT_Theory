#!/usr/bin/env python3
"""Figure 3 -- six analytical-benchmark verifications of the primary
1-D coupled finite-volume solver (``column_fvm_permafoam``).

Panels:
  a -- Carslaw-Jaeger step temperature response (erfc kernel)
  b -- Stefan one-phase Neumann freezing-front position xi(t)
  c -- Theis 1-D pressure-diffusion step (erfc analog)
  d -- Terzaghi 1-D consolidation degree U(T_v)
  e -- Undrained thermo-poroelastic ratio dp/dT = alpha_w/beta_w
  f -- Bonacina enthalpy-budget closure under a freeze-thaw cycle

Each panel overlays the solver result on the analytical solution.
Output: ``outputs/figures/numerical/fig03_benchmarks.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.benchmarks.bonacina import column_integrated_enthalpy
from gt_theory.benchmarks.carslaw_jaeger import step_temperature_response
from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
)
from gt_theory.benchmarks.terzaghi import degree_of_consolidation
from gt_theory.benchmarks.theis import (
    hydraulic_diffusivity,
    step_pressure_response,
)
from gt_theory.benchmarks.undrained_ratio import expected_dp_dT
from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)
from gt_theory.solvers.column_fvm_permafoam import (
    run_column_fvm_permafoam,
)

YEAR_S = 365.25 * 86400.0
ANALYTIC = "black"
NUMERIC = "#d62728"


def _panel_a_carslaw(ax) -> None:
    kappa = 1.0e-6
    phi = 0.10
    rho_r, c_r = 2700.0, 800.0
    rho_w, c_w = 1000.0, 4186.0
    rho_c_eff = (1 - phi) * rho_r * c_r + phi * rho_w * c_w
    lam = kappa * rho_c_eff
    res = run_column_fvm_permafoam(
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
        dTc=0.1,
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
    ax.plot(T_an, res.z, "-", color=ANALYTIC, linewidth=1.4, label="Carslaw-Jaeger erfc")
    ax.plot(res.T[-1], res.z, "o", color=NUMERIC, markersize=3, markevery=5, label="solver")
    ax.invert_yaxis()
    ax.set_xlabel(r"$T(z, t = 100 \mathrm{yr})$  (K)")
    ax.set_ylabel("depth (m)")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.03, 0.97, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def _panel_b_stefan(ax) -> None:
    phi = 0.30
    rho_r, c_r = 2700.0, 800.0
    rho_w, c_w = 1000.0, 4186.0
    rho_c_eff = (1 - phi) * rho_r * c_r + phi * rho_w * c_w
    p = StefanOnePhaseParams(
        T_s=-10.0,
        T_f=0.0,
        lambda_thermal=2.5,
        rho_c_solid=rho_c_eff,
        porosity=phi,
        L_f=3.34e5,
        rho_w=rho_w,
    )
    t_grid = np.linspace(0.05, 0.5, 10) * YEAR_S
    xi_an = stefan_front_position(t_grid, p)
    xi_solver = []
    for t in t_grid:
        res = run_column_fvm_permafoam(
            depth_max_m=4.0,
            dz_m=0.05,
            duration_s=float(t),
            dt_s=float(t) / 400.0,
            porosity=phi,
            lambda_r=2.5,
            lambda_w=2.5,
            lambda_i=2.5,
            rho_r=rho_r,
            c_r=c_r,
            rho_w=rho_w,
            c_w=c_w,
            c_i=c_w * rho_w / 917.0,
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
        S = res.S_i[-1]
        frozen = S > 0.5
        xi_solver.append(float(res.z[frozen][-1]) if frozen.any() else 0.0)
    ax.plot(t_grid / YEAR_S, xi_an, "-", color=ANALYTIC, linewidth=1.4, label="Neumann similarity")
    ax.plot(t_grid / YEAR_S, xi_solver, "o", color=NUMERIC, markersize=4, label="solver")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(r"freezing front $\xi(t)$ (m)")
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.03, 0.97, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def _panel_c_theis(ax) -> None:
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
    res = run_column_fvm_permafoam(
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
        T_f=-100.0,
        dTc=0.5,
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
    ax.plot(p_an / dp, res.z, "-", color=ANALYTIC, linewidth=1.4, label="Theis 1-D analog")
    ax.plot(res.p[-1] / dp, res.z, "o", color=NUMERIC, markersize=3, markevery=5, label="solver")
    ax.invert_yaxis()
    ax.set_xlabel(r"$p(z, t) / \Delta p$")
    ax.set_ylabel("depth (m)")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.03, 0.97, "c", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def _panel_d_terzaghi(ax) -> None:
    # Run the solver under a step initial p_0; degree of consolidation
    # is U(T_v) = 1 - <p(T_v)>/<p(0)>.
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
    duration_s = 1.5 * char_t
    res = run_column_fvm_permafoam(
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
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=0.0,
        p_top=0.0,
        p_init=p0,
    )
    T_v_solver = c_v * res.t / (L * L)
    # Numerical U(T_v): integrate p(z) / (p_0 * L).
    p_int = np.trapezoid(res.p, res.z, axis=1)
    U_solver = 1.0 - p_int / (p0 * L)
    T_v_grid = np.linspace(0.01, 1.5, 60)
    U_an = degree_of_consolidation(T_v_grid)
    ax.plot(T_v_grid, U_an, "-", color=ANALYTIC, linewidth=1.4, label="Terzaghi series")
    ax.plot(T_v_solver[::20], U_solver[::20], "o", color=NUMERIC, markersize=3.5, label="solver")
    ax.set_xlabel(r"dimensionless time  $T_v$")
    ax.set_ylabel(r"degree of consolidation  $U$")
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.03, 0.97, "d", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def _panel_e_undrained(ax) -> None:
    alpha_w = 2.1e-4
    beta_w = 4.5e-10
    expected = expected_dp_dT(alpha_w=alpha_w, beta_w=beta_w)
    res = run_column_fvm_permafoam(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        # (rho c)_eff = (1-phi) rho_r c_r + phi rho_w c_w ~ 2.5e6 at phi=0.15;
        # lambda uniform 2.5 so the (constant-T) conduction is immaterial.
        lambda_r=2.5,
        lambda_w=2.5,
        lambda_i=2.5,
        rho_r=2700.0,
        c_r=815.0,
        T_f=-100.0,
        dTc=0.5,
        # impermeable at the corrected c_v = K/(mu phi beta) so the 60 m
        # probe stays in the undrained limit over 100 yr.
        K_zz=1.0e-23,
        mu=1.0e-3,
        porosity=0.15,
        alpha_w=alpha_w,
        beta_w=beta_w,
        rho_w=1000.0,
        g=0.0,
        gamma_n_alpha_scale=1.0,
        sat=1.0,
        p_top=0.0,
    )
    # Use a probe depth at 60 m where surface drainage hasn't reached.
    i_probe = 30
    T_t = res.T[:, i_probe]
    p_t = res.p[:, i_probe]
    mask = T_t > 0.03
    ax.plot(
        T_t[mask],
        p_t[mask],
        "o",
        color=NUMERIC,
        markersize=2.5,
        label=f"solver  ($z = {res.z[i_probe]:.0f}\\,$m)",
    )
    T_line = np.linspace(0, 1.0, 5)
    ax.plot(
        T_line,
        expected * T_line,
        "-",
        color=ANALYTIC,
        linewidth=1.4,
        label=rf"$\alpha_w / \beta_w = {expected:.2e}$ Pa K$^{{-1}}$",
    )
    ax.set_xlabel(r"$T(z_{\rm probe}, t)$  (K)")
    ax.set_ylabel(r"$p(z_{\rm probe}, t)$  (Pa)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.text(0.03, 0.97, "e", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def _panel_f_bonacina(ax) -> None:
    # Drive a column through a half-cycle of seasonal freezing and
    # plot the column-integrated H(t).  Closure: change in H should
    # equal the net boundary heat input.
    nt = 121
    duration = 4.0 * YEAR_S
    dt = duration / (nt - 1)
    t = np.arange(nt) * dt
    sat_series = 6.0 * np.sin(2.0 * np.pi * t / YEAR_S)
    phi = 0.30
    res = run_column_fvm_permafoam(
        depth_max_m=10.0,
        dz_m=0.25,
        duration_s=duration,
        dt_s=dt,
        porosity=phi,
        lambda_r=2.5,
        lambda_w=0.58,
        lambda_i=2.22,
        rho_r=2700.0,
        c_r=800.0,
        rho_w=1000.0,
        c_w=4186.0,
        c_i=2108.0,
        L_f=3.34e5,
        K_zz=1.0e-15,
        gamma_n_alpha_scale=0.0,
        T_f=0.0,
        dTc=1.0,
        sat=sat_series,
        T_init=0.0,
        picard_max_iter=20,
        picard_tol_K=1e-3,
        picard_omega=0.7,
    )
    props = dict(
        porosity=phi,
        rho_r=2700.0,
        c_r=800.0,
        rho_w=1000.0,
        c_w=4186.0,
        rho_i=917.0,
        c_i=2108.0,
        L_f=3.34e5,
    )
    H_int = np.array(
        [
            column_integrated_enthalpy(res.T[k], res.S_i[k], res.z, T_ref=0.0, **props)
            for k in range(nt)
        ]
    )
    # Normalise so first sample is zero (we're tracking changes).
    H_int -= H_int[0]
    ax.plot(
        t / YEAR_S,
        H_int / 1e9,
        "-",
        color=NUMERIC,
        linewidth=1.0,
        label=r"$\Delta \int H dz$ (solver)",
    )
    # Reference: scale of latent heat per unit area.
    L_v_per_m = phi * 1000.0 * 3.34e5
    ax.axhline(
        L_v_per_m * res.z[-1] / 1e9,
        color=ANALYTIC,
        linestyle="--",
        linewidth=0.8,
        label=r"latent-heat scale  $\rho_w L_f \phi L_z$",
    )
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(r"$\Delta \int H dz$  (GJ m$^{-2}$)")
    ax.legend(loc="center", frameon=False, fontsize=6)
    ax.text(0.03, 0.97, "f", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig03_benchmarks.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.62),
    )
    print("  panel a (Carslaw-Jaeger)...")
    _panel_a_carslaw(axes[0, 0])
    print("  panel b (Stefan)...")
    _panel_b_stefan(axes[0, 1])
    print("  panel c (Theis)...")
    _panel_c_theis(axes[0, 2])
    print("  panel d (Terzaghi)...")
    _panel_d_terzaghi(axes[1, 0])
    print("  panel e (undrained ratio)...")
    _panel_e_undrained(axes[1, 1])
    print("  panel f (Bonacina)...")
    _panel_f_bonacina(axes[1, 2])
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
