#!/usr/bin/env python3
"""Figure 10 -- cross-verification of the two independent numerical
schemes for the same coupled closure.

The finite-volume backward-Euler solver (``column_fvm_permafoam``,
PermaFoam strategy) and the finite-difference Crank-Nicolson solver
(``column_thermo_freeze_coupled``) share no numerical machinery -- they
differ in discretisation (FV vs FD), time integration (implicit Euler
vs Crank-Nicolson), coupling (sequential two-TDMA vs monolithic
block-banded) and advection (upwind vs centred).  Their agreement on
the analytical benchmarks is therefore a strong correctness argument.

Panels:
  a -- pure conduction: both schemes vs the Carslaw-Jaeger erfc profile
       (agreement in space);
  b -- one-phase Stefan freezing front xi(t): both schemes vs the
       Neumann similarity solution (agreement on a moving phase front);
  c -- max |T_FVM - T_CN| through a multi-decade transient
       (agreement in time, quantified).

Output: ``outputs/figures/numerical/fig10_fvm_vs_cn.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.benchmarks.carslaw_jaeger import step_temperature_response
from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
)
from gt_theory.plotting.style import NATURE_2COL_INCH, apply_nature_style
from gt_theory.solvers.column_fvm_permafoam import run_column_fvm_permafoam
from gt_theory.solvers.column_thermo_freeze_coupled import (
    run_column_thermo_freeze_coupled,
)

YEAR_S = 365.25 * 86400.0
ANALYTIC = "black"
C_FVM = "#1f77b4"  # finite volume
C_CN = "#d62728"  # Crank-Nicolson


def _conduction_runs():
    phi = 0.15
    lam = 2.5
    rho_r = 2700.0
    c_r = (2.5e6 - phi * 1000.0 * 4186.0) / ((1.0 - phi) * rho_r)
    kappa = lam / 2.5e6
    dur = 5.0 * YEAR_S
    common = dict(
        depth_max_m=60.0,
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
        g=0.0,
        q_bot=0.0,
    )
    r_fvm = run_column_fvm_permafoam(**common)
    r_cn = run_column_thermo_freeze_coupled(**common)
    return r_fvm, r_cn, dur, kappa


def _panel_a_conduction(ax) -> None:
    r_fvm, r_cn, dur, kappa = _conduction_runs()
    z_fine = np.linspace(0.0, 60.0, 400)
    T_ana = step_temperature_response(z_fine, dur, delta_T=10.0, kappa=kappa)
    ax.plot(T_ana, z_fine, "-", color=ANALYTIC, lw=1.1, label="Carslaw-Jaeger")
    ax.plot(r_cn.T[-1], r_cn.z, "-", color=C_CN, lw=1.0, alpha=0.9, label="Crank-Nicolson (FD)")
    sub = slice(0, None, 6)
    ax.plot(
        r_fvm.T[-1][sub],
        r_fvm.z[sub],
        "o",
        color=C_FVM,
        ms=2.6,
        mfc="none",
        mew=0.7,
        label="finite volume",
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$T$  ($^\circ$C)")
    ax.set_ylabel("depth (m)")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax.text(0.04, 0.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _stefan_front_series(res, t_query):
    """Deepest cell with S_i > 0.5 at each queried time index."""
    xi = np.zeros(t_query.size)
    for j, it in enumerate(t_query):
        frozen = res.S_i[it] > 0.5
        xi[j] = res.z[frozen][-1] if frozen.any() else 0.0
    return xi


def _panel_b_stefan(ax) -> None:
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
    common = dict(
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
    )
    r_fvm = run_column_fvm_permafoam(**common)
    r_cn = run_column_thermo_freeze_coupled(**common)

    t_ana = np.linspace(0.0, dur, 200)
    ax.plot(
        t_ana / 86400.0,
        stefan_front_position(t_ana, p_stefan),
        "-",
        color=ANALYTIC,
        lw=1.1,
        label="Neumann",
    )
    idx = np.linspace(1, r_cn.t.size - 1, 16).astype(int)
    ax.plot(
        r_cn.t[idx] / 86400.0,
        _stefan_front_series(r_cn, idx),
        "s",
        color=C_CN,
        ms=2.8,
        mfc="none",
        mew=0.7,
        label="Crank-Nicolson (FD)",
    )
    idxf = np.linspace(1, r_fvm.t.size - 1, 16).astype(int)
    ax.plot(
        r_fvm.t[idxf] / 86400.0,
        _stefan_front_series(r_fvm, idxf),
        "o",
        color=C_FVM,
        ms=2.6,
        mfc="none",
        mew=0.7,
        label="finite volume",
    )
    ax.set_xlabel("time (days)")
    ax.set_ylabel(r"freezing-front depth  $\xi(t)$  (m)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.text(
        0.96,
        0.04,
        "b",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _panel_c_transient(ax) -> None:
    """Multi-decade conduction + geothermal transient; report the
    instantaneous max |T_FVM - T_CN| (interpolated onto the FV centres)."""
    phi = 0.15
    rho_r = 2700.0
    c_r = (2.5e6 - phi * 1000.0 * 4186.0) / ((1.0 - phi) * rho_r)
    dur = 50.0 * YEAR_S
    t = np.arange(0, 401) * (dur / 400.0)
    sat = 5.0 + 8.0 * np.sin(2.0 * np.pi * t / (10.0 * YEAR_S))
    common = dict(
        depth_max_m=120.0,
        dz_m=1.0,
        duration_s=dur,
        dt_s=dur / 400.0,
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
        sat=sat,
        T_init=5.0,
        p_top=0.0,
        g=0.0,
        q_bot=0.06,
    )
    r_fvm = run_column_fvm_permafoam(**common)
    r_cn = run_column_thermo_freeze_coupled(**common)
    max_diff = np.array(
        [
            np.max(np.abs(r_fvm.T[k] - np.interp(r_fvm.z, r_cn.z, r_cn.T[k])))
            for k in range(r_fvm.t.size)
        ]
    )
    ax.plot(r_fvm.t / YEAR_S, max_diff * 1.0e3, "-", color=C_FVM, lw=1.0)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(r"$\max_z |T_{\rm FV} - T_{\rm CN}|$  (mK)")
    ax.set_ylim(bottom=0.0)
    ax.text(
        0.96, 0.92, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="right"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig10_fvm_vs_cn.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.34),
    )
    _panel_a_conduction(axes[0])
    _panel_b_stefan(axes[1])
    _panel_c_transient(axes[2])
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
