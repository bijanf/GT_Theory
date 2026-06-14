#!/usr/bin/env python3
"""Figure 11 -- advection-dominated verification with grid refinement.

The conduction benchmarks (Fig. 2) and the finite-volume / Crank-Nicolson
cross-verification (Fig. 10) are all conduction- or low-Peclet problems;
neither exercises the advection operator that distinguishes the two
schemes (first-order upwind vs centred).  This figure closes that gap.

A surface temperature step is advected into a column at the geothermal
regime's cell Peclet number by a sustained uniform Darcy flow; the
analytical Ogata-Banks (1961) advection-dispersion solution is the
reference.  We verify the primary finite-volume (upwind) solver against
it under grid refinement, and contrast it with a centred discretisation
of the same advection-diffusion operator -- the discretisation the
Crank-Nicolson reference solver uses.

Panels:
  a -- coarse grid (cell Peclet ~ 5.6): the centred scheme overshoots the
       front, the upwind finite-volume solver stays monotone; both bracket
       the Ogata-Banks profile.
  b -- fine grid (cell Peclet ~ 0.7): both schemes fall on the analytical
       solution.
  c -- L-infinity error vs grid spacing (log-log): both errors collapse
       under refinement -- upwind at ~first order, centred at ~second order
       once the cell Peclet drops below 2 (dashed line) -- so the two
       schemes are consistent and the upwind choice is a robustness, not
       an accuracy, decision.

Output: ``outputs/figures/numerical/fig11_advection_xverify.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

from gt_theory.benchmarks.ogata_banks import ogata_banks_response
from gt_theory.plotting.style import NATURE_2COL_INCH, apply_nature_style
from gt_theory.solvers.column_fvm_permafoam import run_column_fvm_permafoam

YEAR_S = 365.25 * 86400.0
ANALYTIC = "black"
C_FVM = "#1f77b4"  # finite volume (upwind, primary)
C_CEN = "#d62728"  # centred (the Crank-Nicolson reference's advection)

# --- controlled, single-phase, constant-property advection-diffusion config.
# A surface step delta_T advected downward at a sustained uniform Darcy flux
# v (imposed via a fixed head with gravity and the thermo-poro coupling off,
# so the pressure field is steady and the velocity uniform).  Numbers chosen
# so the cell Peclet at the production grid matches the geothermal case.
PHI = 0.02
LAM = 3.0
RHO_R, C_R = 2700.0, 1000.0
RHO_W, C_W = 1000.0, 4186.0
MU = 1.0e-3
K_ZZ = 1.0e-13
V_DARCY = 1.0e-7
L_COL = 2000.0
DUR = 30.0 * YEAR_S
DELTA_T = 10.0
DZ_SWEEP = (40.0, 20.0, 10.0, 5.0, 2.5)

RHO_C = (1.0 - PHI) * RHO_R * C_R + PHI * RHO_W * C_W
KAPPA = LAM / RHO_C
V_T = RHO_W * C_W * V_DARCY / RHO_C  # thermal front velocity
DP_HEAD = V_DARCY * MU / K_ZZ * L_COL  # head that drives V_DARCY (g = 0)
DZ_PE2 = 2.0 * LAM / (RHO_W * C_W * V_DARCY)  # grid spacing at cell Peclet = 2


def cell_peclet(dz: float) -> float:
    return RHO_W * C_W * V_DARCY * dz / LAM


def _fvm_profile(dz: float):
    """Primary finite-volume (upwind) solver on the advection config."""
    nz = int(round(L_COL / dz))
    z = (np.arange(nz) + 0.5) * dz
    p_init = DP_HEAD * (1.0 - z / L_COL)  # steady linear head, drives uniform v
    res = run_column_fvm_permafoam(
        depth_max_m=L_COL,
        dz_m=dz,
        duration_s=DUR,
        dt_s=DUR / 360.0,
        porosity=PHI,
        lambda_r=LAM,
        lambda_w=LAM,
        lambda_i=LAM,
        rho_r=RHO_R,
        c_r=C_R,
        rho_w=RHO_W,
        c_w=C_W,
        mu=MU,
        K_zz=K_ZZ,
        g=0.0,
        T_f=-100.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        sat=DELTA_T,
        T_init=0.0,
        p_top=DP_HEAD,
        p_init=p_init,
        bot_p_bc="dirichlet",
        p_bot=0.0,
        q_bot=0.0,
    )
    return res.z, res.T[-1]


def _centred_profile(dz: float):
    """Minimal cell-centred backward-Euler advection-diffusion solve with
    *centred* advection -- a transparent stand-in for the centred-advection
    discretisation the Crank-Nicolson reference solver uses, on the same
    surface-step problem.  Backward Euler keeps it bounded so the spatial
    overshoot at cell Peclet > 2 is visible without temporal blow-up.
    """
    n = int(round(L_COL / dz))
    z = (np.arange(n) + 0.5) * dz
    dt = DUR / 360.0
    rd = KAPPA * dt / dz**2  # diffusion number
    ca = V_T * dt / dz  # advective Courant number
    lo = np.full(n, -rd - 0.5 * ca)
    di = np.full(n, 1.0 + 2.0 * rd)
    up = np.full(n, -rd + 0.5 * ca)
    di[0] += rd  # half-cell Dirichlet face at z = 0
    di[-1] += up[-1]  # zero-gradient bottom (far field)
    up[-1] = 0.0
    ab = np.zeros((3, n))
    ab[0, 1:] = up[:-1]
    ab[1, :] = di
    ab[2, :-1] = lo[1:]
    bc_top = (2.0 * rd + 0.5 * ca) * DELTA_T
    T = np.zeros(n)
    for _ in range(int(round(DUR / dt))):
        rhs = T.copy()
        rhs[0] += bc_top
        T = solve_banded((1, 1), ab, rhs)
    return z, T


def _linf(z, T, mask_depth=600.0):
    ob = ogata_banks_response(z, DUR, delta_T=DELTA_T, v_T=V_T, kappa=KAPPA)
    m = z < mask_depth
    return float(np.max(np.abs(T[m] - ob[m])))


def _panel_profiles(ax, dz, letter, zmax=300.0, legend=False) -> None:
    z_ob = np.linspace(0.0, zmax, 400)
    T_ob = ogata_banks_response(z_ob, DUR, delta_T=DELTA_T, v_T=V_T, kappa=KAPPA)
    zf, Tf = _fvm_profile(dz)
    zc, Tc = _centred_profile(dz)
    ax.plot(T_ob, z_ob, "-", color=ANALYTIC, lw=1.1, label="Ogata-Banks")
    ax.plot(Tc, zc, "-", color=C_CEN, lw=1.0, alpha=0.9, label="centred")
    sub = slice(0, None, max(1, int(round(2.0 / dz)) if dz < 2.0 else 1))
    ax.plot(
        Tf[sub],
        zf[sub],
        "o",
        color=C_FVM,
        ms=2.6,
        mfc="none",
        mew=0.7,
        label="finite volume (upwind)",
    )
    ax.set_ylim(zmax, 0.0)
    ax.set_xlim(-0.6, 11.0)
    ax.set_xlabel(r"$T$  ($^\circ$C)")
    ax.set_ylabel("depth (m)")
    ax.text(
        0.55,
        0.93,
        rf"$\mathrm{{Pe}}_{{\rm cell}}\approx{cell_peclet(dz):.1f}$",
        transform=ax.transAxes,
        fontsize=6,
        ha="center",
        color="0.3",
    )
    if legend:
        ax.legend(loc="lower right", frameon=False, fontsize=5.5)
    ax.text(0.04, 0.96, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _panel_convergence(ax) -> None:
    dzs = np.array(DZ_SWEEP)
    err_fvm, err_cen = [], []
    for dz in DZ_SWEEP:
        zf, Tf = _fvm_profile(dz)
        zc, Tc = _centred_profile(dz)
        err_fvm.append(_linf(zf, Tf))
        err_cen.append(_linf(zc, Tc))
    err_fvm = np.array(err_fvm)
    err_cen = np.array(err_cen)
    ax.loglog(dzs, err_fvm, "o-", color=C_FVM, ms=3.0, lw=1.0, label="finite volume (upwind)")
    ax.loglog(dzs, err_cen, "s-", color=C_CEN, ms=3.0, lw=1.0, mfc="none", label="centred")
    # first- and second-order guide lines anchored at the finest grid
    g1 = err_fvm[-1] * (dzs / dzs[-1]) ** 1.0
    g2 = err_cen[-1] * (dzs / dzs[-1]) ** 2.0
    ax.loglog(dzs, g1, ":", color="0.5", lw=0.8)
    ax.loglog(dzs, g2, "--", color="0.5", lw=0.8)
    ax.text(dzs[1], g1[1] * 1.25, "slope 1", fontsize=5.5, color="0.4")
    ax.text(dzs[1], g2[1] * 0.55, "slope 2", fontsize=5.5, color="0.4")
    ax.axvline(DZ_PE2, color="0.7", ls="--", lw=0.8)
    ax.text(
        DZ_PE2 * 1.05,
        err_cen.min() * 1.4,
        r"$\mathrm{Pe}_{\rm cell}=2$",
        fontsize=5.5,
        color="0.4",
        rotation=90,
        va="bottom",
    )
    ax.set_xlabel(r"$\Delta z$  (m)")
    ax.set_ylabel(r"$\max_z |T_{\rm num} - T_{\rm OB}|$  (K)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.87), frameon=False, fontsize=5.5)
    ax.text(0.04, 0.96, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig11_advection_xverify.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.34),
    )
    _panel_profiles(axes[0], 40.0, "a", legend=True)
    _panel_profiles(axes[1], 5.0, "b")
    _panel_convergence(axes[2])
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
