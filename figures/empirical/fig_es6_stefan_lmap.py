#!/usr/bin/env python3
"""Supplementary Figure ES6 -- the technical-validation companions to
the main-paper R18 closure-completeness figure (ED5).

Panel a: analytical one-phase Neumann similarity solution
ξ(t) = 2 λ √(κ t) vs the merged-solver recovery.  This is the
deterministic Stefan-problem benchmark for the freezing-front
evolution.

Panel b: world map of log₁₀ ℒ (theory Eq. 279) at the 948
Huang--Pollack sites; the 31 sites with ℒ > 1 (latent-heat-
dominated regime where the merged solver is load-bearing) are
flagged with open black circles.

Output:
``outputs/figures/empirical/fig_es6_stefan_lmap.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
)
from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)
from gt_theory.solvers.column_thermo_freeze_coupled import (
    run_column_thermo_freeze_coupled,
)


YEAR_S = 365.25 * 86400.0


def _panel_a_stefan(ax) -> None:
    phi = 0.30
    rho_r, c_r = 2700.0, 800.0
    rho_w, c_w = 1000.0, 4186.0
    c_uniform = c_w
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
    t_grid = np.linspace(0.05, 0.5, 8) * YEAR_S
    xi_an = stefan_front_position(t_grid, p_stefan)

    xi_solver = []
    for t in t_grid:
        res = run_column_thermo_freeze_coupled(
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
            c_w=c_uniform,
            c_i=c_uniform * rho_w / 917.0,
            L_f=3.34e5,
            K_zz=1.0e-20,
            T_f=0.0,
            dTc=0.5,
            gamma_n_alpha_scale=0.0,
            sat=-10.0,
            T_init=0.0,
            picard_max_iter=15,
            picard_tol_K=1e-4,
        )
        S = res.S_i[-1]
        frozen = S > 0.5
        xi_solver.append(float(res.z[frozen][-1]) if frozen.any() else 0.0)

    ax.plot(
        t_grid / YEAR_S,
        xi_an,
        "-",
        color="black",
        linewidth=1.3,
        label="analytical Neumann",
    )
    ax.plot(
        t_grid / YEAR_S,
        xi_solver,
        "o",
        color="#d62728",
        markersize=4.5,
        label="merged solver",
    )
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(r"freezing front $\xi(t)$ (m)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.text(
        0.03,
        0.97,
        "a",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def _panel_b_lmap(ax, l_path: Path) -> None:
    df = pd.read_parquet(l_path)
    lat = df["lat_deg"].values
    lon = df["lon_deg"].values
    l_vals = np.log10(np.clip(df["L_calL"].values, 1e-4, 1e3))
    sc = ax.scatter(
        lon,
        lat,
        c=l_vals,
        cmap="RdYlBu_r",
        s=5,
        edgecolor="none",
        vmin=-2,
        vmax=2,
    )
    flag = df["needs_freeze_thaw_closure"].values
    ax.scatter(
        lon[flag],
        lat[flag],
        facecolors="none",
        edgecolors="black",
        s=22,
        linewidth=0.6,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.set_xlabel(r"longitude ($^\circ$E)")
    ax.set_ylabel(r"latitude ($^\circ$N)")
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$\log_{10} \mathcal{L}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    n_lf = int(flag.sum())
    ax.text(
        0.03,
        0.97,
        "b",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.03,
        0.08,
        rf"$\mathcal{{L}} > 1$ (latent-heat-dominated):  {n_lf}/948  ({100 * n_lf / len(df):.1f}\%)",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.2"),
    )
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--l",
        default="outputs/global/l_regime.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_es6_stefan_lmap.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42),
        gridspec_kw={"width_ratios": [1.0, 1.5]},
    )
    _panel_a_stefan(ax_a)
    _panel_b_lmap(ax_b, Path(args.l).expanduser().resolve())
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
