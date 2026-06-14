#!/usr/bin/env python3
"""Figure ED1 — solver benchmarks.

Four panels, one per benchmark exercised by ``tests/test_column_coupled.py``:

a. Carslaw-Jaeger uncoupled limit: with `gamma_n_alpha_scale = 0` and a
   step surface T, the solver recovers the half-space erfc kernel.
b. Terzaghi 1-D consolidation: pressure decay against the standard
   series solution at one mid-time snapshot.
c. Undrained thermo-poroelastic ratio: in the K → 0 limit, interior
   dp/dT tracks the linearised α_w/β_w line.
d. Coupling-switch null: with `s = 0`, the pressure field stays at its
   initial/BC state regardless of T evolution.

Output: ``outputs/figures/empirical/fig_ed1_benchmarks.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)
from gt_theory.solvers import (
    carslaw_jaeger_step_analytic,
    run_column_coupled,
)

YEAR_S = 365.25 * 86400.0


# --------------------------------------------------------------------------
# Benchmark generators (mirror tests/test_column_coupled.py)
# --------------------------------------------------------------------------


def _carslaw_jaeger() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    T_theory = carslaw_jaeger_step_analytic(res.z, np.array([duration_s]), 1.0, kappa)[0]
    return res.z, res.T[-1], T_theory


def _terzaghi() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    beta_w = 4.5e-10
    mu = 1.0e-3
    phi = 0.15
    K_zz = 1.0e-13
    rho_w = 1000.0
    L = 100.0
    p0 = 1.0e5
    c_v = K_zz / (mu * rho_w * phi * beta_w)
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
    return z, res.p[n_idx] / p0, p_an / p0, T_dim


def _undrained_ratio() -> tuple[np.ndarray, np.ndarray, float]:
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
        K_zz=1.0e-20,
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
    # Pull (T, p) pairs from all interior nodes deeper than 10 m at final time.
    z = res.z
    mask = z > 10.0
    T_pts = res.T[-1, mask]
    p_pts = res.p[-1, mask]
    expected = alpha_w / beta_w
    return T_pts, p_pts, expected


def _coupling_null() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coupling-switch null + companion s=1 trace, both as max|p|(t).

    s=0 gives p ≡ 0 analytically; s=1 with K → 0 returns the undrained
    thermo-poroelastic response. The contrast is the diagnostic value
    of the panel.
    """
    common = dict(
        depth_max_m=200.0,
        dz_m=2.0,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        K_zz=1.0e-20,
        g=0.0,
        sat=1.0,
        p_top=0.0,
        p_init=0.0,
    )
    res_off = run_column_coupled(**common, gamma_n_alpha_scale=0.0)
    res_on = run_column_coupled(**common, gamma_n_alpha_scale=1.0)
    p_off = np.max(np.abs(res_off.p), axis=1)
    p_on = np.max(np.abs(res_on.p), axis=1)
    return res_off.t / YEAR_S, p_off, p_on, res_off.z


# --------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed1_benchmarks.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.85),
    )
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # --- a. Carslaw-Jaeger uncoupled limit ---
    z, T_num, T_th = _carslaw_jaeger()
    ax_a.plot(T_th, z, color="0.4", linewidth=1.0, label="analytic")
    ax_a.plot(T_num, z, linestyle="--", color="#c0392b", linewidth=0.9, label="solver")
    ax_a.invert_yaxis()
    ax_a.set_xlabel("T (K)")
    ax_a.set_ylabel("depth (m)")
    ax_a.legend(loc="lower right", frameon=False)
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- b. Terzaghi 1-D consolidation ---
    z_t, p_num, p_an, T_dim = _terzaghi()
    ax_b.plot(p_an, z_t, color="0.4", linewidth=1.0, label="analytic")
    ax_b.plot(p_num, z_t, linestyle="--", color="#c0392b", linewidth=0.9, label="solver")
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"$p/p_0$")
    ax_b.set_ylabel("depth (m)")
    ax_b.text(0.05, 0.10, rf"$T = {T_dim:.2f}$", transform=ax_b.transAxes)
    ax_b.legend(loc="lower right", frameon=False)
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- c. Undrained thermo-poroelastic ratio ---
    T_pts, p_pts, expected = _undrained_ratio()
    ax_c.scatter(
        T_pts, p_pts / 1e3, s=14, color="#c0392b", edgecolor="none", alpha=0.7, label="solver"
    )
    Tline = np.linspace(0.0, T_pts.max() * 1.05, 50)
    ax_c.plot(
        Tline,
        expected * Tline / 1e3,
        color="0.4",
        linewidth=1.0,
        label=rf"$\alpha_w / \beta_w \approx {expected:.0f}$ Pa K$^{{-1}}$",
    )
    ax_c.set_xlabel(r"$T$ (K)")
    ax_c.set_ylabel(r"$p$ (kPa)")
    ax_c.legend(loc="upper left", frameon=False)
    ax_c.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- d. Coupling-switch null ---
    t_yr, p_off, p_on, _ = _coupling_null()
    # Plot s=1 in kPa (positive) and s=0 as a flat line at zero on the
    # same linear axis -- the contrast is the diagnostic value of the
    # panel.
    ax_d.plot(
        t_yr,
        p_on / 1e3,
        color="#c0392b",
        linewidth=0.9,
        label=r"$s = 1$  (undrained thermo-poroelastic)",
    )
    ax_d.plot(t_yr, p_off / 1e3, color="0.4", linewidth=1.0, label=r"$s = 0$  (null, $\equiv 0$)")
    ax_d.set_xlabel("time (yr)")
    ax_d.set_ylabel(r"$\max_z |p(z, t)|$ (kPa)")
    p_off_peak = float(np.max(p_off))
    ax_d.text(
        0.05,
        0.08,
        rf"max $|p_{{s=0}}|$ = {p_off_peak:.2e} Pa",
        transform=ax_d.transAxes,
        color="0.3",
    )
    ax_d.legend(loc="upper left", frameon=False)
    ax_d.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # Panel labels.
    for ax, lbl in zip((ax_a, ax_b, ax_c, ax_d), ("a", "b", "c", "d")):
        ax.text(0.02, 0.97, lbl, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
