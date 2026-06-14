#!/usr/bin/env python3
"""Figure ED10 — R18: closure-completeness verification for the
merged T+p+S_i solver.

Four panels:

(a) **Stefan analytical recovery** -- the merged solver reproduces the
    Neumann similarity solution for the one-phase freezing front
    ξ(t) = 2 λ √(κ t) within a few percent.
(b) **VDTBS forward overlay at Umiujaq** -- observed thermistor record
    at 5 m vs. the merged forward and the column_coupled forward.
(c) **Profile-likelihood posterior on s** -- pure-sensible column_coupled
    vs. merged solver on real VDTBS data.
(d) **World map of ℒ at the 948 Huang-Pollack sites** -- coloured by
    regime; the latent-heat-dominated sites (ℒ > 1) are flagged.

Output: ``outputs/figures/empirical/fig_ed10_freeze_thaw_closure.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
    """Merged solver vs analytical Neumann similarity solution."""
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
    t_grid = np.linspace(0.05, 0.5, 10) * YEAR_S
    xi_an = stefan_front_position(t_grid, p_stefan)

    xi_solver = []
    for t in t_grid:
        res = run_column_thermo_freeze_coupled(
            depth_max_m=4.0,
            dz_m=0.05,
            duration_s=float(t),
            dt_s=float(t) / 500.0,
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

    ax.plot(t_grid / YEAR_S, xi_an, "-", color="#1f77b4", linewidth=1.4, label="analytical Neumann")
    ax.plot(t_grid / YEAR_S, xi_solver, "o", color="#d62728", markersize=4, label="merged solver")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel(r"freezing front $\xi(t)$ (m)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.text(0.03, 0.97, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def _panel_b_vdtbs(ax, fwd_path: Path, obs_path: Path) -> None:
    """VDTBS observed T at ~5 m vs merged and column_coupled forwards."""
    fwd = xr.open_dataset(fwd_path)
    obs = xr.open_dataset(obs_path)

    depth_target = 5.0
    idx_fwd = int(np.argmin(np.abs(fwd["depth_m"].values - depth_target)))
    idx_obs = int(np.argmin(np.abs(obs["depth_m"].values - depth_target)))
    obs_T = obs["T_degC"].values[:, idx_obs, :].mean(axis=1)
    obs_times = pd.to_datetime(obs["time"].values)

    # Monthly mean of observed.
    s_obs = pd.Series(obs_T, index=obs_times).resample("MS").mean()
    ax.plot(
        s_obs.index,
        s_obs.values,
        "-",
        color="0.4",
        linewidth=1.0,
        label="VDTBS observed (monthly mean)",
    )
    ax.plot(
        pd.to_datetime(fwd["time"].values),
        fwd["T_merged"].values[:, idx_fwd],
        "-",
        color="#d62728",
        linewidth=1.0,
        label="merged forward",
    )
    ax.plot(
        pd.to_datetime(fwd["time"].values),
        fwd["T_cc"].values[:, idx_fwd],
        "-",
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.8,
        label="column\\_coupled forward",
    )
    ax.axhline(0.0, color="0.5", linewidth=0.3, linestyle=":")
    ax.set_xlabel("date")
    ax.set_ylabel(rf"$T$ at $z = {depth_target}$ m  (deg C)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.text(0.03, 0.97, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def _panel_c_posterior(ax, post_path: Path) -> None:
    """Profile-likelihood comparison."""
    df = pd.read_parquet(post_path)
    s = df["s"].values
    ll_m = df["log_lik_merged"].values
    ll_c = df["log_lik_cc"].values
    # Normalise each curve so max = 0.
    ll_m_n = ll_m - ll_m.max()
    ll_c_n = ll_c - ll_c.max()
    ax.plot(s, ll_m_n, "-o", color="#d62728", markersize=3, linewidth=1.0, label="merged")
    ax.plot(s, ll_c_n, "-o", color="#1f77b4", markersize=3, linewidth=1.0, label="column\\_coupled")
    ax.axhline(
        -1.353,
        color="0.5",
        linewidth=0.5,
        linestyle="--",
        label="90% CI ($\\Delta\\!\\log\\mathcal{L} = -1.35$)",
    )
    ax.set_xlabel(r"$s = \Gamma N_\alpha$  scale")
    ax.set_ylabel(r"$\Delta \log \mathcal{L}$  vs. argmax")
    ax.set_ylim(-30.0, 1.0)
    ax.legend(loc="lower center", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.text(0.03, 0.97, "c", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def _panel_d_lmap(ax, l_path: Path) -> None:
    """World map of ℒ for the 948 sites."""
    df = pd.read_parquet(l_path)
    lat = df["lat_deg"].values
    lon = df["lon_deg"].values
    l_vals = np.log10(np.clip(df["L_calL"].values, 1e-4, 1e3))
    # Map background: thin grey continent outlines via the data itself
    # since we don't ship a base map.  Just plot scatter on lon/lat.
    sc = ax.scatter(
        lon,
        lat,
        c=l_vals,
        cmap="RdYlBu_r",
        s=4,
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
    ax.text(0.03, 0.97, "d", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.text(
        0.03,
        0.08,
        rf"latent-heat-dominated ($\mathcal{{L}} > 1$):  {n_lf}/948  ({100 * n_lf / len(df):.1f}%)",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.2"),
    )
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fwd",
        default="outputs/supersite_umiujaq/forward_runs_merged.nc",
    )
    parser.add_argument(
        "--obs",
        default="data/supersite_umiujaq/processed/vdtbs_thermistors.nc",
    )
    parser.add_argument(
        "--post",
        default="outputs/supersite_umiujaq/posterior_merged.parquet",
    )
    parser.add_argument(
        "--l",
        default="outputs/global/l_regime.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed10_freeze_thaw_closure.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.75),
    )
    _panel_a_stefan(axes[0, 0])
    _panel_b_vdtbs(
        axes[0, 1], Path(args.fwd).expanduser().resolve(), Path(args.obs).expanduser().resolve()
    )
    _panel_c_posterior(axes[1, 0], Path(args.post).expanduser().resolve())
    _panel_d_lmap(axes[1, 1], Path(args.l).expanduser().resolve())
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
