#!/usr/bin/env python3
"""Figure ED5 — R18: closure-completeness at Umiujaq, two-panel story.

Panel a: VDTBS observed temperature at z = 5 m (5-borehole monthly
mean) vs the forward prediction from the pure-sensible
``column_coupled`` solver and from the merged
``column_thermo_freeze_coupled`` solver.  The merged forward tracks
observations better -- a 12.1 % depth-integrated RMS misfit
reduction.

Panel b: profile-likelihood on the coupling parameter s = Γ N_α.
The merged-solver curve peaks at s = 0.5, vs the pure-sensible
curve which is degenerate at the s = 0 boundary.  The R8 published
synthetic-twin headline of s = 0.97 is marked for reference.

This figure replaces the cluttered 4-panel R18 demonstration; the
Stefan analytical benchmark and the global L-regime map are moved
to Supplementary Information §S6.

Output:
``outputs/figures/empirical/fig_ed5_closure_completeness.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


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
        "--out",
        default="outputs/figures/empirical/fig_ed5_closure_completeness.pdf",
    )
    args = parser.parse_args(argv)

    fwd = xr.open_dataset(Path(args.fwd).expanduser().resolve())
    obs = xr.open_dataset(Path(args.obs).expanduser().resolve())
    df = pd.read_parquet(Path(args.post).expanduser().resolve())

    apply_nature_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.40),
    )

    # --------- Panel a: T at 5 m, observed vs the two solvers --------
    depth_target = 5.0
    idx_fwd = int(np.argmin(np.abs(fwd["depth_m"].values - depth_target)))
    idx_obs = int(np.argmin(np.abs(obs["depth_m"].values - depth_target)))
    obs_T = obs["T_degC"].values[:, idx_obs, :].mean(axis=1)
    obs_times = pd.to_datetime(obs["time"].values)
    s_obs = pd.Series(obs_T, index=obs_times).resample("MS").mean()

    ax_a.plot(
        s_obs.index,
        s_obs.values,
        "-",
        color="black",
        linewidth=1.0,
        label="observed (VDTBS, 5-borehole mean)",
    )
    ax_a.plot(
        pd.to_datetime(fwd["time"].values),
        fwd["T_cc"].values[:, idx_fwd],
        "-",
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.85,
        label=r"\texttt{column\_coupled} (no latent heat)",
    )
    ax_a.plot(
        pd.to_datetime(fwd["time"].values),
        fwd["T_merged"].values[:, idx_fwd],
        "-",
        color="#d62728",
        linewidth=1.0,
        label="merged solver (latent heat on)",
    )
    ax_a.axhline(0.0, color="0.6", linewidth=0.4, linestyle=":")
    ax_a.set_xlabel("date")
    ax_a.set_ylabel(rf"$T$  at  $z = {int(depth_target)}$ m  ($^\circ$C)")
    ax_a.legend(loc="lower right", frameon=False, fontsize=6)
    ax_a.text(
        0.03,
        0.97,
        "a",
        transform=ax_a.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    # Headline annotation: the RMS reduction.
    rms_text = (
        r"depth-integrated RMS (0.5--20 m):" + "\n"
        r"  \texttt{column\_coupled}:  1.77 K" + "\n"
        r"  merged:  1.56 K  ($-12.1\,\%$)"
    )
    ax_a.text(
        0.03,
        0.04,
        rms_text,
        transform=ax_a.transAxes,
        fontsize=5.5,
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.25"),
    )
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    # --------- Panel b: profile-likelihood on s ---------
    s_grid = df["s"].values
    ll_m = df["log_lik_merged"].values
    ll_c = df["log_lik_cc"].values
    ll_m_n = ll_m - ll_m.max()
    ll_c_n = ll_c - ll_c.max()

    ax_b.plot(
        s_grid,
        ll_m_n,
        "-o",
        color="#d62728",
        markersize=3.5,
        linewidth=1.1,
        label="merged solver",
    )
    ax_b.plot(
        s_grid,
        ll_c_n,
        "-o",
        color="#1f77b4",
        markersize=3.5,
        linewidth=1.1,
        alpha=0.85,
        label=r"\texttt{column\_coupled}",
    )
    ax_b.axhline(
        -1.353,
        color="0.5",
        linewidth=0.5,
        linestyle="--",
        label=r"$\Delta \log \mathcal{L} = -1.35$ (90\,\% CI)",
    )
    # Mark the published R8 synthetic-twin headline at s = 0.97.
    ax_b.axvline(0.97, color="#666666", linewidth=0.8, linestyle="-.", alpha=0.8)
    ax_b.text(
        0.98,
        ax_b.get_ylim()[1],
        "  R8 synthetic-twin\n  $s = 0.97$",
        fontsize=5.5,
        va="top",
        ha="left",
        color="#666666",
    )
    # Mark the merged-solver MAP at s = 0.5.
    i_map = int(np.argmax(ll_m))
    s_map = float(s_grid[i_map])
    ax_b.axvline(s_map, color="#d62728", linewidth=0.8, linestyle="-.", alpha=0.7)
    ax_b.text(
        s_map - 0.03,
        ax_b.get_ylim()[1],
        f"  merged\n  $s_{{\\mathrm{{MAP}}}} = {s_map:.2f}$",
        fontsize=5.5,
        va="top",
        ha="right",
        color="#d62728",
    )
    ax_b.set_xlabel(r"$s = \Gamma N_\alpha$  coupling scale")
    ax_b.set_ylabel(r"$\Delta \log \mathcal{L}$  vs. argmax")
    ax_b.set_ylim(-30.0, 2.5)
    ax_b.set_xlim(-0.05, 1.85)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6)
    ax_b.text(
        0.03,
        0.97,
        "b",
        transform=ax_b.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    ax_b.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
