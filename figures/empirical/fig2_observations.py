#!/usr/bin/env python3
"""Figure 2 — observations [smoke].

Two-panel placeholder for the Umiujaq observational record, drawn
from the coupled-solver forward run at ``coupling = 1.0``:

a. T(z) snapshots at four times across the 2-year window.
b. p(t) at one piezometer-screen depth (z = 5 m).

# SMOKE: real Nordicana D Borealis VDTBS + D19 Immatsiak series
# will replace these once the operator runs the curl recipe in
# data/supersite_umiujaq/README.md.

Output: ``outputs/figures/empirical/fig2_observations.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="outputs/supersite_umiujaq/forward_runs.nc",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig2_observations.pdf",
    )
    parser.add_argument("--probe-depth", type=float, default=5.0)
    args = parser.parse_args(argv)

    ds = xr.open_dataset(Path(args.input).expanduser().resolve())
    # SMOKE: use coupling=1 as a stand-in for observed.
    s_idx = int(np.argmin(np.abs(ds.coupling.values - 1.0)))
    T = ds["T_degC"].isel(coupling=s_idx).values  # (time, depth)
    p = ds["p_Pa"].isel(coupling=s_idx).values
    t_s = ds["time"].values
    z = ds["depth_m"].values

    t_yr = t_s / YEAR_S

    # Four equally-spaced snapshots across the window.
    n_t = T.shape[0]
    snap_idx = np.linspace(0, n_t - 1, 4).astype(int)
    snap_colors = plt.cm.viridis(np.linspace(0.1, 0.85, 4))

    iz = int(np.argmin(np.abs(z - args.probe_depth)))
    z_used = float(z[iz])

    apply_nature_style()
    fig, (ax_T, ax_p) = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42),
    )

    for k, color in zip(snap_idx, snap_colors):
        ax_T.plot(
            T[k],
            z,
            color=color,
            linewidth=0.9,
            label=f"$t = {t_yr[k]:.2f}$ yr",
        )
    ax_T.invert_yaxis()
    ax_T.set_xlabel(r"$T$ (degC)")
    ax_T.set_ylabel("depth (m)")
    ax_T.legend(loc="lower right", frameon=False)
    ax_T.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_T.text(0.02, 0.97, "a", transform=ax_T.transAxes, fontsize=7, fontweight="bold", va="top")

    ax_p.plot(t_yr, p[:, iz] / 1e3, color="#1f77b4", linewidth=0.8)
    ax_p.set_xlabel("time (yr since 2012-08-01)")
    ax_p.set_ylabel(rf"$p$ at $z = {z_used:.1f}$ m (kPa)")
    ax_p.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_p.text(0.02, 0.97, "b", transform=ax_p.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
