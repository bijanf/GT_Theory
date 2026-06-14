#!/usr/bin/env python3
"""Figure 9 -- 2-D case studies that the 1-D solver structurally
cannot produce.

Panel a-b: lateral basin recharge-discharge cross-section.
  a -- final-time T(x, z) heatmap with Darcy velocity vectors
       overlaid.
  b -- depth-averaged T(x) profile vs the no-flow control.

Panel c-d: 2-D permafrost lateral-edge effect.
  c -- final-time T(x, z) showing curved active-layer base.
  d -- final-time S_i(x, z) showing the asymmetric freezing
       front.

Output:
``outputs/figures/numerical/fig09_2d_cases.pdf``.
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


def _panel_basin_t(ax, ds: xr.Dataset) -> None:
    T_final = ds["T_degC"].values[-1]
    v_x = ds["v_x"].values[-1]
    v_z = ds["v_z"].values[-1]
    x_km = ds["x_m"].values / 1000.0
    z = ds["depth_m"].values
    im = ax.pcolormesh(
        x_km,
        z,
        T_final,
        cmap="RdYlBu_r",
        shading="auto",
        rasterized=True,
    )
    # Sub-sample for arrows.  Plot the Darcy velocity unit-vectors so the
    # flow geometry is visible (magnitude is reported separately in
    # the caption: median |v_x| ~ 3.5e-7 m/s).
    s_x, s_z = 4, 3
    Xg, Zg = np.meshgrid(x_km[::s_x], z[::s_z])
    Ux = v_x[::s_z, ::s_x]
    Uz = v_z[::s_z, ::s_x]
    mag = np.sqrt(Ux * Ux + Uz * Uz)
    mag = np.where(mag > 0, mag, 1.0)
    # Display direction-only: unit vectors with consistent visible length.
    Q = ax.quiver(
        Xg,
        Zg,
        Ux / mag,
        Uz / mag,
        scale=22.0,
        scale_units="width",
        color="black",
        width=0.0035,
        alpha=0.8,
        headwidth=4,
        headlength=5,
        headaxislength=4,
    )
    ax.quiverkey(
        Q,
        0.82,
        1.04,
        1.0,
        r"$\hat{v}_{\rm Darcy}$",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 5},
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (km)")
    ax.set_ylabel("depth (m)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$T$  ($^\circ$C)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.text(
        0.03,
        0.97,
        "a",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="black",
    )


def _panel_basin_top_t(ax, ds: xr.Dataset) -> None:
    T_final = ds["T_degC"].values[-1]
    T_init = ds["T_degC"].values[0]
    x_km = ds["x_m"].values / 1000.0
    z = ds["depth_m"].values
    # Mid-column probe depth: the lateral T asymmetry here combines lateral
    # conduction of the surface-temperature contrast and lateral advection;
    # both are genuinely 2-D (a 1-D column carries no lateral gradient).
    z_probe_m = 300.0
    i_z = int(np.argmin(np.abs(z - z_probe_m)))
    T_probe_final = T_final[i_z, :]
    T_probe_init = T_init[i_z, :]
    ax.axhline(
        float(T_probe_init.mean()),
        color="0.4",
        linewidth=0.9,
        linestyle="--",
        label=rf"$t = 0$  (conductive baseline)",
    )
    ax.plot(
        x_km,
        T_probe_final,
        "-",
        color="#d62728",
        linewidth=1.4,
        label=rf"$t = 500$ yr (advection on)",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel(rf"$T(x, z={z_probe_m:.0f}\,$m$)$  ($^\circ$C)")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.text(
        0.03,
        0.97,
        "b",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def _panel_edge_t(ax, ds: xr.Dataset) -> None:
    T_final = ds["T_degC"].values[-1]
    x = ds["x_m"].values
    z = ds["depth_m"].values
    im = ax.pcolormesh(
        x,
        z,
        T_final,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        vmin=-8.0,
        vmax=8.0,
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$T$  ($^\circ$C)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.text(
        0.03,
        0.97,
        "c",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="white",
    )


def _panel_edge_s(ax, ds: xr.Dataset) -> None:
    S_final = ds["S_i"].values[-1]
    x = ds["x_m"].values
    z = ds["depth_m"].values
    im = ax.pcolormesh(
        x,
        z,
        S_final,
        cmap="Blues",
        shading="auto",
        rasterized=True,
        vmin=0.0,
        vmax=1.0,
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$S_i$", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.text(
        0.03,
        0.97,
        "d",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="black",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--basin",
        default="outputs/cases/lateral_basin_2d.nc",
    )
    parser.add_argument(
        "--edge",
        default="outputs/cases/permafrost_edge_2d.nc",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig09_2d_cases.pdf",
    )
    args = parser.parse_args(argv)

    ds_basin = xr.open_dataset(Path(args.basin).expanduser().resolve())
    ds_edge = xr.open_dataset(Path(args.edge).expanduser().resolve())

    apply_nature_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.72),
    )
    _panel_basin_t(axes[0, 0], ds_basin)
    _panel_basin_top_t(axes[0, 1], ds_basin)
    _panel_edge_t(axes[1, 0], ds_edge)
    _panel_edge_s(axes[1, 1], ds_edge)
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
