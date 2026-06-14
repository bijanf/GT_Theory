"""Shared 4-panel template for the regime case-study figures
(fig04 through fig07).

Each figure has the same layout so the manuscript reader can scan
across regimes:

  a -- T(z, t) heatmap
  b -- p(z, t) heatmap (anomaly relative to t = 0)
  c -- S_i(z, t) heatmap OR diagnostic time-series (for non-permafrost)
  d -- per-case dimensionless-number bar chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


YEAR_S = 365.25 * 86400.0


def _panel_heatmap(
    ax, ds: xr.Dataset, var: str, cmap: str, label: str, vmin=None, vmax=None
) -> None:
    arr = ds[var].values
    z = ds["depth_m"].values
    t_yr = ds["time"].values / YEAR_S
    im = ax.pcolormesh(
        t_yr,
        z,
        arr.T,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("depth (m)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6)


def _panel_dimless_bar(ax, ds: xr.Dataset) -> None:
    keys = ["Pe_T", "L_calL", "Gamma_N_alpha", "N_p"]
    labels = [
        r"$\mathrm{Pe}_T$",
        r"$\mathcal{L}$",
        r"$\Gamma N_\alpha$",
        r"$N_p$",
    ]
    values = [float(ds.attrs.get(f"dimless.{k}", float("nan"))) for k in keys]
    x = np.arange(len(keys))
    bar_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    ax.bar(
        x, np.log10(np.maximum(values, 1e-10)), color=bar_colors, edgecolor="0.25", linewidth=0.4
    )
    ax.axhline(0.0, color="0.5", linewidth=0.4, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\log_{10}$  (dimensionless)")
    ax.set_ylim(-5, 3)
    # Numeric annotations.
    for xi, v in zip(x, values):
        ax.text(
            xi, np.log10(max(v, 1e-10)) + 0.15, f"{v:.2g}", ha="center", va="bottom", fontsize=5.5
        )


def render_case_figure(
    case_nc: Path,
    out_path: Path,
    *,
    panel_c_var: str = "S_i",
    panel_c_label: str = r"$S_i$",
    panel_c_cmap: str = "Blues",
    panel_c_vmin: float | None = 0.0,
    panel_c_vmax: float | None = 1.0,
) -> None:
    ds = xr.open_dataset(case_nc)
    apply_nature_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.72),
    )
    _panel_heatmap(axes[0, 0], ds, "T_degC", "RdBu_r", r"$T$  ($^\circ$C)")
    axes[0, 0].text(
        0.03,
        0.97,
        "a",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="black",
    )
    # Pressure anomaly w.r.t. t=0.
    p_anom = ds["p_Pa"].values - ds["p_Pa"].values[0:1, :]
    ds_p_anom = ds.copy()
    ds_p_anom["p_Pa_anom"] = (("time", "depth_m"), p_anom)
    p_amp = float(np.nanpercentile(np.abs(p_anom), 99) + 1e-12)
    _panel_heatmap(
        axes[0, 1], ds_p_anom, "p_Pa_anom", "PuOr_r", r"$\Delta p$  (Pa)", vmin=-p_amp, vmax=p_amp
    )
    axes[0, 1].text(
        0.03, 0.97, "b", transform=axes[0, 1].transAxes, fontsize=8, fontweight="bold", va="top"
    )
    if panel_c_var in ds:
        _panel_heatmap(
            axes[1, 0],
            ds,
            panel_c_var,
            panel_c_cmap,
            panel_c_label,
            vmin=panel_c_vmin,
            vmax=panel_c_vmax,
        )
    else:
        # Fallback: show |v_darcy| heatmap.
        v_abs = np.log10(np.abs(ds["v_darcy"].values) + 1e-14)
        ds_vlog = ds.copy()
        ds_vlog["v_darcy_log"] = (("time", "depth_m"), v_abs)
        _panel_heatmap(
            axes[1, 0],
            ds_vlog,
            "v_darcy_log",
            "viridis",
            r"$\log_{10} |v_{\rm Darcy}|$ (m s$^{-1}$)",
        )
    axes[1, 0].text(
        0.03, 0.97, "c", transform=axes[1, 0].transAxes, fontsize=8, fontweight="bold", va="top"
    )
    _panel_dimless_bar(axes[1, 1], ds)
    axes[1, 1].text(
        0.03, 0.97, "d", transform=axes[1, 1].transAxes, fontsize=8, fontweight="bold", va="top"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
