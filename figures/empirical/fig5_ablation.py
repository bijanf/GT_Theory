#!/usr/bin/env python3
"""Figure 5 — ablation curves: Δp_rms (coupled vs s=0 baseline) as a
function of the ΓN_α scalar knob, at all three supersites side-by-side.

The plot is the headline diagnostic that the coupled solver's
response is monotone in the new physics: pressure error against the
``coupling=0`` limit increases smoothly as the knob moves from 0 to 1,
with no discontinuities. The Umiujaq panel uses synthetic GST forcing
in its current smoke state; Mont Terri and FORGE use literature-derived
boundary conditions.

Output: ``outputs/figures/empirical/fig5_ablation.pdf``.

Usage::

    python figures/empirical/fig5_ablation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)


SITES = [
    {
        "key": "umiujaq",
        "label": "Umiujaq",
        "nc": "outputs/supersite_umiujaq/forward_runs.nc",
        "probe_depth_m": 5.0,
        "pressure_unit": ("kPa", 1.0e3),
    },
    {
        "key": "mont_terri",
        "label": "Mont Terri HE-D",
        "nc": "outputs/supersite_mont_terri/forward_runs.nc",
        "probe_depth_m": 0.3,
        "pressure_unit": ("MPa", 1.0e6),
    },
    {
        "key": "forge",
        "label": "Utah FORGE",
        "nc": "outputs/supersite_forge/forward_runs.nc",
        "probe_depth_m": 100.0,
        "pressure_unit": ("kPa", 1.0e3),
    },
]


def _delta_p_rms_vs_coupling(
    ds: xr.Dataset, probe_depth_m: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """For each s in ``ds.coupling``, return Δp_rms vs the s=0 column at
    the requested probe depth."""
    z = ds.depth_m.values
    iz = int(np.argmin(np.abs(z - probe_depth_m)))
    z_used = float(z[iz])

    s_values = ds.coupling.values
    i_off = int(np.argmin(np.abs(s_values - 0.0)))
    p_off = ds.p_Pa.isel(coupling=i_off, depth_m=iz).values

    drms = np.empty_like(s_values)
    for k, _s in enumerate(s_values):
        p_k = ds.p_Pa.isel(coupling=k, depth_m=iz).values
        drms[k] = float(np.sqrt(np.mean((p_k - p_off) ** 2)))
    return s_values, drms, z_used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig5_ablation.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args(argv)

    apply_nature_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH / 3.0),
        sharex=True,
    )

    for ax, site in zip(axes, SITES):
        nc_path = Path(site["nc"]).resolve()
        if not nc_path.exists():
            ax.text(
                0.5,
                0.5,
                f"missing\n{nc_path.name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.5",
            )
            continue

        ds = xr.open_dataset(nc_path)
        s_vals, drms_Pa, z_used = _delta_p_rms_vs_coupling(ds, site["probe_depth_m"])
        unit_label, unit_div = site["pressure_unit"]
        drms_scaled = drms_Pa / unit_div

        ax.plot(s_vals, drms_scaled, marker="o", linewidth=1.0, color="#c0392b")
        ax.set_xlabel(r"$\Gamma N_\alpha$ scale  $s$")
        ax.set_ylabel(r"$\Delta p_{\mathrm{rms}}$")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        # Site + unit annotation in the top-right of each panel: makes the
        # kPa <-> MPa swap between sites explicit.
        ax.text(
            0.97,
            0.92,
            f"{site['label']}\n({unit_label})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="0.2",
        )

    panel_labels = ("a", "b", "c")
    for ax, lbl in zip(axes, panel_labels):
        ax.text(0.04, 0.96, lbl, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
