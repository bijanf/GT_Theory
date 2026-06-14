#!/usr/bin/env python3
"""Extended Data Figure 2 — Carslaw-Jaeger benchmark.

The Crank-Nicolson conduction solver in ``gt_theory.solvers.column_1d``
is validated against the analytic semi-infinite step-surface solution
``T(z, t) = dT * erfc(z / (2 sqrt(kappa t)))``.  Three panels:

  (a) Depth profiles at three times (10, 30, 100 yr) — numerical
      vs analytic curves overlaid.
  (b) Pointwise residual ``T_num - T_ana`` as a function of depth at
      t = 100 yr, with the +/- 0.05 K acceptance band.
  (c) Time series at three depths (20, 50, 100 m) — numerical
      crosses, analytic line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.solvers import carslaw_jaeger_step_analytic, run_column_1d

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR_S: float = 365.25 * 86400.0


def build_figure(out_path: Path) -> None:
    kappa = 1.0e-6
    dT = 1.0
    duration_s = 100.0 * YEAR_S
    res = run_column_1d(
        depth_max_m=1000.0,
        dz_m=2.0,
        duration_s=duration_s,
        dt_s=YEAR_S / 12.0,
        kappa=kappa,
        sat=dT,
    )
    t_yr = res.t / YEAR_S

    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.40))

    # (a) Depth profiles at three times.
    ax_a = axes[0]
    for t_yr_test, colour in ((10.0, "#aaaaaa"), (30.0, "#0066cc"), (100.0, "#cc3333")):
        it = int(np.argmin(np.abs(t_yr - t_yr_test)))
        ax_a.plot(
            res.T[it], res.z, color=colour, linewidth=1.0, label=f"t={t_yr_test:.0f} yr (num)"
        )
        T_ana = carslaw_jaeger_step_analytic(
            res.z, np.array([res.t[it]]), delta_T=dT, kappa=kappa
        ).ravel()
        ax_a.plot(
            T_ana,
            res.z,
            color=colour,
            linewidth=0.5,
            linestyle="--",
            label="" if t_yr_test != 100.0 else "analytic",
        )
    ax_a.invert_yaxis()
    ax_a.set_xlim(-0.05, 1.05)
    ax_a.set_xlabel(r"$\Delta T$ (K)")
    ax_a.set_ylabel("Depth (m)")
    ax_a.set_title("a   depth profiles", loc="left", weight="bold")
    ax_a.legend(loc="lower right", frameon=False, fontsize=5.5)
    ax_a.grid(alpha=0.25, linewidth=0.3)

    # (b) Residual at t = 100 yr vs depth.
    ax_b = axes[1]
    it = -1
    T_ana = carslaw_jaeger_step_analytic(
        res.z, np.array([res.t[it]]), delta_T=dT, kappa=kappa
    ).ravel()
    resid = res.T[it] - T_ana
    ax_b.axhspan(-0.05, 0.05, color="#cc7700", alpha=0.18, label="+/- 0.05 K target")
    ax_b.plot(res.z, resid, color="#0066cc", linewidth=1.0)
    ax_b.set_xlim(0, 600)
    ax_b.set_xlabel("Depth (m)")
    ax_b.set_ylabel(r"$T_\mathrm{num} - T_\mathrm{ana}$ (K)")
    rms = float(np.sqrt(np.mean(resid[res.z <= 600] ** 2)))
    ax_b.set_title(f"b   residual at 100 yr (RMS={rms:.3f} K)", loc="left", weight="bold")
    ax_b.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax_b.grid(alpha=0.25, linewidth=0.3)

    # (c) Time series at three depths.
    ax_c = axes[2]
    for z_test, colour in ((20.0, "#0066cc"), (50.0, "#cc7700"), (100.0, "#cc3333")):
        iz = int(np.argmin(np.abs(res.z - z_test)))
        ax_c.plot(t_yr, res.T[:, iz], color=colour, linewidth=1.0, label=f"z={z_test:.0f} m (num)")
        T_ana_t = carslaw_jaeger_step_analytic(
            np.array([res.z[iz]]), res.t[1:], delta_T=dT, kappa=kappa
        ).ravel()
        ax_c.plot(t_yr[1:], T_ana_t, color=colour, linewidth=0.5, linestyle="--")
    ax_c.set_xlabel("years")
    ax_c.set_ylabel(r"$T$ (K)")
    ax_c.set_title("c   time series", loc="left", weight="bold")
    ax_c.legend(loc="lower right", frameon=False, fontsize=5.5)
    ax_c.grid(alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  RMS at 100 yr = {rms:.4f} K", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
