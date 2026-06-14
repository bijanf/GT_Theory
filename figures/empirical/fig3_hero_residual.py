#!/usr/bin/env python3
"""Figure 3 — hero residual [smoke].

T(z) difference between the coupled solver (s = 1) and the
uncoupled baseline (s = 0), at the final time of the smoke window,
with an envelope of three earlier snapshots showing how the
residual builds up.

The headline argument: at Umiujaq, the conduction-only kernel
leaves a depth-dependent residual that the framework's
$\\Gamma N_\\alpha$ predicts. This is the [smoke] version; the
production figure will use real Nordicana D residuals (observed
T minus solver T at each $s$).

Output: ``outputs/figures/empirical/fig3_hero_residual.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gt_theory.plotting.style import (
    NATURE_1COL_INCH,
    apply_nature_style,
)


YEAR_S = 365.25 * 86400.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        default="outputs/supersite_umiujaq/forward_runs.nc",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig3_hero_residual.pdf",
    )
    args = parser.parse_args(argv)

    ds = xr.open_dataset(Path(args.input).expanduser().resolve())
    i_on = int(np.argmin(np.abs(ds.coupling.values - 1.0)))
    i_off = int(np.argmin(np.abs(ds.coupling.values - 0.0)))

    T_on = ds["T_degC"].isel(coupling=i_on).values
    T_off = ds["T_degC"].isel(coupling=i_off).values
    z = ds["depth_m"].values
    t_yr = ds["time"].values / YEAR_S

    residual = T_on - T_off

    # Final-snapshot residual (hero curve) + envelope of three earlier
    # snapshots showing the build-up.
    snap_idx = np.linspace(0, residual.shape[0] - 1, 4).astype(int)
    envelope = residual[snap_idx[:-1]]
    final = residual[snap_idx[-1]]

    apply_nature_style()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(NATURE_1COL_INCH, NATURE_1COL_INCH),
    )

    env_lo = np.minimum.reduce(envelope, axis=0)
    env_hi = np.maximum.reduce(envelope, axis=0)
    ax.fill_betweenx(
        z,
        env_lo,
        env_hi,
        color="#c0392b",
        alpha=0.18,
        label=f"envelope (t < {t_yr[snap_idx[-1]]:.2f} yr)",
    )
    ax.plot(
        final,
        z,
        color="#c0392b",
        linewidth=1.2,
        label=f"$t = {t_yr[snap_idx[-1]]:.2f}$ yr",
    )
    ax.axvline(0.0, color="0.6", linewidth=0.5, linestyle=":")

    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{s=1} - T_{s=0}$ (K)")
    ax.set_ylabel("depth (m)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
