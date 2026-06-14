#!/usr/bin/env python3
"""R18-D: compute ℒ per Theory Eq. 279 at each of the 948 Huang-Pollack
sites and flag sites in the latent-heat-dominated regime (ℒ > 1) where
the original pure-sensible solver ``column_coupled.py`` is inadequate.

Input
-----
``outputs/global/ensemble_summary.parquet`` -- 948-site posterior
summary from the existing global inversion (R3 R59).

Output
------
``outputs/global/l_regime.parquet`` -- per-site ℒ, regime label,
flag for sites where freeze-thaw closure is load-bearing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.diagnostics.regime import (
    RegimeBands,
    classify_regime,
    latent_heat_regime_l,
)
from gt_theory.theory.dimless import RHO_C_EFF_DEFAULT


# Lithology-default porosity: Tibetan / Quaternary sites use higher
# porosity than crystalline-crust Huang-Pollack defaults.  Approximate
# bands by latitude: |phi| > 60 -> Pleistocene/Quaternary (phi ~ 0.30);
# |phi| 30-60 -> mixed crust (phi ~ 0.15); |phi| < 30 -> mixed
# (phi ~ 0.10).  This is the same default convention used by
# theory/dimless.py.
def _porosity_default(lat_deg: np.ndarray) -> np.ndarray:
    """Lithology-default porosity from latitude.

    Low- and mid-latitudes are dominated by crystalline continental
    crust (igneous + metamorphic) with bulk porosity ~ 0.03-0.05
    (Manning & Ingebritsen 1999); high-latitude sites overlie
    Pleistocene/Quaternary sediments and permafrost with bulk
    porosity ~ 0.30 (Walvoord & Striegl 2007).  These are the same
    defaults used by ``theory/dimless.py``.
    """
    abs_lat = np.abs(lat_deg)
    phi = np.full_like(abs_lat, 0.03)
    phi[(abs_lat >= 30.0) & (abs_lat < 60.0)] = 0.05
    phi[abs_lat >= 60.0] = 0.30
    return phi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in",
        dest="in_path",
        default="outputs/global/ensemble_summary.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/global/l_regime.parquet",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.in_path).expanduser().resolve())
    lat = df["lat_deg"].to_numpy()
    porosity = _porosity_default(lat)

    # Characteristic Delta_T is the *seasonal* SAT amplitude at the
    # site, not the long-term anomaly.  We use a simple latitude-band
    # proxy based on the Hartmann (2016) ERA-Interim climatology
    # (Eq. 7.3): tropical = 5 K, mid-lat = 18 K, boreal/Arctic = 28 K.
    # This is the correct scale for ℒ per theory Eq. 279.
    abs_lat = np.abs(lat)
    delta_T = np.where(
        abs_lat < 23.5,
        5.0,
        np.where(abs_lat < 60.0, 18.0, 28.0),
    )

    l_values = latent_heat_regime_l(
        porosity=porosity,
        delta_T_K=delta_T,
        rho_c_eff=RHO_C_EFF_DEFAULT,
    )
    bands = RegimeBands()
    regime = classify_regime(l_values, bands)

    out = df[["site_id", "lat_deg", "lon_deg"]].copy()
    out["porosity_assumed"] = porosity
    out["delta_T_K"] = delta_T
    out["L_calL"] = l_values
    out["regime"] = regime
    out["needs_freeze_thaw_closure"] = l_values > 1.0

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)

    n_perm = int(out["regime"].eq("permafrost").sum())
    n_inter = int(out["regime"].eq("intermediate").sum())
    n_arid = int(out["regime"].eq("arid").sum())
    n_lf = int(out["needs_freeze_thaw_closure"].sum())
    print(f"wrote {out_path}")
    print(f"  N sites: {len(out)}")
    print(f"  permafrost (ℒ >= 1):     {n_perm:4d}  ({100 * n_perm / len(out):.1f}%)")
    print(f"  intermediate (0.1-1):    {n_inter:4d}  ({100 * n_inter / len(out):.1f}%)")
    print(f"  arid (< 0.1):            {n_arid:4d}  ({100 * n_arid / len(out):.1f}%)")
    print(f"  needs freeze-thaw closure (ℒ > 1): {n_lf} ({100 * n_lf / len(out):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
