#!/usr/bin/env python3
"""N1-B: run the four canonical regime case studies and write each
to a NetCDF in ``outputs/cases/``.

Each NetCDF carries the solver state ``T_degC, p_Pa, S_i, v_darcy``
plus the boundary forcing and the case parameter set in the dataset
attributes.  The companion figures (fig04-fig07) and the regime-
diagram synthesis figure (fig08) consume these NetCDFs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gt_theory.cases import CASES
from gt_theory.diagnostics.regime import latent_heat_regime_l
from gt_theory.theory.dimless import L_FUSION, RHO_C_EFF_DEFAULT, RHO_ICE


def _compute_dimless(ds, case_name: str) -> dict[str, float]:
    """Per-case dimensionless-number signature.

    Computes (Pe_T, L, Gamma * N_alpha) using the case's solver output.
    """
    z = ds["depth_m"].values
    T = ds["T_degC"].values
    L_m = float(z.max() - z.min())
    kappa = 1.0e-6
    T_p95 = float(np.nanquantile(T, 0.95))
    T_p05 = float(np.nanquantile(T, 0.05))
    delta_T = max(T_p95 - T_p05, 0.5)
    if "v_darcy" in ds:
        # Median over (depth, time) captures the steady-state advection
        # background; using p95 picks up brief freeze-thaw transients
        # that are not the case's defining flow regime.
        v_d_char = float(np.nanmedian(np.abs(ds["v_darcy"].values)))
    else:
        v_d_char = 0.0
    Pe_T = v_d_char * L_m / kappa
    porosity = float(ds.attrs.get("param.porosity", 0.10))
    L_dim = float(
        latent_heat_regime_l(
            porosity=porosity,
            delta_T_K=delta_T,
            rho_c_eff=RHO_C_EFF_DEFAULT,
        )
    )
    # Gamma * N_alpha proxy: thermal-expansion-driven pressure scale
    # alpha_w * delta_T / beta_w divided by characteristic geological pressure.
    alpha_w = 2.1e-4
    beta_w = 4.5e-10
    p_scale = 1.0e6
    Gamma_N_alpha = (alpha_w * delta_T) / (beta_w * p_scale)
    # N_p: storage / pressure
    N_p = porosity * beta_w * p_scale
    return {
        "Pe_T": Pe_T,
        "L_calL": L_dim,
        "Gamma_N_alpha": Gamma_N_alpha,
        "N_p": N_p,
        "delta_T_K": delta_T,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/cases")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name, runner in CASES.items():
        print(f"running case: {name}")
        ds, _ = runner()
        dl = _compute_dimless(ds, name)
        for k, v in dl.items():
            ds.attrs[f"dimless.{k}"] = float(v)
        out_path = out_dir / f"{name}.nc"
        ds.to_netcdf(out_path)
        T_min, T_max = float(ds["T_degC"].min()), float(ds["T_degC"].max())
        print(
            f"  wrote {out_path}: T range [{T_min:.1f}, {T_max:.1f}] degC; "
            f"Pe_T={dl['Pe_T']:.3g}, L={dl['L_calL']:.3g}, "
            f"GammaNalpha={dl['Gamma_N_alpha']:.3g}, N_p={dl['N_p']:.3g}"
        )
        summary.append({"case": name, **dl})

    # Persist the summary as a small CSV for the regime-placement figure.
    import pandas as pd

    df = pd.DataFrame(summary)
    summary_path = out_dir / "dimless_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
