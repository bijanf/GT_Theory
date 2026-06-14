#!/usr/bin/env python3
"""Mont Terri HE-D forward-run driver: ΓN_α coupling sweep against a
heater-style boundary condition.

Loads ``data/supersite_mont_terri/site_config.yaml`` and runs
``run_column_coupled`` across the same ``gamma_n_alpha_scale`` sweep
used at Umiujaq. The forcing source is the published HE-D heater
schedule (Garitte et al. 2017, Fig 5), reconstructed as a piecewise-
linear T(t) ramp from 15 °C to 100 °C over 18 months.

The smoke-mode path uses the existing Cartesian solver as a stand-in
for the eventual cylindrical/radial solver — the absolute Δp will not
be quantitatively faithful to the real HE-D geometry (factor-of-π
correction from the cylindrical Laplacian) but the *qualitative
demonstration* that ΓN_α drives a multi-MPa pore-pressure rise that
``coupling=0`` cannot produce is preserved.

Output: ``outputs/supersite_mont_terri/forward_runs.nc`` plus a
stderr summary of Δp_rms at a pseudo-thermocouple radius.

Usage::

    python scripts/run_mont_terri_forward.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from gt_theory.solvers import CoupledResult, run_column_coupled

YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def column_scalar_properties(cfg: dict[str, Any]) -> dict[str, float]:
    layer = cfg["physics"]["layers"][0]  # only one layer at Mont Terri
    fluid = cfg["physics"]["fluid"]
    K_hyd = float(layer["K_zz_m_per_s"])
    mu = float(fluid["mu_Pa_s"])
    rho_w = float(fluid["rho_w_kg_per_m3"])
    g = float(fluid["g_m_per_s2"])
    return {
        "lambda_thermal": float(layer["lambda_thermal_W_per_m_per_K"]),
        "rho_c_eff": float(layer["rho_c_eff_J_per_m3_per_K"]),
        "K_zz": K_hyd * mu / (rho_w * g),  # intrinsic permeability (m^2)
        "mu": mu,
        "porosity": float(layer["porosity"]),
        "alpha_w": float(fluid["alpha_w_per_K"]),
        "beta_w": float(fluid["beta_w_per_Pa"]),
        "rho_w": rho_w,
        "g": g,
        "c_w": float(fluid["c_w_J_per_kg_per_K"]),
    }


def heater_schedule_C(nt: int, dt_s: float) -> np.ndarray:
    """Reconstruct the HE-D inner-wall T forcing from Garitte 2017 Fig 5.

    Piecewise-linear: 0-30 d ramp 15 → 60 °C; 30-90 d ramp 60 → 100 °C;
    90-460 d hold at 100 °C; 460-540 d ramp down 100 → 30 °C.
    """
    t_d = np.arange(nt) * dt_s / DAY_S
    T = np.where(
        t_d < 30.0,
        15.0 + (60.0 - 15.0) * t_d / 30.0,
        np.where(
            t_d < 90.0,
            60.0 + (100.0 - 60.0) * (t_d - 30.0) / 60.0,
            np.where(
                t_d < 460.0,
                100.0,
                np.where(
                    t_d < 540.0,
                    100.0 + (30.0 - 100.0) * (t_d - 460.0) / 80.0,
                    30.0,
                ),
            ),
        ),
    )
    return T


def run_coupling_sweep(
    *,
    cfg: dict[str, Any],
    sat: np.ndarray,
    dt_s: float,
    duration_s: float,
    coupling_grid: list[float],
    depth_max_m: float,
) -> xr.Dataset:
    props = column_scalar_properties(cfg)
    dz_m = float(cfg["column"]["dz_m"])
    z_grid = np.arange(0.0, depth_max_m + 0.5 * dz_m, dz_m)
    T_init = np.full_like(z_grid, 15.0)
    p_init = np.full_like(z_grid, 1.0e6)

    T_stack: list[np.ndarray] = []
    p_stack: list[np.ndarray] = []
    v_stack: list[np.ndarray] = []
    z_ref: np.ndarray | None = None
    t_ref: np.ndarray | None = None

    for s in coupling_grid:
        result: CoupledResult = run_column_coupled(
            depth_max_m=depth_max_m,
            dz_m=dz_m,
            duration_s=duration_s,
            dt_s=dt_s,
            **props,
            gamma_n_alpha_scale=s,
            sat=sat,
            p_top=0.0,  # top-p = far-field anomaly = 0
            T_init=T_init,
            p_init=p_init,
        )
        if z_ref is None:
            z_ref = result.z
            t_ref = result.t
        T_stack.append(result.T)
        p_stack.append(result.p)
        v_stack.append(result.v_darcy)

    assert z_ref is not None and t_ref is not None

    return xr.Dataset(
        data_vars={
            "T_degC": (("coupling", "time", "depth_m"), np.stack(T_stack)),
            "p_Pa": (("coupling", "time", "depth_m"), np.stack(p_stack)),
            "v_darcy_m_s": (("coupling", "time", "depth_m"), np.stack(v_stack)),
            "heater_T_C": (("time",), sat),
        },
        coords={
            "coupling": np.asarray(coupling_grid, dtype=float),
            "time": t_ref,
            "depth_m": z_ref,
        },
        attrs={
            "site": cfg["site"]["name"],
            "site_long_name": cfg["site"]["long_name"],
            "operator": cfg["site"]["operator"],
            "dataset_class": cfg["site"]["class"],
            "geometry_smoke": cfg["column"].get("geometry_smoke", "cartesian"),
            "geometry_target": cfg["column"]["geometry"],
            "note": (
                "Smoke run: Cartesian solver as a stand-in for the "
                "1-D radial HE-D geometry; quantitative Δp is not radial-"
                "faithful but the qualitative coupling-vs-uncoupled gap is preserved."
            ),
        },
    )


def summarise(ds: xr.Dataset, *, probe_depth_m: float = 0.3) -> str:
    z = ds.depth_m.values
    iz = int(np.argmin(np.abs(z - probe_depth_m)))
    z_used = float(z[iz])

    s_values = ds.coupling.values
    i_off = int(np.argmin(np.abs(s_values - 0.0)))
    i_on = int(np.argmin(np.abs(s_values - 1.0)))

    T_on = ds.T_degC.isel(coupling=i_on, depth_m=iz).values
    p_on = ds.p_Pa.isel(coupling=i_on, depth_m=iz).values
    p_off = ds.p_Pa.isel(coupling=i_off, depth_m=iz).values

    p_on_max = float(np.max(p_on))
    p_off_max = float(np.max(p_off))
    dp_rms = float(np.sqrt(np.mean((p_on - p_off) ** 2)))
    T_on_max = float(np.max(T_on))

    return "\n".join(
        [
            f"Mont Terri HE-D forward sweep — probe depth z={z_used:.2f} m",
            f"  peak rock T (coupled): {T_on_max:.1f} degC",
            f"  peak pore pressure  (coupled):   {p_on_max / 1e6:.2f} MPa",
            f"  peak pore pressure  (uncoupled): {p_off_max / 1e6:.2f} MPa",
            f"  Δp_rms (coupled vs uncoupled):   {dp_rms / 1e6:.3f} MPa",
            f"  reference (Garitte 2017 digitized): ~1 → ~4 MPa at peak heating",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/supersite_mont_terri/site_config.yaml")
    parser.add_argument(
        "--depth-max",
        type=float,
        default=20.0,
        help="Pseudo-radial column extent; large enough that the thermal front "
        "doesn't reach the bottom in the 18-month campaign.",
    )
    parser.add_argument("--years", type=float, default=1.6)
    parser.add_argument("--dt-days", type=float, default=1.0)
    parser.add_argument("--out", default="outputs/supersite_mont_terri/forward_runs.nc")
    parser.add_argument(
        "--probe-depth",
        type=float,
        default=0.3,
        help="Pseudo-radial distance (m) for the stderr summary "
        "(HE-D thermocouples were at 0.1-1.5 m).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sweep = cfg.get("ablation_sweep", {}).get("gamma_n_alpha_scale", [0.0, 0.25, 0.5, 0.75, 1.0])

    dt_s = args.dt_days * DAY_S
    duration_s = args.years * YEAR_S
    nt = int(round(duration_s / dt_s)) + 1
    sat = heater_schedule_C(nt, dt_s)

    ds = run_coupling_sweep(
        cfg=cfg,
        sat=sat,
        dt_s=dt_s,
        duration_s=duration_s,
        coupling_grid=list(sweep),
        depth_max_m=args.depth_max,
    )
    ds.attrs["forcing_source"] = (
        "Garitte 2017 HE-D heater schedule (reconstructed piecewise-linear)"
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    print(summarise(ds, probe_depth_m=args.probe_depth), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
