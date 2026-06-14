#!/usr/bin/env python3
"""Utah FORGE 16A forward-run driver: ΓN_α coupling sweep against a
circulation-test injection-step boundary condition.

Loads ``data/supersite_forge/site_config.yaml`` and runs
``run_column_coupled`` over a 5-week injection step from the 2024
Extended Circulation Tests (DOE GDR submission 1608). The forcing is
reconstructed as a step-down injection of 60 °C water into the
otherwise 225 °C bottomhole environment.

Smoke-mode caveat: the coupled solver is 1-D Cartesian; the
injector-producer 3-D fracture network at FORGE is the actual
geometry. We use the column as a stand-in for the injector wellbore
and trust that the qualitative ΓN_α gap (Δp from thermal expansion of
the pore fluid) is preserved.

Output: ``outputs/supersite_forge/forward_runs.nc``.

Usage::

    python scripts/run_forge_forward.py
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
    # Use the granite reservoir layer (deeper, thicker) — the smoke-mode
    # column averages over the full 3.3 km, but the reservoir properties
    # dominate.
    layer = [layer for layer in cfg["physics"]["layers"] if layer["name"] == "granite_reservoir"][0]
    fluid = cfg["physics"]["fluid"]
    K_hyd = float(layer["K_zz_m_per_s"])
    mu = float(fluid["mu_Pa_s"])
    rho_w = float(fluid["rho_w_kg_per_m3"])
    g = float(fluid["g_m_per_s2"])
    return {
        "lambda_thermal": float(layer["lambda_thermal_W_per_m_per_K"]),
        "rho_c_eff": float(layer["rho_c_eff_J_per_m3_per_K"]),
        "K_zz": K_hyd * mu / (rho_w * g),
        "mu": mu,
        "porosity": float(layer["porosity"]),
        "alpha_w": float(fluid["alpha_w_per_K"]),
        "beta_w": float(fluid["beta_w_per_Pa"]),
        "rho_w": rho_w,
        "g": g,
        "c_w": float(fluid["c_w_J_per_kg_per_K"]),
    }


def injection_step_C(nt: int, dt_s: float) -> np.ndarray:
    """Synthetic cold-injection step: bottomhole T drops from 225 to
    60 °C at t = 1 day, holds for 30 days, then ramps back to 225 °C
    by t = 35 days."""
    t_d = np.arange(nt) * dt_s / DAY_S
    T = np.where(
        t_d < 1.0,
        225.0,
        np.where(
            t_d < 30.0,
            60.0,
            np.where(t_d < 35.0, 60.0 + (225.0 - 60.0) * (t_d - 30.0) / 5.0, 225.0),
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
) -> xr.Dataset:
    props = column_scalar_properties(cfg)
    depth_max_m = float(cfg["column"]["depth_max_m"])
    dz_m = float(cfg["column"]["dz_m"])
    z_grid = np.arange(0.0, depth_max_m + 0.5 * dz_m, dz_m)
    # Linear T(z) from 15 °C at surface to 225 °C at bottom.
    T_init = 15.0 + (225.0 - 15.0) * z_grid / depth_max_m
    # Hydrostatic pressure 0 at top, ρgh at bottom. Solver wants anomaly:
    # set p_init = 0 (interpreted as anomaly relative to the column's
    # hydrostatic reference) and rely on the Neumann-hydrostatic bottom BC.
    p_init = np.zeros_like(z_grid)

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
            sat=sat,  # bottomhole-driven step; we use 'sat' as the active boundary
            p_top=0.0,
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
            "injection_T_C": (("time",), sat),
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
            "geometry": cfg["column"]["geometry"],
            "note": (
                "Smoke run: 1-D Cartesian column as a stand-in for the FORGE "
                "injector wellbore + fracture network; quantitative Δp is not "
                "3-D-faithful."
            ),
        },
    )


def summarise(ds: xr.Dataset, *, probe_depth_m: float = 2500.0) -> str:
    z = ds.depth_m.values
    iz = int(np.argmin(np.abs(z - probe_depth_m)))
    z_used = float(z[iz])

    s_values = ds.coupling.values
    i_off = int(np.argmin(np.abs(s_values - 0.0)))
    i_on = int(np.argmin(np.abs(s_values - 1.0)))

    T_on = ds.T_degC.isel(coupling=i_on, depth_m=iz).values
    p_on = ds.p_Pa.isel(coupling=i_on, depth_m=iz).values
    p_off = ds.p_Pa.isel(coupling=i_off, depth_m=iz).values

    return "\n".join(
        [
            f"Utah FORGE forward sweep — probe depth z={z_used:.0f} m",
            f"  ΔT (coupled, max excursion at probe): {float(np.ptp(T_on)):.1f} K",
            f"  peak |Δp| (coupled):   {float(np.max(np.abs(p_on))) / 1e6:.2f} MPa",
            f"  peak |Δp| (uncoupled): {float(np.max(np.abs(p_off))) / 1e6:.2f} MPa",
            f"  Δp_rms (coupled vs uncoupled): "
            f"{float(np.sqrt(np.mean((p_on - p_off) ** 2))) / 1e6:.3f} MPa",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/supersite_forge/site_config.yaml")
    parser.add_argument("--years", type=float, default=0.12)  # ~6 weeks
    parser.add_argument("--dt-days", type=float, default=0.1)  # ~2.4 hour solver step
    parser.add_argument("--out", default="outputs/supersite_forge/forward_runs.nc")
    parser.add_argument(
        "--probe-depth",
        type=float,
        default=100.0,
        help="Probe depth (m) for the stderr summary. Default 100 m: the "
        "thermal diffusion depth in granite over a 6-week injection step. "
        "The 2.5 km reservoir signature requires a multi-year run or a "
        "fracture-network conduit (out of scope for the 1-D smoke).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    sweep = cfg.get("ablation_sweep", {}).get("gamma_n_alpha_scale", [0.0, 0.25, 0.5, 0.75, 1.0])

    dt_s = args.dt_days * DAY_S
    duration_s = args.years * YEAR_S
    nt = int(round(duration_s / dt_s)) + 1
    sat = injection_step_C(nt, dt_s)

    ds = run_coupling_sweep(
        cfg=cfg,
        sat=sat,
        dt_s=dt_s,
        duration_s=duration_s,
        coupling_grid=list(sweep),
    )
    ds.attrs["forcing_source"] = "2024 Extended Circulation Tests step-injection (reconstructed)"

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    print(summarise(ds, probe_depth_m=args.probe_depth), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
