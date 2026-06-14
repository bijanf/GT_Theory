#!/usr/bin/env python3
"""Umiujaq forward-run driver: ΓN_α coupling sweep on the Tasiapik
talik column.

Loads ``data/supersite_umiujaq/site_config.yaml``, builds a
column-mean property set from the silt-aquifer-talik layer (the
dominant advective layer; underlying frozen silt is eight orders of
magnitude less permeable and is treated as a no-flow base), and runs
``run_column_coupled`` across ``gamma_n_alpha_scale ∈ {0, 0.25, 0.5,
0.75, 1.0}``.

Surface forcing source:

- ``--forcing nordicana`` — reads
  ``data/supersite_umiujaq/processed/d9_VDTSYBU_Tsurf_daily.nc``
  (operator-prepared from the Nordicana D9 zips per the README).
- ``--forcing synthetic`` (default for smoke runs) — builds a
  realistic SAT-style series for Umiujaq: annual mean −3 °C, seasonal
  amplitude 15 °C, daily noise σ=0.5 K, plus a 0.5 K/decade linear
  warming trend over the 10-year window. The seasonal amplitude and
  mean are taken from the Nordicana D9 station summary; the linear
  trend is the Arctic-amplified land-warming rate from IPCC AR6 WG1
  Atlas.

Output: ``outputs/supersite_umiujaq/forward_runs.nc`` — five solver
runs stacked along a ``coupling`` coord. A stderr summary reports
the RMS difference of T(z=5 m) and head(z=5 m) between the coupled
and uncoupled limits.

Usage::

    python scripts/run_umiujaq_forward.py \\
        --config data/supersite_umiujaq/site_config.yaml \\
        --forcing synthetic \\
        --years 10 \\
        --out outputs/supersite_umiujaq/forward_runs.nc
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

YEAR_S: float = 365.25 * 86400.0
DAY_S: float = 86400.0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse the supersite YAML config."""
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _talik_layer(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the silt-aquifer-talik layer dict (the dominant advective
    layer in the Umiujaq Tasiapik column)."""
    for layer in cfg["physics"]["layers"]:
        if layer["name"] == "silt_aquifer_talik":
            return layer
    raise KeyError("site_config.yaml has no 'silt_aquifer_talik' layer")


def column_scalar_properties(cfg: dict[str, Any]) -> dict[str, float]:
    """Pick the talik silt-aquifer properties as column-scalar inputs
    for ``run_column_coupled``.

    Limitation: the solver takes scalar material properties, so we
    cannot express the layered column directly. We trim the column to
    the talik depth and treat the underlying frozen silt as a no-flow
    base via the default ``neumann_hydrostatic`` bottom-p BC.
    """
    layer = _talik_layer(cfg)
    fluid = cfg["physics"]["fluid"]
    # The solver wants intrinsic permeability k (m^2), not hydraulic
    # conductivity K (m/s).  The site_config records K_zz_m_per_s; convert.
    K_hyd = float(layer["K_zz_m_per_s"])
    mu = float(fluid["mu_Pa_s"])
    rho_w = float(fluid["rho_w_kg_per_m3"])
    g = float(fluid["g_m_per_s2"])
    k_intrinsic = K_hyd * mu / (rho_w * g)

    return {
        "lambda_thermal": float(layer["lambda_thermal_W_per_m_per_K"]),
        "rho_c_eff": float(layer["rho_c_eff_J_per_m3_per_K"]),
        "K_zz": k_intrinsic,
        "mu": mu,
        "porosity": float(layer["porosity"]),
        "alpha_w": float(fluid["alpha_w_per_K"]),
        "beta_w": float(fluid["beta_w_per_Pa"]),
        "rho_w": rho_w,
        "g": g,
        "c_w": float(fluid["c_w_J_per_kg_per_K"]),
    }


def column_geometry(cfg: dict[str, Any]) -> dict[str, float]:
    """Trim the run to the talik base depth so the no-flow assumption
    at the underlying frozen silt is the natural bottom BC."""
    talik = _talik_layer(cfg)
    return {
        "depth_max_m": float(talik["z_bottom_m"]),
        "dz_m": float(cfg["column"]["dz_m"]),
    }


# ---------------------------------------------------------------------------
# Surface forcing
# ---------------------------------------------------------------------------


def synthetic_forcing(
    nt: int,
    dt_s: float,
    *,
    annual_mean_C: float = -3.0,
    seasonal_amp_C: float = 15.0,
    warming_K_per_decade: float = 0.5,
    noise_sigma_C: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Realistic SAT-style ground-surface T series for Umiujaq.

    Builds annual sinusoid + linear warming trend + Gaussian daily
    noise. Values returned in °C anomaly relative to 0 °C reference;
    the solver treats the field as a generic T-anomaly so absolute
    Kelvin / Celsius is a labelling choice.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt_s
    seasonal = -seasonal_amp_C * np.cos(2.0 * np.pi * t / YEAR_S)  # min in mid-January
    trend = warming_K_per_decade * (t / (10.0 * YEAR_S))
    noise = rng.normal(0.0, noise_sigma_C, size=nt)
    return annual_mean_C + seasonal + trend + noise


def nordicana_forcing(path: str | Path, nt: int, dt_s: float) -> np.ndarray:
    """Load operator-prepared D9 daily GST netCDF and resample onto the
    solver time grid by linear interpolation."""
    ds = xr.open_dataset(path)
    var = [v for v in ds.data_vars if "T" in v.upper()][0]
    series = ds[var].values.astype(float)
    n_obs = series.size
    # Interpolate from observation time index onto solver index.
    obs_x = np.linspace(0.0, 1.0, n_obs)
    solver_x = np.linspace(0.0, 1.0, nt)
    return np.interp(solver_x, obs_x, series)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def run_coupling_sweep(
    *,
    cfg: dict[str, Any],
    gst: np.ndarray,
    dt_s: float,
    duration_s: float,
    coupling_grid: list[float],
) -> xr.Dataset:
    """Run the solver across ``coupling_grid`` with shared geometry,
    properties, and GST forcing. Returns a Dataset of (coupling, time,
    depth) arrays for T, p, v_darcy."""
    geom = column_geometry(cfg)
    props = column_scalar_properties(cfg)

    # Initial conditions: linear from surface GST to base T at -2 °C.
    z_grid = np.arange(0.0, geom["depth_max_m"] + 0.5 * geom["dz_m"], geom["dz_m"])
    T_init = np.linspace(gst[0], -2.0, z_grid.size)
    p_init = np.zeros_like(z_grid)  # gauge pressure; bottom BC keeps it hydrostatic

    T_stack = []
    p_stack = []
    v_stack = []
    z_ref: np.ndarray | None = None
    t_ref: np.ndarray | None = None

    for s in coupling_grid:
        result: CoupledResult = run_column_coupled(
            depth_max_m=geom["depth_max_m"],
            dz_m=geom["dz_m"],
            duration_s=duration_s,
            dt_s=dt_s,
            **props,
            gamma_n_alpha_scale=s,
            sat=gst,
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

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("coupling", "time", "depth_m"), np.stack(T_stack)),
            "p_Pa": (("coupling", "time", "depth_m"), np.stack(p_stack)),
            "v_darcy_m_s": (("coupling", "time", "depth_m"), np.stack(v_stack)),
            "gst_degC": (("time",), gst),
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
            "lat_deg": cfg["site"]["lat_deg"],
            "lon_deg": cfg["site"]["lon_deg"],
            "dataset_class": cfg["site"]["class"],
            "config_path": "data/supersite_umiujaq/site_config.yaml",
            "talik_K_hyd_m_s": float(_talik_layer(cfg)["K_zz_m_per_s"]),
            "talik_porosity": float(_talik_layer(cfg)["porosity"]),
            "coupling_grid_meaning": (
                "gamma_n_alpha_scale: scalar multiplier on both cross-coupling "
                "terms (energy advection and thermal expansion); 0 = uncoupled limit"
            ),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------


def summarise(ds: xr.Dataset, *, probe_depth_m: float = 5.0) -> str:
    """One-paragraph stderr summary: RMS(T) and RMS(p) at the
    piezometer-screen depth, coupled vs uncoupled."""
    z = ds.depth_m.values
    iz = int(np.argmin(np.abs(z - probe_depth_m)))
    z_used = float(z[iz])

    s_values = ds.coupling.values
    i_off = int(np.argmin(np.abs(s_values - 0.0)))
    i_on = int(np.argmin(np.abs(s_values - 1.0)))

    T_off = ds.T_degC.isel(coupling=i_off, depth_m=iz).values
    T_on = ds.T_degC.isel(coupling=i_on, depth_m=iz).values
    p_off = ds.p_Pa.isel(coupling=i_off, depth_m=iz).values
    p_on = ds.p_Pa.isel(coupling=i_on, depth_m=iz).values

    def _rms(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    dT_rms = _rms(T_on, T_off)
    dp_rms = _rms(p_on, p_off)
    v_on_mean = float(np.mean(np.abs(ds.v_darcy_m_s.isel(coupling=i_on, depth_m=iz).values)))

    nt = ds.sizes["time"]
    n_coup = ds.sizes["coupling"]
    lines = [
        f"Umiujaq forward sweep — {n_coup} coupling levels × {nt} time steps",
        f"  probe depth: z={z_used:.2f} m (target {probe_depth_m} m)",
        f"  ΔT_rms (coupled vs uncoupled): {dT_rms:.4f} K",
        f"  Δp_rms (coupled vs uncoupled): {dp_rms:.4f} Pa",
        f"  |v_darcy| at probe (coupled): {v_on_mean:.3e} m/s",
        f"  K_hyd talik (m/s): {ds.attrs['talik_K_hyd_m_s']:.2e}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="data/supersite_umiujaq/site_config.yaml",
        help="Path to supersite YAML config",
    )
    parser.add_argument(
        "--forcing",
        choices=["synthetic", "nordicana"],
        default="synthetic",
        help="GST forcing source",
    )
    parser.add_argument(
        "--forcing-file",
        default="data/supersite_umiujaq/processed/d9_VDTSYBU_Tsurf_daily.nc",
        help="Path to operator-prepared D9 forcing netCDF (when --forcing=nordicana)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Integration duration in years (default 10)",
    )
    parser.add_argument(
        "--dt-days",
        type=float,
        default=1.0,
        help="Solver time step in days (default 1 day)",
    )
    parser.add_argument(
        "--out",
        default="outputs/supersite_umiujaq/forward_runs.nc",
        help="Output netCDF path",
    )
    parser.add_argument(
        "--probe-depth",
        type=float,
        default=5.0,
        help="Depth (m) for the stderr RMS summary",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    sweep = cfg.get("ablation_sweep", {}).get("gamma_n_alpha_scale", [0.0, 0.25, 0.5, 0.75, 1.0])

    dt_s = float(args.dt_days) * DAY_S
    duration_s = float(args.years) * YEAR_S
    nt = int(round(duration_s / dt_s)) + 1

    if args.forcing == "synthetic":
        gst = synthetic_forcing(nt, dt_s)
    else:
        gst = nordicana_forcing(args.forcing_file, nt, dt_s)

    ds = run_coupling_sweep(
        cfg=cfg,
        gst=gst,
        dt_s=dt_s,
        duration_s=duration_s,
        coupling_grid=list(sweep),
    )
    ds.attrs["forcing_source"] = args.forcing
    ds.attrs["forcing_synthetic_params"] = (
        "annual_mean=-3C, seasonal_amp=15C, warming=0.5K/decade, sigma=0.5K"
        if args.forcing == "synthetic"
        else "operator-prepared D9"
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    print(summarise(ds, probe_depth_m=args.probe_depth), file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
