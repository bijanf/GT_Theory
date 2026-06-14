"""Case study: permafrost regime (high ℒ, moderate Γ N_α).

A saturated Pleistocene-sediment column with seasonal freeze-thaw
forcing superposed on a multi-decadal warming trend.  Parameters are
chosen so that the dimensionless number ℒ = ρ_i L_f φ / [(ρc)_eff ΔT]
sits firmly in the 10-100 band of Theory Table 1 (latent-heat-
dominated), and that the column exhibits a visible active-layer
seasonal freeze-thaw cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from gt_theory.solvers.column_fvm_permafoam import (
    ColumnFVMResult,
    run_column_fvm_permafoam,
)

YEAR_S = 365.25 * 86400.0


@dataclass(frozen=True)
class PermafrostParams:
    name: str = "permafrost"
    depth_max_m: float = 20.0
    dz_m: float = 0.5
    duration_years: float = 50.0
    dt_per_year: int = 12  # monthly
    porosity: float = 0.30
    lambda_r: float = 2.5
    rho_r: float = 2700.0
    c_r: float = 800.0
    K_zz: float = 1.0e-13
    T_f: float = 0.0
    dTc: float = 1.0
    T_init_surface: float = -3.0
    T_init_geothermal_gradient: float = 0.02  # K/m
    annual_amplitude_K: float = 15.0
    warming_trend_K_per_yr: float = 0.05
    gamma_n_alpha_scale: float = 1.0


def _build_forcing(p: PermafrostParams) -> tuple[np.ndarray, np.ndarray]:
    nt = int(round(p.duration_years * p.dt_per_year)) + 1
    t_s = np.linspace(0.0, p.duration_years * YEAR_S, nt)
    years = t_s / YEAR_S
    sat = (
        p.T_init_surface
        + p.warming_trend_K_per_yr * years
        + p.annual_amplitude_K * np.sin(2.0 * np.pi * years)
    )
    return t_s, sat


def run(params: PermafrostParams | None = None) -> tuple[xr.Dataset, ColumnFVMResult]:
    p = params if params is not None else PermafrostParams()
    t_s, sat = _build_forcing(p)
    dt_s = float(t_s[1] - t_s[0])
    duration_s = float(t_s[-1])

    # Cell-centred control-volume grid: nz cells at (i + 1/2) dz.
    nz = int(round(p.depth_max_m / p.dz_m))
    z = (np.arange(nz) + 0.5) * p.dz_m
    T_init = p.T_init_surface + p.T_init_geothermal_gradient * z

    res = run_column_fvm_permafoam(
        depth_max_m=p.depth_max_m,
        dz_m=p.dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        porosity=p.porosity,
        lambda_r=p.lambda_r,
        rho_r=p.rho_r,
        c_r=p.c_r,
        K_zz=p.K_zz,
        T_f=p.T_f,
        dTc=p.dTc,
        gamma_n_alpha_scale=p.gamma_n_alpha_scale,
        sat=sat,
        p_top=0.0,
        T_init=T_init,
        p_init=0.0,
        q_bot=0.05,
        picard_max_iter=25,
        picard_tol_K=1.0e-3,
        picard_omega=0.7,
    )

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m"), res.T),
            "p_Pa": (("time", "depth_m"), res.p),
            "S_i": (("time", "depth_m"), res.S_i),
            "v_darcy": (("time", "depth_m"), res.v_darcy),
            "T_surface": (("time",), sat),
        },
        coords={
            "time": t_s,
            "depth_m": res.z,
        },
        attrs={
            "case": p.name,
            "regime_expected": "high L (latent heat dominant)",
            "solver": "column_fvm_permafoam (upwind FV)",
            **{f"param.{k}": v for k, v in asdict(p).items()},
        },
    )
    return ds, res
