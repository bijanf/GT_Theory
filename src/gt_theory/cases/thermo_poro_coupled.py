"""Case study: thermo-poro-coupled regime (low ℒ, high Γ N_α).

A Mont-Terri-like indurated-clay column subjected to a heater-test-
style surface temperature ramp under low-permeability conditions.
Pore-fluid thermal expansion in the near-undrained limit produces
a clear pressure response that scales as Γ N_α through the solver's
``gamma_n_alpha_scale`` knob.
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
class ThermoPoroParams:
    name: str = "thermo_poro"
    depth_max_m: float = 50.0
    dz_m: float = 0.5
    duration_years: float = 2.0
    dt_per_year: int = 24
    porosity: float = 0.15
    # Solid-matrix properties.  (1-phi) rho_r c_r + phi rho_w c_w ~ 2.5e6
    # J m^-3 K^-1 reproduces the previous lumped (rho c)_eff at phi = 0.15.
    lambda_r: float = 2.5
    rho_r: float = 2700.0
    c_r: float = 815.0
    K_zz: float = 1.0e-19  # essentially undrained
    T_init_uniform: float = 16.0
    T_top_ramp_target: float = 100.0
    ramp_duration_yr: float = 1.0
    gamma_n_alpha_scale: float = 1.0
    alpha_w: float = 2.1e-4
    beta_w: float = 4.5e-10


def _build_forcing(p: ThermoPoroParams) -> tuple[np.ndarray, np.ndarray]:
    nt = int(round(p.duration_years * p.dt_per_year)) + 1
    t_s = np.linspace(0.0, p.duration_years * YEAR_S, nt)
    years = t_s / YEAR_S
    sat = p.T_init_uniform + (
        np.minimum(years / p.ramp_duration_yr, 1.0) * (p.T_top_ramp_target - p.T_init_uniform)
    )
    return t_s, sat


def run(params: ThermoPoroParams | None = None) -> tuple[xr.Dataset, ColumnFVMResult]:
    p = params if params is not None else ThermoPoroParams()
    t_s, sat = _build_forcing(p)
    dt_s = float(t_s[1] - t_s[0])
    duration_s = float(t_s[-1])

    res = run_column_fvm_permafoam(
        depth_max_m=p.depth_max_m,
        dz_m=p.dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        porosity=p.porosity,
        lambda_r=p.lambda_r,
        lambda_w=p.lambda_r,
        lambda_i=p.lambda_r,
        rho_r=p.rho_r,
        c_r=p.c_r,
        K_zz=p.K_zz,
        alpha_w=p.alpha_w,
        beta_w=p.beta_w,
        T_f=-100.0,
        dTc=0.5,  # never freezes
        gamma_n_alpha_scale=p.gamma_n_alpha_scale,
        sat=sat,
        p_top=0.0,
        T_init=p.T_init_uniform,
        p_init=0.0,
        g=0.0,
        picard_max_iter=25,
        picard_omega=0.7,
    )

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m"), res.T),
            "p_Pa": (("time", "depth_m"), res.p),
            "v_darcy": (("time", "depth_m"), res.v_darcy),
            "T_surface": (("time",), sat),
        },
        coords={
            "time": t_s,
            "depth_m": res.z,
        },
        attrs={
            "case": p.name,
            "regime_expected": "low L, high Gamma N_alpha (thermo-poro coupled)",
            "solver": "column_fvm_permafoam (upwind FV)",
            **{f"param.{k}": v for k, v in asdict(p).items()},
        },
    )
    return ds, res
