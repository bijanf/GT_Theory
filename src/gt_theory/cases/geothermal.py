"""Case study: geothermal-reservoir regime (high Pe_T, low ℒ).

A granite enhanced-geothermal-system (EGS) column with a strong,
*sustained* upward Darcy flow.  Far above 0 °C throughout, so
freeze-thaw plays no role.

Solver choice.  This is the one regime in the suite that is advection
dominated (cell Peclet > 1 at the production grid).  The centred-
difference Crank-Nicolson solvers (``column_coupled`` /
``column_thermo_freeze_coupled``) develop spurious oscillations and
overflow once the flow is strong enough to be a genuine high-Pe_T
showcase, so this case is run with the **first-order-upwind finite-
volume solver** ``column_fvm_permafoam``, which is unconditionally
stable for advection.  (An earlier version of this case used a
hydrostatic-Neumann bottom boundary, which cannot sustain steady
through-flow; the flow only persisted because of a hydraulic-
diffusivity error that has since been corrected.  The sustained head
below, with a fixed bottom pressure, drives genuine steady advection.)
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
class GeothermalParams:
    name: str = "geothermal"
    depth_max_m: float = 2000.0
    dz_m: float = 20.0
    duration_years: float = 30.0
    dt_per_year: int = 12
    porosity: float = 0.02
    lambda_thermal: float = 3.0
    rho_r: float = 2700.0
    c_r: float = 1000.0  # (1-phi) rho_r c_r + phi rho_w c_w ~ 2.7e6
    K_zz: float = 1.0e-15
    mu: float = 1.0e-3
    v_darcy_imposed: float = 1.0e-7  # m/s; sustained via a fixed head
    T_init_surface: float = 15.0
    T_init_geothermal_gradient: float = 0.1  # K/m (200 K at 2000 m)
    annual_amplitude_K: float = 5.0
    q_bot: float = 0.10  # mantle heat flux (W m^-2)
    gamma_n_alpha_scale: float = 1.0


def _build_forcing(p: GeothermalParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nt = int(round(p.duration_years * p.dt_per_year)) + 1
    t_s = np.linspace(0.0, p.duration_years * YEAR_S, nt)
    years = t_s / YEAR_S
    sat = p.T_init_surface + p.annual_amplitude_K * np.sin(2.0 * np.pi * years)
    # Sustained head: choose the top-boundary pressure so the steady
    # gradient drives the target Darcy velocity.  The bottom is held at a
    # fixed pressure (Dirichlet), so the head difference -- and hence the
    # through-flow -- persists for the whole run (unlike a hydrostatic
    # Neumann base, which relaxes to zero flux).
    dp_drive = p.v_darcy_imposed * p.mu / p.K_zz * p.depth_max_m
    p_top = -dp_drive * np.ones_like(t_s)
    return t_s, sat, p_top


def run(params: GeothermalParams | None = None) -> tuple[xr.Dataset, ColumnFVMResult]:
    p = params if params is not None else GeothermalParams()
    t_s, sat, p_top = _build_forcing(p)
    dt_s = float(t_s[1] - t_s[0])
    duration_s = float(t_s[-1])

    nz = int(round(p.depth_max_m / p.dz_m))
    z = (np.arange(nz) + 0.5) * p.dz_m
    T_init = p.T_init_surface + p.T_init_geothermal_gradient * z

    res = run_column_fvm_permafoam(
        depth_max_m=p.depth_max_m,
        dz_m=p.dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        porosity=p.porosity,
        lambda_r=p.lambda_thermal,
        lambda_w=p.lambda_thermal,
        lambda_i=p.lambda_thermal,
        rho_r=p.rho_r,
        c_r=p.c_r,
        K_zz=p.K_zz,
        mu=p.mu,
        T_f=-100.0,
        dTc=0.5,  # never freezes
        gamma_n_alpha_scale=p.gamma_n_alpha_scale,
        sat=sat,
        p_top=p_top,
        T_init=T_init,
        p_init=0.0,
        q_bot=p.q_bot,
        bot_p_bc="dirichlet",
        p_bot=0.0,  # fixed bottom pressure -> sustained head
        picard_max_iter=25,
        picard_omega=0.7,
    )

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m"), res.T),
            "p_Pa": (("time", "depth_m"), res.p),
            "v_darcy": (("time", "depth_m"), res.v_darcy),
            "T_surface": (("time",), sat),
            "p_top": (("time",), p_top),
        },
        coords={
            "time": t_s,
            "depth_m": res.z,
        },
        attrs={
            "case": p.name,
            "regime_expected": "high Pe_T (advection dominant)",
            "solver": "column_fvm_permafoam (upwind FV)",
            **{f"param.{k}": v for k, v in asdict(p).items()},
        },
    )
    return ds, res
