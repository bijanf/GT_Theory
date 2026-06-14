"""Case study: 2-D lateral basin recharge-discharge cross-section.

A vertical cross-section of a continental basin in which lateral
topography drives a pressure gradient on the top boundary
(higher pressure at the recharge end, lower at the discharge end).
The pressure gradient drives horizontal Darcy flow that advects
heat laterally — a regime the 1-D vertical-column solver
structurally cannot represent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from gt_theory.solvers.domain_2d_thermo_freeze_coupled import (
    DIRICHLET,
    NEUMANN,
    BoundaryCondition,
    Domain2DResult,
    run_domain_2d,
)

YEAR_S = 365.25 * 86400.0


@dataclass(frozen=True)
class LateralBasin2DParams:
    name: str = "lateral_basin_2d"
    Lx_m: float = 10000.0  # 10 km basin width
    Lz_m: float = 1000.0  # 1 km depth
    nx: int = 25
    nz: int = 21
    duration_years: float = 500.0
    dt_per_year: float = 1.0  # annual
    porosity: float = 0.10
    lambda_r: float = 2.8
    rho_r: float = 2700.0
    c_r: float = 850.0
    K_xx: float = 1.0e-11  # high-permeability layered aquifer
    K_zz: float = 1.0e-12
    p_recharge: float = 5.0e5  # +5 bar overpressure at recharge (left)
    p_discharge: float = 0.0  # 0 bar at discharge (right)
    T_surface_warm: float = 20.0  # discharge-end surface T
    T_surface_cool: float = 5.0  # recharge-end surface T (higher altitude)
    T_init_uniform: float = 10.0
    geothermal_gradient_K_per_m: float = 0.025  # K/m
    gamma_n_alpha_scale: float = 1.0


def _build_initial_T(p: LateralBasin2DParams) -> np.ndarray:
    z = np.linspace(0.0, p.Lz_m, p.nz)
    T_init = p.T_init_uniform + p.geothermal_gradient_K_per_m * z
    return np.tile(T_init[:, None], (1, p.nx))


def _build_initial_p(p: LateralBasin2DParams) -> np.ndarray:
    # Set initial pressure to interpolate linearly between recharge and
    # discharge so the lateral pressure gradient is present from t=0.
    x = np.linspace(0.0, p.Lx_m, p.nx)
    frac = x / p.Lx_m
    p_top_lin = (1.0 - frac) * p.p_recharge + frac * p.p_discharge
    return np.tile(p_top_lin[None, :], (p.nz, 1))


def run(params: LateralBasin2DParams | None = None) -> tuple[xr.Dataset, Domain2DResult]:
    p = params if params is not None else LateralBasin2DParams()
    nt = int(round(p.duration_years * p.dt_per_year)) + 1
    duration_s = p.duration_years * YEAR_S
    dt_s = duration_s / (nt - 1)

    x_grid = np.linspace(0.0, p.Lx_m, p.nx)
    # Topographic surface temperature: linear gradient from cool
    # recharge end (high altitude) to warm discharge end.
    T_surf_lateral = p.T_surface_cool + (p.T_surface_warm - p.T_surface_cool) * x_grid / p.Lx_m
    p_surf_lateral = p.p_recharge + (p.p_discharge - p.p_recharge) * x_grid / p.Lx_m

    T_init = _build_initial_T(p)
    p_init = _build_initial_p(p)

    res = run_domain_2d(
        Lx_m=p.Lx_m,
        Lz_m=p.Lz_m,
        nx=p.nx,
        nz=p.nz,
        duration_s=duration_s,
        dt_s=dt_s,
        porosity=p.porosity,
        lambda_r=p.lambda_r,
        rho_r=p.rho_r,
        c_r=p.c_r,
        K_xx=p.K_xx,
        K_zz=p.K_zz,
        T_f=-50.0,
        dTc=0.5,  # no freezing in this case
        gamma_n_alpha_scale=p.gamma_n_alpha_scale,
        bc_top={
            "T": BoundaryCondition(DIRICHLET, T_surf_lateral),
            "p": BoundaryCondition(DIRICHLET, p_surf_lateral),
        },
        bc_bottom={
            "T": BoundaryCondition(NEUMANN, 0.08),  # 80 mW/m^2
            "p": BoundaryCondition(NEUMANN, 9810.0),  # hydrostatic
        },
        bc_left={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_right={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        T_init=T_init,
        p_init=p_init,
        picard_max_iter=10,
        picard_tol_K=0.01,
        picard_omega=1.0,
    )

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m", "x_m"), res.T),
            "p_Pa": (("time", "depth_m", "x_m"), res.p),
            "v_x": (("time", "depth_m", "x_m"), res.v_x),
            "v_z": (("time", "depth_m", "x_m"), res.v_z),
            "T_surface_at_top": (("x_m",), T_surf_lateral),
            "p_surface_at_top": (("x_m",), p_surf_lateral),
        },
        coords={
            "time": res.t,
            "depth_m": res.z,
            "x_m": res.x,
        },
        attrs={
            "case": p.name,
            "regime_expected": "2-D lateral advection (basin flow)",
            **{f"param.{k}": v for k, v in asdict(p).items()},
        },
    )
    return ds, res
