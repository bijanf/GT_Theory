"""Case study: 2-D permafrost lateral-edge effect.

A vertical cross-section of a permafrost talik with a lateral
discontinuity in surface temperature — e.g., a south-facing
slope (warm) abutting a north-facing slope (cold).  The lateral
heat conduction across the boundary produces an asymmetric
freezing front that the 1-D solver cannot represent.
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
class PermafrostEdge2DParams:
    name: str = "permafrost_edge_2d"
    Lx_m: float = 100.0
    Lz_m: float = 20.0
    nx: int = 21
    nz: int = 21
    duration_years: float = 10.0
    dt_per_year: int = 12
    porosity: float = 0.30
    lambda_r: float = 2.5
    rho_r: float = 2700.0
    c_r: float = 800.0
    K_xx: float = 1.0e-13
    K_zz: float = 1.0e-13
    T_cold: float = -8.0  # north-facing surface temperature
    T_warm: float = +3.0  # south-facing surface temperature
    edge_x_m: float = 50.0  # location of the lateral discontinuity
    T_init_uniform: float = -3.0
    gamma_n_alpha_scale: float = 0.0


def run(params: PermafrostEdge2DParams | None = None) -> tuple[xr.Dataset, Domain2DResult]:
    p = params if params is not None else PermafrostEdge2DParams()
    nt = int(round(p.duration_years * p.dt_per_year)) + 1
    duration_s = p.duration_years * YEAR_S
    dt_s = duration_s / (nt - 1)

    x_grid = np.linspace(0.0, p.Lx_m, p.nx)
    # Sharp lateral discontinuity in surface T (smoothed slightly by
    # the grid).
    T_surf = np.where(x_grid < p.edge_x_m, p.T_cold, p.T_warm)

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
        T_f=0.0,
        dTc=1.0,
        gamma_n_alpha_scale=p.gamma_n_alpha_scale,
        bc_top={
            "T": BoundaryCondition(DIRICHLET, T_surf),
            "p": BoundaryCondition(DIRICHLET, 0.0),
        },
        bc_bottom={
            "T": BoundaryCondition(NEUMANN, 0.05),
            "p": BoundaryCondition(NEUMANN, 9810.0),
        },
        bc_left={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_right={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        T_init=p.T_init_uniform,
        p_init=0.0,
        picard_max_iter=15,
        picard_tol_K=1.0e-3,
        picard_omega=0.7,
    )

    ds = xr.Dataset(
        data_vars={
            "T_degC": (("time", "depth_m", "x_m"), res.T),
            "p_Pa": (("time", "depth_m", "x_m"), res.p),
            "S_i": (("time", "depth_m", "x_m"), res.S_i),
            "v_x": (("time", "depth_m", "x_m"), res.v_x),
            "v_z": (("time", "depth_m", "x_m"), res.v_z),
            "T_surface_at_top": (("x_m",), T_surf),
        },
        coords={
            "time": res.t,
            "depth_m": res.z,
            "x_m": res.x,
        },
        attrs={
            "case": p.name,
            "regime_expected": "2-D permafrost edge effect (lateral conduction)",
            **{f"param.{k}": v for k, v in asdict(p).items()},
        },
    )
    return ds, res
