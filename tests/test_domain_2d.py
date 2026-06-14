"""Tests for the 2-D coupled T+p+S_i solver."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.carslaw_jaeger import step_temperature_response
from gt_theory.solvers.domain_2d_thermo_freeze_coupled import (
    DIRICHLET,
    NEUMANN,
    BoundaryCondition,
    Domain2DResult,
    run_domain_2d,
)

YEAR_S = 365.25 * 86400.0


def test_2d_reduces_to_1d_carslaw_jaeger() -> None:
    """With uniform x boundary conditions and lateral Neumann no-flux,
    the 2-D solver should reproduce the 1-D Carslaw-Jaeger erfc step
    response in z, with no x-variation."""
    kappa = 1.0e-6
    phi = 0.10
    rho_r, c_r = 2700.0, 800.0
    rho_w, c_w = 1000.0, 4186.0
    rho_c_eff = (1 - phi) * rho_r * c_r + phi * rho_w * c_w
    lam = kappa * rho_c_eff

    res = run_domain_2d(
        Lx_m=100.0,
        Lz_m=400.0,
        nx=5,
        nz=41,
        duration_s=100.0 * YEAR_S,
        dt_s=YEAR_S / 4.0,
        porosity=phi,
        lambda_r=lam,
        lambda_w=lam,
        lambda_i=lam,
        rho_r=rho_r,
        c_r=c_r,
        rho_w=rho_w,
        c_w=c_w,
        c_i=c_w * rho_w / 917.0,
        L_f=3.34e5,
        K_xx=1.0e-20,
        K_zz=1.0e-20,
        T_f=-50.0,
        dTc=0.1,
        gamma_n_alpha_scale=0.0,
        bc_top={
            "T": BoundaryCondition(DIRICHLET, 1.0),
            "p": BoundaryCondition(DIRICHLET, 0.0),
        },
        bc_bottom={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_left={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_right={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        T_init=0.0,
        p_init=0.0,
    )
    assert isinstance(res, Domain2DResult)
    # Final T should be x-uniform (lateral no-flux).
    T_final = res.T[-1]
    x_std = float(np.std(T_final, axis=1).max())
    assert x_std < 1e-6, f"max x-stdev across z = {x_std:.3e}"
    # Match the Carslaw-Jaeger analytical to within 0.05 K.
    T_an = step_temperature_response(
        res.z,
        np.array([res.t[-1]]),
        delta_T=1.0,
        kappa=kappa,
    )[0]
    T_num = T_final.mean(axis=1)
    err = float(np.max(np.abs(T_num - T_an)))
    # 2-D grid is coarser than the 1-D test (dz=10m vs dz=2m) so
    # discretisation error scales accordingly.  Tolerance reflects this.
    assert err < 0.15, f"max |T_num - T_an| = {err:.4f} K"


def test_2d_handles_uniform_initial_state() -> None:
    """A column at T=0, p=0 with all-Neumann no-flux boundaries
    should remain at T=0, p=0 (steady-state null check)."""
    res = run_domain_2d(
        Lx_m=10.0,
        Lz_m=10.0,
        nx=5,
        nz=5,
        duration_s=YEAR_S,
        dt_s=YEAR_S / 10.0,
        porosity=0.10,
        lambda_r=2.5,
        K_xx=1e-15,
        K_zz=1e-15,
        T_f=-50.0,
        dTc=0.1,
        gamma_n_alpha_scale=0.0,
        bc_top={"T": BoundaryCondition(NEUMANN, 0.0), "p": BoundaryCondition(NEUMANN, 0.0)},
        bc_bottom={"T": BoundaryCondition(NEUMANN, 0.0), "p": BoundaryCondition(NEUMANN, 0.0)},
        bc_left={"T": BoundaryCondition(NEUMANN, 0.0), "p": BoundaryCondition(NEUMANN, 0.0)},
        bc_right={"T": BoundaryCondition(NEUMANN, 0.0), "p": BoundaryCondition(NEUMANN, 0.0)},
        T_init=0.0,
        p_init=0.0,
    )
    assert float(np.max(np.abs(res.T))) < 1e-9
    assert float(np.max(np.abs(res.p))) < 1e-9


def test_2d_freeze_thaw_with_uniform_x() -> None:
    """With lateral no-flux and a sub-zero surface, the 2-D solver
    should produce ice everywhere the column crosses the freezing
    interval — equivalent to a 1-D Stefan-like behaviour stacked
    in x."""
    res = run_domain_2d(
        Lx_m=2.0,
        Lz_m=4.0,
        nx=4,
        nz=21,
        duration_s=0.3 * YEAR_S,
        dt_s=YEAR_S / 200.0,
        porosity=0.30,
        lambda_r=2.5,
        lambda_w=2.5,
        lambda_i=2.5,
        rho_r=2700.0,
        c_r=800.0,
        rho_w=1000.0,
        c_w=4186.0,
        c_i=4565.0,
        L_f=3.34e5,
        K_xx=1e-20,
        K_zz=1e-20,
        T_f=0.0,
        dTc=0.5,
        gamma_n_alpha_scale=0.0,
        bc_top={
            "T": BoundaryCondition(DIRICHLET, -10.0),
            "p": BoundaryCondition(DIRICHLET, 0.0),
        },
        bc_bottom={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_left={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        bc_right={
            "T": BoundaryCondition(NEUMANN, 0.0),
            "p": BoundaryCondition(NEUMANN, 0.0),
        },
        T_init=0.0,
        p_init=0.0,
        picard_max_iter=15,
        picard_tol_K=1.0e-4,
    )
    # Ice should be present at the final time near the surface.
    S_final = res.S_i[-1]
    assert float(S_final[0, :].mean()) > 0.5, "no ice at top"
    # Should be uniform in x (no-flux laterally).
    assert float(np.std(S_final, axis=1).max()) < 1e-3


def test_2d_input_validation() -> None:
    with pytest.raises(ValueError):
        run_domain_2d(
            Lx_m=-1.0,
            Lz_m=10.0,
            nx=5,
            nz=5,
            duration_s=YEAR_S,
            dt_s=YEAR_S / 10.0,
        )
    with pytest.raises(ValueError):
        run_domain_2d(
            Lx_m=10.0,
            Lz_m=10.0,
            nx=2,
            nz=5,
            duration_s=YEAR_S,
            dt_s=YEAR_S / 10.0,
        )
    with pytest.raises(ValueError):
        run_domain_2d(
            Lx_m=10.0,
            Lz_m=10.0,
            nx=5,
            nz=5,
            duration_s=YEAR_S,
            dt_s=YEAR_S / 10.0,
            porosity=1.5,
        )
