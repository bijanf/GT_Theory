"""2-D coupled T + p + S_i Crank-Nicolson solver.

The 2-D Cartesian (x, z) analog of
``column_thermo_freeze_coupled.py``.  Solves the same closure
(extended Richards mass conservation + enthalpy-form energy
conservation with apparent-heat-capacity latent heat + Brooks-Corey
relative permeability + geometric-mean effective thermal
conductivity), but on a rectangular 2-D grid.

The 2-D solver unlocks two demonstrations the 1-D solver cannot
produce:

* **Mandel-Cryer overshoot** -- the canonical non-monotonic
  thermo-poroelastic pressure response under a sudden surface load
  with drained lateral boundaries.
* **Lateral basin flow** -- topography-driven recharge-to-discharge
  cross-sections (and the 2-D edge effects in permafrost columns
  with a lateral discontinuity in surface forcing).

Discretisation
--------------
Cartesian grid ``(nx, nz)``, uniform ``dx`` and ``dz``.  Linear
index ``k = j * nx + i`` with ``i`` the x index (fast) and ``j``
the z index (slow).  Per cell we carry two unknowns ``(T_k, p_k)``
ordered as ``[T_0, p_0, T_1, p_1, ...]``, so the global vector
has length ``2 nx nz``.  The sparse Crank-Nicolson matrix is
assembled in COO format and solved by ``scipy.sparse.linalg.spsolve``.

Per cell the stencil is

* T-row: T_self, T_E, T_W, T_N, T_S  (5 entries; pressure couples in
  only via the Picard-lagged Darcy velocity, which is a known
  coefficient at solve time).
* p-row: p_self, p_E, p_W, p_N, p_S, T_self  (6 entries; the
  last one carries the ``- s rho_w phi S_w alpha_w dT/dt``
  thermo-poroelastic coupling).

Boundary conditions
-------------------
Per side, T and p can each be Dirichlet or Neumann; the four sides
(top z=0, bottom z=L_z, left x=0, right x=L_x) are configured
independently via the ``bc_*`` dict arguments.  Mandel-Cryer needs
top T Dirichlet, top p Dirichlet (load applied), bottom Neumann
both, lateral sides p Dirichlet (drained) and T Neumann (insulated).

Picard iteration
----------------
The saturation-dependent properties (apparent heat capacity,
lambda_eff, k_rel) and the Picard-lagged Darcy velocity are
evaluated at the previous Picard iterate; ``omega < 1``
under-relaxation stabilises the iteration through sharp freezing
transitions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from gt_theory.theory.dimless import (
    ALPHA_WATER,
    BETA_WATER,
    G_EARTH,
    L_FUSION,
    MU_WATER,
    RHO_ICE,
    RHO_WATER,
)
from gt_theory.theory.effective_properties import (
    C_ICE,
    C_ROCK_DEFAULT,
    C_WATER,
    LAMBDA_ICE,
    LAMBDA_ROCK_DEFAULT,
    LAMBDA_WATER,
    RHO_ROCK_DEFAULT,
    brooks_corey_k_rel,
    lambda_eff_geometric_mean,
    rho_c_eff_two_phase,
)
from gt_theory.theory.freezing_curve import (
    FreezingCurveParams,
    apparent_volumetric_capacity,
    ice_saturation,
)

YEAR_S: float = 365.25 * 86400.0


# Boundary-condition kinds
DIRICHLET = "dirichlet"
NEUMANN = "neumann"


@dataclass(frozen=True)
class BoundaryCondition:
    """Per-side boundary specification for either T or p.

    kind == 'dirichlet': impose value (T or p) directly on the
    boundary face.  ``value`` can be a scalar (constant in space and
    time), a 1-D array (length nt, time-varying but uniform across
    the boundary face), or a callable ``f(coord, t) -> array`` (full
    spatial and temporal dependence).

    kind == 'neumann': impose the derivative normal to the boundary
    (heat flux for T, pressure gradient for p).  Sign convention:
    positive ``value`` means the field's z-derivative (or
    x-derivative on lateral boundaries) is positive at the face.
    """

    kind: str
    value: float | np.ndarray | Callable[[np.ndarray, float], np.ndarray] = 0.0


@dataclass(frozen=True)
class Domain2DResult:
    """Output of a 2-D coupled solver run."""

    x: np.ndarray  # (nx,)
    z: np.ndarray  # (nz,)
    t: np.ndarray  # (nt,)
    T: np.ndarray  # (nt, nz, nx)
    p: np.ndarray  # (nt, nz, nx)
    S_i: np.ndarray  # (nt, nz, nx)
    v_x: np.ndarray  # (nt, nz, nx)  Darcy velocity x
    v_z: np.ndarray  # (nt, nz, nx)  Darcy velocity z (down +)
    picard_iters: np.ndarray  # (nt - 1,)


def _bc_value(
    bc: BoundaryCondition, coord: np.ndarray, t: float, nt: int = 1, n_step: int = 0
) -> np.ndarray:
    """Resolve a BC value at a given time and coordinate."""
    v = bc.value
    if callable(v):
        return np.asarray(v(coord, t), dtype=float)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full(coord.shape, float(arr))
    if arr.ndim == 1 and arr.size == nt:
        return np.full(coord.shape, float(arr[n_step]))
    if arr.shape == coord.shape:
        return arr
    raise ValueError(f"BC value shape {arr.shape} cannot be broadcast to {coord.shape}")


def _darcy_velocity_2d(
    p: np.ndarray,  # (nz, nx)
    k_rel: np.ndarray,  # (nz, nx)
    *,
    K_xx: float,
    K_zz: float,
    mu: float,
    rho_w: float,
    g: float,
    dx: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centred-difference Darcy velocity (v_x, v_z) on the same grid as p."""
    dp_dx = np.empty_like(p)
    dp_dz = np.empty_like(p)
    dp_dx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dx)
    dp_dx[:, 0] = (p[:, 1] - p[:, 0]) / dx
    dp_dx[:, -1] = (p[:, -1] - p[:, -2]) / dx
    dp_dz[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dz)
    dp_dz[0, :] = (p[1, :] - p[0, :]) / dz
    dp_dz[-1, :] = (p[-1, :] - p[-2, :]) / dz
    v_x = -(k_rel * K_xx / mu) * dp_dx
    v_z = -(k_rel * K_zz / mu) * (dp_dz - rho_w * g)
    return v_x, v_z


def run_domain_2d(
    *,
    Lx_m: float,
    Lz_m: float,
    nx: int,
    nz: int,
    duration_s: float,
    dt_s: float,
    # rock matrix properties
    porosity: float = 0.30,
    lambda_r: float = LAMBDA_ROCK_DEFAULT,
    rho_r: float = RHO_ROCK_DEFAULT,
    c_r: float = C_ROCK_DEFAULT,
    # fluid properties
    lambda_w: float = LAMBDA_WATER,
    lambda_i: float = LAMBDA_ICE,
    rho_w: float = RHO_WATER,
    c_w: float = C_WATER,
    c_i: float = C_ICE,
    L_f: float = L_FUSION,
    mu: float = MU_WATER,
    alpha_w: float = ALPHA_WATER,
    beta_w: float = BETA_WATER,
    g: float = G_EARTH,
    # hydraulic
    K_xx: float = 1.0e-13,
    K_zz: float = 1.0e-13,
    eta_kr: float = 3.0,
    # freezing curve
    T_f: float = 0.0,
    dTc: float = 1.0,
    S_w_residual: float = 0.0,
    # coupling
    gamma_n_alpha_scale: float = 1.0,
    # boundary conditions: each side gets a (T_bc, p_bc) pair
    bc_top: Mapping[str, BoundaryCondition] | None = None,
    bc_bottom: Mapping[str, BoundaryCondition] | None = None,
    bc_left: Mapping[str, BoundaryCondition] | None = None,
    bc_right: Mapping[str, BoundaryCondition] | None = None,
    # initial conditions
    T_init: np.ndarray | float = 0.0,
    p_init: np.ndarray | float = 0.0,
    # Picard control
    picard_tol_K: float = 1.0e-4,
    picard_max_iter: int = 15,
    picard_omega: float = 0.7,
) -> Domain2DResult:
    """Run the 2-D coupled T + p + S_i Crank-Nicolson solver.

    Returns a Domain2DResult with fields of shape (nt, nz, nx)
    (time slowest, x fastest).
    """
    if Lx_m <= 0 or Lz_m <= 0 or duration_s <= 0 or dt_s <= 0:
        raise ValueError("Lx, Lz, duration, dt must all be positive")
    if nx < 3 or nz < 3:
        raise ValueError("nx and nz must be >= 3")
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    if not 0.0 < picard_omega <= 1.0:
        raise ValueError("picard_omega must be in (0, 1]")

    bc_top = bc_top or {
        "T": BoundaryCondition(DIRICHLET, 0.0),
        "p": BoundaryCondition(DIRICHLET, 0.0),
    }
    bc_bottom = bc_bottom or {
        "T": BoundaryCondition(NEUMANN, 0.0),
        "p": BoundaryCondition(NEUMANN, 0.0),
    }
    bc_left = bc_left or {
        "T": BoundaryCondition(NEUMANN, 0.0),
        "p": BoundaryCondition(NEUMANN, 0.0),
    }
    bc_right = bc_right or {
        "T": BoundaryCondition(NEUMANN, 0.0),
        "p": BoundaryCondition(NEUMANN, 0.0),
    }

    dx = Lx_m / (nx - 1)
    dz = Lz_m / (nz - 1)
    x = np.linspace(0.0, Lx_m, nx)
    z = np.linspace(0.0, Lz_m, nz)
    nt = int(round(duration_s / dt_s)) + 1
    t = np.arange(nt) * dt_s

    fc_params = FreezingCurveParams(T_f=T_f, dTc=dTc, S_w_residual=S_w_residual)

    # Initial fields shape (nz, nx).
    if np.ndim(T_init) == 0:
        T = np.full((nz, nx), float(T_init))
    else:
        T = np.asarray(T_init, dtype=float).copy()
        if T.shape != (nz, nx):
            raise ValueError(f"T_init shape {T.shape} != ({nz}, {nx})")
    if np.ndim(p_init) == 0:
        p = np.full((nz, nx), float(p_init))
    else:
        p = np.asarray(p_init, dtype=float).copy()
        if p.shape != (nz, nx):
            raise ValueError(f"p_init shape {p.shape} != ({nz}, {nx})")

    # Output buffers
    T_all = np.empty((nt, nz, nx))
    p_all = np.empty((nt, nz, nx))
    S_all = np.empty((nt, nz, nx))
    vx_all = np.empty((nt, nz, nx))
    vz_all = np.empty((nt, nz, nx))
    picard_iters = np.zeros(nt - 1, dtype=int)

    S_i = ice_saturation(T, fc_params)
    k_rel = brooks_corey_k_rel(S_i, eta=eta_kr, S_w_residual=S_w_residual)
    vx0, vz0 = _darcy_velocity_2d(
        p,
        k_rel,
        K_xx=K_xx,
        K_zz=K_zz,
        mu=mu,
        rho_w=rho_w,
        g=g,
        dx=dx,
        dz=dz,
    )
    T_all[0] = T
    p_all[0] = p
    S_all[0] = S_i
    vx_all[0] = vx0
    vz_all[0] = vz0

    s_coup = float(gamma_n_alpha_scale)

    def idx(i: int, j: int, which: int) -> int:
        """which = 0 for T row, 1 for p row."""
        return 2 * (j * nx + i) + which

    n_eq = 2 * nx * nz

    for n_step in range(nt - 1):
        T_old = T.copy()
        p_old = p.copy()
        T_new = T_old.copy()
        p_new = p_old.copy()

        # Resolve BC values at the new time level.
        t_new = float(t[n_step + 1])
        top_T_val = _bc_value(bc_top["T"], x, t_new, nt, n_step + 1)
        top_p_val = _bc_value(bc_top["p"], x, t_new, nt, n_step + 1)
        bot_T_val = _bc_value(bc_bottom["T"], x, t_new, nt, n_step + 1)
        bot_p_val = _bc_value(bc_bottom["p"], x, t_new, nt, n_step + 1)
        left_T_val = _bc_value(bc_left["T"], z, t_new, nt, n_step + 1)
        left_p_val = _bc_value(bc_left["p"], z, t_new, nt, n_step + 1)
        right_T_val = _bc_value(bc_right["T"], z, t_new, nt, n_step + 1)
        right_p_val = _bc_value(bc_right["p"], z, t_new, nt, n_step + 1)

        for picard in range(picard_max_iter):
            T_prev_iter = T_new.copy()

            # --- Saturation-dependent properties ---
            T_eval = 0.5 * (T_old + T_new)
            S_i_iter = ice_saturation(T_eval, fc_params)
            lam_eff = lambda_eff_geometric_mean(
                S_i_iter,
                porosity=porosity,
                lambda_r=lambda_r,
                lambda_w=lambda_w,
                lambda_i=lambda_i,
            )
            k_rel_iter = brooks_corey_k_rel(
                S_i_iter,
                eta=eta_kr,
                S_w_residual=S_w_residual,
            )
            rho_c_sensible = rho_c_eff_two_phase(
                S_i_iter,
                porosity=porosity,
                rho_r=rho_r,
                c_r=c_r,
                rho_w=rho_w,
                c_w=c_w,
                rho_i=RHO_ICE,
                c_i=c_i,
            )
            C_app = apparent_volumetric_capacity(
                T_eval,
                rho_c_dry=rho_c_sensible,
                porosity=porosity,
                L_f=L_f,
                rho_w=rho_w,
                params=fc_params,
            )

            r_T_x = lam_eff * dt_s / (C_app * dx * dx)
            r_T_z = lam_eff * dt_s / (C_app * dz * dz)
            A_p_node = np.full_like(C_app, rho_w * porosity * beta_w)
            # Mass-flux mobilities rho_w * k_rel K / mu, consistent with the
            # mass storage A_p; the rho_w factors cancel so the pressure
            # diffusivity is the textbook K/(mu phi beta_w).
            K_h_x = rho_w * (k_rel_iter * K_xx) / mu
            K_h_z = rho_w * (k_rel_iter * K_zz) / mu
            r_p_x = K_h_x * dt_s / (A_p_node * dx * dx)
            r_p_z = K_h_z * dt_s / (A_p_node * dz * dz)
            ctp = -s_coup * (rho_w * porosity * (1.0 - S_i_iter) * alpha_w) / A_p_node

            # Picard-lagged Darcy velocity for advection on T equation
            v_x_iter, v_z_iter = _darcy_velocity_2d(
                p_new,
                k_rel_iter,
                K_xx=K_xx,
                K_zz=K_zz,
                mu=mu,
                rho_w=rho_w,
                g=g,
                dx=dx,
                dz=dz,
            )
            adv_x = s_coup * rho_w * c_w * v_x_iter * dt_s / (2.0 * C_app * dx)
            adv_z = s_coup * rho_w * c_w * v_z_iter * dt_s / (2.0 * C_app * dz)

            # --- COO assembly ---
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            rhs = np.zeros(n_eq)

            def add(r: int, c: int, v: float, _rows=rows, _cols=cols, _data=data) -> None:
                _rows.append(r)
                _cols.append(c)
                _data.append(v)

            for j in range(nz):
                for i in range(nx):
                    r_T = idx(i, j, 0)
                    r_p = idx(i, j, 1)
                    # ---- Energy equation row ----
                    rTx = r_T_x[j, i]
                    rTz = r_T_z[j, i]
                    adv_x_ij = adv_x[j, i]
                    adv_z_ij = adv_z[j, i]
                    # Diagonal
                    add(r_T, r_T, 1.0 + rTx + rTz)
                    rhs_T = (1.0 - rTx - rTz) * T_old[j, i]

                    # E neighbour (i+1)
                    if i + 1 < nx:
                        add(r_T, idx(i + 1, j, 0), -0.5 * rTx - 0.5 * adv_x_ij)
                        rhs_T += (0.5 * rTx - 0.5 * adv_x_ij) * T_old[j, i + 1]
                    else:
                        # Right boundary
                        bc = bc_right["T"]
                        if bc.kind == DIRICHLET:
                            # T_{nx} = right_T_val[j]; folds into RHS.
                            rhs_T -= (-0.5 * rTx - 0.5 * adv_x_ij) * right_T_val[j]
                            rhs_T += (0.5 * rTx - 0.5 * adv_x_ij) * right_T_val[j]
                        else:
                            # Neumann: T_{nx} = T_{nx-2} + 2 dx * grad
                            grad = float(right_T_val[j]) / max(lam_eff[j, i], 1e-12)
                            ghost = -0.5 * rTx - 0.5 * adv_x_ij
                            add(r_T, idx(i - 1, j, 0), ghost)
                            rhs_T -= ghost * 2.0 * dx * grad
                            rhs_T += (0.5 * rTx - 0.5 * adv_x_ij) * (
                                T_old[j, i - 1] + 2.0 * dx * grad
                            )
                    # W neighbour (i-1)
                    if i - 1 >= 0:
                        add(r_T, idx(i - 1, j, 0), -0.5 * rTx + 0.5 * adv_x_ij)
                        rhs_T += (0.5 * rTx + 0.5 * adv_x_ij) * T_old[j, i - 1]
                    else:
                        bc = bc_left["T"]
                        if bc.kind == DIRICHLET:
                            rhs_T -= (-0.5 * rTx + 0.5 * adv_x_ij) * left_T_val[j]
                            rhs_T += (0.5 * rTx + 0.5 * adv_x_ij) * left_T_val[j]
                        else:
                            grad = float(left_T_val[j]) / max(lam_eff[j, i], 1e-12)
                            ghost = -0.5 * rTx + 0.5 * adv_x_ij
                            # T_{-1} = T_{1} - 2 dx * grad (outward normal is -x)
                            add(r_T, idx(i + 1, j, 0), ghost)
                            rhs_T -= ghost * (-2.0 * dx * grad)
                            rhs_T += (0.5 * rTx + 0.5 * adv_x_ij) * (
                                T_old[j, i + 1] - 2.0 * dx * grad
                            )
                    # S neighbour (j+1) — z direction, j+1 means deeper
                    if j + 1 < nz:
                        add(r_T, idx(i, j + 1, 0), -0.5 * rTz - 0.5 * adv_z_ij)
                        rhs_T += (0.5 * rTz - 0.5 * adv_z_ij) * T_old[j + 1, i]
                    else:
                        bc = bc_bottom["T"]
                        if bc.kind == DIRICHLET:
                            rhs_T -= (-0.5 * rTz - 0.5 * adv_z_ij) * bot_T_val[i]
                            rhs_T += (0.5 * rTz - 0.5 * adv_z_ij) * bot_T_val[i]
                        else:
                            grad = float(bot_T_val[i]) / max(lam_eff[j, i], 1e-12)
                            ghost = -0.5 * rTz - 0.5 * adv_z_ij
                            add(r_T, idx(i, j - 1, 0), ghost)
                            rhs_T -= ghost * 2.0 * dz * grad
                            rhs_T += (0.5 * rTz - 0.5 * adv_z_ij) * (
                                T_old[j - 1, i] + 2.0 * dz * grad
                            )
                    # N neighbour (j-1) — z direction, j-1 means shallower (top is j=0)
                    if j - 1 >= 0:
                        add(r_T, idx(i, j - 1, 0), -0.5 * rTz + 0.5 * adv_z_ij)
                        rhs_T += (0.5 * rTz + 0.5 * adv_z_ij) * T_old[j - 1, i]
                    else:
                        bc = bc_top["T"]
                        if bc.kind == DIRICHLET:
                            rhs_T -= (-0.5 * rTz + 0.5 * adv_z_ij) * top_T_val[i]
                            rhs_T += (0.5 * rTz + 0.5 * adv_z_ij) * top_T_val[i]
                        else:
                            grad = float(top_T_val[i]) / max(lam_eff[j, i], 1e-12)
                            ghost = -0.5 * rTz + 0.5 * adv_z_ij
                            add(r_T, idx(i, j + 1, 0), ghost)
                            rhs_T -= ghost * (-2.0 * dz * grad)
                            rhs_T += (0.5 * rTz + 0.5 * adv_z_ij) * (
                                T_old[j + 1, i] - 2.0 * dz * grad
                            )
                    rhs[r_T] = rhs_T

                    # ---- Mass equation row ----
                    rpx = r_p_x[j, i]
                    rpz = r_p_z[j, i]
                    ctp_ij = ctp[j, i]
                    add(r_p, r_p, 1.0 + rpx + rpz)
                    add(r_p, r_T, ctp_ij)
                    rhs_p = (1.0 - rpx - rpz) * p_old[j, i] + ctp_ij * T_old[j, i]

                    # Body-force from rho_w g term: only contributes at z boundaries
                    # as a constant offset that we put into the RHS via gradient BC.
                    if i + 1 < nx:
                        add(r_p, idx(i + 1, j, 1), -0.5 * rpx)
                        rhs_p += 0.5 * rpx * p_old[j, i + 1]
                    else:
                        bc = bc_right["p"]
                        if bc.kind == DIRICHLET:
                            rhs_p += 0.5 * rpx * right_p_val[j]
                            rhs_p -= (-0.5 * rpx) * right_p_val[j]
                        else:
                            ghost = -0.5 * rpx
                            add(r_p, idx(i - 1, j, 1), ghost)
                            grad = float(right_p_val[j])
                            rhs_p -= ghost * 2.0 * dx * grad
                            rhs_p += 0.5 * rpx * (p_old[j, i - 1] + 2.0 * dx * grad)
                    if i - 1 >= 0:
                        add(r_p, idx(i - 1, j, 1), -0.5 * rpx)
                        rhs_p += 0.5 * rpx * p_old[j, i - 1]
                    else:
                        bc = bc_left["p"]
                        if bc.kind == DIRICHLET:
                            rhs_p += 0.5 * rpx * left_p_val[j]
                            rhs_p -= (-0.5 * rpx) * left_p_val[j]
                        else:
                            ghost = -0.5 * rpx
                            add(r_p, idx(i + 1, j, 1), ghost)
                            grad = float(left_p_val[j])
                            rhs_p -= ghost * (-2.0 * dx * grad)
                            rhs_p += 0.5 * rpx * (p_old[j, i + 1] - 2.0 * dx * grad)
                    if j + 1 < nz:
                        add(r_p, idx(i, j + 1, 1), -0.5 * rpz)
                        rhs_p += 0.5 * rpz * p_old[j + 1, i]
                    else:
                        bc = bc_bottom["p"]
                        if bc.kind == DIRICHLET:
                            rhs_p += 0.5 * rpz * bot_p_val[i]
                            rhs_p -= (-0.5 * rpz) * bot_p_val[i]
                        else:
                            ghost = -0.5 * rpz
                            add(r_p, idx(i, j - 1, 1), ghost)
                            grad = float(bot_p_val[i])
                            rhs_p -= ghost * 2.0 * dz * grad
                            rhs_p += 0.5 * rpz * (p_old[j - 1, i] + 2.0 * dz * grad)
                    if j - 1 >= 0:
                        add(r_p, idx(i, j - 1, 1), -0.5 * rpz)
                        rhs_p += 0.5 * rpz * p_old[j - 1, i]
                    else:
                        bc = bc_top["p"]
                        if bc.kind == DIRICHLET:
                            rhs_p += 0.5 * rpz * top_p_val[i]
                            rhs_p -= (-0.5 * rpz) * top_p_val[i]
                        else:
                            ghost = -0.5 * rpz
                            add(r_p, idx(i, j + 1, 1), ghost)
                            grad = float(top_p_val[i])
                            rhs_p -= ghost * (-2.0 * dz * grad)
                            rhs_p += 0.5 * rpz * (p_old[j + 1, i] - 2.0 * dz * grad)
                    rhs[r_p] = rhs_p

            # Build sparse and solve.
            A = csc_matrix(
                (data, (rows, cols)),
                shape=(n_eq, n_eq),
            )
            x_sol = spsolve(A, rhs)

            T_new_flat = x_sol[0::2]
            p_new_flat = x_sol[1::2]
            T_new = T_new_flat.reshape(nz, nx)
            p_new = p_new_flat.reshape(nz, nx)
            if picard_omega < 1.0:
                T_new = picard_omega * T_new + (1.0 - picard_omega) * T_prev_iter
                p_new = picard_omega * p_new + (1.0 - picard_omega) * p_new  # no-op

            delta_T = float(np.max(np.abs(T_new - T_prev_iter)))
            if delta_T < picard_tol_K and picard >= 1:
                break

        picard_iters[n_step] = picard + 1
        T = T_new
        p = p_new
        S_i = ice_saturation(T, fc_params)
        k_rel = brooks_corey_k_rel(S_i, eta=eta_kr, S_w_residual=S_w_residual)
        v_x_step, v_z_step = _darcy_velocity_2d(
            p,
            k_rel,
            K_xx=K_xx,
            K_zz=K_zz,
            mu=mu,
            rho_w=rho_w,
            g=g,
            dx=dx,
            dz=dz,
        )
        T_all[n_step + 1] = T
        p_all[n_step + 1] = p
        S_all[n_step + 1] = S_i
        vx_all[n_step + 1] = v_x_step
        vz_all[n_step + 1] = v_z_step

    return Domain2DResult(
        x=x,
        z=z,
        t=t,
        T=T_all,
        p=p_all,
        S_i=S_all,
        v_x=vx_all,
        v_z=vz_all,
        picard_iters=picard_iters,
    )
