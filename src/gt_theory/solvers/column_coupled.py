"""1-D coupled temperature-pressure solver (Crank-Nicolson, block-banded).

This is the solver the framework's $\\Gamma N_\\alpha$ coupling parameter
needs for direct empirical validation; see the accompanying paper for
the design rationale, equations, and the four reference benchmarks.

Equations (dimensional, mass and energy conservation in the accompanying
paper):

    Mass:
        rho_w phi beta_w dp/dt - s * rho_w phi alpha_w dT/dt
        = d/dz [ (K_zz/mu) (dp/dz - rho_w g) ]

    Energy:
        (rho c)_eff dT/dt + s * rho_w c_w v_Darcy dT/dz
        = d/dz [ lambda_eff dT/dz ]

with ``v_Darcy = -(K_zz/mu)(dp/dz - rho_w g)`` evaluated at the previous
time step (Picard-lagged single pass, same lagging trick as the
apparent-heat-capacity ``column_enthalpy`` solver), and ``s`` a scalar
``gamma_n_alpha_scale`` that multiplies BOTH cross-coupling terms; setting
``s = 0`` recovers the uncoupled limit (pure conduction for the energy
equation, pure pressure diffusion for the mass equation) and is the
controlled null hypothesis used in the test suite.

Discretisation
--------------
* Crank-Nicolson in time, centred differences in space, uniform ``dz_m``.
* Unknown ordering is interleaved: row ``2i`` is the energy equation at
  node ``i``, row ``2i+1`` is the mass equation at node ``i``.  The
  cross-coupling between row ``2i`` and node ``i`` of the other variable
  is then within bandwidth 3, so ``scipy.linalg.solve_banded((3, 3), ...)``
  applies -- identical pattern to ``column_enthalpy.py``.
* Boundary conditions: top T Dirichlet (``sat[n]``), top p Dirichlet
  (``p_top[n]``), bottom T Neumann (``lambda dT/dz = q_bot``) via
  ghost-node elimination, bottom p Neumann hydrostatic (``dp/dz = rho_w g``,
  no net Darcy flux) or Dirichlet, user-selectable.

Limitations
-----------
* No phase change (latent heat is handled separately by
  ``column_enthalpy.py``; both solvers are needed for the cold-and-wet
  regime, but the coupled solver targets the $\\Gamma N_\\alpha > 0$
  warm-and-permeable regime).
* No rock-skeleton compressibility; storage is set entirely by
  ``phi * beta_w``.  This is consistent with the accompanying paper's
  equations but means we exhibit Terzaghi 1-D consolidation rather than the
  Mandel-Cryer 2-D overshoot, which requires mechanical equilibrium.
* Single-pass Picard on the advection velocity; for ``Pe_T`` of order 1
  with sharp surface forcing, the user should refine ``dt_s`` rather
  than expect an inner iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from gt_theory.theory.dimless import (
    ALPHA_WATER,
    BETA_WATER,
    G_EARTH,
    MU_WATER,
    RHO_C_EFF_DEFAULT,
    RHO_WATER,
)

YEAR_S: float = 365.25 * 86400.0

C_WATER: float = 4.186e3  # J kg^-1 K^-1, specific heat of liquid water


@dataclass(frozen=True)
class CoupledResult:
    """Output of a transient coupled column run.

    Attributes
    ----------
    z : ndarray, shape (nz,)
        Depth grid (m), positive downward.
    t : ndarray, shape (nt,)
        Time grid (s).
    T : ndarray, shape (nt, nz)
        Temperature field (K, or anomaly K depending on initial condition).
    p : ndarray, shape (nt, nz)
        Pressure field (Pa, anomaly relative to the initial hydrostatic
        profile if ``p_init`` was hydrostatic).
    v_darcy : ndarray, shape (nt, nz)
        Diagnostic Darcy velocity (m s^-1, positive downward),
        ``-(K_zz/mu)(dp/dz - rho_w g)`` evaluated at each saved step.
    gst : ndarray, shape (nt,)
        Applied ground-surface temperature time series.
    p_top : ndarray, shape (nt,)
        Applied top pressure BC time series.
    """

    z: np.ndarray
    t: np.ndarray
    T: np.ndarray
    p: np.ndarray
    v_darcy: np.ndarray
    gst: np.ndarray
    p_top: np.ndarray


def _to_series(value: np.ndarray | float, nt: int, name: str) -> np.ndarray:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return np.full(nt, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.size != nt:
        raise ValueError(f"{name} length ({arr.size}) != nt ({nt})")
    return arr


def _to_profile(value: np.ndarray | float | None, nz: int, name: str, default: float) -> np.ndarray:
    if value is None:
        return np.full(nz, default)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return np.full(nz, float(value))
    arr = np.asarray(value, dtype=float).copy()
    if arr.size != nz:
        raise ValueError(f"{name} length ({arr.size}) != nz ({nz})")
    return arr


def _darcy_velocity(
    p: np.ndarray,
    *,
    K_zz: float,
    mu: float,
    rho_w: float,
    g: float,
    dz_m: float,
) -> np.ndarray:
    """Centred-difference Darcy velocity diagnostic on the same grid as p.

    Endpoints use one-sided differences so the returned array has the
    same length as ``p``.
    """
    dp_dz = np.empty_like(p)
    dp_dz[1:-1] = (p[2:] - p[:-2]) / (2.0 * dz_m)
    dp_dz[0] = (p[1] - p[0]) / dz_m
    dp_dz[-1] = (p[-1] - p[-2]) / dz_m
    return -(K_zz / mu) * (dp_dz - rho_w * g)


def run_column_coupled(
    *,
    depth_max_m: float,
    dz_m: float,
    duration_s: float,
    dt_s: float,
    # thermal properties
    lambda_thermal: float = 2.5,
    rho_c_eff: float = RHO_C_EFF_DEFAULT,
    # hydraulic properties
    K_zz: float = 1.0e-13,
    mu: float = MU_WATER,
    porosity: float = 0.15,
    alpha_w: float = ALPHA_WATER,
    beta_w: float = BETA_WATER,
    rho_w: float = RHO_WATER,
    g: float = G_EARTH,
    c_w: float = C_WATER,
    # coupling switch (0 -> fully uncoupled)
    gamma_n_alpha_scale: float = 1.0,
    # forcing
    sat: np.ndarray | float = 0.0,
    p_top: np.ndarray | float = 0.0,
    q_bot: float = 0.0,
    bot_p_bc: str = "neumann_hydrostatic",
    # initial conditions
    T_init: np.ndarray | float | None = None,
    p_init: np.ndarray | float | None = None,
) -> CoupledResult:
    """Run the 1-D coupled T-p Crank-Nicolson solver.

    Parameters
    ----------
    depth_max_m, dz_m
        Column depth and uniform vertical grid spacing (m).
    duration_s, dt_s
        Total integration time and uniform time step (s).
    lambda_thermal, rho_c_eff
        Effective thermal conductivity (W m^-1 K^-1) and effective
        volumetric heat capacity (J m^-3 K^-1).
    K_zz, mu
        Intrinsic vertical permeability (m^2) and dynamic viscosity (Pa s).
    porosity
        Bulk porosity ``phi``, dimensionless, in (0, 1).
    alpha_w, beta_w
        Thermal expansion coefficient (K^-1) and isothermal compressibility
        (Pa^-1) of the pore fluid.
    rho_w, g, c_w
        Pore-fluid density (kg m^-3), gravity (m s^-2), and specific
        heat (J kg^-1 K^-1).
    gamma_n_alpha_scale
        Scalar multiplier on both cross-coupling terms (advection in the
        energy equation, thermal expansion in the mass equation).  Setting
        it to 0 recovers the fully uncoupled limit.
    sat
        Top-boundary T forcing (scalar or length-``nt`` series, K or
        K-anomaly).  Applied as Dirichlet T at z=0.
    p_top
        Top-boundary p forcing (scalar or length-``nt`` series, Pa, anomaly
        relative to ``p_init`` at z=0).
    q_bot
        Geothermal heat flux at the bottom (W m^-2, positive upward).
    bot_p_bc
        ``"neumann_hydrostatic"`` (default) imposes ``dp/dz = rho_w g``,
        i.e. zero Darcy flux at the base; ``"dirichlet"`` holds the
        bottom pressure fixed at ``p_init[-1]``.
    T_init, p_init
        Initial profiles (length ``nz``) or scalars; ``None`` -> 0.0.

    Returns
    -------
    CoupledResult
    """
    if dz_m <= 0 or dt_s <= 0 or depth_max_m <= 0 or duration_s <= 0:
        raise ValueError("grid/time parameters must be positive")
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    if K_zz <= 0 or mu <= 0:
        raise ValueError("K_zz and mu must be positive")
    if beta_w <= 0:
        raise ValueError("beta_w must be positive")
    if bot_p_bc not in ("neumann_hydrostatic", "dirichlet"):
        raise ValueError(f"unknown bot_p_bc {bot_p_bc!r}")

    z = np.arange(0.0, depth_max_m + 0.5 * dz_m, dz_m)
    nz = z.size
    nt = int(round(duration_s / dt_s)) + 1
    t = np.arange(nt) * dt_s

    sat_series = _to_series(sat, nt, "sat")
    p_top_series = _to_series(p_top, nt, "p_top")

    T = _to_profile(T_init, nz, "T_init", 0.0)
    p = _to_profile(p_init, nz, "p_init", 0.0)
    T[0] = sat_series[0]
    p[0] = p_top_series[0]

    grad_T_bot = q_bot / lambda_thermal
    grad_p_bot = rho_w * g  # hydrostatic; flux = 0

    # Storage coefficient in the mass equation: rho_w phi beta_w dp/dt.
    # Coupling term: rho_w phi alpha_w dT/dt on the mass equation.
    # Energy storage: rho_c_eff dT/dt.
    A_T = rho_c_eff
    A_p = rho_w * porosity * beta_w
    C_pT = rho_w * porosity * alpha_w  # coefficient of dT/dt on mass eqn
    # Mass-flux mobility rho_w * K_zz / mu, consistent with the mass
    # storage A_p = rho_w phi beta_w, so the pressure diffusivity is the
    # textbook c_v = K_zz / (mu phi beta_w) (the rho_w factors cancel).
    K_h = rho_w * K_zz / mu
    s = float(gamma_n_alpha_scale)

    # Crank-Nicolson coefficients for the diffusion terms.
    # Energy:  A_T dT/dt = lambda d^2T/dz^2
    #   r_T = lambda dt / (A_T dz^2)
    # Mass:    A_p dp/dt = K_h d^2p/dz^2
    #   r_p = K_h dt / (A_p dz^2)
    r_T = lambda_thermal * dt_s / (A_T * dz_m * dz_m)
    r_p = K_h * dt_s / (A_p * dz_m * dz_m)

    # Interior equations live at nodes 1..nz-1, so the reduced system
    # has size n_int = nz - 1 in each variable.  We assemble a banded
    # matrix for the interleaved unknown vector
    #   x = [T_1, p_1, T_2, p_2, ..., T_{nz-1}, p_{nz-1}].
    # Bandwidth in this ordering is 3 (sub: T-T at -2, p-T at -1, T-p
    # at +1, T-T at +2 .. actually let's just write it out below).
    n_int = nz - 1
    n_eq = 2 * n_int
    lband = 3
    uband = 3

    # The banded matrix ab has shape (lband + uband + 1, n_eq);
    # ab[uband + i - j, j] is A[i, j] in dense form.
    def _set(ab: np.ndarray, i: int, j: int, val: float) -> None:
        ab[uband + i - j, j] = val

    def _add(ab: np.ndarray, i: int, j: int, val: float) -> None:
        ab[uband + i - j, j] += val

    T_all = np.empty((nt, nz))
    p_all = np.empty((nt, nz))
    v_all = np.empty((nt, nz))
    T_all[0] = T
    p_all[0] = p
    v_all[0] = _darcy_velocity(p, K_zz=K_zz, mu=mu, rho_w=rho_w, g=g, dz_m=dz_m)

    # Constant body-force RHS contribution from the d/dz[K_h (-rho_w g)]
    # term in the mass equation: this is zero on a uniform grid except at
    # the bottom row, where the ghost-node substitution leaves a
    # 2 dz rho_w g residual that goes into the RHS.  We handle it below
    # in the bottom-row tweak.

    for n in range(nt - 1):
        ab = np.zeros((lband + uband + 1, n_eq))
        rhs = np.zeros(n_eq)

        # Off-diagonal coupling coefficients.  The advection term in the
        # energy equation is the only one that needs the previous-step
        # Darcy velocity; the thermal-expansion term in the mass equation
        # is linear and goes directly into A and B.
        v_prev = _darcy_velocity(p, K_zz=K_zz, mu=mu, rho_w=rho_w, g=g, dz_m=dz_m)
        # Centred-advection coefficient in T-eq, scaled by s.
        #   s * rho_w c_w v dT/dz on row i  ->  (s rho_w c_w v_i)/(2 dz)
        #   * (T_{i+1} - T_{i-1})
        # In CN form, divided by A_T (== rho_c_eff), with dt:
        adv_coef = s * rho_w * c_w * v_prev * dt_s / (2.0 * A_T * dz_m)
        # Cross-block coupling on mass row: -s * C_pT * (T^{n+1} - T^n) / dt
        # rearranged to LHS, so the LHS multiplier on T^{n+1}_node is
        #   -s * C_pT / dt        (in units of A_p / dt)
        # We non-dimensionalise by dividing the mass eqn by A_p / dt:
        # row becomes:  p_node^{n+1} - r_p/2 * (p_{node-1} + p_{node+1}) - r_p p_node = ...
        # plus the cross term  -s * (C_pT / A_p) * (T_node^{n+1} - T_node^n)
        # so the matrix entry on the T column is  -s * (C_pT / A_p).
        ctp = -s * (C_pT / A_p)

        for k in range(n_int):
            node = k + 1
            row_T = 2 * k
            row_p = 2 * k + 1
            col_T = row_T
            col_p = row_p

            # ---------- Energy equation (row_T) ----------
            # 1 + r_T on diagonal, -r_T/2 on T_{node-1} and T_{node+1}
            # adv = adv_coef[node]:  +adv on T_{node+1}, -adv on T_{node-1}
            ri = r_T
            adv = adv_coef[node]

            _add(ab, row_T, col_T, 1.0 + ri)
            # off-grid T columns: k-1 -> col_T - 2 (or top BC absorbs); k+1 -> col_T + 2 (or bottom ghost)
            if k > 0:
                _add(ab, row_T, col_T - 2, -0.5 * ri + 0.5 * adv)
            if k < n_int - 1:
                _add(ab, row_T, col_T + 2, -0.5 * ri - 0.5 * adv)

            # RHS: (1 - r_T) T_node^n + (r_T/2)(T_{node-1}^n + T_{node+1}^n)
            #      + adv/2 * (T_{node-1}^n - T_{node+1}^n)
            T_left_n = T[node - 1]
            T_right_n = T[node + 1] if node + 1 < nz else T[node - 1] + 2.0 * dz_m * grad_T_bot
            rhs[row_T] = (1.0 - ri) * T[node] + 0.5 * ri * (T_left_n + T_right_n)
            rhs[row_T] += 0.5 * adv * (T_left_n - T_right_n)

            # Top Dirichlet on T: substitute T_0^{n+1} = sat_series[n+1] into row 0.
            if k == 0:
                gst_new = sat_series[n + 1]
                rhs[row_T] -= -0.5 * ri * gst_new  # = +0.5 r_T gst
                rhs[row_T] += 0.5 * (-adv) * gst_new
                # The T_{node-1} coefficient (which would be -0.5 ri + 0.5 adv)
                # is folded into the RHS via gst_new -- nothing to add to ab.

            # Bottom Neumann on T: T_{nz} = T_{nz-2} + 2 dz grad_T_bot
            if k == n_int - 1:
                # The k+1 column does not exist; instead, the T_{nz} term goes
                # back to T_{nz-2} (which is col_T - 2) with the same coefficient,
                # and the constant 2 dz grad_T_bot goes to the RHS.
                ri_bot = r_T
                adv_bot = adv  # already at node = nz-1
                ghost_coef = -0.5 * ri_bot - 0.5 * adv_bot
                _add(ab, row_T, col_T - 2, ghost_coef)
                rhs[row_T] -= ghost_coef * 2.0 * dz_m * grad_T_bot
                # And on the RHS, T_{node+1}^n = T_{node-1}^n + 2 dz grad
                # which was already used above when computing T_right_n.

            # ---------- Mass equation (row_p) ----------
            # 1 + r_p on diagonal, -r_p/2 on p_{node-1} and p_{node+1}.
            # Cross-coupling: -s C_pT/A_p * (T_node^{n+1} - T_node^n).
            #   LHS: + ctp on col_T == row_T (this node's T)
            #   RHS: + ctp * T_node^n  (move the T^n term to RHS)
            rj = r_p

            _add(ab, row_p, col_p, 1.0 + rj)
            if k > 0:
                _add(ab, row_p, col_p - 2, -0.5 * rj)
            if k < n_int - 1:
                _add(ab, row_p, col_p + 2, -0.5 * rj)
            _add(ab, row_p, col_T, ctp)

            p_left_n = p[node - 1]
            if node + 1 < nz:
                p_right_n = p[node + 1]
            elif bot_p_bc == "neumann_hydrostatic":
                p_right_n = p[node - 1] + 2.0 * dz_m * grad_p_bot
            else:
                # Dirichlet bottom: this row is overwritten below, so the
                # value here is a placeholder (avoid the out-of-range index).
                p_right_n = p[node - 1]
            rhs[row_p] = (1.0 - rj) * p[node] + 0.5 * rj * (p_left_n + p_right_n)
            rhs[row_p] += ctp * T[node]

            # Top Dirichlet on p:
            if k == 0:
                p_new = p_top_series[n + 1]
                rhs[row_p] -= -0.5 * rj * p_new

            # Bottom BC on p:
            if k == n_int - 1:
                if bot_p_bc == "neumann_hydrostatic":
                    ghost_coef_p = -0.5 * rj
                    _add(ab, row_p, col_p - 2, ghost_coef_p)
                    rhs[row_p] -= ghost_coef_p * 2.0 * dz_m * grad_p_bot
                else:
                    # Dirichlet: p_{nz-1} held fixed, but our k = n_int - 1
                    # IS node nz - 1.  Treat as a no-op in this branch (the
                    # node IS the bottom Dirichlet); zero out the row and
                    # impose p_node = p_init[-1].
                    p_fixed = float(p[-1])  # use current value (= p_init at t=0)
                    # Overwrite the row:
                    ab[:, row_p] = 0.0
                    _set(ab, row_p, col_p, 1.0)
                    rhs[row_p] = p_fixed

        x = solve_banded((lband, uband), ab, rhs)

        T_new = np.empty(nz)
        p_new_arr = np.empty(nz)
        T_new[0] = sat_series[n + 1]
        p_new_arr[0] = p_top_series[n + 1]
        T_new[1:] = x[0::2]
        p_new_arr[1:] = x[1::2]

        T = T_new
        p = p_new_arr
        T_all[n + 1] = T
        p_all[n + 1] = p
        v_all[n + 1] = _darcy_velocity(p, K_zz=K_zz, mu=mu, rho_w=rho_w, g=g, dz_m=dz_m)

    return CoupledResult(
        z=z,
        t=t,
        T=T_all,
        p=p_all,
        v_darcy=v_all,
        gst=sat_series,
        p_top=p_top_series,
    )
