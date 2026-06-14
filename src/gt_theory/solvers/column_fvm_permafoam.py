"""1-D coupled T + p + S_i **finite-volume** solver with freeze-thaw,
following the PermaFoam numerical strategy (Orgogozo et al. 2014, 2019).

This is an *independent second numerical scheme* for the same closure
that ``column_thermo_freeze_coupled.py`` solves with a monolithic
Crank-Nicolson finite-difference discretisation.  The two schemes
differ in every numerical choice, which is exactly what makes their
agreement on the analytical benchmarks a strong correctness argument:

==================  ==============================  ==============================
aspect              column_thermo_freeze_coupled    column_fvm_permafoam (this)
==================  ==============================  ==============================
discretisation      finite difference (nodes)       finite volume (cell centres)
time integration    Crank-Nicolson (theta = 1/2)    backward / implicit Euler
coupling            monolithic block-banded          sequential (two TDMA solves)
                    ``solve_banded((3, 3))``         per Picard iteration
advection           centred (in block matrix)        first-order upwind
face properties     node values                      harmonic-mean face values
==================  ==============================  ==============================

The scheme is the one specified in the accompanying paper
("Numerical Scheme for the Coupled Thermal-Hydraulic System Based on
the PermaFoam Approach"), implemented faithfully:

* **Spatial** -- cell-centred FVM on ``N`` equal control volumes of
  thickness ``dz``; cell centres at ``z_i = (i + 1/2) dz``.  Flux
  divergences are evaluated as differences of face fluxes (divergence
  theorem); face permeabilities and conductivities are harmonic means
  of the adjacent cell values (Sec. 4.3, 5.2 of the accompanying paper).
* **Temporal** -- fully implicit (backward Euler), unconditionally
  stable, first-order accurate (Sec. 3.3 of the accompanying paper).
* **Coupling** -- sequential Picard outer loop: solve the mass
  equation for ``p`` (tridiagonal / TDMA), recover the face Darcy
  fluxes, then solve the energy equation for ``T`` (tridiagonal),
  under-relax both with ``omega`` and iterate to a (delta_p, delta_T)
  tolerance (Sec. 3.4, 7 of the accompanying paper).
* **Phase change** -- apparent heat capacity ``C_app = (rho c)_eff -
  rho_i L_f phi dS_i/dT`` (Sec. 5.1 of the accompanying paper), with the
  piecewise-linear freezing curve of ``theory.freezing_curve``.

Closure modules reused unchanged from the theory layer:
``theory.freezing_curve`` (S_i, dS_i/dT, apparent capacity),
``theory.effective_properties`` (geometric-mean lambda_eff,
Brooks-Corey k_rel, two-phase rho c), ``theory.dimless`` (constants).

Boundary conditions (Sec. 6 of the accompanying paper)
---------------------------------------
* **Top (surface)** -- Dirichlet for both ``T`` (= ``sat(t)``) and
  ``p`` (= ``p_top(t)``).  Implemented as a *face-based* Dirichlet at
  ``z = 0`` using the half-cell distance ``dz/2``, so the boundary sits
  exactly on the domain face rather than half a cell below it.  This is
  the conservative finite-volume form of the Dirichlet condition and is
  what lets the solver reproduce the Carslaw-Jaeger / Theis erfc
  similarity solutions (whose boundary is exactly at ``z = 0``) to the
  truncation order of the scheme.
* **Bottom** -- Neumann geothermal heat flux ``q_bot`` (W m^-2) for
  temperature: ``lambda_eff dT/dz = q_bot`` (positive ``q_bot`` =>
  temperature increasing downward => heat entering from below).  For
  pressure the default is zero-gradient ``dp/dz = 0`` (the accompanying paper);
  ``"neumann_hydrostatic"`` (zero total Darcy flux) and ``"dirichlet"``
  (fixed bottom pressure) are also available.

The explicit ``dS_w/dt`` mass-equation source
----------------------------------------------
Sec. 4.1 of the accompanying paper carries the phase-change storage term
``-rho_w phi dS_w/dt`` as an *explicit* right-hand-side source.  In the
sequential scheme this is feasible (it is a source, not a matrix
entry, so it does not wreck the conditioning the way it would in the
monolithic block-banded system -- see the note in
``column_thermo_freeze_coupled.py``).  However, with porosity held
fixed (no poroelastic ``dphi/dt`` to accommodate the volume change of
freezing water) this source is physically *unbalanced*: in an
impermeable freezing column it drives ``Delta p ~ Delta S_w / beta_w``,
i.e. GPa-scale pressures, because the only available sink is fluid
compressibility.  We therefore default ``include_phase_change_storage
= False`` (matching the proven behaviour of the monolithic solver) and
expose it as a flag: turning it on reproduces Sec. 4.1 of the accompanying paper
verbatim and is meaningful only in permeable, well-drained settings
where Darcy flow can carry the source away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

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


@dataclass(frozen=True)
class ColumnFVMResult:
    """Output of a finite-volume coupled T + p + S_i transient run.

    Attributes
    ----------
    z, t
        Cell-centre depths (m) and time (s) grids.  ``z`` are control-
        volume centres at ``(i + 1/2) dz``, offset by ``dz/2`` from the
        node-centred grid of ``column_thermo_freeze_coupled``.
    T : ndarray, shape (nt, nz)
        Temperature field (degC or K following the input convention).
    p : ndarray, shape (nt, nz)
        Pore-pressure field (Pa).
    S_i : ndarray, shape (nt, nz)
        Ice-saturation field (0..1).
    v_darcy : ndarray, shape (nt, nz)
        Cell-centred diagnostic Darcy velocity (m s^-1).
    picard_iters : ndarray, shape (nt - 1,)
        Picard outer-iteration count consumed at each time step.
    """

    z: np.ndarray
    t: np.ndarray
    T: np.ndarray
    p: np.ndarray
    S_i: np.ndarray
    v_darcy: np.ndarray
    picard_iters: np.ndarray


def _to_series(value, nt, name):
    if isinstance(value, (int, float, np.floating, np.integer)):
        return np.full(nt, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.size != nt:
        raise ValueError(f"{name} length ({arr.size}) != nt ({nt})")
    return arr


def _to_profile(value, nz, name, default):
    if value is None:
        return np.full(nz, default)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return np.full(nz, float(value))
    arr = np.asarray(value, dtype=float).copy()
    if arr.size != nz:
        raise ValueError(f"{name} length ({arr.size}) != nz ({nz})")
    return arr


def _harmonic_face(cell_values: np.ndarray) -> np.ndarray:
    """Harmonic mean of adjacent cell values at the ``N-1`` interior
    faces.  Returns 0 where either neighbour is 0 (so that a fully
    frozen, zero-``k_rel`` cell shuts off the face flux).
    """
    a = cell_values[:-1]
    b = cell_values[1:]
    denom = a + b
    out = np.zeros_like(denom)
    nonzero = denom > 0.0
    out[nonzero] = 2.0 * a[nonzero] * b[nonzero] / denom[nonzero]
    return out


def _solve_tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system via ``scipy.linalg.solve_banded``
    (Thomas algorithm).  ``lower[i]`` multiplies ``x_{i-1}`` in row i
    (``lower[0]`` unused); ``upper[i]`` multiplies ``x_{i+1}`` in row i
    (``upper[-1]`` unused).
    """
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def _darcy_cell_centred(p, k_rel, K_zz, mu, rho_w, g, dz, p_top, bot_p_bc):
    """Cell-centred Darcy velocity diagnostic consistent with the
    face-flux discretisation, for output only.
    """
    dp_dz = np.empty_like(p)
    dp_dz[1:-1] = (p[2:] - p[:-2]) / (2.0 * dz)
    # Top: one-sided gradient between cell centres 0 and 1 (spacing dz).
    dp_dz[0] = (p[1] - p[0]) / dz
    # Bottom: mirror the solver's internal q_face[-1] so the diagnostic
    # is consistent with each pressure BC.
    if bot_p_bc == "dirichlet":
        dp_dz[-1] = (p[-1] - p[-2]) / dz
    elif bot_p_bc == "neumann_hydrostatic":
        dp_dz[-1] = rho_w * g  # hydrostatic equilibrium -> zero Darcy flux
    else:  # neumann_zero_gradient
        dp_dz[-1] = 0.0  # gravity-only drainage flux
    return -(k_rel * K_zz / mu) * (dp_dz - rho_w * g)


def run_column_fvm_permafoam(
    *,
    depth_max_m: float,
    dz_m: float,
    duration_s: float,
    dt_s: float,
    # rock matrix properties
    porosity: float = 0.30,
    lambda_r: float = LAMBDA_ROCK_DEFAULT,
    rho_r: float = RHO_ROCK_DEFAULT,
    c_r: float = C_ROCK_DEFAULT,
    # fluid / ice properties
    lambda_w: float = LAMBDA_WATER,
    lambda_i: float = LAMBDA_ICE,
    rho_w: float = RHO_WATER,
    rho_i: float = RHO_ICE,
    c_w: float = C_WATER,
    c_i: float = C_ICE,
    L_f: float = L_FUSION,
    mu: float = MU_WATER,
    alpha_w: float = ALPHA_WATER,
    beta_w: float = BETA_WATER,
    g: float = G_EARTH,
    # hydraulic
    K_zz: float = 1.0e-13,
    eta_kr: float = 3.0,
    # freezing curve
    T_f: float = 0.0,
    dTc: float = 1.0,
    S_w_residual: float = 0.0,
    # coupling switches
    gamma_n_alpha_scale: float = 1.0,
    include_phase_change_storage: bool = False,
    # volumetric sources
    Q_heat: np.ndarray | float = 0.0,
    Q_w: np.ndarray | float = 0.0,
    # forcing and BCs
    sat: np.ndarray | float = 0.0,
    p_top: np.ndarray | float = 0.0,
    q_bot: float = 0.0,
    bot_p_bc: str = "neumann_zero_gradient",
    p_bot: float | None = None,
    T_init: np.ndarray | float | None = None,
    p_init: np.ndarray | float | None = None,
    # Picard control.  Sec. 7 of the accompanying paper recommends eps_T = 1e-4 K; we
    # default to a tighter 1e-5 K because, with under-relaxation, the
    # per-step Picard residual sets a one-signed accuracy floor that
    # accumulates over long runs -- 1e-5 keeps the conduction error
    # stable (<~0.03 K) even at 32 000 steps, where 1e-4 drifts to
    # ~0.19 K.  1e-4 remains adequate for short (<~2000-step) runs.
    picard_tol_K: float = 1.0e-5,
    picard_tol_p: float = 1.0e-2,
    picard_max_iter: int = 20,
    picard_omega: float = 0.7,
) -> ColumnFVMResult:
    """Run the finite-volume PermaFoam-style coupled solver.

    Parameters mirror ``run_column_thermo_freeze_coupled`` so the two
    schemes can be cross-verified on identical configurations.
    Differences: ``bot_p_bc`` defaults to the accompanying paper's zero-gradient
    pressure condition (vs. ``"neumann_hydrostatic"`` in the CN solver);
    ``picard_omega`` defaults to 0.7 (the sequential scheme needs
    under-relaxation, Sec. 3.4 of the accompanying paper); and ``picard_tol_p`` adds a
    pressure convergence tolerance to the temperature one.

    Returns
    -------
    ColumnFVMResult
    """
    if dz_m <= 0 or dt_s <= 0 or depth_max_m <= 0 or duration_s <= 0:
        raise ValueError("grid/time parameters must be positive")
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    if K_zz <= 0 or mu <= 0:
        raise ValueError("K_zz and mu must be positive")
    if beta_w <= 0:
        raise ValueError("beta_w must be positive")
    if bot_p_bc not in ("neumann_zero_gradient", "neumann_hydrostatic", "dirichlet"):
        raise ValueError(f"unknown bot_p_bc {bot_p_bc!r}")
    if picard_tol_K <= 0 or picard_tol_p <= 0 or picard_max_iter < 1:
        raise ValueError("invalid Picard controls")
    if not 0.0 < picard_omega <= 1.0:
        raise ValueError("picard_omega must be in (0, 1]")

    fc = FreezingCurveParams(T_f=T_f, dTc=dTc, S_w_residual=S_w_residual)

    # Cell-centred grid: N control volumes of thickness dz.
    nz = int(round(depth_max_m / dz_m))
    if nz < 3:
        raise ValueError("need at least 3 control volumes")
    dz = depth_max_m / nz
    z = (np.arange(nz) + 0.5) * dz
    nt = int(round(duration_s / dt_s)) + 1
    t = np.arange(nt) * dt_s

    sat_series = _to_series(sat, nt, "sat")
    p_top_series = _to_series(p_top, nt, "p_top")
    Q_heat_prof = _to_profile(Q_heat, nz, "Q_heat", 0.0)
    Q_w_prof = _to_profile(Q_w, nz, "Q_w", 0.0)

    T = _to_profile(T_init, nz, "T_init", 0.0)
    p = _to_profile(p_init, nz, "p_init", 0.0)

    # Fixed bottom pressure for the Dirichlet option: a user-supplied
    # level, or the initial bottom pressure when not given.  Held
    # constant across time steps (not re-pinned to the evolving field).
    p_bot_fixed = float(p_bot) if p_bot is not None else float(p[-1])

    s_coup = float(gamma_n_alpha_scale)
    storage_p = rho_w * porosity * beta_w / dt_s  # mass-eq accumulation

    T_all = np.empty((nt, nz))
    p_all = np.empty((nt, nz))
    S_all = np.empty((nt, nz))
    v_all = np.empty((nt, nz))
    picard_iters_log = np.zeros(nt - 1, dtype=int)

    S_i = ice_saturation(T, fc)
    k_rel0 = brooks_corey_k_rel(S_i, eta=eta_kr, S_w_residual=S_w_residual)
    T_all[0] = T
    p_all[0] = p
    S_all[0] = S_i
    v_all[0] = _darcy_cell_centred(p, k_rel0, K_zz, mu, rho_w, g, dz, p_top_series[0], bot_p_bc)

    inv_dz2 = 1.0 / (dz * dz)

    for n in range(nt - 1):
        T_old = T.copy()
        p_old = p.copy()
        S_w_old = 1.0 - ice_saturation(T_old, fc)
        T_new = T_old.copy()
        p_new = p_old.copy()

        T_surf_new = sat_series[n + 1]
        p_top_new = p_top_series[n + 1]

        picard = 0
        for picard in range(picard_max_iter):
            T_prev = T_new.copy()
            p_prev = p_new.copy()

            # --- saturation-dependent properties at the current iterate ---
            S_i_k = ice_saturation(T_new, fc)
            S_w_k = 1.0 - S_i_k
            k_rel_k = brooks_corey_k_rel(S_i_k, eta=eta_kr, S_w_residual=S_w_residual)
            lam_k = lambda_eff_geometric_mean(
                S_i_k,
                porosity=porosity,
                lambda_r=lambda_r,
                lambda_w=lambda_w,
                lambda_i=lambda_i,
            )
            rho_c_sens = rho_c_eff_two_phase(
                S_i_k,
                porosity=porosity,
                rho_r=rho_r,
                c_r=c_r,
                rho_w=rho_w,
                c_w=c_w,
                rho_i=rho_i,
                c_i=c_i,
            )
            C_app = apparent_volumetric_capacity(
                T_new,
                rho_c_dry=rho_c_sens,
                porosity=porosity,
                L_f=L_f,
                rho_w=rho_w,
                params=fc,
            )

            # =====================================================
            # 1) MASS EQUATION  ->  p_new   (tridiagonal / TDMA)
            # =====================================================
            # Cell and face mobilities  M = rho_w k_rel K_zz / mu.
            M_cell = rho_w * k_rel_k * K_zz / mu
            M_face = _harmonic_face(M_cell)  # interior faces, len nz-1

            diag = np.full(nz, storage_p)
            lower = np.zeros(nz)
            upper = np.zeros(nz)
            rhs = storage_p * p_old.copy()

            cond = M_face * inv_dz2  # interior face conductance
            diag[:-1] += cond
            diag[1:] += cond
            upper[:-1] = -cond
            lower[1:] = -cond

            # Top face-based Dirichlet at z = 0 (half-cell distance).
            ct = 2.0 * M_cell[0] * inv_dz2
            diag[0] += ct
            rhs[0] += ct * p_top_new

            # Bottom pressure BC.
            if bot_p_bc == "dirichlet":
                cb = 2.0 * M_cell[-1] * inv_dz2
                diag[-1] += cb
                rhs[-1] += cb * p_bot_fixed
            # zero-gradient / hydrostatic: no diffusive flux at bottom face.

            # Gravity-driven flux divergence (explicit, on the RHS).
            #   Fg_face = M_face * (-rho_w g);  R_grav = d Fg / dz.
            Fg = np.empty(nz + 1)
            Fg[1:-1] = M_face * (-rho_w * g)
            Fg[0] = M_cell[0] * (-rho_w * g)  # top boundary face
            if bot_p_bc == "neumann_hydrostatic":
                Fg[-1] = 0.0  # zero total Darcy flux
            elif bot_p_bc == "dirichlet":
                Fg[-1] = M_cell[-1] * (-rho_w * g)
            else:  # neumann_zero_gradient
                Fg[-1] = M_cell[-1] * (-rho_w * g)
            rhs += np.diff(Fg) / dz

            # Explicit thermal-expansion coupling.  NOTE on sign: a proper
            # mass balance d/dt(rho_w phi S_w) = div(rho_w q) + Q with
            # rho_w = rho_0[1 + beta_w(p-p0) - alpha_w(T-T0)] puts this
            # term on the RHS as +rho_w phi S_w alpha_w dT/dt, so that
            # heating a sealed column RAISES the pore pressure
            # (dp/dT = +alpha_w/beta_w, Detournay & Cheng 1993).  This is
            # the sign the undrained benchmark and the monolithic
            # column_thermo_freeze_coupled solver both use; the
            # mass-continuity equation in the accompanying paper writes it
            # with a minus sign, which is a sign slip relative to that
            # benchmark -- we use the +.
            rhs += s_coup * rho_w * porosity * S_w_k * alpha_w * (T_new - T_old) / dt_s
            if include_phase_change_storage:
                rhs += -rho_w * porosity * (S_w_k - S_w_old) / dt_s
            rhs += Q_w_prof

            p_star = _solve_tridiag(lower, diag, upper, rhs)
            p_new = picard_omega * p_star + (1.0 - picard_omega) * p_prev

            # =====================================================
            # 2) FACE DARCY FLUXES from the updated pressure
            # =====================================================
            kK_face = M_face / rho_w  # k_rel K_zz / mu at faces
            q_face = np.empty(nz + 1)
            q_face[1:-1] = -kK_face * ((p_new[1:] - p_new[:-1]) / dz - rho_w * g)
            # Top boundary face (half-cell to the Dirichlet p_top).
            q_face[0] = -(M_cell[0] / rho_w) * ((p_new[0] - p_top_new) / (0.5 * dz) - rho_w * g)
            # Bottom boundary face.
            if bot_p_bc == "neumann_hydrostatic":
                q_face[-1] = 0.0
            elif bot_p_bc == "dirichlet":
                q_face[-1] = -(M_cell[-1] / rho_w) * (
                    (p_bot_fixed - p_new[-1]) / (0.5 * dz) - rho_w * g
                )
            else:  # zero-gradient: only gravity drives the bottom flux
                q_face[-1] = -(M_cell[-1] / rho_w) * (-rho_w * g)

            # =====================================================
            # 3) ENERGY EQUATION  ->  T_new   (tridiagonal / TDMA)
            # =====================================================
            lam_face = _harmonic_face(lam_k)  # interior faces, len nz-1

            diagT = C_app / dt_s
            lowerT = np.zeros(nz)
            upperT = np.zeros(nz)
            rhsT = (C_app / dt_s) * T_old + Q_heat_prof

            condT = lam_face * inv_dz2
            diagT[:-1] += condT
            diagT[1:] += condT
            upperT[:-1] += -condT
            lowerT[1:] += -condT

            # Top face-based Dirichlet T_surf at z = 0.
            ctT = 2.0 * lam_k[0] * inv_dz2
            diagT[0] += ctT
            rhsT[0] += ctT * T_surf_new

            # Bottom Neumann geothermal flux: lambda dT/dz = q_bot.
            rhsT[-1] += q_bot / dz

            # Advection (first-order upwind).  Face coefficient
            #   A_face = rho_w c_w q_face;  upwind picks the upstream cell.
            A_face = rho_w * c_w * q_face  # len nz+1
            Ai = A_face[:-1]  # upper face of each cell
            Aip = A_face[1:]  # lower face of each cell
            pos = np.maximum(Aip, 0.0)
            neg = np.minimum(Aip, 0.0)
            pos_u = np.maximum(Ai, 0.0)
            neg_u = np.minimum(Ai, 0.0)
            # +(1/dz)(J_{i+1/2} - J_{i-1/2}) on the LHS:
            diagT += (pos - neg_u) / dz
            upperT += neg / dz
            lowerT += -pos_u / dz
            # Top boundary face: for downward inflow (A>=0) the upstream
            # value is the Dirichlet T_surf (known -> RHS); the spurious
            # ghost coefficient in lowerT[0] is ignored by the solver but
            # we zero it for cleanliness.  For outflow (A<0) the upstream
            # is cell 0, already captured by the -neg_u[0]/dz diagonal term.
            if A_face[0] >= 0.0:
                rhsT[0] += A_face[0] * T_surf_new / dz
            lowerT[0] = 0.0
            # Bottom boundary face: zero-gradient advective flux uses the
            # interior cell value for both in- and outflow.  The vectorised
            # step put only max(A_bot,0) on the diagonal; add min(A_bot,0)
            # so the full A_bot * T_{N-1} term is present, and drop the
            # ignored ghost coefficient in upperT[-1].
            diagT[-1] += neg[-1] / dz
            upperT[-1] = 0.0

            T_star = _solve_tridiag(lowerT, diagT, upperT, rhsT)
            T_new = picard_omega * T_star + (1.0 - picard_omega) * T_prev

            # Convergence is tested on the RAW (un-relaxed) solver
            # increments |x_star - x_prev|, NOT the under-relaxed field
            # increments |x_new - x_prev| = omega |x_star - x_prev|.
            # Testing the relaxed increment would trip the tolerance a
            # factor (1/omega) too early and leave the accepted field
            # ~(1-omega) short of the true backward-Euler solution, with a
            # one-signed per-step lag that accumulates over a long run.
            delta_T = float(np.max(np.abs(T_star - T_prev)))
            delta_p = float(np.max(np.abs(p_star - p_prev)))
            if delta_T < picard_tol_K and delta_p < picard_tol_p and picard >= 1:
                break

        picard_iters_log[n] = picard + 1

        T = T_new
        p = p_new
        S_i = ice_saturation(T, fc)
        k_rel = brooks_corey_k_rel(S_i, eta=eta_kr, S_w_residual=S_w_residual)

        T_all[n + 1] = T
        p_all[n + 1] = p
        S_all[n + 1] = S_i
        v_all[n + 1] = _darcy_cell_centred(p, k_rel, K_zz, mu, rho_w, g, dz, p_top_new, bot_p_bc)

    return ColumnFVMResult(
        z=z,
        t=t,
        T=T_all,
        p=p_all,
        S_i=S_all,
        v_darcy=v_all,
        picard_iters=picard_iters_log,
    )
