"""1-D Crank-Nicolson solver with freeze-thaw via the apparent-heat-
capacity method.

The energy equation in the presence of phase change (Section 4 of
the accompanying paper) is

    rho c_eff(T) * dT/dt = d/dz (lambda * dT/dz)

with ``c_eff(T) = c_solid + L_f * rho_w * phi * dS_i/dT``.  The second
term is the apparent-heat-capacity spike at the freezing interval:
it absorbs the latent heat as the temperature crosses through
[T_f - dTc, T_f].

We use the piecewise-linear freezing curve described in the
accompanying paper:

           | 1                  T <= T_f - dTc
    S_i =  | (T_f - T) / dTc    T_f - dTc < T < T_f
           | 0                  T >= T_f

So ``c_app(T) = c_solid`` outside the interval and
``c_app(T) = c_solid + L_f rho_w phi / dTc`` inside it.

Discretisation
--------------
Variable-coefficient Crank-Nicolson with node-local r_i = lambda * dt /
(c_app(T_i^n) * dz^2):

    (1 + r_i) T_i^{n+1} - (r_i/2)(T_{i-1}^{n+1} + T_{i+1}^{n+1})
        = (1 - r_i) T_i^n + (r_i/2)(T_{i-1}^n + T_{i+1}^n)

c_app is lagged at T^n.  This is single-pass (no Picard inner
iteration); it matches the Stefan analytic to within ~10% for
``dTc >= 2 K`` and degrades for sharper intervals where nodes can
jump across the interval in one step without 'seeing' the latent-heat
spike.  Sharper intervals require a Picard iteration on T_mid (not
yet implemented); see TODO at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

YEAR_S: float = 365.25 * 86400.0

RHO_WATER: float = 1000.0
L_FUSION: float = 3.34e5
C_SOLID_DEFAULT: float = 2.5e6


@dataclass(frozen=True)
class EnthalpyResult:
    z: np.ndarray
    t: np.ndarray
    T: np.ndarray
    S_i: np.ndarray


def _ice_saturation(T: np.ndarray, *, T_f: float, dTc: float) -> np.ndarray:
    Si = np.zeros_like(T)
    Si[T <= T_f - dTc] = 1.0
    interval = (T > T_f - dTc) & (T < T_f)
    Si[interval] = (T_f - T[interval]) / dTc
    return Si


def _apparent_capacity(
    T: np.ndarray,
    *,
    T_f: float,
    dTc: float,
    rho_c_solid: float,
    L_f: float,
    rho_w: float,
    phi: float,
) -> np.ndarray:
    c = np.full_like(T, rho_c_solid)
    interval = (T > T_f - dTc) & (T < T_f)
    c[interval] += L_f * rho_w * phi / dTc
    return c


def run_column_enthalpy(
    *,
    depth_max_m: float,
    dz_m: float,
    duration_s: float,
    dt_s: float,
    lambda_thermal: float = 2.5,
    rho_c_solid: float = C_SOLID_DEFAULT,
    porosity: float = 0.30,
    L_f: float = L_FUSION,
    rho_w: float = RHO_WATER,
    T_f: float = 0.0,
    dTc: float = 1.0,
    sat: np.ndarray | float = -2.0,
    q_bot: float = 0.0,
    T_init: np.ndarray | float = 0.0,
) -> EnthalpyResult:
    if dz_m <= 0 or dt_s <= 0 or depth_max_m <= 0 or duration_s <= 0:
        raise ValueError("grid/time parameters must be positive")
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    if dTc <= 0:
        raise ValueError("dTc must be positive")

    z = np.arange(0.0, depth_max_m + 0.5 * dz_m, dz_m)
    nz = z.size
    nt = int(round(duration_s / dt_s)) + 1
    t = np.arange(nt) * dt_s

    if isinstance(sat, (int, float, np.floating, np.integer)):
        sat_series = np.full(nt, float(sat))
    else:
        sat_series = np.asarray(sat, dtype=float)
    if sat_series.size != nt:
        raise ValueError(f"sat length ({sat_series.size}) != nt ({nt})")

    if isinstance(T_init, (int, float, np.floating, np.integer)):
        T = np.full(nz, float(T_init))
    else:
        T = np.asarray(T_init, dtype=float).copy()
        if T.size != nz:
            raise ValueError(f"T_init length ({T.size}) != nz ({nz})")
    T[0] = sat_series[0]

    grad_bot = q_bot / lambda_thermal

    T_all = np.empty((nt, nz))
    S_all = np.empty((nt, nz))
    T_all[0] = T
    S_all[0] = _ice_saturation(T, T_f=T_f, dTc=dTc)

    n_int = nz - 1

    for n in range(nt - 1):
        c_app = _apparent_capacity(
            T,
            T_f=T_f,
            dTc=dTc,
            rho_c_solid=rho_c_solid,
            L_f=L_f,
            rho_w=rho_w,
            phi=porosity,
        )
        r = lambda_thermal * dt_s / (c_app * dz_m * dz_m)

        # Banded matrix ab (3, n_int) for nodes 1..nz-1.
        # Reduced system row i corresponds to global node (i+1).
        ab = np.zeros((3, n_int))
        for i in range(n_int):
            node = i + 1
            ri = r[node]
            ab[1, i] = 1.0 + ri
            if i > 0:
                ab[2, i - 1] = -0.5 * ri  # sub at row i
            if i < n_int - 1:
                ab[0, i + 1] = -0.5 * ri  # super at row i

        # Bottom Neumann ghost-node elimination: T_{nz} = T_{nz-2} +
        # 2 dz grad_bot, which converts the would-be (-ri/2) T_{nz}
        # term into (-ri/2) T_{nz-2} + (-ri/2)*2 dz grad_bot.
        ri_bot = r[nz - 1]
        ab[2, -2] += -0.5 * ri_bot  # extra sub-diag contribution

        rhs = np.zeros(n_int)
        for i in range(n_int):
            node = i + 1
            ri = r[node]
            T_left = T[node - 1]
            T_right = T[node + 1] if node + 1 < nz else T[node - 1] + 2.0 * dz_m * grad_bot
            rhs[i] = (1.0 - ri) * T[node] + 0.5 * ri * (T_left + T_right)

        # Top Dirichlet: substitute new GST into reduced eqn at i=0.
        gst_new = sat_series[n + 1]
        rhs[0] += 0.5 * r[1] * gst_new

        # Bottom Neumann: the ghost adds 2 dz grad to the RHS bottom.
        rhs[-1] += 0.5 * r[nz - 1] * (2.0 * dz_m * grad_bot)

        T_new_int = solve_banded((1, 1), ab, rhs)
        T = np.empty(nz)
        T[0] = gst_new
        T[1:] = T_new_int
        T_all[n + 1] = T
        S_all[n + 1] = _ice_saturation(T, T_f=T_f, dTc=dTc)

    return EnthalpyResult(z=z, t=t, T=T_all, S_i=S_all)


def neumann_stefan_lambda(stefan: float) -> float:
    """Solve the Neumann transcendental equation
    ``lambda * exp(lambda^2) * erf(lambda) = stefan / sqrt(pi)``
    for the dimensionless freezing-front coefficient.  Valid for
    stefan in (1e-3, 5)."""
    from scipy.special import erf

    def _f(lam: float) -> float:
        return lam * np.exp(lam * lam) * erf(lam) - stefan / np.sqrt(np.pi)

    lo, hi = 1.0e-4, 5.0
    while _f(lo) > 0:
        lo *= 0.1
    while _f(hi) < 0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-10:
            break
    return 0.5 * (lo + hi)


# TODO: add Picard iteration on c_app(T_mid) with the Bonacina
# (1973) enthalpy-integrated form to absorb latent heat when nodes
# jump the freezing interval in one step.  See the git history for a
# previous attempt; it was reverted because the face-averaging
# coupling I used was correct for variable lambda but wrong for
# variable c, and the simpler node-based form below is what the
# implementation needs.
