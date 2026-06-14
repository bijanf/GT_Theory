"""1-D Crank-Nicolson solver for the vertical heat equation with optional
Darcy advection and a Zhang (2005) winter n-factor.

Governing equation
------------------
    dT/dt = kappa * d^2T/dz^2  -  v_darcy * dT/dz

Boundary conditions
-------------------
    Top    (z = 0):   T(0, t) = GST(t)                       (Dirichlet)
    Bottom (z = z_b): k * dT/dz = q_bot                       (Neumann flux)

Surface coupling
----------------
The ground-surface temperature is modelled as
    GST(t) = eta * SAT(t)    where    eta = n_winter if SAT(t) < 0 else 1.0
which captures the dominant snow-insulation effect of Zhang (2005).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc


@dataclass(frozen=True)
class SolverResult:
    """Output of a transient 1-D column run.

    Attributes
    ----------
    z : ndarray, shape (nz,)
        Depth grid in metres, positive downward.
    t : ndarray, shape (nt,)
        Time grid in seconds.
    T : ndarray, shape (nt, nz)
        Temperature field, K (or anomaly K, depending on initial condition).
    gst : ndarray, shape (nt,)
        Applied ground-surface temperature time series.
    """

    z: np.ndarray
    t: np.ndarray
    T: np.ndarray
    gst: np.ndarray


def carslaw_jaeger_step_analytic(
    z: np.ndarray,
    t: np.ndarray,
    delta_T: float,
    kappa: float,
) -> np.ndarray:
    """Analytic solution of the half-space heat equation with a step change
    in surface temperature, with zero initial temperature.

    T(z, t) = dT * erfc( z / (2 * sqrt(kappa * t)) )

    Reference: Carslaw & Jaeger (1959), Conduction of Heat in Solids, §2.5.

    Parameters
    ----------
    z : ndarray, shape (nz,)
        Depths (m).
    t : ndarray, shape (nt,)
        Times (s), all > 0.
    delta_T : float
        Step amplitude (K).
    kappa : float
        Thermal diffusivity (m^2 s^-1).

    Returns
    -------
    T : ndarray, shape (nt, nz)
    """
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    if np.any(t <= 0):
        raise ValueError("Carslaw-Jaeger analytic requires t > 0.")
    arg = z[None, :] / (2.0 * np.sqrt(kappa * t[:, None]))
    return delta_T * erfc(arg)


def _gst_from_sat(sat: np.ndarray, n_winter: float) -> np.ndarray:
    """Apply the Zhang (2005) winter n-factor: multiply sub-zero SAT by
    n_winter, leave above-zero SAT unchanged."""
    if n_winter == 1.0:
        return sat.copy()
    gst = sat.copy()
    cold = sat < 0.0
    gst[cold] = n_winter * sat[cold]
    return gst


def run_column_1d(
    *,
    depth_max_m: float,
    dz_m: float,
    duration_s: float,
    dt_s: float,
    kappa: float,
    q_bot: float = 0.0,
    k_thermal: float = 2.5,
    sat: np.ndarray | float = 0.0,
    n_winter: float = 1.0,
    v_darcy: float = 0.0,
    T_init: np.ndarray | float | None = None,
) -> SolverResult:
    """Run the 1-D Crank-Nicolson solver.

    Parameters
    ----------
    depth_max_m, dz_m
        Column depth and uniform vertical grid spacing (m).
    duration_s, dt_s
        Total integration time and uniform time step (s).
    kappa
        Thermal diffusivity (m^2 s^-1).
    q_bot
        Geothermal heat flux at z = depth_max_m (W m^-2). Sign convention:
        q_bot > 0 means heat flowing upward into the column.
    k_thermal
        Bulk thermal conductivity (W m^-1 K^-1), used only to convert q_bot
        into a temperature gradient.
    sat
        Surface air temperature series (K, anomaly). Either a scalar (held
        constant) or an array of length nt = round(duration_s / dt_s) + 1.
    n_winter
        Zhang (2005) winter n-factor in [0, 1]. 1.0 disables it.
    v_darcy
        Vertical Darcy velocity (m s^-1), positive downward. 0.0 turns
        advection off.
    T_init
        Initial temperature profile. None or scalar means zero / constant
        initial condition. Array must have shape (nz,).

    Returns
    -------
    SolverResult
    """
    if dz_m <= 0 or dt_s <= 0:
        raise ValueError("dz_m and dt_s must be positive.")
    if depth_max_m <= 0 or duration_s <= 0:
        raise ValueError("depth_max_m and duration_s must be positive.")
    if not (0.0 <= n_winter <= 1.0):
        raise ValueError("n_winter must lie in [0, 1].")

    z = np.arange(0.0, depth_max_m + 0.5 * dz_m, dz_m)
    nz = z.size
    nt = int(round(duration_s / dt_s)) + 1
    t = np.arange(nt) * dt_s

    # Build the SAT time series and convert to GST via the n-factor.
    if isinstance(sat, (int, float, np.floating, np.integer)):
        sat_series = np.full(nt, float(sat))
    else:
        sat_arr = np.asarray(sat, dtype=float)
        if sat_arr.size != nt:
            raise ValueError(f"sat length ({sat_arr.size}) != nt ({nt}).")
        sat_series = sat_arr
    gst_series = _gst_from_sat(sat_series, n_winter)

    # Initial condition.
    if T_init is None:
        T = np.zeros(nz)
    elif isinstance(T_init, (int, float, np.floating, np.integer)):
        T = np.full(nz, float(T_init))
    else:
        T_arr = np.asarray(T_init, dtype=float)
        if T_arr.size != nz:
            raise ValueError(f"T_init length ({T_arr.size}) != nz ({nz}).")
        T = T_arr.copy()
    T[0] = gst_series[0]

    # Dimensionless coefficients.
    r = kappa * dt_s / (dz_m * dz_m)  # diffusion number
    c = v_darcy * dt_s / (2.0 * dz_m)  # advection (centred)

    # Crank-Nicolson tridiagonal system for interior nodes 1..nz-2:
    #   a_i T_{i-1}^{n+1} + b_i T_i^{n+1} + c_i T_{i+1}^{n+1} = rhs_i
    # with
    #   a = -r/2 - c/2 , b = 1 + r , c = -r/2 + c/2
    a = -r / 2.0 - c / 2.0
    b = 1.0 + r
    cc = -r / 2.0 + c / 2.0

    # Right-hand side coefficients (explicit half-step).
    a_r = +r / 2.0 + c / 2.0
    b_r = 1.0 - r
    c_r = +r / 2.0 - c / 2.0

    # Bottom Neumann BC: enforce dT/dz = q_bot / k_thermal via a ghost node
    # T_{nz} = T_{nz-2} + 2 * dz_m * (q_bot / k_thermal).
    grad_bot = q_bot / k_thermal

    # Pre-allocate the output and march in time.
    T_all = np.empty((nt, nz))
    T_all[0] = T
    n_int = nz - 1  # equations 1..nz-1 (bottom row absorbs the Neumann ghost)

    # Banded matrix in (l=1, u=1) form for solve_banded.
    ab = np.zeros((3, n_int))
    ab[0, 1:] = cc  # super-diagonal
    ab[1, :] = b  # main diagonal
    ab[2, :-1] = a  # sub-diagonal

    # Bottom-row tweak: ghost-node elimination merges the would-be node nz
    # into row nz-1.  T_{nz} = T_{nz-2} + 2*dz*grad_bot, and the FD stencil
    # at row nz-1 becomes
    #   (a + cc) T_{nz-2}^{n+1} + b T_{nz-1}^{n+1} = rhs - 2 * cc * dz * grad_bot
    # i.e. the sub-diagonal at the last row gains the super-diagonal value.
    ab[2, -2] = a + cc
    ab[1, -1] = b
    ab[0, -1] = 0.0  # no super-diagonal at last row

    for n in range(nt - 1):
        gst_new = gst_series[n + 1]
        rhs = np.zeros(n_int)
        # Interior right-hand side.
        rhs[1:-1] = a_r * T[1:-2] + b_r * T[2:-1] + c_r * T[3:]
        # Top row (index 0 of the reduced system == node 1 of the column).
        rhs[0] = a_r * T[0] + b_r * T[1] + c_r * T[2] - a * gst_new
        # Bottom row absorbs the ghost-node contribution.
        rhs[-1] = (
            a_r * T[-2]
            + b_r * T[-1]
            + c_r * (T[-2] + 2.0 * dz_m * grad_bot)
            - cc * 2.0 * dz_m * grad_bot
        )

        T_new_int = solve_banded((1, 1), ab, rhs)
        T = np.empty(nz)
        T[0] = gst_new
        T[1:] = T_new_int
        T_all[n + 1] = T

    return SolverResult(z=z, t=t, T=T_all, gst=gst_series)
