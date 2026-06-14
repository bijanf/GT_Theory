"""1-D planar analog of the Theis (1935) well solution: pressure
diffusion in a semi-infinite column under a step surface-pressure
boundary.

The classical Theis equation describes the radial pressure
drawdown around a producing well in a confined aquifer.  In our
1-D vertical column the mathematics reduces to the simple heat-
equation analog with hydraulic diffusivity
``c_v = K_zz / (mu phi beta_w)`` taking the place of thermal
diffusivity.  (This is the textbook value: the ``rho_w`` of the mass
storage cancels against the ``rho_w`` of the mass flux.)  The
step-pressure surface boundary then gives

    p(z, t) = delta_p * erfc( z / (2 sqrt(c_v t)) ).

This benchmark verifies the solver's mass-equation operator
independently of any thermal-hydraulic coupling.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def step_pressure_response(
    z: np.ndarray,
    t: np.ndarray | float,
    *,
    delta_p: float,
    c_v: float,
) -> np.ndarray:
    """p(z, t) = delta_p * erfc( z / (2 sqrt(c_v t)) ).

    Same functional form as Carslaw-Jaeger step but with hydraulic
    diffusivity instead of thermal.

    Parameters
    ----------
    z, t
        Depth grid (m) and times (s, > 0).
    delta_p : float
        Step amplitude at the surface (Pa).
    c_v : float
        Hydraulic diffusivity ``K_zz / (mu rho_w phi beta_w)`` (m^2 s^-1).
    """
    if c_v <= 0:
        raise ValueError("hydraulic diffusivity c_v must be positive")
    z_arr = np.atleast_1d(z).astype(float)
    t_arr = np.atleast_1d(t).astype(float)
    if np.any(t_arr <= 0):
        raise ValueError("all t must be > 0")
    eta = z_arr[None, :] / (2.0 * np.sqrt(c_v * t_arr[:, None]))
    out = float(delta_p) * erfc(eta)
    if np.isscalar(t):
        return out[0]
    return out


def hydraulic_diffusivity(
    *,
    K_zz: float,
    mu: float,
    porosity: float,
    beta_w: float,
) -> float:
    """``c_v = K_zz / (mu phi beta_w)`` (m^2 s^-1).

    Bundles the storage parameters into the single diffusion coefficient
    that controls the 1-D pressure-equation timescale.  This is the
    dimensionally consistent (textbook) form: the ``rho_w`` of the mass
    storage ``rho_w phi beta_w`` cancels against the ``rho_w`` of the mass
    flux ``rho_w k_rel K/mu``.  (Earlier revisions of this helper and of
    the solvers carried a spurious extra ``rho_w`` here, making ``c_v``
    a factor ``rho_w`` too small; that has been corrected throughout.)
    """
    if K_zz <= 0 or mu <= 0 or porosity <= 0 or beta_w <= 0:
        raise ValueError("all inputs must be positive")
    return K_zz / (mu * porosity * beta_w)
