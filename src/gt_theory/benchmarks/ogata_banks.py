"""Ogata-Banks (1961) analytical solution for 1-D advection-dispersion
of a surface step in a semi-infinite medium.

This is the advection-dominated verification companion to the coupled
solvers: with latent heat and the thermo-poroelastic coupling turned
off and a sustained uniform Darcy flow, the energy equation reduces to
a linear advection-diffusion equation, whose response to a surface
temperature step is the classical Ogata-Banks solution (Ogata & Banks,
*A solution of the differential equation of longitudinal dispersion in
porous media*, USGS Prof. Paper 411-A, 1961).

It exercises exactly the operator the analytical conduction benchmark
(``carslaw_jaeger``) cannot: the upwind-vs-centred advection term.  The
first-order-upwind finite-volume solver stays monotone at any cell
Peclet number, whereas the centred Crank-Nicolson reference overshoots
on coarse grids; both converge onto this solution under grid
refinement.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx


def ogata_banks_response(
    z: np.ndarray,
    t: float,
    *,
    delta_T: float,
    v_T: float,
    kappa: float,
) -> np.ndarray:
    r"""T(z, t) for a surface step advected-dispersed into a half-space.

    .. math::
        T(z,t) = \frac{\Delta T}{2}\left[
            \mathrm{erfc}\!\left(\frac{z - v_T t}{2\sqrt{\kappa t}}\right)
            + \exp\!\left(\frac{v_T z}{\kappa}\right)
              \mathrm{erfc}\!\left(\frac{z + v_T t}{2\sqrt{\kappa t}}\right)
        \right]

    A semi-infinite column at zero initial temperature with a step
    ``delta_T`` imposed at the surface (``z = 0``) at ``t = 0`` and a
    sustained, uniform advective velocity ``v_T`` directed into the
    column (``z`` positive downward).

    Parameters
    ----------
    z : ndarray
        Depths (m), ``z >= 0``.
    t : float
        Time (s), ``> 0``.
    delta_T : float
        Surface-temperature step amplitude (K).
    v_T : float
        Thermal front (advective) velocity (m s^-1).  For a porous
        column carrying a Darcy flux ``v_Darcy`` this is
        ``v_T = rho_w c_w v_Darcy / (rho c)_eff`` -- the temperature
        front moves more slowly than the fluid because the rock matrix
        must also be heated.
    kappa : float
        Thermal diffusivity ``lambda_eff / (rho c)_eff`` (m^2 s^-1).

    Returns
    -------
    T : ndarray, shape (nz,)
        Temperature profile at time ``t``.

    Notes
    -----
    The second term is evaluated in the overflow-safe form
    ``exp(-b1**2) * erfcx(b2)`` with
    ``b1 = (z - v_T t)/(2 sqrt(kappa t))`` and
    ``b2 = (z + v_T t)/(2 sqrt(kappa t))``.  This is algebraically
    identical to ``exp(v_T z / kappa) * erfc(b2)`` because
    ``v_T z / kappa - b2**2 == -b1**2``, but it never forms the
    ``inf * 0`` product that the naive expression produces at depth
    (where ``v_T z / kappa`` can exceed the float exponent range while
    ``erfc(b2)`` underflows to zero).
    """
    if t <= 0:
        raise ValueError("t must be > 0 (step response is singular at t = 0)")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    z_arr = np.atleast_1d(z).astype(float)
    denom = 2.0 * np.sqrt(kappa * t)
    b1 = (z_arr - v_T * t) / denom
    b2 = (z_arr + v_T * t) / denom
    return 0.5 * float(delta_T) * (erfc(b1) + np.exp(-b1 * b1) * erfcx(b2))
