"""Carslaw-Jaeger (1959) analytical solutions for 1-D heat conduction
in a semi-infinite medium.

This module is the verification companion to ``column_thermo_freeze_coupled``
when latent heat and coupling are turned off; in that limit the solver
must reproduce the classical erfc step response (Carslaw & Jaeger,
*Conduction of Heat in Solids*, 2nd ed., 1959, §2.5).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def step_temperature_response(
    z: np.ndarray,
    t: np.ndarray | float,
    *,
    delta_T: float,
    kappa: float,
) -> np.ndarray:
    """T(z, t) = delta_T * erfc( z / (2 sqrt(kappa t)) ).

    A semi-infinite half-space at zero initial temperature with a
    step change ``delta_T`` at the surface at ``t = 0``.

    Parameters
    ----------
    z : ndarray
        Depths (m), z >= 0.
    t : float or ndarray
        Times (s), all > 0.
    delta_T : float
        Surface-temperature step amplitude (K).
    kappa : float
        Thermal diffusivity ``lambda / (rho c)_eff`` (m^2 s^-1).

    Returns
    -------
    T : ndarray
        If ``t`` is scalar, shape (nz,); if ``t`` is array, shape
        (nt, nz).
    """
    z_arr = np.atleast_1d(z).astype(float)
    t_arr = np.atleast_1d(t).astype(float)
    if np.any(t_arr <= 0):
        raise ValueError("all t must be > 0 (step response is singular at t = 0)")
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    eta = z_arr[None, :] / (2.0 * np.sqrt(kappa * t_arr[:, None]))
    out = float(delta_T) * erfc(eta)
    if np.isscalar(t):
        return out[0]
    return out


def harmonic_temperature_response(
    z: np.ndarray,
    t: np.ndarray | float,
    *,
    amplitude_K: float,
    period_s: float,
    kappa: float,
) -> np.ndarray:
    """T(z, t) = A exp(-z/d) cos(omega t - z/d),  d = sqrt(2 kappa / omega).

    Surface harmonic forcing ``A cos(omega t)`` on a semi-infinite
    half-space.  The damped travelling wave with skin depth ``d``
    is the classical Carslaw-Jaeger solution (Eq. 2.6.iii).
    """
    if period_s <= 0 or kappa <= 0:
        raise ValueError("period_s and kappa must be positive")
    z_arr = np.atleast_1d(z).astype(float)
    t_arr = np.atleast_1d(t).astype(float)
    omega = 2.0 * np.pi / period_s
    d = np.sqrt(2.0 * kappa / omega)
    out = (
        amplitude_K
        * np.exp(-z_arr[None, :] / d)
        * np.cos(omega * t_arr[:, None] - z_arr[None, :] / d)
    )
    if np.isscalar(t):
        return out[0]
    return out
