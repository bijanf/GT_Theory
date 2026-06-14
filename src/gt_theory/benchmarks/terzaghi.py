"""Terzaghi (1943) 1-D consolidation: analytical sin-series solution.

A column of saturated soil of length ``L`` with initial pore-pressure
``p_0``, top Dirichlet drainage boundary (``p_top = 0``), and bottom
no-flow Neumann (``dp/dz = 0``).  The pore-pressure isochrone is

    p(z, t) / p_0 = sum_{m=0}^infinity  (4/((2m+1) pi))
                  * sin((2m+1) pi z / (2 L))
                  * exp(-(2m+1)^2 pi^2 T_v / 4),

with the dimensionless time ``T_v = c_v t / L^2``.

The degree of consolidation

    U(t) = 1 - sum_{m=0}^infinity  (8 / ((2m+1) pi)^2)
                                 * exp(-(2m+1)^2 pi^2 T_v / 4).

This module is the verification companion to ``column_coupled`` /
``column_thermo_freeze_coupled`` when the energy equation is held
isothermal and gravity is set to zero.
"""

from __future__ import annotations

import numpy as np


def isochrone(
    z: np.ndarray,
    T_v: float | np.ndarray,
    *,
    p0: float,
    L: float,
    n_terms: int = 200,
) -> np.ndarray:
    """Terzaghi isochrone p(z, T_v) / p_0.

    Returns the pore-pressure profile at one or several dimensionless
    times.  ``T_v`` is the dimensionless consolidation time
    ``c_v t / L^2``.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    z_arr = np.atleast_1d(z).astype(float)
    T_arr = np.atleast_1d(T_v).astype(float)
    series = np.zeros((T_arr.size, z_arr.size))
    for m in range(n_terms):
        n = 2 * m + 1
        spatial = (4.0 / (n * np.pi)) * np.sin(n * np.pi * z_arr / (2.0 * L))
        temporal = np.exp(-((n * np.pi / 2.0) ** 2) * T_arr[:, None])
        series += float(p0) * spatial[None, :] * temporal
    if np.isscalar(T_v):
        return series[0]
    return series


def degree_of_consolidation(
    T_v: float | np.ndarray,
    *,
    n_terms: int = 200,
) -> np.ndarray | float:
    """U(T_v): degree of consolidation, in [0, 1].

    U = 1 at fully drained, 0 at undrained.
    """
    T_arr = np.atleast_1d(T_v).astype(float)
    s = np.zeros_like(T_arr)
    for m in range(n_terms):
        n = 2 * m + 1
        s += (8.0 / (n * np.pi) ** 2) * np.exp(-((n * np.pi / 2.0) ** 2) * T_arr)
    U = 1.0 - s
    if np.isscalar(T_v):
        return float(U[0])
    return U
