"""Bonacina (1973) enthalpy-budget closure check for freeze--thaw.

The apparent-heat-capacity method tracks latent heat by absorbing
it into a state-dependent c_app(T) inside the freezing interval.
A necessary consistency check is that the integrated enthalpy
budget closes: the change in column-integrated enthalpy over a
time interval should equal the net heat flux through the
boundaries minus any internal sources.

For an adiabatic column with no flow, the enthalpy budget is

    int_0^L  H(z, t)  dz  =  int_0^L  H(z, 0)  dz  +  Q_top(t),

with the enthalpy density

    H(z) = (1 - phi) rho_r c_r T + phi[(1 - S_i) rho_w c_w
            + S_i rho_i c_i] T - phi rho_w L_f S_i,

(the last term is the latent-heat contribution; freezing decreases
H at constant T).

Reference: Bonacina, Comini, Fasano & Primicerio (1973), *Numerical
solution of phase-change problems*, Int. J. Heat Mass Transfer
16, 1825-1832.
"""

from __future__ import annotations

import numpy as np


def enthalpy_density(
    T: np.ndarray,
    S_i: np.ndarray,
    *,
    porosity: float,
    rho_r: float,
    c_r: float,
    rho_w: float,
    c_w: float,
    rho_i: float,
    c_i: float,
    L_f: float,
    T_ref: float = 0.0,
) -> np.ndarray:
    """Volumetric enthalpy H(z, t) (J m^-3).

    The latent-heat contribution is ``- phi rho_w L_f S_i``, i.e.
    freezing decreases H at constant T (latent heat is released
    on freezing, then carried as a negative-S_i contribution in
    this sign convention).
    """
    T_a = np.asarray(T, dtype=float)
    S_a = np.asarray(S_i, dtype=float)
    phi = float(porosity)
    sensible = (
        (1.0 - phi) * rho_r * c_r + phi * ((1.0 - S_a) * rho_w * c_w + S_a * rho_i * c_i)
    ) * (T_a - T_ref)
    latent = -phi * rho_w * L_f * S_a
    return sensible + latent


def column_integrated_enthalpy(
    T: np.ndarray,
    S_i: np.ndarray,
    z: np.ndarray,
    **kwargs,
) -> float:
    """Integrate H(z) over the column using the trapezoidal rule."""
    H = enthalpy_density(T, S_i, **kwargs)
    return float(np.trapezoid(H, z))


def boundary_heat_flux_integrated(
    *,
    lambda_eff_top: float,
    grad_T_top: np.ndarray,
    lambda_eff_bot: float,
    grad_T_bot: np.ndarray,
    t: np.ndarray,
) -> float:
    """Time-integrated net heat input through the top and bottom
    boundaries: ``int_0^t  lambda dT/dz |_top - lambda dT/dz |_bot  dt``.

    Conventions:
        * positive ``grad_T_top``  => heat IN at the top.
        * positive ``grad_T_bot``  => heat OUT at the bottom.
    """
    flux_in = lambda_eff_top * grad_T_top - lambda_eff_bot * grad_T_bot
    return float(np.trapezoid(flux_in, t))
