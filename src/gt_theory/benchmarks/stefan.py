"""Analytical one-phase Neumann similarity solution to the Stefan
problem in a semi-infinite porous medium.

Setup
-----
A semi-infinite column z >= 0 is initially at the freezing temperature
T_f (no super-cooling in the liquid phase).  At t = 0 the surface
temperature is dropped to T_s < T_f and held there.  A freezing front
propagates downward at position

    xi(t) = 2 lambda sqrt(kappa t),

where ``kappa = lambda_thermal / (rho c)_eff`` is the frozen-phase
thermal diffusivity and ``lambda`` solves the transcendental

    lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi),

with the Stefan number

    St = c_solid * (T_f - T_s) / L_f_volumetric,
    L_f_volumetric = rho_w * L_f * phi * (1 - S_w_residual).

Inside the frozen region 0 <= z <= xi(t), the temperature follows an
erf-profile that joins the surface BC and the moving front:

    T(z, t) = T_s + (T_f - T_s) * erf(z / (2 sqrt(kappa t))) / erf(lambda).

Beyond the front (z > xi(t)) the medium is unfrozen at T_f.

References
----------
* Carslaw and Jaeger, *Conduction of Heat in Solids*, 2nd ed., 1959,
  §11.2 (the classical one-phase Neumann solution).
* This is the standard benchmark used by Kurylyk et al. (2014),
  Painter & Karra (2014), and the empirical paper's
  ``column_enthalpy.py`` (where ``neumann_stefan_lambda`` already
  exists and is reused here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from gt_theory.solvers.column_enthalpy import neumann_stefan_lambda


@dataclass(frozen=True)
class StefanOnePhaseParams:
    """Inputs for the one-phase Neumann similarity solution.

    Attributes
    ----------
    T_s : float
        Imposed surface temperature (degC or K).  Must satisfy
        ``T_s < T_f``.
    T_f : float
        Initial uniform medium temperature, taken equal to the bulk
        freezing point.
    lambda_thermal : float
        Effective thermal conductivity of the *frozen* phase
        (W m^-1 K^-1).
    rho_c_solid : float
        Volumetric heat capacity of the *frozen* phase, excluding
        latent heat (J m^-3 K^-1).
    porosity : float
        Pore-volume fraction.
    L_f : float
        Specific latent heat of fusion (J kg^-1).
    rho_w : float
        Liquid-water density (kg m^-3); convention matched to
        ``column_enthalpy.py``.
    S_w_residual : float
        Residual unfrozen water saturation at the front (typical
        0 for pure-ice benchmarks).
    """

    T_s: float
    T_f: float = 0.0
    lambda_thermal: float = 2.5
    rho_c_solid: float = 2.5e6
    porosity: float = 0.30
    L_f: float = 3.34e5
    rho_w: float = 1000.0
    S_w_residual: float = 0.0

    @property
    def stefan_number(self) -> float:
        L_v = self.rho_w * self.L_f * self.porosity * (1.0 - self.S_w_residual)
        return self.rho_c_solid * (self.T_f - self.T_s) / L_v

    @property
    def kappa(self) -> float:
        return self.lambda_thermal / self.rho_c_solid

    @property
    def stefan_lambda(self) -> float:
        return neumann_stefan_lambda(self.stefan_number)


def stefan_front_position(
    t: np.ndarray | float,
    params: StefanOnePhaseParams,
) -> np.ndarray:
    """Position of the freezing front xi(t) = 2 lambda sqrt(kappa t).

    Returns 0 at ``t = 0``.  Vectorised over ``t``.
    """
    t_arr = np.asarray(t, dtype=float)
    lam = params.stefan_lambda
    kap = params.kappa
    return 2.0 * lam * np.sqrt(np.maximum(kap * t_arr, 0.0))


def stefan_temperature_profile(
    z: np.ndarray,
    t: float,
    params: StefanOnePhaseParams,
) -> np.ndarray:
    """Analytical T(z, t) profile.

    Inside the frozen region 0 <= z <= xi(t) the temperature is

        T(z, t) = T_s + (T_f - T_s) erf(z / (2 sqrt(kappa t))) / erf(lambda).

    Beyond the front (z > xi) the medium is at the initial T_f.
    """
    if t <= 0.0:
        out = np.full_like(z, params.T_f, dtype=float)
        out[z <= 0.0] = params.T_s  # apply BC at z=0
        return out
    lam = params.stefan_lambda
    kap = params.kappa
    eta = z / (2.0 * np.sqrt(kap * t))
    xi_t = 2.0 * lam * np.sqrt(kap * t)
    T = np.full_like(z, params.T_f, dtype=float)
    in_frozen = z <= xi_t
    T[in_frozen] = params.T_s + (params.T_f - params.T_s) * erf(eta[in_frozen]) / erf(lam)
    return T
