"""Piecewise-linear freezing characteristic curve and its temperature
derivative (Eq. 17 in the accompanying paper / label
``eq:freezing_curve``):

           | 1 - S_w_res        T <= T_f - dTc
    S_i =  | (T_f - T) / dTc    T_f - dTc < T < T_f
           | 0                  T >= T_f

Here ``T_f`` is the bulk freezing point (0 C for pure water at
atmospheric pressure), ``dTc`` is the freezing interval width, and
``S_w_res`` is the residual unfrozen water saturation in the fully
frozen state (typical for fine-grained soils; 0 for pure ice).

The temperature derivative ``dS_i/dT`` is needed by the apparent-heat-
capacity formulation of the energy equation (Eq. 176 in the accompanying paper):

    [(rho c)_eff + rho_w L_f phi (-dS_i/dT)] dT/dt
        = d/dz (lambda_eff dT/dz) - rho_w c_w q dT/dz + Q

Note the sign convention: ``S_i`` increases as ``T`` decreases, so
``dS_i/dT <= 0``.  The latent-heat term ``rho_w L_f phi |dS_i/dT|``
is the positive apparent-capacity spike inside the freezing interval.

We use ``rho_w`` (liquid water density) as the prefactor in the latent
heat term -- this is the convention of the empirical paper's
``column_enthalpy.py`` and matches the Bonacina (1973) and Nicolsky &
Romanovsky (2007) literature where the latent heat is expressed per
unit pore-water mass.  The accompanying paper writes it as ``rho_i L_f phi
dS_i/dt``; the two forms agree to the ratio ``rho_i / rho_w = 0.917``
which we absorb into the effective ``L_f`` if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FreezingCurveParams:
    """Parameters of the piecewise-linear freezing curve.

    Attributes
    ----------
    T_f : float
        Bulk freezing temperature (same units as the input ``T``;
        typically degC = 0.0 or K = 273.15).
    dTc : float
        Freezing interval width (same units as ``T``).  Must be > 0.
        Sharp phase change requires ``dTc`` small but finite; typical
        production values for permafrost simulations are 0.1-1.0 K.
    S_w_residual : float
        Residual unfrozen water saturation in the fully frozen state.
        0.0 for pure ice; ~0.05-0.15 for fine-grained soils with
        adsorbed water (Romanovsky & Osterkamp 2000).
    """

    T_f: float = 0.0
    dTc: float = 1.0
    S_w_residual: float = 0.0

    def __post_init__(self) -> None:  # noqa: D401
        if self.dTc <= 0:
            raise ValueError(f"dTc must be positive, got {self.dTc}")
        if not 0.0 <= self.S_w_residual < 1.0:
            raise ValueError(f"S_w_residual must be in [0, 1), got {self.S_w_residual}")


def ice_saturation(
    T: np.ndarray | float,
    params: FreezingCurveParams | None = None,
) -> np.ndarray:
    """Piecewise-linear ice saturation S_i(T) per Eq. 17 in the accompanying paper.

    Vectorised over ``T``.  Returns the same shape as ``T`` (or a
    0-d array if ``T`` is a scalar).
    """
    if params is None:
        params = FreezingCurveParams()
    Tf = params.T_f
    dTc = params.dTc
    Sr = params.S_w_residual
    T_arr = np.asarray(T, dtype=float)
    S_i = np.zeros_like(T_arr)
    fully_frozen = T_arr <= Tf - dTc
    interval = (T_arr > Tf - dTc) & (T_arr < Tf)
    S_i[fully_frozen] = 1.0 - Sr
    S_i[interval] = (Tf - T_arr[interval]) / dTc * (1.0 - Sr)
    return S_i


def ice_saturation_derivative(
    T: np.ndarray | float,
    params: FreezingCurveParams | None = None,
) -> np.ndarray:
    """``dS_i/dT`` for the piecewise-linear curve.

    Equals ``-(1 - S_w_res) / dTc`` inside the freezing interval and
    0 elsewhere.  At the interval endpoints the function is
    non-smooth; we adopt the right-continuous convention (the
    derivative is 0 at ``T = T_f`` and equals ``-(1 - S_r)/dTc`` at
    ``T = T_f - dTc``).
    """
    if params is None:
        params = FreezingCurveParams()
    Tf = params.T_f
    dTc = params.dTc
    Sr = params.S_w_residual
    T_arr = np.asarray(T, dtype=float)
    dSi_dT = np.zeros_like(T_arr)
    interval = (T_arr > Tf - dTc) & (T_arr < Tf)
    dSi_dT[interval] = -(1.0 - Sr) / dTc
    return dSi_dT


def apparent_volumetric_capacity(
    T: np.ndarray | float,
    *,
    rho_c_dry: np.ndarray | float,
    porosity: float | np.ndarray,
    L_f: float,
    rho_w: float,
    params: FreezingCurveParams | None = None,
) -> np.ndarray:
    """Apparent volumetric heat capacity (J m^-3 K^-1) including the
    latent-heat spike inside the freezing interval.

    ``rho_c_dry`` is the volumetric heat capacity of the saturated medium
    in the absence of the latent-heat contribution -- it can be a scalar
    constant or a depth-resolved profile; both broadcasts are supported.
    Outside the freezing interval the function returns ``rho_c_dry``;
    inside the interval it adds ``rho_w L_f phi (1 - S_w_res) / dTc``,
    which is the analytical limit of ``rho_w L_f phi |dS_i/dT|`` for the
    piecewise-linear curve.
    """
    if params is None:
        params = FreezingCurveParams()
    T_arr = np.asarray(T, dtype=float)
    base = np.broadcast_to(np.asarray(rho_c_dry, dtype=float), T_arr.shape).copy()
    Tf = params.T_f
    dTc = params.dTc
    Sr = params.S_w_residual
    interval = (T_arr > Tf - dTc) & (T_arr < Tf)
    spike = rho_w * L_f * np.asarray(porosity) * (1.0 - Sr) / dTc
    if np.ndim(spike) == 0:
        base[interval] = base[interval] + float(spike)
    else:
        spike_b = np.broadcast_to(spike, T_arr.shape)
        base = np.where(interval, base + spike_b, base)
    return base
