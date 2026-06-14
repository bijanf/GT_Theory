"""State-dependent effective properties for the merged enthalpy-coupled
solver.

Implements the saturation-dependent closures of the accompanying
paper:

* **Eq. 198 / label ``eq:lambda_eff``** -- geometric-mean thermal
  conductivity:

      lambda_eff = lambda_r^(1 - phi) * lambda_w^(phi * S_w)
                                      * lambda_i^(phi * S_i)

* **Eq. 190 / label ``eq:relperm``** -- Brooks-Corey relative
  permeability:

      k_rel(S_w) = S_w^eta,   eta ~ 3 for many porous media

* **Eq. 220 / label ``eq:rho_c_eff``** -- volumetric heat capacity:

      (rho c)_eff = (1 - phi) rho_r c_r
                    + phi (S_w rho_w c_w + S_i rho_i c_i)

All functions vectorise over ``S_i`` (and therefore implicitly over
``S_w = 1 - S_i``).  Constant-property defaults come from the
``gt_theory.theory.dimless`` module.

Notes
-----
* The geometric-mean form for thermal conductivity is the standard
  parallel/series average used in permafrost simulators (e.g.
  Romanovsky et al. 2009; Lawrence & Slater 2008).  The arithmetic
  mean (with effective-medium corrections) is also defensible; the
  geometric form is what the accompanying paper specifies.
* k_rel approaches zero rapidly when S_w -> 0 (S_i -> 1); the merged
  solver therefore exhibits the expected behaviour that ice formation
  effectively shuts off Darcy flow until thaw, even though we do not
  enforce a hard k_rel = 0 cutoff.
"""

from __future__ import annotations

import numpy as np

from gt_theory.theory.dimless import (
    RHO_ICE,
    RHO_WATER,
)

# Reference thermal conductivities (W m^-1 K^-1) at 0 C for the
# components used in the geometric-mean closure.
LAMBDA_ROCK_DEFAULT: float = 2.5  # quartz-feldspar dominated continental crust
LAMBDA_WATER: float = 0.58  # liquid water
LAMBDA_ICE: float = 2.22  # pure ice

# Reference specific heats (J kg^-1 K^-1)
C_WATER: float = 4.186e3
C_ICE: float = 2.108e3
C_ROCK_DEFAULT: float = 800.0  # silicate rock matrix

# Reference matrix density (kg m^-3) for continental crust
RHO_ROCK_DEFAULT: float = 2700.0


def lambda_eff_geometric_mean(
    S_i: np.ndarray | float,
    *,
    porosity: float,
    lambda_r: float = LAMBDA_ROCK_DEFAULT,
    lambda_w: float = LAMBDA_WATER,
    lambda_i: float = LAMBDA_ICE,
) -> np.ndarray:
    """Effective thermal conductivity by geometric mean.

    ``S_i`` is the ice saturation (0..1); ``S_w = 1 - S_i`` is the
    liquid water saturation.  Equation 198 of the accompanying paper:

        lambda_eff = lambda_r^(1-phi) * lambda_w^(phi S_w)
                                      * lambda_i^(phi S_i)

    Returns the same shape as ``S_i``.
    """
    S_i_arr = np.asarray(S_i, dtype=float)
    S_w_arr = 1.0 - S_i_arr
    phi = float(porosity)
    return (lambda_r ** (1.0 - phi)) * (lambda_w ** (phi * S_w_arr)) * (lambda_i ** (phi * S_i_arr))


def brooks_corey_k_rel(
    S_i: np.ndarray | float,
    *,
    eta: float = 3.0,
    S_w_residual: float = 0.0,
) -> np.ndarray:
    """Brooks-Corey relative permeability k_rel(S_w) = S_w^eta.

    Expressed as a function of ``S_i`` for symmetry with the freezing
    curve interface.  Reduces to ``1.0`` at ``S_i = 0`` (fully liquid)
    and to ``S_w_residual^eta`` at ``S_i = 1 - S_w_residual`` (fully
    frozen with residual unfrozen water).
    """
    S_i_arr = np.asarray(S_i, dtype=float)
    S_w_arr = np.clip(1.0 - S_i_arr, 0.0, 1.0)
    # Optional Brooks-Corey effective saturation; here we use the
    # simple S_w^eta form quoted in the accompanying paper.
    return S_w_arr ** float(eta)


def rho_c_eff_two_phase(
    S_i: np.ndarray | float,
    *,
    porosity: float,
    rho_r: float = RHO_ROCK_DEFAULT,
    c_r: float = C_ROCK_DEFAULT,
    rho_w: float = RHO_WATER,
    c_w: float = C_WATER,
    rho_i: float = RHO_ICE,
    c_i: float = C_ICE,
) -> np.ndarray:
    """Volumetric heat capacity per Eq. 220 of the accompanying paper.

    (rho c)_eff = (1 - phi) rho_r c_r
                  + phi (S_w rho_w c_w + S_i rho_i c_i)

    This is the *sensible* part of the heat capacity.  The latent-heat
    contribution is added separately by
    ``freezing_curve.apparent_volumetric_capacity``.
    """
    S_i_arr = np.asarray(S_i, dtype=float)
    S_w_arr = 1.0 - S_i_arr
    phi = float(porosity)
    return (1.0 - phi) * rho_r * c_r + phi * (S_w_arr * rho_w * c_w + S_i_arr * rho_i * c_i)
