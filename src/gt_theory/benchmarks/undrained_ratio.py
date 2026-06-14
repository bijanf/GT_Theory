"""Undrained-limit thermo-poroelastic pressure response.

In the undrained limit (no fluid mass exchange across the boundary
on the thermal-diffusion timescale), the linearised
thermo-poroelastic constitutive equations under constant total
stress give the instantaneous pressure response to a temperature
perturbation as

    dp/dT = alpha_w / beta_w.

Reference: Detournay & Cheng (1993), *Fundamentals of poroelasticity*,
Eq. 5.10.  This is the rigid-skeleton, fully-confined limit; the
analytic ratio is independent of the rock-matrix properties.
"""

from __future__ import annotations


def expected_dp_dT(*, alpha_w: float, beta_w: float) -> float:
    """``alpha_w / beta_w`` (Pa K^-1).

    Pore-fluid thermal expansion divided by isothermal
    compressibility.  Independent of rock-skeleton parameters in
    the constant-total-stress undrained limit.
    """
    if alpha_w <= 0 or beta_w <= 0:
        raise ValueError("alpha_w and beta_w must be positive")
    return alpha_w / beta_w
