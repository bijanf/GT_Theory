"""The nine dimensionless numbers of the coupled thermal-hydraulic
framework (Section 3 of the accompanying paper).

We adopt the scaling conventions of the accompanying paper:

    N_p     = S_s * Delta_p / (phi)                    storage / pressure
    N_s     = phi                                       saturation storage
    N_alpha = phi * S_w * beta_w * Delta_p              thermal expansion storage
    Gamma   = (rho_0 * alpha_w * Delta_T) / mu          thermal-hydraulic coupling
    N_Q     = Q_w * tau / phi                           water source
    Fo      = kappa * tau / L^2                         Fourier
    Pe_T    = v_darcy * L / kappa                       thermal Peclet
    L_calL  = rho_i * L_f * phi / ((rho c)_eff * Delta_T)   latent heat
    Q_calQ  = q_dot * tau / ((rho c)_eff * Delta_T)     heat source

These nine numbers partition the subsurface into regimes (Section 7 of
the accompanying paper).  The two diagrams that anchor Figure 5 are:

* Pe_T - L_calL  (heat transport: conduction vs advection vs latent)
* N_p   - Gamma * N_alpha  (mass transport: pressure vs thermal feedback)

This module computes the nine numbers from a curated set of per-site
parameters, with sensible lithology defaults when site-specific values
are unavailable.  Defaults can be overridden per site or globally.
"""

from __future__ import annotations

from dataclasses import dataclass

YEAR_S: float = 365.25 * 86400.0

# --- physical constants (mks units) -------------------------------------
RHO_WATER: float = 1000.0  # kg m^-3
RHO_ICE: float = 917.0  # kg m^-3
L_FUSION: float = 3.34e5  # J kg^-1 (latent heat of fusion of water)
ALPHA_WATER: float = 2.1e-4  # K^-1 (thermal expansion of liquid water at ~10 C)
BETA_WATER: float = 4.5e-10  # Pa^-1 (isothermal compressibility of water)
MU_WATER: float = 1.0e-3  # Pa s (dynamic viscosity, ~20 C)
G_EARTH: float = 9.81  # m s^-2

# Default effective volumetric heat capacity for saturated continental crust.
RHO_C_EFF_DEFAULT: float = 2.5e6  # J m^-3 K^-1


@dataclass(frozen=True)
class SiteDimensionlessParams:
    """All inputs needed to compute the nine numbers at one site.

    Fields with `None` are treated as defaults from the per-method
    overrides below.
    """

    L_m: float = 600.0  # characteristic depth (m)
    tau_s: float = 100.0 * YEAR_S  # characteristic time (s)
    kappa_m2_s: float = 1.0e-6  # thermal diffusivity (m^2 s^-1)
    rho_c_eff: float = RHO_C_EFF_DEFAULT  # effective volumetric heat capacity
    porosity: float = 0.15  # phi
    sat_water: float = 1.0  # S_w (assume saturated)
    sat_ice: float = 0.0  # S_i (ice fraction); reset for permafrost sites
    specific_storage: float = 1.0e-5  # S_s (m^-1)
    delta_T_K: float = 1.0  # characteristic Delta_T at the surface (K)
    delta_p_Pa: float = 1.0e5  # characteristic pressure scale (Pa); ~1 bar default
    v_darcy_m_s: float = 0.0  # vertical Darcy velocity (m s^-1); set per site
    Q_w_kg_m3_s: float = 0.0  # climatic water source (kg m^-3 s^-1)
    q_dot_W_m3: float = 0.0  # volumetric heat source (W m^-3)
    beta_w_inv_Pa: float = BETA_WATER
    alpha_w_inv_K: float = ALPHA_WATER
    mu_Pa_s: float = MU_WATER
    rho_w_kg_m3: float = RHO_WATER
    L_f_J_per_kg: float = L_FUSION
    rho_i_kg_m3: float = RHO_ICE


@dataclass(frozen=True)
class DimensionlessNumbers:
    N_p: float
    N_s: float
    N_alpha: float
    Gamma: float
    N_Q: float
    Fo: float
    Pe_T: float
    L_calL: float  # latent-heat number
    Q_calQ: float  # heat source number


def compute_site_dimless(p: SiteDimensionlessParams) -> DimensionlessNumbers:
    """Compute the nine numbers for a single site."""
    N_p = p.specific_storage * p.delta_p_Pa / max(p.porosity, 1.0e-12)
    N_s = p.porosity
    N_alpha = p.porosity * p.sat_water * p.beta_w_inv_Pa * p.delta_p_Pa
    Gamma = (p.rho_w_kg_m3 * p.alpha_w_inv_K * p.delta_T_K) / p.mu_Pa_s
    N_Q = (p.Q_w_kg_m3_s * p.tau_s) / max(p.porosity, 1.0e-12)
    Fo = p.kappa_m2_s * p.tau_s / (p.L_m * p.L_m)
    Pe_T = p.v_darcy_m_s * p.L_m / max(p.kappa_m2_s, 1.0e-12)
    L_calL = (p.rho_i_kg_m3 * p.L_f_J_per_kg * max(p.porosity, 1.0e-12)) / (
        p.rho_c_eff * max(p.delta_T_K, 1.0e-12)
    )
    Q_calQ = (p.q_dot_W_m3 * p.tau_s) / (p.rho_c_eff * max(p.delta_T_K, 1.0e-12))
    return DimensionlessNumbers(
        N_p=float(N_p),
        N_s=float(N_s),
        N_alpha=float(N_alpha),
        Gamma=float(Gamma),
        N_Q=float(N_Q),
        Fo=float(Fo),
        Pe_T=float(Pe_T),
        L_calL=float(L_calL),
        Q_calQ=float(Q_calQ),
    )


def default_params_from_site(
    *,
    lat_deg: float,
    mean_kappa_w_m_k: float | None = None,
    delta_T_K: float = 1.0,
    porosity: float | None = None,
    sat_ice: float | None = None,
    v_darcy_m_s: float = 0.0,
) -> SiteDimensionlessParams:
    """Build a :class:`SiteDimensionlessParams` from the metadata that a
    borehole catalog row typically carries.

    Heuristic defaults:

    * Permafrost (|lat| >= 55, mean_kappa not warmer than typical):
      ice fraction ~ 0.2, porosity ~ 0.30 (silty material).
    * Otherwise: temperate saturated rock (phi = 0.15, S_w = 1, S_i = 0).
    """
    is_permafrost = abs(lat_deg) >= 55.0
    phi = porosity if porosity is not None else (0.30 if is_permafrost else 0.15)
    si = sat_ice if sat_ice is not None else (0.20 if is_permafrost else 0.0)
    return SiteDimensionlessParams(
        porosity=phi,
        sat_water=1.0 - si,
        sat_ice=si,
        delta_T_K=delta_T_K,
        v_darcy_m_s=v_darcy_m_s,
        kappa_m2_s=1.0e-6,  # generic; the inversion estimates a per-site value
    )
