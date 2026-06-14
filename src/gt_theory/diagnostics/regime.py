"""ℒ regime number per Theory Eq. 279.

  ℒ = rho_i * L_f * phi / [ (rho c)_eff * Delta_T ]

Captures the ratio of pore-ice latent-heat storage to sensible-heat
storage on the characteristic surface-temperature scale.  ℒ > 1
denotes a regime where the latent-heat buffer is the dominant
thermal sink/source and the pure-sensible solver
``column_coupled.py`` is inadequate.  Theory paper Table 1 brackets:

  - Permafrost (shallow)         ℒ ∼ 10 -- 100
  - Geothermal reservoir         ℒ ∼ 0.1 -- 1
  - Arid basin interior          ℒ ∼ 0.01 -- 0.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gt_theory.theory.dimless import L_FUSION, RHO_C_EFF_DEFAULT, RHO_ICE


@dataclass(frozen=True)
class RegimeBands:
    permafrost: float = 10.0
    intermediate: float = 1.0
    arid: float = 0.1


def latent_heat_regime_l(
    *,
    porosity: float | np.ndarray,
    delta_T_K: float | np.ndarray,
    rho_c_eff: float | np.ndarray = RHO_C_EFF_DEFAULT,
    rho_i: float = RHO_ICE,
    L_f: float = L_FUSION,
) -> np.ndarray:
    """Per-site ℒ.

    All scalar inputs broadcast.  ``delta_T_K`` is the characteristic
    temperature scale; for the empirical paper we use the
    inter-quantile range of the recent ground-surface temperature
    anomaly distribution at the site (gst_p95 - gst_p05).
    """
    phi = np.asarray(porosity, dtype=float)
    dT = np.asarray(delta_T_K, dtype=float)
    rc = np.asarray(rho_c_eff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rho_i * L_f * phi) / (rc * np.where(dT > 0, dT, np.nan))


def classify_regime(
    l_values: np.ndarray,
    bands: RegimeBands | None = None,
) -> np.ndarray:
    """Per-site label: 'permafrost', 'intermediate', 'arid', or
    'undetermined' for NaN."""
    bands = bands if bands is not None else RegimeBands()
    out = np.full(l_values.shape, "undetermined", dtype=object)
    permafrost = l_values >= bands.intermediate
    intermediate = (l_values >= bands.arid) & (l_values < bands.intermediate)
    arid = (l_values < bands.arid) & np.isfinite(l_values)
    out[permafrost] = "permafrost"
    out[intermediate] = "intermediate"
    out[arid] = "arid"
    return out
