"""Regime classification and occupancy permutation test for the two
hero diagrams (Figure 5 of the empirical companion):

* Pe_T - L_calL : heat-transport regimes (conduction/advection x
  sensible/latent).
* N_p - Gamma*N_alpha : mass-transport regimes (pressure-dominated /
  saturation-dominated; weak / strong thermal-hydraulic feedback).

Transition criteria (per Section 7.4 of the accompanying paper):
    Pe_T  ~ 1   : conduction <-> advection
    L_calL ~ 1  : sensible <-> latent dominance
    N_p   ~ 1   : pressure-diffusion <-> saturation-dominated storage
    Gamma*N_alpha ~ 1 : weak <-> strong thermal-hydraulic coupling
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PetLQuadrant(IntEnum):
    CONDUCTION_SENSIBLE = 0  # Pe_T << 1, L << 1
    CONDUCTION_LATENT = 1  # Pe_T << 1, L >> 1
    ADVECTION_SENSIBLE = 2  # Pe_T >> 1, L << 1
    ADVECTION_LATENT = 3  # Pe_T >> 1, L >> 1


class NpGNaQuadrant(IntEnum):
    PRESSURE_WEAK = 0  # N_p >> 1, Gamma*N_alpha << 1
    PRESSURE_STRONG = 1  # N_p >> 1, Gamma*N_alpha >> 1
    SATURATION_WEAK = 2  # N_p << 1, Gamma*N_alpha << 1
    SATURATION_STRONG = 3  # N_p << 1, Gamma*N_alpha >> 1


def assign_pet_l_quadrant(Pe_T: float, L: float, threshold: float = 1.0) -> PetLQuadrant:
    is_adv = Pe_T >= threshold
    is_lat = L >= threshold
    if is_adv and is_lat:
        return PetLQuadrant.ADVECTION_LATENT
    if is_adv and not is_lat:
        return PetLQuadrant.ADVECTION_SENSIBLE
    if (not is_adv) and is_lat:
        return PetLQuadrant.CONDUCTION_LATENT
    return PetLQuadrant.CONDUCTION_SENSIBLE


def assign_np_gna_quadrant(
    N_p: float, Gamma_N_alpha: float, threshold: float = 1.0
) -> NpGNaQuadrant:
    is_sat = N_p < threshold
    is_strong = Gamma_N_alpha >= threshold
    if (not is_sat) and is_strong:
        return NpGNaQuadrant.PRESSURE_STRONG
    if (not is_sat) and (not is_strong):
        return NpGNaQuadrant.PRESSURE_WEAK
    if is_sat and is_strong:
        return NpGNaQuadrant.SATURATION_STRONG
    return NpGNaQuadrant.SATURATION_WEAK


@dataclass(frozen=True)
class RegimeOccupancy:
    """Result of a permutation test that asks whether the observed
    co-occurrence of an external label (e.g. Koppen zone) with the
    predicted regime quadrant is more concentrated than chance.
    """

    statistic: float  # observed normalised mutual information (or chi2)
    null_mean: float
    null_sd: float
    p_value: float
    n_permutations: int


def _normalised_mutual_information(a: np.ndarray, b: np.ndarray, *, eps: float = 1.0e-12) -> float:
    """Normalised mutual information between two categorical labellings
    of the same N samples.  Returns a value in [0, 1].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    cats_a, ia = np.unique(a, return_inverse=True)
    cats_b, ib = np.unique(b, return_inverse=True)
    n = a.size
    contingency = np.zeros((cats_a.size, cats_b.size), dtype=float)
    np.add.at(contingency, (ia, ib), 1.0)
    p_ab = contingency / max(n, 1)
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms = p_ab * np.log((p_ab + eps) / (p_a @ p_b + eps))
    mi = float(np.nansum(mi_terms))
    h_a = float(-np.nansum(p_a * np.log(p_a + eps)))
    h_b = float(-np.nansum(p_b * np.log(p_b + eps)))
    denom = max(0.5 * (h_a + h_b), eps)
    return max(mi / denom, 0.0)


def permutation_test_occupancy(
    predicted_quadrants: np.ndarray,
    external_labels: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 20260522,
) -> RegimeOccupancy:
    """Test whether the predicted regime labelling carries information
    about an external categorical labelling (e.g. Koppen zone, GST
    sign).  H0: no association.

    Uses normalised mutual information as the statistic and a
    permutation test under H0 to derive a one-sided p-value.
    """
    rng = np.random.default_rng(seed)
    observed = _normalised_mutual_information(predicted_quadrants, external_labels)
    null = np.empty(n_permutations)
    shuffled = external_labels.copy()
    for k in range(n_permutations):
        rng.shuffle(shuffled)
        null[k] = _normalised_mutual_information(predicted_quadrants, shuffled)
    # One-sided: how often does the null exceed the observed?
    p = float((np.sum(null >= observed) + 1.0) / (n_permutations + 1.0))
    return RegimeOccupancy(
        statistic=observed,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        p_value=p,
        n_permutations=n_permutations,
    )
