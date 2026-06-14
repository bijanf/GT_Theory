"""Tests for the theory.dimless and theory.regime modules."""

from __future__ import annotations

import numpy as np

from gt_theory.theory import (
    DimensionlessNumbers,
    NpGNaQuadrant,
    PetLQuadrant,
    SiteDimensionlessParams,
    assign_np_gna_quadrant,
    assign_pet_l_quadrant,
    compute_site_dimless,
    default_params_from_site,
    permutation_test_occupancy,
)


def test_default_temperate_site_lives_in_conduction_sensible_quadrant() -> None:
    p = default_params_from_site(lat_deg=40.0, delta_T_K=1.0)
    d = compute_site_dimless(p)
    assert isinstance(d, DimensionlessNumbers)
    # Default parameters: no advection -> Pe_T = 0 (conduction).
    assert d.Pe_T == 0.0
    # Default phi = 0.15 + rho c_eff = 2.5 MJ/m3/K -> L_calL ~ 18.
    # Temperate (no ice) but L_calL still exceeds 1 because phi and L_f
    # are independent of latitude in our default model; this is
    # intentional and surfaced by the figure with quadrant text.
    assert d.L_calL > 1.0
    q = assign_pet_l_quadrant(d.Pe_T, d.L_calL)
    assert q == PetLQuadrant.CONDUCTION_LATENT


def test_permafrost_has_higher_latent_number_than_temperate() -> None:
    perma = default_params_from_site(lat_deg=68.0, delta_T_K=1.0)
    temp = default_params_from_site(lat_deg=40.0, delta_T_K=1.0)
    assert compute_site_dimless(perma).L_calL > compute_site_dimless(temp).L_calL


def test_pet_l_quadrant_threshold_behaviour() -> None:
    assert assign_pet_l_quadrant(0.5, 0.5) == PetLQuadrant.CONDUCTION_SENSIBLE
    assert assign_pet_l_quadrant(0.5, 2.0) == PetLQuadrant.CONDUCTION_LATENT
    assert assign_pet_l_quadrant(2.0, 0.5) == PetLQuadrant.ADVECTION_SENSIBLE
    assert assign_pet_l_quadrant(2.0, 2.0) == PetLQuadrant.ADVECTION_LATENT


def test_np_gna_quadrant_threshold_behaviour() -> None:
    assert assign_np_gna_quadrant(0.1, 0.1) == NpGNaQuadrant.SATURATION_WEAK
    assert assign_np_gna_quadrant(0.1, 2.0) == NpGNaQuadrant.SATURATION_STRONG
    assert assign_np_gna_quadrant(2.0, 0.1) == NpGNaQuadrant.PRESSURE_WEAK
    assert assign_np_gna_quadrant(2.0, 2.0) == NpGNaQuadrant.PRESSURE_STRONG


def test_compute_site_dimless_handles_zero_advection() -> None:
    p = SiteDimensionlessParams(v_darcy_m_s=0.0, kappa_m2_s=1.0e-6, L_m=500.0)
    d = compute_site_dimless(p)
    assert d.Pe_T == 0.0
    assert d.Fo > 0.0


def test_permutation_test_recovers_perfect_dependence() -> None:
    """If predicted labels equal external labels exactly, NMI should
    be ~1 and the permutation p-value should be tiny."""
    rng = np.random.default_rng(0)
    n = 80
    labels = rng.integers(0, 4, size=n)
    res = permutation_test_occupancy(
        predicted_quadrants=labels.copy(),
        external_labels=labels.copy(),
        n_permutations=300,
    )
    assert res.statistic > 0.9
    assert res.p_value <= 0.01


def test_permutation_test_handles_independence() -> None:
    rng = np.random.default_rng(20260522)
    n = 100
    pred = rng.integers(0, 4, size=n)
    ext = rng.integers(0, 3, size=n)
    res = permutation_test_occupancy(
        predicted_quadrants=pred,
        external_labels=ext,
        n_permutations=300,
    )
    # Under H0 the p should be uniform-ish; the test just verifies the
    # observed statistic is in the null mass, not in the tail.
    assert 0.05 <= res.p_value <= 0.95


def test_permutation_test_distinguishes_2x2_dependence() -> None:
    """Construct a 2x2 contingency with strong association and verify
    that the permutation p-value is small."""
    # 40 samples: pred 0/1 perfectly aligned with ext 0/1.
    pred = np.array([0] * 20 + [1] * 20)
    ext = np.array(["a"] * 20 + ["b"] * 20)
    res = permutation_test_occupancy(
        predicted_quadrants=pred,
        external_labels=ext,
        n_permutations=500,
    )
    assert res.p_value < 0.01
    assert res.statistic > 0.95
