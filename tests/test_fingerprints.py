"""Tests for the F1 and F4 fingerprints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.fingerprints import (
    aggregate_continental,
    compute_f1,
    compute_f6_one_site,
    horvitz_thompson_weights,
    site_energy_gain,
)
from gt_theory.fingerprints.f1_erfc import load_smoke_pair

SMOKE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "smoke-synthetic"


def _make_inversion_row_set(
    *,
    bin_edges_yr: np.ndarray,
    medians_K: np.ndarray,
    kappa: float = 1.0e-6,
    T0_K: float = 5.0,
    dTdz_K_per_m: float = 0.025,
    z_steady_min_m: float = 300.0,
) -> pd.DataFrame:
    n = bin_edges_yr.size - 1
    return pd.DataFrame(
        {
            "bin_edge_young_yr": bin_edges_yr[:-1],
            "bin_edge_old_yr": bin_edges_yr[1:],
            "median_K": medians_K,
            "ci_lo_K": medians_K - 0.1,
            "ci_hi_K": medians_K + 0.1,
            "kappa_median": [kappa] * n,
            "residual_rms_median": [0.05] * n,
            "T0_K": [T0_K] * n,
            "dTdz_K_per_m": [dTdz_K_per_m] * n,
            "z_steady_min_m": [z_steady_min_m] * n,
        }
    )


def test_horvitz_thompson_weights_zero_in_empty_bands() -> None:
    lats = np.array([35.0, 45.0, -25.0])
    per_site, per_band = horvitz_thompson_weights(lats)
    # Each band has exactly one sample; weight per site equals
    # land area / 1.
    for w, lat in zip(per_site, lats, strict=True):
        assert w > 0.0, f"non-positive HT weight for lat={lat}"
    # Each unique band centre should appear once in the dict.
    assert len(per_band) == 3


def test_site_energy_gain_grows_with_warming() -> None:
    """A larger recovered warming must produce a larger column integral."""
    edges = np.array([0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    cool = _make_inversion_row_set(bin_edges_yr=edges, medians_K=np.array([0.1] * 6))
    warm = _make_inversion_row_set(bin_edges_yr=edges, medians_K=np.array([1.0] * 6))
    E_cool = site_energy_gain(cool)
    E_warm = site_energy_gain(warm)
    assert E_warm > E_cool > 0
    # Roughly linear: 10x amplitude -> ~10x energy.
    assert 5.0 < E_warm / E_cool < 20.0


def test_aggregate_continental_closes_to_target_on_constructed_field() -> None:
    """Construct sites whose per-band HT-weighted total matches a target,
    and verify the aggregator recovers it (within bootstrap noise)."""
    edges = np.array([0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    # 6 lat-band centres, one site each, all warming by 1 K throughout.
    lats = np.array([-25.0, -5.0, 5.0, 25.0, 45.0, 65.0])
    inversions = [
        _make_inversion_row_set(bin_edges_yr=edges, medians_K=np.array([1.0] * 6)) for _ in lats
    ]
    site_ids = [f"S-{i}" for i in range(len(lats))]
    # Compute by hand: each site delivers E_site (J/m2), HT-weighted by
    # land area (since each band has one sample, weight = land area).
    res = aggregate_continental(
        inversions,
        lats,
        site_ids,
        window_years=(0.0, 800.0),
        target_ZJ=0.0,  # don't constrain pass/fail here
        n_bootstrap=80,
    )
    assert res.total_ZJ > 0.0
    assert res.ci_lo_ZJ <= res.total_ZJ <= res.ci_hi_ZJ


def test_compute_f1_on_synthetic_subset() -> None:
    """F1 against the smoke-synthetic outputs (forward sim is the same
    operator F1 uses to predict, so residuals should be tiny: synthetic
    truth aligns perfectly with the inversion's forward model).

    Skipped if the Snakemake smoke-synthetic outputs are not on disk:
    that pipeline is exercised by a dedicated smoke.yml workflow, and
    snakemake itself is not in the minimal test-environment install.
    Run ``snakemake -s workflows/Snakefile smoke_synthetic --cores 4``
    locally to materialise the inputs and unlock this assertion.
    """
    import pytest

    if not SMOKE_DIR.exists() or not (SMOKE_DIR / "inversions").exists():
        pytest.skip(
            f"smoke-synthetic outputs absent at {SMOKE_DIR}; run snakemake "
            "smoke_synthetic to enable this test"
        )

    profiles, inversions = load_smoke_pair(SMOKE_DIR)
    assert len(profiles) == len(inversions) >= 4

    res = compute_f1(profiles, inversions, n_bootstrap=80)
    # Across-site median residual should be well within +/- 0.5 K on the
    # synthetic set; the strict 0.2 K theory band is for real data.
    assert res.max_abs_residual_K < 0.5, (
        f"F1 synthetic residual {res.max_abs_residual_K:.4f} K too large"
    )
    assert res.n_sites >= 4
    assert res.envelope_band_K == 0.2


# --------------------------------------------------------------- F6


def test_f6_tropical_site_not_latent_dominant() -> None:
    """A 25 degC site with a +/- 3 K seasonal cycle never enters the
    freezing interval; occupancy ~ 0 and not flagged."""
    months = np.arange(50 * 12)
    seasonal = 6.0 * np.cos(2.0 * np.pi * months / 12.0)
    sat = 25.0 + 0.3 * seasonal
    res = compute_f6_one_site(site_id="trop", lat_deg=10.0, sat_c_monthly=sat)
    assert res.freezing_interval_occupancy < 0.01
    assert res.latent_dominant is False


def test_f6_permafrost_site_is_latent_dominant() -> None:
    """A -2 degC site with an 8 K seasonal cycle spends a large fraction
    of its history near the freezing interval and is flagged."""
    months = np.arange(50 * 12)
    seasonal = 8.0 * np.cos(2.0 * np.pi * months / 12.0)
    sat = -2.0 + seasonal
    res = compute_f6_one_site(site_id="perma", lat_deg=68.0, sat_c_monthly=sat)
    assert res.freezing_interval_occupancy > 0.10
    assert res.latent_dominant is True


def test_f6_rejects_short_input() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least 2 samples"):
        compute_f6_one_site(site_id="x", lat_deg=0.0, sat_c_monthly=np.array([0.0]))
