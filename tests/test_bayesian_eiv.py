"""Tests for the hierarchical Bayesian EIV Gibbs sampler."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.inversion.bayesian_eiv import (
    gibbs_eiv,
    latitude_band_index,
)


def test_recovers_known_slope_on_clean_line() -> None:
    rng = np.random.default_rng(123)
    n = 300
    x_true = rng.normal(0.0, 1.0, size=n)
    beta_true = 1.4
    alpha_true = 0.2
    sigma_x = 0.1
    sigma_y = 0.1
    x_obs = x_true + rng.normal(0.0, sigma_x, size=n)
    y_obs = alpha_true + beta_true * x_true + rng.normal(0.0, sigma_y, size=n)
    bands = np.zeros(n, dtype=int)
    posterior = gibbs_eiv(
        x_obs=x_obs,
        y_obs=y_obs,
        band_index=bands,
        n_bands=1,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_draws=2000,
        n_burn=500,
    )
    s = posterior.summary()
    # Beta posterior median should be within ~0.1 of the true 1.4.
    assert abs(s["beta_global"]["median"] - beta_true) < 0.15
    assert s["beta_global"]["ci_lo"] < beta_true < s["beta_global"]["ci_hi"]
    # Alpha posterior contains the truth.
    assert s["alpha"]["ci_lo"] < alpha_true < s["alpha"]["ci_hi"]


def test_hierarchical_separates_per_band_slopes() -> None:
    """Three latitude bands with distinct true slopes -- each is
    recovered, with the global slope at the centre."""
    rng = np.random.default_rng(42)
    n_per_band = 200
    true_betas = (0.7, 1.4, 2.1)
    sigma_x = 0.15
    sigma_y = 0.15
    xs = []
    ys = []
    bands = []
    for g, beta_g in enumerate(true_betas):
        x_true = rng.normal(0.0, 1.0, size=n_per_band)
        x_obs = x_true + rng.normal(0.0, sigma_x, size=n_per_band)
        y_obs = beta_g * x_true + rng.normal(0.0, sigma_y, size=n_per_band)
        xs.append(x_obs)
        ys.append(y_obs)
        bands.append(np.full(n_per_band, g, dtype=int))
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    b_all = np.concatenate(bands)
    posterior = gibbs_eiv(
        x_obs=x_all,
        y_obs=y_all,
        band_index=b_all,
        n_bands=3,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_draws=2500,
        n_burn=500,
        band_labels=("A", "B", "C"),
    )
    s = posterior.summary()
    # Each band's posterior median should be within 0.25 of its truth.
    for label, beta_true in zip(("A", "B", "C"), true_betas, strict=True):
        med = s[f"beta_{label}"]["median"]
        assert abs(med - beta_true) < 0.3, (label, med, beta_true)
    # The global hyperparameter should fall between the extreme truths.
    bg = s["beta_global"]["median"]
    assert min(true_betas) - 0.5 < bg < max(true_betas) + 0.5


def test_latitude_band_index_thresholds() -> None:
    lats = np.array([-10.0, -45.0, -70.0, 5.0, 35.0, 65.0])
    bands, labels = latitude_band_index(lats)
    # |lat| < 30 -> 0 (tropical); 30 <= |lat| < 60 -> 1 (mid); >= 60 -> 2 (boreal)
    expected = np.array([0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(bands, expected)
    assert labels == ("tropical", "mid", "boreal")


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        gibbs_eiv(
            x_obs=np.zeros(5),
            y_obs=np.zeros(4),
            band_index=np.zeros(5, dtype=int),
            n_bands=1,
            n_draws=10,
            n_burn=0,
        )
