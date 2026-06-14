"""Synthetic-twin tests for the hierarchical bootstrap Tikhonov
inversion.

We construct a known GST history, forward-simulate the corresponding
present-day temperature profile via the analytic forward operator, add
a geothermal gradient and 0.05 K Gaussian noise, and assert that the
posterior recovers the truth within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.inversion import (
    build_forward_operator,
    default_bin_edges_yr,
    detrend_geothermal,
    invert_posterior,
)


def _make_synthetic_profile(
    *,
    bin_amplitudes_K: np.ndarray,
    kappa: float = 1.0e-6,
    z: np.ndarray | None = None,
    bin_edges_yr: np.ndarray | None = None,
    T0: float = 5.0,
    dTdz: float = 0.025,
    noise_K: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic borehole profile T_obs(z) = T0 + dTdz*z + dT_anom(z) + noise."""
    rng = np.random.default_rng() if rng is None else rng
    z = np.linspace(20.0, 600.0, 30) if z is None else z
    edges = default_bin_edges_yr() if bin_edges_yr is None else bin_edges_yr
    G_true = build_forward_operator(z, edges, kappa)
    dT_anom = G_true @ bin_amplitudes_K
    T_steady = T0 + dTdz * z
    T_obs = T_steady + dT_anom + rng.normal(0.0, noise_K, size=z.size)
    return z, T_obs, dT_anom


def test_detrend_recovers_steady_state() -> None:
    """detrend_geothermal should recover the linear background within
    floating-point tolerance when no anomaly is present."""
    z = np.linspace(20.0, 600.0, 30)
    T_steady = 4.5 + 0.022 * z
    dT, T0, dTdz = detrend_geothermal(z, T_steady, z_steady_min_m=300.0)
    assert T0 == pytest.approx(4.5, abs=1e-6)
    assert dTdz == pytest.approx(0.022, abs=1e-8)
    np.testing.assert_allclose(dT, 0.0, atol=1e-6)


def test_forward_operator_shape_and_units() -> None:
    edges = default_bin_edges_yr()
    z = np.linspace(20.0, 600.0, 25)
    G = build_forward_operator(z, edges, kappa=1.0e-6)
    assert G.shape == (z.size, edges.size - 1)
    # erfc differences are in [0, 1]; the recent-bin column dominates near surface.
    assert np.all(G >= -1e-12)
    assert np.all(G <= 1.0 + 1e-12)
    # Surface (z=0) row would be all zeros (erfc(0) - erfc(0) = 0); deep rows decay.
    assert G[0, 0] > G[-1, 0]


def test_inversion_rejects_bad_inputs() -> None:
    z = np.linspace(20.0, 600.0, 30)
    dT = np.zeros_like(z)
    with pytest.raises(ValueError, match="sigma_T"):
        invert_posterior(z, dT, sigma_T=-0.05)
    with pytest.raises(ValueError, match="lam"):
        invert_posterior(z, dT, lam=-1.0)
    with pytest.raises(ValueError, match="n_bootstrap"):
        invert_posterior(z, dT, n_bootstrap=0)


def test_inversion_recovers_smooth_warming() -> None:
    """A smoothly decaying GST history (current-day +1.5 K decaying back
    in time) should be recovered with bin-wise posterior median within
    +/- 0.4 K of the truth, and the cumulative warming integral within
    +/- 30%.  We use a smooth truth because the first-difference
    smoothness prior implicit in the inversion biases sharp-step
    reconstructions toward smoother shapes (Beltrami et al. 2006)."""
    rng = np.random.default_rng(20260522)
    edges = default_bin_edges_yr()
    # Smooth exponential decay back in time, in K.
    truth = np.array([1.5, 1.1, 0.7, 0.4, 0.15, 0.0])
    z, T_obs, _ = _make_synthetic_profile(
        bin_amplitudes_K=truth,
        bin_edges_yr=edges,
        rng=rng,
    )
    dT, *_ = detrend_geothermal(z, T_obs, z_steady_min_m=300.0)
    post = invert_posterior(
        z,
        dT,
        bin_edges_yr=edges,
        sigma_T=0.05,
        n_bootstrap=200,
        lam=0.05,
        rng=rng,
    )
    # Bin-wise: median within 0.4 K of truth for every bin.
    bin_errors = np.abs(post.median - truth)
    assert np.all(bin_errors < 0.4), (
        f"bin medians {post.median.round(3)} vs truth {truth.round(3)} "
        f"(errors {bin_errors.round(3)})"
    )
    # Cumulative: time-integrated warming within 30%.
    durations = np.diff(edges)
    integral_truth = float((truth * durations).sum())
    integral_post = float((post.median * durations).sum())
    rel = abs(integral_post - integral_truth) / abs(integral_truth)
    assert rel < 0.30, (
        f"posterior integrated warming {integral_post:.2f} K-yr deviates "
        f"{rel * 100:.1f}% from truth {integral_truth:.2f} K-yr"
    )


def test_inversion_kappa_prior_marginal() -> None:
    """Marginal kappa samples should reflect the log-normal prior:
    geometric mean within ~10% of exp(log_kappa_mean) over 500 draws."""
    rng = np.random.default_rng(20260522)
    z = np.linspace(20.0, 600.0, 30)
    dT = np.zeros_like(z)
    post = invert_posterior(z, dT, n_bootstrap=500, rng=rng)
    geomean = float(np.exp(np.mean(np.log(post.kappa_samples))))
    assert geomean == pytest.approx(1.0e-6, rel=0.10)
    # And sd of log kappa close to prior sd 0.20.
    assert np.std(np.log(post.kappa_samples)) == pytest.approx(0.20, abs=0.04)


def test_inversion_coverage_for_smooth_truths() -> None:
    """Frequentist coverage of the 90% credible interval for the recent
    bin, when the underlying GST history is smooth, should be at least
    0.6 over 30 independent draws.  Sharp-step truths systematically
    under-cover because of Tikhonov bias toward smoothness; smooth
    truths are the realistic regime where coverage is meaningful."""
    edges = default_bin_edges_yr()
    rng = np.random.default_rng(20260522)
    hits = 0
    n_seeds = 30
    for seed in range(n_seeds):
        srng = np.random.default_rng(seed)
        # Smooth truth: monotone decreasing amplitudes back in time, with
        # a random recent amplitude.
        recent_truth = float(srng.uniform(0.2, 2.0))
        decay = srng.uniform(0.4, 0.7)
        truth = recent_truth * decay ** np.arange(6, dtype=float)
        z, T_obs, _ = _make_synthetic_profile(
            bin_amplitudes_K=truth,
            bin_edges_yr=edges,
            rng=srng,
        )
        dT, *_ = detrend_geothermal(z, T_obs, z_steady_min_m=300.0)
        post = invert_posterior(
            z,
            dT,
            bin_edges_yr=edges,
            sigma_T=0.05,
            n_bootstrap=120,
            lam=0.05,
            rng=rng,
        )
        if post.ci_lo[0] <= recent_truth <= post.ci_hi[0]:
            hits += 1
    coverage = hits / n_seeds
    assert coverage >= 0.60, f"recent-bin 90% CI coverage = {coverage:.2f}"
