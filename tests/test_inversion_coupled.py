"""Synthetic-twin tests for the coupled T+p Bayesian inverter.

Generate observations from a known parameter vector via the same
forward operator, add Gaussian noise, then verify the adaptive
Metropolis-Hastings ensemble recovers the truth within a credible
interval on at least the most identifiable parameter
(``gamma_n_alpha_scale``, the headline ΓN_α knob).

Budget: tiny column + short window + few walkers/steps so the test
completes in well under a minute on a laptop. Production runs use
~250-walker × 200-step budgets driven from ``scripts/`` (TBD in
P2b).
"""

from __future__ import annotations

import numpy as np

from gt_theory.inversion import (
    CoupledPosteriorResult,
    coupled_forward,
    invert_coupled_posterior,
)

YEAR_S = 365.25 * 86400.0


def _build_forward_kwargs(
    *,
    sat: np.ndarray,
    t_obs_s: np.ndarray,
    z_obs_m: np.ndarray,
    z_piezo_m: float,
    duration_s: float,
    dt_s: float,
) -> dict:
    return {
        "depth_max_m": 2.0,
        "dz_m": 0.5,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "sat": sat,
        "t_obs_s": t_obs_s,
        "z_obs_m": z_obs_m,
        "z_piezo_m": z_piezo_m,
    }


def _synthetic_observations(
    *,
    truth: np.ndarray,
    forward_kwargs: dict,
    sigma_T: float,
    sigma_p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    T_clean, p_clean = coupled_forward(truth, **forward_kwargs)
    T_obs = T_clean + rng.normal(0.0, sigma_T, size=T_clean.shape)
    p_obs = p_clean + rng.normal(0.0, sigma_p, size=p_clean.shape)
    return T_obs, p_obs


def test_coupled_forward_returns_expected_shapes() -> None:
    duration_s = 30.0 * 86400.0
    dt_s = 5.0 * 86400.0
    nt = int(round(duration_s / dt_s)) + 1
    sat = -3.0 + 5.0 * np.sin(2.0 * np.pi * np.arange(nt) / nt)
    t_obs_s = np.linspace(0.0, duration_s, 4)
    z_obs_m = np.array([0.5, 1.0, 1.5])

    truth = np.array([-5.5, 0.40, 1.80, 0.0, 1.0])
    forward_kwargs = _build_forward_kwargs(
        sat=sat,
        t_obs_s=t_obs_s,
        z_obs_m=z_obs_m,
        z_piezo_m=1.0,
        duration_s=duration_s,
        dt_s=dt_s,
    )

    T_pred, p_pred = coupled_forward(truth, **forward_kwargs)
    assert T_pred.shape == (t_obs_s.size, z_obs_m.size)
    assert p_pred.shape == (t_obs_s.size,)
    assert np.all(np.isfinite(T_pred))
    assert np.all(np.isfinite(p_pred))


def test_synthetic_twin_recovers_gamma_scale_within_90pct_ci() -> None:
    """The headline check: a synthetic-twin observation generated with
    ``gamma_n_alpha_scale = 1.0`` should produce a 90% CI that brackets
    the truth.
    """
    rng = np.random.default_rng(2026_05_22)

    duration_s = 90.0 * 86400.0
    dt_s = 5.0 * 86400.0
    nt = int(round(duration_s / dt_s)) + 1
    t_solver = np.arange(nt) * dt_s
    # GST: large seasonal swing so the column actually feels the forcing
    sat = -3.0 + 10.0 * np.sin(2.0 * np.pi * t_solver / YEAR_S)

    t_obs_s = np.linspace(0.0, duration_s, 6)
    z_obs_m = np.array([0.5, 1.0, 1.5])
    forward_kwargs = _build_forward_kwargs(
        sat=sat,
        t_obs_s=t_obs_s,
        z_obs_m=z_obs_m,
        z_piezo_m=1.0,
        duration_s=duration_s,
        dt_s=dt_s,
    )

    truth = np.array([-5.0, 0.40, 1.80, 0.0, 1.0])  # log10_K, phi, lam, dT, scale
    sigma_T = 0.05
    sigma_p = 50.0  # Pa
    T_obs, p_obs = _synthetic_observations(
        truth=truth,
        forward_kwargs=forward_kwargs,
        sigma_T=sigma_T,
        sigma_p=sigma_p,
        rng=rng,
    )

    # Initial points clustered (within prior) but NOT at truth, so the
    # chain has to actually move.
    n_walkers = 6
    init = np.empty((n_walkers, 5))
    init[:, 0] = rng.uniform(-6.0, -4.5, size=n_walkers)
    init[:, 1] = rng.uniform(0.34, 0.46, size=n_walkers)
    init[:, 2] = rng.uniform(1.50, 2.10, size=n_walkers)
    init[:, 3] = rng.normal(0.0, 0.2, size=n_walkers)
    init[:, 4] = rng.uniform(0.4, 1.6, size=n_walkers)

    result: CoupledPosteriorResult = invert_coupled_posterior(
        T_obs=T_obs,
        p_obs=p_obs,
        sigma_T=sigma_T,
        sigma_p=sigma_p,
        forward_kwargs=forward_kwargs,
        n_walkers=n_walkers,
        n_steps=200,
        n_burn=80,
        initial_thetas=init,
        rng=rng,
        truth=truth,
    )

    assert result.chains.shape == (n_walkers, 120, 5)
    assert np.all(result.accept_rate > 0.0)
    assert np.any(np.isfinite(result.log_post))

    # 90% CI on the headline parameter must bracket the truth.
    lo, hi = result.credible_interval("gamma_n_alpha_scale", level=0.90)
    assert lo <= truth[4] <= hi, (
        f"90% CI on gamma_n_alpha_scale [{lo:.2f}, {hi:.2f}] does not bracket truth {truth[4]:.2f}"
    )


def test_log_posterior_returns_negative_inf_outside_prior() -> None:
    """A sample at log10_K = -10 (outside the prior box) should give
    -inf, signalling immediate rejection by the MH step."""
    from gt_theory.inversion import build_log_posterior

    duration_s = 30.0 * 86400.0
    dt_s = 5.0 * 86400.0
    sat = np.zeros(int(round(duration_s / dt_s)) + 1)
    t_obs_s = np.array([0.0, duration_s])
    z_obs_m = np.array([1.0])
    forward_kwargs = _build_forward_kwargs(
        sat=sat,
        t_obs_s=t_obs_s,
        z_obs_m=z_obs_m,
        z_piezo_m=1.0,
        duration_s=duration_s,
        dt_s=dt_s,
    )
    T_obs = np.zeros((t_obs_s.size, z_obs_m.size))
    p_obs = np.zeros(t_obs_s.size)
    lp = build_log_posterior(
        T_obs=T_obs,
        p_obs=p_obs,
        sigma_T=0.1,
        sigma_p=100.0,
        forward_kwargs=forward_kwargs,
    )

    out_of_prior = np.array([-10.0, 0.40, 1.80, 0.0, 1.0])
    assert lp(out_of_prior) == -np.inf
