"""Joint Bayesian inversion of (T_profile, p_series) for the
ΓN_α coupling parameter plus the reduced talik property vector.

The forward operator is :func:`gt_theory.solvers.run_column_coupled`
(the block-banded 1-D coupled Crank-Nicolson). Each posterior draw
runs the solver once, so we cannot afford the ``invert_posterior``
bootstrap pattern. Instead we use an adaptive Metropolis-Hastings
ensemble: ``n_walkers`` independent chains, Gaussian random-walk
proposals with covariance Σ adapted from the running covariance of
each chain's recent samples, scaled by ``2.38**2 / d`` (the
Roberts-Rosenthal optimal-acceptance heuristic for d-dim Gaussian
targets).

Reduced parameter vector (default, 5 scalars):

============= ============================== ==================================
Name          Description                    Prior
============= ============================== ==================================
log10_K_hyd   talik hydraulic conductivity   uniform(-7, -4)   m/s
porosity      talik porosity                 uniform( 0.30, 0.50)
lambda_th     talik thermal conductivity     uniform( 1.20, 2.50) W/m/K
gst_offset    additive offset on GST forcing normal(0, 0.5)    K
gamma_scale   ΓN_α coupling knob             uniform( 0, 2)
============= ============================== ==================================

The observation operator is linear interpolation of the simulated
T(t, z) onto a 2-D observation grid ``(t_obs, z_obs)`` for the
thermistor stack and onto a 1-D ``(t_obs,)`` series at ``z_piezo``
for the piezometer. Likelihood is Gaussian with separate σ_T and
σ_p; both treated as known hyperparameters (informed by the
Nordicana D documentation and instrument-noise tables).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gt_theory.solvers import run_column_coupled

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoupledPosteriorResult:
    """Adaptive-MH ensemble posterior for the coupled T+p inversion.

    Attributes
    ----------
    param_names : tuple of str
        Names of the reduced parameters in column order.
    chains : ndarray, shape (n_walkers, n_steps, n_params)
        Post-burn-in samples for each walker.
    log_post : ndarray, shape (n_walkers, n_steps)
        Per-sample log posterior.
    accept_rate : ndarray, shape (n_walkers,)
        Per-walker post-burn-in acceptance rate.
    truth : ndarray or None, shape (n_params,)
        Ground-truth parameter vector (set by synthetic-twin tests; None
        for real-data runs).
    """

    param_names: tuple[str, ...]
    chains: np.ndarray
    log_post: np.ndarray
    accept_rate: np.ndarray
    truth: np.ndarray | None = None

    @property
    def flat_samples(self) -> np.ndarray:
        return self.chains.reshape(-1, self.chains.shape[-1])

    def credible_interval(
        self, parameter: str | int, *, level: float = 0.90
    ) -> tuple[float, float]:
        """Two-sided credible interval at the requested level."""
        idx = parameter if isinstance(parameter, int) else self.param_names.index(parameter)
        samples = self.flat_samples[:, idx]
        lo_q = 0.5 - 0.5 * level
        hi_q = 0.5 + 0.5 * level
        return float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))

    def median(self, parameter: str | int) -> float:
        idx = parameter if isinstance(parameter, int) else self.param_names.index(parameter)
        return float(np.median(self.flat_samples[:, idx]))


# ---------------------------------------------------------------------------
# Forward operator wrapper
# ---------------------------------------------------------------------------


def coupled_forward(
    theta: np.ndarray,
    *,
    depth_max_m: float,
    dz_m: float,
    duration_s: float,
    dt_s: float,
    sat: np.ndarray,
    t_obs_s: np.ndarray,
    z_obs_m: np.ndarray,
    z_piezo_m: float,
    rho_w: float = 1000.0,
    mu: float = 1.3e-3,
    g: float = 9.81,
    c_w: float = 4186.0,
    alpha_w: float = 2.1e-4,
    beta_w: float = 4.6e-10,
    rho_c_eff: float = 2.5e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run :func:`run_column_coupled` and interpolate the result onto
    the observation grid.

    Parameters
    ----------
    theta
        Reduced parameter vector
        ``(log10_K_hyd, porosity, lambda_th, gst_offset, gamma_scale)``.
    depth_max_m, dz_m, duration_s, dt_s
        Solver grid + time step (held fixed across the chain).
    sat
        Surface-air-T series of length ``nt``.
    t_obs_s, z_obs_m
        Observation time grid (s) and thermistor depths (m).
    z_piezo_m
        Single piezometer screen depth (m).

    Returns
    -------
    T_pred : ndarray, shape (len(t_obs_s), len(z_obs_m))
        Predicted thermistor stack on the observation grid.
    p_pred : ndarray, shape (len(t_obs_s),)
        Predicted piezometric pressure at ``z_piezo_m`` on the
        observation time grid.
    """
    log10_K, phi, lam, dT, scale = (float(x) for x in theta)
    K_hyd = 10.0**log10_K
    k_intrinsic = K_hyd * mu / (rho_w * g)

    res = run_column_coupled(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        lambda_thermal=lam,
        rho_c_eff=rho_c_eff,
        K_zz=k_intrinsic,
        mu=mu,
        porosity=phi,
        alpha_w=alpha_w,
        beta_w=beta_w,
        rho_w=rho_w,
        g=g,
        c_w=c_w,
        gamma_n_alpha_scale=scale,
        sat=sat + dT,
        p_top=0.0,
    )

    # Bilinear interpolation T(t, z): first interp along t for each z, then
    # along z.  Both axes are uniform in the solver output.
    T_at_t = np.empty((t_obs_s.size, res.z.size))
    for j in range(res.z.size):
        T_at_t[:, j] = np.interp(t_obs_s, res.t, res.T[:, j])
    T_pred = np.empty((t_obs_s.size, z_obs_m.size))
    for i in range(t_obs_s.size):
        T_pred[i] = np.interp(z_obs_m, res.z, T_at_t[i])

    p_at_t = np.empty(res.z.size)
    p_pred = np.empty(t_obs_s.size)
    for j in range(res.z.size):
        p_at_t[j] = 0.0  # placeholder; overwrite below
    p_at_t_t = np.empty((t_obs_s.size, res.z.size))
    for j in range(res.z.size):
        p_at_t_t[:, j] = np.interp(t_obs_s, res.t, res.p[:, j])
    for i in range(t_obs_s.size):
        p_pred[i] = np.interp(z_piezo_m, res.z, p_at_t_t[i])

    return T_pred, p_pred


# ---------------------------------------------------------------------------
# Priors and posterior
# ---------------------------------------------------------------------------


def _log_uniform(x: float, lo: float, hi: float) -> float:
    return 0.0 if lo <= x <= hi else -np.inf


def _log_normal(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2


def default_log_prior(theta: np.ndarray) -> float:
    log10_K, phi, lam, dT, scale = theta
    return (
        _log_uniform(log10_K, -7.0, -4.0)
        + _log_uniform(phi, 0.30, 0.50)
        + _log_uniform(lam, 1.20, 2.50)
        + _log_normal(dT, 0.0, 0.5)
        + _log_uniform(scale, 0.0, 2.0)
    )


def build_log_posterior(
    *,
    T_obs: np.ndarray,
    p_obs: np.ndarray,
    sigma_T: float,
    sigma_p: float,
    forward_kwargs: dict,
    log_prior: Callable[[np.ndarray], float] = default_log_prior,
) -> Callable[[np.ndarray], float]:
    """Build a closure that evaluates log posterior at ``theta``."""

    def _log_post(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        try:
            T_pred, p_pred = coupled_forward(theta, **forward_kwargs)
        except (ValueError, RuntimeError):
            return -np.inf
        ll_T = -0.5 * np.sum(((T_pred - T_obs) / sigma_T) ** 2)
        ll_p = -0.5 * np.sum(((p_pred - p_obs) / sigma_p) ** 2)
        return float(lp + ll_T + ll_p)

    return _log_post


# ---------------------------------------------------------------------------
# Adaptive Metropolis-Hastings ensemble
# ---------------------------------------------------------------------------


def adaptive_mh(
    log_post: Callable[[np.ndarray], float],
    *,
    initial_thetas: np.ndarray,
    n_steps: int,
    n_burn: int,
    proposal_cov: np.ndarray | None = None,
    adapt_after: int = 50,
    adapt_every: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Metropolis-Hastings on ``n_walkers`` independent chains.

    Each walker maintains its own running mean and covariance of the
    last ``adapt_after`` samples and proposes from
    ``N(theta_curr, (2.38**2 / d) * Sigma)``.

    Returns
    -------
    chains : ndarray, shape (n_walkers, n_steps - n_burn, d)
    log_post_chain : ndarray, shape (n_walkers, n_steps - n_burn)
    accept_rate : ndarray, shape (n_walkers,)
    """
    rng = np.random.default_rng() if rng is None else rng
    initial_thetas = np.asarray(initial_thetas, dtype=float)
    if initial_thetas.ndim != 2:
        raise ValueError("initial_thetas must be (n_walkers, d)")
    n_walkers, d = initial_thetas.shape
    if proposal_cov is None:
        proposal_cov = 0.01 * np.eye(d)
    proposal_cov = np.asarray(proposal_cov, dtype=float)
    if proposal_cov.shape != (d, d):
        raise ValueError("proposal_cov must be (d, d)")
    if n_burn >= n_steps:
        raise ValueError("n_burn must be < n_steps")

    factor = (2.38**2) / d
    walker_covs = [factor * proposal_cov.copy() for _ in range(n_walkers)]
    walker_history: list[list[np.ndarray]] = [[] for _ in range(n_walkers)]

    samples = np.empty((n_walkers, n_steps, d))
    log_post_chain = np.empty((n_walkers, n_steps))
    accepts = np.zeros(n_walkers, dtype=int)

    theta_curr = initial_thetas.copy()
    log_post_curr = np.array([log_post(t) for t in theta_curr])

    for step in range(n_steps):
        for w in range(n_walkers):
            prop = theta_curr[w] + rng.multivariate_normal(np.zeros(d), walker_covs[w])
            lp_prop = log_post(prop)
            log_alpha = lp_prop - log_post_curr[w]
            if np.log(rng.uniform()) < log_alpha:
                theta_curr[w] = prop
                log_post_curr[w] = lp_prop
                accepts[w] += 1
            samples[w, step] = theta_curr[w]
            log_post_chain[w, step] = log_post_curr[w]
            walker_history[w].append(theta_curr[w].copy())

            if (
                step >= adapt_after
                and step % adapt_every == 0
                and len(walker_history[w]) >= 2 * d + 2
            ):
                hist = np.asarray(walker_history[w][-adapt_after:])
                emp_cov = np.cov(hist.T)
                # blend a small ridge to keep positive-definite
                walker_covs[w] = factor * (emp_cov + 1.0e-6 * np.eye(d))

    chains = samples[:, n_burn:, :]
    log_post_chain = log_post_chain[:, n_burn:]
    accept_rate = accepts / float(n_steps)
    return chains, log_post_chain, accept_rate


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


PARAM_NAMES_DEFAULT: tuple[str, ...] = (
    "log10_K_hyd_m_s",
    "porosity",
    "lambda_th_W_m_K",
    "gst_offset_K",
    "gamma_n_alpha_scale",
)


def invert_coupled_posterior(
    *,
    T_obs: np.ndarray,
    p_obs: np.ndarray,
    sigma_T: float,
    sigma_p: float,
    forward_kwargs: dict,
    n_walkers: int = 8,
    n_steps: int = 400,
    n_burn: int = 100,
    initial_thetas: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    truth: np.ndarray | None = None,
    param_names: tuple[str, ...] = PARAM_NAMES_DEFAULT,
) -> CoupledPosteriorResult:
    """Joint T+p adaptive-MH inversion.

    Parameters
    ----------
    T_obs : ndarray, shape (n_t_obs, n_z_obs)
        Observed thermistor stack on the same grid as
        ``forward_kwargs["t_obs_s"]`` x ``forward_kwargs["z_obs_m"]``.
    p_obs : ndarray, shape (n_t_obs,)
        Observed piezometric pressure at ``forward_kwargs["z_piezo_m"]``.
    sigma_T, sigma_p
        Observation noise standard deviations (K, Pa).
    forward_kwargs
        Dict of solver/grid parameters passed verbatim to
        :func:`coupled_forward`.
    n_walkers, n_steps, n_burn
        Adaptive-MH ensemble size and chain length.
    initial_thetas
        Optional ``(n_walkers, d)`` starting points; if ``None``, draws
        independent random points from the prior support.
    rng
        Optional numpy Generator.
    truth
        Optional ground-truth vector recorded in the result for the
        synthetic-twin test.
    """
    rng = np.random.default_rng() if rng is None else rng
    log_post = build_log_posterior(
        T_obs=T_obs,
        p_obs=p_obs,
        sigma_T=sigma_T,
        sigma_p=sigma_p,
        forward_kwargs=forward_kwargs,
    )

    if initial_thetas is None:
        # Random draws from broad-prior support; we sample uniformly for the
        # uniform-prior parameters and from the normal for the gst_offset.
        init = np.empty((n_walkers, 5))
        init[:, 0] = rng.uniform(-6.5, -4.5, size=n_walkers)
        init[:, 1] = rng.uniform(0.32, 0.48, size=n_walkers)
        init[:, 2] = rng.uniform(1.30, 2.30, size=n_walkers)
        init[:, 3] = rng.normal(0.0, 0.2, size=n_walkers)
        init[:, 4] = rng.uniform(0.2, 1.8, size=n_walkers)
        initial_thetas = init

    chains, log_post_chain, accept_rate = adaptive_mh(
        log_post,
        initial_thetas=initial_thetas,
        n_steps=n_steps,
        n_burn=n_burn,
        rng=rng,
    )

    return CoupledPosteriorResult(
        param_names=param_names,
        chains=chains,
        log_post=log_post_chain,
        accept_rate=accept_rate,
        truth=truth,
    )
