"""Hierarchical Bayesian errors-in-variables (EIV) Gibbs sampler.

Closes the R17 W3 item from the editorial verdict: replace the
unstable Deming regression (1.70 → 6.1 → 46 depending on assumed
variance ratio) and the biased OLS fallback (β = 1.70) with a
proper Bayesian errors-in-variables model that gives a stable
posterior on the SAT-GST coupling slope.

Model:

    X_i = x_i* + ε_X,    ε_X ~ N(0, σ_X²)
    Y_i = α + β_{g(i)} x_i* + ε_Y,    ε_Y ~ N(0, σ_Y²)

with ``g(i)`` the latitude-band index (boreal / mid / tropical),
``β_g ~ N(β_global, τ²)`` a hierarchical-pooling prior, and
``β_global`` itself given a weakly-informative normal prior.

The Gibbs sampler cycles through four conditional updates per
iteration (Carroll, Ruppert & Stefanski 2006, §4.7):

1.  Latent ``x_i*`` per site, normal posterior given (α, β_g, σ_X, σ_Y).
2.  Intercept ``α``, normal posterior given (x_i*, β_g).
3.  Band slopes ``β_g``, normal posterior pooled toward ``β_global``.
4.  Hyperparameters ``β_global, τ²``, normal / inverse-gamma posterior.

All observation noises ``σ_X, σ_Y`` are treated as **known**
(supplied by the caller); their default values are calibrated to
the F2 use case (CRU TS observation noise + Huang-Pollack
inversion posterior width). A full sampler over ``σ_X, σ_Y`` is
straightforward to add but not needed for the current iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EIVPosterior:
    """Posterior samples from the hierarchical EIV Gibbs sampler."""

    alpha: np.ndarray  # (n_draws,)
    beta_global: np.ndarray  # (n_draws,)
    beta_band: np.ndarray  # (n_draws, n_bands)
    tau: np.ndarray  # (n_draws,)
    band_labels: tuple[str, ...]

    def summary(self) -> dict[str, dict[str, float]]:
        """Posterior median + 5--95 % CI for each scalar."""

        def s(arr: np.ndarray) -> dict[str, float]:
            return {
                "median": float(np.median(arr)),
                "ci_lo": float(np.quantile(arr, 0.05)),
                "ci_hi": float(np.quantile(arr, 0.95)),
            }

        out = {
            "alpha": s(self.alpha),
            "beta_global": s(self.beta_global),
            "tau": s(self.tau),
        }
        for k, label in enumerate(self.band_labels):
            out[f"beta_{label}"] = s(self.beta_band[:, k])
        return out


def gibbs_eiv(
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    band_index: np.ndarray,
    n_bands: int,
    sigma_x: float = 0.5,
    sigma_y: float = 0.5,
    n_draws: int = 4000,
    n_burn: int = 1000,
    beta_global_prior_mean: float = 1.0,
    beta_global_prior_sd: float = 1.0,
    tau_prior_shape: float = 2.0,
    tau_prior_scale: float = 0.5,
    seed: int = 20260523,
    band_labels: tuple[str, ...] | None = None,
) -> EIVPosterior:
    """Run the hierarchical EIV Gibbs sampler.

    Parameters
    ----------
    x_obs, y_obs
        Same-length 1-D arrays of observed ΔSAT and ΔGST.
    band_index
        Integer array (same length as x_obs) with values in
        ``[0, n_bands)`` -- the latitude-band index of each site.
    n_bands
        Number of latitude bands (typically 3: tropical / mid / boreal).
    sigma_x, sigma_y
        Known observation noise on ΔSAT and ΔGST (K). Defaults
        calibrated to F2.
    n_draws, n_burn
        Total MCMC iterations and burn-in to discard.
    beta_global_prior_mean, beta_global_prior_sd
        Normal prior on the global slope.
    tau_prior_shape, tau_prior_scale
        Inverse-gamma prior on the between-band slope variance.
    seed
        RNG seed.
    band_labels
        Optional band names for the posterior summary.

    Returns
    -------
    EIVPosterior
        Posterior samples after burn-in.
    """
    rng = np.random.default_rng(seed)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    band_index = np.asarray(band_index, dtype=int)
    if not (x_obs.shape == y_obs.shape == band_index.shape):
        raise ValueError("x_obs, y_obs, band_index must have the same shape")
    n = x_obs.size

    # Initialise from OLS.
    x_bar = x_obs.mean()
    y_bar = y_obs.mean()
    sxx = float(np.sum((x_obs - x_bar) ** 2))
    sxy = float(np.sum((x_obs - x_bar) * (y_obs - y_bar)))
    beta_init = sxy / sxx if sxx > 0 else 1.0
    alpha_init = y_bar - beta_init * x_bar

    alpha = alpha_init
    beta_band = np.full(n_bands, beta_init, dtype=float)
    beta_global = beta_init
    tau = max(0.5, 0.1 * abs(beta_init))
    x_star = x_obs.copy()

    alpha_chain = np.empty(n_draws)
    beta_global_chain = np.empty(n_draws)
    beta_band_chain = np.empty((n_draws, n_bands))
    tau_chain = np.empty(n_draws)

    inv_sx2 = 1.0 / (sigma_x * sigma_x)
    inv_sy2 = 1.0 / (sigma_y * sigma_y)

    for k in range(n_draws):
        # 1. Sample x_i* | rest.
        beta_per_site = beta_band[band_index]
        precision = inv_sx2 + (beta_per_site**2) * inv_sy2
        mean = (x_obs * inv_sx2 + beta_per_site * (y_obs - alpha) * inv_sy2) / precision
        std = 1.0 / np.sqrt(precision)
        x_star = rng.normal(mean, std)

        # 2. Sample alpha | rest. Flat prior; normal posterior.
        residual = y_obs - beta_per_site * x_star
        alpha_post_mean = residual.mean()
        alpha_post_sd = sigma_y / np.sqrt(n)
        alpha = rng.normal(alpha_post_mean, alpha_post_sd)

        # 3. Sample beta_g | rest. Posterior pools data + N(β_global, τ²).
        for g in range(n_bands):
            mask = band_index == g
            x_g = x_star[mask]
            y_g = y_obs[mask] - alpha
            n_g = x_g.size
            if n_g == 0:
                beta_band[g] = rng.normal(beta_global, tau)
                continue
            data_precision = (x_g * x_g).sum() * inv_sy2
            data_term = (x_g * y_g).sum() * inv_sy2
            prior_precision = 1.0 / (tau * tau)
            prior_term = beta_global * prior_precision
            post_precision = data_precision + prior_precision
            post_mean = (data_term + prior_term) / post_precision
            post_sd = 1.0 / np.sqrt(post_precision)
            beta_band[g] = rng.normal(post_mean, post_sd)

        # 4a. Sample beta_global | rest.
        bg_prior_prec = 1.0 / (beta_global_prior_sd * beta_global_prior_sd)
        bg_data_prec = n_bands / (tau * tau)
        bg_post_prec = bg_prior_prec + bg_data_prec
        bg_post_mean = (
            beta_global_prior_mean * bg_prior_prec + beta_band.mean() * bg_data_prec
        ) / bg_post_prec
        bg_post_sd = 1.0 / np.sqrt(bg_post_prec)
        beta_global = rng.normal(bg_post_mean, bg_post_sd)

        # 4b. Sample tau² ~ inverse-gamma | beta_band, beta_global.
        ss = float(np.sum((beta_band - beta_global) ** 2))
        post_shape = tau_prior_shape + 0.5 * n_bands
        post_scale = tau_prior_scale + 0.5 * ss
        # numpy.random.gamma is shape, scale; invert for inverse-gamma.
        tau2 = 1.0 / rng.gamma(post_shape, 1.0 / post_scale)
        tau = float(np.sqrt(max(tau2, 1.0e-12)))

        alpha_chain[k] = alpha
        beta_global_chain[k] = beta_global
        beta_band_chain[k] = beta_band
        tau_chain[k] = tau

    sl = slice(n_burn, None)
    if band_labels is None:
        band_labels = tuple(f"band_{g}" for g in range(n_bands))
    elif len(band_labels) != n_bands:
        raise ValueError("band_labels length must equal n_bands")
    return EIVPosterior(
        alpha=alpha_chain[sl],
        beta_global=beta_global_chain[sl],
        beta_band=beta_band_chain[sl],
        tau=tau_chain[sl],
        band_labels=band_labels,
    )


def latitude_band_index(
    lats_deg: np.ndarray,
    *,
    tropical_max: float = 30.0,
    boreal_min: float = 60.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map |latitude| to a 3-class band index for the hierarchical EIV.

    Returns ``(band_index, band_labels)`` where ``band_index`` is
    0 = tropical, 1 = mid, 2 = boreal.
    """
    abs_lat = np.abs(np.asarray(lats_deg, dtype=float))
    bands = np.full(abs_lat.shape, 1, dtype=int)
    bands[abs_lat < tropical_max] = 0
    bands[abs_lat >= boreal_min] = 2
    return bands, ("tropical", "mid", "boreal")


__all__ = ["EIVPosterior", "gibbs_eiv", "latitude_band_index"]
