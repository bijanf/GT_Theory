"""Hierarchical bootstrap Tikhonov inversion of borehole temperature
profiles for ground-surface-temperature (GST) histories.

Theory
------
On a half-space with zero initial anomaly, a step change of `dT` in
surface temperature from time `a` to time `b` (years before present)
produces a present-day depth-profile anomaly

    Delta T_step(z; a, b, kappa)
        = dT * [ erfc( z / (2*sqrt(kappa*a)) )
                - erfc( z / (2*sqrt(kappa*b)) ) ].

Discretising the GST history into `n_bins` rectangular pieces with
amplitudes `s = (s_1, ..., s_n)` over edges `t_0 < t_1 < ... < t_n` in
years before present (oldest = t_n), the present-day anomaly is the
linear combination

    Delta T(z_i) = sum_k G_ik * s_k,
    G_ik = erfc( z_i / (2 sqrt(kappa t_{k-1})) ) - erfc( z_i / (2 sqrt(kappa t_k)) ).

The forward operator depends on `kappa`. We treat `kappa` as a random
parameter drawn from a log-normal prior, and reconstruct `s` for each
draw by Tikhonov-regularised least squares:

    s_hat = argmin || G s - dT_obs ||^2 + lam^2 || L s ||^2

where `L` is the first-difference operator (penalising bin-to-bin
roughness). Repeating this with bootstrap perturbations of the
observation by Gaussian noise of standard deviation sigma_T yields a
joint posterior over (kappa, GST history). Percentile-based credible
intervals follow.

Conventions
-----------
* `z`        — depths in metres, positive downward.
* `t` / `a,b`/ bin edges — years before present, all positive.
* `kappa`    — thermal diffusivity in m^2 s^-1.
* `s` / GST  — anomaly in K relative to the pre-anomaly mean GST.

Reference
---------
Mareschal & Beltrami (1992); Pollack & Huang (2000); Beltrami et al. (2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq
from scipy.special import erfc

YEAR_S: float = 365.25 * 86400.0

DEFAULT_SIGMA_T: float = 0.05
DEFAULT_LOG_KAPPA_MEAN: float = float(np.log(1.0e-6))
DEFAULT_LOG_KAPPA_SD: float = 0.20
DEFAULT_N_BOOTSTRAP: int = 500


def default_bin_edges_yr() -> np.ndarray:
    """A pragmatic GST-history binning: 0, 25, 50, 100, 200, 400, 800 yr
    before present.  Logarithmically coarsens away from the surface in
    line with the diffusion kernel's depth resolution.
    """
    return np.array([0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0])


@dataclass(frozen=True)
class PosteriorResult:
    """Bootstrap posterior summary.

    Attributes
    ----------
    bin_edges_yr : ndarray, shape (n_bins+1,)
        Bin edges in years before present, sorted ascending.
    samples : ndarray, shape (n_bootstrap, n_bins)
        Per-draw recovered GST amplitudes (K).
    median : ndarray, shape (n_bins,)
        Posterior median GST per bin (K).
    ci_lo : ndarray, shape (n_bins,)
        5th-percentile credible bound (K).
    ci_hi : ndarray, shape (n_bins,)
        95th-percentile credible bound (K).
    kappa_samples : ndarray, shape (n_bootstrap,)
        Drawn thermal diffusivities (m^2 s^-1).
    residual_rms : ndarray, shape (n_bootstrap,)
        Per-draw RMS of the observation-fit residual (K).
    """

    bin_edges_yr: np.ndarray
    samples: np.ndarray
    median: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    kappa_samples: np.ndarray
    residual_rms: np.ndarray

    @property
    def bin_centers_yr(self) -> np.ndarray:
        edges = self.bin_edges_yr
        return 0.5 * (edges[:-1] + edges[1:])


def build_forward_operator(
    z: np.ndarray,
    bin_edges_yr: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Build the (nz, n_bins) forward operator G mapping bin amplitudes
    to present-day depth-profile anomaly.

    Implements the analytic step-response in a half-space (zero IC).
    """
    z = np.atleast_1d(z).astype(float)
    edges = np.asarray(bin_edges_yr, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges_yr must be 1-D with at least 2 entries.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges_yr must be strictly increasing.")
    if edges[0] < 0:
        raise ValueError("bin_edges_yr must be non-negative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    # `a_s` is the younger edge of each bin (smaller years-before-present);
    # `b_s` is the older edge.  Decomposing a unit rectangular pulse into
    # "step on at b_s, persisting to present" MINUS "step on at a_s,
    # persisting to present" gives the present-day contribution
    #     erfc(z / (2 sqrt(kappa * b_s))) - erfc(z / (2 sqrt(kappa * a_s))).
    # Because b_s > a_s, the first term is larger, so the contribution is
    # non-negative (for a +1 K pulse).
    a_s = edges[:-1] * YEAR_S
    b_s = edges[1:] * YEAR_S
    # Floor a_s to avoid division by zero when the youngest bin starts at
    # t = 0 (today): erfc at infinite argument is 0, which is the correct
    # limit (no time for the diffusive front to begin).
    a_s_safe = np.maximum(a_s, 1.0e-3)

    arg_a = z[:, None] / (2.0 * np.sqrt(kappa * a_s_safe[None, :]))
    arg_b = z[:, None] / (2.0 * np.sqrt(kappa * b_s[None, :]))
    return erfc(arg_b) - erfc(arg_a)


def detrend_geothermal(
    z: np.ndarray,
    T_obs: np.ndarray,
    *,
    z_steady_min_m: float = 300.0,
) -> tuple[np.ndarray, float, float]:
    """Subtract the steady-state geothermal gradient.

    Fits T_steady(z) = T0 + dTdz * z to the deep portion z >= z_steady_min_m
    (assumed transient-free) and returns the residual anomaly.

    Parameters
    ----------
    z, T_obs
        Borehole depth (m) and temperature (K) profiles.
    z_steady_min_m
        Minimum depth (m) for the steady-state fit.

    Returns
    -------
    dT, T0, dTdz
        Anomaly profile (K), intercept (K), and gradient (K m^-1).
    """
    z = np.asarray(z, dtype=float)
    T = np.asarray(T_obs, dtype=float)
    deep = z >= z_steady_min_m
    if deep.sum() < 3:
        raise ValueError(
            f"Need >= 3 samples at z >= {z_steady_min_m} m for steady-state fit; "
            f"got {int(deep.sum())}."
        )
    A = np.column_stack([np.ones(deep.sum()), z[deep]])
    coef, *_ = lstsq(A, T[deep])
    T0 = float(coef[0])
    dTdz = float(coef[1])
    return T - (T0 + dTdz * z), T0, dTdz


def _first_difference_matrix(n: int) -> np.ndarray:
    L = np.zeros((n - 1, n))
    for i in range(n - 1):
        L[i, i] = -1.0
        L[i, i + 1] = +1.0
    return L


def invert_posterior(
    z: np.ndarray,
    dT_obs: np.ndarray,
    *,
    bin_edges_yr: np.ndarray | None = None,
    sigma_T: float = DEFAULT_SIGMA_T,
    log_kappa_mean: float = DEFAULT_LOG_KAPPA_MEAN,
    log_kappa_sd: float = DEFAULT_LOG_KAPPA_SD,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    lam: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PosteriorResult:
    """Hierarchical bootstrap Tikhonov inversion of a single profile.

    Parameters
    ----------
    z, dT_obs
        Depths (m) and detrended temperature anomalies (K).  Use
        :func:`detrend_geothermal` to obtain `dT_obs` from a raw log.
    bin_edges_yr
        GST history bin edges in years before present.  Default:
        :func:`default_bin_edges_yr`.
    sigma_T
        Standard deviation of the assumed measurement noise (K).
    log_kappa_mean, log_kappa_sd
        Hyperparameters of the log-normal prior on `kappa`.
    n_bootstrap
        Number of bootstrap draws.
    lam
        Tikhonov regularisation strength on the first-difference
        smoothness operator.
    rng
        numpy Generator for reproducibility.  Default: `np.random.default_rng()`.

    Returns
    -------
    PosteriorResult
    """
    z = np.asarray(z, dtype=float)
    dT_obs = np.asarray(dT_obs, dtype=float)
    if z.shape != dT_obs.shape:
        raise ValueError(f"z and dT_obs shape mismatch: {z.shape} vs {dT_obs.shape}.")
    if sigma_T <= 0:
        raise ValueError("sigma_T must be positive.")
    if lam < 0:
        raise ValueError("lam must be non-negative.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    edges = default_bin_edges_yr() if bin_edges_yr is None else np.asarray(bin_edges_yr, float)
    n_bins = edges.size - 1
    L = _first_difference_matrix(n_bins)

    rng = np.random.default_rng() if rng is None else rng

    samples = np.empty((n_bootstrap, n_bins))
    kappa_samples = np.empty(n_bootstrap)
    residual_rms = np.empty(n_bootstrap)

    nz = z.size
    rhs_pad = np.zeros(n_bins - 1)  # zero target for the regularisation rows

    for b in range(n_bootstrap):
        kappa_b = float(np.exp(rng.normal(log_kappa_mean, log_kappa_sd)))
        G_b = build_forward_operator(z, edges, kappa_b)

        eps_b = rng.normal(0.0, sigma_T, size=nz)
        y_b = dT_obs + eps_b

        # Stacked least-squares: [G; lam*L] s = [y; 0]
        A_b = np.vstack([G_b, lam * L])
        rhs_b = np.concatenate([y_b, rhs_pad])
        s_b, *_ = lstsq(A_b, rhs_b, lapack_driver="gelsd")

        kappa_samples[b] = kappa_b
        samples[b] = s_b
        residual_rms[b] = float(np.sqrt(np.mean((G_b @ s_b - y_b) ** 2)))

    median = np.median(samples, axis=0)
    ci_lo = np.percentile(samples, 5.0, axis=0)
    ci_hi = np.percentile(samples, 95.0, axis=0)

    return PosteriorResult(
        bin_edges_yr=edges,
        samples=samples,
        median=median,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        kappa_samples=kappa_samples,
        residual_rms=residual_rms,
    )
