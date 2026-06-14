"""Bootstrap confidence intervals for the empirical-validation fingerprints.

Two estimators relevant to the manuscript's claims:

* :func:`bca_ci` — Efron's Bias-Corrected and Accelerated (BCa) interval.
  Corrects the percentile bootstrap for both bias (the observed statistic
  sits at the wrong quantile of the bootstrap distribution) and skewness
  (the rate of change of the statistic's standard error with the
  parameter, estimated via jackknife).  Replaces the naive percentile
  bootstrap currently used in F1, F3 and F4.

* :func:`spatial_block_ci` — block bootstrap that resamples spatial
  bins rather than individual sites.  The Huang--Pollack archive is
  heavily clustered (Canada + Eurasia + USA), so a site-level
  resample treats spatially correlated boreholes as independent
  degrees of freedom and artificially narrows confidence intervals.
  Resampling 5° × 5° lat--lon blocks preserves the spatial
  correlation length of climate forcings and inflates the CI to a
  defensible width.

References
----------
Efron & Tibshirani, *An Introduction to the Bootstrap* (1993), §14.3.
Davison & Hinkley, *Bootstrap Methods and their Application* (1997).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BootstrapResult:
    """Container for a bootstrap confidence interval.

    Attributes
    ----------
    point : float
        The observed statistic on the original sample.
    lo, hi : float
        Lower and upper interval endpoints at ``level``.
    level : float
        Coverage of the interval (e.g. 0.90 for a 90% CI).
    n_bootstrap : int
        Number of bootstrap draws used.
    method : str
        Short tag identifying the method ("percentile", "bca",
        "spatial_block").
    """

    point: float
    lo: float
    hi: float
    level: float
    n_bootstrap: int
    method: str


def naive_percentile_ci(
    sample: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    level: float = 0.90,
    seed: int = 20260522,
) -> BootstrapResult:
    """Plain percentile bootstrap — included only as a reference point
    against which :func:`bca_ci` and :func:`spatial_block_ci` should be
    compared.  Use one of those for any reported manuscript number."""
    sample = np.asarray(sample)
    rng = np.random.default_rng(seed)
    n = sample.shape[0]
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(statistic_fn(sample[idx]))
    alpha = (1.0 - level) / 2.0
    lo = float(np.nanpercentile(boot, 100.0 * alpha))
    hi = float(np.nanpercentile(boot, 100.0 * (1.0 - alpha)))
    point = float(statistic_fn(sample))
    return BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        level=level,
        n_bootstrap=n_bootstrap,
        method="percentile",
    )


def bca_ci(
    sample: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    level: float = 0.90,
    seed: int = 20260522,
) -> BootstrapResult:
    """Bias-Corrected and Accelerated (BCa) bootstrap CI.

    Parameters
    ----------
    sample
        The original sample as a 1-D numpy array (each row is one
        observation).  For multi-column samples, wrap the row index
        in the statistic and pass ``np.arange(n)`` here.
    statistic_fn
        Callable that takes a resampled array of the same shape as
        ``sample`` and returns a scalar.
    n_bootstrap
        Bootstrap draws.
    level
        Coverage of the interval (0.90 = 90% CI).
    seed
        RNG seed.

    Returns
    -------
    :class:`BootstrapResult`
    """
    sample = np.asarray(sample)
    n = sample.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 observations for BCa; got {n}")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(statistic_fn(sample[idx]))

    theta_hat = float(statistic_fn(sample))

    # Bias correction z0: standard normal quantile of the fraction of
    # bootstrap draws below the observed statistic.
    finite = np.isfinite(boot)
    boot_finite = boot[finite]
    if boot_finite.size < 10:
        # Bootstrap distribution is too degenerate; fall back to percentile.
        alpha = (1.0 - level) / 2.0
        lo = float(np.nanpercentile(boot_finite, 100.0 * alpha))
        hi = float(np.nanpercentile(boot_finite, 100.0 * (1.0 - alpha)))
        return BootstrapResult(
            point=theta_hat,
            lo=lo,
            hi=hi,
            level=level,
            n_bootstrap=n_bootstrap,
            method="bca-fallback-percentile",
        )

    p0 = float((boot_finite < theta_hat).sum()) / boot_finite.size
    # Clamp p0 away from {0,1} so norm.ppf is finite.
    p0 = min(max(p0, 1.0 / (4.0 * n_bootstrap)), 1.0 - 1.0 / (4.0 * n_bootstrap))
    z0 = float(norm.ppf(p0))

    # Acceleration a: jackknife on the original sample.
    jackknife = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jackknife[i] = float(statistic_fn(sample[mask]))
    jk_mean = float(np.mean(jackknife))
    num = float(np.sum((jk_mean - jackknife) ** 3))
    den = 6.0 * (float(np.sum((jk_mean - jackknife) ** 2)) ** 1.5)
    a = num / den if den > 0 else 0.0

    alpha = (1.0 - level) / 2.0
    z_lo = norm.ppf(alpha)
    z_hi = norm.ppf(1.0 - alpha)
    adj_lo = norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    adj_hi = norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    # Clamp adjusted quantiles to [eps, 1-eps] for robustness.
    eps = 1.0 / (4.0 * n_bootstrap)
    adj_lo = float(min(max(adj_lo, eps), 1.0 - eps))
    adj_hi = float(min(max(adj_hi, eps), 1.0 - eps))

    lo = float(np.nanpercentile(boot_finite, 100.0 * adj_lo))
    hi = float(np.nanpercentile(boot_finite, 100.0 * adj_hi))
    return BootstrapResult(
        point=theta_hat,
        lo=lo,
        hi=hi,
        level=level,
        n_bootstrap=n_bootstrap,
        method="bca",
    )


def spatial_block_ci(
    sample: np.ndarray,
    lat_deg: Sequence[float],
    lon_deg: Sequence[float],
    statistic_fn: Callable[[np.ndarray], float],
    *,
    block_size_deg: float = 5.0,
    n_bootstrap: int = 2000,
    level: float = 0.90,
    seed: int = 20260522,
) -> BootstrapResult:
    """Spatial block bootstrap that resamples lat--lon bins (size
    ``block_size_deg``) rather than individual sites.

    Parameters
    ----------
    sample
        The original sample.  First axis is the site index; remaining
        axes are passed through unchanged.
    lat_deg, lon_deg
        Latitude / longitude of each site in ``sample``.  Must have
        length equal to ``sample.shape[0]``.
    statistic_fn
        Callable taking a resampled array of variable size (the number
        of sites in the resampled blocks is not constant) and returning
        a scalar.
    block_size_deg
        Width of the lat--lon binning grid (degrees).  Default 5°,
        roughly the synoptic correlation length of monthly CRU SAT.
    n_bootstrap, level, seed
        As for :func:`bca_ci`.
    """
    sample = np.asarray(sample)
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    n = sample.shape[0]
    if lat.size != n or lon.size != n:
        raise ValueError("lat_deg and lon_deg must have length sample.shape[0]")

    # Assign each site to a (lat-bin, lon-bin) cell index.
    lat_bin = np.floor(lat / block_size_deg).astype(int)
    lon_bin = np.floor(lon / block_size_deg).astype(int)
    # Pack (lat_bin, lon_bin) into a single integer key for uniquing.
    cell_ids = lat_bin * 10000 + lon_bin
    unique_cells, inv = np.unique(cell_ids, return_inverse=True)
    n_cells = unique_cells.size

    # Pre-group site indices by cell so resampling is cheap.
    members: list[np.ndarray] = [np.where(inv == k)[0] for k in range(n_cells)]

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        cell_draw = rng.integers(0, n_cells, size=n_cells)
        # Concatenate site indices from the resampled cells.
        idx_pieces = [members[k] for k in cell_draw]
        idx = np.concatenate(idx_pieces) if idx_pieces else np.empty(0, dtype=int)
        if idx.size < 2:
            boot[b] = np.nan
            continue
        boot[b] = float(statistic_fn(sample[idx]))

    alpha = (1.0 - level) / 2.0
    lo = float(np.nanpercentile(boot, 100.0 * alpha))
    hi = float(np.nanpercentile(boot, 100.0 * (1.0 - alpha)))
    point = float(statistic_fn(sample))
    return BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        level=level,
        n_bootstrap=n_bootstrap,
        method=f"spatial_block_{block_size_deg:.1f}deg_({n_cells}cells)",
    )
