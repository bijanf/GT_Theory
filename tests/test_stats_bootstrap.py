"""Tests for the BCa and spatial-block bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.stats import bca_ci, naive_percentile_ci, spatial_block_ci


def test_bca_recovers_known_mean_ci_coverage() -> None:
    """For a standard-normal sample of n=200, BCa 90% CI should bracket
    the population mean 0 about 90% of the time across repeated draws."""
    rng = np.random.default_rng(20260522)
    n_trials = 80
    hits = 0
    for k in range(n_trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=200)
        res = bca_ci(sample, lambda x: float(np.mean(x)), n_bootstrap=300, seed=k)
        if res.lo <= 0.0 <= res.hi:
            hits += 1
    # 90% nominal, 80 trials -> Wilson 95% CI on coverage ~ [0.81, 0.96].
    # Require coverage at least 0.78 (loose to allow Monte-Carlo noise).
    assert hits / n_trials >= 0.78, f"BCa coverage of mean = {hits / n_trials:.2f}"


def test_bca_differs_from_percentile_on_skewed_data() -> None:
    """For a lognormal sample, BCa applies a non-trivial bias and skew
    correction.  Compare BCa to the naive percentile on identical data
    and verify the endpoints are not identical (the correction is
    being applied)."""
    rng = np.random.default_rng(0)
    sample = rng.lognormal(mean=0.0, sigma=1.5, size=200)
    bca = bca_ci(sample, lambda x: float(np.mean(x)), n_bootstrap=1000, seed=1)
    pct = naive_percentile_ci(sample, lambda x: float(np.mean(x)), n_bootstrap=1000, seed=1)
    # At least one endpoint should differ by >0.5% of the point estimate.
    rel_diff = max(abs(bca.lo - pct.lo), abs(bca.hi - pct.hi)) / abs(bca.point)
    assert rel_diff > 0.005, (
        f"BCa correction had no effect; bca=[{bca.lo:.3f}, {bca.hi:.3f}], "
        f"pct=[{pct.lo:.3f}, {pct.hi:.3f}]"
    )


def test_bca_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 3 observations"):
        bca_ci(np.array([1.0, 2.0]), lambda x: float(np.mean(x)))


def test_spatial_block_inflates_ci_relative_to_naive_on_clustered_data() -> None:
    """Synthesize 200 sites clustered into ~10 spatial blocks.  The
    spatial-block bootstrap should produce a wider CI than the naive
    percentile bootstrap that treats the 200 sites as independent."""
    rng = np.random.default_rng(20260522)
    n_blocks = 10
    sites_per_block = 20
    block_lats = rng.uniform(-60.0, 60.0, size=n_blocks)
    block_lons = rng.uniform(-180.0, 180.0, size=n_blocks)
    block_means = rng.normal(loc=1.0, scale=0.4, size=n_blocks)
    lat = np.repeat(block_lats, sites_per_block) + rng.normal(0.0, 0.5, n_blocks * sites_per_block)
    lon = np.repeat(block_lons, sites_per_block) + rng.normal(0.0, 0.5, n_blocks * sites_per_block)
    values = np.repeat(block_means, sites_per_block) + rng.normal(
        0.0, 0.05, n_blocks * sites_per_block
    )

    naive = naive_percentile_ci(values, lambda x: float(np.mean(x)), n_bootstrap=800, seed=0)
    block = spatial_block_ci(
        values, lat, lon, lambda x: float(np.mean(x)), block_size_deg=10.0, n_bootstrap=800, seed=0
    )
    naive_width = naive.hi - naive.lo
    block_width = block.hi - block.lo
    assert block_width > naive_width, (
        f"spatial block bootstrap should widen CI on clustered data; "
        f"got naive={naive_width:.4f}, block={block_width:.4f}"
    )
    # Sanity: factor of at least 1.5 wider in this strongly clustered toy.
    assert block_width / naive_width >= 1.5


def test_spatial_block_rejects_mismatched_latlon() -> None:
    with pytest.raises(ValueError, match="must have length"):
        spatial_block_ci(
            np.array([1.0, 2.0, 3.0]),
            lat_deg=[0.0, 0.0],
            lon_deg=[0.0, 0.0],
            statistic_fn=lambda x: float(np.mean(x)),
            n_bootstrap=10,
        )


def test_bca_method_tag() -> None:
    sample = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    res = bca_ci(sample, lambda x: float(np.mean(x)), n_bootstrap=300)
    assert res.method in {"bca", "bca-fallback-percentile"}
    assert res.lo < res.point < res.hi
