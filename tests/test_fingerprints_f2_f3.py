"""Tests for F2 (SAT-GST coupling) and F3 (latitudinal amplification)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gt_theory.fingerprints import compute_f2, compute_f3
from gt_theory.fingerprints.f2_coupling import _deming_slope, _ols_slope

# --------------------------------------------------------------- helpers


def _toy_inversions(deltas_K: np.ndarray) -> list[pd.DataFrame]:
    rows = []
    for d in deltas_K:
        rows.append(
            pd.DataFrame(
                {
                    "site_id": ["X"],
                    "bin_edge_young_yr": [0.0],
                    "bin_edge_old_yr": [25.0],
                    "median_K": [float(d)],
                    "ci_lo_K": [float(d) - 0.1],
                    "ci_hi_K": [float(d) + 0.1],
                    "kappa_median": [1.0e-6],
                    "residual_rms_median": [0.05],
                    "T0_K": [5.0],
                    "dTdz_K_per_m": [0.025],
                    "z_steady_min_m": [300.0],
                }
            )
        )
    return rows


def _toy_cru(site_ids: list[str], sat_offsets: np.ndarray) -> pd.DataFrame:
    """Build a CRU-like long-form table.  Baseline (1901-1960) is zero;
    the recent window (2000-2024) sits at ``sat_offsets[i]`` for site i."""
    months = pd.date_range("1901-01-01", "2024-12-01", freq="MS")
    frames = []
    for sid, off in zip(site_ids, sat_offsets, strict=True):
        vals = np.where(months.year >= 2000, off, 0.0)
        frames.append(pd.DataFrame({"site_id": sid, "time": months, "sat_c": vals}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------- regressors


def test_ols_slope_matches_known_line() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = 0.7 * x + 0.1
    assert _ols_slope(x, y) == pytest.approx(0.7, abs=1e-10)


def test_deming_slope_lambda_one_is_ols_for_clean_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 1.3 * x + rng.normal(scale=1e-9, size=50)  # near-zero noise
    assert _deming_slope(x, y, lam=1.0) == pytest.approx(1.3, abs=1e-3)


# --------------------------------------------------------------- F2


def test_f2_recovers_unit_slope_on_perfect_data() -> None:
    """If Delta GST equals Delta SAT site by site, both OLS and Deming
    slopes should be ~1.0."""
    deltas = np.array([0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.4])
    inversions = _toy_inversions(deltas)
    site_ids = [f"S{i}" for i in range(len(deltas))]
    cru = _toy_cru(site_ids, deltas)
    res = compute_f2(inversions=inversions, site_ids=site_ids, cru_sat=cru, n_bootstrap=80)
    np.testing.assert_allclose(res.delta_sat_K, deltas, atol=1e-9)
    np.testing.assert_allclose(res.delta_gst_K, deltas, atol=1e-9)
    assert res.ols_slope == pytest.approx(1.0, abs=1e-6)
    assert res.deming_slope == pytest.approx(1.0, abs=1e-6)


def test_f2_handles_missing_sites() -> None:
    deltas = np.array([0.4, 0.8, 1.2])
    inversions = _toy_inversions(deltas)
    site_ids = ["A", "B", "C"]
    # Only A and C have CRU rows.
    cru = _toy_cru(["A", "C"], np.array([0.4, 1.2]))
    res = compute_f2(inversions=inversions, site_ids=site_ids, cru_sat=cru, n_bootstrap=40)
    # delta_sat is NaN for B.
    assert np.isnan(res.delta_sat_K[1])
    # OLS slope ignores NaN entries.
    assert res.ols_slope == pytest.approx(1.0, abs=1e-6)


def test_f2_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_f2(inversions=[], site_ids=["A"], cru_sat=pd.DataFrame())
    with pytest.raises(ValueError, match="columns"):
        compute_f2(
            inversions=_toy_inversions(np.array([0.5])),
            site_ids=["A"],
            cru_sat=pd.DataFrame({"wrong_col": [0]}),
        )


# --------------------------------------------------------------- F3


def test_f3_ratio_above_one_for_boreal_amplified_warming() -> None:
    """Construct 3 boreal sites with stronger warming and 3 equator
    sites with weaker warming; the SAT ratio must exceed 1 and the GST
    ratio must reflect the constructed contrast."""
    site_ids = [
        "B1",
        "B2",
        "B3",  # |lat| >= 50, ramp 2 K
        "E1",
        "E2",
        "E3",  # |lat| <= 20, ramp 1 K
    ]
    lats = np.array([60.0, -60.0, 70.0, 10.0, -5.0, 15.0])
    sat_offsets = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    gst_offsets = np.array([1.8, 1.6, 1.7, 0.9, 0.8, 1.0])
    inversions = _toy_inversions(gst_offsets)
    cru = _toy_cru(site_ids, sat_offsets)
    res = compute_f3(
        inversions=inversions, site_ids=site_ids, lats_deg=lats, cru_sat=cru, n_bootstrap=80
    )
    assert res.n_boreal == 3
    assert res.n_equator == 3
    assert res.sat_ratio == pytest.approx(2.0, abs=1e-6)
    assert 1.5 < res.gst_ratio < 2.3
