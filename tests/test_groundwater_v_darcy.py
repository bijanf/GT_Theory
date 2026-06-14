"""Tests for the per-site Darcy-velocity proxy from GGMN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gt_theory.io.ggmn import load_ggmn
from gt_theory.io.groundwater_v_darcy import (
    K_HYD_LITH_MS,
    site_v_darcy,
)


@pytest.fixture
def ggmn_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "StnID": [10] * 6 + [11] * 6,
            "Lat": [40.0] * 6 + [-30.0] * 6,
            "Lon": [-100.0] * 6 + [25.0] * 6,
            "IntegerYear": list(range(2010, 2016)) * 2,
            # Station 10: depth rising +0.4 m/yr → head falling -0.4 m/yr.
            # Station 11: depth flat → head flat.
            "DepthToWater_m": [
                10.0,
                10.4,
                10.8,
                11.2,
                11.6,
                12.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ],
        }
    )
    p = tmp_path / "ggmn.csv"
    df.to_csv(p, index=False)
    return p


def test_v_darcy_nonzero_at_station_with_trend(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    est = site_v_darcy(
        ds,
        lat_deg=40.0,
        lon_deg=-100.0,
        depth_m=300.0,
        lithology="sandstone",
    )
    # Station 10 trend = -0.4 m/yr (head falling); sandstone K = 1e-5 m/s.
    # |v_darcy| = K * |dh/dt| / SECONDS_PER_YEAR / L_z
    #           = 1e-5 * 0.4 / 3.155e7 / 300
    #           ~ 4.2e-16 m/s in magnitude, with negative sign.
    assert est.nearest_station_id == 10
    assert est.head_trend_m_per_yr == pytest.approx(-0.4, abs=1e-9)
    assert est.K_hyd_m_per_s == K_HYD_LITH_MS["sandstone"]
    assert est.v_darcy_m_per_s != 0.0
    assert np.sign(est.v_darcy_m_per_s) == np.sign(est.head_trend_m_per_yr)


def test_v_darcy_zero_at_flat_station(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    est = site_v_darcy(
        ds,
        lat_deg=-30.0,
        lon_deg=25.0,
        depth_m=300.0,
        lithology="limestone",
    )
    assert est.nearest_station_id == 11
    assert est.head_trend_m_per_yr == pytest.approx(0.0, abs=1e-9)
    assert est.v_darcy_m_per_s == pytest.approx(0.0, abs=1e-20)


def test_v_darcy_unknown_lithology_falls_back(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    est = site_v_darcy(
        ds,
        lat_deg=40.0,
        lon_deg=-100.0,
        depth_m=300.0,
        lithology="missing-lithology-key",
    )
    assert est.lithology == "missing-lithology-key"
    assert est.K_hyd_m_per_s == K_HYD_LITH_MS["unknown"]


def test_v_darcy_no_station_within_max_distance(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    est = site_v_darcy(
        ds,
        lat_deg=80.0,
        lon_deg=160.0,
        depth_m=300.0,
        lithology="sandstone",
        max_distance_km=200.0,
    )
    # 80 N, 160 E is far from both synthetic stations; max=200 km excludes.
    assert est.nearest_station_id is None
    assert est.v_darcy_m_per_s == 0.0


def test_v_darcy_K_scales_with_lithology(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    sandstone = site_v_darcy(
        ds,
        lat_deg=40.0,
        lon_deg=-100.0,
        depth_m=300.0,
        lithology="sandstone",
    )
    granite = site_v_darcy(
        ds,
        lat_deg=40.0,
        lon_deg=-100.0,
        depth_m=300.0,
        lithology="granite",
    )
    # K_sandstone / K_granite = 1e-5 / 1e-10 = 1e5; v_darcy ratio same.
    ratio = abs(sandstone.v_darcy_m_per_s) / abs(granite.v_darcy_m_per_s)
    assert ratio == pytest.approx(K_HYD_LITH_MS["sandstone"] / K_HYD_LITH_MS["granite"], rel=1e-9)
