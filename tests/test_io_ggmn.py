"""Tests for the GGMN annual-time-series reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gt_theory.io.ggmn import (
    head_trend_per_station,
    load_ggmn,
    nearest_station,
)


@pytest.fixture
def ggmn_csv(tmp_path: Path) -> Path:
    """Synthetic two-station / four-year GGMN annual file."""
    df = pd.DataFrame(
        {
            "StnID": [
                1001,
                1001,
                1001,
                1001,
                1002,
                1002,
                1002,
                1002,
            ],
            "Lat": [40.0] * 4 + [-30.0] * 4,
            "Lon": [-100.0] * 4 + [25.0] * 4,
            "IntegerYear": [2010, 2011, 2012, 2013] * 2,
            # Station 1001: depth rises (water table falls) at +0.5 m/yr.
            # Station 1002: depth constant (water table flat).
            "DepthToWater_m": [
                10.0,
                10.5,
                11.0,
                11.5,
                5.0,
                5.0,
                5.0,
                5.0,
            ],
        }
    )
    p = tmp_path / "ggmn_synth.csv"
    df.to_csv(p, index=False)
    return p


def test_load_ggmn_shape_and_coords(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    assert ds["depth_to_water_m"].shape == (2, 4)
    assert list(ds["station"].values) == [1001, 1002]
    assert list(ds["year"].values) == [2010, 2011, 2012, 2013]
    assert ds["lat"].values.tolist() == [40.0, -30.0]
    assert ds["lon"].values.tolist() == [-100.0, 25.0]


def test_load_ggmn_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ggmn(tmp_path / "absent.csv")


def test_load_ggmn_rejects_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_ggmn(bad)


def test_nearest_station(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    near = nearest_station(ds, lat_deg=41.0, lon_deg=-101.0)
    assert near is not None
    assert near.station_id == 1001
    # Roughly ~150 km from (40, -100) to (41, -101)
    assert 50.0 < near.distance_km < 200.0


def test_nearest_station_with_max_distance(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    # 0,0 is far from both synthetic stations; tight max_distance returns None.
    near = nearest_station(ds, lat_deg=0.0, lon_deg=0.0, max_distance_km=100.0)
    assert near is None


def test_head_trend_per_station(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    trend = head_trend_per_station(ds, min_years=3)
    # Station 1001: depth rising +0.5 m/yr, so head FALLING at -0.5 m/yr.
    assert trend.sel(station=1001).item() == pytest.approx(-0.5, abs=1e-9)
    # Station 1002: flat.
    assert trend.sel(station=1002).item() == pytest.approx(0.0, abs=1e-9)


def test_head_trend_min_years(ggmn_csv: Path) -> None:
    ds = load_ggmn(ggmn_csv)
    trend = head_trend_per_station(ds, min_years=10)
    assert np.isnan(trend.values).all()
