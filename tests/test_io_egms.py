"""Tests for the EGMS PS-InSAR CSV reader.

We exercise the parser against synthetic CSVs in ``tmp_path`` --
schemas mirror the real EGMS Level-3 ortho-vertical / ortho-east
column conventions. Confirm against the bundled README once
operator downloads land.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from gt_theory.io.egms import (
    load_egms_csv,
    nearest_point,
    points_in_bbox,
)


def _write_csv(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")


def test_egms_reader_round_trips_metadata_and_time_series(tmp_path: Path) -> None:
    p = tmp_path / "EGMS_L3_E12N40_E12N42_VU.csv"
    _write_csv(
        p,
        """
        pid,latitude,longitude,height,velocity,velocity_std,20200115,20200127,20200208
        a01,47.230,7.150,650.0,-1.20,0.20,0.0,-0.4,-0.8
        a02,47.232,7.155,651.0,-0.80,0.15,0.0,-0.3,-0.5
        a03,47.234,7.160,652.0,+0.10,0.25,0.0,+0.05,+0.10
        """,
    )

    ds = load_egms_csv(p)
    assert ds.sizes["point"] == 3
    assert ds.sizes["time"] == 3
    assert list(ds["point"].values) == ["a01", "a02", "a03"]
    assert ds["lat"].values[0] == pytest.approx(47.230)
    assert ds["velocity_mm_yr"].values[1] == pytest.approx(-0.80)
    assert ds["displacement_mm"].values[0, -1] == pytest.approx(-0.8)
    # Time axis should be sorted.
    times = ds["time"].values
    assert (np.diff(times.astype("int64")) > 0).all()
    assert ds.attrs["license"].startswith("CC-BY 4.0")


def test_egms_reader_synthesises_pid_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "no_pid.csv"
    _write_csv(
        p,
        """
        latitude,longitude,velocity,20200101,20200201
        47.0,7.0,-1.5,0.0,-0.5
        47.5,7.5,-2.0,0.0,-0.6
        """,
    )
    ds = load_egms_csv(p)
    assert list(ds["point"].values) == ["pt_000000", "pt_000001"]
    assert np.isnan(ds["velocity_std_mm_yr"].values).all()


def test_egms_reader_missing_required_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        """
        latitude,velocity,20200101
        47.0,-1.0,0.0
        """,
    )
    with pytest.raises(KeyError, match="longitude"):
        load_egms_csv(p)


def test_egms_reader_handles_no_time_series_columns(tmp_path: Path) -> None:
    p = tmp_path / "vel_only.csv"
    _write_csv(
        p,
        """
        pid,latitude,longitude,velocity
        a1,47.0,7.0,-1.2
        a2,47.5,7.5,-0.8
        """,
    )
    ds = load_egms_csv(p)
    assert ds.sizes["time"] == 0
    assert ds["displacement_mm"].shape == (2, 0)


def test_points_in_bbox_clips_correctly(tmp_path: Path) -> None:
    p = tmp_path / "bbox.csv"
    _write_csv(
        p,
        """
        pid,latitude,longitude,velocity,20200101
        a,47.0,7.0,-1.0,0.0
        b,48.0,8.0,-1.0,0.0
        c,49.0,9.0,-1.0,0.0
        """,
    )
    ds = load_egms_csv(p)
    sub = points_in_bbox(ds, lat_lo=47.5, lat_hi=48.5, lon_lo=7.5, lon_hi=8.5)
    assert sub.sizes["point"] == 1
    assert sub["lat"].values[0] == pytest.approx(48.0)


def test_nearest_point_returns_closest(tmp_path: Path) -> None:
    p = tmp_path / "nearest.csv"
    _write_csv(
        p,
        """
        pid,latitude,longitude,velocity,20200101
        a,47.0,7.0,-1.0,0.0
        b,47.235,7.155,-1.0,0.0
        c,49.0,9.0,-1.0,0.0
        """,
    )
    ds = load_egms_csv(p)
    pt = nearest_point(ds, lat_deg=47.235, lon_deg=7.155)
    assert str(pt["point"].values) == "b"


def test_egms_reader_filenotfound_points_at_readme(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="README"):
        load_egms_csv(tmp_path / "missing.csv")
