"""Tests for the Nordicana D / Borealis readers used at the Umiujaq
supersite.

We exercise the parsers against synthetic CSV fixtures written into
``tmp_path``. The schemas mirror what the readers document (header
comments + one datetime column + one or more value columns); confirm
against the real bundled READMEs once the operator has curl'd the
data.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gt_theory.io import (
    align_to_hourly,
    load_d9_climate_series,
    load_immatsiak_head_series,
    load_vdtbs_thermistors,
)

# --------------------------------------------------------------- VDTBS thermistors


def _write_csv(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")


def test_load_vdtbs_thermistors_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "vdtbs_borehole_1.csv"
    _write_csv(
        p,
        """
        # Subsurface ground temperature data, Tasiapik Valley
        # DOI: 10.5683/SP3/QSRW0I
        datetime,T_0p5_m,T_1p0_m,T_3p0_m,T_10p0_m
        2012-08-01T00:00:00,8.10,5.20,1.30,-1.20
        2012-08-01T01:00:00,8.05,5.22,1.32,-1.20
        2012-08-01T02:00:00,8.00,5.25,1.34,-1.21
        """,
    )

    ds = load_vdtbs_thermistors(p)
    assert isinstance(ds, xr.Dataset)
    assert ds["T_degC"].dims == ("time", "depth_m")
    assert ds["T_degC"].shape == (3, 4)
    np.testing.assert_array_equal(ds["depth_m"].values, [0.5, 1.0, 3.0, 10.0])
    assert ds["T_degC"].isel(time=0, depth_m=0).item() == pytest.approx(8.10)
    assert ds.attrs["source_doi"] == "10.5683/SP3/QSRW0I"


def test_load_vdtbs_thermistors_rejects_unknown_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        """
        # no parseable depth columns
        datetime,foo,bar
        2012-08-01T00:00:00,1,2
        """,
    )
    with pytest.raises(ValueError, match="No thermistor depth columns matched"):
        load_vdtbs_thermistors(p)


def test_load_vdtbs_thermistors_missing_time_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    _write_csv(
        p,
        """
        timestamp,T_1p0_m
        2012-08-01T00:00:00,1.0
        """,
    )
    with pytest.raises(KeyError, match="Time column"):
        load_vdtbs_thermistors(p)


# --------------------------------------------------------------- Immatsiak head


def test_load_immatsiak_head_series_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "immatsiak1_head.csv"
    _write_csv(
        p,
        """
        # Immatsiak network of groundwater monitoring at Umiujaq
        # DOI: 10.5885/45309SL-15611D6EC6D34E23
        DateTime,Head_m,Flag
        2012-08-01T00:00:00,2.13,0
        2012-08-01T01:00:00,2.14,0
        2012-08-01T02:00:00,2.15,0
        """,
    )

    ds = load_immatsiak_head_series(p, station="Immatsiak_1")
    assert ds["head_m"].dims == ("time",)
    assert ds["head_m"].size == 3
    assert ds.attrs["station"] == "Immatsiak_1"
    assert ds["head_m"].isel(time=-1).item() == pytest.approx(2.15)


# --------------------------------------------------------------- D9 climate


def test_load_d9_climate_series_records_units_and_station(tmp_path: Path) -> None:
    p = tmp_path / "d9_VDTSYBU_Tair.csv"
    _write_csv(
        p,
        """
        # Climate station data from the Umiujaq region
        # DOI: 10.5885/45120SL-067305A53E914AF0
        DateTime,T_air_max_C
        2012-08-01T12:00:00,12.5
        2012-08-02T12:00:00,11.8
        """,
    )

    ds = load_d9_climate_series(
        p,
        value_col="T_air_max_C",
        variable_name="T_air_max_degC",
        units="degC",
        station="VDTSYBU",
    )
    assert ds["T_air_max_degC"].attrs["units"] == "degC"
    assert ds.attrs["station"] == "VDTSYBU"
    assert ds["T_air_max_degC"].size == 2


# --------------------------------------------------------------- alignment


def test_align_to_hourly_reindexes_two_products(tmp_path: Path) -> None:
    p1 = tmp_path / "T.csv"
    _write_csv(
        p1,
        """
        datetime,T_1p0_m
        2012-08-01T00:30:00,1.0
        2012-08-01T01:30:00,1.1
        2012-08-01T02:30:00,1.2
        """,
    )
    p2 = tmp_path / "head.csv"
    _write_csv(
        p2,
        """
        DateTime,Head_m
        2012-08-01T00:15:00,2.0
        2012-08-01T01:45:00,2.1
        2012-08-01T03:00:00,2.2
        """,
    )

    t_ds = load_vdtbs_thermistors(p1)
    p_ds = load_immatsiak_head_series(p2)
    t_aligned, p_aligned = align_to_hourly(
        t_ds, p_ds, start="2012-08-01 00:00", end="2012-08-01 03:00"
    )

    expected_times = pd.date_range("2012-08-01 00:00", "2012-08-01 03:00", freq="1h")
    np.testing.assert_array_equal(t_aligned["time"].values, expected_times.values)
    np.testing.assert_array_equal(p_aligned["time"].values, expected_times.values)
    # Each is a (1+, 1) shape with nearest-match within 2H tolerance.
    assert np.isfinite(t_aligned["T_degC"]).any()
    assert np.isfinite(p_aligned["head_m"]).any()


def test_loaders_raise_filenotfound_with_helpful_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="README.md"):
        load_vdtbs_thermistors(missing)
