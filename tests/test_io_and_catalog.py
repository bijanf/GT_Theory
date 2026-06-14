"""Tests for the Huang-Pollack reader, the borehole catalog loader, and
the CRU TS helpers (without requiring a CRU TS file on disk)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from gt_theory.catalog import Catalog, load_catalog, resolve_data_root
from gt_theory.io import (
    BoreholeProfile,
    extract_sat_at_point,
    monthly_anomaly,
    normalise_longitudes,
    parse_huang_pollack,
)

FIXTURE = Path(__file__).parent / "data" / "huang-2013-AU-10.txt"


# --------------------------------------------------------------- parser


def test_parse_huang_pollack_extracts_metadata_and_data() -> None:
    p = parse_huang_pollack(FIXTURE)
    assert isinstance(p, BoreholeProfile)
    assert p.site_id == "AU-10"
    assert p.country == "Australia"
    assert p.lat_deg == pytest.approx(-34.0)
    assert p.lon_deg == pytest.approx(151.25)
    assert p.max_depth_m == pytest.approx(449.58)
    assert p.measurement_year == pytest.approx(1972.51, abs=1e-3)
    assert p.prior_gst_c == pytest.approx(15.21, abs=1e-3)
    assert p.mean_conductivity_w_m_k == pytest.approx(2.7215, abs=1e-4)
    assert p.mean_gradient_k_per_km == pytest.approx(28.159, abs=1e-3)
    assert p.depth_m.shape == p.temperature_c.shape
    assert p.depth_m.size >= 50
    assert p.depth_m[0] == pytest.approx(22.86)
    assert p.depth_m[-1] == pytest.approx(449.58)
    # Temperatures monotonically increase with depth at this site.
    assert np.all(np.diff(p.temperature_c) > 0)


def test_parse_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        parse_huang_pollack(tmp_path / "does_not_exist.txt")


def test_parse_rejects_file_without_data(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("# only a header\n# Site_Name: BAD\n")
    with pytest.raises(ValueError, match="No data header"):
        parse_huang_pollack(bad)


def test_parse_rejects_file_without_lat_lon(tmp_path: Path) -> None:
    bad = tmp_path / "nocoords.txt"
    bad.write_text(
        "# Site_Name: NO\n"
        "Depth_m    Temperature_Celsius    Notes\n"
        "10.0  5.0\n20.0  5.1\n30.0  5.2\n40.0  5.3\n50.0  5.4\n"
    )
    with pytest.raises(ValueError, match="lat/lon"):
        parse_huang_pollack(bad)


# --------------------------------------------------------------- catalog


def test_catalog_loads_smoke_10_subset() -> None:
    cat = load_catalog()
    assert isinstance(cat, Catalog)
    assert cat.schema_version == 1
    assert "smoke-10" in cat.subsets
    ids = cat.subset_ids("smoke-10")
    assert len(ids) == 10
    # Each smoke-10 ID has a per-site metadata entry.
    for sid in ids:
        m = cat.site(sid)
        assert -90.0 <= m.lat_deg <= 90.0
        assert -180.0 <= m.lon_deg <= 180.0
        assert m.qc_tier in {1, 2, 3}


def test_catalog_subset_missing_raises() -> None:
    cat = load_catalog()
    with pytest.raises(KeyError, match="not in catalog"):
        cat.subset_ids("does-not-exist")


def test_catalog_site_missing_raises() -> None:
    cat = load_catalog()
    with pytest.raises(KeyError, match="not in catalog"):
        cat.site("ZZ-no-such-site")


def test_resolve_data_root_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cat = load_catalog()
    monkeypatch.setenv("GT_THEORY_BOREHOLE_ROOT", str(tmp_path))
    root = resolve_data_root(cat)
    assert root == tmp_path.resolve()


# ----------------------------------------------------------- CRU helpers


def _toy_cru_dataarray() -> xr.DataArray:
    """Build a minimal 4-time x 3-lat x 4-lon DataArray emulating CRU TS."""
    time = xr.date_range("2000-01-01", periods=12, freq="MS")
    lats = np.array([-30.0, 0.0, 30.0])
    lons = np.array([10.0, 20.0, 30.0, 200.0])  # last one >180 to exercise normalise
    data = (
        np.arange(time.size)[:, None, None] + 0.1 * lats[None, :, None] + 0.01 * lons[None, None, :]
    )
    return xr.DataArray(
        data,
        coords={"time": time, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="tmp",
    )


def test_extract_sat_at_point_returns_timeseries() -> None:
    da = _toy_cru_dataarray()
    ts = extract_sat_at_point(da, lat_deg=0.0, lon_deg=20.0)
    assert ts.dims == ("time",)
    assert ts.size == 12


def test_monthly_anomaly_zeros_on_no_seasonal_drift() -> None:
    da = _toy_cru_dataarray()
    # The synthetic field has a linear-in-month component; anomaly should
    # remove it only if baseline equals the full record (it does).
    anom = monthly_anomaly(da)
    # mean of anomaly over time at each (lat,lon) should be ~0.
    np.testing.assert_allclose(anom.mean(dim="time"), 0.0, atol=1e-10)


def test_normalise_longitudes_signs_and_sorts() -> None:
    da = _toy_cru_dataarray()
    out = normalise_longitudes(da, to_signed=True)
    lons_new = out["lon"].values
    assert np.all(np.diff(lons_new) > 0)
    assert lons_new.min() >= -180.0
    assert lons_new.max() <= 180.0
    # The lon=200 should map to -160.
    assert -160.0 in lons_new
