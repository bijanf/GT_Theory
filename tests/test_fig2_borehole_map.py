"""Smoke test for Figure 2 — confirms the renderer runs against a tiny
synthetic catalog without needing GTV's full archive.

Cartopy ships its own data download in some setups; the test uses a
plain matplotlib axes path by stubbing in a minimal catalog and asking
the renderer to write to a tmp PDF.  We assert the file exists and is
non-trivial in size, not its pixel content.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

# Cartopy is an optional dependency (in [project.optional-dependencies]
# plot, not in dev), so the figure-2 renderer cannot be exercised on
# minimal CI environments.  Skip the whole module if it's missing
# rather than letting the import fail at collection time.
pytest.importorskip("cartopy", reason="cartopy not installed in this environment")

matplotlib.use("Agg")

# Add scripts/figures dirs to sys.path so we can import the module
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "figures"))


@pytest.fixture
def tiny_catalog(tmp_path: Path) -> Path:
    csv = tmp_path / "all_sites.csv"
    df = pd.DataFrame(
        {
            "site_id": ["AA-1", "BB-2", "CC-3", "AU-10", "ZA-AD3"],
            "country": ["Country A", "Country B", "Country C", "Australia", "South Africa"],
            "lat_deg": [10.0, -45.0, 60.0, -34.0, -25.0],
            "lon_deg": [20.0, 50.0, -100.0, 151.25, 29.15],
            "max_depth_m": [350.0, 220.0, 180.0, 449.58, 500.0],
            "measurement_year": [1980.0, 1990.0, 2000.0, 1972.51, 1989.0],
            "prior_gst_c": [10.0, 5.0, -1.0, 15.21, 19.0],
            "mean_kappa_w_m_k": [2.5, 2.5, 2.5, 2.7, 2.6],
            "mean_grad_k_per_km": [25.0, 25.0, 25.0, 28.0, 27.0],
            "n_samples": [60, 30, 20, 57, 50],
        }
    )
    df.to_csv(csv, index=False)
    return csv


@pytest.fixture
def tiny_curated(tmp_path: Path) -> Path:
    yml = tmp_path / "boreholes.yaml"
    yml.write_text(
        "schema_version: 1\n"
        "default_data_root: '.'\n"
        "subsets:\n"
        "  smoke-10: [AU-10, ZA-AD3]\n"
        "sites: {}\n"
    )
    return yml


def test_build_figure_writes_pdf(tmp_path: Path, tiny_catalog: Path, tiny_curated: Path) -> None:
    # Avoid cartopy's natural-earth download in restricted CI environments
    # by stubbing the rendering call to a no-op cache directory.
    os.environ.setdefault("CARTOPY_DATA_DIR", str(tmp_path / "cartopy"))

    from fig2_borehole_map import build_figure

    out = tmp_path / "fig2.pdf"
    build_figure(tiny_catalog, tiny_curated, out)
    assert out.exists()
    assert out.stat().st_size > 10_000  # vector PDFs from matplotlib are usually >10 kB
    # Quick sniff: PDF magic.
    assert out.read_bytes()[:5] == b"%PDF-"
