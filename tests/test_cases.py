"""Smoke tests for the four canonical regime case studies.

Each case should produce a finite, bounded forward solution and
land in the regime quadrant the theory predicts.
"""

from __future__ import annotations

import numpy as np

from gt_theory.cases import (
    run_arid_basin,
    run_geothermal,
    run_permafrost,
    run_thermo_poro,
)


def _check_finite(ds) -> None:
    for var in ds.data_vars:
        arr = ds[var].values
        assert np.all(np.isfinite(arr)), f"{var} contains non-finite values"


def test_permafrost_case_runs() -> None:
    ds, _ = run_permafrost()
    _check_finite(ds)
    # Should have sub-zero temperatures somewhere (winter).
    assert float(ds["T_degC"].min()) < -1.0
    # Should have non-zero ice fraction at some point.
    assert float(ds["S_i"].max()) > 0.5


def test_geothermal_case_runs() -> None:
    ds, _ = run_geothermal()
    _check_finite(ds)
    # Should be hot at depth.
    assert float(ds["T_degC"].max()) > 100.0
    # Should have non-zero Darcy velocity.
    assert float(np.abs(ds["v_darcy"].values).max()) > 1e-9


def test_arid_basin_case_runs() -> None:
    ds, _ = run_arid_basin()
    _check_finite(ds)
    # Should NOT have sub-zero temperatures.
    assert float(ds["T_degC"].min()) > 5.0
    # Should have negligible Darcy velocity.
    assert float(np.abs(ds["v_darcy"].values).max()) < 1e-8


def test_thermo_poro_case_runs() -> None:
    ds, _ = run_thermo_poro()
    _check_finite(ds)
    # Should have heated up significantly.
    assert float(ds["T_degC"].max()) > 50.0
    # Should have built up substantial pressure (undrained heating).
    assert float(ds["p_Pa"].max()) > 1.0e5
