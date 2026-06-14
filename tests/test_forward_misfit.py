"""Tests for gt_theory.diagnostics.forward_misfit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gt_theory.diagnostics.forward_misfit import rms_misfit_on_common_grid


def _make_obs_ds(
    t: pd.DatetimeIndex,
    depths: np.ndarray,
    T: np.ndarray,
) -> xr.Dataset:
    return xr.Dataset(
        data_vars={"T_degC": (("time", "depth_m"), T)},
        coords={"time": t, "depth_m": depths},
    )


def test_perfect_recovery_zero_misfit() -> None:
    nt, nz = 50, 5
    times = pd.date_range("2010-01-01", periods=nt, freq="D")
    depths = np.linspace(1.0, 10.0, nz)
    T_obs = np.broadcast_to(np.arange(nt, dtype=float)[:, None], (nt, nz)).copy()
    obs = _make_obs_ds(times, depths, T_obs)

    fwd_t_s = (np.arange(nt) * 86400.0).astype(float)
    fwd_z = depths.copy()
    fwd_T = T_obs.copy()
    fwd_t0 = times[0].to_numpy()
    res = rms_misfit_on_common_grid(
        forward_T_K=fwd_T,
        forward_t_s=fwd_t_s,
        forward_z_m=fwd_z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        obs_borehole_dim=None,
        depth_min_m=0.0,
    )
    assert res.rms_total_K == pytest.approx(0.0)


def test_constant_bias_recovery() -> None:
    nt, nz = 30, 4
    times = pd.date_range("2010-01-01", periods=nt, freq="D")
    depths = np.array([2.0, 5.0, 10.0, 15.0])
    T_obs = np.broadcast_to(np.arange(nt, dtype=float)[:, None], (nt, nz)).copy()
    obs = _make_obs_ds(times, depths, T_obs)
    fwd_t_s = (np.arange(nt) * 86400.0).astype(float)
    fwd_z = depths.copy()
    # Forward is observed + 2 K everywhere.
    fwd_T = T_obs + 2.0
    fwd_t0 = times[0].to_numpy()
    res = rms_misfit_on_common_grid(
        forward_T_K=fwd_T,
        forward_t_s=fwd_t_s,
        forward_z_m=fwd_z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        obs_borehole_dim=None,
        depth_min_m=0.0,
    )
    assert res.rms_total_K == pytest.approx(2.0)
    assert np.allclose(res.bias_per_depth_K, 2.0)


def test_nan_handling_in_observations() -> None:
    nt, nz = 20, 3
    times = pd.date_range("2010-01-01", periods=nt, freq="D")
    depths = np.array([1.0, 5.0, 10.0])
    T_obs = np.broadcast_to(np.arange(nt, dtype=float)[:, None], (nt, nz)).copy()
    T_obs[10:, 2] = np.nan
    obs = _make_obs_ds(times, depths, T_obs)
    fwd_t_s = (np.arange(nt) * 86400.0).astype(float)
    fwd_z = depths.copy()
    fwd_T = T_obs.copy()
    fwd_t0 = times[0].to_numpy()
    res = rms_misfit_on_common_grid(
        forward_T_K=fwd_T,
        forward_t_s=fwd_t_s,
        forward_z_m=fwd_z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        obs_borehole_dim=None,
        depth_min_m=0.0,
    )
    # NaNs ignored -> still zero misfit at the valid cells.
    assert res.rms_total_K == pytest.approx(0.0)
    # n_cells should count only finite pairs.
    assert res.n_cells == 50  # 20*3 - 10 NaN obs


def test_borehole_dimension_averaged() -> None:
    nt, nz, nb = 10, 3, 2
    times = pd.date_range("2010-01-01", periods=nt, freq="D")
    depths = np.array([1.0, 5.0, 10.0])
    T = np.zeros((nt, nz, nb))
    T[:, :, 0] = 1.0
    T[:, :, 1] = 3.0
    obs = xr.Dataset(
        data_vars={"T_degC": (("time", "depth_m", "borehole"), T)},
        coords={"time": times, "depth_m": depths, "borehole": ["BH1", "BH2"]},
    )
    fwd_t_s = (np.arange(nt) * 86400.0).astype(float)
    fwd_z = depths.copy()
    fwd_T = 2.0 * np.ones((nt, nz))  # equals the borehole-mean
    fwd_t0 = times[0].to_numpy()
    res = rms_misfit_on_common_grid(
        forward_T_K=fwd_T,
        forward_t_s=fwd_t_s,
        forward_z_m=fwd_z,
        forward_t0=fwd_t0,
        observed_ds=obs,
        obs_borehole_dim="borehole",
        depth_min_m=0.0,
    )
    # Borehole mean = 2.0 everywhere; forward is also 2.0 -> zero misfit.
    assert res.rms_total_K == pytest.approx(0.0)
