"""Misfit diagnostic: compare a forward-modelled T(z, t) field against
observed thermistor records on the common (time, depth) grid.

The merged solver returns ``T(nt, nz)`` on its own uniform grids; the
VDTBS thermistor product returns ``T_degC(time, depth_m)`` (with a
borehole dimension that we average over here, after rejecting any
borehole with insufficient finite coverage).  This module handles the
projection of one onto the other and computes the per-depth and
total-volume RMS misfits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class MisfitResult:
    """Outcome of a forward-vs-observed comparison.

    Attributes
    ----------
    rms_total_K : float
        Root-mean-square misfit over all common (time, depth) cells.
    rms_per_depth_K : np.ndarray
        RMS misfit profile vs. depth (length nz_common).
    bias_per_depth_K : np.ndarray
        Mean (forward - observed) bias profile vs. depth.
    common_depths_m : np.ndarray
        Depth grid used for the comparison (m).
    n_cells : int
        Number of finite (forward, observed) pairs that entered the
        statistic.
    """

    rms_total_K: float
    rms_per_depth_K: np.ndarray
    bias_per_depth_K: np.ndarray
    common_depths_m: np.ndarray
    n_cells: int


def _times_to_seconds_since(reference: np.datetime64, times: np.ndarray) -> np.ndarray:
    return (times - reference) / np.timedelta64(1, "s")


def rms_misfit_on_common_grid(
    *,
    forward_T_K: np.ndarray,  # shape (nt_fwd, nz_fwd)
    forward_t_s: np.ndarray,  # shape (nt_fwd,) seconds since forward t0
    forward_z_m: np.ndarray,  # shape (nz_fwd,)
    forward_t0: np.datetime64,
    observed_ds: xr.Dataset,
    obs_T_var: str = "T_degC",
    obs_time_dim: str = "time",
    obs_depth_dim: str = "depth_m",
    obs_borehole_dim: str | None = "borehole",
    depth_min_m: float = 0.5,
    depth_max_m: float | None = None,
) -> MisfitResult:
    """Project a forward T(z, t) field onto the observed thermistor
    grid and return RMS misfit statistics.

    The observed dataset is collapsed to a single (time, depth) field
    by taking the nan-mean across the borehole dimension (if any).
    Both forward and observed temperatures must be in the same units
    (degC or K -- the misfit is sign-preserving and unit-agnostic).
    """
    obs = observed_ds
    if obs_borehole_dim is not None and obs_borehole_dim in obs.dims:
        obs_T = obs[obs_T_var].mean(dim=obs_borehole_dim, skipna=True)
    else:
        obs_T = obs[obs_T_var]

    obs_times = obs[obs_time_dim].values
    obs_depths = obs[obs_depth_dim].values

    # Restrict to a depth window where both the forward solution and
    # the observations have coverage (skip the very-near-surface
    # noisy/freeze-thaw 0 m thermistor; cap at the borehole bottom).
    z_lo = float(depth_min_m)
    z_hi = (
        float(depth_max_m)
        if depth_max_m is not None
        else float(min(float(forward_z_m.max()), float(obs_depths.max())))
    )
    obs_mask = (obs_depths >= z_lo) & (obs_depths <= z_hi)
    common_depths = obs_depths[obs_mask]
    obs_T_sub = obs_T.values[:, obs_mask]

    # Interpolate forward solution onto observed depth grid (linear).
    fwd_T_on_obs_depth = np.array(
        [np.interp(common_depths, forward_z_m, forward_T_K[k]) for k in range(forward_T_K.shape[0])]
    )
    # Interpolate forward solution from forward times onto obs times.
    obs_times_s = _times_to_seconds_since(forward_t0, obs_times)
    fwd_T_on_obs_grid = np.empty((obs_times.size, common_depths.size))
    for jd in range(common_depths.size):
        fwd_T_on_obs_grid[:, jd] = np.interp(
            obs_times_s,
            forward_t_s,
            fwd_T_on_obs_depth[:, jd],
            left=np.nan,
            right=np.nan,
        )

    finite = np.isfinite(obs_T_sub) & np.isfinite(fwd_T_on_obs_grid)
    if finite.sum() == 0:
        return MisfitResult(
            rms_total_K=float("nan"),
            rms_per_depth_K=np.full(common_depths.size, np.nan),
            bias_per_depth_K=np.full(common_depths.size, np.nan),
            common_depths_m=common_depths,
            n_cells=0,
        )
    diff = fwd_T_on_obs_grid - obs_T_sub
    rms_total = float(np.sqrt(np.mean(diff[finite] ** 2)))

    rms_per_depth = np.full(common_depths.size, np.nan)
    bias_per_depth = np.full(common_depths.size, np.nan)
    for jd in range(common_depths.size):
        col_mask = finite[:, jd]
        if col_mask.sum() < 5:
            continue
        d = diff[col_mask, jd]
        rms_per_depth[jd] = float(np.sqrt(np.mean(d**2)))
        bias_per_depth[jd] = float(np.mean(d))

    return MisfitResult(
        rms_total_K=rms_total,
        rms_per_depth_K=rms_per_depth,
        bias_per_depth_K=bias_per_depth,
        common_depths_m=common_depths,
        n_cells=int(finite.sum()),
    )
