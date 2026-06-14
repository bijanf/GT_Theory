"""InSAR-vs-solver vertical-displacement cross-check.

Given the coupled solver's depth-time fields ``T(z, t)`` and
``p(z, t)`` from :class:`gt_theory.solvers.CoupledResult`, the
expected surface vertical displacement (mm, positive upward) is
the column integral of the volumetric strain:

    u_z(t) = ∫₀^L  [ alpha_eff * (T(z, t) - T(z, 0))
                    + beta_eff * (p(z, t) - p(z, 0)) ] dz

with the effective thermal-expansion coefficient

    alpha_eff = phi * alpha_w + (1 - phi) * alpha_solid

(typically ~3-5×10⁻⁵ K⁻¹ for water-saturated continental rocks),
and the effective pressure-compaction coefficient

    beta_eff = phi * beta_w + (1 - phi) * beta_solid

(typically ~5×10⁻¹¹ Pa⁻¹). The sign convention is geodetic: positive
``u_z`` corresponds to surface uplift, which is what PS-InSAR
ortho-vertical (VU) products report.

The cross-check then asks: does adding the framework's
$\\Gamma N_\\alpha$ coupling term improve the residual against the
observed PS-InSAR time series? The diagnostic is

    R = 1 - RMS(u_obs - u_solver, s=1) / RMS(u_obs - u_solver, s=0)

A positive ``R`` means coupling helps; ``R = 0`` means it doesn't;
the framework predicts ``R > 0`` at advection-dominated sites and
``R ≈ 0`` at conduction-dominated sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from gt_theory.solvers import CoupledResult

# Defaults consistent with typical saturated continental crust.
ALPHA_WATER: float = 2.1e-4  # K^-1
ALPHA_SOLID: float = 1.5e-5  # K^-1 (quartz-feldspar mean)
BETA_WATER: float = 4.5e-10  # Pa^-1
BETA_SOLID: float = 1.0e-11  # Pa^-1 (typical rock matrix)


def predict_surface_displacement(
    result: CoupledResult,
    *,
    porosity: float = 0.15,
    alpha_w: float = ALPHA_WATER,
    alpha_solid: float = ALPHA_SOLID,
    beta_w: float = BETA_WATER,
    beta_solid: float = BETA_SOLID,
) -> np.ndarray:
    """Compute u_z(t) in millimetres from a coupled-solver result.

    Parameters
    ----------
    result
        Output of :func:`gt_theory.solvers.run_column_coupled`.
    porosity, alpha_w, alpha_solid, beta_w, beta_solid
        Per-layer constitutive parameters; defaults are
        water-saturated continental crust.

    Returns
    -------
    ndarray, shape (nt,)
        Surface vertical displacement time series (mm, positive
        upward), referenced to the initial (t=0) state.
    """
    z = result.z
    T = result.T - result.T[0]
    p = result.p - result.p[0]

    alpha_eff = porosity * alpha_w + (1.0 - porosity) * alpha_solid
    beta_eff = porosity * beta_w + (1.0 - porosity) * beta_solid

    # Volumetric strain at each (t, z); integrate over z using the trapezoidal
    # rule. Result in metres -> convert to mm.
    strain_thermal = alpha_eff * T
    strain_pressure = beta_eff * p
    strain = strain_thermal + strain_pressure  # (nt, nz)

    u_z_m = np.trapezoid(strain, x=z, axis=1)
    return u_z_m * 1.0e3  # mm


@dataclass(frozen=True)
class InSARSolverComparison:
    """Aggregated cross-check result for one supersite."""

    t_common: np.ndarray  # common time axis (datetime64 or float seconds)
    u_obs_mm: np.ndarray  # observed vertical displacement
    u_solver_on_mm: np.ndarray  # solver with s=1
    u_solver_off_mm: np.ndarray  # solver with s=0
    rms_on_mm: float
    rms_off_mm: float
    residual_reduction: float  # 1 - rms_on/rms_off


def compare_insar_solver(
    insar_ds: xr.Dataset,
    *,
    solver_t_s: np.ndarray,
    u_solver_on_mm: np.ndarray,
    u_solver_off_mm: np.ndarray,
    reference_time: pd.Timestamp,
) -> InSARSolverComparison:
    """Interpolate observed + simulated u_z onto a common time axis
    and report RMS residuals.

    Parameters
    ----------
    insar_ds
        Output of :func:`gt_theory.io.egms.load_egms_csv`, already
        subset to a single PS point via :func:`nearest_point`. The
        Dataset must have a ``time`` coord (datetime64) and a
        ``displacement_mm`` variable indexed on ``(time,)`` only
        (after the ``isel`` from nearest_point).
    solver_t_s
        Solver time axis in seconds since ``reference_time``.
    u_solver_on_mm, u_solver_off_mm
        Solver surface-displacement series in mm at s=1 and s=0.
    reference_time
        Calendar timestamp corresponding to ``solver_t_s = 0``.
    """
    obs_t = pd.to_datetime(insar_ds["time"].values)
    obs_t_s = (obs_t - reference_time).total_seconds().to_numpy()

    # Restrict the common axis to the observational support.
    obs_min, obs_max = obs_t_s.min(), obs_t_s.max()
    common_mask = (solver_t_s >= obs_min) & (solver_t_s <= obs_max)
    if int(common_mask.sum()) < 2:
        raise ValueError("Solver time axis does not overlap the InSAR observation window.")
    t_common = solver_t_s[common_mask]
    obs_at_common = np.interp(t_common, obs_t_s, insar_ds["displacement_mm"].values)

    u_on = u_solver_on_mm[common_mask]
    u_off = u_solver_off_mm[common_mask]

    rms_on = float(np.sqrt(np.mean((obs_at_common - u_on) ** 2)))
    rms_off = float(np.sqrt(np.mean((obs_at_common - u_off) ** 2)))
    if rms_off <= 0:
        red = 0.0
    else:
        red = float(1.0 - rms_on / rms_off)

    return InSARSolverComparison(
        t_common=t_common,
        u_obs_mm=obs_at_common,
        u_solver_on_mm=u_on,
        u_solver_off_mm=u_off,
        rms_on_mm=rms_on,
        rms_off_mm=rms_off,
        residual_reduction=red,
    )


def residual_reduction(
    insar_ds: xr.Dataset,
    *,
    solver_t_s: np.ndarray,
    u_solver_on_mm: np.ndarray,
    u_solver_off_mm: np.ndarray,
    reference_time: pd.Timestamp,
) -> float:
    """Thin wrapper that returns only the scalar residual-reduction
    statistic (useful for inversion-loss closures)."""
    return compare_insar_solver(
        insar_ds,
        solver_t_s=solver_t_s,
        u_solver_on_mm=u_solver_on_mm,
        u_solver_off_mm=u_solver_off_mm,
        reference_time=reference_time,
    ).residual_reduction
