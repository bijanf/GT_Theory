"""F1 — erfc-attenuation envelope test.

For every borehole, the theory predicts that the present-day temperature
anomaly profile lies within the erfc envelope traced out by the
recovered GST history:

    Delta T(z) ≈ G(z; bin_edges, kappa) @ s_recovered.

F1 collects the residual ``Delta T_obs(z) - Delta T_pred(z)`` across all
sites at a common depth grid and asks whether the median is within
+/- 0.2 K across the 20-600 m window the Huang-Pollack archive
restricts itself to.  A deviation outside that envelope flags either
(i) significant Darcy advection violating the pure-conduction kernel,
or (ii) a phase-change contribution at high latitudes.

The function consumes the same parquet that ``scripts/invert_profile.py``
emits, plus the per-site detrended profile parquet.  Both are produced
by the smoke-10 / full Snakemake targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.inversion import build_forward_operator


@dataclass(frozen=True)
class F1Result:
    depth_m: np.ndarray  # common depth grid (m)
    median_residual_K: np.ndarray  # site-median residual at each depth
    ci_lo_K: np.ndarray
    ci_hi_K: np.ndarray
    n_sites: int
    envelope_band_K: float  # the +/- target (default 0.2 K)
    passes_envelope: bool  # True iff |median| <= envelope_band across all depths

    @property
    def max_abs_residual_K(self) -> float:
        return float(np.max(np.abs(self.median_residual_K)))


def _resample_profile_to_grid(
    z_obs: np.ndarray,
    dT_obs: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Linear-interpolate a per-site detrended profile onto ``z_grid``,
    leaving NaN outside the observed depth range."""
    out = np.interp(z_grid, z_obs, dT_obs, left=np.nan, right=np.nan)
    return out


def compute_f1(
    profiles: list[pd.DataFrame],
    inversions: list[pd.DataFrame],
    *,
    depth_grid_m: np.ndarray | None = None,
    envelope_band_K: float = 0.2,
    n_bootstrap: int = 200,
    seed: int = 20260522,
) -> F1Result:
    """Compute F1 across a population of sites.

    Parameters
    ----------
    profiles
        One DataFrame per site, columns ``depth_m`` and ``temperature_c``.
    inversions
        One DataFrame per site, the output of ``invert_profile.py``.  Must
        contain ``bin_edge_young_yr``, ``bin_edge_old_yr``, ``median_K``,
        ``T0_K``, ``dTdz_K_per_m``, ``z_steady_min_m``, ``kappa_median``.
    depth_grid_m
        Common depth grid for cross-site comparison.  Default 20-600 m
        in 20 m steps, matching the Huang-Pollack 1-Variables window.
    envelope_band_K
        The +/- envelope width that the theory predicts (0.2 K is the
        specification in the accompanying paper).
    n_bootstrap
        Site-resample bootstrap draws for the CI on the cross-site
        median.

    Returns
    -------
    F1Result
    """
    if len(profiles) != len(inversions):
        raise ValueError(
            f"profiles ({len(profiles)}) and inversions ({len(inversions)}) "
            "must contain the same number of sites."
        )
    if depth_grid_m is None:
        depth_grid_m = np.arange(20.0, 601.0, 20.0)
    depth_grid_m = np.asarray(depth_grid_m, dtype=float)

    rng = np.random.default_rng(seed)
    residuals = np.full((len(profiles), depth_grid_m.size), np.nan)

    for i, (prof, inv) in enumerate(zip(profiles, inversions, strict=True)):
        z = prof["depth_m"].to_numpy(dtype=float)
        T = prof["temperature_c"].to_numpy(dtype=float)
        T0 = float(inv["T0_K"].iloc[0])
        dTdz = float(inv["dTdz_K_per_m"].iloc[0])
        dT_obs = T - (T0 + dTdz * z)

        edges = np.concatenate(
            [
                inv["bin_edge_young_yr"].to_numpy(dtype=float),
                inv["bin_edge_old_yr"].to_numpy(dtype=float)[-1:],
            ]
        )
        kappa = float(inv["kappa_median"].iloc[0])
        G = build_forward_operator(z, edges, kappa)
        s = inv["median_K"].to_numpy(dtype=float)
        dT_pred = G @ s

        residuals[i] = _resample_profile_to_grid(z, dT_obs - dT_pred, depth_grid_m)

    # Cross-site median residual at each depth.
    median = np.nanmedian(residuals, axis=0)

    # Bootstrap CI by resampling sites.
    n_sites = len(profiles)
    boot = np.empty((n_bootstrap, depth_grid_m.size))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sites, size=n_sites)
        boot[b] = np.nanmedian(residuals[idx], axis=0)
    ci_lo = np.nanpercentile(boot, 5.0, axis=0)
    ci_hi = np.nanpercentile(boot, 95.0, axis=0)

    passes = bool(np.all(np.abs(median) <= envelope_band_K))

    return F1Result(
        depth_m=depth_grid_m,
        median_residual_K=median,
        ci_lo_K=ci_lo,
        ci_hi_K=ci_hi,
        n_sites=n_sites,
        envelope_band_K=envelope_band_K,
        passes_envelope=passes,
    )


def load_smoke_pair(subset_dir: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Convenience: read a smoke-N subset's per-site profile + inversion
    parquets and return them paired by site_id."""
    prof_dir = subset_dir / "profiles"
    inv_dir = subset_dir / "inversions"
    site_ids = sorted(p.stem for p in prof_dir.glob("*.parquet"))
    profiles = []
    inversions = []
    for sid in site_ids:
        pp = prof_dir / f"{sid}.parquet"
        ip = inv_dir / f"{sid}.parquet"
        if not (pp.exists() and ip.exists()):
            continue
        profiles.append(pd.read_parquet(pp))
        inversions.append(pd.read_parquet(ip))
    return profiles, inversions
