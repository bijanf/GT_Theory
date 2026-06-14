"""F6 — Latent-heat dominance fingerprint.

Theory prediction (Section 7.2 of the accompanying paper): in permafrost regions the
latent-heat number ``L_calL >> 1``, so the freeze-thaw enthalpy term
slows the propagation of surface signals downward.  The empirical
signature is that the borehole's temperature field spends a substantial
fraction of its history within the freezing interval ``[T_f - dTc,
T_f]``, where the latent-heat sink is active.

We measure that signature with a single-solver diagnostic that does
not require comparing against an enthalpy solver:

  occupancy = fraction of (depth, time) cells in the conduction-only
              forward simulation where ``T_f - dTc < T(z, t) < T_f``,
              integrated over the 0-50 m depth window and the CRU
              SAT history.

A site is flagged as 'latent-heat dominant' when ``occupancy >=
threshold`` (default 0.05 = 5% of the (z, t) domain).  The threshold
is calibrated so that mid-latitude sites with brief winter dips below
zero are not flagged, while permafrost sites with persistent
near-freezing temperatures are.

Why not the two-solver RMSE?
---------------------------
A first attempt compared the conduction-only solver against the
column_enthalpy solver and reported the RMSE between their present-day
profiles.  The enthalpy solver's variable-r Crank-Nicolson assembly
diverges from the column_1d's constant-r form even with no phase
change present, contaminating the diagnostic.  This single-solver
occupancy version sidesteps that bug and is what the manuscript
will report; the two-solver RMSE diagnostic is recoverable once
column_enthalpy is hardened to agree with column_1d on the
no-phase-change benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gt_theory.solvers import run_column_1d

YEAR_S: float = 365.25 * 86400.0


@dataclass(frozen=True)
class F6SiteResult:
    site_id: str
    lat_deg: float
    freezing_interval_occupancy: float  # fraction in [0, 1]
    latent_dominant: bool


@dataclass(frozen=True)
class F6Result:
    sites: list[F6SiteResult]
    threshold: float
    T_f: float
    dTc: float

    @property
    def n_dominant(self) -> int:
        return sum(1 for s in self.sites if s.latent_dominant)

    @property
    def fraction_dominant(self) -> float:
        return self.n_dominant / max(len(self.sites), 1)


def compute_f6_one_site(
    *,
    site_id: str,
    lat_deg: float,
    sat_c_monthly: np.ndarray,
    kappa_m2_s: float = 1.0e-6,
    depth_max_m: float = 50.0,
    dz_m: float = 1.0,
    dt_s: float = YEAR_S / 12.0,
    T_f: float = 0.0,
    dTc_K: float = 1.0,
    z_window_min_m: float = 0.0,
    z_window_max_m: float = 50.0,
    threshold: float = 0.05,
) -> F6SiteResult:
    """Forward-simulate the conduction-only column under monthly SAT
    forcing and report the fraction of the (depth, time) cells whose
    temperature lies inside the freezing interval (T_f - dTc, T_f).

    A site with ``occupancy >= threshold`` is flagged as latent-heat
    dominant.
    """
    sat = np.asarray(sat_c_monthly, dtype=float)
    nt = sat.size
    duration_s = (nt - 1) * dt_s
    if duration_s <= 0:
        raise ValueError("sat_c_monthly must have at least 2 samples")
    T_init = float(np.nanmean(sat))

    res = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa_m2_s,
        sat=sat,
        n_winter=1.0,
        T_init=T_init,
    )

    z_mask = (res.z >= z_window_min_m) & (res.z <= z_window_max_m)
    T_field = res.T[:, z_mask]
    in_interval = (T_field > T_f - dTc_K) & (T_field < T_f)
    occupancy = float(np.mean(in_interval))
    return F6SiteResult(
        site_id=site_id,
        lat_deg=lat_deg,
        freezing_interval_occupancy=occupancy,
        latent_dominant=occupancy >= threshold,
    )


def compute_f6(
    *,
    site_ids: list[str],
    lats_deg: np.ndarray,
    sat_panel: dict[str, np.ndarray],
    threshold: float = 0.05,
    T_f: float = 0.0,
    dTc_K: float = 1.0,
    **kwargs,
) -> F6Result:
    if len(site_ids) != len(lats_deg):
        raise ValueError("site_ids and lats_deg length mismatch")
    results: list[F6SiteResult] = []
    for sid, lat in zip(site_ids, lats_deg, strict=True):
        if sid not in sat_panel:
            continue
        r = compute_f6_one_site(
            site_id=sid,
            lat_deg=float(lat),
            sat_c_monthly=sat_panel[sid],
            T_f=T_f,
            dTc_K=dTc_K,
            threshold=threshold,
            **kwargs,
        )
        results.append(r)
    return F6Result(sites=results, threshold=threshold, T_f=T_f, dTc=dTc_K)
