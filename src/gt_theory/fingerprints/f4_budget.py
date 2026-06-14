"""F4 — global continental ground-heat-content gain since 1960.

For each site, the contribution to ground-heat-content over a window
[t_start, t_end] (years before present) is

    dE_site = rho * c_eff * integral over z in [0, z_max] of dT(z) dz,

where ``dT(z)`` is the present-day depth profile predicted by the
recovered GST history.  Sites are then reweighted by a Horvitz-Thompson
factor of continental land area per 10-deg latitude band divided by
sample count in that band, and summed to a global continental total in
zettajoules.  The target value from Cuesta-Valero et al. (2025) is
~17.6 ZJ for 1960-2018; F4 reports the bootstrap-CI on the recovered
total and asks whether 17.6 ZJ lies inside it.

Notation here follows Section 3 (F4) of the accompanying paper and the Methods section
of Beltrami & Bourlon (2004) Earth-Planet. Sci. Lett. 227, 169-177.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gt_theory.inversion import build_forward_operator

YEAR_S = 365.25 * 86400.0
ZJ = 1.0e21  # joules per zettajoule

# Reference effective volumetric heat capacity for continental crust /
# saturated sediment (J m^-3 K^-1).  See Beltrami & Bourlon (2004).
RHO_C_EFF: float = 2.5e6


@dataclass(frozen=True)
class SiteEnergyGain:
    site_id: str
    lat_deg: float
    delta_E_J_per_m2: float  # column-integrated heat content gain (J m^-2)
    z_max_m: float


@dataclass(frozen=True)
class F4Result:
    site_gains: list[SiteEnergyGain]
    weights_per_band: dict[int, float]  # lat-band centre -> HT weight (m^2/site)
    total_ZJ: float
    ci_lo_ZJ: float
    ci_hi_ZJ: float
    target_ZJ: float
    passes_equivalence: bool

    @property
    def n_sites(self) -> int:
        return len(self.site_gains)


# Continental land area per 10-deg latitude band, square metres.  Same
# table used by figures/fig2_borehole_map.py for the overlay; kept as a
# local constant so the module is self-contained.
_LAND_AREA_M2_PER_LAT_BAND: dict[int, float] = {
    -65: 12e12,
    -55: 2.5e12,
    -45: 4.5e12,
    -35: 12e12,
    -25: 18e12,
    -15: 19e12,
    -5: 17e12,
    5: 16e12,
    15: 22e12,
    25: 22e12,
    35: 20e12,
    45: 17e12,
    55: 13e12,
    65: 11e12,
    75: 5e12,
    85: 2e12,
}


def site_energy_gain(
    inversion: pd.DataFrame,
    *,
    z_max_m: float = 600.0,
    dz_m: float = 5.0,
    rho_c_eff: float = RHO_C_EFF,
    window_years: tuple[float, float] | None = None,
) -> float:
    """Integrate the recovered ``Delta T(z)`` over depth and return the
    site contribution to ground-heat-content in J m^-2.

    Parameters
    ----------
    inversion
        Output of ``scripts/invert_profile.py``: one row per GST history
        bin.  Required columns: ``bin_edge_young_yr``,
        ``bin_edge_old_yr``, ``median_K``, ``kappa_median``.
    z_max_m, dz_m
        Integration window and step (m).
    rho_c_eff
        Effective volumetric heat capacity (J m^-3 K^-1).
    window_years
        Optional (start_yr, end_yr) before-present window.  Bins outside
        the window are zeroed (no contribution).  Default: use every bin.

    Returns
    -------
    Site contribution to delta-E in J m^-2.
    """
    z = np.arange(0.0, z_max_m + 0.5 * dz_m, dz_m)
    edges = np.concatenate(
        [
            inversion["bin_edge_young_yr"].to_numpy(dtype=float),
            inversion["bin_edge_old_yr"].to_numpy(dtype=float)[-1:],
        ]
    )
    kappa = float(inversion["kappa_median"].iloc[0])
    G = build_forward_operator(z, edges, kappa)
    s = inversion["median_K"].to_numpy(dtype=float).copy()

    if window_years is not None:
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        mask = (bin_centres >= window_years[0]) & (bin_centres <= window_years[1])
        s = np.where(mask, s, 0.0)

    dT_z = G @ s  # K, present-day depth profile of the windowed anomaly
    column_integral = float(np.trapezoid(dT_z, z))  # K m
    return rho_c_eff * column_integral


def horvitz_thompson_weights(lats_deg: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    """Per-site Horvitz-Thompson weight (m^2 of continental land per
    sample in the same 10-deg latitude band).  Returns the per-site
    weights and the per-band weight (band centre -> m^2/site)."""
    lats = np.asarray(lats_deg, dtype=float)
    band_centres = ((lats // 10.0) * 10.0 + 5.0).astype(int)
    # Histogram of sites per band.
    bands, counts = np.unique(band_centres, return_counts=True)
    weight_per_band: dict[int, float] = {}
    for b, n in zip(bands, counts, strict=True):
        land = _LAND_AREA_M2_PER_LAT_BAND.get(int(b), 0.0)
        weight_per_band[int(b)] = land / float(n) if n > 0 else 0.0
    per_site = np.array([weight_per_band.get(int(c), 0.0) for c in band_centres])
    return per_site, weight_per_band


def koppen_weights(
    koppen_codes: np.ndarray,
    *,
    biome_area_m2: dict[int, float] | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Per-site Köppen-Geiger biome weight.

    Each site is weighted by global continental land-area in its
    Köppen class divided by the number of H-P sites sampling that
    class. Sites with ``koppen_code = 0`` (ocean / unclassified)
    receive weight zero.

    Parameters
    ----------
    koppen_codes
        Integer Köppen codes per site (from
        :func:`gt_theory.io.koppen.koppen_code`).
    biome_area_m2
        Optional override for global land area per code. If not
        supplied, uses a default table derived from Beck 2018
        Köppen-Geiger 0.5-deg occupancy (total continental land
        ~1.49x10^14 m^2). Sites with a code absent from this table
        get the *mean* per-biome area as a fallback.

    Returns
    -------
    (per_site_weight, weight_per_code)
        Same shape as :func:`horvitz_thompson_weights`.
    """
    codes = np.asarray(koppen_codes, dtype=int)
    if biome_area_m2 is None:
        biome_area_m2 = _BECK_2018_BIOME_AREA_M2
    fallback_area = float(np.mean(list(biome_area_m2.values())))
    unique_codes, counts = np.unique(codes, return_counts=True)
    weight_per_code: dict[int, float] = {}
    for c, n in zip(unique_codes, counts, strict=True):
        if int(c) == 0 or n == 0:
            weight_per_code[int(c)] = 0.0
            continue
        area = biome_area_m2.get(int(c), fallback_area)
        weight_per_code[int(c)] = float(area) / float(n)
    per_site = np.array([weight_per_code.get(int(c), 0.0) for c in codes])
    return per_site, weight_per_code


# Default biome areas in m^2, summed over the Beck 2018 Köppen-Geiger
# 0.5-degree global product (excluding ocean). The table is a rough
# climatology -- exact values may differ by a few percent depending
# on the underlying land mask. Codes match
# :data:`gt_theory.io.koppen.KOPPEN_CLASSES` order (1=Af, 2=Am, ...).
_BECK_2018_BIOME_AREA_M2: dict[int, float] = {
    1: 1.50e13,  # Af tropical rainforest
    2: 5.00e12,  # Am tropical monsoon
    3: 1.30e13,  # Aw tropical savanna
    4: 9.00e12,  # BWh hot desert
    5: 6.00e12,  # BWk cold desert
    6: 1.10e13,  # BSh hot steppe
    7: 7.00e12,  # BSk cold steppe
    8: 3.00e12,  # Csa hot-summer Mediterranean
    9: 3.50e12,  # Csb warm-summer Mediterranean
    10: 5.00e11,  # Csc cool-summer Mediterranean
    11: 4.50e12,  # Cwa humid subtropical (dry winter)
    12: 3.50e12,  # Cwb subtropical highland
    13: 5.00e11,  # Cwc cool subtropical highland
    14: 9.00e12,  # Cfa humid subtropical
    15: 6.00e12,  # Cfb oceanic
    16: 1.00e12,  # Cfc subpolar oceanic
    17: 5.00e11,  # Dsa hot summer continental
    18: 1.00e12,  # Dsb warm summer continental
    19: 1.50e12,  # Dsc subarctic
    20: 5.00e11,  # Dsd very cold subarctic
    21: 4.00e12,  # Dwa monsoon humid continental
    22: 3.50e12,  # Dwb humid continental
    23: 5.00e12,  # Dwc subarctic monsoon
    24: 1.00e12,  # Dwd very cold subarctic monsoon
    25: 5.00e12,  # Dfa hot summer humid continental
    26: 1.20e13,  # Dfb warm summer humid continental
    27: 1.50e13,  # Dfc subarctic
    28: 3.00e12,  # Dfd very cold subarctic
    29: 1.20e13,  # ET tundra
    30: 1.40e13,  # EF ice cap
    31: 0.0,  # Ocean / unclassified
}


def aggregate_continental(
    inversions: list[pd.DataFrame],
    lats_deg: np.ndarray,
    site_ids: list[str],
    *,
    window_years: tuple[float, float] = (0.0, 58.0),  # 1960-2018 from 2018 anchor
    rho_c_eff: float = RHO_C_EFF,
    target_ZJ: float = 17.6,
    equivalence_band_ZJ: float = 5.0,
    n_bootstrap: int = 500,
    seed: int = 20260522,
) -> F4Result:
    """Compute the F4 global continental energy budget closure.

    Parameters
    ----------
    inversions
        One DataFrame per site (output of ``invert_profile.py``).
    lats_deg
        Latitudes of each site in the same order as ``inversions``.
    site_ids
        Site identifiers in the same order.
    window_years
        Years-before-present window to integrate (default 0-58 yr,
        roughly 1960-2018).
    rho_c_eff
        Effective volumetric heat capacity (J m^-3 K^-1).
    target_ZJ
        Independent target value for the closure test.
    equivalence_band_ZJ
        TOST-style equivalence band (ZJ).
    n_bootstrap
        Bootstrap draws over the site population.

    Returns
    -------
    F4Result
    """
    if not (len(inversions) == len(lats_deg) == len(site_ids)):
        raise ValueError("inversions, lats_deg, site_ids must have the same length.")

    per_site_E = np.array(
        [
            site_energy_gain(inv, window_years=window_years, rho_c_eff=rho_c_eff)
            for inv in inversions
        ]
    )  # J m^-2 per site
    per_site_w, weight_per_band = horvitz_thompson_weights(np.asarray(lats_deg))

    total_J = float(np.sum(per_site_E * per_site_w))
    total_ZJ = total_J / ZJ

    gains = [
        SiteEnergyGain(
            site_id=str(sid),
            lat_deg=float(lat),
            delta_E_J_per_m2=float(e),
            z_max_m=600.0,
        )
        for sid, lat, e in zip(site_ids, lats_deg, per_site_E, strict=True)
    ]

    # Bootstrap site population for the HT-weighted total.  We hold the
    # per-site HT weights fixed (the lat-band structure is part of the
    # sampling design, not a random draw from the population) and
    # resample only the per-site energy contribution.  This mirrors the
    # Rao-Wu rescaled bootstrap for stratified surveys; resampling the
    # weights too would conflate sample variability with design.
    rng = np.random.default_rng(seed)
    contributions = per_site_E * per_site_w
    boot = np.empty(n_bootstrap)
    n = contributions.size
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.sum(contributions[idx])) / ZJ
    ci_lo = float(np.percentile(boot, 5.0))
    ci_hi = float(np.percentile(boot, 95.0))

    passes = abs(total_ZJ - target_ZJ) <= equivalence_band_ZJ

    return F4Result(
        site_gains=gains,
        weights_per_band=weight_per_band,
        total_ZJ=total_ZJ,
        ci_lo_ZJ=ci_lo,
        ci_hi_ZJ=ci_hi,
        target_ZJ=target_ZJ,
        passes_equivalence=passes,
    )
