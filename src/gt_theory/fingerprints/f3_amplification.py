"""F3 — Latitudinal amplification of the surface-to-subsurface signal.

Compare the boreal-to-equator ratio of Delta GST (recovered from
boreholes) with the Delta SAT ratio (observed from CRU TS).  Theory
prediction in the accompanying paper: GST ratio ~ 0.94, SAT ratio ~ 1.49.  A
substantial gap signals that the subsurface does NOT directly inherit
surface Arctic amplification, consistent with snow/permafrost
decoupling at high latitudes.

Latitude bands
--------------
boreal  : |lat| >= 50 deg
equator : |lat| <= 20 deg

The thresholds match Section 3 (F3) of the accompanying paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gt_theory.fingerprints.f2_coupling import _window_mean


@dataclass(frozen=True)
class F3Result:
    gst_boreal_K: float
    gst_equator_K: float
    sat_boreal_K: float
    sat_equator_K: float
    gst_ratio: float  # boreal / equator
    sat_ratio: float
    gst_ratio_ci: tuple[float, float]
    sat_ratio_ci: tuple[float, float]
    n_boreal: int
    n_equator: int


def _ratio(a: float, b: float) -> float:
    return float("nan") if (b == 0 or not np.isfinite(b)) else a / b


def compute_f3(
    *,
    inversions: list[pd.DataFrame],
    site_ids: list[str],
    lats_deg: np.ndarray,
    cru_sat: pd.DataFrame,
    boreal_lat: float = 50.0,
    equator_lat: float = 20.0,
    baseline_window: tuple[int, int] = (1901, 1960),
    recent_window: tuple[int, int] = (2000, 2024),
    n_bootstrap: int = 500,
    seed: int = 20260522,
) -> F3Result:
    if not (len(inversions) == len(site_ids) == len(lats_deg)):
        raise ValueError("inversions, site_ids, lats_deg length mismatch")

    lats = np.asarray(lats_deg, dtype=float)
    delta_sat = np.full(len(site_ids), np.nan)
    delta_gst = np.full(len(site_ids), np.nan)
    for i, sid in enumerate(site_ids):
        sat = cru_sat[cru_sat["site_id"] == sid]
        if sat.empty:
            continue
        delta_sat[i] = _window_mean(sat, *recent_window) - _window_mean(sat, *baseline_window)
        delta_gst[i] = float(inversions[i].iloc[0]["median_K"])

    boreal = (np.abs(lats) >= boreal_lat) & np.isfinite(delta_sat) & np.isfinite(delta_gst)
    equator = (np.abs(lats) <= equator_lat) & np.isfinite(delta_sat) & np.isfinite(delta_gst)

    gst_b = float(np.mean(delta_gst[boreal])) if boreal.sum() > 0 else float("nan")
    gst_e = float(np.mean(delta_gst[equator])) if equator.sum() > 0 else float("nan")
    sat_b = float(np.mean(delta_sat[boreal])) if boreal.sum() > 0 else float("nan")
    sat_e = float(np.mean(delta_sat[equator])) if equator.sum() > 0 else float("nan")

    gst_ratio = _ratio(gst_b, gst_e)
    sat_ratio = _ratio(sat_b, sat_e)

    # Bootstrap CIs by resampling sites within each band.
    rng = np.random.default_rng(seed)
    idx_b = np.where(boreal)[0]
    idx_e = np.where(equator)[0]
    boot_g = np.full(n_bootstrap, np.nan)
    boot_s = np.full(n_bootstrap, np.nan)
    if idx_b.size > 0 and idx_e.size > 0:
        for k in range(n_bootstrap):
            bi = rng.choice(idx_b, size=idx_b.size, replace=True)
            ei = rng.choice(idx_e, size=idx_e.size, replace=True)
            boot_g[k] = _ratio(float(np.mean(delta_gst[bi])), float(np.mean(delta_gst[ei])))
            boot_s[k] = _ratio(float(np.mean(delta_sat[bi])), float(np.mean(delta_sat[ei])))
    gst_ci = (float(np.nanpercentile(boot_g, 5.0)), float(np.nanpercentile(boot_g, 95.0)))
    sat_ci = (float(np.nanpercentile(boot_s, 5.0)), float(np.nanpercentile(boot_s, 95.0)))

    return F3Result(
        gst_boreal_K=gst_b,
        gst_equator_K=gst_e,
        sat_boreal_K=sat_b,
        sat_equator_K=sat_e,
        gst_ratio=gst_ratio,
        sat_ratio=sat_ratio,
        gst_ratio_ci=gst_ci,
        sat_ratio_ci=sat_ci,
        n_boreal=int(boreal.sum()),
        n_equator=int(equator.sum()),
    )
