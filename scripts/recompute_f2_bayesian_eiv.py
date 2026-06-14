#!/usr/bin/env python3
"""R17 W3 — recompute the F2 SAT-GST coupling slope with a
hierarchical Bayesian errors-in-variables model.

Closes editor item 3: the F2 statistics dodge (Deming unstable,
OLS = 1.70 as biased fallback). This script:

1. Loads ``outputs/global/ensemble_summary.parquet`` (per-site
   ΔGST posterior median).
2. Extracts per-site monthly CRU TS SAT at each H-P (lat, lon)
   from ``data/raw/cru_ts/``.
3. Computes per-site ΔSAT as the difference between
   recent-window (2000-2024) mean and baseline-window (1901-1960)
   mean.
4. Runs the hierarchical EIV Gibbs sampler from
   :mod:`gt_theory.inversion.bayesian_eiv` with the three
   latitude bands (tropical / mid / boreal).
5. Compares the EIV posterior median to OLS and Deming-λ=4.
6. Emits ``outputs/global/f2_bayesian_eiv.parquet`` with
   posterior samples for each band slope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.inversion.bayesian_eiv import (
    gibbs_eiv,
    latitude_band_index,
)
from gt_theory.io.cru_ts import extract_sat_at_point, load_cru_ts


DEFAULT_SUMMARY = "outputs/global/ensemble_summary.parquet"
DEFAULT_CRU = Path("data/raw/cru_ts/cru_ts4.09.1901.2024.tmp.dat.nc")
DEFAULT_OUT = "outputs/global/f2_bayesian_eiv.parquet"

BASELINE_WINDOW = (1901, 1960)
RECENT_WINDOW = (2000, 2024)


def _delta_sat_at_site(
    cru_da,
    *,
    lat_deg: float,
    lon_deg: float,
) -> float:
    """Δ SAT = recent-window mean - baseline-window mean (K)."""
    try:
        series = extract_sat_at_point(
            cru_da,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            tolerance_deg=1.0,
        )
    except Exception:
        return float("nan")
    times = pd.to_datetime(series["time"].values)
    years = times.year
    baseline_mask = (years >= BASELINE_WINDOW[0]) & (years <= BASELINE_WINDOW[1])
    recent_mask = (years >= RECENT_WINDOW[0]) & (years <= RECENT_WINDOW[1])
    sat = series.values.astype(float)
    if baseline_mask.sum() < 50 or recent_mask.sum() < 24:
        return float("nan")
    return float(np.nanmean(sat[recent_mask]) - np.nanmean(sat[baseline_mask]))


def _ols(x: np.ndarray, y: np.ndarray) -> float:
    x_bar = x.mean()
    y_bar = y.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx == 0:
        return float("nan")
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    return sxy / sxx


def _deming(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    x_bar = x.mean()
    y_bar = y.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    syy = float(np.sum((y - y_bar) ** 2))
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    discriminant = (syy - lam * sxx) ** 2 + 4.0 * lam * sxy * sxy
    if discriminant < 0 or sxy == 0:
        return float("nan")
    return float(((syy - lam * sxx) + np.sqrt(discriminant)) / (2.0 * sxy))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cru", default=str(DEFAULT_CRU))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n-draws", type=int, default=4000)
    parser.add_argument("--n-burn", type=int, default=1000)
    args = parser.parse_args(argv)

    summary = pd.read_parquet(Path(args.summary).expanduser().resolve())
    print(f"loaded {len(summary)} sites from {args.summary}", flush=True)

    print(f"loading CRU TS {args.cru}", flush=True)
    cru = load_cru_ts(args.cru)

    print("extracting ΔSAT per site...", flush=True)
    delta_sat = np.empty(len(summary))
    for i, (_, row) in enumerate(summary.iterrows()):
        delta_sat[i] = _delta_sat_at_site(
            cru,
            lat_deg=float(row["lat_deg"]),
            lon_deg=float(row["lon_deg"]),
        )
        if (i + 1) % 200 == 0 or i + 1 == len(summary):
            print(
                f"  {i + 1}/{len(summary)}  "
                f"(finite ΔSAT: {int(np.isfinite(delta_sat[: i + 1]).sum())})",
                flush=True,
            )
    delta_gst = summary["gst_median_recent_K"].to_numpy(dtype=float)
    lats = summary["lat_deg"].to_numpy(dtype=float)

    finite = np.isfinite(delta_sat) & np.isfinite(delta_gst)
    n_finite = int(finite.sum())
    print(f"\nrunning hierarchical EIV Gibbs on {n_finite} sites...", flush=True)
    x = delta_sat[finite]
    y = delta_gst[finite]
    band_idx, band_labels = latitude_band_index(lats[finite])

    posterior = gibbs_eiv(
        x_obs=x,
        y_obs=y,
        band_index=band_idx,
        n_bands=3,
        sigma_x=0.5,
        sigma_y=0.5,
        n_draws=args.n_draws,
        n_burn=args.n_burn,
        beta_global_prior_mean=1.0,
        beta_global_prior_sd=2.0,
        band_labels=band_labels,
    )
    s = posterior.summary()

    ols_slope = _ols(x, y)
    deming_l4 = _deming(x, y, lam=4.0)
    deming_l1 = _deming(x, y, lam=1.0)

    print("\n=== R17 W3 headline ===")
    print(f"sites with finite ΔSAT + ΔGST: {n_finite}")
    print(f"OLS slope (biased, attenuated):  {ols_slope:.3f}")
    print(f"Deming λ=1 (orthogonal):         {deming_l1:.3f}")
    print(f"Deming λ=4 (per the accompanying paper): {deming_l4:.3f}")
    print(
        f"EIV β_global (posterior median): {s['beta_global']['median']:.3f}  "
        f"(90% CI [{s['beta_global']['ci_lo']:.3f}, "
        f"{s['beta_global']['ci_hi']:.3f}])"
    )
    for label in band_labels:
        b = s[f"beta_{label}"]
        print(
            f"  β_{label:<8s}: median {b['median']:.3f}  "
            f"90% CI [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]"
        )

    # Save posterior samples + per-site x, y.
    rows = []
    n_draws_kept = posterior.beta_global.size
    for k in range(n_draws_kept):
        rows.append(
            {
                "draw": k,
                "alpha": posterior.alpha[k],
                "beta_global": posterior.beta_global[k],
                "tau": posterior.tau[k],
                **{
                    f"beta_{label}": float(posterior.beta_band[k, g])
                    for g, label in enumerate(band_labels)
                },
            }
        )
    df_draws = pd.DataFrame(rows)
    df_draws.attrs["ols_slope"] = float(ols_slope)
    df_draws.attrs["deming_lambda_4"] = float(deming_l4)
    df_draws.attrs["deming_lambda_1"] = float(deming_l1)
    df_draws.attrs["n_sites_used"] = n_finite

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_draws.to_parquet(out_path)
    # Also save the per-site (x, y, band) pairs for the figure.
    df_pairs = pd.DataFrame(
        {
            "site_id": summary.loc[finite, "site_id"].to_numpy(),
            "lat_deg": lats[finite],
            "delta_sat_K": x,
            "delta_gst_K": y,
            "band_index": band_idx,
            "band_label": [band_labels[g] for g in band_idx],
        }
    )
    pairs_path = out_path.with_name(out_path.stem + "_pairs.parquet")
    df_pairs.to_parquet(pairs_path)
    print(f"\nwrote {out_path}  ({n_draws_kept} draws)")
    print(f"wrote {pairs_path}  ({n_finite} site pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
