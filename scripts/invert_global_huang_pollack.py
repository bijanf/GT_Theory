#!/usr/bin/env python3
"""Global Huang-Pollack ensemble inverter.

Runs the analytic-erfc Bayesian inverter
(:func:`gt_theory.inversion.invert_posterior`) over every parseable
borehole in the Huang-Pollack archive. Saves a per-site posterior
(samples + median + 90 % CI) plus an aggregate summary parquet.

The H-P data are temperature-only steady-state logs, so the
inverter constrains kappa + GST history under the conduction-only
kernel. This is the natural global reference for the empirical
paper's regime-diagram claim: the 951-site cloud sits in the
Pe_T ~ 0 quadrant, and the three supersites with co-located T+p
populate the high-coupling quadrants the framework predicts.

Outputs
-------
- ``outputs/global/posteriors/<site_id>.npz`` -- per-site
  posterior samples, kappa draws, residual-RMS draws.
- ``outputs/global/ensemble_summary.parquet`` -- one row per site
  with: lat, lon, max_depth_m, posterior-median kappa, posterior-
  median GST per bin, residual-RMS median, QC tier.

Usage
-----

    python scripts/invert_global_huang_pollack.py
    python scripts/invert_global_huang_pollack.py --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gt_theory.catalog import load_catalog, resolve_data_root
from gt_theory.inversion import (
    PosteriorResult,
    default_bin_edges_yr,
    detrend_geothermal,
    invert_posterior,
)
from gt_theory.io import iter_borehole_archive


def _invert_one(
    profile, *, n_bootstrap: int, rng: np.random.Generator
) -> tuple[PosteriorResult, dict[str, Any]] | None:
    """Detrend + invert a single profile. Returns None if the profile
    is too shallow / too sparse for the standard z_steady_min_m fit."""
    z = profile.depth_m
    T = profile.temperature_c
    if z.size < 8 or z[-1] < 200.0:
        return None
    try:
        # Adapt the steady-state floor to the deepest 30 % of the profile.
        z_steady = float(max(150.0, z[int(0.65 * z.size)]))
        dT, T0, dTdz = detrend_geothermal(z, T, z_steady_min_m=z_steady)
    except ValueError:
        return None
    post = invert_posterior(
        z=z,
        dT_obs=dT,
        bin_edges_yr=default_bin_edges_yr(),
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    meta = {
        "site_id": profile.site_id,
        "lat_deg": profile.lat_deg,
        "lon_deg": profile.lon_deg,
        "max_depth_m": profile.max_depth_m,
        "n_points": int(z.size),
        "geo_gradient_K_per_km": dTdz * 1000.0,
        "geo_intercept_C": T0,
        "z_steady_min_m": z_steady,
    }
    return post, meta


def _save_posterior(post: PosteriorResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        bin_edges_yr=post.bin_edges_yr,
        samples=post.samples.astype(np.float32),
        median=post.median.astype(np.float32),
        ci_lo=post.ci_lo.astype(np.float32),
        ci_hi=post.ci_hi.astype(np.float32),
        kappa_samples=post.kappa_samples.astype(np.float32),
        residual_rms=post.residual_rms.astype(np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default="outputs/global",
        help="Output root for posteriors + summary parquet",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=300,
        help="Bootstrap draws per site (default 300)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N sites (smoke test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260523,
    )
    args = parser.parse_args(argv)

    cat = load_catalog()
    root = resolve_data_root(cat)
    out_dir = Path(args.out_dir).expanduser().resolve()
    post_dir = out_dir / "posteriors"
    post_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(args.seed)
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    for k, profile in enumerate(iter_borehole_archive(root)):
        if args.limit is not None and n_ok >= args.limit:
            break
        rng = np.random.default_rng(rng_master.integers(2**32))
        result = _invert_one(profile, n_bootstrap=args.n_bootstrap, rng=rng)
        if result is None:
            n_skip += 1
            continue
        post, meta = result
        _save_posterior(post, post_dir / f"{profile.site_id}.npz")
        meta.update(
            {
                "kappa_median": float(np.median(post.kappa_samples)),
                "kappa_p05": float(np.quantile(post.kappa_samples, 0.05)),
                "kappa_p95": float(np.quantile(post.kappa_samples, 0.95)),
                "residual_rms_median": float(np.median(post.residual_rms)),
                "gst_median_recent_K": float(post.median[0]),
                "gst_p05_recent_K": float(post.ci_lo[0]),
                "gst_p95_recent_K": float(post.ci_hi[0]),
            }
        )
        rows.append(meta)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  {n_ok} ok / {n_skip} skipped  ({time.time() - t0:.1f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    summary_path = out_dir / "ensemble_summary.parquet"
    df.to_parquet(summary_path)
    print(f"wrote {summary_path}  ({len(df)} sites, {n_skip} skipped)", file=sys.stderr)
    print(f"posteriors -> {post_dir}/", file=sys.stderr)
    print(f"total wall time: {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
