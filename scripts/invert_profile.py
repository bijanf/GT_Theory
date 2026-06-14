#!/usr/bin/env python3
"""Invert one borehole profile to a posterior GST history via the
hierarchical bootstrap-Tikhonov machinery in ``gt_theory.inversion``.

Reads the profile parquet from the ingest step, runs the inversion, and
writes a summary parquet with one row per GST history bin.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gt_theory.inversion import (
    default_bin_edges_yr,
    detrend_geothermal,
    invert_posterior,
)


def _seed_from(site_id: str, seed_base: int) -> int:
    h = hashlib.sha1(site_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) ^ seed_base) & 0xFFFFFFFF


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", required=True, type=str)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--n-bootstrap", required=True, type=int)
    parser.add_argument("--seed-base", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.profile)
    z = df["depth_m"].to_numpy(dtype=np.float64)
    T = df["temperature_c"].to_numpy(dtype=np.float64)

    # Pick the deepest possible steady-state window that still has at
    # least 3 samples for the linear fit.  Try 300 m first (the F1
    # convention); fall back to progressively shallower thresholds.
    candidates = [300.0, 250.0, 200.0, 150.0, 100.0, max(60.0, 0.5 * z.max())]
    z_steady_min = None
    for c in candidates:
        if int(np.sum(z >= c)) >= 3:
            z_steady_min = c
            break
    if z_steady_min is None:
        # Last resort: use the deepest three points.
        z_steady_min = float(np.sort(z)[-3])
    dT, T0, dTdz = detrend_geothermal(z, T, z_steady_min_m=z_steady_min)

    rng = np.random.default_rng(_seed_from(args.site_id, args.seed_base))
    edges = default_bin_edges_yr()
    post = invert_posterior(
        z=z,
        dT_obs=dT,
        bin_edges_yr=edges,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )

    out = pd.DataFrame(
        {
            "site_id": [args.site_id] * (edges.size - 1),
            "bin_edge_young_yr": edges[:-1].astype(np.float64),
            "bin_edge_old_yr": edges[1:].astype(np.float64),
            "median_K": post.median.astype(np.float64),
            "ci_lo_K": post.ci_lo.astype(np.float64),
            "ci_hi_K": post.ci_hi.astype(np.float64),
            "kappa_median": float(np.median(post.kappa_samples)),
            "residual_rms_median": float(np.median(post.residual_rms)),
            "T0_K": T0,
            "dTdz_K_per_m": dTdz,
            "z_steady_min_m": z_steady_min,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(
        f"wrote {len(out)} bins -> {args.out} (T0={T0:.2f} K, dTdz={dTdz:.4f} K/m)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
