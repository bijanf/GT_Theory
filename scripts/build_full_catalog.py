#!/usr/bin/env python3
"""Scan the Huang-Pollack archive on disk and emit a CSV with one row per
site (site_id, country, lat_deg, lon_deg, max_depth_m, measurement_year).

This is a one-shot helper meant to be run on a machine that has the
archive present, e.g. a developer's laptop with the raw data dir.  The
emitted CSV is committed under ``catalogs/`` so that the figure pipeline
and CI smoke runs do not need the raw data.

Usage::

    python scripts/build_full_catalog.py \\
        --archive data/raw/boreholes/huang2000 \\
        --out catalogs/all_sites.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gt_theory.io import iter_borehole_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="Fail on first parse error.")
    args = parser.parse_args(argv)

    if not args.archive.exists():
        print(f"ERROR: archive {args.archive} does not exist", file=sys.stderr)
        return 2

    rows = []
    for profile in iter_borehole_archive(args.archive, strict=args.strict):
        rows.append(
            {
                "site_id": profile.site_id,
                "country": profile.country,
                "lat_deg": profile.lat_deg,
                "lon_deg": profile.lon_deg,
                "max_depth_m": profile.max_depth_m,
                "measurement_year": profile.measurement_year,
                "prior_gst_c": profile.prior_gst_c,
                "mean_kappa_w_m_k": profile.mean_conductivity_w_m_k,
                "mean_grad_k_per_km": profile.mean_gradient_k_per_km,
                "n_samples": int(profile.depth_m.size),
            }
        )

    df = pd.DataFrame(rows).sort_values(["country", "site_id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} sites -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
