#!/usr/bin/env python3
"""Real-borehole ingestion: resolve a site_id via the curated catalog to a
Huang-Pollack archive file, parse it, and emit a uniform parquet for the
downstream solve/invert rules.

This is the laptop / HPC ingest rule for smoke-10 and full subsets.  It
expects the GT_THEORY_BOREHOLE_ROOT environment variable to point at the
local archive (or the catalog's default_data_root to be valid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gt_theory.catalog import load_catalog, resolve_data_root
from gt_theory.io import find_borehole_file, parse_huang_pollack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", required=True, type=str)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    cat = load_catalog(args.catalog)
    data_root = resolve_data_root(cat)
    try:
        archive_path = find_borehole_file(args.site_id, data_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    profile = parse_huang_pollack(archive_path)
    df = pd.DataFrame(
        {
            "depth_m": profile.depth_m,
            "temperature_c": profile.temperature_c,
        }
    )
    df.attrs["site_id"] = profile.site_id
    df.attrs["lat_deg"] = profile.lat_deg
    df.attrs["lon_deg"] = profile.lon_deg
    df.attrs["max_depth_m"] = profile.max_depth_m
    df.attrs["measurement_year"] = profile.measurement_year
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out} ({profile.country})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
