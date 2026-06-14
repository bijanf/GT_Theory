#!/usr/bin/env python3
"""Concatenate per-site inversion parquet summaries into one tidy table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    frames = []
    for p in args.inputs:
        frames.append(pd.read_parquet(p))
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(
        f"gathered {len(args.inputs)} site files, {len(out)} rows -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
