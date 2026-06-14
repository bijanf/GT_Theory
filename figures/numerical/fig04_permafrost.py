#!/usr/bin/env python3
"""Figure 4 -- permafrost regime case study (high ℒ, moderate Γ N_α).

Output: ``outputs/figures/numerical/fig04_permafrost.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures.numerical._case_figure import render_case_figure


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nc", default="outputs/cases/permafrost.nc")
    p.add_argument(
        "--out",
        default="outputs/figures/numerical/fig04_permafrost.pdf",
    )
    args = p.parse_args(argv)
    render_case_figure(
        Path(args.nc).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        panel_c_var="S_i",
        panel_c_label=r"$S_i$",
        panel_c_cmap="Blues",
        panel_c_vmin=0.0,
        panel_c_vmax=1.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
