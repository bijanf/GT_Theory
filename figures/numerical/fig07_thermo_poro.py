#!/usr/bin/env python3
"""Figure 7 -- thermo-poro-coupled regime case study (low ℒ, high
Γ N_α).

Output: ``outputs/figures/numerical/fig07_thermo_poro.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures.numerical._case_figure import render_case_figure


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nc", default="outputs/cases/thermo_poro.nc")
    p.add_argument(
        "--out",
        default="outputs/figures/numerical/fig07_thermo_poro.pdf",
    )
    args = p.parse_args(argv)
    render_case_figure(
        Path(args.nc).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        panel_c_var="MISSING_FORCE_FALLBACK",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
