#!/usr/bin/env python3
"""Figure 5 -- geothermal-reservoir regime case study
(high Pe_T, low ℒ).

Output: ``outputs/figures/numerical/fig05_geothermal.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from figures.numerical._case_figure import render_case_figure


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nc", default="outputs/cases/geothermal.nc")
    p.add_argument(
        "--out",
        default="outputs/figures/numerical/fig05_geothermal.pdf",
    )
    args = p.parse_args(argv)
    # Geothermal has no ice; panel c shows |v_Darcy| instead.
    render_case_figure(
        Path(args.nc).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        panel_c_var="MISSING_FORCE_FALLBACK",
        panel_c_label=r"$\log_{10} |v_{\rm Darcy}|$ (m s$^{-1}$)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
