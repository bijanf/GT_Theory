#!/usr/bin/env python3
"""Figure ED6 — R17 W2: Zhang 2005 winter n-factor scenarios for the
F3 boreal/equator subsurface warming ratio.

Output: ``outputs/figures/empirical/fig_ed6_f3_nfactor.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)
from gt_theory.theory.zhang_nfactor import SCENARIOS, nfactor_series


SCEN_COLOR = {
    "raw": "#7f7f7f",
    "constant_heavy": "#1f77b4",
    "declining_snow_insulation": "#2ca02c",
    "no_insulation": "#d62728",
}
SCEN_LABEL = {
    "raw": "raw (no n-factor)",
    "constant_heavy": r"$n_w = 0.5$ (heavy snow)",
    "declining_snow_insulation": r"$n_w$: 0.6 → 0.9 (snow loss)",
    "no_insulation": r"$n_w = 1$ (no insulation)",
}


def _band_ratio(values: np.ndarray, lats: np.ndarray) -> float:
    abs_lat = np.abs(lats)
    boreal = (abs_lat >= 50.0) & np.isfinite(values)
    equator = (abs_lat <= 20.0) & np.isfinite(values)
    if boreal.sum() == 0 or equator.sum() == 0:
        return float("nan")
    b = float(np.mean(values[boreal]))
    e = float(np.mean(values[equator]))
    return b / e if abs(e) > 1.0e-6 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in",
        dest="in_path",
        default="outputs/global/f3_nfactor_corrected.parquet",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig_ed6_f3_nfactor.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_parquet(Path(args.in_path).expanduser().resolve())

    apply_nature_style()
    fig = plt.figure(figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.42))

    # Panel a: n_w(t) scenarios.
    ax_a = fig.add_subplot(1, 3, 1)
    years = np.arange(1900, 2025)
    for scen in SCENARIOS:
        nw = nfactor_series(scenario=scen, years=years)
        ax_a.plot(years, nw, linewidth=1.4, color=SCEN_COLOR[scen], label=SCEN_LABEL[scen])
    ax_a.set_xlabel("year")
    ax_a.set_ylabel(r"$n_w(t)$")
    ax_a.set_ylim(0.0, 1.1)
    ax_a.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_a.legend(loc="lower right", frameon=False)
    ax_a.text(0.03, 0.97, "a", transform=ax_a.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel b: boreal/equator ratio per scenario.
    ax_b = fig.add_subplot(1, 3, 2)
    lats = df["lat_deg"].to_numpy()
    scens = ("raw",) + SCENARIOS
    ratios = []
    for scen in scens:
        vals = df[f"delta_gst_{scen}"].to_numpy()
        ratios.append(_band_ratio(vals, lats))
    bar_colors = [SCEN_COLOR[s] for s in scens]
    ax_b.bar(
        range(len(scens)),
        ratios,
        color=bar_colors,
        edgecolor="0.25",
        linewidth=0.4,
    )
    ax_b.axhline(1.49, color="#c0392b", linewidth=1.0, linestyle="--", label="target (SAT) = 1.49")
    ax_b.axhline(
        0.94, color="0.4", linewidth=1.0, linestyle=":", label="theory-paper observed = 0.94"
    )
    # Use plain text scenario labels for the x-ticks (mathtext in the
    # full labels of the legend breaks when split on whitespace).
    short_labels = {
        "raw": "raw",
        "constant_heavy": "heavy snow",
        "declining_snow_insulation": "snow loss",
        "no_insulation": "no insulation",
    }
    ax_b.set_xticks(range(len(scens)))
    ax_b.set_xticklabels(
        [short_labels[s] for s in scens],
        rotation=20,
        ha="right",
    )
    ax_b.set_ylabel(r"boreal / equator $\Delta T_{\rm GST}$ ratio")
    ax_b.set_ylim(0.0, 1.7)
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax_b.text(0.03, 0.97, "b", transform=ax_b.transAxes, fontsize=7, fontweight="bold", va="top")

    # Panel c: per-site Δ vs latitude scatter.
    ax_c = fig.add_subplot(1, 3, 3)
    for scen in ("raw", "constant_heavy", "declining_snow_insulation"):
        ax_c.scatter(
            df["lat_deg"],
            df[f"delta_gst_{scen}"],
            s=6,
            alpha=0.5,
            color=SCEN_COLOR[scen],
            edgecolor="none",
            label=SCEN_LABEL[scen],
        )
    ax_c.axhline(0.0, color="0.6", linewidth=0.4, linestyle=":")
    ax_c.set_xlabel("latitude (deg)")
    ax_c.set_ylabel(r"$\Delta T_{\rm GST}^{\rm 10m}$ recent - baseline  (K)")
    ax_c.legend(loc="upper left", frameon=False)
    ax_c.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax_c.text(0.03, 0.97, "c", transform=ax_c.transAxes, fontsize=7, fontweight="bold", va="top")

    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}  ({len(df)} sites)")
    for scen, r in zip(scens, ratios):
        print(f"  {scen:30s} ratio = {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
