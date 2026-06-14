#!/usr/bin/env python3
"""Figure 8 -- regime-diagram synthesis: the four case studies
plotted on the theory paper's two regime diagrams.

Panel a: (Pe_T, ℒ) plane (heat transport: conduction vs advection
vs latent).
Panel b: (N_p, Γ N_α) plane (mass transport: pressure vs thermal
feedback).

Each case is a labelled marker; the regime boundary lines from
the theory paper are overlaid as dashed.

Output:
``outputs/figures/numerical/fig08_regime_placement.pdf``.
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


CASE_COLOURS = {
    "permafrost": "#1f77b4",  # blue
    "geothermal": "#d62728",  # red
    "arid_basin": "#7f7f7f",  # grey
    "thermo_poro": "#2ca02c",  # green
}
CASE_LABELS = {
    "permafrost": "permafrost",
    "geothermal": "geothermal",
    "arid_basin": "arid basin",
    "thermo_poro": "thermo-poro coupled",
}


def _panel_pe_l(ax, df: pd.DataFrame) -> None:
    # Cases with no through-flow have Pe_T -> 0, which a log axis cannot
    # render; clip such markers to just inside the left edge so they sit
    # in the far conduction-dominated corner (Pe_T << 1) rather than
    # vanishing.
    pe_floor = 1.4e-5
    for _, row in df.iterrows():
        c = row["case"]
        pe_disp = max(float(row["Pe_T"]), pe_floor)
        ax.scatter(
            pe_disp,
            row["L_calL"],
            s=80,
            color=CASE_COLOURS[c],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=CASE_LABELS[c],
        )
        ax.annotate(
            CASE_LABELS[c],
            (pe_disp, row["L_calL"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=6.5,
            color=CASE_COLOURS[c],
        )
    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=0.6, label=r"$\mathrm{Pe}_T = 1$")
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=0.6, label=r"$\mathcal{L} = 1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1e3)
    ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel(r"$\mathrm{Pe}_T = v_{\rm Darcy} L / \kappa$")
    ax.set_ylabel(r"$\mathcal{L} = \rho_i L_f \phi / [(\rho c)_{\rm eff} \Delta T]$")
    # Regime quadrant annotations.
    ax.text(
        0.04,
        50,
        "latent-heat\ndominated",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="left",
        va="center",
    )
    ax.text(
        50,
        50,
        "latent +\nadvection",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="center",
        va="center",
    )
    ax.text(
        50,
        0.005,
        "advection\ndominated",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="center",
        va="center",
    )
    ax.text(
        0.04,
        0.005,
        "conduction\ndominated",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="left",
        va="center",
    )
    ax.text(0.03, 0.97, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, which="both", linestyle=":", linewidth=0.3, alpha=0.5)


def _panel_np_gamma(ax, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        c = row["case"]
        ax.scatter(
            row["N_p"],
            row["Gamma_N_alpha"],
            s=80,
            color=CASE_COLOURS[c],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=CASE_LABELS[c],
        )
        ax.annotate(
            CASE_LABELS[c],
            (row["N_p"], row["Gamma_N_alpha"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=6.5,
            color=CASE_COLOURS[c],
        )
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.6, label=r"$\Gamma N_\alpha = 1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 1e-2)
    ax.set_ylim(1e-1, 1e3)
    ax.set_xlabel(r"$N_p = \phi \beta_w \Delta p$")
    ax.set_ylabel(r"$\Gamma N_\alpha = \alpha_w \Delta T / \beta_w \Delta p$")
    ax.text(
        2e-7,
        200,
        "thermo-poro\ncoupling matters",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="left",
        va="center",
    )
    ax.text(
        2e-7,
        0.3,
        "thermal\nfeedback\nnegligible",
        transform=ax.transData,
        fontsize=6,
        color="0.3",
        ha="left",
        va="center",
    )
    ax.text(0.03, 0.97, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    ax.grid(True, which="both", linestyle=":", linewidth=0.3, alpha=0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        default="outputs/cases/dimless_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="outputs/figures/numerical/fig08_regime_placement.pdf",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(Path(args.summary).expanduser().resolve())

    apply_nature_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.45),
    )
    _panel_pe_l(ax_a, df)
    _panel_np_gamma(ax_b, df)
    fig.tight_layout()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
