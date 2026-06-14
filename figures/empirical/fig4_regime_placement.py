#!/usr/bin/env python3
"""Figure 4 — regime placement: where the three supersites sit
relative to the Huang-Pollack borehole cloud in the
(Pe_T, L_calL) and (N_p, Γ·N_α) quadrants.

The framework's empirical claim is that the regime diagrams predict
*a priori* where coupling matters. This figure tests that claim by
showing that the three supersites — Umiujaq, Mont Terri, FORGE —
populate distinct quadrants from the Huang-Pollack background cloud,
which sits in the conduction-dominated corner.

Output: ``outputs/figures/empirical/fig4_regime_placement.pdf``.

Usage::

    python figures/empirical/fig4_regime_placement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gt_theory.catalog import load_catalog
from gt_theory.plotting.style import (
    NATURE_2COL_INCH,
    apply_nature_style,
)
from gt_theory.theory.dimless import (
    SiteDimensionlessParams,
    compute_site_dimless,
    default_params_from_site,
)


# Pe_T floor for log-scale visibility; the conduction-only Huang-Pollack
# sites have v_darcy = 0 by construction, so they map to Pe_T = 0 which
# log axes cannot show. We clip to PE_FLOOR and jitter horizontally so
# the cloud is visible.
PE_FLOOR = 1.0e-3

SUPERSITE_DEFS = {
    # Umiujaq: shallow silt aquifer talik. Pe_T from Fortier 2023 estimated
    # Darcy velocity ~ 1e-7 m/s over a 10 m talik thickness.
    "umiujaq": {
        "label": "Umiujaq",
        "config": "data/supersite_umiujaq/site_config.yaml",
        "v_darcy_m_s": 1.0e-7,
        "L_m": 10.0,
        "kappa_m2_s": 7.2e-7,  # silt aquifer thermal diffusivity
        "delta_T_K": 15.0,  # surface-air seasonal amplitude
        "color": "#1f77b4",
        "marker": "o",
    },
    # Mont Terri HE-D: indurated low-K clay heated 15 -> 100 degC.
    # K_zz so low (~1e-13 m/s) that Pe_T is essentially zero, but Γ is large
    # because of the 85 K thermal excursion.
    "mont_terri": {
        "label": "Mont Terri HE-D",
        "config": "data/supersite_mont_terri/site_config.yaml",
        "v_darcy_m_s": 1.0e-12,
        "L_m": 2.0,
        "kappa_m2_s": 9.1e-7,
        "delta_T_K": 85.0,
        "color": "#d62728",
        "marker": "s",
    },
    # Utah FORGE: 60 -> 225 degC injection step at 2-3 km depth in granite.
    # High-Pe_T, high-Γ corner; the regime-diagram anchor.
    "forge": {
        "label": "Utah FORGE",
        "config": "data/supersite_forge/site_config.yaml",
        "v_darcy_m_s": 1.0e-5,  # injection-driven order-of-magnitude estimate
        "L_m": 3300.0,
        "kappa_m2_s": 1.24e-6,
        "delta_T_K": 165.0,
        "color": "#2ca02c",
        "marker": "^",
    },
}


def _supersite_dimless(definition: dict) -> tuple[float, float, float, float]:
    cfg_path = Path(definition["config"]).resolve()
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Use the first talik / reservoir layer for porosity.
    layer = cfg["physics"]["layers"][0]
    fluid = cfg["physics"]["fluid"]
    params = SiteDimensionlessParams(
        L_m=definition["L_m"],
        kappa_m2_s=definition["kappa_m2_s"],
        porosity=float(layer["porosity"]),
        delta_T_K=definition["delta_T_K"],
        v_darcy_m_s=definition["v_darcy_m_s"],
        alpha_w_inv_K=float(fluid["alpha_w_per_K"]),
        beta_w_inv_Pa=float(fluid["beta_w_per_Pa"]),
        mu_Pa_s=float(fluid["mu_Pa_s"]),
        rho_w_kg_m3=float(fluid["rho_w_kg_per_m3"]),
    )
    d = compute_site_dimless(params)
    pe_t = max(d.Pe_T, PE_FLOOR)
    return pe_t, d.L_calL, d.N_p, d.Gamma * d.N_alpha


def _hp_cloud(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Pe_T, L_calL, N_p, Γ·N_α) for every site in the loaded
    catalog. Pe_T is jittered horizontally below 1e-2 because the
    catalog records do not carry v_darcy."""
    cat = load_catalog()
    pe_t = []
    l_cal = []
    n_p = []
    g_na = []
    for site_id, meta in cat.sites.items():
        params = default_params_from_site(
            lat_deg=meta.lat_deg,
            delta_T_K=2.0,  # generic surface-warming amplitude over the H-P era
        )
        d = compute_site_dimless(params)
        # Jitter Pe_T below the floor so the cloud is visible on log axis.
        pe = max(d.Pe_T, PE_FLOOR) * rng.uniform(0.5, 1.5)
        pe_t.append(pe)
        l_cal.append(d.L_calL)
        n_p.append(d.N_p)
        g_na.append(d.Gamma * d.N_alpha)
    return (np.asarray(pe_t), np.asarray(l_cal), np.asarray(n_p), np.asarray(g_na))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs/figures/empirical/fig4_regime_placement.pdf",
    )
    args = parser.parse_args(argv)

    apply_nature_style()
    rng = np.random.default_rng(20260522)

    pe_hp, l_hp, np_hp, ga_hp = _hp_cloud(rng)

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.45),
    )

    # --- Pe_T vs L_calL panel ---
    ax_l.scatter(
        pe_hp,
        l_hp,
        s=12,
        alpha=0.45,
        color="0.55",
        edgecolor="none",
        label="Huang-Pollack cloud",
    )
    for key, defn in SUPERSITE_DEFS.items():
        pe, lc, _, _ = _supersite_dimless(defn)
        ax_l.scatter(
            pe,
            lc,
            s=80,
            marker=defn["marker"],
            color=defn["color"],
            edgecolor="white",
            linewidth=1.0,
            label=defn["label"],
            zorder=5,
        )
        ax_l.scatter(
            pe,
            lc,
            s=80,
            marker=defn["marker"],
            color="none",
            edgecolor="black",
            linewidth=0.6,
            zorder=6,
        )
    ax_l.set_xscale("log")
    ax_l.set_xlabel(r"$\mathrm{Pe}_T$")
    ax_l.set_ylabel(r"$\mathcal{L}$  (latent-heat number)")
    ax_l.axvline(1.0, linestyle="--", color="0.7", linewidth=0.6)
    ax_l.axhline(1.0, linestyle="--", color="0.7", linewidth=0.6)
    ax_l.text(0.02, 0.97, "a", transform=ax_l.transAxes, fontsize=7, fontweight="bold", va="top")
    ax_l.legend(loc="best", frameon=False)
    ax_l.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    # --- N_p vs Γ·N_α panel ---
    ax_r.scatter(
        np_hp,
        ga_hp,
        s=12,
        alpha=0.45,
        color="0.55",
        edgecolor="none",
    )
    label_offsets = {
        "umiujaq": (1.6, 1.6),
        "mont_terri": (0.55, 1.7),
        "forge": (1.6, 0.55),
    }
    for key, defn in SUPERSITE_DEFS.items():
        _, _, npv, gv = _supersite_dimless(defn)
        # Two-pass scatter for the white outline trick: filled marker first,
        # then unfilled black outline on top.
        ax_r.scatter(
            npv,
            gv,
            s=80,
            marker=defn["marker"],
            color=defn["color"],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        ax_r.scatter(
            npv,
            gv,
            s=80,
            marker=defn["marker"],
            color="none",
            edgecolor="black",
            linewidth=0.6,
            zorder=6,
        )
        dx, dy = label_offsets.get(key, (1.5, 1.5))
        ax_r.annotate(
            defn["label"],
            xy=(npv, gv),
            xytext=(npv * dx, gv * dy),
            arrowprops=dict(arrowstyle="-", color="0.35", linewidth=0.4),
            color="0.15",
        )
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel(r"$N_p$")
    ax_r.set_ylabel(r"$\Gamma \cdot N_\alpha$")
    ax_r.axvline(1.0, linestyle="--", color="0.7", linewidth=0.6)
    ax_r.axhline(1.0, linestyle="--", color="0.7", linewidth=0.6)
    ax_r.text(0.02, 0.97, "b", transform=ax_r.transAxes, fontsize=7, fontweight="bold", va="top")
    ax_r.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
