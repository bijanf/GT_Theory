#!/usr/bin/env python3
"""Extended Data Figure 1 — Synthetic-twin GST recovery.

Demonstrates that the hierarchical bootstrap-Tikhonov inversion
recovers known GST histories from forward-simulated borehole profiles.
Four panels, Nature 1-column width:

  (a) Step truth: +1.5 K at the most-recent bin, zero elsewhere.
  (b) Ramp truth: linear 0 -> +1.2 K over the recent two bins.
  (c) Sinusoid truth: cos(2 pi t / 200 yr), amplitude 0.7 K.
  (d) Coverage diagnostic: across 50 random smooth truths, fraction
      that fall inside the 90% posterior credible interval per bin.

Each panel shows truth + posterior median + 90% credible band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gt_theory.inversion import (
    build_forward_operator,
    default_bin_edges_yr,
    detrend_geothermal,
    invert_posterior,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_profile(
    truth_K: np.ndarray,
    z: np.ndarray,
    edges: np.ndarray,
    *,
    T0: float,
    dTdz: float,
    noise_K: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    G = build_forward_operator(z, edges, 1.0e-6)
    dT = G @ truth_K
    T = T0 + dTdz * z + dT + rng.normal(0.0, noise_K, size=z.size)
    return z, T


def _invert_once(z: np.ndarray, T_obs: np.ndarray, edges: np.ndarray, rng: np.random.Generator):
    dT, _T0, _dTdz = detrend_geothermal(z, T_obs, z_steady_min_m=300.0)
    return invert_posterior(
        z, dT, bin_edges_yr=edges, sigma_T=0.05, n_bootstrap=200, lam=0.05, rng=rng
    )


def build_figure(out_path: Path) -> None:
    edges = default_bin_edges_yr()
    centres = 0.5 * (edges[:-1] + edges[1:])
    z = np.linspace(20.0, 600.0, 30)

    truths = {
        "step": np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "ramp": np.array([1.2, 0.8, 0.4, 0.2, 0.0, 0.0]),
        "sinusoid": 0.7 * np.cos(2.0 * np.pi * centres / 200.0),
    }

    rng = np.random.default_rng(20260522)
    posteriors = {}
    for name, tr in truths.items():
        _, T_obs = _synthetic_profile(tr, z, edges, T0=5.0, dTdz=0.025, noise_K=0.05, rng=rng)
        posteriors[name] = _invert_once(z, T_obs, edges, rng=rng)

    # Coverage over 50 random smooth truths.
    n_seeds = 50
    hits = np.zeros(edges.size - 1)
    for seed in range(n_seeds):
        srng = np.random.default_rng(seed + 100)
        recent = float(srng.uniform(0.2, 2.0))
        decay = float(srng.uniform(0.4, 0.7))
        tr = recent * decay ** np.arange(edges.size - 1, dtype=float)
        _, T_obs = _synthetic_profile(tr, z, edges, T0=5.0, dTdz=0.025, noise_K=0.05, rng=srng)
        post = _invert_once(z, T_obs, edges, rng=rng)
        for b in range(edges.size - 1):
            if post.ci_lo[b] <= tr[b] <= post.ci_hi[b]:
                hits[b] += 1
    coverage = hits / n_seeds

    from gt_theory.plotting import NATURE_1COL_INCH, apply_nature_style

    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(NATURE_1COL_INCH * 2.2, NATURE_1COL_INCH * 1.6))
    panels = [
        ("a", "step", "step at 0-25 yr"),
        ("b", "ramp", "exponential decay"),
        ("c", "sinusoid", "200-yr sinusoid"),
    ]
    for ax, (lbl, key, title) in zip(axes.ravel()[:3], panels, strict=True):
        tr = truths[key]
        post = posteriors[key]
        ax.fill_between(
            centres, post.ci_lo, post.ci_hi, color="#0066cc", alpha=0.25, label="90% CI"
        )
        ax.plot(centres, post.median, color="#0066cc", linewidth=1.0, label="posterior median")
        ax.plot(centres, tr, color="black", linewidth=1.0, linestyle="--", label="truth")
        ax.set_xscale("log")
        ax.set_xlabel("years before present")
        ax.set_ylabel(r"$\Delta$GST (K)")
        ax.set_title(f"{lbl}   {title}", loc="left", weight="bold")
        ax.grid(alpha=0.3, linewidth=0.3)
        if lbl == "a":
            ax.legend(loc="upper right", frameon=False, fontsize=5.5)

    ax_d = axes[1, 1]
    ax_d.axhline(0.90, color="black", linewidth=0.4, alpha=0.5, label="nominal 90%")
    ax_d.bar(np.arange(coverage.size), coverage, color="#0066cc", edgecolor="white", alpha=0.85)
    ax_d.set_xticks(np.arange(coverage.size))
    ax_d.set_xticklabels(
        [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(coverage.size)],
        rotation=30,
        fontsize=5,
    )
    ax_d.set_ylabel("frequentist coverage")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title(f"d   90% CI coverage over {n_seeds} seeds", loc="left", weight="bold")
    ax_d.legend(loc="lower right", frameon=False, fontsize=5.5)
    ax_d.grid(axis="y", alpha=0.25, linewidth=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    build_figure(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
