#!/usr/bin/env python3
"""Figure A — Direct profile contrast (conduction vs conduction + Darcy
advection) at four anomalous high-gradient sites.

Selected from the Huang-Pollack archive by exceptional steady-state
thermal gradient (>50 K km^{-1}), indicating either an active
geothermal basin or a strongly advective regime where the
conduction-only kernel of Section 7 is expected to fail.  The four
sites are:

  - US-GSST-3       (38.37 N, 118.81 W, 121 K/km, Great Basin, NV)
  - CN-XZ-naqu903   (31.38 N,  91.88 E,  94 K/km, Tibetan Plateau)
  - US-KEN95        (43.80 N,  99.80 W,  84 K/km, Black Hills, SD)
  - US-MT2-14       (46.72 N, 112.30 W,  72 K/km, Montana)

For each site, we (i) load the observed temperature profile and the
inversion's recovered binned GST history, (ii) reconstruct a piecewise-
constant 800-year surface forcing from those bins, and (iii) run the
column_1d solver twice on this forcing: once with v_darcy = 0 (pure
conduction) and once with v_darcy = v_fit minimising the squared
residual against the observed anomaly in the 50-200 m window, over
the physical range v in [1e-9, 1e-7] m s^{-1}.  The advection-added
curve is the simplest available proxy for the Pe_T-> non-zero regime
the framework predicts.

When ``--coupled`` is set (default), a third curve is overlaid: the
fully-coupled T-p solver (``run_column_coupled``) is run with
``gamma_n_alpha_scale = 1`` and ``K_zz`` set to the value implied by
the Pe_T best-fit v_darcy (``K_zz = v_best * mu / (rho_w g)``).  This
isolates the marginal contribution of the Gamma N_alpha cross-coupling:
the dashed black curve is conduction only, the orange curve is
conduction + prescribed advection (Pe_T sweep), and the red curve adds
the two-way thermal-expansion feedback to the same advective regime.
The per-site stderr summary reports all three RMS values so the reader
can see whether Gamma N_alpha tightens the fit beyond Pe_T alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_theory.solvers import run_column_1d, run_column_coupled
from gt_theory.theory.dimless import G_EARTH, MU_WATER, RHO_WATER

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR_S: float = 365.25 * 86400.0

SITE_IDS = ["US-GSST-3", "CN-XZ-naqu903", "US-KEN95", "US-MT2-14"]
SITE_LABELS = {
    "US-GSST-3": "US-GSST-3 (Great Basin, NV; 121 K/km)",
    "CN-XZ-naqu903": "CN-XZ-naqu903 (Tibetan Plateau; 94 K/km)",
    "US-KEN95": "US-KEN95 (Black Hills, SD; 84 K/km)",
    "US-MT2-14": "US-MT2-14 (Montana; 72 K/km)",
}


def _load_site(subset_dir: Path, site_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof = pd.read_parquet(subset_dir / "profiles" / f"{site_id}.parquet")
    inv = pd.read_parquet(subset_dir / "inversions" / f"{site_id}.parquet")
    return prof, inv


def _piecewise_constant_gst(inv: pd.DataFrame, dt_yr: float) -> np.ndarray:
    """Reconstruct a piecewise-constant monthly GST series from the
    inversion's binned recovery, running from the oldest bin's start
    backwards in time up to the present.  Returns a 1-D array; first
    element is the oldest sample, last element is the present."""
    edges = np.concatenate(
        [
            inv["bin_edge_young_yr"].to_numpy(dtype=float),
            inv["bin_edge_old_yr"].to_numpy(dtype=float)[-1:],
        ]
    )
    s = inv["median_K"].to_numpy(dtype=float)
    duration_yr = float(edges[-1])
    n_steps = int(round(duration_yr / dt_yr)) + 1
    t_yr_before_present = np.linspace(duration_yr, 0.0, n_steps)
    gst = np.zeros_like(t_yr_before_present)
    for k in range(len(s)):
        in_bin = (t_yr_before_present >= edges[k]) & (t_yr_before_present < edges[k + 1])
        gst[in_bin] = s[k]
    return gst


def _forward(
    gst_series: np.ndarray,
    *,
    kappa: float,
    v_darcy: float,
    depth_max_m: float,
    dz_m: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run column_1d on the piecewise-constant forcing.  Returns (z,
    final_T_profile)."""
    duration_s = (gst_series.size - 1) * dt_s
    res = run_column_1d(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        kappa=kappa,
        sat=gst_series,
        n_winter=1.0,
        v_darcy=v_darcy,
        T_init=0.0,
    )
    return res.z, res.T[-1]


def _forward_coupled(
    gst_series: np.ndarray,
    *,
    kappa: float,
    K_zz: float,
    depth_max_m: float,
    dz_m: float,
    dt_s: float,
    rho_c_eff: float = 2.5e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the coupled T-p solver on the same forcing as ``_forward`` with
    ``gamma_n_alpha_scale = 1``.  Returns (z, final_T_profile)."""
    lam_th = kappa * rho_c_eff
    duration_s = (gst_series.size - 1) * dt_s
    res = run_column_coupled(
        depth_max_m=depth_max_m,
        dz_m=dz_m,
        duration_s=duration_s,
        dt_s=dt_s,
        lambda_thermal=lam_th,
        rho_c_eff=rho_c_eff,
        K_zz=K_zz,
        mu=MU_WATER,
        rho_w=RHO_WATER,
        g=G_EARTH,
        gamma_n_alpha_scale=1.0,
        sat=gst_series,
        p_top=0.0,
        T_init=0.0,
        p_init=0.0,
    )
    return res.z, res.T[-1]


def _fit_v_darcy(
    gst_series: np.ndarray,
    z_obs: np.ndarray,
    dT_obs: np.ndarray,
    *,
    kappa: float,
    depth_max_m: float,
    dz_m: float,
    dt_s: float,
    z_fit_min_m: float = 50.0,
    z_fit_max_m: float = 200.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Grid-search v_darcy in [1e-9, 1e-7] m s^-1 (positive downward)
    to minimise the L2 residual against the observed anomaly in the
    [z_fit_min_m, z_fit_max_m] window.  Returns (v_best, z, dT_pred)."""
    candidates = np.concatenate(
        [
            np.array([0.0]),
            np.geomspace(1e-10, 1e-7, 20),
            -np.geomspace(1e-10, 1e-7, 20),
        ]
    )
    best_v = 0.0
    best_loss = np.inf
    best_dT = None
    best_z = None
    for v in candidates:
        z_sim, T_end = _forward(
            gst_series,
            kappa=kappa,
            v_darcy=float(v),
            depth_max_m=depth_max_m,
            dz_m=dz_m,
            dt_s=dt_s,
        )
        pred_at_obs = np.interp(z_obs, z_sim, T_end, left=np.nan, right=np.nan)
        mask = (z_obs >= z_fit_min_m) & (z_obs <= z_fit_max_m) & np.isfinite(pred_at_obs)
        if mask.sum() < 3:
            continue
        loss = float(np.sum((dT_obs[mask] - pred_at_obs[mask]) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_v = float(v)
            best_dT = T_end
            best_z = z_sim
    assert best_z is not None
    assert best_dT is not None
    return best_v, best_z, best_dT


def build_figure(subset_dir: Path, out_path: Path, *, include_coupled: bool = True) -> None:
    from gt_theory.plotting import NATURE_2COL_INCH, apply_nature_style

    apply_nature_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(NATURE_2COL_INCH, NATURE_2COL_INCH * 0.8),
        sharex=False,
        sharey=False,
    )
    fits: dict[str, dict] = {}

    for ax, sid in zip(axes.ravel(), SITE_IDS, strict=True):
        prof, inv = _load_site(subset_dir, sid)
        z_obs = prof["depth_m"].to_numpy(dtype=float)
        T = prof["temperature_c"].to_numpy(dtype=float)
        T0 = float(inv["T0_K"].iloc[0])
        dTdz = float(inv["dTdz_K_per_m"].iloc[0])
        dT_obs = T - (T0 + dTdz * z_obs)
        kappa = float(inv["kappa_median"].iloc[0])

        dt_yr = 1.0
        dt_s = dt_yr * YEAR_S
        depth_max_m = max(z_obs.max() + 50.0, 250.0)
        dz_m = 5.0

        gst_series = _piecewise_constant_gst(inv, dt_yr)

        # v = 0 (pure conduction).
        z_cond, T_cond = _forward(
            gst_series,
            kappa=kappa,
            v_darcy=0.0,
            depth_max_m=depth_max_m,
            dz_m=dz_m,
            dt_s=dt_s,
        )
        # Best-fit v_darcy.
        v_best, z_adv, T_adv = _fit_v_darcy(
            gst_series,
            z_obs,
            dT_obs,
            kappa=kappa,
            depth_max_m=depth_max_m,
            dz_m=dz_m,
            dt_s=dt_s,
        )

        # Residual reductions in the 50-200 m window.  Define the
        # helper with the loop variables captured as defaults so ruff
        # B023 is satisfied (each iteration binds the current site's
        # data into the closure).
        def _rms(z_sim, T_sim, z_o=z_obs, dT_o=dT_obs):
            pred = np.interp(z_o, z_sim, T_sim, left=np.nan, right=np.nan)
            mask = (z_o >= 50.0) & (z_o <= 200.0) & np.isfinite(pred)
            return (
                float(np.sqrt(np.mean((dT_o[mask] - pred[mask]) ** 2)))
                if mask.sum() > 1
                else np.nan
            )

        rms_cond = _rms(z_cond, T_cond)
        rms_adv = _rms(z_adv, T_adv)

        # Coupled forward at the K_zz that the best-fit v_darcy implies.
        # v_Darcy = (K_zz / mu) * (rho_w g)  =>  K_zz = |v| * mu / (rho_w g)
        if include_coupled:
            K_zz_eff = abs(v_best) * MU_WATER / (RHO_WATER * G_EARTH)
            K_zz_eff = max(K_zz_eff, 1.0e-20)
            z_cpl, T_cpl = _forward_coupled(
                gst_series,
                kappa=kappa,
                K_zz=K_zz_eff,
                depth_max_m=depth_max_m,
                dz_m=dz_m,
                dt_s=dt_s,
            )
            rms_cpl = _rms(z_cpl, T_cpl)
        else:
            K_zz_eff = float("nan")
            rms_cpl = float("nan")
            z_cpl = None
            T_cpl = None

        fits[sid] = {
            "v_best_m_s": v_best,
            "K_zz_eff_m2": K_zz_eff,
            "rms_cond_K": rms_cond,
            "rms_adv_K": rms_adv,
            "rms_coupled_K": rms_cpl,
        }

        ax.scatter(
            dT_obs,
            z_obs,
            s=12,
            c="#003366",
            edgecolor="white",
            linewidths=0.4,
            label="observed",
            zorder=5,
        )
        ax.plot(
            T_cond,
            z_cond,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label=rf"$v_\mathrm{{Darcy}}=0$ (RMS={rms_cond:.2f} K)",
        )
        ax.plot(
            T_adv,
            z_adv,
            color="#cc7700",
            linewidth=1.2,
            label=rf"$v={v_best:.1e}$ m/s (RMS={rms_adv:.2f} K)",
        )
        if include_coupled and z_cpl is not None and T_cpl is not None:
            ax.plot(
                T_cpl,
                z_cpl,
                color="#cc0000",
                linewidth=1.2,
                linestyle="-",
                label=rf"coupled $\Gamma N_\alpha$ (RMS={rms_cpl:.2f} K)",
            )
        ax.axvline(0.0, color="black", linewidth=0.3, alpha=0.4)
        ax.set_ylim(z_obs.max() + 20, 0)
        ax.set_xlabel(r"$\Delta T(z)$ (K)")
        ax.set_ylabel("Depth (m)")
        ax.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.4)
        ax.set_title(SITE_LABELS[sid], loc="left", weight="bold", fontsize=6)
        ax.grid(alpha=0.25, linewidth=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Per-site best-fit Darcy velocity and RMS reduction:", file=sys.stderr)
    for sid, f in fits.items():
        delta_pe = (f["rms_cond_K"] - f["rms_adv_K"]) / max(f["rms_cond_K"], 1e-6) * 100.0
        rms_cpl = f["rms_coupled_K"]
        if np.isfinite(rms_cpl):
            delta_cpl = (f["rms_adv_K"] - rms_cpl) / max(f["rms_adv_K"], 1e-6) * 100.0
            print(
                f"  {sid:18s}  v={f['v_best_m_s']:+.1e} m/s  "
                f"K_zz={f['K_zz_eff_m2']:.1e} m^2  "
                f"RMS cond={f['rms_cond_K']:.3f} K  adv={f['rms_adv_K']:.3f} K  "
                f"coupled={rms_cpl:.3f} K  "
                f"Pe_T->{delta_pe:+.1f}%  GammaN_alpha->{delta_cpl:+.1f}%",
                file=sys.stderr,
            )
        else:
            print(
                f"  {sid:18s}  v={f['v_best_m_s']:+.1e} m/s  "
                f"RMS cond={f['rms_cond_K']:.3f} K  adv={f['rms_adv_K']:.3f} K  "
                f"reduction={delta_pe:+.1f}%",
                file=sys.stderr,
            )
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=REPO_ROOT / "outputs" / "full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-coupled",
        action="store_true",
        help="Skip the coupled-solver curve (faster smoke run; default off).",
    )
    args = parser.parse_args(argv)
    build_figure(args.subset_dir, args.out, include_coupled=not args.no_coupled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
