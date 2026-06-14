"""Gaussian-process emulator over the coupled T-p forward operator.

The raw ``run_column_coupled`` call is ~100 ms per forward; an MCMC
inversion at one site needs ~10^4 forwards, so the global 951-site
batch is ~10^9 forwards = infeasible. This module trains a
PCA-reduced GP ensemble that amortises one forward to ~100 us, a
1000x speedup, with median accuracy well below the observational
noise floor.

Pipeline:

1. Sample ``n_train + n_test`` parameter vectors from the prior
   support via Latin Hypercube (``scipy.stats.qmc.LatinHypercube``).
2. Run ``run_column_coupled`` on a small training column
   (depth_max=6 m, dz=0.5 m, duration=2 yr, dt=5 d) at each
   parameter point; extract ``T(z_obs, t_obs)`` and
   ``p(z_piezo, t_obs)`` flattened into one fixed-length output
   vector per sample.
3. PCA-reduce the training-set outputs to ``n_pca`` components.
4. Fit one ``GaussianProcessRegressor`` per PCA component, each
   over the 5-D input space, with anisotropic RBF + WhiteKernel.
5. ``predict`` inverts the PCA basis to return the full
   ``(n_t_obs, n_z_obs)`` T field and ``(n_t_obs,)`` p series.

Reduced parameter vector (must match
``inversion.bayes_coupled.PARAM_NAMES_DEFAULT``):

   (log10_K_hyd, porosity, lambda_th, gst_offset, gamma_n_alpha_scale)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from gt_theory.inversion.bayes_coupled import coupled_forward

YEAR_S = 365.25 * 86400.0
DAY_S = 86400.0


# Prior box used to draw the training Latin-hypercube. Matches the
# uniform supports in ``inversion.bayes_coupled.default_log_prior``.
PARAM_BOX_LO = np.array([-7.0, 0.30, 1.20, -1.0, 0.0])
PARAM_BOX_HI = np.array([-4.0, 0.50, 2.50, +1.0, 2.0])
PARAM_NAMES = (
    "log10_K_hyd_m_s",
    "porosity",
    "lambda_th_W_m_K",
    "gst_offset_K",
    "gamma_n_alpha_scale",
)


@dataclass(frozen=True)
class EmulatorSpec:
    """Solver grid + observation manifest baked into the emulator."""

    depth_max_m: float = 6.0
    dz_m: float = 0.5
    duration_s: float = 2.0 * YEAR_S
    dt_s: float = 5.0 * DAY_S
    z_obs_m: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    t_obs_yr: tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
    z_piezo_m: float = 5.0
    sat_annual_mean_C: float = -3.0
    sat_seasonal_amp_C: float = 10.0
    sat_seed: int = 20260523

    @property
    def t_obs_s(self) -> np.ndarray:
        return np.asarray(self.t_obs_yr, dtype=float) * YEAR_S

    @property
    def z_obs(self) -> np.ndarray:
        return np.asarray(self.z_obs_m, dtype=float)

    @property
    def sat(self) -> np.ndarray:
        nt = int(round(self.duration_s / self.dt_s)) + 1
        t = np.arange(nt) * self.dt_s
        return self.sat_annual_mean_C + self.sat_seasonal_amp_C * np.sin(2.0 * np.pi * t / YEAR_S)


class CoupledForwardEmulator:
    """PCA-reduced GP ensemble over the coupled forward operator.

    Trained with :meth:`fit` from a Latin-hypercube training set;
    queried with :meth:`predict_T_p(theta)` which returns the
    ``(n_t_obs, n_z_obs)`` T field and ``(n_t_obs,)`` p series.

    Attributes
    ----------
    spec : EmulatorSpec
        Frozen column + observation manifest.
    pca : sklearn.decomposition.PCA
    gps : list[GaussianProcessRegressor]
        One GP per retained PCA component.
    """

    def __init__(self, spec: EmulatorSpec) -> None:
        self.spec = spec
        self.pca: PCA | None = None
        self.gps: list[GaussianProcessRegressor] = []
        self._n_T: int = len(spec.z_obs_m) * len(spec.t_obs_yr)
        self._n_p: int = len(spec.t_obs_yr)
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    # ------------------------------------------------------------- training

    def _forward_kwargs(self) -> dict:
        return {
            "depth_max_m": self.spec.depth_max_m,
            "dz_m": self.spec.dz_m,
            "duration_s": self.spec.duration_s,
            "dt_s": self.spec.dt_s,
            "sat": self.spec.sat,
            "t_obs_s": self.spec.t_obs_s,
            "z_obs_m": self.spec.z_obs,
            "z_piezo_m": self.spec.z_piezo_m,
        }

    def _run_one(self, theta: np.ndarray) -> np.ndarray:
        T_pred, p_pred = coupled_forward(theta, **self._forward_kwargs())
        # Pressure varies over many orders of magnitude across the prior
        # box; an asinh transform keeps the sign and compresses dynamic
        # range so PCA can find meaningful directions.
        p_compressed = np.arcsinh(p_pred / 1.0e3)
        return np.concatenate([T_pred.ravel(), p_compressed.ravel()])

    def _decompress(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a stacked output vector into T (K) and p (Pa)."""
        T = y[: self._n_T].reshape(len(self.spec.t_obs_yr), len(self.spec.z_obs_m))
        p_compressed = y[self._n_T :]
        p = np.sinh(p_compressed) * 1.0e3
        return T, p

    def fit(
        self,
        *,
        n_train: int = 1000,
        n_pca: int = 12,
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ) -> CoupledForwardEmulator:
        rng = np.random.default_rng() if rng is None else rng
        sampler = qmc.LatinHypercube(d=len(PARAM_BOX_LO), seed=int(rng.integers(2**32)))
        u = sampler.random(n_train)
        thetas = PARAM_BOX_LO + u * (PARAM_BOX_HI - PARAM_BOX_LO)

        Y = np.empty((n_train, self._n_T + self._n_p))
        for k, th in enumerate(thetas):
            Y[k] = self._run_one(th)
            if verbose and (k + 1) % 100 == 0:
                print(f"  training forward {k + 1}/{n_train}", flush=True)

        self._y_mean = Y.mean(axis=0)
        self._y_std = Y.std(axis=0) + 1.0e-12
        Yn = (Y - self._y_mean) / self._y_std

        n_pca = min(n_pca, Yn.shape[0], Yn.shape[1])
        self.pca = PCA(n_components=n_pca)
        Z = self.pca.fit_transform(Yn)

        # Matern (nu=2.5) is smoother-than-RBF but tolerates the
        # non-stationary, non-analytic curvature of the coupled
        # forward better than the default RBF.
        base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(len(PARAM_BOX_LO)),
            length_scale_bounds=(1e-3, 1e4),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1.0))
        self.gps = []
        for j in range(n_pca):
            gp = GaussianProcessRegressor(
                kernel=base_kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                alpha=1e-8,
            )
            gp.fit(thetas, Z[:, j])
            self.gps.append(gp)
            if verbose:
                print(
                    f"  fit GP {j + 1}/{n_pca}  marginal log-lik={gp.log_marginal_likelihood_value_:.1f}",
                    flush=True,
                )
        return self

    # ------------------------------------------------------------- prediction

    def predict(self, theta: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Emulator not fit yet; call .fit() first.")
        theta_2d = np.atleast_2d(theta)
        z_pred = np.column_stack([gp.predict(theta_2d) for gp in self.gps])
        yn_pred = self.pca.inverse_transform(z_pred)
        y_pred = yn_pred * self._y_std + self._y_mean
        return y_pred[0] if theta.ndim == 1 else y_pred

    def predict_T_p(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = self.predict(theta)
        return self._decompress(y)

    # ------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> CoupledForwardEmulator:
        return joblib.load(Path(path).expanduser().resolve())

    # ------------------------------------------------------------- evaluation

    def evaluate(
        self,
        *,
        n_test: int = 200,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Held-out accuracy on a fresh Latin-hypercube test set.

        Reports two error families:

        * **rel_T** -- |emul - true| RMS in K, plus the same divided by
          the training-set per-channel std on the T outputs (a unit
          where 1.0 means "as much error as the data varies").
        * **rel_p** -- same on the p outputs.

        The pass criterion for the inversion is rel_T_to_std < 0.2 and
        rel_p_to_std < 0.2; emulator noise then sits ~5x below the
        natural variability of the forward output and the MCMC
        likelihood is not biased by emulator drift.
        """
        rng = np.random.default_rng() if rng is None else rng
        sampler = qmc.LatinHypercube(d=len(PARAM_BOX_LO), seed=int(rng.integers(2**32)))
        u = sampler.random(n_test)
        thetas = PARAM_BOX_LO + u * (PARAM_BOX_HI - PARAM_BOX_LO)

        # Per-channel training stds (in *compressed* output space).
        assert self._y_std is not None  # set during fit(); required before scoring
        std_T = self._y_std[: self._n_T]
        std_p_compressed = self._y_std[self._n_T :]

        err_T_K = np.empty(n_test)
        err_p_Pa = np.empty(n_test)
        rel_T = np.empty(n_test)
        rel_p = np.empty(n_test)
        for k, th in enumerate(thetas):
            # Truth in (T_K, p_Pa) physical units.
            T_true, p_true = coupled_forward(th, **self._forward_kwargs())
            T_emul, p_emul = self.predict_T_p(th)
            err_T_K[k] = float(np.sqrt(np.mean((T_emul - T_true) ** 2)))
            err_p_Pa[k] = float(np.sqrt(np.mean((p_emul - p_true) ** 2)))
            # Relative to per-channel training std (in compressed space
            # for p so the comparison is consistent with the GP fit).
            rel_T[k] = err_T_K[k] / float(std_T.mean())
            p_emul_compressed = np.arcsinh(p_emul / 1.0e3)
            p_true_compressed = np.arcsinh(p_true / 1.0e3)
            rel_p[k] = float(
                np.sqrt(np.mean((p_emul_compressed - p_true_compressed) ** 2))
                / float(std_p_compressed.mean())
            )

        return {
            "median_err_T_K": float(np.median(err_T_K)),
            "p95_err_T_K": float(np.quantile(err_T_K, 0.95)),
            "median_err_p_Pa": float(np.median(err_p_Pa)),
            "p95_err_p_Pa": float(np.quantile(err_p_Pa, 0.95)),
            "median_rel_T": float(np.median(rel_T)),
            "median_rel_p": float(np.median(rel_p)),
            "n_test": float(n_test),
        }
