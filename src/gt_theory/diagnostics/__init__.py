"""Diagnostic post-processing utilities for the gt_theory solvers."""

from gt_theory.diagnostics.forward_misfit import (
    MisfitResult,
    rms_misfit_on_common_grid,
)
from gt_theory.diagnostics.regime import (
    RegimeBands,
    classify_regime,
    latent_heat_regime_l,
)

__all__ = [
    "MisfitResult",
    "RegimeBands",
    "classify_regime",
    "latent_heat_regime_l",
    "rms_misfit_on_common_grid",
]
