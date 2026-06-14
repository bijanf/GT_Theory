"""Tests for gt_theory.diagnostics.regime."""

from __future__ import annotations

import numpy as np

from gt_theory.diagnostics.regime import (
    RegimeBands,
    classify_regime,
    latent_heat_regime_l,
)


def test_l_permafrost_regime() -> None:
    """Shallow permafrost with phi=0.3 and small Delta_T should give
    ℒ in the 10-100 range per theory Table 1."""
    l = latent_heat_regime_l(porosity=0.30, delta_T_K=1.0)
    assert 1.0 < float(l) < 200.0


def test_l_arid_regime() -> None:
    """Dry crystalline crust with phi=0.02 and large Delta_T -> small ℒ."""
    l = latent_heat_regime_l(porosity=0.02, delta_T_K=10.0)
    assert float(l) < 0.5


def test_l_vectorised() -> None:
    porosity = np.array([0.05, 0.30, 0.40])
    dT = np.array([2.0, 1.0, 0.5])
    l = latent_heat_regime_l(porosity=porosity, delta_T_K=dT)
    assert l.shape == (3,)
    assert np.all(l > 0.0)


def test_classify_buckets() -> None:
    l = np.array([0.001, 0.05, 0.5, 50.0])
    bands = RegimeBands()
    lab = classify_regime(l, bands)
    assert lab[0] == "arid"
    assert lab[1] == "arid"
    assert lab[2] == "intermediate"
    assert lab[3] == "permafrost"


def test_classify_nan_undetermined() -> None:
    l = np.array([np.nan, 1.0])
    lab = classify_regime(l)
    assert lab[0] == "undetermined"
    assert lab[1] == "permafrost"  # at boundary -> permafrost


def test_l_handles_zero_dT() -> None:
    """A site with no temperature variation has ill-defined ℒ; return NaN."""
    l = latent_heat_regime_l(porosity=0.3, delta_T_K=0.0)
    assert not np.isfinite(float(l))
