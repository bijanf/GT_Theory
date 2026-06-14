"""Tests for gt_theory.benchmarks.terzaghi."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.terzaghi import (
    degree_of_consolidation,
    isochrone,
)


def test_isochrone_initial_state_unity() -> None:
    """At T_v -> 0 the isochrone should equal p_0 in the interior."""
    z = np.linspace(0.0, 1.0, 51)
    L = 1.0
    p = isochrone(z, T_v=1e-6, p0=1.0, L=L)
    # Interior nodes (excluding the top boundary) should be close to 1.
    assert p[10:].mean() > 0.95


def test_isochrone_full_consolidation_zero() -> None:
    """At T_v >> 1 the isochrone should approach zero everywhere."""
    z = np.linspace(0.0, 1.0, 21)
    p = isochrone(z, T_v=5.0, p0=1.0, L=1.0)
    assert float(np.max(np.abs(p))) < 1e-3


def test_degree_of_consolidation_monotone() -> None:
    Tv = np.linspace(0.01, 2.0, 30)
    U = degree_of_consolidation(Tv)
    assert U[0] < 0.2
    assert U[-1] > 0.99
    assert np.all(np.diff(U) > 0)


def test_degree_of_consolidation_classic_value() -> None:
    """U(T_v = 0.197) is approximately 0.50 (Terzaghi's classical value)."""
    U = float(degree_of_consolidation(0.197))
    assert U == pytest.approx(0.50, abs=5e-3)


def test_isochrone_validation() -> None:
    with pytest.raises(ValueError):
        isochrone(np.array([0.5]), T_v=0.1, p0=1.0, L=-1.0)
