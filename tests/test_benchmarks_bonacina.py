"""Tests for gt_theory.benchmarks.bonacina enthalpy machinery."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.benchmarks.bonacina import (
    column_integrated_enthalpy,
    enthalpy_density,
)


def _props():
    return dict(
        porosity=0.30,
        rho_r=2700.0,
        c_r=800.0,
        rho_w=1000.0,
        c_w=4186.0,
        rho_i=917.0,
        c_i=2108.0,
        L_f=3.34e5,
    )


def test_enthalpy_at_reference_state() -> None:
    """At T = T_ref and S_i = 0, the sensible part is zero and the
    latent part is zero, so H = 0."""
    H = enthalpy_density(
        np.array([0.0]),
        np.array([0.0]),
        T_ref=0.0,
        **_props(),
    )
    assert H[0] == pytest.approx(0.0, abs=1e-6)


def test_enthalpy_latent_term_freezing() -> None:
    """At T = T_ref and S_i = 1, only the latent term survives, and it
    is negative (freezing releases heat to the surroundings, i.e. the
    column's enthalpy density decreases)."""
    H = enthalpy_density(
        np.array([0.0]),
        np.array([1.0]),
        T_ref=0.0,
        **_props(),
    )
    props = _props()
    expected = -props["porosity"] * props["rho_w"] * props["L_f"]
    assert H[0] == pytest.approx(expected, rel=1e-6)


def test_enthalpy_sensible_warming() -> None:
    """At S_i = 0, dH/dT equals (rho c)_eff."""
    T0 = enthalpy_density(np.array([0.0]), np.array([0.0]), T_ref=0.0, **_props())
    T1 = enthalpy_density(np.array([1.0]), np.array([0.0]), T_ref=0.0, **_props())
    p = _props()
    expected = (1 - p["porosity"]) * p["rho_r"] * p["c_r"] + p["porosity"] * p["rho_w"] * p["c_w"]
    assert (T1[0] - T0[0]) == pytest.approx(expected, rel=1e-6)


def test_column_integrated_enthalpy_uniform_state() -> None:
    """For a uniform-property column at constant T and S_i, the
    column-integrated H equals H * column length."""
    L = 10.0
    z = np.linspace(0.0, L, 21)
    T = 1.0 * np.ones_like(z)
    Si = np.zeros_like(z)
    H_int = column_integrated_enthalpy(T, Si, z, T_ref=0.0, **_props())
    H_density = float(
        enthalpy_density(
            np.array([1.0]),
            np.array([0.0]),
            T_ref=0.0,
            **_props(),
        )[0]
    )
    assert H_int == pytest.approx(H_density * L, rel=1e-12)


def test_enthalpy_density_vectorised() -> None:
    """Vector inputs return vector outputs of the same shape."""
    T = np.array([0.0, 1.0, 2.0, -1.0])
    S = np.array([0.0, 0.0, 0.5, 1.0])
    H = enthalpy_density(T, S, T_ref=0.0, **_props())
    assert H.shape == (4,)
    assert np.all(np.isfinite(H))
