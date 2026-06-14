"""Tests for the programmatic Köppen-Geiger classifier."""

from __future__ import annotations

import numpy as np
import pytest

from gt_theory.io.koppen import (
    KOPPEN_CLASSES,
    koppen_class_from_monthly,
    koppen_code,
)


def _monthly_constant(t_c: float, p_mm: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full(12, t_c, dtype=float), np.full(12, p_mm, dtype=float)


def test_tropical_rainforest_is_Af() -> None:
    t, p = _monthly_constant(t_c=27.0, p_mm=200.0)
    cls = koppen_class_from_monthly(t_monthly_c=t, p_monthly_mm=p, lat_deg=0.0)
    assert cls == "Af"


def test_polar_icecap_is_EF() -> None:
    t, p = _monthly_constant(t_c=-25.0, p_mm=10.0)
    cls = koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=80.0,
    )
    assert cls == "EF"


def test_tundra_is_ET() -> None:
    t = np.array([-20.0, -20.0, -15.0, -8.0, 0.0, 5.0, 8.0, 5.0, 0.0, -8.0, -15.0, -20.0])
    p = np.full(12, 20.0)
    cls = koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=70.0,
    )
    assert cls == "ET"


def test_hot_desert_is_BWh() -> None:
    t, p = _monthly_constant(t_c=28.0, p_mm=5.0)
    cls = koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=22.0,
    )
    assert cls == "BWh"


def test_mediterranean_is_Cs() -> None:
    # NH mediterranean: cool wet winter, hot dry summer.
    t = np.array([8, 9, 11, 14, 18, 22, 26, 26, 22, 18, 13, 9], dtype=float)
    p = np.array([90, 80, 70, 50, 30, 10, 5, 5, 30, 70, 90, 100], dtype=float)
    cls = koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=40.0,
    )
    assert cls is not None
    assert cls.startswith("Cs")


def test_continental_dfb_class_recovers() -> None:
    # Cold-month < -3, warm month > 10, year-round precip.
    t = np.array([-5, -3, 2, 8, 14, 19, 21, 20, 14, 7, 1, -4], dtype=float)
    p = np.full(12, 60.0)
    cls = koppen_class_from_monthly(
        t_monthly_c=t,
        p_monthly_mm=p,
        lat_deg=55.0,
    )
    assert cls is not None
    assert cls.startswith("D")


def test_invalid_length_rejected() -> None:
    with pytest.raises(ValueError, match="length 12"):
        koppen_class_from_monthly(
            t_monthly_c=np.zeros(11),
            p_monthly_mm=np.zeros(12),
            lat_deg=0.0,
        )


def test_koppen_code_round_trip() -> None:
    for k, name in enumerate(KOPPEN_CLASSES, start=1):
        assert koppen_code(name) == k
    assert koppen_code(None) == 0
    assert koppen_code("not-a-class") == 0


def test_nan_inputs_return_none() -> None:
    t = np.full(12, np.nan)
    p = np.full(12, 100.0)
    cls = koppen_class_from_monthly(t_monthly_c=t, p_monthly_mm=p, lat_deg=0.0)
    assert cls is None
