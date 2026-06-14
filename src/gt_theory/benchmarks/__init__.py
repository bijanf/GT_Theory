"""Analytical and numerical benchmarks for the gt_theory solvers."""

from gt_theory.benchmarks.bonacina import (
    boundary_heat_flux_integrated,
    column_integrated_enthalpy,
    enthalpy_density,
)
from gt_theory.benchmarks.carslaw_jaeger import (
    harmonic_temperature_response,
    step_temperature_response,
)
from gt_theory.benchmarks.stefan import (
    StefanOnePhaseParams,
    stefan_front_position,
    stefan_temperature_profile,
)
from gt_theory.benchmarks.terzaghi import (
    degree_of_consolidation,
    isochrone,
)
from gt_theory.benchmarks.theis import (
    hydraulic_diffusivity,
    step_pressure_response,
)
from gt_theory.benchmarks.undrained_ratio import expected_dp_dT

__all__ = [
    "StefanOnePhaseParams",
    "boundary_heat_flux_integrated",
    "column_integrated_enthalpy",
    "degree_of_consolidation",
    "enthalpy_density",
    "expected_dp_dT",
    "harmonic_temperature_response",
    "hydraulic_diffusivity",
    "isochrone",
    "stefan_front_position",
    "stefan_temperature_profile",
    "step_pressure_response",
    "step_temperature_response",
]
