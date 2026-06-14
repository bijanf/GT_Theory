"""Canonical regime case studies for the numerical-companion paper.

Each case parameterises a 1-D column whose dimensionless-number
signature places it firmly inside one of the four regime quadrants
identified by the theory paper:

  - **permafrost**:        high mathcal{L} (latent heat dominant)
  - **geothermal**:        high Pe_T (advection dominant)
  - **arid_basin**:        low everything (pure-conduction baseline)
  - **thermo_poro**:       high Gamma N_alpha (thermo-poro coupling)

Each module exposes a single ``run()`` entry point returning the
forward-solution dataset together with the case's dimensionless-
number signature.
"""

from gt_theory.cases.arid_basin import run as run_arid_basin
from gt_theory.cases.geothermal import run as run_geothermal
from gt_theory.cases.permafrost import run as run_permafrost
from gt_theory.cases.thermo_poro_coupled import run as run_thermo_poro

CASES = {
    "permafrost": run_permafrost,
    "geothermal": run_geothermal,
    "arid_basin": run_arid_basin,
    "thermo_poro": run_thermo_poro,
}

__all__ = [
    "CASES",
    "run_arid_basin",
    "run_geothermal",
    "run_permafrost",
    "run_thermo_poro",
]
