# GT_Theory — coupled thermal–hydraulic dynamics of the continental subsurface

[![tests](https://github.com/bijanf/GT_Theory/actions/workflows/tests.yml/badge.svg)](https://github.com/bijanf/GT_Theory/actions/workflows/tests.yml)
[![lint](https://github.com/bijanf/GT_Theory/actions/workflows/lint.yml/badge.svg)](https://github.com/bijanf/GT_Theory/actions/workflows/lint.yml)
[![smoke](https://github.com/bijanf/GT_Theory/actions/workflows/smoke.yml/badge.svg)](https://github.com/bijanf/GT_Theory/actions/workflows/smoke.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo%20on%20release-lightgrey.svg)](https://zenodo.org/)

Research code for coupled heat and water transport in the continental
subsurface: finite-volume and finite-difference solvers with freeze–thaw phase
change and thermo-poroelastic coupling, analytical benchmarks, Bayesian
inversion of borehole temperature profiles, empirical fingerprints against a
global borehole archive, and a dimensionless regime analysis.

The framework organises the subsurface into three dynamically coupled
interfaces — Land Surface–Atmosphere, Sediment–Water–Rock, and Crust–Mantle —
and derives the dimensionless numbers that partition the system into predicted
thermal/hydraulic regimes.

## What's here

- **Solvers** (`src/gt_theory/solvers/`) — a 1-D finite-volume backward-Euler
  column solver (`column_fvm_permafoam`, the primary scheme), a finite-difference
  Crank–Nicolson reference (`column_thermo_freeze_coupled`), a 2-D solver
  (`domain_2d_thermo_freeze_coupled`), and lighter conduction/enthalpy columns.
  Coupled temperature–pressure–ice-saturation transport with an apparent-heat-
  capacity latent-heat treatment and thermo-poroelastic pressure coupling.
- **Analytical benchmarks** (`src/gt_theory/benchmarks/`) — Carslaw–Jaeger
  conduction, one-phase Stefan freezing, Ogata–Banks advection–dispersion,
  Terzaghi consolidation, Theis drawdown, Bonacina phase change, and the
  undrained thermal-pressurisation ratio, each with a convergence test.
- **Bayesian inversion** (`src/gt_theory/inversion/`) — hierarchical and
  errors-in-variables inversion of borehole profiles, with a forward emulator.
- **Empirical fingerprints** (`src/gt_theory/fingerprints/`) — F1–F6:
  erfc envelope, T–p coupling, surface-amplitude amplification, heat budget,
  diffusive lag, and latent-heat closure.
- **Theory & regimes** (`src/gt_theory/theory/`, `diagnostics/`) — dimensionless
  numbers, effective properties, freezing curves, n-factors, regime placement.
- **Data ingest** (`src/gt_theory/io/`) — Huang–Pollack boreholes, CRU TS,
  EGMS/InSAR, GGMN groundwater levels, Köppen classes, Nordicana.
- **Pipeline** — a Snakemake DAG (`workflows/Snakefile`) from synthetic smoke
  test to the full borehole archive, with a PIK-HPC profile and SLURM scripts.
- **Figures** (`figures/`) — publication-quality figure recipes (render to
  `outputs/figures/`).

## Install

```bash
# Reproducible binary-pinned environment (recommended)
pixi install
pixi run pytest -q

# …or a plain virtualenv
pip install -e ".[dev,plot,io]"
pytest -q
```

## Quick start

```bash
# unit + benchmark + inversion tests
pytest -q

# end-to-end synthetic smoke run (no external data needed)
PYTHONPATH=src snakemake -s workflows/Snakefile smoke_synthetic --cores 4

# render a figure (writes to outputs/figures/)
PYTHONPATH=src python figures/numerical/fig10_fvm_vs_cn.py

# full ~800-borehole archive run on the PIK HPC
PYTHONPATH=src snakemake -s workflows/Snakefile --profile workflows/profiles/pik-hpc full
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/gt_theory/` | solvers, benchmarks, inversion, theory, fingerprints, IO, plotting |
| `figures/` | figure recipes (`figures/numerical/`, `figures/empirical/`, root) |
| `scripts/` | data download, ingest, inversion, and analysis drivers |
| `catalogs/` | borehole site list (`boreholes.yaml`, `all_sites.csv`) + QC flags |
| `data/` | per-dataset acquisition notes (`data/*/README.md`) + site configs |
| `workflows/` | Snakemake DAG, rules, and the PIK-HPC profile |
| `slurm/` | SLURM submission scripts for PIK |
| `tests/` | pytest suite (solvers, benchmarks, inversion, fingerprints, IO) |
| `.github/workflows/` | CI: tests, lint, smoke, release |

## Data

The external datasets (Huang–Pollack borehole archive, CRU TS, EGMS/InSAR,
GGMN, Köppen, supersite monitoring) are not redistributed here. Each
`data/<source>/README.md` documents how to obtain the corresponding dataset and
where the ingest code expects it. The synthetic smoke run needs none of them.

## Tests & CI

`pytest` runs the full suite (unit, analytical-benchmark convergence,
inversion recovery, IO, and Hypothesis property tests). GitHub Actions runs
`tests`, `lint` (Ruff), and a synthetic `smoke` end-to-end run on every push.

## Citation

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI is minted from the GitHub
release on publication.

## License

MIT — see [`LICENSE`](LICENSE).
