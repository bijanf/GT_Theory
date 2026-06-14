#!/usr/bin/env bash
#SBATCH --job-name=gt-smoke10
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/p/tmp/%u/gt_theory/logs/smoke10-%j.out
#SBATCH --error=/p/tmp/%u/gt_theory/logs/smoke10-%j.err
#
# Run the smoke-10 Snakemake pipeline on PIK HPC.  Requires the env
# created by ``slurm/setup_pik_env.sh`` plus the project rsync'd to
# /p/tmp/$USER/gt_theory/repo and the borehole archive at
# /p/tmp/$USER/gt_theory/data/raw/boreholes/huang2000.

set -euo pipefail

REPO="/p/tmp/${USER}/gt_theory/repo"
ARCHIVE="/p/tmp/${USER}/gt_theory/data/raw/boreholes/huang2000"

cd "${REPO}"

# Activate the micromamba env without a full shell init.
export PATH="${HOME}/.local/bin:${PATH}"
eval "$(${HOME}/.local/bin/micromamba shell hook -s bash)"
micromamba activate gt-theory

# Point the catalog loader at the staged archive.
export GT_THEORY_BOREHOLE_ROOT="${ARCHIVE}"

# Run smoke-10 with local cores (no SLURM-within-SLURM fanout for this
# small subset; the full archive uses the pik-hpc Snakemake profile and
# real per-site array submission).
snakemake -s workflows/Snakefile smoke_10 --cores "${SLURM_CPUS_PER_TASK:-2}"
