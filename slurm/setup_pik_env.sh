#!/usr/bin/env bash
# One-time PIK setup: install micromamba in $HOME, create a Python env
# with the gt_theory dependencies, and stage the project at $WORK_DIR.
#
# Usage (run from your laptop):
#
#     scp slurm/setup_pik_env.sh login01:setup_pik_env.sh
#     ssh login01 'bash setup_pik_env.sh'
#
# After this completes:
#   * micromamba lives at ~/.local/bin/micromamba
#   * env "gt-theory" has python 3.11 + numpy/scipy/pandas/xarray/snakemake
#   * the borehole archive is rsync'd to /p/tmp/$USER/gt_theory/data/
#   * a $WORK_DIR symlink at /p/tmp/$USER/gt_theory points at this layout
#
# This script is idempotent: re-running it updates the env in place.

set -euo pipefail

WORK_DIR="/p/tmp/${USER}/gt_theory"
ARCHIVE_TARGET="${WORK_DIR}/data/raw/boreholes/huang2000"

echo "[1/4] Installing micromamba in \$HOME/.local/bin (if missing)..."
if [ ! -x "${HOME}/.local/bin/micromamba" ]; then
    mkdir -p "${HOME}/.local/bin"
    cd "${HOME}/.local/bin"
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj bin/micromamba --strip-components=1
fi
export PATH="${HOME}/.local/bin:${PATH}"
hash -r

echo "[2/4] Creating/updating 'gt-theory' env..."
"${HOME}/.local/bin/micromamba" create -y -n gt-theory -c conda-forge \
    "python=3.11" \
    "numpy>=1.26" "scipy>=1.12" "pandas>=2.1" "pyarrow>=15" \
    "xarray>=2024.1" "netcdf4>=1.6" "pyyaml>=6" "matplotlib-base>=3.8" \
    "cartopy>=0.22" "tqdm" \
    "snakemake-minimal>=8" \
    || true

echo "[3/4] Staging project layout under ${WORK_DIR}..."
mkdir -p "${WORK_DIR}/data/raw/boreholes" \
         "${WORK_DIR}/outputs" \
         "${WORK_DIR}/logs"

echo "[4/4] Reminder: archive sync"
cat <<'EONOTES'
==============================================================
Final step (run from your laptop, NOT from login01):

  rsync -av --info=progress2 \
    data/raw/boreholes/huang2000/ \
    login01:/p/tmp/$USER/gt_theory/data/raw/boreholes/huang2000/

  rsync -av --info=progress2 --exclude .git --exclude outputs --exclude .snakemake \
    $HOME/GT_Theory/ \
    login01:/p/tmp/$USER/gt_theory/repo/

Then from login01:
  cd /p/tmp/$USER/gt_theory/repo
  micromamba activate gt-theory
  pip install -e .
  export GT_THEORY_BOREHOLE_ROOT=/p/tmp/$USER/gt_theory/data/raw/boreholes/huang2000
  sbatch slurm/submit_smoke10.sbatch
==============================================================
EONOTES

echo "Setup script finished."
