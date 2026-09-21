#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"

if (($# == 0)); then
  echo "usage: validate.sbatch CONFIG_PATH [CONFIG_PATH ...]" >&2
  exit 2
fi
CONFIG_PATHS=("$@")
CONFIG_PATH="${CONFIG_PATHS[0]}"
# Slurm executes a spool copy, so locate the repository from the real config.
ROOT_DIR="$(cd -- "$(dirname -- "${CONFIG_PATH}")/.." && pwd)"
VENV_PATH="${VENV_PATH:?VENV_PATH must name the aarch64 training environment}"

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${ROOT_DIR}"
python -m pytest -q tests
for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
  python -m rlinf_modified.train --compose-only --config "${CONFIG_PATH}"
done
