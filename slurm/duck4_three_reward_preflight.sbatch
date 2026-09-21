#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=duck4-three-reward-preflight
#SBATCH --output=/path/to/rlinf_release/outputs/preflight_regression/%x-%j.out

set -euo pipefail

ROOT_DIR=/path/to/rlinf_release
module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
source "${ROOT_DIR}/.venv_aarch64/bin/activate"
export PYTHONPATH="${ROOT_DIR}/src"

python -m rlinf_modified.train --preflight-only \
  --config "${ROOT_DIR}/configs/duck4_b128_mb32_u50_traj_reward_fixediface_side_rm.yaml"
python -m rlinf_modified.train --preflight-only \
  --config "${ROOT_DIR}/configs/duck4_b128_mb32_u50_terminal_mse_reward_fixediface_side_rm.yaml"
python -m rlinf_modified.train --preflight-only \
  --config "${ROOT_DIR}/configs/duck4_b128_mb32_u50_combined_mse_reward_fixediface_side_rm.yaml"
