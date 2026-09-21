#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=final-frame-metadata-regression
#SBATCH --output=/path/to/rlinf_release/outputs/final_frame_metadata_regression/%x-%j.out

set -euo pipefail

ROOT_DIR=/path/to/rlinf_release
module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
source "${ROOT_DIR}/.venv_aarch64/bin/activate"
export PYTHONPATH="${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/src"
python -m pytest "${ROOT_DIR}/tests/test_config_tasks_launch.py" -q
