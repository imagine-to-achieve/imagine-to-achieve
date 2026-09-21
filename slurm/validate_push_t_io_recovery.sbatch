#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --job-name=push_t_io_recovery_check
set -euo pipefail
ROOT_DIR="${1:?project root required}"
OUTPUT_DIR="${ROOT_DIR}/outputs/push_t_view_weight_ablation_fp32_h13_20260904/recovery_20260905"
module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
source "${ROOT_DIR}/.venv_aarch64/bin/activate"
GCC_RUNTIME_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world:${ROOT_DIR}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TMPDIR="/scratch/local/${SLURM_JOB_ID}/io_recovery_test_tmp"
mkdir -p "${TMPDIR}" "${OUTPUT_DIR}"
cd "${ROOT_DIR}"
python -m rlinf.utils.diagnostic_io --probe-dir "${ROOT_DIR}/outputs/push_t_combined_mse_fp32_h13_original_15u_s5" > "${OUTPUT_DIR}/compute_write_probe.log" 2>&1
python -m rlinf.utils.diagnostic_io --probe-dir "${ROOT_DIR}/outputs/push_t_combined_mse_fp32_h13_no_wrist_15u_s5" >> "${OUTPUT_DIR}/compute_write_probe.log" 2>&1
python -m pytest -q -p no:cacheprovider tests/test_diagnostic_io.py tests/test_push_t_view_weights.py tests/test_ur5_adapter_autocast.py --junitxml="${OUTPUT_DIR}/tests.xml" > "${OUTPUT_DIR}/tests.log" 2>&1
python scripts/validate_push_t_weight_composition.py > "${OUTPUT_DIR}/composition.log" 2>&1
