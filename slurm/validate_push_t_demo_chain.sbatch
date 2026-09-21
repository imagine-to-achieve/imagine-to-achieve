#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=push_t_demo_check
set -euo pipefail
ROOT_DIR="${1:?project root required}"
OUTPUT_DIR="${2:?output directory required}"
CASE_LIST="${3:-states,actions,hold}"
module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
source "${ROOT_DIR}/.venv_aarch64/bin/activate"
GCC_RUNTIME_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world:${ROOT_DIR}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export TMPDIR="/scratch/local/${SLURM_JOB_ID}/push_t_demo_tmp"
mkdir -p "${TMPDIR}" "${OUTPUT_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor" TRITON_CACHE_DIR="${TMPDIR}/triton"
cd "${ROOT_DIR}"
episodes=(0 53 123 6)
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/validate_push_t_demo_chain.py \
    --config "${ROOT_DIR}/outputs/push_t_combined_mse_ctrl50_8n4g_base50_s5/resolved_config.yaml" \
    --output "${OUTPUT_DIR}" --episode "${episodes[$gpu]}" --cases "${CASE_LIST}" \
    > "${OUTPUT_DIR}/episode_${episodes[$gpu]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
