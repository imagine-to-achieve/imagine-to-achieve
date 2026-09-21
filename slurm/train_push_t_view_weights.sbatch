#!/usr/bin/env bash
# Resources and dependency chain are supplied by submit_push_t_weight_ablation.py.
set -euo pipefail
CONFIG_PATH="${2:?expected --config PATH}"
ROOT_DIR="$(cd -- "$(dirname -- "${CONFIG_PATH}")/.." && pwd)"
COMPARISON_DIR="${ROOT_DIR}/outputs/push_t_view_weight_ablation_fp32_h13_20260904"
module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
source "${ROOT_DIR}/.venv_aarch64/bin/activate"
GCC_RUNTIME_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world:${ROOT_DIR}/src"
export PYTHONDONTWRITEBYTECODE=1
export RLINF_PROBE_SHARED_WRITES=1
export MPLCONFIGDIR="/scratch/local/${SLURM_JOB_ID}/push_t_comparison_mpl"
mkdir -p "${MPLCONFIGDIR}"
cd "${ROOT_DIR}"
python scripts/verify_push_t_ablation_sources.py
quota_args=(--path "$(python -m rlinf_modified.launch value --config "${CONFIG_PATH}" --field runtime.output_dir)")
if [[ "${RLINF_ALLOW_EXPIRED_DISK_POOL_AFTER_WRITE_PROBE:-0}" == "1" ]]; then
    quota_args+=(--allow-expired-disk-pool-after-write-probe)
fi
python scripts/check_push_t_storage_quota.py "${quota_args[@]}"
finish_comparison_segment() {
    local code=$?
    trap - EXIT
    python scripts/summarize_push_t_weight_ablation.py > "${COMPARISON_DIR}/summary_job_${SLURM_JOB_ID}.log" 2>&1 || true
    exit "${code}"
}
trap finish_comparison_segment EXIT
bash "${ROOT_DIR}/slurm/submit.sbatch" "$@"
