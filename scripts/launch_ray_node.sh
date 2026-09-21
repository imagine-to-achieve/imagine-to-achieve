#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?config path is required}"
HEAD_NODE="${2:?Ray head node is required}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${VENV_PATH:?VENV_PATH is required}/bin/activate"
# Ray daemons must inherit vendored modules before spawning actors.
export PYTHONPATH="${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world:${ROOT_DIR}/src"
export RLINF_NODE_RANK="${SLURM_PROCID:?SLURM_PROCID is required}"

SCRATCH_ROOT="/scratch/local/${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
SCRATCH_DIR="${SCRATCH_ROOT}/node_${RLINF_NODE_RANK}"
case "${SCRATCH_DIR}" in
  /scratch/local/[0-9]*/node_[0-9]*) ;;
  *) echo "refusing unsafe scratch path: ${SCRATCH_DIR}" >&2; exit 2 ;;
esac
mkdir -p "${SCRATCH_DIR}"/{tmp,ray,hf,xdg,pycache,wandb,diagnostics}
export TMPDIR="${SCRATCH_DIR}/tmp"
export RAY_TMPDIR="${SCRATCH_DIR}/ray"
export HF_HOME="${SCRATCH_DIR}/hf"
export XDG_CACHE_HOME="${SCRATCH_DIR}/xdg"
export PYTHONPYCACHEPREFIX="${SCRATCH_DIR}/pycache"
export WANDB_DIR="${SCRATCH_DIR}/wandb"
# Keep failures immediately visible and prevent each Ctrl-World pipeline from
# flooding the shared Slurm stream with its own 50-step progress bar.
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=1

OUTPUT_DIR="$(python -m rlinf_modified.launch value --config "${CONFIG_PATH}" --field runtime.output_dir)"
READY_TIMEOUT="$(python -m rlinf_modified.launch value --config "${CONFIG_PATH}" --field distributed.ray_ready_timeout_seconds)"
OBJECT_STORE_BYTES="$(python -m rlinf_modified.launch value --config "${CONFIG_PATH}" --field distributed.ray_object_store_bytes)"
NODES="$(python -m rlinf_modified.launch value --config "${CONFIG_PATH}" --field slurm.nodes)"
mkdir -p "${OUTPUT_DIR}/allocation" "${OUTPUT_DIR}/heartbeats"
DONE_FILE="${OUTPUT_DIR}/allocation/${SLURM_JOB_ID}.done"
HEARTBEAT_FILE="${OUTPUT_DIR}/heartbeats/slurm_rank_${RLINF_NODE_RANK}.json"
TRAIN_PID=""

write_heartbeat() {
  local phase="$1"
  local temporary="${HEARTBEAT_FILE}.tmp.${SLURM_JOB_ID}.${RLINF_NODE_RANK}"
  printf '{"job_id":"%s","node_rank":%s,"phase":"%s","host":"%s","segment_start":%s,"segment_end":%s}\n' \
    "${SLURM_JOB_ID}" "${RLINF_NODE_RANK}" "${phase}" "$(hostname)" \
    "${RLINF_SEGMENT_START:-0}" "${RLINF_SEGMENT_END:-0}" >"${temporary}"
  mv -f -- "${temporary}" "${HEARTBEAT_FILE}"
}

allocation_exit_code() {
  local process_exit_code="$1"
  local done_exit_code=""
  if [[ -f "${DONE_FILE}" ]]; then
    IFS= read -r done_exit_code <"${DONE_FILE}" || true
  fi
  if [[ "${done_exit_code}" =~ ^[0-9]+$ ]] && ((done_exit_code != 0)); then
    printf '%s\n' "${done_exit_code}"
  else
    printf '%s\n' "${process_exit_code}"
  fi
}

preserve_failure_diagnostics() {
  local failure_code="$1"
  local process_exit_code="$2"
  if ((failure_code == 0)); then
    return 0
  fi

  local diagnostic_dir="${OUTPUT_DIR}/diagnostics/job_${SLURM_JOB_ID}/node_${RLINF_NODE_RANK}"
  local ray_logs="${RAY_TMPDIR}/session_latest/logs"
  mkdir -p "${diagnostic_dir}/ray_logs"
  printf '{"job_id":"%s","node_rank":%s,"host":"%s","process_exit_code":%s,"allocation_exit_code":%s,"segment_start":%s,"segment_end":%s}\n' \
    "${SLURM_JOB_ID}" "${RLINF_NODE_RANK}" "$(hostname)" "${process_exit_code}" "${failure_code}" \
    "${RLINF_SEGMENT_START:-0}" "${RLINF_SEGMENT_END:-0}" \
    >"${diagnostic_dir}/metadata.json" || true

  # Ray stores actor exceptions in per-worker files that are not guaranteed to
  # reach the driver's Slurm stream. Copy every top-level regular log before
  # `ray stop` and scratch cleanup; node-specific directories avoid collisions.
  if [[ -d "${ray_logs}" ]]; then
    local -a ray_log_files=()
    mapfile -d '' -t ray_log_files < <(find "${ray_logs}" -maxdepth 1 -type f -print0)
    local ray_log
    for ray_log in "${ray_log_files[@]}"; do
      cp -a -- "${ray_log}" "${diagnostic_dir}/ray_logs/" || true
    done
  fi
}

cleanup() {
  local exit_code=$?
  local effective_exit_code
  write_heartbeat "exit_${exit_code}" || true
  if [[ "${RLINF_NODE_RANK}" == "0" ]]; then
    printf '%s\n' "${exit_code}" >"${DONE_FILE}.tmp"
    mv -f -- "${DONE_FILE}.tmp" "${DONE_FILE}"
  fi
  effective_exit_code="$(allocation_exit_code "${exit_code}")"
  if ((effective_exit_code != 0)); then
    write_heartbeat "allocation_failure_${effective_exit_code}" || true
    preserve_failure_diagnostics "${effective_exit_code}" "${exit_code}" || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  # Retry publication of locally spooled scalar diagnostics after workers stop.
  local view_spool="${SCRATCH_ROOT}/reward_view_records/$(basename -- "${OUTPUT_DIR}")"
  if [[ -d "${view_spool}" ]]; then
    python -m rlinf.utils.diagnostic_io --source "${view_spool}" \
      --destination "${OUTPUT_DIR}/reward_view_records" || true
  fi
  # Wait for late Ray children to stop creating pycache files.
  for _cleanup_attempt in {1..10}; do
    if rm -rf -- "${SCRATCH_DIR}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  trap - EXIT
  exit "${exit_code}"
}
forward_signal() {
  local signal_name="$1"
  write_heartbeat "signal_${signal_name,,}"
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -s "${signal_name}" "${TRAIN_PID}"
  fi
}
trap cleanup EXIT
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

python -m rlinf_modified.launch validate-allocation --check-gpus --config "${CONFIG_PATH}"
if [[ "${RLINF_PROBE_SHARED_WRITES:-0}" == "1" ]]; then
  python -m rlinf.utils.diagnostic_io --probe-dir "${OUTPUT_DIR}"
fi
HEAD_IP="$(getent ahostsv4 "${HEAD_NODE}" | awk 'NR == 1 {print $1}')"
if [[ -z "${HEAD_IP}" ]]; then
  echo "could not resolve Ray head node ${HEAD_NODE}" >&2
  exit 1
fi
export RAY_ADDRESS="${HEAD_IP}:6379"
write_heartbeat ray_start

if [[ "${RLINF_NODE_RANK}" == "0" ]]; then
  ray start --head --node-ip-address="${HEAD_IP}" --port=6379 \
    --object-store-memory="${OBJECT_STORE_BYTES}" --temp-dir="${RAY_TMPDIR}"
  python -m rlinf_modified.launch wait-ray --nodes "${NODES}" --timeout "${READY_TIMEOUT}"
  write_heartbeat train
  python -m rlinf_modified.train --config "${CONFIG_PATH}" &
  TRAIN_PID=$!
  wait "${TRAIN_PID}"
else
  deadline=$((SECONDS + READY_TIMEOUT))
  until ray start --address="${RAY_ADDRESS}" --temp-dir="${RAY_TMPDIR}"; do
    if ((SECONDS >= deadline)); then
      echo "Ray worker ${RLINF_NODE_RANK} timed out joining ${RAY_ADDRESS}" >&2
      exit 1
    fi
    sleep 2
  done
  write_heartbeat worker_ready
  while [[ ! -f "${DONE_FILE}" ]]; do
    sleep 5
  done
fi
