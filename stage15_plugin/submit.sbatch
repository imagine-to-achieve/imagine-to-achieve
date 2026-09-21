#!/usr/bin/env bash
# Resources are supplied by rlinf_modified.launch from the selected YAML.
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1

set -euo pipefail

module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"

CONFIG_PATH=""
while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "--config must be non-empty" >&2
  exit 2
fi
ROOT_DIR="$(cd -- "$(dirname -- "${CONFIG_PATH}")/.." && pwd)"
# Slurm executes a spool copy, so BASH_SOURCE cannot locate the repository.

VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv_aarch64}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "missing aarch64 Python 3.11 environment: ${VENV_PATH}" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export VENV_PATH
export PYTHONPATH="${ROOT_DIR}/src"
python -m rlinf_modified.launch validate-allocation --config "${CONFIG_PATH}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export RLINF_HEAD_NODE="${HEAD_NODE}"
export RLINF_CONFIG="${CONFIG_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# PyTorch 2.10 renamed the NCCL flight-recorder buffer variable.
export TORCH_FR_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export FI_PROVIDER=sockets

# One task per node owns exactly the four Slurm-mounted GPUs.
exec srun --label --kill-on-bad-exit=1 \
  --nodes="${SLURM_JOB_NUM_NODES}" \
  --ntasks="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 \
  "${ROOT_DIR}/stage15_plugin/launch_ray_node.sh" "${CONFIG_PATH}" "${HEAD_NODE}"
