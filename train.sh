#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: ./train.sh {run|preflight|submit|submit-dry-run} --config PATH [--start-update N] [--until-update N]" >&2
  exit 2
fi
shift

CONFIG_PATH=""
START_UPDATE=""
UNTIL_UPDATE=""
while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --start-update)
      START_UPDATE="${2:-}"
      shift 2
      ;;
    --until-update)
      UNTIL_UPDATE="${2:-}"
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
CONFIG_PATH="$(realpath -- "${CONFIG_PATH}")"

VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv_aarch64}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "missing standalone environment: ${VENV_PATH}; run scripts/create_env.sh on an aarch64 compute node" >&2
  exit 1
fi

if [[ "${MODE}" == "submit" || "${MODE}" == "submit-dry-run" ]]; then
  # X86 login nodes cannot execute the aarch64 compute venv.
  export VENV_PATH
  export PYTHONPATH="${ROOT_DIR}/src"
  SUBMIT_PYTHON="${RLINF_SUBMIT_PYTHON:-python3}"
  SUBMIT_RANGE_ARGS=()
  if [[ -n "${START_UPDATE}" ]]; then
    SUBMIT_RANGE_ARGS+=(--start-update "${START_UPDATE}")
  fi
  if [[ -n "${UNTIL_UPDATE}" ]]; then
    SUBMIT_RANGE_ARGS+=(--until-update "${UNTIL_UPDATE}")
  fi
  if [[ "${MODE}" == "submit" ]]; then
    exec "${SUBMIT_PYTHON}" -m rlinf_modified.launch submit --config "${CONFIG_PATH}" "${SUBMIT_RANGE_ARGS[@]}"
  fi
  exec "${SUBMIT_PYTHON}" -m rlinf_modified.launch submit --dry-run --config "${CONFIG_PATH}" "${SUBMIT_RANGE_ARGS[@]}"
fi

if [[ -n "${START_UPDATE}" || -n "${UNTIL_UPDATE}" ]]; then
  echo "--start-update/--until-update are only valid for submit modes" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export VENV_PATH
export PYTHONPATH="${ROOT_DIR}/src"

case "${MODE}" in
  run)
    exec python -m rlinf_modified.train --config "${CONFIG_PATH}"
    ;;
  preflight)
    exec python -m rlinf_modified.train --preflight-only --config "${CONFIG_PATH}"
    ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac
