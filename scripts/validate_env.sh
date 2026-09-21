#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv_aarch64}"

module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
GCC_RUNTIME_DIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/rlinf_runtime"

python -u "${ROOT_DIR}/scripts/validate_env.py"
