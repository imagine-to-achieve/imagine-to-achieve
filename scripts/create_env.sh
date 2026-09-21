#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv_aarch64}"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "run this script on an aarch64 compute node" >&2
  exit 1
fi
UV_BIN="${UV_BIN:-uv}"
"${UV_BIN}" venv --seed --python 3.11 "${VENV_PATH}"
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  --find-links https://whl.natten.org \
  --index-strategy unsafe-best-match \
  -r "${ROOT_DIR}/requirements-aarch64-cu130.txt"
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  --index-url https://pypi.org/simple \
  "cmake>=3.21" "pybind11[global]" ninja
NVTE_FRAMEWORK=pytorch \
NVTE_CUDA_ARCHS=90 \
NVTE_BUILD_MAX_JOBS=8 \
NVTE_BUILD_THREADS_PER_JOB=2 \
"${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" \
  --no-build-isolation \
  "git+https://github.com/NVIDIA/TransformerEngine.git@v2.12"
"${VENV_PATH}/bin/python" -m pip check
"${ROOT_DIR}/scripts/validate_env.sh"
