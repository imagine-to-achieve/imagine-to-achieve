#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:?project root required}"
OUTPUT_DIR="${2:?output directory required}"
case "${SLURM_PROCID:?Slurm process rank required}" in
  0) DEMO_CASE=states ;;
  1) DEMO_CASE=hold ;;
  *) exit 2 ;;
esac
mkdir -p "${OUTPUT_DIR}/${DEMO_CASE}"
exec bash "${ROOT_DIR}/slurm/validate_push_t_demo_chain.sbatch" \
  "${ROOT_DIR}" "${OUTPUT_DIR}/${DEMO_CASE}" "${DEMO_CASE}"
