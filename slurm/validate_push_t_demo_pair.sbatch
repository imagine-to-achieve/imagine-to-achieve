#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --job-name=push_t_demo_fp32
set -euo pipefail
ROOT_DIR="${1:?project root required}"
OUTPUT_DIR="${2:?output directory required}"
exec srun --label --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  bash -l "${ROOT_DIR}/scripts/validate_push_t_demo_pair_node.sh" "${ROOT_DIR}" "${OUTPUT_DIR}"
