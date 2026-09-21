#!/usr/bin/env bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

module load "${BUILDENV_MODULE:?set BUILDENV_MODULE to your cluster CUDA build-environment module}"

ROOT_DIR="/path/to/rlinf_release"
VENV_PATH="/path/to/workspace/github/rlinf-cosmos3-ctrl-world/.venv_aarch64"
source "${VENV_PATH}/bin/activate"
export RLINF_STAGE15_MODE=a_exact
export PYTHONPATH="${ROOT_DIR}/stage15_plugin:${ROOT_DIR}/third_party/rlinf_runtime:${ROOT_DIR}/third_party:${ROOT_DIR}/third_party/ctrl_world:${ROOT_DIR}/src"

cd "${ROOT_DIR}"
python - <<'PY'
from pathlib import Path

from rlinf_modified.config import load_config
from rlinf_modified.engine.real import RealTrainer
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

assert hasattr(RealTrainer, "_stage15_original_overrides")
assert hasattr(EmbodiedRunner, "_stage15_original_commit_training_success")
assert hasattr(EmbodiedFSDPActor, "_stage15_original_run_training")
assert hasattr(
    MultiStepRolloutWorker,
    "_stage15_original_set_post_update_eval_context",
)

config_path = Path("configs/stage15_a_exact_u_mc4.yaml")
config = load_config(config_path)
trainer = RealTrainer(config)
overrides = trainer._overrides(str(config.cosmos.checkpoint_config_path))
by_key = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in overrides}
assert by_key["runner.rollout_only"] == "false"
assert by_key["runner.first_update_smoke.enabled"] == "true"
assert by_key["runner.save_interval"] == "-1"
assert by_key["runner.checkpoint_milestone_interval"] == "-1"
assert by_key["runner.val_check_interval"] == "1"
assert by_key["duck.evaluation.enabled"] == "true"
assert by_key["++algorithm.fpo_ratio_granularity"] == "per_action"
assert by_key["++actor.model.cosmos.fpo_score_parameterization"] == "velocity"
assert by_key["actor.model.cosmos.fpo_num_mc_samples"] == "4"
print("STAGE15_PLUGIN_SMOKE_PASS")
PY
