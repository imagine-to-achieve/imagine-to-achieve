# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _cfg_get(container, dotted_key, default=None):
    current = container
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _print_config_summary(cfg) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if os.environ.get("RLINF_PRINT_FULL_CONFIG", "0") in ("1", "true", "True"):
        print(json.dumps(cfg_dict, indent=2))
        return

    summary_keys = (
        "runner.max_epochs",
        "runner.save_interval",
        "runner.val_check_interval",
        "env.train.total_num_envs",
        "env.train.group_size",
        "env.train.max_episode_steps",
        "actor.model.model_type",
        "actor.model.num_action_chunks",
        "actor.model.cosmos.backend",
        "actor.model.cosmos.num_steps",
        "env.train.ctrl_world_cfg.num_inference_steps",
        "env.train.ctrl_world_cfg.decode_chunk_size",
        "algorithm.adv_type",
        "algorithm.normalize_advantages",
        "actor.micro_batch_size",
        "actor.global_batch_size",
    )
    summary = {key: _cfg_get(cfg_dict, key) for key in summary_keys}
    summary["log_path"] = _cfg_get(cfg_dict, "runner.logger.log_path")
    summary["experiment_name"] = _cfg_get(cfg_dict, "runner.logger.experiment_name")
    print("[Config] compact summary; set RLINF_PRINT_FULL_CONFIG=1 for full Hydra config")
    print(json.dumps(summary, indent=2))


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    _print_config_summary(cfg)

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group = None
    rollout_group = None
    if not cfg.runner.get("replay_only", False):
        # Create actor worker group
        actor_placement = component_placement.get_strategy("actor")

        if cfg.algorithm.loss_type == "embodied_sac":
            from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

            actor_worker_cls = EmbodiedSACFSDPPolicy
        elif cfg.algorithm.loss_type == "embodied_dagger":
            from rlinf.workers.actor.fsdp_dagger_policy_worker import (
                EmbodiedDAGGERFSDPPolicy,
            )

            actor_worker_cls = EmbodiedDAGGERFSDPPolicy
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor
        actor_group = actor_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )
        # Create rollout worker group
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
