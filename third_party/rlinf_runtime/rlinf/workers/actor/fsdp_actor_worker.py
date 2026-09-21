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

import hashlib
import json
import os
import time
from collections.abc import Mapping
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils import _pytree

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.cross_rank_grpo import (
    derive_stable_seed,
    map_semantic_seed_to_uint32,
    select_cross_rank_collective_device,
    validate_and_compute_group_advantages,
    validate_and_compute_group_suffix_advantages,
    validate_cross_rank_runtime_contract,
)
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    calculate_episode_scores,
    kl_penalty,
    postprocess_embodied_advantages_outputs,
    preprocess_embodied_advantages_inputs,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.envs.world_model.duck_episode_contract import (
    assert_duck_episode_split_unchanged,
    load_duck_episode_split,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cosmos.fpo import (
    COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD,
    resolve_cosmos_replay_objective,
)
from rlinf.models.embodiment.cosmos.artifacts import CosmosArtifactWriter
from rlinf.models.embodiment.cosmos.diagnostics import (
    ActionRatioExplosionGuard,
    compute_action_logprob_diagnostics,
    compute_advantage_logprob_alignment,
    compute_action_param_checksum,
    compute_chain_diagnostics,
    compute_gradient_diagnostics,
    validate_action_logprobs_finite,
    warn_if_frozen_video_grads,
)
from rlinf.rewards.resnet_reward_model import classify_terminal_probabilities
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def build_duck_trajectory_provenance(
    *,
    split_manifest: str | Path,
    success_models: Mapping,
) -> dict:
    """Build scalar duck trajectory provenance without hashing model weights.

    The launch-time validator owns the expensive checkpoint hash verification.
    Actor ranks only require the configured SHA-256 values and copy them into
    records, avoiding a four-checkpoint read on every rank.
    """
    contract = load_duck_episode_split(str(split_manifest), "training")
    assert_duck_episode_split_unchanged(contract)
    configured_colors = {str(color) for color in success_models}
    expected_colors = set(contract.color_order)
    if configured_colors != expected_colors:
        raise ValueError(
            "Duck success models must exactly match manifest colors: "
            f"expected {sorted(expected_colors)}, got {sorted(configured_colors)}."
        )

    models: dict[str, dict[str, str]] = {}
    for color in contract.color_order:
        model_cfg = success_models[color]
        model_root = str(model_cfg.get("from_pretrained", "")).strip()
        artifact_name = str(model_cfg.get("artifact_name", "")).strip()
        digest = str(model_cfg.get("sha256", "")).strip().lower()
        if not model_root or not artifact_name:
            raise ValueError(
                f"Duck success model {color!r} requires from_pretrained and "
                "artifact_name."
            )
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(
                f"Duck success model {color!r} requires a configured SHA-256."
            )
        checkpoint_path = (
            Path(model_root).expanduser() / artifact_name
        ).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Duck success-model checkpoint does not exist: {checkpoint_path}"
            )
        models[color] = {"path": str(checkpoint_path), "sha256": digest}

    episodes: dict[int, dict[str, str]] = {}
    for color, episode_ids in contract.episodes_by_color.items():
        for episode_id in episode_ids:
            episodes[int(episode_id)] = {
                "color": color,
                "model_path": models[color]["path"],
                "model_sha256": models[color]["sha256"],
            }
    if set(episodes) != set(contract.episodes):
        raise ValueError("Duck training color pools do not cover the manifest split.")
    return {
        "manifest_path": contract.manifest_path,
        "manifest_sha256": contract.manifest_sha256,
        "episodes": episodes,
    }


def index_duck_seed_manifest_groups(
    seed_manifest: Mapping,
    *,
    expected_update_id: int,
) -> dict[tuple[int, int], dict]:
    """Index and strictly validate the duck reset fields in one seed manifest."""
    manifest_update = int(seed_manifest.get("update_id", -1))
    if manifest_update != int(expected_update_id):
        raise ValueError(
            "Duck seed manifest update does not match the actor update: "
            f"{manifest_update} != {expected_update_id}."
        )
    groups = seed_manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Duck seed manifest must contain non-empty groups.")
    indexed: dict[tuple[int, int], dict] = {}
    for raw_group in groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("Duck seed-manifest groups must be mappings.")
        group = dict(raw_group)
        key = (int(group["logical_round_id"]), int(group["group_slot"]))
        if key in indexed:
            raise ValueError(f"Duck seed manifest duplicates group key {key}.")
        if "reset_episode" not in group or not str(group.get("reset_color", "")):
            raise ValueError(
                f"Duck seed-manifest group {key} lacks reset_episode/reset_color."
            )
        indexed[key] = group
    return indexed


def validate_duck_trajectory_seed_alignment(
    *,
    seed_groups: Mapping[tuple[int, int], Mapping],
    logical_round_id: int,
    group_slot: int,
    reset_episode: int,
    reset_color: str,
    reset_seed: int,
    member_id: int,
    action_noise_seeds: list[int],
    ctrl_seed: int,
) -> None:
    """Abort immediately when rollout reset provenance drifts from its manifest."""
    key = (int(logical_round_id), int(group_slot))
    if key not in seed_groups:
        raise ValueError(f"Duck seed manifest is missing group key {key}.")
    group = seed_groups[key]
    expected = (
        int(group["reset_episode"]),
        str(group["reset_color"]),
        int(group["reset_seed"]),
    )
    actual = (int(reset_episode), str(reset_color), int(reset_seed))
    if actual != expected:
        raise ValueError(
            "Duck rollout reset provenance differs from seed manifest for "
            f"group {key}: actual={actual}, expected={expected}."
        )
    if int(ctrl_seed) != int(group.get("ctrl_seed", -1)):
        raise ValueError(
            f"Duck rollout Ctrl seed differs from seed manifest for group {key}."
        )
    members = group.get("members")
    if not isinstance(members, list):
        raise ValueError(f"Duck seed-manifest group {key} has no members.")
    matches = [
        member
        for member in members
        if int(member.get("member_id", -1)) == int(member_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Duck seed-manifest group {key} does not uniquely contain member "
            f"{member_id}."
        )
    chunks = matches[0].get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 5:
        raise ValueError(
            f"Duck seed-manifest member {member_id} must contain five chunks."
        )
    ordered_chunks = sorted(chunks, key=lambda chunk: int(chunk["chunk_id"]))
    if [int(chunk["chunk_id"]) for chunk in ordered_chunks] != list(range(5)):
        raise ValueError(
            f"Duck seed-manifest member {member_id} has invalid chunk IDs."
        )
    expected_action_seeds = [
        int(chunk["cosmos_joint_seed_int63"]) for chunk in ordered_chunks
    ]
    if [int(seed) for seed in action_noise_seeds] != expected_action_seeds:
        raise ValueError(
            "Duck rollout action seeds differ from seed manifest for "
            f"group {key}, member {member_id}."
        )


def duck_target_frame_provenance(cfg: Mapping, episode_id: int) -> dict:
    """Validate and materialize the same-episode duck target-frame contract."""
    train_cfg = cfg["env"]["train"]
    goal_cfg = cfg["reward"]["terminal_goal"]
    initial_frame = int(train_cfg.get("initial_frame_index", -1))
    terminal_frame = int(train_cfg.get("terminal_goal_frame_index", 0))
    same_episode = bool(train_cfg.get("terminal_goal_same_episode", False))
    reference_mode = str(goal_cfg.get("reference_mode", ""))
    if (
        initial_frame != 0
        or terminal_frame != -1
        or not same_episode
        or reference_mode != "same_episode_last_frame"
    ):
        raise ValueError(
            "Duck target-frame contract requires initial frame 0 and the same "
            "episode's final frame (-1)."
        )
    return {
        "initial_state_episode": int(episode_id),
        "initial_state_frame": initial_frame,
        "terminal_goal_reference_episode": int(episode_id),
        "terminal_goal_reference_frame": terminal_frame,
        "terminal_goal_reference_mode": reference_mode,
    }


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def _nested_tensor_bytes_by_device(value) -> dict[str, int]:
    """Count logical tensor bytes by device in a nested replay batch."""
    totals: dict[str, int] = {}
    if isinstance(value, torch.Tensor):
        device = value.device.type
        totals[device] = value.numel() * value.element_size()
        return totals
    if isinstance(value, dict):
        for child in value.values():
            for device, size in _nested_tensor_bytes_by_device(child).items():
                totals[device] = totals.get(device, 0) + size
    return totals


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def build_trajectory_aware_shuffle_id(
    *, num_chunks: int, num_trajectories: int, seed: int
) -> torch.Tensor:
    """Shuffle trajectories while keeping every trajectory's chunks adjacent."""
    if num_chunks <= 0 or num_trajectories <= 0:
        raise ValueError("num_chunks and num_trajectories must be positive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    trajectory_order = torch.randperm(num_trajectories, generator=generator)
    chunk_offsets = (
        torch.arange(num_chunks, dtype=torch.long).unsqueeze(0)
        * num_trajectories
    )
    return (trajectory_order.unsqueeze(1) + chunk_offsets).reshape(-1)


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)

        self.bucket_capacity = 128 * 1024 * 1024

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def divide_model_to_bucket(self, state_dict, has_visual):
        bucket_capacity = self.bucket_capacity
        model_bucket_list = []
        current_capacity = 0
        model_bucket = {}
        for key, val in state_dict.items():
            name = key
            if "_extra_state" in name:
                continue
            if has_visual:
                if name.startswith("model.language_model."):
                    name = "model." + name[21:]
                # NOTE:
                # if transformers version is 4.56.1 or older(not tested),
                # the following line should be uncommented

                # elif name.startswith("model."):
                #     name = name[6:]

            model_bucket[name] = val
            current_capacity += (
                val.numel() * val.element_size() * torch.distributed.get_world_size()
            )

            if current_capacity >= bucket_capacity:
                model_bucket_list.append(model_bucket)
                current_capacity = 0
                model_bucket = {}

        if len(model_bucket) > 0:
            model_bucket_list.append(model_bucket)
        return model_bucket_list

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())
        if self._weight_dst_rank_in_rollout is not None:
            rollout_dtype = None
            if self._cfg.get("sync_precision", None) is not None:
                rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)
            model_bucket_list = self.divide_model_to_bucket(
                self.rollout_state_dict, has_visual
            )
            self.log_debug(
                f"[sync_model_to_rollout rank-{self._rank}] length of model_bucket_list: {len(model_bucket_list)}"
            )
            for bucket_idx, model_bucket in enumerate(model_bucket_list):
                buffer = {}
                for k, v in model_bucket.items():
                    if isinstance(v, DTensor):
                        v = v.full_tensor()
                    if rollout_dtype is not None:
                        v = v.to(rollout_dtype)
                    if not self.is_pipeline:
                        v = reduce_tensor(v)
                    buffer[k] = v
                if bucket_idx == 0:
                    buffer["bucket_length"] = len(model_bucket_list)
                if not self.is_pipeline:
                    self.send(
                        buffer,
                        self._rollout_group_name,
                        self._weight_dst_rank_in_rollout,
                    )
                else:
                    for weight_dst_rank in self._weight_dst_rank_in_rollout:
                        self.send(
                            buffer,
                            self._rollout_group_name,
                            weight_dst_rank,
                        )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits
        logits.div_(self.cfg.algorithm.sampling_params.temperature)
        if self.enable_dynamic_batch_size:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)
        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)
            if self.enable_dynamic_batch_size:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]
            return logprobs, entropy
        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: RolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    prev_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result, compute_ref_logprobs
                    )

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = prev_logprobs
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = prev_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                rollout_prev_logprobs = prev_logprobs
                recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                advantages = advantages * torch.clamp(
                    (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                    min=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._optimizer_state_cpu_during_backward = bool(
            self.cfg.actor.get("optimizer_state_cpu_during_backward", False)
        )
        self._rollout_batch_cpu_resident = bool(
            self.cfg.actor.get("rollout_batch_cpu_resident", False)
        )
        if self._optimizer_state_cpu_during_backward and not self.enable_offload:
            raise ValueError(
                "actor.optimizer_state_cpu_during_backward requires "
                "actor.enable_offload=true."
            )
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas,
            accel_min_ctas=min_ctas,
            disable_same_device_ipc=bool(
                cfg.rollout.get("sync_weight_disable_same_device_ipc", False)
            ),
        )

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        diagnostics_cfg = cfg.algorithm.get("diagnostics", {})
        self._cosmos_action_ratio_guard = ActionRatioExplosionGuard(
            threshold=diagnostics_cfg.get("action_ratio_explosion_threshold", 10.0),
            patience=diagnostics_cfg.get("action_ratio_explosion_patience", 2),
        )
        self._cosmos_baseline_logprob_abs_tolerance = diagnostics_cfg.get(
            "baseline_logprob_abs_tolerance", None
        )
        self._cosmos_target_kl_k2 = diagnostics_cfg.get("target_kl_k2", None)
        self._cosmos_target_kl_multiplier = float(
            diagnostics_cfg.get("target_kl_multiplier", 1.5)
        )
        self._cosmos_require_multi_update_policy_drift = bool(
            diagnostics_cfg.get("require_multi_update_policy_drift", False)
        )
        self._cosmos_policy_drift_tolerance = float(
            diagnostics_cfg.get("policy_drift_tolerance", 1.0e-7)
        )
        cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
        self._cosmos_pin_trainable_param_storage = bool(
            SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.COSMOS
            and str(cosmos_cfg.get("backend", "")).lower() in {"native", "edge4b"}
            and cosmos_cfg.get("pin_trainable_param_storage", True)
        )
        self._cosmos_native_fsdp_cpu_offload = bool(
            SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.COSMOS
            and str(cosmos_cfg.get("backend", "")).lower() == "native"
            and cosmos_cfg.get("fsdp_cpu_offload", False)
        )
        self._logged_native_cpu_optimizer_residency = False
        self._cosmos_artifact_writer = CosmosArtifactWriter.from_model_cfg(
            self.cfg.actor.model
        )
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            if self._actor_parameter_offload_enabled():
                self.offload_param_and_grad()
            else:
                if self._cosmos_native_fsdp_cpu_offload:
                    residency_message = (
                        "internal FSDP2 owns CPU shard residency; optimizer "
                        "state remains on CPU"
                    )
                else:
                    residency_message = (
                        "internal FSDP2 shards remain on GPU; optimizer state "
                        "follows actor phase residency"
                    )
                self.log_on_first_rank(
                    "Keeping native Cosmos trainable Parameter identities pinned "
                    f"while {residency_message}."
                )
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _actor_parameter_offload_enabled(self) -> bool:
        """Whether model parameter storage may migrate between RL phases.

        Native Cosmos exposes internally FSDP2-sharded parameters through an
        outer registered proxy. Calling proxy.to(cpu/cuda) preserves the Python
        Parameter identity but replaces its DTensor local storage. It does not
        run the owning FSDPModule reset_sharded_param() hook, so FSDP replay can
        keep gathering stale storage while AdamW updates the new proxy storage.
        Keep those relatively small action shards resident; optimizer state is
        still offloaded independently.
        """

        return bool(
            self.enable_offload
            and not getattr(self, "_cosmos_pin_trainable_param_storage", False)
        )

    def load_optimizer(self, device_id: int) -> None:
        """Keep AdamW state beside CPU-offloaded native FSDP2 shards.

        The generic FSDP actor onloads optimizer state before training. Native
        Cosmos instead lets its internal FSDP2 policy offload sharded params
        and gradients to CPU, so AdamW state must remain on CPU as well.
        """

        if getattr(self, "_cosmos_native_fsdp_cpu_offload", False):
            non_cpu_state = [
                (str(key), str(value.device))
                for state in self.optimizer.state.values()
                if isinstance(state, dict)
                for key, value in state.items()
                if torch.is_tensor(value) and value.device.type != "cpu"
            ]
            if non_cpu_state:
                self.offload_optimizer()
            if not self._logged_native_cpu_optimizer_residency:
                self.log_on_first_rank(
                    "Keeping AdamW state on CPU for Native Cosmos internal "
                    "FSDP2 CPU offload."
                )
                self._logged_native_cpu_optimizer_residency = True
            self.is_optimizer_offloaded = True
            return
        super().load_optimizer(device_id)

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            # Plain nn.Module.load_state_dict has no DTensor awareness and
            # hard-crashes on any parameter that FSDP2's fully_shard() has
            # already sharded (e.g. an internally-parallelized Cosmos native
            # backbone). set_model_state_dict is the same DCP API used for
            # regular checkpoint resume (strategy/base.py) and transparently
            # handles both plain and DTensor-backed parameters.
            set_model_state_dict(
                model,
                model_state_dict=model_dict,
                options=StateDictOptions(
                    full_state_dict=True, broadcast_from_rank0=True
                ),
            )

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        parameter_offload_enabled = self._actor_parameter_offload_enabled()
        if parameter_offload_enabled and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        native_state_builder = (
            getattr(self.model, "native_rollout_state_dict", None)
            if self.cfg.actor.model.model_type == "cosmos"
            else None
        )
        native_binding_validator = getattr(
            self.model, "validate_native_trainable_bindings", None
        )
        if callable(native_state_builder):
            if callable(native_binding_validator):
                native_binding_validator(self.optimizer)
            state_dict = native_state_builder(cpu_offload=False)
            if callable(native_binding_validator):
                native_binding_validator(self.optimizer)
        else:
            state_dict = self.get_model_state_dict(
                cpu_offload=False, full_state_dict=True
            )
        sync_metrics = {}
        try:
            if self.cfg.actor.model.model_type == "cosmos":
                sync_metrics["sync/action_param_checksum"] = (
                    compute_action_param_checksum(state_dict.items()).detach().cpu()
                )
            for rank in self._weight_dst_rank_in_rollout:
                # The full state dict is roughly 11 GiB. A fire-and-forget
                # async send can keep that GPU buffer alive long after the
                # rollout worker has loaded the weights. Use a synchronous
                # send here (the receiver is already posted by the runner) so
                # the buffer lifetime ends before rollout generation starts.
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    rank,
                    async_op=False,
                    options=self._sync_weight_comm_options,
                )
        finally:
            del state_dict
            if parameter_offload_enabled and not self.is_weight_offloaded:
                self.offload_param_and_grad(offload_grad=True)
            clear_memory(sync=True, trim_cpu=True, collect_ipc=True)
        return sync_metrics

    def _offload_actor_state(self) -> None:
        """Return Actor model, gradients, and optimizer state to CPU."""
        if not self.enable_offload:
            return
        if not self.is_optimizer_offloaded:
            self.offload_optimizer()
        if (
            self._actor_parameter_offload_enabled()
            and not self.is_weight_offloaded
        ):
            self.offload_param_and_grad(offload_grad=True)
        clear_memory(sync=True, trim_cpu=True)

    def _defer_optimizer_onload_for_backward(self) -> bool:
        """Whether Adam state stays on CPU until each optimizer step.

        Native Cosmos replay has a much larger activation peak than the
        optimizer step. Loading Adam's first- and second-moment tensors at the
        beginning of run_training needlessly overlaps those states with every
        forward/backward microbatch. With actor offload enabled, keep the state
        on CPU while gradients are accumulated and move it to the device only
        after the final backward has released its graph.
        """

        return bool(
            self.enable_offload
            and getattr(self, "_optimizer_state_cpu_during_backward", False)
        )

    def _onload_optimizer_for_step(self) -> bool:
        """Onload deferred optimizer state and report whether it was moved."""

        if (
            not self._defer_optimizer_onload_for_backward()
            or not self.is_optimizer_offloaded
        ):
            return False
        self.load_optimizer(self.device)
        return True

    def _offload_optimizer_after_step(self, onloaded_for_step: bool) -> None:
        """Restore CPU optimizer residency after a deferred optimizer step."""

        if onloaded_for_step and not self.is_optimizer_offloaded:
            self.offload_optimizer()

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        # The previous rollout is no longer needed once its actor update has
        # finished. Clear it before waiting for the next Env trajectory so a
        # complete old replay never overlaps the next five-chunk collection.
        self._release_rollout_batch()

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        actor_channel_cfg = self.cfg.runner.get("actor_channel", {})
        route_by_node = bool(actor_channel_cfg.get("route_by_node", False))
        if route_by_node and not input_channel.is_distributed:
            raise RuntimeError(
                "runner.actor_channel.route_by_node requires a distributed "
                "Actor channel."
            )
        channel_key = (
            input_channel.node_local_key("actor_trajectory")
            if route_by_node
            else None
        )
        recv_list = []
        for _ in range(split_num):
            if channel_key is None:
                work = input_channel.get(async_op=True)
            else:
                work = input_channel.get(key=channel_key, async_op=True)
            try:
                trajectory: Trajectory = await work.async_wait()
            finally:
                # AsyncChannelCommWork otherwise owns a completed callback
                # chain whose Future result still references the original
                # shared-memory trajectory. The merged Actor batch below is
                # an independent torch.cat allocation, so the communication
                # work can release its copy immediately.
                work.release()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        # Drop the source Trajectory objects as soon as torch.cat has produced
        # the private Actor batch. Explicit cleanup matters for Native Cosmos:
        # one received replay is several GiB and cyclic GC otherwise leaves
        # its shared mappings charged through the next update.
        recv_list.clear()
        del recv_list
        del trajectory
        del work
        clear_memory(sync=False, trim_cpu=True)

        if getattr(self, "_rollout_batch_cpu_resident", False):
            before = _nested_tensor_bytes_by_device(self.rollout_batch)
            self.rollout_batch = put_tensor_device(self.rollout_batch, "cpu")
            after = _nested_tensor_bytes_by_device(self.rollout_batch)
            non_cpu_after = {
                device: size for device, size in after.items() if device != "cpu"
            }
            if non_cpu_after:
                raise RuntimeError(
                    "Actor rollout batch CPU residency failed: "
                    f"remaining_non_cpu_bytes={non_cpu_after}."
                )
            clear_memory(sync=True, trim_cpu=True, collect_ipc=True)
            self.log_on_first_rank(
                "Actor rollout batch moved to CPU before advantage/replay: "
                f"before_bytes={before}, after_bytes={after}."
            )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _release_rollout_batch(self) -> None:
        """Drop replay tensors and return free CPU heap pages between RL steps."""
        rollout_batch = getattr(self, "rollout_batch", None)
        if rollout_batch is not None:
            self.rollout_batch = None
            del rollout_batch
        clear_memory(sync=False, trim_cpu=True)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            if self.cfg.algorithm.adv_type == "grpo_action_suffix":
                action_reward_threshold = self.cfg.algorithm.get(
                    "reward_filter_action_threshold", 0.9
                )
                reward_filter_mask = (reward_matrix > action_reward_threshold).any(
                    dim=(1, 2)
                )  # [n_prompts]
            else:
                reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
                mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

                # mask
                reward_filter_mask = (
                    mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
                ) & (
                    mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
                )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def _cross_rank_group_enabled(self) -> bool:
        cross_rank_group = self.cfg.algorithm.get("cross_rank_group", None)
        return bool(
            cross_rank_group is not None
            and cross_rank_group.get("enabled", False)
        )

    def _compute_cross_rank_grpo_advantages(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Gather only scalar rollout records and compute explicit-ID GRPO."""
        batch_size = int(self.rollout_batch["rewards"].shape[1])
        local_error = 0
        try:
            prepared = preprocess_embodied_advantages_inputs(
                rewards=self.rollout_batch["rewards"],
                dones=self.rollout_batch["dones"],
                loss_mask=self.rollout_batch.get("loss_mask", None),
                loss_mask_sum=self.rollout_batch.get("loss_mask_sum", None),
                reward_type=self.cfg.algorithm.reward_type,
                adv_type=str(self.cfg.algorithm.adv_type),
            )
            local_scores = calculate_episode_scores(
                prepared["rewards"], prepared["dones"]
            ).to(torch.float32)
            flat_loss_mask = prepared["loss_mask"]
            local_valid = (
                torch.ones(batch_size, dtype=torch.float32)
                if flat_loss_mask is None
                else flat_loss_mask.any(dim=0).to(torch.float32)
            )
            local_step_rewards = prepared["rewards"].to(torch.float32).cpu()
            local_step_dones = prepared["dones"].to(torch.float32).cpu()
            local_step_loss_mask = (
                torch.ones_like(local_step_rewards, dtype=torch.float32)
                if flat_loss_mask is None
                else flat_loss_mask.to(torch.float32).cpu()
            )
        except Exception:
            local_error = 1
            local_scores = torch.zeros(batch_size, dtype=torch.float32)
            local_valid = torch.zeros(batch_size, dtype=torch.float32)
            prepared = None
            local_step_rewards = torch.zeros(1, batch_size)
            local_step_dones = torch.zeros(2, batch_size)
            local_step_loss_mask = torch.zeros(1, batch_size)

        metadata_names = (
            "rollout_uid",
            "global_group_id",
            "group_member_id",
            "reset_seed",
            "vision_noise_seed",
            "action_noise_seed",
            "ctrl_world_noise_seed",
            "reset_state_ids",
            "source_env_rank",
            "local_env_id",
            "update_id",
            "logical_round_id",
            "physical_wave_id",
            "group_slot",
            "reset_episode",
            "shuffle_seed",
        )
        local_metadata = {}
        for field_name in metadata_names:
            try:
                value = self.rollout_batch.get(field_name)
                if (
                    not torch.is_tensor(value)
                    or value.ndim < 2
                    or value.shape[1] != batch_size
                    or value.numel() % batch_size != 0
                ):
                    raise ValueError(f"malformed metadata field {field_name}")
                flattened = value.reshape(value.shape[0], batch_size, -1)[
                    :, :, 0
                ]
                if field_name not in (
                    "vision_noise_seed",
                    "action_noise_seed",
                    "ctrl_world_noise_seed",
                    "reset_state_ids",
                    "chunk_id",
                    "seed_nonce",
                ) and not torch.equal(
                    flattened, flattened[:1].expand_as(flattened)
                ):
                    raise ValueError(
                        f"metadata field {field_name} changed within a trajectory"
                    )
                local_metadata[field_name] = flattened[0].to(torch.int64).cpu()
            except Exception:
                local_error = 1
                local_metadata[field_name] = torch.full(
                    (batch_size,), -1, dtype=torch.int64
                )

        try:
            actions = self.rollout_batch.get("actions")
            if not torch.is_tensor(actions):
                raise ValueError("missing or malformed actions")
            flattened_actions = (
                actions.transpose(0, 1).reshape(batch_size, -1).float()
            )
            weights = (
                torch.arange(
                    1,
                    flattened_actions.shape[1] + 1,
                    dtype=torch.float32,
                )
                .remainder(997)
                .add(1)
                .div(997.0)
            )
            action_checksums = (flattened_actions.cpu() * weights).sum(dim=1)
        except Exception:
            local_error = 1
            action_checksums = torch.zeros(batch_size, dtype=torch.float32)

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "cross-rank GRPO requires an initialized actor WORLD process group"
            )
        collective_device = select_cross_rank_collective_device(
            torch.distributed.get_backend(),
            accelerator_device_type=Worker.torch_device_type,
            local_rank=int(os.environ.get("LOCAL_RANK", self._rank)),
        )

        started_at = time.monotonic()
        count_and_error = torch.tensor(
            [batch_size, local_error],
            dtype=torch.int64,
            device=collective_device,
        )
        gathered_count_and_error = [
            torch.empty_like(count_and_error)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(
            gathered_count_and_error, count_and_error
        )
        counts = [int(item[0].item()) for item in gathered_count_and_error]
        max_rows = max(counts)

        local_ids = torch.full(
            (max_rows, len(metadata_names)),
            -1,
            dtype=torch.int64,
            device=collective_device,
        )
        for column, field_name in enumerate(metadata_names):
            local_ids[:batch_size, column] = local_metadata[field_name].to(
                collective_device
            )
        n_steps = int(local_step_rewards.shape[0])
        value_width = 3 + n_steps + (n_steps + 1) + n_steps
        local_values = torch.zeros(
            max_rows, value_width, dtype=torch.float32, device=collective_device
        )
        local_values[:batch_size, 0] = local_scores.to(collective_device)
        local_values[:batch_size, 1] = local_valid.to(collective_device)
        local_values[:batch_size, 2] = action_checksums.to(collective_device)
        reward_start = 3
        dones_start = reward_start + n_steps
        mask_start = dones_start + n_steps + 1
        local_values[:batch_size, reward_start:dones_start] = (
            local_step_rewards.transpose(0, 1).to(collective_device)
        )
        local_values[:batch_size, dones_start:mask_start] = (
            local_step_dones.transpose(0, 1).to(collective_device)
        )
        local_values[:batch_size, mask_start:] = (
            local_step_loss_mask.transpose(0, 1).to(collective_device)
        )

        gathered_ids = [
            torch.empty_like(local_ids)
            for _ in range(torch.distributed.get_world_size())
        ]
        gathered_values = [
            torch.empty_like(local_values)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_ids, local_ids)
        torch.distributed.all_gather(gathered_values, local_values)
        gather_ms = (time.monotonic() - started_at) * 1000.0

        all_ids = torch.cat(
            [value[:count] for value, count in zip(gathered_ids, counts, strict=True)]
        )
        all_values = torch.cat(
            [
                value[:count]
                for value, count in zip(gathered_values, counts, strict=True)
            ]
        )
        if any(int(item[1].item()) for item in gathered_count_and_error):
            raise ValueError(
                "one or more actor ranks supplied malformed cross-rank GRPO "
                "metadata or episode scores"
            )

        assert prepared is not None
        if str(self.cfg.algorithm.adv_type) == "grpo_action_suffix":
            global_advantages, metrics = (
                validate_and_compute_group_suffix_advantages(
                    rollout_uid=all_ids[:, 0],
                    global_group_id=all_ids[:, 1],
                    group_member_id=all_ids[:, 2],
                    rewards=all_values[:, reward_start:dones_start].transpose(0, 1),
                    dones=all_values[:, dones_start:mask_start]
                    .transpose(0, 1)
                    .to(torch.bool),
                    loss_mask=all_values[:, mask_start:].transpose(0, 1).to(
                        torch.bool
                    ),
                    group_size=int(self.cfg.algorithm.group_size),
                    gamma=float(self.cfg.algorithm.get("gamma", 0.95)),
                )
            )
            global_uid_to_row = {
                int(uid): row for row, uid in enumerate(all_ids[:, 0].tolist())
            }
            local_rows = torch.tensor(
                [
                    global_uid_to_row[int(uid)]
                    for uid in local_metadata["rollout_uid"].tolist()
                ],
                dtype=torch.long,
                device=global_advantages.device,
            )
            expanded = global_advantages.index_select(1, local_rows).to(
                self.rollout_batch["rewards"].device
            )
        else:
            global_advantages, metrics = validate_and_compute_group_advantages(
                rollout_uid=all_ids[:, 0],
                global_group_id=all_ids[:, 1],
                group_member_id=all_ids[:, 2],
                scores=all_values[:, 0],
                group_size=int(self.cfg.algorithm.group_size),
            )
            advantage_by_uid = {
                int(uid): advantage
                for uid, advantage in zip(
                    all_ids[:, 0].tolist(), global_advantages, strict=True
                )
            }
            local_advantages = torch.stack(
                [
                    advantage_by_uid[int(uid)]
                    for uid in local_metadata["rollout_uid"].tolist()
                ]
            ).to(self.rollout_batch["rewards"].device)
            loss_mask = prepared["loss_mask"]
            if loss_mask is None:
                loss_mask = torch.ones(
                    prepared["n_steps"],
                    prepared["batch_size"],
                    dtype=local_advantages.dtype,
                    device=local_advantages.device,
                )
            else:
                loss_mask = loss_mask.to(local_advantages.device)
            expanded = (
                torch.zeros_like(loss_mask) + local_advantages.reshape(1, -1)
            ) * loss_mask
        result = postprocess_embodied_advantages_outputs(
            advantages=expanded,
            num_chunk=prepared["num_chunk"],
            chunk_size=prepared["chunk_size"],
        )

        metrics.update(
            validate_cross_rank_runtime_contract(
                metadata={
                    field_name: all_ids[:, column]
                    for column, field_name in enumerate(metadata_names)
                },
                validity=all_values[:, 1],
                action_checksums=all_values[:, 2],
                group_size=int(self.cfg.algorithm.group_size),
                members_per_rank=int(
                    self.cfg.algorithm.cross_rank_group.members_per_rank
                ),
                per_member_vision_seed=bool(
                    self.cfg.algorithm.cross_rank_group.get(
                        "per_member_vision_seed", False
                    )
                ),
            )
        )
        metrics["cross_rank_group_gather_ms"] = float(gather_ms)
        return result, metrics

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "suffix_success_anchor_enabled": self.cfg.algorithm.get(
                "suffix_success_anchor_enabled", False
            ),
            "suffix_success_threshold": self.cfg.algorithm.get(
                "suffix_success_threshold", 0.9
            ),
            "suffix_success_trace_decay": self.cfg.algorithm.get(
                "suffix_success_trace_decay", 0.95
            ),
            "suffix_gamma_min": self.cfg.algorithm.get("suffix_gamma_min", 0.9),
            "suffix_gamma_max": self.cfg.algorithm.get("suffix_gamma_max", 1.0),
            "suffix_latent_gamma_enabled": self.cfg.algorithm.get(
                "suffix_latent_gamma_enabled", False
            ),
            "suffix_latent_gamma_beta": self.cfg.algorithm.get(
                "suffix_latent_gamma_beta", 0.02
            ),
            "suffix_latent_gamma_min": self.cfg.algorithm.get(
                "suffix_latent_gamma_min", 0.98
            ),
            "suffix_latent_gamma_after_success": self.cfg.algorithm.get(
                "suffix_latent_gamma_after_success", True
            ),
            "latent_motion": self.rollout_batch.get("latent_motion", None),
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        # Mutated in place by compute_grpo_action_suffix_advantages (no-op for
        # other adv_types) with the per-chunk reward before/after gamma
        # discounting, so we can report per-action-chunk reward trends.
        chunk_reward_log: dict[str, torch.Tensor] = {}
        kwargs["chunk_reward_log"] = chunk_reward_log

        cross_rank_metrics = {}
        if self._cross_rank_group_enabled():
            advantages_and_returns, cross_rank_metrics = (
                self._compute_cross_rank_grpo_advantages()
            )
        else:
            advantages_and_returns = calculate_adv_and_returns(**kwargs)
        if (
            self.cfg.algorithm.get("libero_verifier_weight_advantages", False)
            and self.rollout_batch.get("libero_verified_weight", None) is not None
        ):
            verifier_weight = self.rollout_batch["libero_verified_weight"].to(
                advantages_and_returns["advantages"].device,
                dtype=advantages_and_returns["advantages"].dtype,
            )
            if verifier_weight.shape != advantages_and_returns["advantages"].shape:
                if (
                    advantages_and_returns["advantages"].shape[-1] == 1
                    and verifier_weight.shape[:-1]
                    == advantages_and_returns["advantages"].shape[:-1]
                ):
                    verifier_weight = verifier_weight.mean(dim=-1, keepdim=True)
                if (
                    verifier_weight.shape[:-1]
                    == advantages_and_returns["advantages"].shape[:-1]
                    and verifier_weight.shape[-1] == 1
                ):
                    verifier_weight = verifier_weight.expand_as(
                        advantages_and_returns["advantages"]
                    )
                else:
                    raise ValueError(
                        "libero_verified_weight shape "
                        f"{verifier_weight.shape} does not match advantages shape "
                        f"{advantages_and_returns['advantages'].shape}"
                    )
            advantages_and_returns["advantages"] = (
                advantages_and_returns["advantages"] * verifier_weight
            )

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})
        # NOTE: chunk_reward_log tensors are diagnostics-only (per-chunk and
        # per-group shapes, not the [n_chunk_step, bsz, ...] shape every
        # other rollout_batch entry has), so they must stay out of
        # self.rollout_batch -- run_training()'s process_nested_dict_for_train
        # reshapes+shuffle-indexes every tensor in there assuming that common
        # shape and will IndexError on anything that doesn't match it.
        self._initialize_trajectory_records()

        rollout_metrics = compute_rollout_metrics(
            self.rollout_batch, chunk_reward_log=chunk_reward_log
        )
        rollout_metrics.update(
            {f"train/{key}": value for key, value in cross_rank_metrics.items()}
        )
        return rollout_metrics

    def _trajectory_record_cfg(self):
        """Return the optional production trajectory-record configuration."""
        return self.cfg.algorithm.get("trajectory_records", {})

    @staticmethod
    def _record_tensor_values(value: torch.Tensor) -> list[float]:
        """Convert one diagnostic tensor to finite JSON floats."""
        return [
            float(item)
            for item in value.detach().to(torch.float64).cpu().reshape(-1).tolist()
        ]

    def _trajectory_record_gpu_index(self) -> int:
        """Return the scheduler-assigned physical GPU slot on this node."""
        gpu_index = int(getattr(self, "_local_accelerator_rank", -1))
        if gpu_index < 0:
            raise ValueError(
                "trajectory recorder requires a scheduler-assigned "
                "LOCAL_ACCELERATOR_RANK"
            )
        return gpu_index

    def _initialize_trajectory_records(self) -> None:
        """Build one scalar-only record per locally owned trajectory."""
        record_cfg = self._trajectory_record_cfg()
        if not bool(record_cfg.get("enabled", False)):
            self._trajectory_records = {}
            return
        provenance_mode = str(record_cfg.get("provenance_mode", "strict"))
        if provenance_mode not in {"strict", "lightweight_analysis"}:
            raise ValueError(
                "trajectory recorder provenance_mode must be strict or "
                f"lightweight_analysis, got {provenance_mode!r}"
            )

        checksum_provenance = {
            "manifest_path": None,
            "manifest_sha256": None,
            "mode": provenance_mode,
            "artifacts": {},
        }
        version_provenance = {}
        input_audit_sha256 = None
        terminal_goal_provenance = {}
        seed_manifest_path: str | Path = ""
        seed_manifest_sha256 = None
        seed_manifest = {}
        duck_provenance = None
        duck_seed_groups = None
        lightweight_duck_episodes = None
        split_manifest = str(record_cfg.get("split_manifest", "")).strip()

        if provenance_mode == "strict":
            checksum_manifest_path = Path(
                str(record_cfg.get("checkpoint_checksum_manifest", ""))
            )
            if not checksum_manifest_path.is_file():
                raise FileNotFoundError(
                    "trajectory recorder checksum manifest is missing: "
                    f"{checksum_manifest_path}"
                )
            checksum_payload = checksum_manifest_path.read_bytes()
            checksum_manifest = json.loads(checksum_payload)
            if checksum_manifest.get("mode") != "full":
                raise ValueError(
                    "Formal trajectory recording requires full input checksums, got "
                    f"{checksum_manifest.get('mode')!r}."
                )
            compact_checksums = {}
            for label, artifact in checksum_manifest.get("artifacts", {}).items():
                digest = (
                    artifact.get("sha256")
                    or artifact.get("manifest_sha256")
                    or artifact.get("sampled_sha256")
                )
                if not digest:
                    raise ValueError(f"Input checksum is missing for {label}.")
                compact_checksums[label] = str(digest)
            checksum_provenance = {
                "manifest_path": str(checksum_manifest_path),
                "manifest_sha256": hashlib.sha256(checksum_payload).hexdigest(),
                "mode": str(checksum_manifest["mode"]),
                "artifacts": compact_checksums,
            }
            run_manifest_path = Path(str(record_cfg.get("run_manifest", "")))
            if not run_manifest_path.is_file():
                raise FileNotFoundError(
                    f"trajectory recorder run manifest is missing: {run_manifest_path}"
                )
            run_manifest_payload = run_manifest_path.read_bytes()
            run_manifest = json.loads(run_manifest_payload)
            version_provenance = {
                "manifest_path": str(run_manifest_path),
                "manifest_sha256": hashlib.sha256(
                    run_manifest_payload
                ).hexdigest(),
                "repo_revision": run_manifest.get("repo_revision"),
                "cosmos_framework_revision": run_manifest.get(
                    "cosmos_framework_revision"
                ),
                "ctrl_world_revision": run_manifest.get("ctrl_world_revision"),
            }
            input_audit_path = Path(str(record_cfg.get("input_audit", "")))
            if not input_audit_path.is_file():
                raise FileNotFoundError(
                    f"trajectory recorder input audit is missing: {input_audit_path}"
                )
            input_audit_sha256 = hashlib.sha256(
                input_audit_path.read_bytes()
            ).hexdigest()
            terminal_metadata_path = Path(
                str(record_cfg.get("terminal_goal_metadata", ""))
            )
            if not terminal_metadata_path.is_file():
                raise FileNotFoundError(
                    "trajectory recorder terminal-goal metadata is missing: "
                    f"{terminal_metadata_path}"
                )
            terminal_metadata_payload = terminal_metadata_path.read_bytes()
            terminal_metadata = json.loads(terminal_metadata_payload)
            terminal_goal_provenance = {
                "metadata_path": str(terminal_metadata_path),
                "metadata_sha256": hashlib.sha256(
                    terminal_metadata_payload
                ).hexdigest(),
                "saved_sha256": terminal_metadata.get("saved_sha256", {}),
            }
            seed_manifest_path = (
                Path(str(record_cfg.get("seed_manifest_dir", "")))
                / f"update_{int(self.global_step):04d}.json"
            )
            if not seed_manifest_path.is_file():
                raise FileNotFoundError(
                    "trajectory recorder seed manifest is missing: "
                    f"{seed_manifest_path}"
                )
            seed_manifest_payload = seed_manifest_path.read_bytes()
            seed_manifest_sha256 = hashlib.sha256(seed_manifest_payload).hexdigest()
            seed_manifest = json.loads(seed_manifest_payload)
            if split_manifest:
                reward_cfg = self.cfg.get("reward", {})
                success_models = reward_cfg.get("success_models", {})
                duck_provenance = build_duck_trajectory_provenance(
                    split_manifest=split_manifest,
                    success_models=success_models,
                )
                duck_seed_groups = index_duck_seed_manifest_groups(
                    seed_manifest,
                    expected_update_id=int(self.global_step),
                )
        elif split_manifest:
            # Retain scalar analysis records without requiring
            # preparation-only audit manifests or hashing model weights per rank.
            contract = load_duck_episode_split(split_manifest, "training")
            assert_duck_episode_split_unchanged(contract)
            active_colors = tuple(
                str(color)
                for color in self.cfg.get("duck", {}).get(
                    "colors", contract.color_order
                )
            )
            unknown_colors = set(active_colors) - set(contract.color_order)
            if unknown_colors:
                raise ValueError(
                    "trajectory recorder received Duck colors outside the "
                    f"validated split: {sorted(unknown_colors)}"
                )
            lightweight_duck_episodes = {}
            for color in active_colors:
                for episode_id in contract.episodes_by_color[color]:
                    lightweight_duck_episodes[int(episode_id)] = {
                        "color": color,
                        "manifest_path": contract.manifest_path,
                        "manifest_sha256": contract.manifest_sha256,
                    }
        required = (
            "rollout_uid",
            "global_group_id",
            "group_member_id",
            "source_env_rank",
            "local_env_id",
            "reset_seed",
            "vision_noise_seed",
            "action_noise_seed",
            "ctrl_world_noise_seed",
            "update_id",
            "logical_round_id",
            "physical_wave_id",
            "group_slot",
            "chunk_id",
            "reset_episode",
            "seed_nonce",
            "shuffle_seed",
            "retry_count",
            "video_similarity_rewards",
            "terminal_goal_rewards",
            "continuous_combined_rewards",
            "reward_model_probabilities",
            "rewards",
            "advantages",
            "prev_logprobs",
        )
        missing = [
            name
            for name in required
            if not torch.is_tensor(self.rollout_batch.get(name))
        ]
        if missing:
            raise ValueError(f"trajectory recorder missing fields: {missing}")
        batch_size = int(self.rollout_batch["rewards"].shape[1])
        num_chunks = int(self.rollout_batch["rewards"].shape[0])
        for name in required:
            value = self.rollout_batch[name]
            if value.ndim < 2 or tuple(value.shape[:2]) != (num_chunks, batch_size):
                raise ValueError(
                    "trajectory recorder field has an invalid leading shape: "
                    f"{name}={tuple(value.shape)}, "
                    f"expected ({num_chunks}, {batch_size}, ...)"
                )

        def field(name: str, trajectory_index: int) -> torch.Tensor | None:
            value = self.rollout_batch.get(name)
            if not torch.is_tensor(value):
                return None
            return value[:, trajectory_index].detach().cpu()

        def first_int(name: str, trajectory_index: int, default: int = -1) -> int:
            value = field(name, trajectory_index)
            if value is None or value.numel() == 0:
                return default
            return int(value.reshape(-1)[0].item())

        records: dict[int, dict] = {}
        gpu_index = self._trajectory_record_gpu_index()
        success_cfg = self.cfg.reward.get("success_classifier", {})
        trajectory_frames = int(
            record_cfg.get(
                "trajectory_frames",
                num_chunks * int(self.cfg.actor.model.num_action_chunks),
            )
        )
        for trajectory_index in range(batch_size):
            uid = first_int("rollout_uid", trajectory_index)
            update_id = first_int(
                "update_id", trajectory_index, int(self.global_step)
            )
            logical_round_id = first_int(
                "logical_round_id", trajectory_index
            )
            group_slot = first_int("group_slot", trajectory_index)
            member_id = first_int("group_member_id", trajectory_index)
            reset_episode = first_int("reset_episode", trajectory_index)
            reset_seed = first_int("reset_seed", trajectory_index)
            ctrl_seed_tensor = field("ctrl_world_noise_seed", trajectory_index)
            ctrl_seed_values = {
                int(value)
                for value in ctrl_seed_tensor.reshape(-1).tolist()
            }
            if len(ctrl_seed_values) != 1:
                raise ValueError(
                    "Duck trajectory changed Ctrl-World seed between chunks."
                )
            ctrl_seed = next(iter(ctrl_seed_values))
            action_seed_tensor = field("action_noise_seed", trajectory_index)
            action_noise_seeds = [
                int(value)
                for value in action_seed_tensor.reshape(num_chunks, -1)[:, 0]
                .tolist()
            ]
            duck_record_provenance = None
            if duck_provenance is not None:
                episode_provenance = duck_provenance["episodes"].get(
                    reset_episode
                )
                if episode_provenance is None:
                    raise ValueError(
                        "Duck training rollout used an episode outside the "
                        f"frozen training split: {reset_episode}."
                    )
                reset_color = str(episode_provenance["color"])
                validate_duck_trajectory_seed_alignment(
                    seed_groups=duck_seed_groups,
                    logical_round_id=logical_round_id,
                    group_slot=group_slot,
                    reset_episode=reset_episode,
                    reset_color=reset_color,
                    reset_seed=reset_seed,
                    member_id=member_id,
                    action_noise_seeds=action_noise_seeds,
                    ctrl_seed=ctrl_seed,
                )
                duck_record_provenance = {
                    "episode": reset_episode,
                    "color": reset_color,
                    "split": "train",
                    "series": "train_pre_update",
                    "policy_stage": "pre_update",
                    "model_path": episode_provenance["model_path"],
                    "model_hash": episode_provenance["model_sha256"],
                    "success_model_path": episode_provenance["model_path"],
                    "success_model_sha256": episode_provenance[
                        "model_sha256"
                    ],
                    "reward_model_path": episode_provenance["model_path"],
                    "reward_model_sha256": episode_provenance[
                        "model_sha256"
                    ],
                    "split_manifest_path": duck_provenance["manifest_path"],
                    "split_manifest_sha256": duck_provenance[
                        "manifest_sha256"
                    ],
                    **duck_target_frame_provenance(
                        self.cfg, reset_episode
                    ),
                }
            elif lightweight_duck_episodes is not None:
                episode_provenance = lightweight_duck_episodes.get(reset_episode)
                if episode_provenance is None:
                    raise ValueError(
                        "Duck training rollout used an episode outside the active "
                        f"validated color subset: {reset_episode}."
                    )
                duck_record_provenance = {
                    "episode": reset_episode,
                    "color": str(episode_provenance["color"]),
                    "split": "train",
                    "series": "train_pre_update",
                    "policy_stage": "pre_update",
                    "split_manifest_path": episode_provenance["manifest_path"],
                    "split_manifest_sha256": episode_provenance[
                        "manifest_sha256"
                    ],
                }
            video_rewards = field("video_similarity_rewards", trajectory_index)
            terminal_rewards = field("terminal_goal_rewards", trajectory_index)
            combined_rewards = field(
                "continuous_combined_rewards", trajectory_index
            )
            if video_rewards is None:
                video_rewards = field("rewards", trajectory_index)
            if terminal_rewards is None:
                terminal_rewards = torch.zeros_like(video_rewards)
            if combined_rewards is None:
                combined_rewards = field("rewards", trajectory_index)
            trajectory_reward = float(video_rewards.to(torch.float64).sum().item())
            lastframe_reward = float(
                terminal_rewards.to(torch.float64).sum().item()
            )
            combined_reward = float(
                combined_rewards.to(torch.float64).sum().item()
            )
            probability_tensor = field(
                "reward_model_probabilities", trajectory_index
            )
            if probability_tensor is None:
                probability_tensor = torch.zeros(4, dtype=torch.float32)
            decision = classify_terminal_probabilities(
                probability_tensor.reshape(1, -1), success_cfg
            )
            final_probabilities = self._record_tensor_values(
                decision["sampled_probabilities"][0]
            )
            probability_max = float(decision["probability_max"][0].item())
            last_probability = float(decision["last_probability"][0].item())
            positive_ratio = float(decision["positive_ratio"][0].item())
            model_probability = float(
                decision["reported_probability"][0].item()
            )
            model_success = bool(decision["success"][0].item())
            success_threshold = float(decision["threshold"].item())
            training_reward_source = str(
                self.cfg.reward.get("training_source", "continuous_combined")
            )
            success_reward = (
                combined_reward
                if training_reward_source == "success_binary"
                else 0.0
            )
            advantages = field("advantages", trajectory_index)
            chunk_advantages = [
                float(value.to(torch.float64).mean().item())
                for value in advantages
            ]
            chunk_combined_rewards = [
                float(value.to(torch.float64).sum().item())
                for value in combined_rewards
            ]
            old_logprobs = field("prev_logprobs", trajectory_index)
            old_chunk_logprobs = [
                float(value.to(torch.float64).sum().item())
                for value in old_logprobs
            ]
            nonce_tensor = field("seed_nonce", trajectory_index)
            retry_tensor = field("retry_count", trajectory_index)
            retry_counts_per_chunk = (
                [
                    int(value)
                    for value in retry_tensor.reshape(num_chunks, -1)[:, 0]
                    .tolist()
                ]
                if retry_tensor is not None
                else [0] * num_chunks
            )
            record = {
                "update": update_id,
                "logical_round": logical_round_id,
                "physical_wave": first_int("physical_wave_id", trajectory_index),
                "group_slot": group_slot,
                "group_id": first_int("global_group_id", trajectory_index),
                "member_id": member_id,
                "rollout_uid": uid,
                "source_env_rank": first_int("source_env_rank", trajectory_index),
                "local_env_id": first_int("local_env_id", trajectory_index),
                "actor_rank": int(self._rank),
                "node": os.uname().nodename,
                # LOCAL_RANK is always zero when Ray isolates each worker to
                # one visible GPU. The scheduler preserves the physical
                # node-local slot separately as LOCAL_ACCELERATOR_RANK.
                "gpu": gpu_index,
                "reset_episode": reset_episode,
                "reset_seed": reset_seed,
                "cosmos_joint_seeds": [
                    int(value)
                    for value in action_seed_tensor.reshape(num_chunks, -1)[:, 0]
                    .tolist()
                ],
                "cosmos_joint_seeds_uint32": [
                    map_semantic_seed_to_uint32(
                        "cosmos_joint_initial_noise", int(value)
                    )
                    for value in action_seed_tensor.reshape(
                        num_chunks, -1
                    )[:, 0].tolist()
                ],
                "cosmos_seed_nonces": (
                    [
                        int(value)
                        for value in nonce_tensor.reshape(num_chunks, -1)[:, 0]
                        .tolist()
                    ]
                    if nonce_tensor is not None
                    else [0] * num_chunks
                ),
                "vision_action_seed_equal": bool(
                    torch.equal(
                        field("vision_noise_seed", trajectory_index),
                        action_seed_tensor,
                    )
                ),
                "ctrl_seed": ctrl_seed,
                "shuffle_seed": first_int("shuffle_seed", trajectory_index),
                "shuffle_version": "trajectory-aware-v1",
                "seed_derivation_version": "blake2b-int63-v2",
                "seed_manifest_path": str(seed_manifest_path),
                "seed_manifest_sha256": seed_manifest_sha256,
                "retry_counts_per_chunk": retry_counts_per_chunk,
                "retry_count": int(sum(retry_counts_per_chunk)),
                "max_retries_per_sampling_call": int(
                    self.cfg.algorithm.get("rollout_retry", {}).get(
                        "max_retries", 0
                    )
                ),
                "trajectory_frames": trajectory_frames,
                "success": model_success,
                "model_success": model_success,
                "success_threshold": success_threshold,
                "success_decision_rule": decision["rule"],
                "success_last4_probabilities": final_probabilities[-4:],
                "success_terminal_probabilities": final_probabilities,
                "success_terminal_sample_indices": [
                    int(value) for value in decision["sample_indices"].tolist()
                ],
                "success_probability_max": probability_max,
                "success_last_probability": last_probability,
                "success_positive_ratio": positive_ratio,
                "model_success_probability": model_probability,
                "training_reward_source": training_reward_source,
                "success_reward": success_reward,
                "trajectory_mse": -trajectory_reward / trajectory_frames,
                "trajectory_similarity": trajectory_reward / trajectory_frames,
                "trajectory_reward": trajectory_reward,
                "lastframe_mse": -lastframe_reward / trajectory_frames,
                "lastframe_similarity": lastframe_reward / trajectory_frames,
                "lastframe_reward": lastframe_reward,
                "combined_reward": combined_reward,
                "chunk_rewards": chunk_combined_rewards,
                "chunk_advantages": chunk_advantages,
                "old_chunk_logprobs": old_chunk_logprobs,
                "old_logprob": float(sum(old_chunk_logprobs)),
                "sampler_label": "unipc_euler_surrogate",
                "optimizer_observations": [],
                "valid": True,
                "validity_status": "valid",
                "exception": None,
                "provenance_mode": provenance_mode,
                "checkpoint_checksum_manifest": (
                    str(record_cfg.get("checkpoint_checksum_manifest", ""))
                    if provenance_mode == "strict"
                    else None
                ),
                "input_checksums": checksum_provenance,
                "input_audit_sha256": input_audit_sha256,
                "version_checksums": version_provenance,
                "terminal_goal_checksums": terminal_goal_provenance,
            }
            if duck_record_provenance is not None:
                record.update(duck_record_provenance)
            records[uid] = record
        self._trajectory_records = records

    def _update_trajectory_record_ratios(
        self,
        batch: dict[str, torch.Tensor],
        *,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
    ) -> None:
        """Accumulate per-chunk PPO diagnostics without retaining replay tensors."""
        records = getattr(self, "_trajectory_records", None)
        if not records:
            return
        uid_tensor = batch.get("rollout_uid")
        chunk_tensor = batch.get("chunk_id")
        if not torch.is_tensor(uid_tensor):
            raise ValueError("trajectory recorder requires rollout_uid in train batch")
        batch_size = int(old_logprobs.shape[0])
        for index in range(batch_size):
            uid = int(uid_tensor[index].reshape(-1)[0].item())
            record = records[uid]
            old = old_logprobs[index].detach().to(torch.float64).reshape(-1)
            new = new_logprobs[index].detach().to(torch.float64).reshape(-1)
            raw_log_ratio = new - old
            log_ratio = raw_log_ratio.clamp(
                min=float(self.cfg.algorithm.get("clip_log_ratio_min", -20.0)),
                max=float(self.cfg.algorithm.get("clip_log_ratio_max", 20.0)),
            )
            ratio = log_ratio.exp()
            clipped = (ratio < 1.0 - float(self.cfg.algorithm.clip_ratio_low)) | (
                ratio > 1.0 + float(self.cfg.algorithm.clip_ratio_high)
            )
            chunk_id = (
                int(chunk_tensor[index].reshape(-1)[0].item())
                if torch.is_tensor(chunk_tensor)
                else -1
            )
            record["optimizer_observations"].append(
                {
                    "optimizer_step": int(self.optimizer_steps),
                    "chunk_id": chunk_id,
                    "old_logprob": float(old.sum().item()),
                    "new_logprob": float(new.sum().item()),
                    "log_ratio_mean": float(raw_log_ratio.mean().item()),
                    "ratio_mean": float(ratio.mean().item()),
                    "ratio_min": float(ratio.min().item()),
                    "ratio_max": float(ratio.max().item()),
                    "clip_fraction": float(clipped.float().mean().item()),
                }
            )

    def _finalize_trajectory_records(self) -> None:
        """Write rank-local JSONL and atomically merge all scalar shards."""
        records = getattr(self, "_trajectory_records", None)
        record_cfg = self._trajectory_record_cfg()
        if not records or not bool(record_cfg.get("enabled", False)):
            return
        output_dir = Path(str(record_cfg.output_dir))
        shard_dir = output_dir / ".rank_shards"
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        update_id = int(getattr(self, "global_step", self.version))
        for record in records.values():
            observations = record["optimizer_observations"]
            expected_chunks = int(
                self.cfg.algorithm.get("trajectory_chunks", 5)
            )
            if len(observations) != expected_chunks:
                raise ValueError(
                    "trajectory record has an invalid optimizer observation "
                    f"count: {len(observations)} != {expected_chunks}"
                )
            observations_by_chunk = {
                int(item["chunk_id"]): item for item in observations
            }
            if sorted(observations_by_chunk) != list(range(expected_chunks)):
                raise ValueError(
                    "trajectory record is missing or duplicates optimizer "
                    f"chunk observations: {sorted(observations_by_chunk)}"
                )
            ordered = [
                observations_by_chunk[chunk_id]
                for chunk_id in range(expected_chunks)
            ]
            record["new_chunk_logprobs"] = [
                float(item["new_logprob"]) for item in ordered
            ]
            record["chunk_raw_log_ratios"] = [
                float(new - old)
                for old, new in zip(
                    record["old_chunk_logprobs"],
                    record["new_chunk_logprobs"],
                    strict=True,
                )
            ]
            record["chunk_log_ratios"] = [
                min(
                    float(self.cfg.algorithm.get("clip_log_ratio_max", 20.0)),
                    max(
                        float(self.cfg.algorithm.get("clip_log_ratio_min", -20.0)),
                        value,
                    ),
                )
                for value in record["chunk_raw_log_ratios"]
            ]
            record["chunk_ratios"] = [
                float(np.exp(value)) for value in record["chunk_log_ratios"]
            ]
            record["chunk_clip_indicators"] = [
                bool(item["clip_fraction"] > 0) for item in ordered
            ]
            record["chunk_clip_fractions"] = [
                float(item["clip_fraction"]) for item in ordered
            ]
            record["new_logprob"] = float(sum(record["new_chunk_logprobs"]))
            raw_joint_log_ratio = record["new_logprob"] - record["old_logprob"]
            clamped_joint_log_ratio = min(
                float(self.cfg.algorithm.get("clip_log_ratio_max", 20.0)),
                max(
                    float(self.cfg.algorithm.get("clip_log_ratio_min", -20.0)),
                    raw_joint_log_ratio,
                ),
            )
            record["raw_log_ratio"] = float(raw_joint_log_ratio)
            record["log_ratio"] = float(clamped_joint_log_ratio)
            record["ratio"] = float(np.exp(clamped_joint_log_ratio))
            record["action_log_ratio_mean"] = float(
                np.mean([item["log_ratio_mean"] for item in ordered])
            )
            record["action_ratio_mean"] = float(
                np.mean([item["ratio_mean"] for item in ordered])
            )
            record["clip_indicator"] = bool(
                any(item["clip_fraction"] > 0 for item in ordered)
            )
            record["clip_fraction"] = float(
                np.mean([item["clip_fraction"] for item in ordered])
            )
            finite_scalars = (
                record["trajectory_mse"],
                record["lastframe_mse"],
                record["combined_reward"],
                record["old_logprob"],
                record["new_logprob"],
                record["ratio"],
            )
            if not all(np.isfinite(value) for value in finite_scalars):
                raise ValueError(
                    f"non-finite trajectory record for uid={record['rollout_uid']}"
                )
        shard_path = shard_dir / (
            f"rank_{self._rank:05d}_update_{update_id:04d}.jsonl"
        )
        shard_tmp = shard_path.with_suffix(".jsonl.tmp")
        with shard_tmp.open("w", encoding="utf-8") as handle:
            for record in sorted(records.values(), key=lambda item: item["rollout_uid"]):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(shard_tmp, shard_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._rank == 0:
            merged = []
            for rank in range(self._world_size):
                path = shard_dir / f"rank_{rank:05d}_update_{update_id:04d}.jsonl"
                with path.open(encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            expected = int(record_cfg.get("expected_trajectories", 512))
            if len(merged) != expected:
                raise ValueError(
                    f"trajectory record count {len(merged)} != {expected}"
                )
            if len({item["rollout_uid"] for item in merged}) != expected:
                raise ValueError("trajectory records contain duplicate rollout_uid")
            grouped: dict[int, list[dict]] = {}
            for record in merged:
                grouped.setdefault(int(record["group_id"]), []).append(record)
            expected_group_size = int(self.cfg.algorithm.group_size)
            expected_group_count = expected // expected_group_size
            if len(grouped) != expected_group_count:
                raise ValueError(
                    f"trajectory records contain {len(grouped)} groups; "
                    f"expected {expected_group_count}"
                )
            if any(len(group) != expected_group_size for group in grouped.values()):
                raise ValueError("trajectory records contain incomplete GRPO groups")
            for group in grouped.values():
                member_ids = sorted(int(item["member_id"]) for item in group)
                if member_ids != list(range(expected_group_size)):
                    raise ValueError(
                        "trajectory records contain incomplete or duplicate "
                        f"group members: {member_ids}"
                    )
                nodes = {str(item["node"]) for item in group}
                if len(nodes) != 1:
                    raise ValueError(
                        "one-group-per-node contract violated: "
                        f"group spans nodes {sorted(nodes)}"
                    )
                gpu_counts = {}
                for item in group:
                    gpu = int(item["gpu"])
                    gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                if gpu_counts != {0: 4, 1: 4, 2: 4, 3: 4}:
                    raise ValueError(
                        "one-group-per-node contract requires four trajectories "
                        f"on each of four GPUs, got {gpu_counts}"
                    )
                rewards = torch.tensor(
                    [item["combined_reward"] for item in group], dtype=torch.float64
                )
                mean = float(rewards.mean().item())
                std = float(rewards.std(unbiased=True).item())
                for record in group:
                    record["group_mean"] = mean
                    record["group_std"] = std
                    record["normalized_advantage"] = (
                        record["combined_reward"] - mean
                    ) / (std + 1e-6)
            required_record_fields = (
                "group_mean",
                "group_std",
                "normalized_advantage",
                "chunk_rewards",
                "chunk_advantages",
                "old_logprob",
                "new_logprob",
                "raw_log_ratio",
                "log_ratio",
                "ratio",
                "clip_indicator",
                "cosmos_joint_seeds",
                "ctrl_seed",
                "shuffle_seed",
                "retry_count",
                "checkpoint_checksum_manifest",
                "input_checksums",
                "valid",
                "validity_status",
                "exception",
            )
            per_chunk_fields = (
                "cosmos_joint_seeds",
                "cosmos_joint_seeds_uint32",
                "cosmos_seed_nonces",
                "retry_counts_per_chunk",
                "chunk_rewards",
                "chunk_advantages",
                "old_chunk_logprobs",
                "new_chunk_logprobs",
                "chunk_raw_log_ratios",
                "chunk_log_ratios",
                "chunk_ratios",
                "chunk_clip_indicators",
                "chunk_clip_fractions",
                "optimizer_observations",
            )
            finite_scalar_fields = (
                "trajectory_mse",
                "trajectory_similarity",
                "trajectory_reward",
                "lastframe_mse",
                "lastframe_similarity",
                "lastframe_reward",
                "combined_reward",
                "group_mean",
                "group_std",
                "normalized_advantage",
                "old_logprob",
                "new_logprob",
                "raw_log_ratio",
                "log_ratio",
                "ratio",
                "clip_fraction",
            )
            expected_ctrl_seed = int(
                self.cfg.algorithm.cross_rank_group.fixed_ctrl_seed
            )
            for record in merged:
                missing = [
                    name for name in required_record_fields if name not in record
                ]
                if missing:
                    raise ValueError(
                        "formal trajectory record is missing fields for "
                        f"uid={record['rollout_uid']}: {missing}"
                    )
                invalid_lengths = {
                    name: len(record[name])
                    for name in per_chunk_fields
                    if len(record[name]) != expected_chunks
                }
                if invalid_lengths:
                    raise ValueError(
                        "formal trajectory record has invalid chunk lengths for "
                        f"uid={record['rollout_uid']}: {invalid_lengths}"
                    )
                if len(record["success_last4_probabilities"]) != 4:
                    raise ValueError(
                        "formal trajectory record must contain four success "
                        f"probabilities for uid={record['rollout_uid']}"
                    )
                observation_chunk_ids = sorted(
                    int(item["chunk_id"])
                    for item in record["optimizer_observations"]
                )
                if observation_chunk_ids != list(range(expected_chunks)):
                    raise ValueError(
                        "formal trajectory record has invalid optimizer chunk IDs: "
                        f"{observation_chunk_ids}"
                    )
                if (
                    record["valid"] is not True
                    or record["validity_status"] != "valid"
                    or record["exception"] is not None
                ):
                    raise ValueError(
                        "formal trajectory record is not valid: "
                        f"uid={record['rollout_uid']}"
                    )
                if not record["vision_action_seed_equal"]:
                    raise ValueError("formal rollout used unequal vision/action seeds")
                if int(record["ctrl_seed"]) != expected_ctrl_seed:
                    raise ValueError(
                        "formal trajectory record has the wrong Ctrl seed: "
                        f"{record['ctrl_seed']} != {expected_ctrl_seed}"
                    )
                if int(record["shuffle_seed"]) < 0:
                    raise ValueError("formal trajectory record has no shuffle seed")
                if int(record["retry_count"]) != sum(
                    int(value) for value in record["retry_counts_per_chunk"]
                ):
                    raise ValueError("formal trajectory retry count is inconsistent")
                artifacts = record["input_checksums"].get("artifacts", {})
                if (
                    record.get("provenance_mode") == "strict"
                    and (not record["checkpoint_checksum_manifest"] or not artifacts)
                ):
                    raise ValueError(
                        "formal trajectory record has no checkpoint checksums"
                    )
                finite_values = [record[name] for name in finite_scalar_fields]
                for name in (
                    "success_last4_probabilities",
                    "chunk_rewards",
                    "chunk_advantages",
                    "old_chunk_logprobs",
                    "new_chunk_logprobs",
                    "chunk_raw_log_ratios",
                    "chunk_log_ratios",
                    "chunk_ratios",
                    "chunk_clip_fractions",
                ):
                    finite_values.extend(record[name])
                if not all(np.isfinite(value) for value in finite_values):
                    raise ValueError(
                        "formal trajectory record contains NaN or Inf: "
                        f"uid={record['rollout_uid']}"
                    )
                expected_advantage = (
                    record["combined_reward"] - record["group_mean"]
                ) / (record["group_std"] + 1e-6)
                if not np.isclose(
                    record["normalized_advantage"],
                    expected_advantage,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "formal trajectory normalized advantage is inconsistent"
                    )
            merged.sort(
                key=lambda item: (
                    item["update"],
                    item["logical_round"],
                    item["group_slot"],
                    item["member_id"],
                )
            )
            final_path = output_dir / f"update_{update_id:04d}.jsonl"
            final_tmp = final_path.with_suffix(".jsonl.tmp")
            with final_tmp.open("w", encoding="utf-8") as handle:
                for record in merged:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            os.replace(final_tmp, final_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _get_global_valid_loss_mask_count(self) -> int | None:
        """Count valid training entries after reward filtering across all actor ranks."""
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if loss_mask is None:
            return None

        local_valid_count = int(loss_mask.count_nonzero().item())
        if torch.distributed.is_initialized():
            return all_reduce_int(
                local_valid_count, op=torch.distributed.ReduceOp.SUM
            )
        return local_valid_count

    def _build_skipped_update_metrics(
        self, global_valid_loss_mask_count: int
    ) -> dict[str, float]:
        """Emit stable metrics when a rollout produces no trainable GRPO samples."""
        metrics = {
            "actor/skipped_update": 1.0,
            "actor/skipped_update_zero_valid_samples": 1.0,
            "actor/valid_loss_mask_count": float(global_valid_loss_mask_count),
            "actor/total_loss": 0.0,
            "actor/grad_norm": 0.0,
        }
        if self.optimizer.param_groups:
            metrics["actor/lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def _recompute_cosmos_actor_replay_prev_logprobs(self) -> None:
        """Rebase PPO old logprobs onto the memory-saving Actor replay.

        A resized Actor replay has different vision-chain conditioning from the
        original Rollout trajectory. Comparing its current logprobs against
        Rollout-time logprobs would therefore produce a meaningless PPO ratio.
        Re-evaluate every saved action once under the still-unmodified actor and
        freeze that result as the old-policy baseline before any optimizer step.
        """

        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.COSMOS:
            return
        cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
        actor_replay_size = cosmos_cfg.get("actor_replay_size", None)
        replay_objective = resolve_cosmos_replay_objective(
            cosmos_cfg.get("replay_objective", None)
        )
        per_mc_ratio = bool(
            replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            and self.cfg.algorithm.get("fpo_ratio_granularity", "per_action")
            == "per_mc"
        )
        replay_rebase_required = bool(
            replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            or actor_replay_size is not None
            or cosmos_cfg.get("replay_eval_mode", False)
        )
        recompute_enabled = cosmos_cfg.get(
            "actor_replay_recompute_prev_logprobs",
            replay_rebase_required,
        )
        if not replay_rebase_required or not recompute_enabled:
            return

        previous = self.rollout_batch["prev_logprobs"]
        rollout_size = int(previous.shape[0])
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        if rollout_size % micro_batch_size != 0:
            raise ValueError(
                "Cosmos Actor replay old-logprob recompute requires rollout size "
                f"{rollout_size} to be divisible by micro batch size "
                f"{micro_batch_size}."
            )

        replay_batches = split_dict_to_chunk(
            self.rollout_batch,
            rollout_size // micro_batch_size,
        )
        replay_logprobs = []
        replay_pair_scores = []
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"{Worker.torch_device_type}:{local_rank}"
        started_at = time.monotonic()
        replay_reason = (
            "fpo_action_head"
            if replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            else "resized_or_eval_replay"
        )
        self.log_on_first_rank(
            "Recomputing Cosmos PPO old scores before optimizer updates: "
            f"reason={replay_reason}, samples={rollout_size}, "
            f"micro_batch_size={micro_batch_size}, "
            f"actor_replay_size={actor_replay_size}."
        )

        was_training = self.model.training
        self.model.eval()
        try:
            for batch_idx, batch in enumerate(replay_batches, start=1):
                batch = put_tensor_device(batch, device)
                forward_inputs = batch.get("forward_inputs", None)
                with self.amp_context:
                    output_dict = self.model(
                        forward_inputs=forward_inputs,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        # Use the same grad-enabled eval forward as PPO replay.
                        # Some fused kernels select a different numerical path
                        # under no_grad; detach immediately after this call.
                        track_grad=True,
                        return_replay_diagnostics=per_mc_ratio,
                    )
                replay_logprobs.append(output_dict["logprobs"].detach().cpu())
                if per_mc_ratio:
                    replay_pair_scores.append(
                        -output_dict["fpo_pair_losses"].detach().float().cpu()
                    )
                del output_dict, batch
                if batch_idx % 10 == 0 or batch_idx == len(replay_batches):
                    self.log_on_first_rank(
                        "Cosmos Actor replay old-logprob recompute progress: "
                        f"{batch_idx}/{len(replay_batches)} microbatches."
                    )
        finally:
            self.model.train(was_training)

        recomputed = torch.cat(replay_logprobs, dim=0).to(
            device=previous.device,
            dtype=previous.dtype,
        )
        if recomputed.shape != previous.shape:
            raise RuntimeError(
                "Cosmos Actor replay old-logprob shape mismatch: "
                f"recomputed={tuple(recomputed.shape)}, "
                f"rollout={tuple(previous.shape)}."
            )
        delta = (recomputed.detach().cpu() - previous.detach().cpu()).abs()
        self.rollout_batch["prev_logprobs"] = recomputed
        if per_mc_ratio:
            recomputed_pair_scores = torch.cat(replay_pair_scores, dim=0).to(
                device=previous.device,
            )
            if recomputed_pair_scores.shape[0] != rollout_size:
                raise RuntimeError(
                    "Cosmos per-MC old-score batch size mismatch: "
                    f"{tuple(recomputed_pair_scores.shape)}."
                )
            self.rollout_batch["prev_fpo_pair_scores"] = recomputed_pair_scores
        self.log_on_first_rank(
            "Finished Cosmos Actor replay old-logprob recompute before any "
            f"optimizer update in {time.monotonic() - started_at:.1f}s; "
            f"rollout-baseline abs_delta_mean={delta.float().mean().item():.6f}, "
            f"abs_delta_max={delta.float().max().item():.6f}."
        )

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # NOTE: This must be set before importing openpi.training.data_loader
            if self.cfg.actor.get("sft_data_path", None):
                os.environ["HF_LEROBOT_HOME"] = self.cfg.actor.sft_data_path

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ):
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: x.to(self.device) if x is not None else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        sft_losses = self.model(
            data={"observation": observation, "actions": actions},
            forward_type=ForwardType.SFT,
        )
        # Ensure losses is a tensor and handle different return types
        if isinstance(sft_losses, list | tuple):
            sft_losses = torch.stack(sft_losses)
        elif not isinstance(sft_losses, torch.Tensor):
            sft_losses = torch.tensor(
                sft_losses, device=self.device, dtype=torch.float32
            )

        sft_loss = sft_losses.mean()
        metrics_data["sft_loss"] = sft_loss.clone().detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if (
            self.is_optimizer_offloaded
            and not self._defer_optimizer_onload_for_backward()
        ):
            self.load_optimizer(self.device)
        prepare_native_cpu_storage = getattr(
            self.model, "prepare_native_cpu_offload_training_storage", None
        )
        if callable(prepare_native_cpu_storage):
            preparation = prepare_native_cpu_storage(self.optimizer)
            moved_count = int(preparation.get("moved_parameter_count", 0))
            if moved_count:
                self.log_on_first_rank(
                    "Prepared Native Cosmos FSDP2 CPU-offload storage for "
                    f"{moved_count} action parameters before replay."
                )

        is_cosmos_model = (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.COSMOS
        )
        cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
        replay_objective = (
            resolve_cosmos_replay_objective(
                cosmos_cfg.get("replay_objective", None)
            )
            if is_cosmos_model
            else None
        )
        per_mc_ratio = bool(
            replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            and self.cfg.algorithm.get("fpo_ratio_granularity", "per_action")
            == "per_mc"
        )
        if is_cosmos_model and (
            cosmos_cfg.get("replay_eval_mode", False)
            or replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
        ):
            self.model.eval()
        else:
            self.model.train()
        native_binding_validator = getattr(
            self.model, "validate_native_trainable_bindings", None
        )
        if is_cosmos_model and callable(native_binding_validator):
            native_binding_validator(self.optimizer)
        global_valid_loss_mask_count = self._get_global_valid_loss_mask_count()
        if global_valid_loss_mask_count == 0:
            self.log_on_first_rank(
                "Rollout batch has zero valid samples after reward filtering; "
                "skipping actor update for this GRPO rollout."
            )
            skipped_metrics = self._build_skipped_update_metrics(
                global_valid_loss_mask_count
            )
            self._release_rollout_batch()
            self._offload_actor_state()
            return skipped_metrics

        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        trajectory_aware = bool(
            self.cfg.algorithm.get("trajectory_aware_minibatches", False)
        )
        if self._cross_rank_group_enabled():
            shuffle_seed = derive_stable_seed(
                "actor_shuffle",
                int(self.cfg.actor.seed),
                int(getattr(self, "global_step", self.version)),
            )
        else:
            shuffle_seed = self.cfg.actor.seed + self._rank
        if trajectory_aware:
            num_chunks = int(self.rollout_batch["prev_logprobs"].shape[0])
            num_trajectories = int(self.rollout_batch["prev_logprobs"].shape[1])
            expected_chunks = int(
                self.cfg.algorithm.get("trajectory_chunks", num_chunks)
            )
            if num_chunks != expected_chunks:
                raise ValueError(
                    f"Expected {expected_chunks} chunks per trajectory, got "
                    f"{num_chunks}."
                )
            shuffle_id = build_trajectory_aware_shuffle_id(
                num_chunks=num_chunks,
                num_trajectories=num_trajectories,
                seed=shuffle_seed,
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(shuffle_seed)
            shuffle_id = torch.randperm(rollout_size, generator=generator)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        self._recompute_cosmos_actor_replay_prev_logprobs()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        if trajectory_aware and batch_size_per_rank % expected_chunks != 0:
            raise ValueError(
                "Per-rank minibatch must contain complete trajectories: "
                f"{batch_size_per_rank} actor samples is not divisible by "
                f"{expected_chunks} chunks."
            )
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        planned_optimizer_steps = (
            int(update_epoch) * rollout_size // batch_size_per_rank
        )
        expected_optimizer_steps = int(
            self.cfg.algorithm.get(
                "expected_optimizer_steps_per_update", planned_optimizer_steps
            )
        )
        if planned_optimizer_steps != expected_optimizer_steps:
            raise ValueError(
                "Optimizer-step contract mismatch: planned "
                f"{planned_optimizer_steps}, expected {expected_optimizer_steps}."
            )
        optimizer_steps_this_update = 0
        for update_epoch_index in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    is_cosmos_model = (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.COSMOS
                    )
                    if is_cosmos_model:
                        # Cosmos inference initialization disables autograd via
                        # thread-local torch.set_grad_enabled(False). Restore
                        # actor-training semantics for replay and PPO loss.
                        torch.set_grad_enabled(True)
                    direction_probe_only = (
                        is_cosmos_model
                        and self.cfg.algorithm.get("diagnostics", {}).get(
                            "single_update_direction_check", False
                        )
                        and update_epoch_index == 1
                    )
                    if is_cosmos_model:
                        # Do not depend on an inference backend default for the
                        # actor autograd contract. Diagnostic-only replay is
                        # explicitly no-grad; every optimizer replay is explicit.
                        kwargs["track_grad"] = not direction_probe_only
                        kwargs["return_replay_diagnostics"] = per_mc_ratio
                    grad_context = torch.no_grad() if direction_probe_only else nullcontext()
                    with grad_context:
                        with self.amp_context:
                            output_dict = self.model(
                                forward_inputs=forward_inputs,
                                compute_logprobs=True,
                                compute_entropy=(
                                    self.cfg.algorithm.entropy_bonus > 0
                                    and not direction_probe_only
                                ),
                                compute_values=compute_values,
                                use_cache=False,
                                **kwargs,
                            )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    loss_logprobs = output_dict["logprobs"]
                    loss_old_logprobs = prev_logprobs
                    loss_logprob_type = self.cfg.algorithm.logprob_type
                    loss_single_action_dim = self.cfg.actor.model.get(
                        "action_dim", 7
                    )
                    if per_mc_ratio:
                        if "prev_fpo_pair_scores" not in batch:
                            raise RuntimeError("Missing fixed old per-MC FPO scores.")
                        loss_logprobs = -output_dict["fpo_pair_losses"].float()
                        loss_old_logprobs = batch["prev_fpo_pair_scores"].float()
                        loss_logprob_type = "action_level"
                        loss_single_action_dim = 1
                    cosmos_metrics = {}
                    if is_cosmos_model:
                        if per_mc_ratio:
                            diagnostic_old_logprobs = loss_old_logprobs
                            diagnostic_new_logprobs = loss_logprobs
                        else:
                            diagnostic_old_logprobs = prev_logprobs.reshape(
                                prev_logprobs.shape[0], -1
                            ).sum(dim=-1)
                            diagnostic_new_logprobs = output_dict[
                                "logprobs"
                            ].reshape(
                                output_dict["logprobs"].shape[0], -1
                            ).sum(dim=-1)
                        validate_action_logprobs_finite(
                            old_logprobs=diagnostic_old_logprobs,
                            new_logprobs=diagnostic_new_logprobs,
                        )
                        diagnostic_loss_mask = loss_mask
                        if diagnostic_loss_mask is not None:
                            diagnostic_loss_mask = (
                                diagnostic_loss_mask.reshape(diagnostic_loss_mask.shape[0], -1)
                                .amax(dim=-1)
                            )
                        cosmos_metrics.update(
                            compute_action_logprob_diagnostics(
                                old_logprobs=diagnostic_old_logprobs,
                                new_logprobs=diagnostic_new_logprobs,
                                loss_mask=diagnostic_loss_mask,
                                clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
                                clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
                            )
                        )
                        self._cosmos_action_ratio_guard.check(
                            cosmos_metrics["action/ratio_max"]
                        )
                        cosmos_metrics.update(compute_chain_diagnostics(forward_inputs))
                        diagnostics_cfg = self.cfg.algorithm.get("diagnostics", {})
                        if diagnostics_cfg.get("fpo_alignment", False) or (
                            diagnostics_cfg.get("single_update_direction_check", False)
                            and update_epoch_index == 1
                        ):
                            cosmos_metrics.update(
                                compute_advantage_logprob_alignment(
                                    old_logprobs=diagnostic_old_logprobs,
                                    new_logprobs=diagnostic_new_logprobs,
                                    advantages=advantages,
                                    loss_mask=diagnostic_loss_mask,
                                )
                            )
                        self._cosmos_artifact_writer.write_actor_logprob_summary(
                            rank=self._rank,
                            optimizer_step=self.optimizer_steps,
                            microbatch_idx=idx,
                            old_logprobs=diagnostic_old_logprobs,
                            new_logprobs=diagnostic_new_logprobs,
                        )

                    if direction_probe_only:
                        metrics_data = {
                            key: value.detach().cpu().item()
                            if torch.is_tensor(value)
                            else value
                            for key, value in cosmos_metrics.items()
                        }
                        append_to_dict(metrics, metrics_data)
                        continue

                    self._update_trajectory_record_ratios(
                        batch,
                        old_logprobs=prev_logprobs,
                        new_logprobs=output_dict["logprobs"],
                    )

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": loss_logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": loss_single_action_dim,
                        "logprobs": loss_logprobs,
                        "values": output_dict.get("values", None),
                        "old_logprobs": loss_old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "loss_agg_func": self.loss_agg_func,
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", None),
                        "clip_log_ratio_min": self.cfg.algorithm.get(
                            "clip_log_ratio_min", None
                        ),
                        "clip_log_ratio_max": self.cfg.algorithm.get(
                            "clip_log_ratio_max", None
                        ),
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                        "lambda_action": self.cfg.algorithm.get("lambda_action", 1.0),
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(
                        0.0, device=Worker.torch_platform.current_device()
                    )
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
                    if is_cosmos_model and self.cfg.algorithm.entropy_bonus > 0:
                        cosmos_metrics["action/entropy"] = entropy_loss.detach()
                    if is_cosmos_model:
                        metrics_data.update(
                            {
                                key: value.detach().cpu().item()
                                if torch.is_tensor(value)
                                else value
                                for key, value in cosmos_metrics.items()
                            }
                        )

                    if self.enable_sft_co_train:
                        self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    if is_cosmos_model and not loss.requires_grad:
                        raise RuntimeError(
                            "Cosmos actor loss is detached before backward: "
                            f"grad_enabled={torch.is_grad_enabled()}, "
                            f"logprobs_requires_grad="
                            f"{output_dict['logprobs'].requires_grad}."
                        )
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                if self.cfg.actor.model.model_type == "cosmos":
                    grad_metrics = compute_gradient_diagnostics(self.model)
                    append_to_dict(
                        metrics,
                        {
                            key: value.detach().cpu().item()
                            if torch.is_tensor(value)
                            else value
                            for key, value in grad_metrics.items()
                        },
                    )
                    diagnostics_cfg = self.cfg.algorithm.get("diagnostics", {})
                    if diagnostics_cfg.get("warn_frozen_video_grad", True):
                        for warning in warn_if_frozen_video_grads(self.model):
                            self.log_on_first_rank(
                                f"Cosmos frozen video grad warning: {warning}"
                            )

                self.torch_platform.empty_cache()

                optimizer_onloaded_for_step = self._onload_optimizer_for_step()
                try:
                    grad_norm, lr_list = self.optimizer_step()
                finally:
                    self._offload_optimizer_after_step(
                        optimizer_onloaded_for_step
                    )
                optimizer_steps_this_update += 1
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        if optimizer_steps_this_update != expected_optimizer_steps:
            raise RuntimeError(
                "Executed optimizer-step count mismatch: "
                f"{optimizer_steps_this_update} != {expected_optimizer_steps}."
            )
        append_to_dict(
            metrics,
            {
                "actor/optimizer_steps_per_update": float(
                    optimizer_steps_this_update
                ),
                "actor/trajectory_aware_minibatches": float(trajectory_aware),
                "actor/shuffle_seed": float(shuffle_seed),
                "hardware/optimizer_state_cpu_during_backward": float(
                    self._defer_optimizer_onload_for_backward()
                ),
            },
        )
        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            total_memory = float(
                torch.cuda.get_device_properties(cuda_device).total_memory
            )
            local_memory = torch.tensor(
                [
                    float(torch.cuda.max_memory_allocated(cuda_device)),
                    float(torch.cuda.max_memory_reserved(cuda_device)),
                    total_memory,
                ],
                dtype=torch.float64,
                device=torch.device("cuda", cuda_device),
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    local_memory, op=torch.distributed.ReduceOp.MAX
                )
            peak_allocated, peak_reserved, largest_device = (
                float(value) for value in local_memory.detach().cpu().tolist()
            )
            append_to_dict(
                metrics,
                {
                    "hardware/gpu_peak_allocated_fraction_max": (
                        peak_allocated / largest_device
                    ),
                    "hardware/gpu_peak_reserved_fraction_max": (
                        peak_reserved / largest_device
                    ),
                    "hardware/gpu_total_memory_gib": (
                        largest_device / float(1024**3)
                    ),
                },
            )
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self._finalize_trajectory_records()

        # No subsequent phase needs the replay tensors. In particular, model
        # checkpointing only consumes model/optimizer state. Releasing here
        # prevents the previous full Native denoising chains and videos from
        # remaining resident throughout the next rollout.
        self._release_rollout_batch()
        self._offload_actor_state()

        return mean_metric_dict

    def discard_rollout_batch(self) -> dict[str, float]:
        """Release a validated rollout without loading or updating the actor."""
        self._release_rollout_batch()
        self._offload_actor_state()
        return {
            "actor/rollout_only": 1.0,
            "actor/skipped_update": 1.0,
        }

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        self.global_step = int(global_step)
        # This is called before rollout collection, so the peak covers the
        # complete rollout + replay + backward/optimizer cycle for the update.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_replay_buffer_warmup_status(self) -> dict[str, int | bool]:
        """Report replay-buffer readiness for rollout warmup scheduling."""
        return {
            "has_replay_buffer": False,
            "is_ready": True,
            "buffer_size": 0,
            "min_buffer_size": 0,
        }
