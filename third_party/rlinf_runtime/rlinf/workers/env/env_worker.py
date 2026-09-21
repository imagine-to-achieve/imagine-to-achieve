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

import asyncio
import hashlib
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Literal

import imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.cross_rank_grpo import build_cross_rank_rollout_metadata
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.world_model.duck_episode_contract import (
    load_duck_episode_split,
    select_duck_episode_colors,
)
from rlinf.envs.wrappers import RecordVideo
from rlinf.models.embodiment.cosmos.artifacts import CosmosArtifactWriter
from rlinf.models.embodiment.cosmos.camera_layout import (
    stitch_cosmos_three_view_chw,
)
from rlinf.models.embodiment.cosmos.diagnostics import find_video_range_warnings
from rlinf.models.embodiment.cosmos.fpo import (
    COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD,
    resolve_cosmos_replay_objective,
)
from rlinf.rewards import (
    compute_terminal_goal_reward,
    compute_video_clip_similarity_reward,
    compute_video_dino_similarity_reward,
    compute_video_similarity_reward,
    render_comparison_grid_frames,
)
from rlinf.rewards.resnet_reward_model import (
    classify_terminal_probabilities,
    resolve_eval_success_model_provenance,
)
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import update_nested_cfg
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory


def _encode_sha256_hex_batch(digests: list[str | None]) -> torch.Tensor:
    """Encode fixed-width SHA-256 hex strings as a tensor-safe byte matrix."""
    rows: list[list[int]] = []
    for index, digest in enumerate(digests):
        if not isinstance(digest, str):
            raise ValueError(
                f"SHA-256 digest at index {index} is missing or is not a string."
            )
        try:
            raw_digest = bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError(
                f"SHA-256 digest at index {index} is not valid hexadecimal."
            ) from exc
        if len(raw_digest) != hashlib.sha256().digest_size:
            raise ValueError(
                f"SHA-256 digest at index {index} decoded to "
                f"{len(raw_digest)} bytes; expected {hashlib.sha256().digest_size}."
            )
        rows.append(list(raw_digest))
    if not rows:
        return torch.empty((0, hashlib.sha256().digest_size), dtype=torch.uint8)
    return torch.tensor(rows, dtype=torch.uint8)


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.cosmos_comparison_video_cnt = 0
        self.cosmos_comparison_frames = []
        self.eval_cosmos_comparison_video_cnt = 0
        self.eval_cosmos_comparison_frames = []
        self.all_cosmos_comparison_frames: list[list[np.ndarray]] = []
        self.eval_all_cosmos_comparison_frames: list[list[np.ndarray]] = []
        self.global_step = 0
        self.rollout_phase = 0
        self.best_rollout_comparison_frames: list[list[np.ndarray]] = []
        self.best_rollout_episode_rewards: torch.Tensor | None = None
        self.best_rollout_chunk_rewards: list[list[float]] = []
        self.best_rollout_chunk_mses: list[list[float]] = []
        self.best_rollout_successes: torch.Tensor | None = None
        self.best_rollout_final_side_frames: torch.Tensor | None = None
        self.best_rollout_final_side_frame_sha256: list[str | None] = []
        self.best_rollout_final_side_frame_episodes: list[int | None] = []
        self._comparison_streams: dict[str, dict[int, dict[str, Any]]] = {}
        self.should_stop = False
        self._cosmos_previous_similarity_scores: torch.Tensor | None = None
        self._terminal_goal_reference: dict[str, Any] | None = None
        self._terminal_goal_references: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._terminal_goal_dataset: Any | None = None
        self._post_update_eval_context: dict[str, Any] | None = None
        self._post_update_eval_records: list[dict[str, Any]] = []
        self._cross_rank_reset_contract_cache: dict[str, Any] | None = None

        self.env_list = []
        self.eval_env_list = []
        self.libero_verifier_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self._cosmos_artifact_writer = CosmosArtifactWriter.from_model_cfg(
            self.cfg.actor.model
        )

        # Env configurations
        self.enable_offload = self.cfg.env.train.get("enable_offload", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            total_eval_envs = int(self.cfg.env.eval.total_num_envs)
            if total_eval_envs % self.stage_num != 0:
                raise ValueError(
                    "env.eval.total_num_envs must be divisible by pipeline_stage_num"
                )
            eval_envs_per_stage = total_eval_envs // self.stage_num
            eval_collective_group_size = int(
                self.cfg.rollout.get("sparse_eval_collective_group_size", 1)
            )
            self.eval_mapping_world_size = CommMapper.get_collective_aligned_world_size(
                eval_envs_per_stage, self._world_size, eval_collective_group_size
            )
            if self._rank < self.eval_mapping_world_size:
                self.eval_num_envs_per_stage = CommMapper.get_rank_batch_size(
                    eval_envs_per_stage, self.eval_mapping_world_size, self._rank
                )
            else:
                self.eval_num_envs_per_stage = 0
            self.eval_env_active = self.eval_num_envs_per_stage > 0
        else:
            self.eval_mapping_world_size = 0
            self.eval_env_active = False
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

        if self._track_rollout_candidate_videos() and self.stage_num != 1:
            raise ValueError(
                "Best/worst-rollout video tracking currently requires "
                f"rollout.pipeline_stage_num=1, got {self.stage_num}."
            )
        self._validate_comparison_stream_config()

    def set_global_step(self, global_step: int) -> None:
        """Set the runner step used for restart-safe best-video names."""
        self.global_step = int(global_step)

    def set_rollout_context(self, global_step: int, rollout_phase: int) -> None:
        """Set restart-safe identifiers for the next training rollout."""
        self.global_step = int(global_step)
        self.rollout_phase = int(rollout_phase)

    def set_post_update_eval_context(
        self,
        global_step: int,
        episode_ids: list[int] | tuple[int, ...],
        base_seed: int = 42,
        execution_episode_ids: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Pin this rank's sparse post-update evaluation episode and RNG seed."""
        if self.stage_num != 1:
            raise ValueError(
                "post-update duck evaluation requires pipeline_stage_num=1"
            )
        official_ids = tuple(sorted(int(value) for value in episode_ids))
        if not official_ids or len(set(official_ids)) != len(official_ids):
            raise ValueError(
                "post-update eval official episode IDs must be non-empty and unique"
            )
        if execution_episode_ids is None:
            execution_ids = official_ids
        else:
            execution_ids = tuple(int(value) for value in execution_episode_ids)
        shard_degree = int(self.cfg.rollout.sparse_eval_collective_group_size)
        padding = (-len(official_ids)) % shard_degree
        expected_execution = official_ids + tuple(
            official_ids[index % len(official_ids)] for index in range(padding)
        )
        # Red Duck's 10 official episodes execute as 12 so every
        # active Cosmos HSDP shard has all four ranks participating.
        if execution_ids != expected_execution:
            raise ValueError("post-update eval execution IDs violate shard padding")
        expected_total = int(self.cfg.env.eval.total_num_envs)
        if len(execution_ids) != expected_total:
            raise ValueError(
                "post-update eval execution count must match env.eval.total_num_envs: "
                f"{len(execution_ids)} != {expected_total}"
            )
        if self._rank < self.eval_mapping_world_size:
            local_start, local_stop = CommMapper.get_rank_batch_range(
                len(execution_ids), self.eval_mapping_world_size, self._rank
            )
        else:
            local_start = local_stop = len(execution_ids)
        local_count = local_stop - local_start
        local_episode_ids = execution_ids[local_start:local_stop]
        local_padding = tuple(
            index >= len(official_ids) for index in range(local_start, local_stop)
        )
        if any(local_padding) and not all(local_padding):
            raise RuntimeError(
                "Official and padding eval episodes may not share one env worker batch"
            )
        local_seeds = tuple(
            int(base_seed) + len(official_ids) * (int(global_step) - 1)
            + (index % len(official_ids))
            for index in range(local_start, local_start + local_count)
        )
        if local_count != self.eval_num_envs_per_stage:
            raise ValueError(
                "post-update eval sparse shard does not match initialized env count: "
                f"{local_count} != {self.eval_num_envs_per_stage}"
            )

        self.global_step = int(global_step)
        self._post_update_eval_records = []
        self._post_update_eval_context = {
            "global_step": int(global_step),
            "base_seed": int(base_seed),
            "active": bool(local_count),
            "official_episode_ids": official_ids,
            "execution_episode_ids": execution_ids,
            "local_padding": local_padding,
            "local_episode_ids": local_episode_ids,
            "local_seeds": local_seeds,
            "local_start": int(local_start),
        }
        if not local_count:
            return dict(self._post_update_eval_context)
        if len(self.eval_env_list) != self.stage_num:
            raise RuntimeError(
                "active post-update eval rank has no initialized eval env"
            )
        for env in self.eval_env_list:
            setter = self._get_wrapped_env_attr(env, "set_post_update_eval_context")
            if not callable(setter):
                raise RuntimeError(
                    "eval environment does not expose set_post_update_eval_context()"
                )
            setter(
                int(global_step),
                local_episode_ids,
                base_seed=int(base_seed),
                seeds=local_seeds,
            )
        provenance_getter = self._get_wrapped_env_attr(
            self.eval_env_list[0], "get_success_model_provenance"
        )
        provenance = provenance_getter() if callable(provenance_getter) else {}
        generic_eval_cfg = self.cfg.get("post_update_evaluation", {})
        active_variants = tuple(
            str(value)
            for value in generic_eval_cfg.get(
                "variants", self.cfg.get("duck", {}).get("colors", [])
            )
        )
        configured_models = self.cfg.reward.get("success_models", {})
        eval_ctrl_world_cfg = self.cfg.env.eval.get("ctrl_world_cfg", {})
        success_model_input_cfg = eval_ctrl_world_cfg.get("reward_model", {})
        success_model_camera_key = str(
            success_model_input_cfg.get("camera_key", "")
        )
        success_model_view_index = success_model_input_cfg.get(
            "ctrl_world_view_index", None
        )
        if success_model_view_index is not None:
            success_model_view_index = int(success_model_view_index)
        resolved_model_provenance = [
            resolve_eval_success_model_provenance(
                episode_id,
                active_variants=active_variants,
                runtime_provenance=provenance,
                configured_models=configured_models,
            )
            for episode_id in local_episode_ids
        ]
        for local_index, (episode_id, seed, is_padding) in enumerate(
            zip(local_episode_ids, local_seeds, local_padding, strict=True)
        ):
            color, model = resolved_model_provenance[local_index]
            segment_count = int(self.n_eval_chunk_steps)
            self._post_update_eval_records.append(
                {
                    "step": int(global_step),
                    "episode": int(episode_id),
                    # Stable source identity for evaluation artifacts. The
                    # per-rank comparison-video writer uses this same local
                    # batch order, so these fields bind each scalar record to
                    # its video without reconstructing placement afterwards.
                    "rank": int(self._rank),
                    "env_id": int(local_index),
                    "color": color,
                    "split": "validation",
                    "series": "eval_post_update",
                    "policy_stage": "post_update",
                    "seed": int(seed),
                    "success_model_path": model.get("path"),
                    "success_model_sha256": model.get("sha256"),
                    "model_hash": model.get("sha256"),
                    "reward_model_sha256": model.get("sha256"),
                    "success_model_camera_key": success_model_camera_key,
                    "success_model_ctrl_world_view_index": success_model_view_index,
                    "rng_manifest": {
                        "schema_version": 1,
                        "seed_formula": (
                            "base_seed + evaluation_episode_count * "
                            "zero_based_update_index + sorted_episode_index"
                        ),
                        "base_seed": int(base_seed),
                        "evaluation_seed": int(seed),
                        "zero_based_update_index": int(global_step) - 1,
                        "sorted_episode_index": int((local_start + local_index) % len(official_ids)),
                        "cosmos": {
                            "seed_derivation_version": "constant-eval-seed-v1",
                            "visual_base_seed": int(seed),
                            "action_base_seed": int(seed),
                            "segment_count": segment_count,
                            "visual_segment_seeds": [int(seed)] * segment_count,
                            "action_segment_seeds": [int(seed)] * segment_count,
                        },
                        "ctrl_world": {
                            "seed_derivation_version": "blake2b-int63-v2",
                            "namespace": (
                                "post_update_eval_ctrl_world_denoise"
                            ),
                            "base_denoise_seed": int(seed),
                            "denoise_calls": [],
                            "denoise_seeds": [],
                        },
                    },
                    "frame_probabilities": [],
                    "video_mse_per_chunk": [],
                    "_rng_completed_segments": 0,
                    "_padding_execution": bool(is_padding),
                    "combined_reward": 0.0,
                    "terminal_goal_mse": None,
                    "terminal_goal_reference_episode": None,
                    "terminal_goal_reference_frame": None,
                    "complete": False,
                    "valid": True,
                    "exception": None,
                    "error": None,
                }
            )
        return dict(self._post_update_eval_context)

    def _sync_post_update_eval_rng_manifest(self) -> None:
        """Copy actual lower-env denoise seeds into episode-level records."""
        context = self._post_update_eval_context
        if context is None or not context.get("active", False):
            return
        getter = self._get_wrapped_env_attr(
            self.eval_env_list[0], "get_post_update_eval_rng_manifest"
        )
        if not callable(getter):
            raise RuntimeError(
                "eval environment does not expose its Ctrl-World RNG manifest"
            )
        lower_manifest = getter()
        if lower_manifest.get("seed_derivation_version") != "blake2b-int63-v2":
            raise RuntimeError("Ctrl-World eval seed derivation version changed")
        base_seeds = [int(value) for value in lower_manifest.get("base_seeds", [])]
        records = self._post_update_eval_records
        if base_seeds != [int(record["seed"]) for record in records]:
            raise RuntimeError(
                "Ctrl-World eval base seeds disagree with episode records"
            )
        completed_segments = {
            int(record["_rng_completed_segments"]) for record in records
        }
        if len(completed_segments) != 1:
            raise RuntimeError("eval records disagree on completed segment count")
        segment_index = next(iter(completed_segments))
        for call in lower_manifest.get("denoise_calls", []):
            call_index = int(call["call_index"])
            derived_seeds = [int(value) for value in call["derived_seeds"]]
            if len(derived_seeds) != len(records):
                raise RuntimeError(
                    "Ctrl-World denoise seed count disagrees with eval batch"
                )
            for env_index, record in enumerate(records):
                ctrl_manifest = record["rng_manifest"]["ctrl_world"]
                existing_indices = {
                    int(item["call_index"])
                    for item in ctrl_manifest["denoise_calls"]
                }
                if call_index in existing_indices:
                    continue
                window_index = sum(
                    int(item["segment_index"]) == segment_index
                    for item in ctrl_manifest["denoise_calls"]
                )
                derived_seed = derived_seeds[env_index]
                ctrl_manifest["denoise_calls"].append(
                    {
                        "segment_index": segment_index,
                        "window_index": int(window_index),
                        "call_index": call_index,
                        "derived_seed": derived_seed,
                    }
                )
                ctrl_manifest["denoise_seeds"].append(derived_seed)

    def consume_post_update_eval_records(self) -> list[dict[str, Any]]:
        """Return completed local eval records and clear the worker buffer."""
        self._sync_post_update_eval_rng_manifest()
        records = self._post_update_eval_records
        if any(not record.get("complete", False) for record in records):
            incomplete = [
                record["episode"]
                for record in records
                if not record["complete"]
            ]
            raise RuntimeError(
                f"post-update evaluation is incomplete for episodes {incomplete}"
            )
        expected_segments = set(range(int(self.n_eval_chunk_steps)))
        official_records: list[dict[str, Any]] = []
        for record in records:
            rng_manifest = record.get("rng_manifest", {})
            cosmos = rng_manifest.get("cosmos", {})
            if len(cosmos.get("visual_segment_seeds", [])) != len(
                expected_segments
            ) or len(cosmos.get("action_segment_seeds", [])) != len(
                expected_segments
            ):
                raise RuntimeError(
                    f"post-update Cosmos RNG manifest is incomplete for "
                    f"episode {record['episode']}"
                )
            ctrl_calls = rng_manifest.get("ctrl_world", {}).get(
                "denoise_calls", []
            )
            actual_segments = {
                int(call["segment_index"]) for call in ctrl_calls
            }
            if actual_segments != expected_segments:
                raise RuntimeError(
                    f"post-update Ctrl-World RNG manifest is incomplete for "
                    f"episode {record['episode']}: segments={actual_segments}"
                )
            record.pop("_rng_completed_segments", None)
            is_padding = bool(record.pop("_padding_execution", False))
            # Padding only satisfies Cosmos collectives and must
            # never enter official Duck eval records or aggregate metrics.
            if not is_padding:
                official_records.append(record)
        self._post_update_eval_records = []
        return official_records

    def _cross_rank_group_cfg(self):
        cross_rank_group = self.cfg.algorithm.get("cross_rank_group", None)
        if cross_rank_group is None or not cross_rank_group.get("enabled", False):
            return None
        return cross_rank_group

    def _cross_rank_reset_contract(self, cross_rank_group: Any) -> dict[str, Any]:
        """Resolve inline or manifest-backed reset scheduling fields once."""
        if self._cross_rank_reset_contract_cache is not None:
            return self._cross_rank_reset_contract_cache
        episode_ids = cross_rank_group.get("reset_episode_ids", None)
        episodes_by_color = cross_rank_group.get(
            "reset_episode_ids_by_color", None
        )
        color_order = cross_rank_group.get(
            "reset_episode_color_order",
            cross_rank_group.get("color_order", None),
        )
        groups_per_color = int(
            cross_rank_group.get(
                "groups_per_color",
                cross_rank_group.get("episodes_per_color_per_update", 2),
            )
        )
        logical_rounds = int(
            cross_rank_group.get("logical_rounds_per_update", 4)
        )
        if logical_rounds <= 0 or groups_per_color % logical_rounds != 0:
            raise ValueError(
                "groups_per_color must be divisible by logical_rounds_per_update: "
                f"{groups_per_color} / {logical_rounds}"
            )
        # Config stores the per-update color total; the scheduler consumes
        # the number of draws assigned to one logical round.
        groups_per_color_per_round = groups_per_color // logical_rounds
        manifest_path = cross_rank_group.get(
            "reset_episode_allowlist_manifest", None
        )
        if manifest_path is not None:
            dotted_key = str(
                cross_rank_group.get(
                    "reset_episode_allowlist_key", "training.episodes"
                )
            )
            split_name = dotted_key.split(".", 1)[0]
            loaded = load_duck_episode_split(str(manifest_path), split_name)
            if dotted_key != f"{loaded.name}.episodes":
                raise ValueError(
                    "reset_episode_allowlist_key must select the validated split's "
                    f"episode list, got {dotted_key!r}"
                )
            # Keep the frozen four-color manifest validation, then
            # restrict reset scheduling to the configured active Duck color subset.
            selected_episode_ids, selected_pools, selected_order = (
                select_duck_episode_colors(loaded, color_order)
            )
            if episode_ids is not None and tuple(
                int(value) for value in episode_ids
            ) != selected_episode_ids:
                raise ValueError(
                    "inline reset_episode_ids does not match the selected frozen "
                    "manifest colors"
                )
            episode_ids = selected_episode_ids
            episodes_by_color = selected_pools
            color_order = selected_order
        self._cross_rank_reset_contract_cache = {
            "episode_ids": episode_ids,
            "episodes_by_color": episodes_by_color,
            "color_order": color_order,
            "groups_per_color": groups_per_color_per_round,
            "groups_per_color_per_update": groups_per_color,
        }
        return self._cross_rank_reset_contract_cache

    def _build_cross_rank_metadata(
        self, *, stage_id: int, chunk_index: int, rollout_epoch_index: int = 0
    ) -> dict[str, torch.Tensor]:
        cross_rank_group = self._cross_rank_group_cfg()
        if cross_rank_group is None:
            return {}
        if stage_id != 0:
            raise ValueError("cross-rank groups require pipeline_stage_num=1")
        if not 0 <= int(chunk_index) <= self.n_train_chunk_steps:
            raise ValueError(
                "cross-rank chunk_index must address a rollout chunk or its "
                f"terminal boundary, got {chunk_index} for "
                f"{self.n_train_chunk_steps} chunks"
            )
        if int(chunk_index) == self.n_train_chunk_steps:
            # The env sends one terminal observation so value-based policies
            # can bootstrap. It is not a sixth sampled trajectory chunk and
            # therefore must not consume a Cosmos semantic seed.
            return {}
        semantic_topology = bool(
            cross_rank_group.get("semantic_topology_v2", False)
        )
        groups_per_wave = (
            self._world_size
            * int(cross_rank_group.members_per_rank)
            // int(self.cfg.algorithm.group_size)
        )
        waves_per_logical_round = int(
            cross_rank_group.get("waves_per_logical_round", 1)
        )
        if waves_per_logical_round <= 0:
            raise ValueError("waves_per_logical_round must be positive")
        logical_round_id = rollout_epoch_index // waves_per_logical_round
        wave_in_round = rollout_epoch_index % waves_per_logical_round
        reset_contract = self._cross_rank_reset_contract(cross_rank_group)
        return build_cross_rank_rollout_metadata(
            base_seed=int(cross_rank_group.get("seed_base", self.cfg.actor.seed)),
            global_step=self.global_step,
            rollout_phase=self.rollout_phase,
            source_env_rank=self._rank,
            local_batch_size=self.train_num_envs_per_stage,
            members_per_rank=int(cross_rank_group.members_per_rank),
            group_size=int(self.cfg.algorithm.group_size),
            chunk_index=chunk_index,
            per_member_vision_seed=bool(
                cross_rank_group.get("per_member_vision_seed", False)
            ),
            logical_round_id=logical_round_id if semantic_topology else None,
            physical_wave_id=rollout_epoch_index,
            group_slot_offset=wave_in_round * groups_per_wave,
            logical_rounds_per_update=int(
                cross_rank_group.get("logical_rounds_per_update", 4)
            ),
            group_slots_per_round=int(
                cross_rank_group.get("group_slots_per_logical_round", 8)
            ),
            chunks_per_trajectory=int(
                cross_rank_group.get(
                    "chunks_per_trajectory", self.n_train_chunk_steps
                )
            ),
            reset_episode_min=int(
                cross_rank_group.get("reset_episode_min", 5)
            ),
            reset_episode_max=int(
                cross_rank_group.get("reset_episode_max", 39)
            ),
            reset_episode_ids=reset_contract["episode_ids"],
            reset_episode_ids_by_color=reset_contract["episodes_by_color"],
            reset_episode_color_order=reset_contract["color_order"],
            groups_per_color=reset_contract["groups_per_color"],
            fixed_ctrl_seed=(
                int(cross_rank_group.fixed_ctrl_seed)
                if cross_rank_group.get("fixed_ctrl_seed", None) is not None
                else None
            ),
        )

    @staticmethod
    def _attach_cross_rank_metadata(
        obs: dict[str, Any], metadata: dict[str, torch.Tensor]
    ) -> None:
        for field_name, value in metadata.items():
            obs[field_name] = value

    def _set_env_cross_rank_context(
        self,
        env,
        metadata: dict[str, torch.Tensor],
        *,
        reset_rollout: bool,
    ) -> None:
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            method_name = (
                "set_cross_rank_rollout_context"
                if reset_rollout
                else "set_cross_rank_chunk_context"
            )
            setter = getattr(current, method_name, None)
            if setter is not None:
                setter(metadata)
                return
            current = getattr(current, "env", None)
        raise ValueError(
            "cross-rank groups require an environment implementing "
            "set_cross_rank_rollout_context"
        )

    def _save_best_rollout_video(self) -> bool:
        video_cfg = self.cfg.env.train.get("video_cfg", {})
        return bool(video_cfg.get("save_best_rollout", False))

    def _save_worst_rollout_video(self) -> bool:
        video_cfg = self.cfg.env.train.get("video_cfg", {})
        return bool(video_cfg.get("save_worst_rollout", False))

    def _save_success_failure_mse_extremes(self) -> bool:
        video_cfg = self.cfg.env.train.get("video_cfg", {})
        return bool(video_cfg.get("save_success_failure_mse_extremes", False))

    def _trajectory_artifact_cfg(self):
        return self.cfg.algorithm.get("trajectory_records", {})

    def _save_rollout_final_side_frames(self) -> bool:
        return bool(
            self._trajectory_artifact_cfg().get("save_final_side_frames", False)
        )

    def _save_mse_extreme_final_frames(self) -> bool:
        return bool(
            self._trajectory_artifact_cfg().get(
                "save_mse_extreme_final_frames", False
            )
        )

    def _track_rollout_summaries(self) -> bool:
        return (
            self._track_rollout_candidate_videos()
            or self._save_rollout_final_side_frames()
            or self._save_mse_extreme_final_frames()
        )

    def _save_global_mse_extremes_only(self) -> bool:
        video_cfg = self.cfg.env.train.get("video_cfg", {})
        return bool(video_cfg.get("global_mse_extremes_only", False))

    def _track_rollout_candidate_videos(self) -> bool:
        return (
            self._save_best_rollout_video()
            or self._save_worst_rollout_video()
            or self._save_success_failure_mse_extremes()
        )

    def _stream_all_cosmos_comparison(self, mode: str = "train") -> bool:
        video_cfg = self.cfg.env[mode].get("video_cfg", {})
        """Reject invalid or mutually exclusive comparison recording modes."""
        return bool(video_cfg.get("stream_all_cosmos_comparison", False))

    def _validate_comparison_stream_config(self) -> None:
        if self._save_mse_extreme_final_frames():
            if not self._save_rollout_final_side_frames():
                raise ValueError(
                    "MSE-extreme final frames require all rollout final frames"
                )
            if not self._cosmos_video_similarity_enabled():
                raise ValueError(
                    "MSE-extreme final frames require video similarity"
                )
            if not self._success_classifier_diagnostic_enabled():
                raise ValueError(
                    "MSE-extreme final frames require success diagnostics"
                )
        if self._save_success_failure_mse_extremes():
            if not self._cosmos_video_similarity_enabled():
                raise ValueError(
                    "success/failure MSE-extreme videos require video similarity"
                )
            if not self._success_classifier_diagnostic_enabled():
                raise ValueError(
                    "success/failure MSE-extreme videos require success diagnostics"
                )
        if not self._stream_all_cosmos_comparison("train"):
            return
        video_cfg = self.cfg.env.train.video_cfg
        conflicts = [
            name
            for name in (
                "save_video",
                "save_all_trajectories",
                "save_best_rollout",
                "save_worst_rollout",
                "save_success_failure_mse_extremes",
            )
            if bool(video_cfg.get(name, False))
        ]
        if conflicts:
            raise ValueError(
                "stream_all_cosmos_comparison must replace buffered video "
                f"recording; disable {', '.join(conflicts)}"
            )
        if self.stage_num != 1:
            raise ValueError(
                "stream_all_cosmos_comparison currently requires "
                f"rollout.pipeline_stage_num=1, got {self.stage_num}."
            )
        if self._cross_rank_group_cfg() is None:
            raise ValueError(
                "stream_all_cosmos_comparison requires cross-rank trajectory "
                "metadata for stable video identities"
            )

    def _reset_best_rollout_video_buffer(self) -> None:
        self.best_rollout_comparison_frames = []
        self.best_rollout_episode_rewards = None
        self.best_rollout_chunk_rewards = []
        self.best_rollout_chunk_mses = []
        self.best_rollout_successes = None
        self.best_rollout_final_side_frames = None
        self.best_rollout_final_side_frame_sha256 = []
        self.best_rollout_final_side_frame_episodes = []

    def init_worker(self):
        if (
            self._terminal_goal_reward_enabled()
            and str(
                self.cfg.reward.terminal_goal.get("reference_mode", "fixed")
            ).lower()
            == "fixed"
        ):
            self._get_terminal_goal_reference()
        elif self._terminal_goal_reward_enabled():
            self._save_terminal_goal_contract_metadata()
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }

        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num,
                active_world_size=self.eval_mapping_world_size,
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num,
                active_world_size=self.eval_mapping_world_size,
            )
        self.log_info(f"Env worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Env worker initialized with src_ranks: {self.src_ranks}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        if not self.only_eval:
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            # Active duck-eval ranks construct a second Ctrl-World instance.
            # Move the train instance to CPU before that construction so two
            # full diffusion stacks are never resident on one GPU at once.
            if (
                self.enable_eval
                and self.eval_env_active
                and self.enable_offload
            ):
                for env in self.env_list:
                    if hasattr(env, "offload"):
                        env.offload()
        if self.enable_eval and self.eval_env_active:
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.cfg.env.eval.get("enable_offload", False):
                for env in self.eval_env_list:
                    if hasattr(env, "offload"):
                        env.offload()
        if (
            self.cfg.env.get("libero_verifier", None) is not None
            and self.cfg.env.libero_verifier.get("enabled", False)
        ):
            verifier_env_cls = get_env_cls(
                self.cfg.env.libero_verifier.env_type, self.cfg.env.libero_verifier
            )
            verifier_total_num_envs = int(self.cfg.env.libero_verifier.total_num_envs)
            verifier_parallel_size = self._world_size * self.stage_num
            if verifier_total_num_envs % verifier_parallel_size != 0:
                raise ValueError(
                    "env.libero_verifier.total_num_envs must be divisible by "
                    f"env_worker_world_size * pipeline_stage_num ({verifier_parallel_size}), "
                    f"got {verifier_total_num_envs}."
                )
            verifier_num_envs_per_stage = (
                verifier_total_num_envs // verifier_parallel_size
            )
            self.libero_verifier_env_list = self._setup_env_and_wrappers(
                env_cls=verifier_env_cls,
                env_cfg=self.cfg.env.libero_verifier,
                num_envs_per_stage=verifier_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()

    def update_env_cfg(self):
        # train env
        train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
        if train_override_cfgs is not None:
            assert len(train_override_cfgs) > self._rank, (
                f"{len(train_override_cfgs)=} > {self._rank=}"
            )

            general_train_override_cfg = OmegaConf.to_container(
                self.cfg.env.train.get("override_cfg", {}), resolve=True
            )
            override_cfg = OmegaConf.to_container(
                train_override_cfgs[self._rank], resolve=True
            ).copy()

            base_cfg = {}
            base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
            base_cfg = update_nested_cfg(base_cfg, override_cfg)
            setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))

        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is not None:
            assert len(eval_override_cfgs) > self._rank, (
                f"{len(eval_override_cfgs)=} > {self._rank=}"
            )

            general_eval_override_cfg = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            eval_override_cfg = OmegaConf.to_container(
                eval_override_cfgs[self._rank], resolve=True
            ).copy()
            base_eval_cfg = {}
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    stats_sample_ratio=getattr(
                        env_cfg.data_collection, "stats_sample_ratio", 0.1
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _libero_verifier_enabled(self) -> bool:
        return (
            self.cfg.env.get("libero_verifier", None) is not None
            and self.cfg.env.libero_verifier.get("enabled", False)
        )

    @staticmethod
    def _get_wrapped_env_attr(env, attr_name: str):
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, attr_name):
                value = getattr(current, attr_name)
                if value is not None:
                    return value
            current = getattr(current, "env", None)
        return None

    def _get_env_reset_state_ids(self, env) -> torch.Tensor | None:
        reset_state_ids = self._get_wrapped_env_attr(env, "reset_state_ids")
        if reset_state_ids is None:
            return None
        if isinstance(reset_state_ids, torch.Tensor):
            return reset_state_ids.detach().cpu().to(torch.long).contiguous()
        return torch.as_tensor(reset_state_ids, dtype=torch.long).cpu().contiguous()

    def _verify_world_model_success_in_libero(
        self, stage_id: int
    ) -> dict[str, torch.Tensor]:
        if not self._libero_verifier_enabled():
            return {}
        if not self.libero_verifier_env_list:
            return {}

        rollout_result = self.rollout_results[stage_id]
        if not rollout_result.actions:
            return {}

        trajectory = rollout_result.to_trajectory()
        if (
            trajectory.actions is None
            or trajectory.terminations is None
            or trajectory.reset_state_ids is None
        ):
            return {}

        num_steps, batch_size = trajectory.actions.shape[:2]
        num_action_chunks = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        verifier_env = self.libero_verifier_env_list[stage_id]
        verifier_num_envs = int(self._get_wrapped_env_attr(verifier_env, "num_envs"))
        if verifier_num_envs != batch_size:
            raise ValueError(
                f"LIBERO verifier env count ({verifier_num_envs}) must match rollout batch ({batch_size})."
            )

        aligned_terminations = Trajectory._align_field_to_traj_len(
            trajectory.terminations, num_steps
        )
        wm_success = aligned_terminations.reshape(
            num_steps, batch_size, -1
        ).any(dim=(0, 2))

        reset_state_ids = trajectory.reset_state_ids.reshape(
            num_steps, batch_size, -1
        )[:, :, 0]
        if not torch.equal(
            reset_state_ids, reset_state_ids[:1].expand_as(reset_state_ids)
        ):
            self.log_on_first_rank(
                "LIBERO verifier currently expects one reset_state_id per rollout trajectory; "
                "skipping verification because reset_state_ids changed within a rollout."
            )
            return {}
        reset_state_ids = reset_state_ids[0]

        if not wm_success.any():
            rollout_result.libero_verified_success = torch.zeros(
                num_steps, batch_size, num_action_chunks, dtype=torch.bool
            )
            rollout_result.libero_verified_weight = torch.ones(
                num_steps, batch_size, num_action_chunks, dtype=torch.float32
            )
            return {
                "libero_verifier/wm_success_rate": torch.zeros(1),
                "libero_verifier/libero_success_rate_on_wm_success": torch.zeros(1),
                "libero_verifier/verified_success_rate": torch.zeros(1),
                "libero_verifier/false_positive_rate": torch.zeros(1),
                "libero_verifier/mean_weight": torch.ones(1),
            }

        actions = trajectory.actions.reshape(
            num_steps, batch_size, num_action_chunks, -1
        )
        if actions.shape[-1] < action_dim:
            raise ValueError(
                f"Rollout action dim {actions.shape[-1]} is smaller than configured action_dim {action_dim}."
            )
        action_sequences = (
            actions[..., :action_dim]
            .permute(1, 0, 2, 3)
            .reshape(batch_size, num_steps * num_action_chunks, action_dim)
            .contiguous()
        )
        action_sequences = prepare_actions(
            raw_chunk_actions=action_sequences,
            env_type=self.cfg.env.libero_verifier.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=None,
        )

        rollout_action_sequences = self._get_wrapped_env_attr(
            verifier_env, "rollout_action_sequences"
        )
        if rollout_action_sequences is None:
            raise AttributeError(
                "LIBERO verifier env does not expose rollout_action_sequences"
            )

        libero_success = rollout_action_sequences(
            action_sequences=action_sequences,
            reset_state_ids=reset_state_ids,
            active_mask=wm_success,
            chunk_size=num_action_chunks,
        )
        libero_success = libero_success.to(torch.bool)
        verified_success = wm_success.cpu() & libero_success
        false_positive = wm_success.cpu() & ~libero_success

        verified_weight = torch.ones(batch_size, dtype=torch.float32)
        verified_weight[verified_success] = float(
            self.cfg.algorithm.get("libero_verified_success_weight", 2.0)
        )
        verified_weight[false_positive] = float(
            self.cfg.algorithm.get("libero_false_positive_weight", 0.5)
        )

        rollout_result.libero_verified_success = (
            verified_success.view(1, batch_size, 1)
            .expand(num_steps, batch_size, num_action_chunks)
            .contiguous()
        )
        rollout_result.libero_verified_weight = (
            verified_weight.view(1, batch_size, 1)
            .expand(num_steps, batch_size, num_action_chunks)
            .contiguous()
        )

        return {
            "libero_verifier/wm_success_rate": wm_success.float().mean().view(1),
            "libero_verifier/libero_success_rate_on_wm_success": (
                libero_success[wm_success.cpu()].float().mean().view(1)
                if wm_success.any()
                else torch.zeros(1)
            ),
            "libero_verifier/verified_success_rate": verified_success.float()
            .mean()
            .view(1),
            "libero_verifier/false_positive_rate": false_positive.float()
            .mean()
            .view(1),
            "libero_verifier/mean_weight": verified_weight.mean().view(1),
        }

    def _setup_dst_ranks(
        self, batch_size: int, *, active_world_size: int | None = None
    ) -> list[tuple[int, int]]:
        """Compute rollout peer ranks for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(rollout_rank, batch_size)`` tuples this env worker should send
            env outputs to.
        """
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
            src_active_world_size=active_world_size,
            dst_active_world_size=active_world_size,
        )

    def _setup_src_ranks(
        self, batch_size: int, *, active_world_size: int | None = None
    ) -> list[tuple[int, int]]:
        """Compute rollout source ranks and sizes for receiving action chunks."""
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
            src_active_world_size=active_world_size,
            dst_active_world_size=active_world_size,
        )

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    @staticmethod
    def _to_cpu_if_possible(value: Any) -> Any:
        return value.cpu() if hasattr(value, "cpu") else value

    @staticmethod
    def _slice_metric_by_done_mask(value: Any, done_mask: torch.Tensor) -> Any:
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == done_mask.shape[0]:
            return value[done_mask]
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == done_mask.shape[0]
        ):
            return value[done_mask.cpu().numpy()]
        return value

    def _collect_episode_metrics(
        self,
        env_info: dict[str, Any],
        infos: Any,
        done_mask: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(infos, dict):
            return
        episode_info = infos.get("episode")
        if not isinstance(episode_info, dict):
            return

        for key, value in episode_info.items():
            metric = (
                self._slice_metric_by_done_mask(value, done_mask)
                if done_mask is not None
                else value
            )
            env_info[key] = self._to_cpu_if_possible(metric)

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) and infos_list else {}
        if infos is None:
            infos = {}
        for key, value in infos.items():
            if key.startswith(("self_feedback/", "reward/success_model/", "episode/")):
                env_info[key] = self._to_cpu_if_possible(value)
        world_model_video_chunk = infos.get("world_model_video_chunk")
        world_model_video_chunk_main = infos.get("world_model_video_chunk_main")
        world_model_video_chunk_wrist = infos.get("world_model_video_chunk_wrist")
        world_model_video_chunk_extra = infos.get("world_model_video_chunk_extra")
        ctrl_world_video_chunk = infos.get("ctrl_world_video_chunk")
        ctrl_world_video_chunk_wrist = infos.get("ctrl_world_video_chunk_wrist")
        ctrl_world_video_chunk_extra = infos.get("ctrl_world_video_chunk_extra")
        ctrl_world_video_chunk_raw = infos.get("ctrl_world_video_chunk_raw")
        ctrl_world_video_chunk_raw_wrist = infos.get("ctrl_world_video_chunk_raw_wrist")
        ctrl_world_video_chunk_raw_extra = infos.get("ctrl_world_video_chunk_raw_extra")
        reward_model_frame_probabilities = infos.get("chunk_raw_rewards")
        rollout_retry_count = infos.get("rollout_retry_count")
        for metric_name in (
            "forward_dynamics_action_raw_min",
            "forward_dynamics_action_raw_max",
            "forward_dynamics_action_normalized_min",
            "forward_dynamics_action_normalized_max",
            "forward_dynamics_action_saturation_fraction",
        ):
            metric_value = infos.get(metric_name)
            if metric_value is not None:
                env_info[f"cosmos/{metric_name}"] = (
                    torch.as_tensor(metric_value, dtype=torch.float32)
                    .mean()
                    .cpu()
                    .reshape(1)
                )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    self._collect_episode_metrics(env_info, infos)
            else:
                self._collect_episode_metrics(env_info, infos)
        elif chunk_dones.any():
            done_mask = chunk_dones[:, -1]
            self._collect_episode_metrics(env_info, infos, done_mask=done_mask)
            self._collect_episode_metrics(
                env_info, infos.get("final_info"), done_mask=done_mask
            )

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            final_info = infos.get("final_info")
            if isinstance(final_info, dict) and "intervene_action" in final_info:
                intervene_actions = final_info["intervene_action"]
                intervene_flags = final_info["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dynamic_gammas=infos.get("dynamic_gammas"),
            latent_motion=infos.get("latent_motion"),
            reset_state_ids=infos.get(
                "reset_state_ids", self._get_env_reset_state_ids(self.env_list[stage_id])
            ),
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        if ctrl_world_video_chunk is not None:
            env_info["_ctrl_world_video_chunk"] = ctrl_world_video_chunk
        if ctrl_world_video_chunk_wrist is not None:
            env_info["_ctrl_world_video_chunk_wrist"] = ctrl_world_video_chunk_wrist
        if ctrl_world_video_chunk_extra is not None:
            env_info["_ctrl_world_video_chunk_extra"] = ctrl_world_video_chunk_extra
        if reward_model_frame_probabilities is not None:
            env_info["_reward_model_frame_probabilities"] = (
                reward_model_frame_probabilities
            )
        if rollout_retry_count is not None:
            env_info["_ctrl_world_rollout_retry_count"] = rollout_retry_count
        if ctrl_world_video_chunk_raw is not None:
            env_info["_ctrl_world_video_chunk_raw"] = ctrl_world_video_chunk_raw
        if ctrl_world_video_chunk_raw_wrist is not None:
            env_info["_ctrl_world_video_chunk_raw_wrist"] = ctrl_world_video_chunk_raw_wrist
        if ctrl_world_video_chunk_raw_extra is not None:
            env_info["_ctrl_world_video_chunk_raw_extra"] = ctrl_world_video_chunk_raw_extra
        if world_model_video_chunk is not None:
            env_info["_world_model_video_chunk"] = world_model_video_chunk
        if world_model_video_chunk_main is not None:
            env_info["_world_model_video_chunk_main"] = world_model_video_chunk_main
        if world_model_video_chunk_wrist is not None:
            env_info["_world_model_video_chunk_wrist"] = world_model_video_chunk_wrist
        if world_model_video_chunk_extra is not None:
            env_info["_world_model_video_chunk_extra"] = world_model_video_chunk_extra
        return env_output, env_info

    def _cosmos_video_similarity_enabled(self) -> bool:
        reward_cfg = self.cfg.get("reward", {})
        reward_source = str(reward_cfg.get("source", "video_similarity"))
        return (
            self._is_cosmos_model()
            and reward_source == "video_similarity"
            and reward_cfg.get("video_similarity", {}).get("enabled", False)
        )

    def _terminal_goal_reward_enabled(self) -> bool:
        return bool(
            self._is_cosmos_model()
            and self.cfg.get("reward", {})
            .get("terminal_goal", {})
            .get("enabled", False)
        )

    def _success_classifier_diagnostic_enabled(self) -> bool:
        return bool(
            self._is_cosmos_model()
            and self.cfg.get("reward", {})
            .get("success_classifier", {})
            .get("enabled", False)
        )

    def _training_reward_source(self) -> str:
        source = str(
            self.cfg.get("reward", {}).get(
                "training_source", "continuous_combined"
            )
        )
        allowed = {
            "continuous_combined",
            "trajectory_mse",
            "terminal_goal_mse",
            "success_binary",
        }
        if source not in allowed:
            raise ValueError(
                f"Unsupported reward.training_source {source!r}; "
                f"expected one of {sorted(allowed)}"
            )
        return source

    def _select_training_rewards(
        self,
        *,
        trajectory_rewards: torch.Tensor,
        terminal_rewards: torch.Tensor | None,
        continuous_rewards: torch.Tensor,
        success_rewards: torch.Tensor | None,
    ) -> torch.Tensor:
        source = self._training_reward_source()
        if source == "continuous_combined":
            return continuous_rewards
        if source == "trajectory_mse":
            return trajectory_rewards
        if source == "terminal_goal_mse":
            if terminal_rewards is None:
                raise RuntimeError(
                    "terminal_goal_mse selected without terminal reward diagnostics"
                )
            return terminal_rewards
        if success_rewards is None:
            raise RuntimeError(
                "success_binary selected without terminal classifier diagnostics"
            )
        return success_rewards

    @staticmethod
    def _terminal_goal_image_to_uint8(image: torch.Tensor) -> np.ndarray:
        image = image.detach().cpu()
        if image.dim() != 3:
            raise ValueError("Terminal goal image must have three dimensions.")
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        elif image.shape[-1] not in (1, 3):
            raise ValueError(f"Cannot infer image channels for {image.shape}.")
        image = image.to(torch.float32)
        if image.numel() and float(image.min().item()) < 0.0:
            image = (image + 1.0) / 2.0
        return image.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()

    def _save_terminal_goal_contract_metadata(self) -> None:
        """Write static provenance for same-episode goal selection."""
        if getattr(self, "_rank", 0) != 0:
            return
        goal_cfg = self.cfg.reward.terminal_goal
        output_dir_value = goal_cfg.get("reference_output_dir", None)
        if not output_dir_value:
            return
        output_dir = Path(str(output_dir_value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_camera_keys = {
            name: str(value)
            for name, value in goal_cfg.dataset_camera_keys.items()
        }
        manifest_path_value = self.cfg.env.train.get("episode_manifest_path", None)
        manifest_path = (
            Path(str(manifest_path_value)).expanduser().resolve()
            if manifest_path_value is not None
            else None
        )
        metadata = {
            "reference_mode": "same_episode_last_frame",
            "dataset_path": str(
                goal_cfg.get("dataset_path", self.cfg.env.train.initial_image_path)
            ),
            "dataset_camera_keys": dataset_camera_keys,
            "frame_index": -1,
            "episode_manifest_path": str(manifest_path) if manifest_path else None,
            "episode_manifest_sha256": (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_path is not None and manifest_path.is_file()
                else None
            ),
            "episode_manifest_split": str(
                self.cfg.env.train.get("episode_manifest_split", "training")
            ),
            "window_size": int(goal_cfg.get("window_size", 4)),
            "reward_scale": float(goal_cfg.get("reward_scale", 160.0)),
            "size": [int(value) for value in goal_cfg.get("size", [192, 320])],
            "view_weights": {
                name: float(value)
                for name, value in goal_cfg.get("view_weights", {}).items()
            },
            "saved_sha256": {},
        }
        metadata_path = output_dir / "metadata.json"
        temporary_path = output_dir / "metadata.json.tmp"
        temporary_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, metadata_path)

    def _save_terminal_goal_reference(
        self, reference: dict[str, Any], goal_cfg: Any
    ) -> None:
        if getattr(self, "_rank", 0) != 0:
            return
        output_dir_value = goal_cfg.get("reference_output_dir", None)
        if not output_dir_value:
            return
        output_dir = Path(str(output_dir_value)).expanduser()
        if reference["metadata"].get("reference_mode") == "same_episode_last_frame":
            output_dir = output_dir / (
                f"episode_{int(reference['metadata']['episode_index']):04d}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}
        for view_name in ("main", "wrist", "extra"):
            output_path = output_dir / f"target_{view_name}.png"
            temporary_path = output_path.with_name(
                f"{output_path.stem}.tmp{output_path.suffix}"
            )
            imageio.imwrite(
                temporary_path,
                self._terminal_goal_image_to_uint8(reference[view_name]),
            )
            os.replace(temporary_path, output_path)
            saved_paths[view_name] = str(output_path)
        composite = stitch_cosmos_three_view_chw(
            reference["main"],
            reference["wrist"],
            reference["extra"],
            layout="main_top",
        )
        composite_path = output_dir / "target_main_top.png"
        temporary_composite_path = composite_path.with_name(
            f"{composite_path.stem}.tmp{composite_path.suffix}"
        )
        imageio.imwrite(
            temporary_composite_path,
            self._terminal_goal_image_to_uint8(composite),
        )
        os.replace(temporary_composite_path, composite_path)
        saved_paths["main_top"] = str(composite_path)
        saved_sha256 = {}
        for view_name, saved_path in saved_paths.items():
            digest = hashlib.sha256()
            with Path(saved_path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            saved_sha256[view_name] = digest.hexdigest()
        metadata = {
            **reference["metadata"],
            "window_size": int(goal_cfg.get("window_size", 4)),
            "reward_scale": float(goal_cfg.get("reward_scale", 160.0)),
            "size": [int(value) for value in goal_cfg.get("size", [192, 320])],
            "view_weights": {
                name: float(value)
                for name, value in goal_cfg.get("view_weights", {}).items()
            },
            "saved_paths": saved_paths,
            "saved_sha256": saved_sha256,
        }
        metadata_path = output_dir / "metadata.json"
        temporary_metadata_path = output_dir / "metadata.json.tmp"
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata_path, metadata_path)

    def _get_terminal_goal_reference(
        self, episode_index: int | None = None
    ) -> dict[str, Any]:
        goal_cfg = self.cfg.reward.terminal_goal
        reference_mode = str(goal_cfg.get("reference_mode", "fixed")).lower()
        if reference_mode not in {"fixed", "same_episode_last_frame"}:
            raise ValueError(
                "terminal_goal.reference_mode must be 'fixed' or "
                f"'same_episode_last_frame', got {reference_mode!r}."
            )
        if reference_mode == "fixed" and self._terminal_goal_reference is not None:
            return self._terminal_goal_reference
        if reference_mode == "same_episode_last_frame":
            if episode_index is None:
                raise ValueError(
                    "same_episode_last_frame terminal goals require an episode id"
                )
            episode_index = int(episode_index)
            cached = self._terminal_goal_references.pop(episode_index, None)
            if cached is not None:
                self._terminal_goal_references[episode_index] = cached
                return cached
        dataset_path = str(
            goal_cfg.get("dataset_path", self.cfg.env.train.initial_image_path)
        )
        camera_cfg = goal_cfg.dataset_camera_keys
        camera_keys = {
            "main": str(camera_cfg.main),
            "wrist": str(camera_cfg.wrist),
            "extra": str(camera_cfg.extra),
        }
        if len(set(camera_keys.values())) != 3:
            raise ValueError("Terminal goal requires three distinct camera keys.")
        from rlinf.data.datasets.lerobot_book import (
            LeRobotBookTrajectoryDatasetWrapper,
        )

        if self._terminal_goal_dataset is None:
            self._terminal_goal_dataset = LeRobotBookTrajectoryDatasetWrapper(
                data_dir=dataset_path,
                camera_keys=tuple(camera_keys.values()),
            )
        dataset = self._terminal_goal_dataset
        frame = dataset.load_frame_views(
            episode_index=(
                episode_index
                if reference_mode == "same_episode_last_frame"
                else int(goal_cfg.get("dataset_episode_index", -1))
            ),
            frame_index=(
                -1
                if reference_mode == "same_episode_last_frame"
                else int(goal_cfg.get("dataset_frame_index", -1))
            ),
        )
        reference = {
            view_name: frame["images"][image_key].cpu().contiguous()
            for view_name, image_key in camera_keys.items()
        }
        reference["metadata"] = {
            "dataset_path": dataset_path,
            "episode_index": int(frame["episode_index"]),
            "frame_index": int(frame["frame_index"]),
            "timestamp": float(frame["timestamp"]),
            "dataset_camera_keys": camera_keys,
            "ctrl_world_view_indices": {"main": 2, "wrist": 0, "extra": 1},
            "reference_mode": reference_mode,
        }
        if reference_mode == "fixed":
            self._terminal_goal_reference = reference
        else:
            self._terminal_goal_references[int(frame["episode_index"])] = reference
            cache_size = int(goal_cfg.get("cache_size", 8))
            if cache_size <= 0:
                raise ValueError("terminal_goal.cache_size must be positive")
            while len(self._terminal_goal_references) > cache_size:
                self._terminal_goal_references.popitem(last=False)
        self._save_terminal_goal_reference(reference, goal_cfg)
        return reference

    def _apply_terminal_goal_reward(
        self,
        rewards: torch.Tensor,
        env_output: EnvOutput,
        env_info: dict[str, Any],
        *,
        ctrl_world_video_chunk: torch.Tensor,
        ctrl_world_video_chunk_wrist: torch.Tensor,
        ctrl_world_video_chunk_extra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._terminal_goal_reward_enabled():
            return rewards, None
        done_mask = env_output.dones
        if done_mask.dim() > 1:
            done_mask = done_mask[..., -1]
        done_mask = done_mask.to(device=rewards.device, dtype=torch.bool)
        terminal_component = torch.zeros_like(rewards, dtype=torch.float32)
        if done_mask.any():
            goal_cfg = self.cfg.reward.terminal_goal
            logical_videos = {
                "main": ctrl_world_video_chunk,
                "wrist": ctrl_world_video_chunk_wrist,
                "extra": ctrl_world_video_chunk_extra,
            }
            source_mapping = goal_cfg.get(
                "ctrl_world_source_mapping",
                {"main": "main", "wrist": "wrist", "extra": "extra"},
            )
            mapping = {
                name: str(source_mapping.get(name, name))
                for name in ("main", "wrist", "extra")
            }
            if sorted(mapping.values()) != ["extra", "main", "wrist"]:
                raise ValueError(
                    "terminal_goal.ctrl_world_source_mapping must be a "
                    f"permutation of main/wrist/extra, got {mapping}."
                )
            reference_mode = str(goal_cfg.get("reference_mode", "fixed")).lower()
            reset_episode_ids = env_output.reset_state_ids
            if reference_mode == "same_episode_last_frame":
                if reset_episode_ids is None:
                    raise ValueError(
                        "per-episode terminal goals require EnvOutput.reset_state_ids"
                    )
                reset_episode_ids = (
                    torch.as_tensor(reset_episode_ids)
                    .detach()
                    .cpu()
                    .to(torch.int64)
                    .reshape(-1)
                )
                if reset_episode_ids.numel() != rewards.shape[0]:
                    raise ValueError(
                        "reset_state_ids must contain one episode id per reward row"
                    )
                unique_episode_ids = sorted(
                    set(
                        int(value)
                        for value in reset_episode_ids[done_mask.cpu()].tolist()
                    )
                )
            else:
                unique_episode_ids = [None]

            goal_rewards = torch.zeros(
                rewards.shape[0], dtype=torch.float32, device=rewards.device
            )
            per_episode_mse = torch.full(
                (rewards.shape[0],), float("nan"), dtype=torch.float32
            )
            per_view_episode_mse = {
                name: torch.full((rewards.shape[0],), float("nan"), dtype=torch.float32)
                for name in ("main", "wrist", "extra")
            }
            reference_episode = torch.full(
                (rewards.shape[0],), -1, dtype=torch.int64
            )
            reference_frame = torch.full(
                (rewards.shape[0],), -1, dtype=torch.int64
            )
            for requested_episode_id in unique_episode_ids:
                reference = self._get_terminal_goal_reference(requested_episode_id)
                if requested_episode_id is None:
                    selected = done_mask
                else:
                    selected = done_mask & (
                        reset_episode_ids.to(done_mask.device)
                        == requested_episode_id
                    )
                selected_rewards, selected_stats = compute_terminal_goal_reward(
                    logical_videos[mapping["main"]][selected],
                    reference["main"],
                    ctrl_world_video_chunk_wrist=logical_videos[mapping["wrist"]][
                        selected
                    ],
                    ctrl_world_video_chunk_extra=logical_videos[mapping["extra"]][
                        selected
                    ],
                    goal_wrist_image=reference["wrist"],
                    goal_extra_image=reference["extra"],
                    window_size=int(goal_cfg.get("window_size", 4)),
                    size=goal_cfg.get("size", [192, 320]),
                    view_weights=goal_cfg.view_weights,
                    range_policy=goal_cfg.get("range_policy", "error"),
                    reward_scale=float(goal_cfg.get("reward_scale", 160.0)),
                    return_stats=True,
                )
                goal_rewards[selected] = selected_rewards.to(rewards.device)
                selected_cpu = selected.detach().cpu()
                per_episode_mse[selected_cpu] = selected_stats[
                    "per_episode_mse"
                ].detach().cpu()
                for goal_name, physical_name in mapping.items():
                    per_view_episode_mse[physical_name][selected_cpu] = selected_stats[
                        f"{goal_name}_per_episode_mse"
                    ].detach().cpu()
                reference_episode[selected_cpu] = int(
                    reference["metadata"]["episode_index"]
                )
                reference_frame[selected_cpu] = int(
                    reference["metadata"]["frame_index"]
                )
            terminal_component[done_mask, -1] = goal_rewards[done_mask]
            env_info["reward/terminal_goal_mse"] = per_episode_mse[done_mask.cpu()]
            for name, values in per_view_episode_mse.items():
                env_info[f"reward/terminal_goal_{name}_mse"] = values[done_mask.cpu()]
            env_info["reward/terminal_goal"] = goal_rewards[
                done_mask
            ].detach().cpu()
            env_info["reward/terminal_goal_reference_episode"] = reference_episode[
                done_mask.cpu()
            ]
            env_info["reward/terminal_goal_reference_frame"] = reference_frame[
                done_mask.cpu()
            ]
        return rewards.to(torch.float32) + terminal_component, terminal_component

    def _apply_success_classifier_diagnostic(
        self,
        rewards: torch.Tensor,
        env_output: EnvOutput,
        env_info: dict[str, Any],
    ) -> torch.Tensor | None:
        if not self._success_classifier_diagnostic_enabled():
            return None
        probabilities = env_info.pop("_reward_model_frame_probabilities", None)
        if probabilities is None:
            raise KeyError(
                "Success diagnostic requires Ctrl-World chunk_raw_rewards."
            )
        probabilities = probabilities.to(rewards.device, dtype=torch.float32)
        if probabilities.dim() != 2 or probabilities.shape[0] != rewards.shape[0]:
            raise ValueError(
                "Success probabilities must be [B,T] with matching batch."
            )
        if not torch.isfinite(probabilities).all() or (
            (probabilities < 0).any() or (probabilities > 1).any()
        ):
            raise ValueError("Success probabilities must be finite and in [0,1].")
        cfg = self.cfg.reward.success_classifier
        decision = classify_terminal_probabilities(probabilities, cfg)
        done_mask = env_output.dones
        if done_mask.dim() > 1:
            done_mask = done_mask[..., -1]
        done_mask = done_mask.to(rewards.device, dtype=torch.bool)
        if decision["rule"] == "terminal_positive_ratio":
            recorded = probabilities.detach().clone()
        else:
            recorded = torch.zeros_like(rewards, dtype=torch.float32)
        success_rewards = torch.zeros_like(rewards, dtype=torch.float32)
        if done_mask.any():
            final_probabilities = decision["sampled_probabilities"]
            successes = decision["success"]
            reward_value = float(cfg.get("reward_value", 1.0))
            success_rewards[done_mask, -1] = (
                successes[done_mask].to(torch.float32) * reward_value
            )
            env_info["reward/success_binary"] = success_rewards[
                done_mask, -1
            ].detach().cpu()
            if decision["rule"] == "legacy_max_window":
                window_size = int(final_probabilities.shape[1])
                recorded[done_mask, -window_size:] = final_probabilities[done_mask]
            env_info["reward/success"] = successes[done_mask].float().detach().cpu()
            env_info["reward/success_probability"] = decision[
                "reported_probability"
            ][done_mask].detach().cpu()
            env_info["reward/success_last_probability"] = decision[
                "last_probability"
            ][done_mask].detach().cpu()
            env_info["reward/success_positive_ratio"] = decision[
                "positive_ratio"
            ][done_mask].detach().cpu()
            env_info["reward/success_probability_max"] = decision[
                "probability_max"
            ][done_mask].detach().cpu()
            env_info["reward/success_final_probabilities"] = final_probabilities[
                done_mask
            ].detach().cpu()
            env_info["reward/success_sample_indices"] = decision[
                "sample_indices"
            ].detach().cpu()
            # The decision rule is string metadata, not a numeric runtime
            # metric. It is retained in trajectory and post-update eval
            # records as `success_decision_rule`; putting it in env_metrics
            # makes the Tensor aggregation at the end of rollout fail.
            env_info["cosmos/success_rate"] = successes[
                done_mask
            ].float().mean().detach().cpu().reshape(1)
        env_info["_success_binary_rewards"] = (
            success_rewards.detach().cpu().contiguous()
        )
        return recorded

    def _is_cosmos_model(self) -> bool:
        return self.cfg.actor.model.model_type == "cosmos"

    @staticmethod
    def _parse_perceptual_frame_index(reward_cfg) -> int | None:
        """clip_frame_index config value -> int, or None for mean-over-frames.

        `reward.video_similarity.clip_frame_index` defaults to -1 (last frame
        only, matching the plan doc's original "last-frame MSE" framing).
        Set it to null/"mean" to instead average per-frame CLIP/DINO cosine
        similarity over every frame in the chunk, matching the actual
        training reward's reward_type=chunk_level aggregation (which sums
        all per-frame pixel rewards, not just the last one) far more closely.
        """
        value = reward_cfg.get("clip_frame_index", -1)
        if value is None or str(value).lower() == "mean":
            return None
        return int(value)

    def _is_cosmos_self_feedback(self, mode: str = "train") -> bool:
        return self.cfg.env[mode].env_type == "cosmos_self_ctrl_world"

    def _set_policy_feedback(
        self, env: Any, imagined_video_chunk: torch.Tensor | None, *, mode: str
    ) -> None:
        setter = self._get_wrapped_env_attr(env, "set_policy_feedback")
        if callable(setter):
            setter(imagined_video_chunk)
            return
        if self._is_cosmos_self_feedback(mode):
            raise RuntimeError(
                "cosmos_self_ctrl_world environment does not expose "
                "set_policy_feedback()"
            )

    def _retain_video_payload_in_actor_batch(self) -> bool:
        """Whether rollout-only videos should be retained in Actor replay."""
        rollout_cfg = self.cfg.get("rollout", {})
        return bool(rollout_cfg.get("retain_video_payload_in_actor_batch", True))

    def _retain_imagined_video_for_actor_replay(self) -> bool:
        """FPO replay must retain the joint sample imagined video."""
        cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
        objective = resolve_cosmos_replay_objective(
            cosmos_cfg.get("replay_objective", None)
        )
        return objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD

    @staticmethod
    def _is_rollout_only_video_payload_key(key: str) -> bool:
        return key.startswith(("ctrl_world_video_chunk", "world_model_video_chunk"))

    def _actor_replay_forward_inputs(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Drop locally consumed video tensors before buffering Actor replay.

        Cosmos imagination and Ctrl/world-model frames are consumed by
        self-feedback, rewards, success diagnostics, and bounded artifact
        recording on the Env rank. Native action-logprob replay does not read
        them. Keeping those videos in every five-chunk trajectory makes a
        ChannelWorker retain hundreds of GiB before the Actor update starts.
        """
        if (
            not self._is_cosmos_model()
            or self._retain_video_payload_in_actor_batch()
        ):
            return forward_inputs
        retain_imagined_video = self._retain_imagined_video_for_actor_replay()
        return {
            key: value
            for key, value in forward_inputs.items()
            if (
                key != "imagined_video_chunk" or retain_imagined_video
            ) and not self._is_rollout_only_video_payload_key(key)
        }

    def _attach_cosmos_video_reward(
        self,
        rollout_result: RolloutResult,
        env_output: EnvOutput,
        env_info: dict[str, Any],
        default_rewards: torch.Tensor | None,
        *,
        comparison_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        ctrl_world_video_chunk = env_info.pop("_ctrl_world_video_chunk", None)
        ctrl_world_video_chunk_wrist = env_info.pop("_ctrl_world_video_chunk_wrist", None)
        ctrl_world_video_chunk_extra = env_info.pop("_ctrl_world_video_chunk_extra", None)
        ctrl_world_video_chunk_raw = env_info.pop("_ctrl_world_video_chunk_raw", None)
        ctrl_world_video_chunk_raw_wrist = env_info.pop("_ctrl_world_video_chunk_raw_wrist", None)
        ctrl_world_video_chunk_raw_extra = env_info.pop("_ctrl_world_video_chunk_raw_extra", None)
        world_model_video_chunk = env_info.pop("_world_model_video_chunk", None)
        world_model_video_chunk_main = env_info.pop("_world_model_video_chunk_main", None)
        world_model_video_chunk_wrist = env_info.pop("_world_model_video_chunk_wrist", None)
        world_model_video_chunk_extra = env_info.pop("_world_model_video_chunk_extra", None)
        retain_video_payload = self._retain_video_payload_in_actor_batch()
        if (
            self._is_cosmos_model()
            and retain_video_payload
            and ctrl_world_video_chunk is not None
        ):
            rollout_result.forward_inputs["ctrl_world_video_chunk"] = (
                ctrl_world_video_chunk.detach().cpu().contiguous()
            )
        if (
            self._is_cosmos_model()
            and retain_video_payload
            and world_model_video_chunk is not None
        ):
            rollout_result.forward_inputs["world_model_video_chunk"] = (
                world_model_video_chunk.detach().cpu().contiguous()
            )

        if not self._cosmos_video_similarity_enabled():
            # Comparison-video recording is independent of the reward source.
            # In success-model mode the pulse rewards are already present in
            # ``default_rewards``; returning before buffering would silently
            # disable best/worst-rollout videos.
            record_best_worst = (
                self._save_best_rollout_video()
                or self._save_worst_rollout_video()
            )
            stream_comparison = self._stream_all_cosmos_comparison("train")
            if record_best_worst or stream_comparison:
                if record_best_worst and default_rewards is None:
                    raise ValueError(
                        "Best/worst-rollout video selection requires rewards."
                    )
                imagined_video_chunk = rollout_result.forward_inputs.get(
                    "imagined_video_chunk"
                )
                if not isinstance(imagined_video_chunk, torch.Tensor):
                    raise KeyError(
                        "Best/worst-rollout comparison video requires "
                        "imagined_video_chunk."
                    )
                comparison_ctrl_world_video_chunk = (
                    world_model_video_chunk
                    if world_model_video_chunk is not None
                    else (
                        ctrl_world_video_chunk_raw
                        if ctrl_world_video_chunk_raw is not None
                        else ctrl_world_video_chunk
                    )
                )
                if not isinstance(comparison_ctrl_world_video_chunk, torch.Tensor):
                    raise KeyError(
                        "Best/worst-rollout comparison video requires "
                        "Ctrl-World or world-model frames."
                    )
                comparison_ctrl_world_video_chunk_wrist = (
                    ctrl_world_video_chunk_raw_wrist
                    if ctrl_world_video_chunk_raw_wrist is not None
                    else (
                        world_model_video_chunk_wrist
                        if world_model_video_chunk is not None
                        else ctrl_world_video_chunk_wrist
                    )
                )
                comparison_ctrl_world_video_chunk_extra = (
                    ctrl_world_video_chunk_raw_extra
                    if ctrl_world_video_chunk_raw_extra is not None
                    else (
                        world_model_video_chunk_extra
                        if world_model_video_chunk is not None
                        else ctrl_world_video_chunk_extra
                    )
                )
                if retain_video_payload and world_model_video_chunk_main is not None:
                    rollout_result.forward_inputs["world_model_video_chunk_main"] = (
                        world_model_video_chunk_main.detach().cpu().contiguous()
                    )
                if record_best_worst:
                    self._buffer_best_rollout_comparison_video(
                        imagined_video_chunk=imagined_video_chunk,
                        ctrl_world_video_chunk=comparison_ctrl_world_video_chunk,
                        ctrl_world_video_chunk_wrist=(
                            comparison_ctrl_world_video_chunk_wrist
                        ),
                        ctrl_world_video_chunk_extra=(
                            comparison_ctrl_world_video_chunk_extra
                        ),
                        rewards=default_rewards,
                        chunk_mse=None,
                        successes=None,
                        done_mask=env_output.dones,
                        reward_cfg=self.cfg.reward.video_similarity,
                    )
                if stream_comparison:
                    self._stream_cosmos_comparison_video(
                        imagined_video_chunk=imagined_video_chunk,
                        ctrl_world_video_chunk=comparison_ctrl_world_video_chunk,
                        ctrl_world_video_chunk_wrist=(
                            comparison_ctrl_world_video_chunk_wrist
                        ),
                        ctrl_world_video_chunk_extra=(
                            comparison_ctrl_world_video_chunk_extra
                        ),
                        reward_cfg=self.cfg.reward.video_similarity,
                        metadata=comparison_metadata,
                    )
            return default_rewards
        reference_video_chunk = (
            world_model_video_chunk
            if world_model_video_chunk is not None
            else ctrl_world_video_chunk
        )
        if reference_video_chunk is None:
            raise KeyError(
                "Cosmos video similarity reward requires world_model_video_chunk "
                "or ctrl_world_video_chunk."
            )
        if "imagined_video_chunk" not in rollout_result.forward_inputs:
            raise KeyError(
                "Cosmos video similarity reward requires imagined_video_chunk."
            )

        reward_cfg = self.cfg.reward.video_similarity
        alignment_mode = reward_cfg.get("alignment_mode", "legacy")
        cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
        camera_layout = reward_cfg.get(
            "camera_layout", cosmos_cfg.get("camera_layout", "main_top")
        )
        rewards, reward_stats = compute_video_similarity_reward(
            rollout_result.forward_inputs["imagined_video_chunk"],
            reference_video_chunk,
            size=reward_cfg.get("size", reward_cfg.get("video_similarity_size", None)),
            ctrl_world_video_chunk_wrist=ctrl_world_video_chunk_wrist,
            ctrl_world_video_chunk_extra=ctrl_world_video_chunk_extra,
            alignment_mode=alignment_mode,
            camera_layout=camera_layout,
            metric=reward_cfg.get("metric", "neg_mse"),
            range_policy=reward_cfg.get("range_policy", "error"),
            return_stats=True,
            composite_view_stats=reward_cfg.get("reference_is_composite", False),
            view_weights=reward_cfg.get("view_weights", None),
        )
        alignment_stats = {alignment_mode: reward_stats}
        if reward_cfg.get("log_counterfactual_alignment", False):
            counterfactual_mode = (
                "aligned" if alignment_mode == "legacy" else "legacy"
            )
            _, counterfactual_stats = compute_video_similarity_reward(
                rollout_result.forward_inputs["imagined_video_chunk"],
                reference_video_chunk,
                size=reward_cfg.get(
                    "size", reward_cfg.get("video_similarity_size", None)
                ),
                ctrl_world_video_chunk_wrist=ctrl_world_video_chunk_wrist,
                ctrl_world_video_chunk_extra=ctrl_world_video_chunk_extra,
                alignment_mode=counterfactual_mode,
                camera_layout=camera_layout,
                metric=reward_cfg.get("metric", "neg_mse"),
                range_policy=reward_cfg.get("range_policy", "error"),
                return_stats=True,
                composite_view_stats=reward_cfg.get("reference_is_composite", False),
                view_weights=(reward_cfg.get("view_weights", None) if (
                    counterfactual_mode == "aligned"
                    or reward_cfg.get("reference_is_composite", False)
                ) else None),
            )
            alignment_stats[counterfactual_mode] = counterfactual_stats
        reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        rewards = rewards.to(dtype=torch.float32) * reward_scale
        if not torch.isfinite(rewards).all():
            raise ValueError("Cosmos video similarity reward produced NaN or Inf.")
        video_similarity_rewards = rewards.detach().clone()
        if (
            ctrl_world_video_chunk is None
            or ctrl_world_video_chunk_wrist is None
            or ctrl_world_video_chunk_extra is None
        ) and self._terminal_goal_reward_enabled():
            raise ValueError(
                "Terminal reward requires all three Ctrl-World video views."
            )
        continuous_rewards, terminal_goal_rewards = self._apply_terminal_goal_reward(
            rewards,
            env_output,
            env_info,
            ctrl_world_video_chunk=ctrl_world_video_chunk,
            ctrl_world_video_chunk_wrist=ctrl_world_video_chunk_wrist,
            ctrl_world_video_chunk_extra=ctrl_world_video_chunk_extra,
        )
        reward_model_probabilities = self._apply_success_classifier_diagnostic(
            continuous_rewards, env_output, env_info
        )
        success_binary_rewards = env_info.pop(
            "_success_binary_rewards", None
        )
        rewards = self._select_training_rewards(
            trajectory_rewards=video_similarity_rewards,
            terminal_rewards=terminal_goal_rewards,
            continuous_rewards=continuous_rewards,
            success_rewards=success_binary_rewards,
        )
        env_info["_video_similarity_rewards"] = (
            video_similarity_rewards.detach().cpu().contiguous()
        )
        if terminal_goal_rewards is not None:
            env_info["_terminal_goal_rewards"] = (
                terminal_goal_rewards.detach().cpu().contiguous()
            )
        env_info["_continuous_combined_rewards"] = (
            rewards.detach().cpu().contiguous()
        )
        if reward_model_probabilities is not None:
            env_info["_reward_model_probabilities"] = (
                reward_model_probabilities.detach().cpu().contiguous()
            )
        env_info["cosmos/video_similarity_reward_mean"] = (
            reward_stats["reward_mean"].detach().cpu().reshape(1)
        )
        env_info["cosmos/video_similarity_reward_std"] = (
            reward_stats["reward_std"].detach().cpu().reshape(1)
        )
        env_info["cosmos/video_similarity_reward_min"] = (
            reward_stats["reward_min"].detach().cpu().reshape(1)
        )
        env_info["cosmos/video_similarity_reward_max"] = (
            reward_stats["reward_max"].detach().cpu().reshape(1)
        )
        env_info["cosmos/video_similarity_mse_mean"] = (
            reward_stats["mse_mean"].detach().cpu().reshape(1)
        )
        env_info["reward/video_mse"] = reward_stats["mse_mean"].detach().cpu().reshape(1)
        env_info["reward/video_pixel_mse"] = reward_stats["pixel_mse_mean"].detach().cpu().reshape(1)
        if reward_cfg.get("view_weights", None) is not None:
            self._record_view_reward_diagnostics(
                reward_stats, env_output, env_info, comparison_metadata
            )
        for mode, mode_stats in alignment_stats.items():
            prefix = f"cosmos/video_similarity_alignment/{mode}"
            env_info[f"{prefix}/mse_mean"] = (
                mode_stats["mse_mean"].detach().cpu().reshape(1)
            )
            env_info[f"{prefix}/reward_mean"] = (
                mode_stats["reward_mean"].detach().cpu().reshape(1)
            )
            for view_name in ("main", "wrist", "extra"):
                stats_key = f"{view_name}_mse_mean"
                if stats_key in mode_stats:
                    env_info[f"{prefix}/{view_name}_mse_mean"] = (
                        mode_stats[stats_key].detach().cpu().reshape(1)
                    )
        for view_name in ("main", "wrist", "extra"):
            stats_key = f"{view_name}_mse_mean"
            if stats_key not in reward_stats:
                continue
            value = reward_stats[stats_key].detach().cpu().reshape(1)
            env_info[f"cosmos/video_similarity_{view_name}_mse_mean"] = value
            env_info[f"reward/video_mse/{view_name}"] = value
        env_info["cosmos/imagined_video_min"] = (
            reward_stats["imagined_min"].detach().cpu().reshape(1)
        )
        env_info["cosmos/imagined_video_max"] = (
            reward_stats["imagined_max"].detach().cpu().reshape(1)
        )
        env_info["cosmos/ctrl_world_video_min"] = (
            reward_stats["ctrl_world_min"].detach().cpu().reshape(1)
        )
        env_info["cosmos/ctrl_world_video_max"] = (
            reward_stats["ctrl_world_max"].detach().cpu().reshape(1)
        )
        env_info["cosmos/reference_video_min"] = (
            reward_stats["ctrl_world_min"].detach().cpu().reshape(1)
        )
        env_info["cosmos/reference_video_max"] = (
            reward_stats["ctrl_world_max"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/imagined_min"] = (
            reward_stats["imagined_min"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/imagined_max"] = (
            reward_stats["imagined_max"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/ctrl_world_min"] = (
            reward_stats["ctrl_world_min"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/ctrl_world_max"] = (
            reward_stats["ctrl_world_max"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/reference_min"] = (
            reward_stats["reference_min"].detach().cpu().reshape(1)
        )
        env_info["reward/video_range_min_max/reference_max"] = (
            reward_stats["reference_max"].detach().cpu().reshape(1)
        )
        comparison_ctrl_world_video_chunk = (
            ctrl_world_video_chunk_raw
            if ctrl_world_video_chunk_raw is not None
            else ctrl_world_video_chunk
        )
        if world_model_video_chunk is not None:
            comparison_ctrl_world_video_chunk = world_model_video_chunk
            ctrl_world_video_chunk_wrist = world_model_video_chunk_wrist
            ctrl_world_video_chunk_extra = world_model_video_chunk_extra
            if retain_video_payload and world_model_video_chunk_main is not None:
                rollout_result.forward_inputs["world_model_video_chunk_main"] = (
                    world_model_video_chunk_main.detach().cpu().contiguous()
                )
        comparison_ctrl_world_video_chunk_wrist = (
            ctrl_world_video_chunk_raw_wrist
            if ctrl_world_video_chunk_raw_wrist is not None
            else ctrl_world_video_chunk_wrist
        )
        comparison_ctrl_world_video_chunk_extra = (
            ctrl_world_video_chunk_raw_extra
            if ctrl_world_video_chunk_raw_extra is not None
            else ctrl_world_video_chunk_extra
        )
        if self._track_rollout_summaries():
            self._buffer_best_rollout_comparison_video(
                imagined_video_chunk=rollout_result.forward_inputs[
                    "imagined_video_chunk"
                ],
                ctrl_world_video_chunk=comparison_ctrl_world_video_chunk,
                ctrl_world_video_chunk_wrist=comparison_ctrl_world_video_chunk_wrist,
                ctrl_world_video_chunk_extra=comparison_ctrl_world_video_chunk_extra,
                rewards=rewards,
                chunk_mse=reward_stats["per_frame_mse"].mean(dim=1),
                successes=env_info.get("reward/success"),
                done_mask=env_output.dones,
                reward_cfg=reward_cfg,
                final_side_video=ctrl_world_video_chunk_raw_extra,
                rollout_metadata=comparison_metadata,
            )
        if self._stream_all_cosmos_comparison("train"):
            self._stream_cosmos_comparison_video(
                imagined_video_chunk=rollout_result.forward_inputs[
                    "imagined_video_chunk"
                ],
                ctrl_world_video_chunk=comparison_ctrl_world_video_chunk,
                ctrl_world_video_chunk_wrist=(
                    comparison_ctrl_world_video_chunk_wrist
                ),
                ctrl_world_video_chunk_extra=(
                    comparison_ctrl_world_video_chunk_extra
                ),
                reward_cfg=reward_cfg,
                metadata=comparison_metadata,
            )
        if self.cfg.env.train.get("video_cfg", {}).get("save_video", False):
            self._save_cosmos_comparison_video(
                imagined_video_chunk=rollout_result.forward_inputs["imagined_video_chunk"],
                ctrl_world_video_chunk=comparison_ctrl_world_video_chunk,
                ctrl_world_video_chunk_wrist=comparison_ctrl_world_video_chunk_wrist,
                ctrl_world_video_chunk_extra=comparison_ctrl_world_video_chunk_extra,
                reward_cfg=reward_cfg,
            )

        diagnostics_cfg = self.cfg.get("algorithm", {}).get("diagnostics", {})
        if diagnostics_cfg.get("warn_video_range", True):
            for warning in find_video_range_warnings(
                imagined_video_chunk=rollout_result.forward_inputs["imagined_video_chunk"],
                ctrl_world_video_chunk=reference_video_chunk,
            ):
                self.log_on_first_rank(f"Cosmos video range warning: {warning}")
        if env_output.rewards is not None:
            rewards = rewards.to(env_output.rewards.device)
        return rewards

    @staticmethod
    def _comparison_metadata_vector(
        metadata: dict[str, torch.Tensor] | None,
        name: str,
        batch_size: int,
    ) -> list[int]:
        if metadata is None or name not in metadata:
            raise ValueError(
                "streamed comparison video requires trajectory metadata "
                f"field {name!r}"
            )
        values = torch.as_tensor(metadata[name]).detach().cpu().reshape(-1)
        if values.numel() != batch_size:
            raise ValueError(
                f"comparison metadata {name} has {values.numel()} values, "
                f"expected {batch_size}"
            )
        return [int(value) for value in values.tolist()]

    def _abort_comparison_stream(self, mode: str, env_id: int) -> None:
        streams = self._comparison_streams.get(mode, {})
        state = streams.pop(env_id, None)
        if state is None:
            return
        try:
            state["writer"].close()
        except Exception:
            pass
        try:
            Path(state["partial_path"]).unlink(missing_ok=True)
        except OSError:
            pass

    def _stream_cosmos_comparison_video(
        self,
        *,
        imagined_video_chunk: torch.Tensor,
        ctrl_world_video_chunk: torch.Tensor,
        ctrl_world_video_chunk_wrist: torch.Tensor | None,
        ctrl_world_video_chunk_extra: torch.Tensor | None,
        reward_cfg: Any,
        metadata: dict[str, torch.Tensor] | None,
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Append one comparison chunk directly to per-trajectory MP4 files.

        The encoder pipe is synchronous and unqueued: if x264 or the filesystem
        falls behind, rollout waits instead of retaining another full chunk in
        host memory. Final files become visible only after every chunk closes.
        """
        if mode != "train":
            raise ValueError(
                "streamed comparison recording currently supports train only"
            )
        batch_size = int(imagined_video_chunk.shape[0])
        field_names = (
            "rollout_uid",
            "global_group_id",
            "group_member_id",
            "source_env_rank",
            "local_env_id",
            "update_id",
            "logical_round_id",
            "physical_wave_id",
            "group_slot",
            "chunk_id",
            "reset_episode",
        )
        fields = {
            name: self._comparison_metadata_vector(metadata, name, batch_size)
            for name in field_names
        }
        chunk_frames = max(
            self._video_timesteps(imagined_video_chunk),
            self._video_timesteps(ctrl_world_video_chunk),
        )
        expected_frames = self.n_train_chunk_steps * chunk_frames
        video_cfg = self.cfg.env[mode].video_cfg
        fps = int(video_cfg.get("fps", None) or 30)
        encoder_threads = int(
            video_cfg.get("stream_comparison_encoder_threads", 1)
        )
        if fps <= 0:
            raise ValueError("streamed comparison fps must be positive")
        if encoder_threads <= 0:
            raise ValueError("stream_comparison_encoder_threads must be positive")
        streams = self._comparison_streams.setdefault(mode, {})

        for env_id in range(batch_size):
            local_env_id = fields["local_env_id"][env_id]
            source_rank = fields["source_env_rank"][env_id]
            chunk_id = fields["chunk_id"][env_id]
            if local_env_id != env_id or source_rank != self._rank:
                raise ValueError(
                    "streamed comparison source identity mismatch: "
                    f"rank={source_rank}/{self._rank}, env={local_env_id}/{env_id}"
                )
            state = streams.get(env_id)
            if state is None:
                if chunk_id != 0:
                    raise ValueError(
                        f"comparison stream for env {env_id} starts at "
                        f"chunk {chunk_id}"
                    )
                update_id = fields["update_id"][env_id]
                logical_round = fields["logical_round_id"][env_id]
                rollout_uid = fields["rollout_uid"][env_id]
                output_dir = (
                    Path(str(video_cfg.video_base_dir))
                    / str(
                        video_cfg.get(
                            "stream_comparison_subdir",
                            "cosmos_vs_ctrlworld_all",
                        )
                    )
                    / f"update_{update_id:04d}"
                    / f"round_{logical_round:03d}"
                    / f"rank_{self._rank:05d}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                final_path = output_dir / f"uid_{rollout_uid}.mp4"
                partial_path = output_dir / f"uid_{rollout_uid}.partial.mp4"
                partial_path.unlink(missing_ok=True)
                metadata_path = final_path.with_suffix(".json")
                metadata_path.with_suffix(".json.tmp").unlink(missing_ok=True)
                writer = imageio.get_writer(
                    str(partial_path),
                    fps=fps,
                    codec="libx264",
                    ffmpeg_params=["-threads", str(encoder_threads)],
                )
                state = {
                    "writer": writer,
                    "partial_path": partial_path,
                    "final_path": final_path,
                    "metadata_path": metadata_path,
                    "rollout_uid": rollout_uid,
                    "global_group_id": fields["global_group_id"][env_id],
                    "group_member_id": fields["group_member_id"][env_id],
                    "source_env_rank": source_rank,
                    "local_env_id": local_env_id,
                    "update_id": update_id,
                    "logical_round_id": logical_round,
                    "physical_wave_id": fields["physical_wave_id"][env_id],
                    "group_slot": fields["group_slot"][env_id],
                    "reset_episode": fields["reset_episode"][env_id],
                    "chunks": 0,
                    "frames": 0,
                    "frames_per_chunk": chunk_frames,
                    "expected_frames": expected_frames,
                    "fps": fps,
                    "height": None,
                    "width": None,
                }
                streams[env_id] = state
            expected_identity = (
                state["rollout_uid"],
                state["update_id"],
                state["logical_round_id"],
            )
            actual_identity = (
                fields["rollout_uid"][env_id],
                fields["update_id"][env_id],
                fields["logical_round_id"][env_id],
            )
            if (
                actual_identity != expected_identity
                or chunk_id != state["chunks"]
                or chunk_frames != state["frames_per_chunk"]
            ):
                self._abort_comparison_stream(mode, env_id)
                raise ValueError(
                    "streamed comparison metadata changed within one trajectory: "
                    f"identity={actual_identity}/{expected_identity}, "
                    f"chunk={chunk_id}/{state['chunks']}, "
                    f"frames={chunk_frames}/{state['frames_per_chunk']}"
                )
            try:
                frames = render_comparison_grid_frames(
                    imagined_video_chunk[env_id : env_id + 1],
                    ctrl_world_video_chunk[env_id : env_id + 1],
                    ctrl_world_video_chunk_wrist[env_id : env_id + 1]
                    if ctrl_world_video_chunk_wrist is not None
                    else None,
                    ctrl_world_video_chunk_extra[env_id : env_id + 1]
                    if ctrl_world_video_chunk_extra is not None
                    else None,
                    size=reward_cfg.get(
                        "size", reward_cfg.get("video_similarity_size", None)
                    ),
                    range_policy=reward_cfg.get("range_policy", "error"),
                    frame_offset=int(state["frames"]),
                    total_frames=expected_frames,
                    camera_layout=video_cfg.get("camera_layout", "main_top"),
                    reference_is_composite=reward_cfg.get(
                        "reference_is_composite", False
                    ),
                    reference_label=reward_cfg.get(
                        "reference_label", "Ctrl-World predicted"
                    ),
                )
                if (
                    frames.ndim != 4
                    or frames.shape[-1] != 3
                    or frames.shape[0] != chunk_frames
                ):
                    raise ValueError(
                        "comparison renderer returned invalid shape "
                        f"{frames.shape}; expected {chunk_frames} RGB frames"
                    )
                height, width = int(frames.shape[1]), int(frames.shape[2])
                if state["height"] is None:
                    state["height"], state["width"] = height, width
                elif (height, width) != (state["height"], state["width"]):
                    raise ValueError(
                        "comparison frame size changed within trajectory"
                    )
                for frame in frames:
                    state["writer"].append_data(frame)
                state["frames"] += int(frames.shape[0])
                state["chunks"] += 1
                del frames
            except Exception:
                self._abort_comparison_stream(mode, env_id)
                raise

    def _flush_streamed_cosmos_comparison(self, mode: str = "train") -> None:
        streams = self._comparison_streams.pop(mode, {})
        expected_streams = int(self.train_num_envs_per_stage)
        errors: list[str] = []
        if len(streams) != expected_streams:
            errors.append(
                f"stream count {len(streams)} != expected {expected_streams}"
            )
        for env_id, state in sorted(streams.items()):
            partial_path = Path(state["partial_path"])
            try:
                state["writer"].close()
                if state["chunks"] != self.n_train_chunk_steps:
                    raise ValueError(
                        f"chunks {state['chunks']} != {self.n_train_chunk_steps}"
                    )
                if state["frames"] != state["expected_frames"]:
                    raise ValueError(
                        f"frames {state['frames']} != {state['expected_frames']}"
                    )
                if not partial_path.is_file() or partial_path.stat().st_size <= 0:
                    raise ValueError("encoder produced no MP4 data")
                final_path = Path(state["final_path"])
                os.replace(partial_path, final_path)
                metadata = {
                    "status": "complete",
                    "rollout_uid": state["rollout_uid"],
                    "global_group_id": state["global_group_id"],
                    "group_member_id": state["group_member_id"],
                    "source_env_rank": state["source_env_rank"],
                    "local_env_id": state["local_env_id"],
                    "update_id": state["update_id"],
                    "logical_round_id": state["logical_round_id"],
                    "physical_wave_id": state["physical_wave_id"],
                    "group_slot": state["group_slot"],
                    "reset_episode": state["reset_episode"],
                    "chunks": state["chunks"],
                    "frames": state["frames"],
                    "fps": state["fps"],
                    "height": state["height"],
                    "width": state["width"],
                    "file_size_bytes": final_path.stat().st_size,
                    "path": str(final_path),
                }
                metadata_path = Path(state["metadata_path"])
                metadata_tmp = metadata_path.with_suffix(".json.tmp")
                metadata_tmp.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(metadata_tmp, metadata_path)
            except Exception as exc:
                partial_path.unlink(missing_ok=True)
                errors.append(f"env {env_id}: {exc}")
        if errors:
            raise RuntimeError(
                "streamed comparison video finalization failed: "
                + "; ".join(errors)
            )

    def _save_cosmos_comparison_video(
        self,
        imagined_video_chunk: torch.Tensor,
        ctrl_world_video_chunk: torch.Tensor,
        ctrl_world_video_chunk_wrist: torch.Tensor | None,
        ctrl_world_video_chunk_extra: torch.Tensor | None,
        reward_cfg: Any,
        *,
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Buffer a policy-imagination versus reference-world-model grid.

        For the legacy pipeline the reference is Ctrl-World's three-view
        output; the Cosmos-forward-dynamics pipeline supplies a ready-made
        composite instead. One ``env_interact_step`` only
        covers a single action chunk, so writing a video here directly would
        produce a clip as short as one chunk (e.g. ~1.1s) instead of spanning
        the whole episode like the main rollout video does. Frames accumulate
        across chunk-steps and are only written to disk in
        `_flush_cosmos_comparison_video`, called from `finish_rollout`
        alongside the main video flush.
        """
        try:
            comparison_frames = (
                self.cosmos_comparison_frames
                if mode == "train"
                else self.eval_cosmos_comparison_frames
            )
            total_chunk_steps = (
                self.n_train_chunk_steps if mode == "train" else self.n_eval_chunk_steps
            )
            if self.cfg.env[mode].video_cfg.get("save_all_trajectories", False):
                attr = (
                    "all_cosmos_comparison_frames"
                    if mode == "train"
                    else "eval_all_cosmos_comparison_frames"
                )
                all_buffers = getattr(self, attr)
                batch_size = int(imagined_video_chunk.shape[0])
                if len(all_buffers) != batch_size:
                    all_buffers = [[] for _ in range(batch_size)]
                    setattr(self, attr, all_buffers)
                for env_id in range(batch_size):
                    env_frames = render_comparison_grid_frames(
                        imagined_video_chunk[env_id : env_id + 1],
                        ctrl_world_video_chunk[env_id : env_id + 1],
                        ctrl_world_video_chunk_wrist[env_id : env_id + 1]
                        if ctrl_world_video_chunk_wrist is not None
                        else None,
                        ctrl_world_video_chunk_extra[env_id : env_id + 1]
                        if ctrl_world_video_chunk_extra is not None
                        else None,
                        size=reward_cfg.get(
                            "size", reward_cfg.get("video_similarity_size", None)
                        ),
                        range_policy=reward_cfg.get("range_policy", "error"),
                        frame_offset=len(all_buffers[env_id]),
                        total_frames=total_chunk_steps
                        * int(imagined_video_chunk.shape[1]),
                        camera_layout=self.cfg.env[mode].video_cfg.get(
                            "camera_layout", "main_top"
                        ),
                        reference_is_composite=reward_cfg.get("reference_is_composite", False),
                        reference_label=reward_cfg.get("reference_label", "Ctrl-World predicted"),
                    )
                    all_buffers[env_id].extend(list(env_frames))
                return
            frames = render_comparison_grid_frames(
                imagined_video_chunk,
                ctrl_world_video_chunk,
                ctrl_world_video_chunk_wrist,
                ctrl_world_video_chunk_extra,
                size=reward_cfg.get("size", reward_cfg.get("video_similarity_size", None)),
                range_policy=reward_cfg.get("range_policy", "error"),
                # Keep the exact 10D ordering convention: the renderer
                # resamples Ctrl-World's raw timeline to the Cosmos plan's
                # frame count, then appends chunks in rollout order.
                frame_offset=len(comparison_frames),
                total_frames=total_chunk_steps * int(imagined_video_chunk.shape[1]),
                camera_layout=self.cfg.env[mode].video_cfg.get(
                    "camera_layout", "main_top"
                ),
                reference_is_composite=reward_cfg.get("reference_is_composite", False),
                reference_label=reward_cfg.get("reference_label", "Ctrl-World predicted"),
            )
            comparison_frames.extend(list(frames))
        except Exception as exc:
            self.log_on_first_rank(
                f"Failed to render Cosmos-vs-Ctrl-World comparison frames: {exc}"
            )

    @staticmethod
    def _video_timesteps(video: torch.Tensor) -> int:
        """Return T for either [B,T,H,W,C] or [B,C,T,H,W]."""
        if video.ndim != 5:
            raise ValueError(f"Expected a 5D video tensor, got {video.shape}.")
        if video.shape[-1] in (1, 3):
            return int(video.shape[1])
        if video.shape[1] in (1, 3):
            return int(video.shape[2])
        raise ValueError(f"Cannot infer channel dimension for {video.shape}.")


    def _capture_training_final_side_frames(
        self,
        *,
        final_side_video: torch.Tensor | None,
        done_mask: torch.Tensor | None,
        rollout_metadata: dict[str, torch.Tensor] | None,
    ) -> None:
        """Capture the exact classifier-facing final d405_1 frame per rollout.

        Frames remain uint8 tensors until the runner atomically packs the full
        cross-rank step into one NPZ. This avoids one filesystem inode per
        rollout while preserving byte-exact inputs for future classifiers.
        """
        if not self._save_rollout_final_side_frames():
            return
        if final_side_video is None or done_mask is None or rollout_metadata is None:
            raise ValueError(
                "final-side-frame recording requires video, done mask, and rollout metadata"
            )
        frames = torch.as_tensor(final_side_video).detach().cpu()
        if frames.ndim != 5:
            raise ValueError(
                f"final side video must be 5D, got {tuple(frames.shape)}"
            )
        if frames.shape[-1] == 3:
            frames = frames[:, -1]
        elif frames.shape[1] == 3:
            frames = frames[:, :, -1].permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"final side video has no RGB channel axis: {tuple(frames.shape)}"
            )
        if frames.dtype != torch.uint8:
            raise ValueError(
                "classifier-facing final side frames must remain exact uint8 data"
            )
        batch_size = int(frames.shape[0])
        done_rows = (
            torch.as_tensor(done_mask)
            .detach()
            .cpu()
            .reshape(batch_size, -1)[:, -1]
            .bool()
        )
        if not done_rows.any():
            return

        def metadata_rows(name: str) -> torch.Tensor:
            value = rollout_metadata.get(name)
            if value is None:
                raise KeyError(f"final-side-frame metadata is missing {name}")
            rows = torch.as_tensor(value).detach().cpu().reshape(-1)
            if rows.numel() != batch_size:
                raise ValueError(
                    f"final-side-frame {name} count {rows.numel()} != {batch_size}"
                )
            return rows

        rollout_uids = metadata_rows("rollout_uid")
        episodes = metadata_rows("reset_episode")
        if self.best_rollout_final_side_frames is None:
            self.best_rollout_final_side_frames = torch.empty_like(frames)
        elif self.best_rollout_final_side_frames.shape != frames.shape:
            raise ValueError(
                "final-side-frame tensor shape changed within one rollout: "
                f"{tuple(self.best_rollout_final_side_frames.shape)} != "
                f"{tuple(frames.shape)}"
            )
        for env_id in torch.nonzero(done_rows, as_tuple=False).reshape(-1).tolist():
            episode = int(episodes[env_id].item())
            # Reading the uid here is a fail-closed metadata cardinality check.
            int(rollout_uids[env_id].item())
            frame = frames[env_id].contiguous()
            self.best_rollout_final_side_frames[env_id].copy_(frame)
            digest = hashlib.sha256(frame.numpy().tobytes(order="C")).hexdigest()
            self.best_rollout_final_side_frame_sha256[env_id] = digest
            self.best_rollout_final_side_frame_episodes[env_id] = episode

    def _buffer_best_rollout_comparison_video(
        self,
        *,
        imagined_video_chunk: torch.Tensor,
        ctrl_world_video_chunk: torch.Tensor,
        ctrl_world_video_chunk_wrist: torch.Tensor | None,
        ctrl_world_video_chunk_extra: torch.Tensor | None,
        rewards: torch.Tensor,
        chunk_mse: torch.Tensor | None,
        successes: torch.Tensor | None,
        done_mask: torch.Tensor | None,
        reward_cfg: Any,
        final_side_video: torch.Tensor | None = None,
        rollout_metadata: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Accumulate per-env frames and episode rewards for local selection."""
        batch_size = int(rewards.shape[0])
        chunk_rewards = (
            rewards.detach()
            .to(dtype=torch.float32)
            .reshape(batch_size, -1)
            .sum(dim=1)
            .cpu()
        )
        if self.best_rollout_episode_rewards is None:
            self.best_rollout_episode_rewards = torch.zeros_like(chunk_rewards)
            self.best_rollout_comparison_frames = [[] for _ in range(batch_size)]
            self.best_rollout_chunk_rewards = [[] for _ in range(batch_size)]
            self.best_rollout_chunk_mses = [[] for _ in range(batch_size)]
            self.best_rollout_successes = torch.full((batch_size,), -1, dtype=torch.int8)
            self.best_rollout_final_side_frames = None
            self.best_rollout_final_side_frame_sha256 = [None] * batch_size
            self.best_rollout_final_side_frame_episodes = [None] * batch_size
        if self.best_rollout_episode_rewards.shape != chunk_rewards.shape:
            raise ValueError(
                "Best-rollout reward batch changed within one rollout: "
                f"{self.best_rollout_episode_rewards.shape} != {chunk_rewards.shape}."
            )
        self.best_rollout_episode_rewards += chunk_rewards
        for env_id in range(batch_size):
            self.best_rollout_chunk_rewards[env_id].append(
                float(chunk_rewards[env_id].item())
            )

        if chunk_mse is not None:
            mse_rows = torch.as_tensor(chunk_mse).detach().to(dtype=torch.float32).cpu().reshape(-1)
            if mse_rows.numel() != batch_size:
                raise ValueError(f"Rollout chunk MSE count {mse_rows.numel()} != {batch_size}.")
            for env_id in range(batch_size):
                self.best_rollout_chunk_mses[env_id].append(float(mse_rows[env_id].item()))
        elif (
            self._save_success_failure_mse_extremes()
            or self._save_mse_extreme_final_frames()
        ):
            raise ValueError("MSE-extreme selection requires per-env chunk MSE.")

        if successes is not None:
            if done_mask is None or self.best_rollout_successes is None:
                raise ValueError("Success values require a rollout done mask.")
            done_rows = (
                torch.as_tensor(done_mask)
                .detach()
                .cpu()
                .reshape(batch_size, -1)[:, -1]
                .bool()
            )
            success_rows = torch.as_tensor(successes).detach().cpu().reshape(-1).bool()
            if success_rows.numel() != int(done_rows.sum().item()):
                raise ValueError(
                    "Final success count does not match completed trajectories: "
                    f"{success_rows.numel()} != {int(done_rows.sum().item())}."
                )
            self.best_rollout_successes[done_rows] = success_rows.to(torch.int8)

        self._capture_training_final_side_frames(
            final_side_video=final_side_video,
            done_mask=done_mask,
            rollout_metadata=rollout_metadata,
        )
        if not self._track_rollout_candidate_videos():
            return

        total_frames = self.n_train_chunk_steps * self._video_timesteps(
            imagined_video_chunk
        )
        for env_id in range(batch_size):
            try:
                frames = render_comparison_grid_frames(
                    imagined_video_chunk[env_id : env_id + 1],
                    ctrl_world_video_chunk[env_id : env_id + 1],
                    ctrl_world_video_chunk_wrist[env_id : env_id + 1]
                    if ctrl_world_video_chunk_wrist is not None
                    else None,
                    ctrl_world_video_chunk_extra[env_id : env_id + 1]
                    if ctrl_world_video_chunk_extra is not None
                    else None,
                    size=reward_cfg.get(
                        "size", reward_cfg.get("video_similarity_size", None)
                    ),
                    range_policy=reward_cfg.get("range_policy", "error"),
                    frame_offset=len(self.best_rollout_comparison_frames[env_id]),
                    total_frames=total_frames,
                    camera_layout=reward_cfg.get(
                        "camera_layout",
                        self.cfg.actor.model.get("cosmos", {}).get(
                            "camera_layout", "main_top"
                        ),
                    ),
                    reference_is_composite=reward_cfg.get("reference_is_composite", False),
                    reference_label=reward_cfg.get("reference_label", "Ctrl-World predicted"),
                )
                self.best_rollout_comparison_frames[env_id].extend(list(frames))
            except Exception as exc:
                self.log_on_first_rank(
                    "Failed to render best-rollout candidate frames for "
                    f"env {env_id}: {exc}"
                )

    def _write_rollout_candidate_video(
        self,
        *,
        env_id: int,
        base_dir_key: str,
        default_subdir: str,
        candidate_label: str | None = None,
        rollout_uid: int | None = None,
    ) -> None:
        """Write one env's already-buffered frames out as this rank's candidate."""
        if (
            env_id >= len(self.best_rollout_comparison_frames)
            or not self.best_rollout_comparison_frames[env_id]
        ):
            self.log_on_first_rank(
                f"No rendered frames for local {default_subdir} env "
                f"{env_id} on rank {self._rank}."
            )
            return

        video_cfg = self.cfg.env.train.video_cfg
        base_dir = str(
            video_cfg.get(
                base_dir_key,
                os.path.join(str(video_cfg.video_base_dir), default_subdir),
            )
        )
        candidate_dir = os.path.join(
            f"{base_dir}_candidates",
            f"step_{self.global_step + 1:06d}",
        )
        os.makedirs(candidate_dir, exist_ok=True)
        label_prefix = f"{candidate_label}_" if candidate_label else ""
        candidate_name = f"{label_prefix}rank_{self._rank}.mp4"
        if rollout_uid is not None:
            # One update can contain several rollout epochs. A category/rank name
            # lets a later epoch overwrite an earlier candidate before the
            # runner selects the global winner.
            candidate_name = f"{label_prefix}rank_{self._rank}_uid_{int(rollout_uid)}.mp4"
        candidate_path = os.path.join(candidate_dir, candidate_name)
        fps = int(video_cfg.get("fps", None) or 7)
        video_writer = imageio.get_writer(candidate_path, fps=fps)
        try:
            for frame in self.best_rollout_comparison_frames[env_id]:
                video_writer.append_data(frame)
        finally:
            video_writer.close()

    def _write_local_mse_extreme_candidates(
        self,
        *,
        trajectory_mses: torch.Tensor,
        current_rollout_uids: torch.Tensor | None,
    ) -> None:
        """Write this rank's candidate for each populated success/MSE category."""
        if self.best_rollout_successes is None:
            raise ValueError("MSE-extreme candidates require success judgments.")
        specs = (
            ("success_mse_min", True, True),
            ("success_mse_max", True, False),
            ("failure_mse_min", False, True),
            ("failure_mse_max", False, False),
        )
        for category, success_value, use_minimum in specs:
            eligible = torch.nonzero(
                self.best_rollout_successes == int(success_value),
                as_tuple=False,
            ).reshape(-1)
            if eligible.numel() == 0:
                continue
            eligible_mses = trajectory_mses[eligible]
            local_offset = (
                torch.argmin(eligible_mses)
                if use_minimum
                else torch.argmax(eligible_mses)
            )
            env_id = int(eligible[int(local_offset.item())].item())
            self._write_rollout_candidate_video(
                env_id=env_id,
                base_dir_key="mse_extremes_base_dir",
                default_subdir="mse_extremes",
                candidate_label=category,
                rollout_uid=(
                    int(current_rollout_uids[env_id].item())
                    if current_rollout_uids is not None
                    else None
                ),
            )

    def _flush_best_rollout_video_candidate(self) -> dict[str, torch.Tensor]:
        """Write this rank's best/worst candidates and return local episode scores."""
        if self.best_rollout_episode_rewards is None:
            return {}

        episode_rewards = self.best_rollout_episode_rewards.clone()
        summary = {"_best_rollout_episode_rewards": episode_rewards}
        chunk_reward_max = torch.tensor(
            [
                max(rewards) if rewards else float("nan")
                for rewards in self.best_rollout_chunk_rewards
            ],
            dtype=torch.float32,
        )
        chunk_reward_mean = torch.tensor(
            [
                sum(rewards) / len(rewards) if rewards else float("nan")
                for rewards in self.best_rollout_chunk_rewards
            ],
            dtype=torch.float32,
        )
        summary["_best_rollout_chunk_reward_max"] = chunk_reward_max
        summary["_best_rollout_chunk_reward_mean"] = chunk_reward_mean
        trajectory_mses = None
        if (
            self._save_success_failure_mse_extremes()
            or self._save_mse_extreme_final_frames()
        ):
            if any(not values for values in self.best_rollout_chunk_mses):
                raise ValueError("Every trajectory must have chunk MSE values.")
            trajectory_mses = torch.tensor(
                [sum(values) / len(values) for values in self.best_rollout_chunk_mses],
                dtype=torch.float32,
            )
            if self.best_rollout_successes is None or torch.any(self.best_rollout_successes < 0):
                raise ValueError("Every trajectory must have a final success judgment.")
            summary["_best_rollout_trajectory_mse"] = trajectory_mses
            summary["_best_rollout_success"] = self.best_rollout_successes.bool().clone()
        if self._save_rollout_final_side_frames():
            if (
                self.best_rollout_final_side_frames is None
                or any(
                    digest is None
                    for digest in self.best_rollout_final_side_frame_sha256
                )
                or any(
                    episode is None
                    for episode in self.best_rollout_final_side_frame_episodes
                )
            ):
                raise ValueError(
                    "Every completed trajectory must have one final side frame"
                )
            if int(self.best_rollout_final_side_frames.shape[0]) != int(
                episode_rewards.numel()
            ):
                raise ValueError(
                    "Final-side-frame batch does not match rollout summary"
                )
            summary["_best_rollout_final_side_frame"] = (
                self.best_rollout_final_side_frames.clone()
            )
            summary["_best_rollout_final_side_frame_sha256"] = (
                _encode_sha256_hex_batch(
                    self.best_rollout_final_side_frame_sha256
                )
            )
            summary["_best_rollout_final_side_frame_episode"] = torch.tensor(
                [
                    int(episode)
                    for episode in self.best_rollout_final_side_frame_episodes
                ],
                dtype=torch.int64,
            )
        current_rollout_uids = None
        if self._cross_rank_group_cfg() is not None and self.rollout_results:
            rollout_result = self.rollout_results[0]
            for field_name in (
                "rollout_uid",
                "global_group_id",
                "group_member_id",
                "reset_episode",
            ):
                values = getattr(rollout_result, field_name)
                if not values:
                    raise ValueError(
                        f"best-rollout summary is missing {field_name}"
                    )
                # The container accumulates all rollout epochs, but this
                # summary is flushed once per epoch. Use the latest chunk's
                # trajectory identity instead of repeating epoch 0 metadata.
                current_values = values[-1].reshape(-1)
                if current_values.numel() != episode_rewards.numel():
                    raise ValueError(
                        f"best-rollout {field_name} count "
                        f"{current_values.numel()} != episode reward count "
                        f"{episode_rewards.numel()}"
                    )
                summary[f"_best_rollout_{field_name}"] = current_values
                if field_name == "rollout_uid":
                    current_rollout_uids = current_values
        try:
            if (
                self._save_success_failure_mse_extremes()
                and not self._save_global_mse_extremes_only()
            ):
                if trajectory_mses is None:
                    raise ValueError("MSE-extreme summary was not constructed.")
                self._write_local_mse_extreme_candidates(
                    trajectory_mses=trajectory_mses,
                    current_rollout_uids=current_rollout_uids,
                )
            if self._save_best_rollout_video():
                best_env = int(torch.argmax(episode_rewards).item())
                self._write_rollout_candidate_video(
                    env_id=best_env,
                    base_dir_key="best_rollout_base_dir",
                    default_subdir="best_rollout",
                    rollout_uid=(
                        int(current_rollout_uids[best_env].item())
                        if current_rollout_uids is not None
                        else None
                    ),
                )
            if self._save_worst_rollout_video():
                worst_env = int(torch.argmin(episode_rewards).item())
                self._write_rollout_candidate_video(
                    env_id=worst_env,
                    base_dir_key="worst_rollout_base_dir",
                    default_subdir="worst_rollout",
                    rollout_uid=(
                        int(current_rollout_uids[worst_env].item())
                        if current_rollout_uids is not None
                        else None
                    ),
                )
        except Exception as exc:
            self.log_on_first_rank(
                f"Failed to save local rollout candidate video: {exc}"
            )
        finally:
            if not self._save_global_mse_extremes_only():
                self._reset_best_rollout_video_buffer()
        return summary

    def write_selected_mse_extreme_candidates(
        self, selections: dict[str, dict[str, int | float | bool]]
    ) -> dict[str, int]:
        """Encode only this rank's globally selected MSE-extreme trajectories."""
        allowed_categories = {
            "success_mse_min",
            "success_mse_max",
            "failure_mse_min",
            "failure_mse_max",
        }
        written = 0
        try:
            if not self._save_global_mse_extremes_only():
                return {"mse_extreme_candidate_videos_written": 0}
            if self.best_rollout_episode_rewards is None:
                raise RuntimeError("Global MSE-extreme selection has no buffered rollout.")
            for category, winner in selections.items():
                if category not in allowed_categories:
                    raise ValueError(f"Unknown MSE-extreme category: {category}")
                if int(winner["rank"]) != self._rank:
                    continue
                env_id = int(winner["env_id"])
                if not 0 <= env_id < len(self.best_rollout_comparison_frames):
                    raise ValueError(
                        f"Selected env {env_id} is outside rank {self._rank}'s buffer."
                    )
                rollout_uid = winner.get("rollout_uid")
                self._write_rollout_candidate_video(
                    env_id=env_id,
                    base_dir_key="mse_extremes_base_dir",
                    default_subdir="mse_extremes",
                    candidate_label=category,
                    rollout_uid=(
                        int(rollout_uid) if rollout_uid is not None else None
                    ),
                )
                written += 1
        finally:
            self._reset_best_rollout_video_buffer()
        return {"mse_extreme_candidate_videos_written": written}

    def _flush_cosmos_comparison_video(
        self, *, mode: Literal["train", "eval"] = "train"
    ) -> None:
        """Write the current rollout epoch's buffered comparison frames to a
        single mp4 and reset the buffer for the next epoch."""
        comparison_frames = (
            self.cosmos_comparison_frames
            if mode == "train"
            else self.eval_cosmos_comparison_frames
        )
        if self.cfg.env[mode].video_cfg.get("save_all_trajectories", False):
            attr = (
                "all_cosmos_comparison_frames"
                if mode == "train"
                else "eval_all_cosmos_comparison_frames"
            )
            all_buffers = getattr(self, attr)
            if not any(all_buffers):
                return
            try:
                output_dir = os.path.join(
                    self.cfg.env[mode].video_cfg.video_base_dir,
                    self.cfg.env[mode].video_cfg.get("comparison_subdir", "cosmos_vs_ctrlworld"),
                    f"rank_{self._rank}",
                )
                fps = int(self.cfg.env[mode].video_cfg.get("fps", None) or 7)
                video_cnt = (
                    self.cosmos_comparison_video_cnt
                    if mode == "train"
                    else self.eval_cosmos_comparison_video_cnt
                )
                for env_id, env_frames in enumerate(all_buffers):
                    if not env_frames:
                        continue
                    env_dir = os.path.join(output_dir, f"env_{env_id}")
                    os.makedirs(env_dir, exist_ok=True)
                    mp4_path = os.path.join(env_dir, f"{video_cnt}.mp4")
                    video_writer = imageio.get_writer(mp4_path, fps=fps)
                    try:
                        for frame in env_frames:
                            video_writer.append_data(frame)
                    finally:
                        video_writer.close()
                    if (
                        mode == "eval"
                        and env_id < len(self._post_update_eval_records)
                    ):
                        self._post_update_eval_records[env_id][
                            "comparison_video_path"
                        ] = str(Path(mp4_path).resolve())
                if mode == "train":
                    self.cosmos_comparison_video_cnt += 1
                else:
                    self.eval_cosmos_comparison_video_cnt += 1
            except Exception as exc:
                self.log_on_first_rank(
                    f"Failed to save per-trajectory Cosmos comparison videos: {exc}"
                )
            finally:
                setattr(self, attr, [])
            return

        if not comparison_frames:
            return
        try:
            # Every EnvWorker rank runs this same code independently in its
            # own OS process; without a per-rank path component, all ranks
            # race to open/write/close the SAME mp4 file concurrently, which
            # corrupts the container intermittently (works sometimes, fails
            # at frame 0 or at a chunk boundary other times, depending on
            # which rank's writer "wins"). Mirror RecordVideo's per-env
            # `seed_{...}` subdirectory convention with `rank_{self._rank}`.
            output_dir = os.path.join(
                self.cfg.env[mode].video_cfg.video_base_dir,
                self.cfg.env[mode].video_cfg.get("comparison_subdir", "cosmos_vs_ctrlworld"),
                f"rank_{self._rank}",
            )
            os.makedirs(output_dir, exist_ok=True)
            fps = int(self.cfg.env[mode].video_cfg.get("fps", None) or 7)
            video_cnt = (
                self.cosmos_comparison_video_cnt
                if mode == "train"
                else self.eval_cosmos_comparison_video_cnt
            )
            mp4_path = os.path.join(output_dir, f"{video_cnt}.mp4")
            # Use the same incremental get_writer()/append_data() pattern as
            # RecordVideo._save_video (rlinf/envs/wrappers/record_video.py) —
            # imageio.mimwrite() on the full ~40-frame list produced a
            # container that decoded fine up to the first chunk boundary (8
            # frames) then failed with "Invalid data found when processing
            # input" on every frame after, while the per-frame writer loop
            # verified to decode cleanly end to end.
            video_writer = imageio.get_writer(mp4_path, fps=fps)
            try:
                for frame in comparison_frames:
                    video_writer.append_data(frame)
            finally:
                video_writer.close()
            if mode == "train":
                self.cosmos_comparison_video_cnt += 1
            else:
                self.eval_cosmos_comparison_video_cnt += 1
        except Exception as exc:
            self.log_on_first_rank(
                f"Failed to save Cosmos-vs-Ctrl-World comparison video: {exc}"
            )
        finally:
            if mode == "train":
                self.cosmos_comparison_frames = []
            else:
                self.eval_cosmos_comparison_frames = []

    def _record_view_reward_diagnostics(self, stats, env_output, env_info, metadata):
        """Small per-chunk sidecar permits rescoring either arm with both weights."""
        record_cfg = self.cfg.algorithm.get("trajectory_records", {})
        if not bool(record_cfg.get("enabled", False)):
            return
        from rlinf.utils.diagnostic_io import make_view_reward_spool

        spool = getattr(self, "_view_reward_diagnostic_spool", None)
        if spool is None:
            spool = make_view_reward_spool(
                Path(str(record_cfg.output_dir)).parent, self._rank
            )
            self._view_reward_diagnostic_spool = spool
        views = (("main", "main"), ("wrist", "wrist"), ("side", "extra"))
        done = torch.as_tensor(env_output.dones)
        if done.dim() > 1:
            done = done[..., -1]
        record = {
            "global_step": int(self.global_step),
            "rank": int(self._rank),
            "metadata": {
                name: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                for name, value in (metadata or {}).items()
            },
            "episode_ids": torch.as_tensor(env_output.reset_state_ids).reshape(-1).cpu().tolist(),
            "done": done.reshape(-1).cpu().tolist(),
            "trajectory_view_mse": {
                physical: stats[f"{key}_per_frame_mse"].mean(dim=1).detach().cpu().tolist()
                for physical, key in views
            },
            "trajectory_weighted_mse": stats["per_frame_mse"].mean(dim=1).detach().cpu().tolist(),
            "trajectory_pixel_mse": stats["pixel_per_frame_mse"].mean(dim=1).detach().cpu().tolist(),
            "terminal_view_mse_done": {
                physical: env_info[f"reward/terminal_goal_{key}_mse"].tolist()
                for physical, key in views if f"reward/terminal_goal_{key}_mse" in env_info
            },
        }
        persisted = spool.append(record, publish=bool(done.any().item()))
        env_info["diagnostics/view_records_io_pending"] = torch.tensor(
            [0.0 if persisted else 1.0], dtype=torch.float32
        )

    def _record_post_update_eval_chunk(
        self,
        *,
        probabilities: torch.Tensor | None,
        video_mse_per_env: torch.Tensor | None,
        combined_rewards: torch.Tensor,
        done_mask: torch.Tensor,
        env_info: dict[str, Any],
        final_side_video: torch.Tensor | None,
    ) -> None:
        """Accumulate raw post-update diagnostics into one record per episode."""
        context = self._post_update_eval_context
        if context is None or not context.get("active", False):
            return
        records = self._post_update_eval_records
        local_count = len(records)
        if local_count != int(combined_rewards.shape[0]):
            raise ValueError(
                "post-update record count does not match eval batch: "
                f"{local_count} != {combined_rewards.shape[0]}"
            )
        self._sync_post_update_eval_rng_manifest()
        probability_rows = (
            torch.as_tensor(probabilities).detach().cpu().to(torch.float32)
            if probabilities is not None
            else None
        )
        mse_rows = (
            torch.as_tensor(video_mse_per_env)
            .detach()
            .cpu()
            .to(torch.float32)
            .reshape(-1)
            if video_mse_per_env is not None
            else None
        )
        reward_rows = (
            torch.as_tensor(combined_rewards)
            .detach()
            .cpu()
            .to(torch.float32)
            .reshape(local_count, -1)
        )
        done_rows = torch.as_tensor(done_mask).detach().cpu().bool().reshape(-1)
        final_side_frames = None
        if self._save_rollout_final_side_frames():
            if final_side_video is None:
                raise RuntimeError(
                    "post-update evaluation requires raw d405_1 side frames"
                )
            final_side_frames = torch.as_tensor(final_side_video).detach().cpu()
            if final_side_frames.ndim != 5:
                raise ValueError(
                    "post-update final side video must be 5D, got "
                    f"{tuple(final_side_frames.shape)}"
                )
            if final_side_frames.shape[-1] == 3:
                final_side_frames = final_side_frames[:, -1]
            elif final_side_frames.shape[1] == 3:
                final_side_frames = (
                    final_side_frames[:, :, -1].permute(0, 2, 3, 1)
                )
            else:
                raise ValueError(
                    "post-update final side video has no RGB channel axis: "
                    f"{tuple(final_side_frames.shape)}"
                )
            if final_side_frames.dtype != torch.uint8:
                raise ValueError(
                    "post-update final side frames must remain exact uint8 data"
                )
            if int(final_side_frames.shape[0]) != local_count:
                raise ValueError(
                    "post-update final side frame count does not match eval batch"
                )
        terminal_mse = env_info.get("reward/terminal_goal_mse")
        reference_episode = env_info.get("reward/terminal_goal_reference_episode")
        reference_frame = env_info.get("reward/terminal_goal_reference_frame")
        done_indices = torch.nonzero(done_rows, as_tuple=False).reshape(-1).tolist()
        terminal_by_env = {}
        for offset, env_id in enumerate(done_indices):
            terminal_by_env[env_id] = {
                "mse": float(torch.as_tensor(terminal_mse).reshape(-1)[offset].item())
                if terminal_mse is not None
                else None,
                "episode": int(
                    torch.as_tensor(reference_episode).reshape(-1)[offset].item()
                )
                if reference_episode is not None
                else None,
                "frame": int(
                    torch.as_tensor(reference_frame).reshape(-1)[offset].item()
                )
                if reference_frame is not None
                else None,
            }
        success_cfg = self.cfg.reward.get("success_classifier", {})
        for env_id, record in enumerate(records):
            if probability_rows is not None:
                record["frame_probabilities"].extend(
                    float(value)
                    for value in probability_rows[env_id].reshape(-1).tolist()
                )
            if mse_rows is not None:
                record["video_mse_per_chunk"].append(float(mse_rows[env_id].item()))
            view_chunks = record.setdefault("trajectory_view_mse_per_chunk", {})
            for physical, key in (("main", "main"), ("wrist", "wrist"), ("side", "extra")):
                values = env_info.get(f"cosmos/video_{key}_mse_per_env")
                if values is not None:
                    view_chunks.setdefault(physical, []).append(float(values[env_id].item()))
            record["combined_reward"] += float(reward_rows[env_id].sum().item())
            record["_rng_completed_segments"] += 1
            if not bool(done_rows[env_id].item()):
                continue
            probability_tensor = torch.tensor(
                record["frame_probabilities"], dtype=torch.float32
            ).reshape(1, -1)
            decision = classify_terminal_probabilities(
                probability_tensor, success_cfg
            )
            sampled = [
                float(value)
                for value in decision["sampled_probabilities"][0].tolist()
            ]
            success = bool(decision["success"][0].item())
            record["success_last4_probabilities"] = sampled[-4:]
            record["success_terminal_probabilities"] = sampled
            record["success_terminal_sample_indices"] = [
                int(value) for value in decision["sample_indices"].tolist()
            ]
            record["success_probability_max"] = float(
                decision["probability_max"][0].item()
            )
            record["success_last_probability"] = float(
                decision["last_probability"][0].item()
            )
            record["success_positive_ratio"] = float(
                decision["positive_ratio"][0].item()
            )
            record["model_success_probability"] = float(
                decision["reported_probability"][0].item()
            )
            record["success"] = success
            record["model_success"] = success
            record["success_threshold"] = float(decision["threshold"].item())
            record["success_decision_rule"] = decision["rule"]
            record["video_mse"] = (
                float(np.mean(record["video_mse_per_chunk"]))
                if record["video_mse_per_chunk"]
                else None
            )
            record["trajectory_mse"] = record["video_mse"]
            record["video_similarity_mse"] = record["video_mse"]
            terminal = terminal_by_env.get(env_id, {})
            record["terminal_goal_mse"] = terminal.get("mse")
            record["lastframe_mse"] = terminal.get("mse")
            record["terminal_goal_reference_episode"] = terminal.get("episode")
            record["terminal_goal_reference_frame"] = terminal.get("frame")
            record["trajectory_view_mse"] = {
                name: float(np.mean(values)) for name, values in view_chunks.items()
                if values
            }
            done_offset = done_indices.index(env_id)
            record["terminal_view_mse"] = {
                physical: float(env_info[f"reward/terminal_goal_{key}_mse"][done_offset].item())
                for physical, key in (("main", "main"), ("wrist", "wrist"), ("side", "extra"))
                if f"reward/terminal_goal_{key}_mse" in env_info
            }
            if final_side_frames is not None:
                frame = np.ascontiguousarray(
                    final_side_frames[env_id].contiguous().numpy()
                )
                record["_final_side_frame"] = frame
                record["final_side_frame_sha256"] = hashlib.sha256(
                    frame.tobytes(order="C")
                ).hexdigest()
            record["complete"] = True

    def env_evaluate_step(
        self,
        raw_actions: torch.Tensor,
        stage_id: int,
        imagined_video_chunk: torch.Tensor | None = None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        diagnostics_cfg = self.cfg.algorithm.get("diagnostics", {})
        action_override = str(
            diagnostics_cfg.get("eval_action_override", "policy")
        ).lower()
        if action_override == "policy":
            capture_path = diagnostics_cfg.get("eval_capture_action_path", None)
            if capture_path:
                # Reward noise-floor diagnostics need a frozen (action, video)
                # pair to replay across many seeds. Cosmos's action and its
                # imagined_video_chunk come from the SAME joint sample (see
                # huggingface_worker.py::predict()) -- capturing only the
                # action and later feeding it to Ctrl-World while Cosmos is
                # left to independently resample its own video every replica
                # breaks that pairing and invalidates any similarity-based
                # comparison. Capture both here, as a side effect of a normal
                # policy eval step, instead of adding a separate capture mode.
                capture = {
                    "action": torch.as_tensor(raw_actions[0:1]).clone(),
                    "reset_id": self.cfg.env.eval.get("specific_reset_id", None),
                    "seed": self.cfg.env.eval.get("seed", None),
                }
                if isinstance(imagined_video_chunk, torch.Tensor):
                    capture["imagined_video_chunk"] = (
                        imagined_video_chunk[0:1].detach().cpu().clone()
                    )
                torch.save(capture, capture_path)
        elif action_override == "zero":
            raw_actions = np.zeros_like(raw_actions)
        elif action_override == "random":
            raw_actions = np.random.uniform(-0.25, 0.25, size=raw_actions.shape).astype(
                raw_actions.dtype
            )
        elif action_override in ("fixed", "reversed"):
            fixed_path = diagnostics_cfg.get("eval_fixed_action_path", None)
            if not fixed_path:
                raise ValueError(
                    f"algorithm.diagnostics.eval_action_override={action_override!r} requires "
                    "algorithm.diagnostics.eval_fixed_action_path to be set."
                )
            frozen_capture = torch.load(fixed_path, map_location="cpu")
            fixed_action = frozen_capture["action"].numpy()
            if fixed_action.shape[1:] != raw_actions.shape[1:]:
                raise ValueError(
                    f"Captured action shape {fixed_action.shape} does not match "
                    f"live action shape {raw_actions.shape} (chunk/action_dim mismatch)."
                )
            if action_override == "reversed":
                # Magnitude-matched "bad" action for Exp 6: exact same L2 norm
                # per dimension as the real captured policy action, just
                # negated -- unlike independent uniform sampling (the old
                # "random" mode), this can't be confounded by action
                # magnitude (see plan doc's "验证结果" section).
                fixed_action = -fixed_action
            raw_actions = np.broadcast_to(fixed_action, raw_actions.shape).astype(
                raw_actions.dtype
            ).copy()

            frozen_video = frozen_capture.get("imagined_video_chunk", None)
            if isinstance(frozen_video, torch.Tensor):
                # Cosmos's own video generation is frozen during RL and is not
                # conditioned on the sampled action's value (video and action
                # are two outputs of one joint sample, not action-then-video) --
                # so replaying a fixed/reversed action for Ctrl-World while
                # letting Cosmos independently resample a new video every
                # replica would test "Ctrl-World's video vs an unrelated,
                # freely-varying Cosmos reference," not "do the two models
                # agree on the same action." Pin Cosmos's side to the exact
                # video captured alongside this action so only Ctrl-World's
                # input actually varies across fixed/reversed/replicas.
                if not isinstance(imagined_video_chunk, torch.Tensor):
                    raise RuntimeError(
                        f"algorithm.diagnostics.eval_action_override={action_override!r} "
                        "found a captured imagined_video_chunk in "
                        f"{fixed_path!r}, but no live imagined_video_chunk was received "
                        "from the rollout worker to replace -- video similarity/"
                        "comparison must be enabled to use frozen-video replay."
                    )
                if frozen_video.shape[1:] != imagined_video_chunk.shape[1:]:
                    raise ValueError(
                        f"Captured imagined_video_chunk shape {tuple(frozen_video.shape)} "
                        f"does not match live shape {tuple(imagined_video_chunk.shape)}."
                    )
                batch_size = imagined_video_chunk.shape[0]
                imagined_video_chunk = frozen_video.expand(
                    batch_size, *frozen_video.shape[1:]
                ).to(imagined_video_chunk.device, dtype=imagined_video_chunk.dtype)
        else:
            raise ValueError(
                "algorithm.diagnostics.eval_action_override must be one of "
                f"policy, zero, random, fixed, reversed; got {action_override!r}."
            )
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}
        eval_video_rewards: torch.Tensor | None = None
        eval_video_mse_per_env: torch.Tensor | None = None

        if self._is_cosmos_self_feedback("eval"):
            self._set_policy_feedback(
                self.eval_env_list[stage_id], imagined_video_chunk, mode="eval"
            )
        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) and infos_list else {}
        if infos is None:
            infos = {}
        for key, value in infos.items():
            if key.startswith(("self_feedback/", "reward/success_model/", "episode/")):
                env_info[key] = self._to_cpu_if_possible(value)

        if self._needs_eval_imagined_video() and not isinstance(imagined_video_chunk, torch.Tensor):
            raise RuntimeError(
                "Eval video similarity/comparison was enabled but no imagined Cosmos video "
                "was received from the rollout worker."
            )

        if self._save_eval_cosmos_comparison():
            if not isinstance(imagined_video_chunk, torch.Tensor):
                raise RuntimeError(
                    "Eval comparison video was enabled but no imagined Cosmos video "
                    "was received from the rollout worker."
                )
            world_model_video_chunk = infos.get("world_model_video_chunk")
            if isinstance(world_model_video_chunk, torch.Tensor):
                ctrl_world_video_chunk = world_model_video_chunk
                ctrl_world_video_chunk_wrist = infos.get("world_model_video_chunk_wrist")
                ctrl_world_video_chunk_extra = infos.get("world_model_video_chunk_extra")
            else:
                ctrl_world_video_chunk = infos.get("ctrl_world_video_chunk_raw")
                ctrl_world_video_chunk_wrist = infos.get(
                    "ctrl_world_video_chunk_raw_wrist"
                )
                ctrl_world_video_chunk_extra = infos.get(
                    "ctrl_world_video_chunk_raw_extra"
                )
                if ctrl_world_video_chunk is None:
                    ctrl_world_video_chunk = infos.get("ctrl_world_video_chunk")
                    ctrl_world_video_chunk_wrist = infos.get("ctrl_world_video_chunk_wrist")
                    ctrl_world_video_chunk_extra = infos.get("ctrl_world_video_chunk_extra")
            if not isinstance(ctrl_world_video_chunk, torch.Tensor):
                raise RuntimeError(
                    "Eval comparison video requires world-model chunk frames in infos."
                )
            self._save_cosmos_comparison_video(
                imagined_video_chunk=imagined_video_chunk,
                ctrl_world_video_chunk=ctrl_world_video_chunk,
                ctrl_world_video_chunk_wrist=ctrl_world_video_chunk_wrist,
                ctrl_world_video_chunk_extra=ctrl_world_video_chunk_extra,
                reward_cfg=self.cfg.reward.video_similarity,
                mode="eval",
            )

        if self._cosmos_video_similarity_enabled():
            if not isinstance(imagined_video_chunk, torch.Tensor):
                raise RuntimeError("Eval video similarity requires imagined Cosmos frames.")
            ctrl_world_video_chunk = infos.get("world_model_video_chunk")
            ctrl_world_video_chunk_wrist = infos.get(
                "world_model_video_chunk_wrist"
            )
            ctrl_world_video_chunk_extra = infos.get(
                "world_model_video_chunk_extra"
            )
            if not isinstance(ctrl_world_video_chunk, torch.Tensor):
                ctrl_world_video_chunk = infos.get("ctrl_world_video_chunk")
                ctrl_world_video_chunk_wrist = infos.get(
                    "ctrl_world_video_chunk_wrist"
                )
                ctrl_world_video_chunk_extra = infos.get(
                    "ctrl_world_video_chunk_extra"
                )
            if not isinstance(ctrl_world_video_chunk, torch.Tensor):
                ctrl_world_video_chunk = infos.get("ctrl_world_video_chunk_raw")
                ctrl_world_video_chunk_wrist = infos.get(
                    "ctrl_world_video_chunk_raw_wrist"
                )
                ctrl_world_video_chunk_extra = infos.get(
                    "ctrl_world_video_chunk_raw_extra"
                )
            if not isinstance(ctrl_world_video_chunk, torch.Tensor):
                raise RuntimeError("Eval video similarity requires world-model chunk frames.")
            reward_cfg = self.cfg.reward.video_similarity
            cosmos_cfg = self.cfg.actor.model.get("cosmos", {})
            camera_layout = reward_cfg.get(
                "camera_layout", cosmos_cfg.get("camera_layout", "main_top")
            )
            metric = self.cfg.reward.video_similarity.get("metric", "neg_mse")
            if metric == "clip_cosine":
                # Exp 7: Cosmos3 and Ctrl-World decode through unrelated VAEs
                # (Wan2.2 vs SVD's own), so their pre-decode latents are not
                # directly comparable -- compare the decoded pixel frames in a
                # shared frozen CLIP embedding space instead. See
                # rlinf/rewards/video_similarity.py::compute_video_clip_similarity_reward.
                clip_model_path = reward_cfg.get("clip_model_path", None)
                if not clip_model_path:
                    raise ValueError(
                        "reward.video_similarity.metric='clip_cosine' requires "
                        "reward.video_similarity.clip_model_path to be set."
                    )
                eval_video_rewards, eval_reward_stats = (
                    compute_video_clip_similarity_reward(
                        imagined_video_chunk,
                        ctrl_world_video_chunk,
                        clip_model_path=str(clip_model_path),
                        frame_index=self._parse_perceptual_frame_index(reward_cfg),
                        range_policy=self.cfg.reward.video_similarity.get(
                            "range_policy", "error"
                        ),
                        return_stats=True,
                    )
                )
                env_info["cosmos/video_similarity_reward_mean"] = (
                    eval_reward_stats["reward_mean"].detach().cpu().reshape(1)
                )
                # Reuse the same key the diagnostic harness reads regardless
                # of metric -- for clip_cosine this holds per-env cosine
                # similarity, not MSE (higher is better, range [-1, 1]).
                env_info["cosmos/video_similarity_per_env_chunk_mse"] = (
                    eval_reward_stats["per_env_reward"].detach().cpu()
                )
            elif metric == "dino_cosine":
                # Exp 7 follow-up: DINOv2 preserves fine-grained spatial
                # detail better than CLIP's language-aligned semantics -- see
                # rlinf/rewards/video_similarity.py::compute_video_dino_similarity_reward.
                dino_model_path = reward_cfg.get("dino_model_path", None)
                if not dino_model_path:
                    raise ValueError(
                        "reward.video_similarity.metric='dino_cosine' requires "
                        "reward.video_similarity.dino_model_path to be set."
                    )
                eval_video_rewards, eval_reward_stats = (
                    compute_video_dino_similarity_reward(
                        imagined_video_chunk,
                        ctrl_world_video_chunk,
                        dino_model_path=str(dino_model_path),
                        frame_index=self._parse_perceptual_frame_index(reward_cfg),
                        range_policy=self.cfg.reward.video_similarity.get(
                            "range_policy", "error"
                        ),
                        return_stats=True,
                    )
                )
                env_info["cosmos/video_similarity_reward_mean"] = (
                    eval_reward_stats["reward_mean"].detach().cpu().reshape(1)
                )
                env_info["cosmos/video_similarity_per_env_chunk_mse"] = (
                    eval_reward_stats["per_env_reward"].detach().cpu()
                )
            else:
                eval_video_rewards, eval_reward_stats = compute_video_similarity_reward(
                    imagined_video_chunk,
                    ctrl_world_video_chunk,
                    size=self.cfg.reward.video_similarity.get(
                        "size", self.cfg.reward.video_similarity.get("video_similarity_size", None)
                    ),
                    ctrl_world_video_chunk_wrist=ctrl_world_video_chunk_wrist,
                    ctrl_world_video_chunk_extra=ctrl_world_video_chunk_extra,
                    metric=metric,
                    range_policy=self.cfg.reward.video_similarity.get("range_policy", "error"),
                    alignment_mode=reward_cfg.get("alignment_mode", "legacy"),
                    camera_layout=camera_layout,
                    return_stats=True,
                    composite_view_stats=self.cfg.reward.video_similarity.get(
                        "reference_is_composite", False
                    ),
                    view_weights=reward_cfg.get("view_weights", None),
                )
                env_info["cosmos/video_similarity_reward_mean"] = (
                    eval_reward_stats["reward_mean"].detach().cpu().reshape(1)
                )
                env_info["cosmos/video_similarity_mse_mean"] = (
                    eval_reward_stats["mse_mean"].detach().cpu().reshape(1)
                )
                # Raw per-env chunk MSE (mean over frames only, batch dim kept
                # intact), for reward noise-floor/separability diagnostics that
                # need the full per-repeat sequence rather than a batch-wide mean.
                env_info["cosmos/video_similarity_per_env_chunk_mse"] = (
                    eval_reward_stats["per_frame_mse"].mean(dim=1).detach().cpu()
                )
                eval_video_mse_per_env = (
                    eval_reward_stats["per_frame_mse"].mean(dim=1).detach().cpu()
                )
                for name in ("main", "wrist", "extra"):
                    view_key = f"{name}_per_frame_mse"
                    if view_key in eval_reward_stats:
                        env_info[f"cosmos/video_{name}_mse_per_env"] = (
                            eval_reward_stats[view_key].mean(dim=1).detach().cpu()
                        )
                env_info["cosmos/reference_video_min"] = (
                    eval_reward_stats["reference_min"].detach().cpu().reshape(1)
                )
                env_info["cosmos/reference_video_max"] = (
                    eval_reward_stats["reference_max"].detach().cpu().reshape(1)
                )
                for view_name in ("main", "wrist", "extra"):
                    key = f"{view_name}_mse_mean"
                    if key in eval_reward_stats:
                        env_info[f"cosmos/video_similarity_{view_name}_mse_mean"] = (
                            eval_reward_stats[key].detach().cpu().reshape(1)
                        )
            env_info["cosmos/eval_action_override_code"] = torch.tensor(
                [{"policy": 0.0, "zero": 1.0, "random": 2.0, "fixed": 3.0, "reversed": 4.0}[action_override]]
            )

        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if eval_video_rewards is None:
            eval_video_rewards = torch.zeros_like(
                chunk_terminations, dtype=torch.float32
            )
        elif eval_video_rewards.dim() == 1:
            expanded = torch.zeros_like(chunk_terminations, dtype=torch.float32)
            expanded[:, -1] = eval_video_rewards.to(expanded.device)
            eval_video_rewards = expanded
        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=eval_video_rewards,
            reset_state_ids=infos.get(
                "reset_state_ids",
                self._get_env_reset_state_ids(self.eval_env_list[stage_id]),
            ),
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
        )
        continuous_eval_rewards = eval_video_rewards
        terminal_eval_rewards = None
        if self._terminal_goal_reward_enabled():
            required_views = (
                infos.get("ctrl_world_video_chunk"),
                infos.get("ctrl_world_video_chunk_wrist"),
                infos.get("ctrl_world_video_chunk_extra"),
            )
            if not all(isinstance(value, torch.Tensor) for value in required_views):
                raise RuntimeError(
                    "post-update terminal-goal evaluation requires all Ctrl-World views"
                )
            continuous_eval_rewards, terminal_eval_rewards = (
                self._apply_terminal_goal_reward(
                    eval_video_rewards,
                    env_output,
                    env_info,
                    ctrl_world_video_chunk=required_views[0],
                    ctrl_world_video_chunk_wrist=required_views[1],
                    ctrl_world_video_chunk_extra=required_views[2],
                )
            )
        probabilities = infos.get("chunk_raw_rewards")
        success_eval_rewards = None
        if self._success_classifier_diagnostic_enabled():
            if probabilities is None:
                raise RuntimeError(
                    "post-update success diagnostics require frame probabilities"
                )
            env_info["_reward_model_frame_probabilities"] = probabilities
            self._apply_success_classifier_diagnostic(
                continuous_eval_rewards, env_output, env_info
            )
            success_eval_rewards = env_info.pop(
                "_success_binary_rewards", None
            )
        selected_eval_rewards = self._select_training_rewards(
            trajectory_rewards=eval_video_rewards,
            terminal_rewards=terminal_eval_rewards,
            continuous_rewards=continuous_eval_rewards,
            success_rewards=success_eval_rewards,
        )
        env_output.rewards = selected_eval_rewards.detach().cpu().contiguous()
        self._record_post_update_eval_chunk(
            probabilities=probabilities,
            video_mse_per_env=eval_video_mse_per_env,
            combined_rewards=selected_eval_rewards,
            done_mask=chunk_dones[:, -1],
            env_info=env_info,
            final_side_video=infos.get("ctrl_world_video_chunk_raw_extra"),
        )

        if chunk_dones.any():
            done_mask = chunk_dones[:, -1]
            self._collect_episode_metrics(env_info, infos, done_mask=done_mask)
            self._collect_episode_metrics(
                env_info, infos.get("final_info"), done_mask=done_mask
            )

        return env_output, env_info

    def _save_eval_cosmos_comparison(self) -> bool:
        return bool(self.cfg.env.eval.video_cfg.get("save_cosmos_comparison", False))

    def _needs_eval_imagined_video(self) -> bool:
        return (
            self._save_eval_cosmos_comparison()
            or self._cosmos_video_similarity_enabled()
            or self._is_cosmos_self_feedback("eval")
        )

    def recv_eval_imagined_video(self, input_channel: Channel) -> torch.Tensor:
        """Receive and concatenate optional eval video shards by source rank."""
        video_shards = []
        for src_rank, expected_size in self.src_ranks["eval"]:
            video = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra="eval_imagined_video"
                )
            )
            if not isinstance(video, torch.Tensor):
                video = torch.as_tensor(video)
            if video.shape[0] != expected_size:
                raise ValueError(
                    "Eval imagined-video batch size does not match its action shard: "
                    f"expected {expected_size}, got {tuple(video.shape)}."
                )
            video_shards.append(video)
        return torch.cat(video_shards, dim=0).contiguous()

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        best_rollout_summary = {}
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                    self.env_list[i].wait_for_pending_saves()
                self.env_list[i].update_reset_state_ids()
            if self.cfg.env.train.video_cfg.get("save_video", False):
                self._flush_cosmos_comparison_video()
            if self._stream_all_cosmos_comparison("train"):
                self._flush_streamed_cosmos_comparison("train")
            if self._track_rollout_summaries():
                best_rollout_summary = self._flush_best_rollout_video_candidate()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                    self.eval_env_list[i].wait_for_pending_saves()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()
            if self._save_eval_cosmos_comparison():
                self._flush_cosmos_comparison_video(mode="eval")
        return best_rollout_summary

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        """Split one env batch dict into size-specified sub-batches along dim-0.

        Tensor values are chunked on dim-0; list values are sliced proportionally;
        nested dict values are split recursively.

        Args:
            env_batch: Env output dictionary produced by ``EnvOutput.to_dict``.
            sizes: Batch sizes for each destination rank.
            mode: Rollout mode used for list-length validation.

        Returns:
            A list of split env batches, one item per destination rank.
        """
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            output_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    reset_state_ids=self._get_env_reset_state_ids(
                        self.env_list[stage_id]
                    ),
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    reset_state_ids=self._get_env_reset_state_ids(
                        self.env_list[stage_id]
                    ),
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    @staticmethod
    def _concatenate_metric_lists(
        metric_lists: dict[str, list[Any]], *, context: str
    ) -> dict[str, torch.Tensor]:
        """Validate and concatenate numeric worker metrics with useful errors."""
        concatenated: dict[str, torch.Tensor] = {}
        for key, values in metric_lists.items():
            if not values:
                continue
            non_tensor_types = sorted(
                {type(value).__name__ for value in values if not torch.is_tensor(value)}
            )
            if non_tensor_types:
                raise TypeError(
                    f"{context} metric {key!r} must contain only tensors before "
                    f"aggregation; found {', '.join(non_tensor_types)}"
                )
            normalized_values = [
                value.reshape(1) if value.ndim == 0 else value for value in values
            ]
            concatenated[key] = (
                torch.cat(normalized_values, dim=0).contiguous().cpu()
            )
        return concatenated

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        actor_channel_cfg = self.cfg.runner.get("actor_channel", {})
        route_by_node = bool(actor_channel_cfg.get("route_by_node", False))
        if route_by_node and not channel.is_distributed:
            raise RuntimeError(
                "runner.actor_channel.route_by_node requires a distributed "
                "Actor channel."
            )
        channel_key = (
            channel.node_local_key("actor_trajectory")
            if route_by_node
            else None
        )
        send_works = []
        for trajectory in trajectories:
            if channel_key is None:
                work = channel.put(trajectory, async_op=True)
            else:
                work = channel.put(trajectory, key=channel_key, async_op=True)
            if work is not None:
                send_works.append(work)

        # Async Channel sends retain their source dataclass and every tensor it
        # contains until communication completes. Waiting here is safe because
        # Actor workers have already posted their receives, and it prevents a
        # full trajectory copy from remaining live through Actor training and
        # the next rollout update.
        for work in send_works:
            await work.async_wait()
        del trajectories
        clear_memory(sync=False, trim_cpu=True)

    async def _run_interact_once(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            self._cosmos_previous_similarity_scores = None
            if self._cross_rank_group_cfg() is not None:
                for stage_id in range(self.stage_num):
                    initial_metadata = self._build_cross_rank_metadata(
                        stage_id=stage_id,
                        chunk_index=0,
                        rollout_epoch_index=epoch,
                    )
                    self._set_env_cross_rank_context(
                        self.env_list[stage_id],
                        initial_metadata,
                        reset_rollout=True,
                    )
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_outputs[stage_id]
                self._attach_cross_rank_metadata(
                    env_output.obs,
                    self._build_cross_rank_metadata(
                        stage_id=stage_id,
                        chunk_index=0,
                        rollout_epoch_index=epoch,
                    ),
                )
                env_batch = env_output.to_dict()
                self.send_env_batch(
                    output_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )
                # dones/terminations/truncations need trajectory_length + 1
                # entries (one per state boundary, states 0..T for T chunks),
                # so this pre-loop reset state (always non-done, state 0) has
                # to be recorded here. Without it, the chunk_step loop below
                # only ever appends states 1..T, and advantages.py's
                # `dones[step + 1]` lookup silently reads one state ahead of
                # what it means to -- e.g. it reads state T's (the actual
                # final chunk's) done flag when computing chunk T-2's return,
                # cutting the discounted-return chain short by one chunk.
                self.rollout_results[stage_id].append_step_result(
                    ChunkStepResult(
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                    )
                )

            for chunk_step in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    chunk_metadata = self._build_cross_rank_metadata(
                        stage_id=stage_id,
                        chunk_index=chunk_step,
                        rollout_epoch_index=epoch,
                    )
                    if chunk_metadata:
                        self._set_env_cross_rank_context(
                            self.env_list[stage_id],
                            chunk_metadata,
                            reset_rollout=False,
                        )
                    self._attach_cross_rank_metadata(
                        env_output.obs, chunk_metadata
                    )
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    if self._is_cosmos_self_feedback("train"):
                        self._set_policy_feedback(
                            self.env_list[stage_id],
                            rollout_result.forward_inputs.get("imagined_video_chunk"),
                            mode="train",
                        )
                    next_env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    default_rewards = self.compute_bootstrap_rewards(
                        next_env_output, rollout_result.bootstrap_values
                    )
                    rewards = self._attach_cosmos_video_reward(
                        rollout_result,
                        next_env_output,
                        env_info,
                        default_rewards,
                        comparison_metadata=chunk_metadata,
                    )
                    video_similarity_rewards = env_info.pop(
                        "_video_similarity_rewards", None
                    )
                    terminal_goal_rewards = env_info.pop(
                        "_terminal_goal_rewards", None
                    )
                    continuous_combined_rewards = env_info.pop(
                        "_continuous_combined_rewards", None
                    )
                    reward_model_probabilities = env_info.pop(
                        "_reward_model_probabilities", None
                    )
                    ctrl_retry_count = env_info.pop(
                        "_ctrl_world_rollout_retry_count", None
                    )
                    native_retry_count = rollout_result.forward_inputs.get(
                        "rollout_retry_count"
                    )
                    if native_retry_count is None:
                        native_retry_count = torch.zeros(
                            rewards.shape[0], 1, dtype=torch.int64
                        )
                    else:
                        native_retry_count = torch.as_tensor(
                            native_retry_count, dtype=torch.int64
                        ).detach().cpu().reshape(rewards.shape[0], -1)[:, :1]
                    if ctrl_retry_count is None:
                        ctrl_retry_count = torch.zeros_like(native_retry_count)
                    else:
                        ctrl_retry_count = torch.as_tensor(
                            ctrl_retry_count, dtype=torch.int64
                        ).detach().cpu().reshape(rewards.shape[0], -1)[:, :1]
                    retry_count = native_retry_count + ctrl_retry_count
                    if self._is_cosmos_model():
                        actions_for_diversity = torch.as_tensor(
                            rollout_result.actions, dtype=torch.float32
                        )
                        cross_rank_enabled = self._cross_rank_group_cfg() is not None
                        group_size = int(self.cfg.algorithm.group_size)
                        if (
                            not cross_rank_enabled
                            and
                            group_size > 1
                            and actions_for_diversity.shape[0] % group_size == 0
                        ):
                            grouped_actions = actions_for_diversity.reshape(
                                -1, group_size, actions_for_diversity[0].numel()
                            )
                            pairwise = torch.cdist(grouped_actions, grouped_actions)
                            upper = torch.triu(
                                torch.ones(
                                    group_size,
                                    group_size,
                                    dtype=torch.bool,
                                    device=pairwise.device,
                                ),
                                diagonal=1,
                            )
                            env_info["cosmos/action_group_pairwise_l2_mean"] = (
                                pairwise[:, upper].mean().detach().cpu().reshape(1)
                            )
                            env_info["cosmos/action_group_member_std_mean"] = (
                                grouped_actions.std(dim=1).mean().detach().cpu().reshape(1)
                            )
                    if self._is_cosmos_model():
                        self._cosmos_artifact_writer.write_rollout_chunk(
                            rank=self._rank,
                            stage_id=stage_id,
                            rollout_epoch=epoch,
                            chunk_step=chunk_step,
                            actions=rollout_result.actions,
                            prev_logprobs=rollout_result.prev_logprobs,
                            rewards=rewards,
                            dones=next_env_output.dones,
                            forward_inputs=rollout_result.forward_inputs,
                            sample_metadata={
                                field_name: chunk_metadata[field_name]
                                for field_name in (
                                    "rollout_uid",
                                    "global_group_id",
                                    "group_member_id",
                                )
                                if field_name in chunk_metadata
                            },
                        )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=self._actor_replay_forward_inputs(
                            rollout_result.forward_inputs
                        ),
                        versions=rollout_result.versions,
                        dones=next_env_output.dones,
                        truncations=next_env_output.truncations,
                        terminations=next_env_output.terminations,
                        rewards=rewards,
                        video_similarity_rewards=video_similarity_rewards,
                        terminal_goal_rewards=terminal_goal_rewards,
                        continuous_combined_rewards=continuous_combined_rewards,
                        reward_model_probabilities=reward_model_probabilities,
                        dynamic_gammas=next_env_output.dynamic_gammas,
                        latent_motion=next_env_output.latent_motion,
                        reset_state_ids=next_env_output.reset_state_ids,
                        rollout_uid=chunk_metadata.get("rollout_uid"),
                        global_group_id=chunk_metadata.get("global_group_id"),
                        group_member_id=chunk_metadata.get("group_member_id"),
                        source_env_rank=chunk_metadata.get("source_env_rank"),
                        local_env_id=chunk_metadata.get("local_env_id"),
                        reset_seed=chunk_metadata.get("reset_seed"),
                        vision_noise_seed=chunk_metadata.get("vision_noise_seed"),
                        action_noise_seed=chunk_metadata.get("action_noise_seed"),
                        ctrl_world_noise_seed=chunk_metadata.get(
                            "ctrl_world_noise_seed"
                        ),
                        update_id=chunk_metadata.get("update_id"),
                        logical_round_id=chunk_metadata.get("logical_round_id"),
                        physical_wave_id=chunk_metadata.get("physical_wave_id"),
                        group_slot=chunk_metadata.get("group_slot"),
                        chunk_id=chunk_metadata.get("chunk_id"),
                        reset_episode=chunk_metadata.get("reset_episode"),
                        seed_nonce=chunk_metadata.get("seed_nonce"),
                        shuffle_seed=chunk_metadata.get("shuffle_seed"),
                        retry_count=retry_count,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    self._attach_cross_rank_metadata(
                        next_env_output.obs,
                        self._build_cross_rank_metadata(
                            stage_id=stage_id,
                            chunk_index=chunk_step + 1,
                            rollout_epoch_index=epoch,
                        ),
                    )
                    env_batch = next_env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            next_env_output.final_obs
                            if next_env_output.dones.any()
                            and self.cfg.env.train.auto_reset
                            else next_env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = next_env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                # dones/truncations/terminations for this same final state
                # were already recorded by the last chunk_step iteration
                # above (env_output here is that same state, unchanged) --
                # appending them again would reintroduce the extra trailing
                # duplicate the pre-loop append above exists to avoid. Only
                # prev_values (the bootstrap value estimate for this final
                # state, needed by GAE) belongs in this trailing entry.
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

            self.store_last_obs_and_intervened_info(env_outputs)
            best_rollout_summary = self.finish_rollout()
            for key, value in best_rollout_summary.items():
                env_metrics[key].append(value)

        if actor_channel is not None:
            if self._libero_verifier_enabled():
                for stage_id in range(self.stage_num):
                    verification_metrics = self._verify_world_model_success_in_libero(
                        stage_id
                    )
                    for key, value in verification_metrics.items():
                        env_metrics[key].append(value.cpu())

            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )

            # The Channel owns complete trajectory copies after the awaited
            # puts. Do not keep the per-step source buffers on EnvWorker until
            # the next call to interact(), especially while Actor training and
            # checkpointing run in the same long-lived Ray process.
            self.rollout_results = []
            clear_memory(sync=False, trim_cpu=True)

        env_metrics = self._concatenate_metric_lists(
            env_metrics, context="train rollout"
        )

        return env_metrics

    async def _run_replay_only_once(self) -> dict[str, torch.Tensor]:
        env_metrics = defaultdict(list)
        action_shape = (
            self.train_num_envs_per_stage,
            self.cfg.actor.model.num_action_chunks,
            self.cfg.actor.model.action_dim,
        )

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    dummy_actions = torch.zeros(action_shape, dtype=torch.float32)
                    env_output, env_info = self.env_interact_step(
                        dummy_actions, stage_id
                    )
                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        env_metrics = self._concatenate_metric_lists(
            env_metrics, context="replay rollout"
        )

        return env_metrics

    @Worker.timer("replay_only_interact")
    async def replay_only_interact(self):
        env_metrics = await self._run_replay_only_once()

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            output_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        if not self.eval_env_active:
            return {}
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    extracted_obs, infos = self.eval_env_list[
                        stage_id
                    ].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    imagined_video_chunk = (
                        self.recv_eval_imagined_video(input_channel)
                        if self._needs_eval_imagined_video()
                        else None
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions,
                        stage_id,
                        imagined_video_chunk=imagined_video_chunk,
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        context = self._post_update_eval_context
        if (
            context is not None
            and context.get("active", False)
            and all(context.get("local_padding", ()))
        ):
            return {}
        eval_metrics = self._concatenate_metric_lists(
            eval_metrics, context="evaluation rollout"
        )

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num
