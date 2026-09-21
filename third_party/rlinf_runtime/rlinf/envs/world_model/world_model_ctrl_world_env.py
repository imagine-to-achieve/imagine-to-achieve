# Copyright 2026 The RLinf Authors.
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

import importlib
import gc
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from rlinf.algorithms.cross_rank_grpo import derive_stable_seed
from rlinf.data.datasets.lerobot_book import LeRobotBookTrajectoryDatasetWrapper
from rlinf.data.datasets.lerobot_world_model import LeRobotTrajectoryDatasetWrapper
from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.duck_episode_contract import (
    assert_duck_episode_split_unchanged,
    load_duck_episode_split,
)
from rlinf.envs.world_model.eef_pose_adapter import (
    ActionToEEFPoseAdapter,
    EEFAdapterConfig,
    apply_eef_rigid_alignment,
    analytic_libero_delta_actions_to_eef_states,
    analytic_ur5_rot6d_delta_actions_to_eef_states,
    compose_learned_eef_delta_to_states,
    compute_eef_rigid_alignment,
    resample_eef_state_sequence_by_fps,
    resample_joint_state_sequence_by_fps,
    state_to_ctrl_world_eef_state,
    ur5_joint_actions_to_eef_states,
)
from rlinf.utils.logging import get_logger, quiet_third_party_progress_bars
from rlinf.utils.utils import trim_cpu_allocator

quiet_third_party_progress_bars()

__all__ = [
    "CtrlWorldEnv",
    "CosmosSelfFeedbackCtrlWorldEnv",
    "ACTION_TO_EEF_ADAPTER_CHOICES",
    "LEARNED_ACTION_TO_EEF_ADAPTERS",
]

# Single source of truth for legal `action_to_eef_adapter` config values.
# Import this instead of re-listing the strings (e.g. in CLI `choices=[...]`
# or test membership checks) so a typo fails fast instead of silently
# falling through to an unhandled adapter name.
ACTION_TO_EEF_ADAPTER_CHOICES = (
    "analytic",
    "analytic_ur5_rot6d",
    "learned",
    "joint_learned_fk",
    "none",
)

# Subset of ACTION_TO_EEF_ADAPTER_CHOICES that requires the learned
# ActionToEEFPoseAdapter checkpoint to be loaded.
LEARNED_ACTION_TO_EEF_ADAPTERS = frozenset({"learned", "joint_learned_fk"})


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _plan_ctrl_world_windows(
    total_emit_frames: int,
    window_frames: int,
    emit_frames: int,
    *,
    use_lookahead_latent: bool = False,
) -> list[tuple[int, int, int]]:
    """Plan overlapping Ctrl-World calls as (start, request, emit) tuples."""
    if total_emit_frames <= 0:
        raise ValueError("total_emit_frames must be positive")
    if window_frames <= 0 or emit_frames <= 0 or emit_frames > window_frames:
        raise ValueError(
            "window_frames and emit_frames must satisfy "
            f"1 <= emit_frames <= window_frames, got {emit_frames}, {window_frames}"
        )
    if use_lookahead_latent and emit_frames >= window_frames:
        raise ValueError(
            "lookahead rollout requires one non-emitted boundary frame: "
            f"emit_frames={emit_frames} must be smaller than "
            f"window_frames={window_frames}"
        )
    return [
        (start, window_frames, min(emit_frames, total_emit_frames - start))
        for start in range(0, total_emit_frames, emit_frames)
    ]

@dataclass
class _CtrlWorldArgs:
    """Minimal argument bundle required by the Ctrl-World constructor."""

    svd_model_path: str
    clip_model_path: str
    action_dim: int
    num_history: int
    num_frames: int
    text_cond: bool


class CtrlWorldEnv(BaseWorldEnv):
    """World-model environment based on Ctrl-World.

    High-level flow:
    1. `reset`: select initial frames from the dataset and initialize latent and
       history state.
    2. `chunk_step`: consume an action chunk with shape `[B, chunk, action_dim]`,
       predict the next latent/image chunk with Ctrl-World, score it with the
       reward model, and return chunk-level rewards and termination signals.
    3. Maintain internal state across chunks, including `current_latent`,
       `history_latents`, `action_history`, and episode metrics.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # Ctrl-World runtime configuration.
        self.seed_offset = seed_offset
        self.ctrl_world_cfg = self.cfg.ctrl_world_cfg
        self.reward_source = str(self.cfg.get("reward_source", "legacy_reward_model"))
        self.inference_dtype = self._to_torch_dtype(
            self.ctrl_world_cfg.get("dtype", "bf16")
        )

        # Groups align reset-state ids with GRPO groups.
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_ordered_reset_state_ids = bool(
            cfg.get("use_ordered_reset_state_ids", False)
        )
        self.random_reset_state_ids = bool(
            cfg.get("random_reset_state_ids", False)
        )
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        configured_reset_ids = cfg.get("reset_episode_ids", None)
        manifest_path = cfg.get("episode_manifest_path", None)
        manifest_split = cfg.get("episode_manifest_split", None)
        loaded_split = None
        if manifest_path is not None:
            if manifest_split is None:
                raise ValueError(
                    "episode_manifest_split is required with episode_manifest_path"
                )
            loaded_split = load_duck_episode_split(
                str(manifest_path),
                str(manifest_split),
                verify_source_hashes=bool(
                    cfg.get("verify_episode_manifest_source_hashes", True)
                ),
            )
            if configured_reset_ids is None:
                configured_reset_ids = loaded_split.episodes
            elif (
                tuple(int(value) for value in configured_reset_ids)
                != loaded_split.episodes
            ):
                raise ValueError(
                    "reset_episode_ids does not exactly match the frozen manifest split"
                )
        self.reset_episode_ids = (
            tuple(int(value) for value in configured_reset_ids)
            if configured_reset_ids is not None
            else None
        )
        if self.reset_episode_ids is not None:
            if not self.reset_episode_ids:
                raise ValueError("reset_episode_ids must not be empty")
            if len(set(self.reset_episode_ids)) != len(self.reset_episode_ids):
                raise ValueError("reset_episode_ids must not contain duplicates")
            invalid_ids = [
                episode_id
                for episode_id in self.reset_episode_ids
                if not 0 <= episode_id < len(self.dataset)
            ]
            if invalid_ids:
                raise ValueError(
                    "reset_episode_ids contains ids outside the dataset: "
                    f"{invalid_ids}; dataset size={len(self.dataset)}"
                )
        self.reset_split = str(
            cfg.get(
                "reset_split",
                loaded_split.name if loaded_split is not None else "unspecified",
            )
        )
        self._episode_contract = loaded_split
        # Episode ids to keep out of ordered/random reset sampling (e.g. a
        # fixed held-out eval split). Never filters `specific_reset_id`
        # itself, so eval can still be pinned to exactly these ids.
        self.exclude_reset_ids = sorted(
            {int(x) for x in (cfg.get("exclude_reset_ids", None) or [])}
        )
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self._ordered_reset_cursor = 0
        self.common_random_numbers_within_group = _as_bool(
            self.ctrl_world_cfg.get("common_random_numbers_within_group", False)
        )
        self.common_noise_seed = int(
            self.ctrl_world_cfg.get("common_noise_seed", self.seed)
        )
        fixed_denoise_seed = self.ctrl_world_cfg.get("fixed_denoise_seed", None)
        if isinstance(fixed_denoise_seed, str):
            fixed_denoise_seed = fixed_denoise_seed.strip()
            if fixed_denoise_seed.lower() in {"", "none", "null"}:
                fixed_denoise_seed = None
        if fixed_denoise_seed is not None:
            try:
                fixed_denoise_seed = int(fixed_denoise_seed)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "ctrl_world_cfg.fixed_denoise_seed must be an integer or null."
                ) from error
            if fixed_denoise_seed < 0:
                raise ValueError("fixed_denoise_seed must be non-negative.")
        self.fixed_denoise_seed = fixed_denoise_seed
        self._fixed_denoise_seeds: tuple[int, ...] | None = None
        self._post_update_eval_denoise_seed_log: list[dict[str, object]] = []
        self._post_update_eval_context: dict[str, object] | None = None
        self._per_member_vision_seed = _as_bool(
            self.ctrl_world_cfg.get("per_member_vision_seed", False)
        )
        self._common_noise_call_index = 0
        # Incremented on every full rollout reset. Members of one GRPO group
        # share noise within a rollout, while successive RL steps see a new
        # deterministic Ctrl-World noise stream.
        self._common_noise_rollout_index = -1
        self._cross_rank_metadata: dict[str, torch.Tensor] | None = None

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        self.chunk = int(self.ctrl_world_cfg.chunk)
        self.policy_chunk = self.chunk
        self.cosmos_action_fps = float(
            self.ctrl_world_cfg.get(
                "cosmos_action_fps",
                self.ctrl_world_cfg.get("policy_action_fps", self.ctrl_world_cfg.get("fps", 7)),
            )
        )
        self.ctrl_condition_fps = float(
            self.ctrl_world_cfg.get("ctrl_condition_fps", self.ctrl_world_cfg.get("fps", 7))
        )
        self.action_fps_resample = _as_bool(
            self.ctrl_world_cfg.get("action_fps_resample", False)
        )
        if self.cosmos_action_fps <= 0 or self.ctrl_condition_fps <= 0:
            raise ValueError(
                "cosmos_action_fps and ctrl_condition_fps must be positive, "
                f"got {self.cosmos_action_fps}, {self.ctrl_condition_fps}"
            )
        default_ctrl_world_chunk = self.policy_chunk
        if self.action_fps_resample:
            default_ctrl_world_chunk = int(
                round(self.policy_chunk * self.ctrl_condition_fps / self.cosmos_action_fps)
            )
        self.ctrl_world_chunk = int(
            self.ctrl_world_cfg.get("ctrl_world_chunk", default_ctrl_world_chunk)
        )
        if self.ctrl_world_chunk <= 0:
            raise ValueError(f"ctrl_world_chunk must be positive, got {self.ctrl_world_chunk}")
        self.ctrl_world_internal_rollout = _as_bool(
            self.ctrl_world_cfg.get("ctrl_world_internal_rollout", False)
        )
        self.ctrl_world_window_frames = int(
            self.ctrl_world_cfg.get("ctrl_world_window_frames", min(5, self.ctrl_world_chunk))
        )
        self.ctrl_world_window_emit_frames = int(
            self.ctrl_world_cfg.get(
                "ctrl_world_window_emit_frames",
                max(1, min(self.ctrl_world_window_frames - 1, self.ctrl_world_chunk)),
            )
        )
        self.ctrl_world_window_use_lookahead_latent = _as_bool(
            self.ctrl_world_cfg.get("ctrl_world_window_use_lookahead_latent", False)
        )

        if self.ctrl_world_internal_rollout:
            if self.ctrl_world_window_frames <= 0:
                raise ValueError(
                    f"ctrl_world_window_frames must be positive, got {self.ctrl_world_window_frames}"
                )
            if not (1 <= self.ctrl_world_window_emit_frames <= self.ctrl_world_window_frames):
                raise ValueError(
                    "ctrl_world_window_emit_frames must be in "
                    f"[1, ctrl_world_window_frames], got {self.ctrl_world_window_emit_frames}"
                )
            if (
                self.ctrl_world_window_use_lookahead_latent
                and self.ctrl_world_window_emit_frames
                >= self.ctrl_world_window_frames
            ):
                raise ValueError(
                    "lookahead Ctrl-World rollout requires "
                    "ctrl_world_window_emit_frames < ctrl_world_window_frames"
                )
        self.condition_frame_length = int(
            self.ctrl_world_cfg.get("condition_frame_length", 1)
        )
        self.num_history = int(self.ctrl_world_cfg.get("num_history", 6))
        self.ctrl_world_window_history_mode = str(
            self.ctrl_world_cfg.get("ctrl_world_window_history_mode", "dense")
        )
        if self.ctrl_world_window_history_mode not in {"dense", "sparse_window"}:
            raise ValueError(
                "ctrl_world_window_history_mode must be 'dense' or 'sparse_window', "
                f"got {self.ctrl_world_window_history_mode!r}"
            )
        self.ctrl_world_history_idx = list(
            self.ctrl_world_cfg.get("ctrl_world_history_idx", [0, 0, -12, -9, -6, -3])
        )
        if len(self.ctrl_world_history_idx) != self.num_history:
            raise ValueError(
                f"ctrl_world_history_idx must have {self.num_history} entries, "
                f"got {len(self.ctrl_world_history_idx)}"
            )
        min_history_bank = max(
            self.num_history,
            max((abs(int(idx)) for idx in self.ctrl_world_history_idx if int(idx) < 0), default=0),
        )
        self.ctrl_world_history_bank_size = int(
            self.ctrl_world_cfg.get(
                "ctrl_world_history_bank_size",
                max(self.num_history * 4, min_history_bank),
            )
        )
        if self.ctrl_world_history_bank_size < min_history_bank:
            raise ValueError(
                "ctrl_world_history_bank_size is too small for ctrl_world_history_idx: "
                f"{self.ctrl_world_history_bank_size} < {min_history_bank}"
            )
        self.action_dim = int(self.ctrl_world_cfg.get("action_dim", 7))
        # Raw policy action width, which can differ from Ctrl-World's own
        # native EEF-state conditioning dimension (`action_dim` above, fixed
        # by the pretrained Ctrl-World checkpoint's action_encoder). Defaults
        # to `action_dim` for policies whose raw action already matches
        # Ctrl-World's condition width (e.g. 7D joint/delta_eef policies).
        self.policy_action_dim = int(
            self.ctrl_world_cfg.get("policy_action_dim", self.action_dim)
        )
        self.policy_action_type = str(
            self.ctrl_world_cfg.get("policy_action_type", "delta_eef")
        ).lower()
        self.policy_gripper_type = str(
            self.ctrl_world_cfg.get("policy_gripper_type", "passthrough")
        ).lower()
        self.action_to_eef_adapter = str(
            self.ctrl_world_cfg.get("action_to_eef_adapter", "none")
        ).lower()
        if self.action_to_eef_adapter not in ACTION_TO_EEF_ADAPTER_CHOICES:
            raise ValueError(
                f"Unsupported action_to_eef_adapter: {self.action_to_eef_adapter!r}. "
                f"Expected one of {ACTION_TO_EEF_ADAPTER_CHOICES}."
            )
        # Dispatch table for the adapters handled after the joint_absolute
        # preamble in _convert_policy_actions_to_ctrl_world (joint_learned_fk
        # is handled entirely inside that preamble, since it needs the
        # joint-space current/next state, not just actions_np).
        self._action_to_eef_adapter_dispatch = {
            "analytic": self._convert_via_analytic,
            "analytic_ur5_rot6d": self._convert_via_analytic_ur5_rot6d,
            "learned": self._convert_via_learned,
            "none": self._convert_via_none,
        }
        self.joint_action_adapter = str(
            self.ctrl_world_cfg.get("joint_action_adapter", "none")
        ).lower()
        self.wm_env_type = str(self.cfg.get("wm_env_type", "")).lower()
        self.adapter_position_scale = float(
            self.ctrl_world_cfg.get("adapter_position_scale", 0.05)
        )
        self.adapter_rotation_scale = float(
            self.ctrl_world_cfg.get("adapter_rotation_scale", 0.5)
        )
        self.adapter_translation_gain = float(
            self.ctrl_world_cfg.get("adapter_translation_gain", 0.15)
        )
        self.adapter_rotation_gain = float(
            self.ctrl_world_cfg.get("adapter_rotation_gain", 1.0)
        )
        self.adapter_translation_frame = str(
            self.ctrl_world_cfg.get("adapter_translation_frame", "body")
        ).lower()
        self.adapter_translation_sign = float(
            self.ctrl_world_cfg.get("adapter_translation_sign", 1.0)
        )
        self.adapter_rotation_mode = str(
            self.ctrl_world_cfg.get("adapter_rotation_mode", "body")
        ).lower()
        self.adapter_gripper_open = float(
            self.ctrl_world_cfg.get("adapter_gripper_open", 0.04)
        )
        self.adapter_gripper_close = float(
            self.ctrl_world_cfg.get("adapter_gripper_close", 0.0)
        )
        self.adapter_gripper_hard_clamp = (
            self.ctrl_world_cfg.get("adapter_gripper_hard_clamp", False) is True
        )
        self.adapter_gripper_hard_threshold = float(
            self.ctrl_world_cfg.get("adapter_gripper_hard_threshold", 0.0)
        )
        self.adapter_gripper_hard_mode = str(
            self.ctrl_world_cfg.get("adapter_gripper_hard_mode", "policy_sign")
        ).lower()
        self.joint_fk_tcp_xyz = tuple(
            float(x) for x in self.ctrl_world_cfg.get("joint_fk_tcp_xyz", [0.0, 0.0, 0.18])
        )
        self.joint_fk_tcp_rotvec = tuple(
            float(x) for x in self.ctrl_world_cfg.get("joint_fk_tcp_rotvec", [0.0, 0.0, 0.0])
        )

        self.learned_eef_adapter = None
        self.eef_adapter_state_mean = None
        self.eef_adapter_state_std = None
        self.eef_adapter_action_mean = None
        self.eef_adapter_action_std = None
        self.eef_adapter_delta_mean = None
        self.eef_adapter_delta_std = None
        if self.action_to_eef_adapter in LEARNED_ACTION_TO_EEF_ADAPTERS:
            self._load_learned_action_to_eef_adapter()

        self.image_size = tuple(self.ctrl_world_cfg.get("image_size", [192, 320]))
        self.policy_image_size = self.ctrl_world_cfg.get("policy_image_size", [480, 640])
        self.policy_image_size = (
            tuple(self.policy_image_size) if self.policy_image_size is not None else None
        )
        # Ctrl-World internally uses vertically stacked three-view images.
        self.full_image_size = (self.image_size[0] * 3, self.image_size[1])
        self.per_view_vae_codec = _as_bool(
            self.ctrl_world_cfg.get("per_view_vae_codec", False)
        )
        self.model_camera_ids = [
            int(value)
            for value in self.ctrl_world_cfg.get("model_camera_ids", [0, 1, 2])
        ]
        if sorted(self.model_camera_ids) != [0, 1, 2]:
            raise ValueError(
                "model_camera_ids must be a permutation of the three source "
                f"camera indices, got {self.model_camera_ids}"
            )
        self.model_view_order = [
            str(value)
            for value in self.ctrl_world_cfg.get(
                "model_view_order", ["source:0", "source:1", "source:2"]
            )
        ]
        if len(self.model_view_order) != 3:
            raise ValueError("model_view_order must contain exactly three labels")
        self.strict_native_resolution = _as_bool(
            self.ctrl_world_cfg.get("strict_native_resolution", False)
        )
        self.expected_latent_size = (
            self.full_image_size[0] // 8,
            self.full_image_size[1] // 8,
        )
        if self.strict_native_resolution:
            if self.image_size != (480, 640):
                raise ValueError(
                    "7D native Ctrl-World requires image_size=(480, 640), got "
                    f"{self.image_size}"
                )
            if self.full_image_size != (1440, 640):
                raise ValueError(
                    f"Expected native full_image_size=(1440, 640), got {self.full_image_size}"
                )
            if self.expected_latent_size != (180, 80):
                raise ValueError(
                    f"Expected native latent size=(180, 80), got {self.expected_latent_size}"
                )

        self.main_view_index = int(self.ctrl_world_cfg.get("main_view_index", 1))
        self.wrist_view_index = int(self.ctrl_world_cfg.get("wrist_view_index", 2))
        if self.main_view_index == self.wrist_view_index:
            raise ValueError("main_view_index and wrist_view_index must be different")
        self.extra_view_index = next(
            idx for idx in range(3) if idx not in {self.main_view_index, self.wrist_view_index}
        )

        self.num_inference_steps = int(self.ctrl_world_cfg.get("num_inference_steps", 50))
        self.decode_chunk_size = int(self.ctrl_world_cfg.get("decode_chunk_size", 7))
        self.guidance_scale = float(self.ctrl_world_cfg.get("guidance_scale", 1.0))
        self.fps = int(self.ctrl_world_cfg.get("fps", 7))
        self.motion_bucket_id = int(self.ctrl_world_cfg.get("motion_bucket_id", 127))
        self.frame_level_cond = bool(self.ctrl_world_cfg.get("frame_level_cond", True))
        self.his_cond_zero = bool(self.ctrl_world_cfg.get("his_cond_zero", False))
        self.text_cond = bool(self.ctrl_world_cfg.get("text_cond", True))
        self.use_raw_reset_policy_views = _as_bool(
            self.ctrl_world_cfg.get("use_raw_reset_policy_views", False)
        )
        if not (0 <= self.main_view_index <= 2 and 0 <= self.wrist_view_index <= 2):
            raise ValueError("main_view_index and wrist_view_index must be in [0, 2]")
        self.reward_model_view_index = self.main_view_index
        self.reward_model_camera_key = ""
        success_model_required = (
            getattr(self, "reward_source", "legacy_reward_model")
            == "success_model"
            or bool(
                self.ctrl_world_cfg.get(
                    "reward_model_diagnostic_only", False
                )
            )
        )
        if success_model_required:
            source_main_index = self.model_camera_ids[self.main_view_index]
            source_wrist_index = self.model_camera_ids[self.wrist_view_index]
            if (source_main_index, source_wrist_index) != (0, 2):
                raise ValueError(
                    "success model or diagnostic requires physical camera mapping "
                    "front=source camera 0 and wrist=source camera 2; got "
                    f"model_camera_ids={self.model_camera_ids}, "
                    f"main_view_index={self.main_view_index}, "
                    f"wrist_view_index={self.wrist_view_index}"
                )
            routed_reward_cfg = self.ctrl_world_cfg.get("reward_models", None)
            reward_cfg = self.ctrl_world_cfg.get("reward_model", {})
            camera_keys = list(self.cfg.get("initial_image_camera_keys", []))
            expected_camera_key = str(reward_cfg.get("camera_key", "")).strip()
            if not expected_camera_key:
                raise ValueError(
                    "success_model reward_model.camera_key must be configured"
                )
            reward_model_view_index = int(
                reward_cfg.get("ctrl_world_view_index", self.main_view_index)
            )
            if not 0 <= reward_model_view_index <= 2:
                raise ValueError(
                    "success_model ctrl_world_view_index must be in [0, 2], got "
                    f"{reward_model_view_index}"
                )
            source_reward_index = self.model_camera_ids[reward_model_view_index]
            if camera_keys:
                if source_reward_index >= len(camera_keys):
                    raise ValueError(
                        "success_model source camera index is outside "
                        f"initial_image_camera_keys: {source_reward_index} >= "
                        f"{len(camera_keys)}"
                    )
                if camera_keys[source_reward_index] != expected_camera_key:
                    raise ValueError(
                        "success_model camera_key does not match its configured "
                        f"physical source view {source_reward_index}: "
                        f"{camera_keys[source_reward_index]!r} != "
                        f"{expected_camera_key!r}"
                    )
            view_roles = {
                self.main_view_index: "main",
                self.wrist_view_index: "wrist",
                self.extra_view_index: "side",
            }
            reward_model_view_role = view_roles[reward_model_view_index]
            logical_camera_key = str(
                self.ctrl_world_cfg.get("camera_keys", {}).get(
                    reward_model_view_role, ""
                )
            ).strip()
            if logical_camera_key and logical_camera_key != expected_camera_key:
                raise ValueError(
                    "success_model camera_key does not match the configured "
                    f"{reward_model_view_role} view: {logical_camera_key!r} != "
                    f"{expected_camera_key!r}"
                )
            self.reward_model_view_index = reward_model_view_index
            self.reward_model_camera_key = expected_camera_key

        self.ctrl_world_pipeline_cls = None
        self.model = self._build_ctrl_world_model().eval().to(self.device, self.inference_dtype)
        self.reward_model = self._load_reward_model()
        if self.reward_model is not None:
            self.reward_model = self.reward_model.eval().to(self.device)

        # Action normalization statistics used before world-model input.
        self.action_stats = self._load_action_stats()

        # Dataset images are in [0, 1], while Ctrl-World expects [-1, 1].
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        # State caches used during rollout.
        self.current_obs = None
        self.current_wrist_obs = None
        self.current_extra_view_obs = None
        self.current_latent = None
        self.history_latents = None
        self.history_latent_bank = None
        self.current_states = None
        self.current_policy_states = None
        self.joint_fk_to_ctrl_alignment = None
        self.initial_image_state_representation = str(
            cfg.get("initial_image_state_representation", "auto")
        ).strip().lower()
        if self.initial_image_state_representation not in {"auto", "joint", "eef"}:
            raise ValueError(
                "initial_image_state_representation must be one of "
                "{'auto', 'joint', 'eef'}, got "
                f"{self.initial_image_state_representation!r}"
            )
        self.initial_joint_state_dataset = self._build_initial_joint_state_dataset(cfg)
        self.action_history = torch.zeros(
            self.num_envs,
            self.num_history,
            self.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_history_bank = None

        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        self._is_offloaded = False
        if not torch.is_tensor(self.elapsed_steps):
            self.elapsed_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )

    def _validate_cross_rank_metadata(
        self, metadata: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
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
        )
        normalized = {}
        for field_name in required:
            value = metadata.get(field_name)
            if value is None:
                raise ValueError(f"missing cross-rank metadata field {field_name}")
            value = torch.as_tensor(value, dtype=torch.int64).reshape(-1).cpu()
            if value.numel() != self.num_envs:
                raise ValueError(
                    f"cross-rank metadata {field_name} has {value.numel()} rows; "
                    f"expected {self.num_envs}"
                )
            normalized[field_name] = value

        # Semantic-topology fields are optional for the legacy cross-rank
        # layout, but production runs carry them end to end.  Preserve them
        # here instead of silently dropping ``reset_episode`` and then
        # re-deriving a potentially different dataset row from reset_seed.
        for field_name in (
            "update_id",
            "logical_round_id",
            "physical_wave_id",
            "group_slot",
            "chunk_id",
            "reset_episode",
            "seed_nonce",
            "shuffle_seed",
        ):
            value = metadata.get(field_name)
            if value is None:
                continue
            value = torch.as_tensor(value, dtype=torch.int64).reshape(-1).cpu()
            if value.numel() != self.num_envs:
                raise ValueError(
                    f"cross-rank metadata {field_name} has {value.numel()} rows; "
                    f"expected {self.num_envs}"
                )
            normalized[field_name] = value

        shared_fields = ["global_group_id", "reset_seed", "ctrl_world_noise_seed"]
        if not self._per_member_vision_seed:
            shared_fields.append("vision_noise_seed")
        for start in range(0, self.num_envs, self.group_size):
            stop = start + self.group_size
            for field_name in shared_fields:
                if normalized[field_name][start:stop].unique().numel() != 1:
                    raise ValueError(
                        f"cross-rank group has inconsistent {field_name}"
                    )
            if normalized["action_noise_seed"][start:stop].unique().numel() != (
                stop - start
            ):
                raise ValueError("cross-rank group has duplicate action_noise_seed")
            if self._per_member_vision_seed:
                if not torch.equal(
                    normalized["vision_noise_seed"][start:stop],
                    normalized["action_noise_seed"][start:stop],
                ):
                    raise ValueError(
                        "cross-rank group has vision_noise_seed != "
                        "action_noise_seed under per_member_vision_seed"
                    )
                namespace_fields = ("reset_seed", "action_noise_seed", "ctrl_world_noise_seed")
            else:
                namespace_fields = (
                    "reset_seed",
                    "vision_noise_seed",
                    "action_noise_seed",
                    "ctrl_world_noise_seed",
                )
            namespace_values = {
                int(normalized[field_name][start].item())
                for field_name in namespace_fields
            }
            if len(namespace_values) != len(namespace_fields):
                raise ValueError("cross-rank seed namespace collision detected")
        return normalized

    def set_cross_rank_rollout_context(
        self, metadata: dict[str, torch.Tensor]
    ) -> None:
        """Install a new rollout context and choose restart-safe reset ids."""
        self._cross_rank_metadata = self._validate_cross_rank_metadata(metadata)
        self._common_noise_call_index = 0
        self._update_cross_rank_reset_state_ids()

    def set_cross_rank_chunk_context(
        self, metadata: dict[str, torch.Tensor]
    ) -> None:
        """Update per-chunk model-noise seeds without changing reset state."""
        normalized = self._validate_cross_rank_metadata(metadata)
        if self._cross_rank_metadata is not None and not torch.equal(
            normalized["rollout_uid"], self._cross_rank_metadata["rollout_uid"]
        ):
            raise ValueError("cross-rank rollout_uid changed within one rollout")
        self._cross_rank_metadata = normalized

    def _update_cross_rank_reset_state_ids(self) -> None:
        assert self._cross_rank_metadata is not None
        if self.specific_reset_id is not None:
            # Fixed reset ids are already rank-independent.
            metadata = self._cross_rank_metadata
            self._cross_rank_metadata = None
            try:
                self.update_reset_state_ids()
            finally:
                self._cross_rank_metadata = metadata
            return
        allowed_ids = self._allowed_reset_episode_ids()
        if not allowed_ids:
            raise ValueError("no reset states remain after applying exclude_reset_ids")
        reset_ids = []
        reset_seeds = self._cross_rank_metadata["reset_seed"]
        semantic_episodes = self._cross_rank_metadata.get("reset_episode")
        for start in range(0, self.num_envs, self.group_size):
            if semantic_episodes is None:
                reset_id = allowed_ids[
                    int(reset_seeds[start].item()) % len(allowed_ids)
                ]
            else:
                group_episodes = semantic_episodes[start : start + self.group_size]
                if group_episodes.unique().numel() != 1:
                    raise ValueError(
                        "cross-rank group has inconsistent reset_episode"
                    )
                reset_id = int(group_episodes[0].item())
                if reset_id not in allowed_ids:
                    raise ValueError(
                        "semantic reset_episode is unavailable or excluded: "
                        f"{reset_id}; allowed={allowed_ids}"
                    )
            reset_ids.extend([reset_id] * self.group_size)
        self.reset_state_ids = torch.tensor(
            reset_ids, dtype=torch.long, device=self.device
        )

    def _allowed_reset_episode_ids(self) -> list[int]:
        """Return the explicit reset allowlist after exclusions."""
        if getattr(self, "_episode_contract", None) is not None:
            assert_duck_episode_split_unchanged(self._episode_contract)
        configured = (
            self.reset_episode_ids
            if self.reset_episode_ids is not None
            else tuple(range(len(self.dataset)))
        )
        excluded = set(self.exclude_reset_ids)
        return [episode_id for episode_id in configured if episode_id not in excluded]

    def _normalize_specific_reset_ids(self) -> list[int]:
        """Validate fixed reset ids without legacy modulo remapping."""
        if hasattr(self.specific_reset_id, "__iter__") and not isinstance(
            self.specific_reset_id, (str, bytes)
        ):
            ids = [int(value) for value in self.specific_reset_id]
        else:
            ids = [int(self.specific_reset_id)]
        if not ids:
            raise ValueError("specific_reset_id list must not be empty")
        if self.reset_episode_ids is None:
            ids = [episode_id % len(self.dataset) for episode_id in ids]
        allowed = set(self._allowed_reset_episode_ids())
        invalid = [episode_id for episode_id in ids if episode_id not in allowed]
        if invalid:
            raise ValueError(
                "specific_reset_id contains episodes outside reset_episode_ids: "
                f"{invalid}; split={self.reset_split!r}"
            )
        return ids

    def set_post_update_eval_context(
        self,
        global_step: int,
        episode_ids: list[int] | tuple[int, ...],
        *,
        base_seed: int = 42,
        seeds: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """Pin eval resets and Ctrl-World noise for one post-update pass."""
        episode_ids = tuple(int(value) for value in episode_ids)
        if len(episode_ids) != self.num_envs:
            raise ValueError(
                "post-update eval requires one episode per local env: "
                f"{len(episode_ids)} != {self.num_envs}"
            )
        if seeds is None:
            seeds = tuple(
                int(base_seed) + 8 * (int(global_step) - 1) + index
                for index in range(len(episode_ids))
            )
        else:
            seeds = tuple(int(value) for value in seeds)
        if len(seeds) != len(episode_ids) or any(seed < 0 for seed in seeds):
            raise ValueError(
                "post-update eval seeds must be non-negative and per-episode"
            )

        self.specific_reset_id = episode_ids
        self._fixed_denoise_seeds = seeds
        self._post_update_eval_denoise_seed_log = []
        self.fixed_denoise_seed = seeds[0] if len(set(seeds)) == 1 else None
        self.common_noise_seed = seeds[0]
        self._generator.manual_seed(seeds[0])
        self._common_noise_rollout_index = -1
        self._common_noise_call_index = 0
        self.is_start = True
        self.update_reset_state_ids()
        self._post_update_eval_context = {
            "global_step": int(global_step),
            "episode_ids": episode_ids,
            "seeds": seeds,
            "split": self.reset_split,
        }
        return dict(self._post_update_eval_context)

    def get_post_update_eval_rng_manifest(self) -> dict[str, object]:
        """Return the actual Ctrl-World denoise seeds used by this eval pass."""
        if self._fixed_denoise_seeds is None:
            return {}
        return {
            "schema_version": 1,
            "seed_derivation_version": "blake2b-int63-v2",
            "namespace": "post_update_eval_ctrl_world_denoise",
            "base_seeds": list(self._fixed_denoise_seeds),
            "denoise_calls": [
                {
                    "call_index": int(call["call_index"]),
                    "derived_seeds": list(call["derived_seeds"]),
                }
                for call in self._post_update_eval_denoise_seed_log
            ],
        }

    def get_success_model_provenance(self) -> dict[str, dict[str, str]]:
        """Return routed success-model paths and SHA-256 values."""
        provenance = getattr(self.reward_model, "provenance", None)
        return provenance() if callable(provenance) else {}

    def _build_initial_joint_state_dataset(self, cfg):
        joint_state_path = self.ctrl_world_cfg.get(
            "initial_joint_state_path", cfg.get("initial_joint_state_path", None)
        )
        if joint_state_path is None:
            return None
        return LeRobotBookTrajectoryDatasetWrapper(
            joint_state_path,
            camera_keys=cfg.get("initial_image_camera_keys", None),
        )

    def _read_initial_joint_state(self, dataset_index: int) -> Optional[np.ndarray]:
        if self.initial_joint_state_dataset is None:
            return None
        episode = self.initial_joint_state_dataset.episodes[int(dataset_index)]
        first_frame = self.initial_joint_state_dataset._get_first_frame(episode)
        return np.asarray(first_frame["observation.state"], dtype=np.float32).reshape(-1)

    def _build_dataset(self, cfg):
        """Build the reset-state dataset."""
        dataset_type = str(cfg.get("initial_image_dataset_type", "npy")).lower()
        if dataset_type == "lerobot":
            return LeRobotTrajectoryDatasetWrapper(cfg.initial_image_path)
        if dataset_type == "lerobot_book":
            return LeRobotBookTrajectoryDatasetWrapper(
                cfg.initial_image_path,
                camera_keys=cfg.get("initial_image_camera_keys", None),
            )
        if dataset_type == "npy":
            return NpyTrajectoryDatasetWrapper(
                cfg.initial_image_path, enable_kir=cfg.get("enable_kir", False)
            )
        raise ValueError(f"Unsupported initial_image_dataset_type: {dataset_type}")
        
    def _to_torch_dtype(self, dtype: str) -> torch.dtype:
        """Map a config dtype string to a torch dtype."""
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        key = str(dtype).lower()
        if key not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype_map[key]

    def _install_diffusers_text_to_video_compat(self):
        """Alias the legacy diffusers text-to-video module used by Ctrl-World."""
        legacy_pkg = "diffusers.pipelines.text_to_video_synthesis"
        legacy_mod = f"{legacy_pkg}.pipeline_text_to_video_synth"
        try:
            importlib.import_module(legacy_mod)
            return
        except ModuleNotFoundError as exc:
            if exc.name not in {legacy_pkg, legacy_mod}:
                raise

        deprecated_pkg = "diffusers.pipelines.deprecated.text_to_video_synthesis"
        deprecated_mod = f"{deprecated_pkg}.pipeline_text_to_video_synth"
        package = importlib.import_module(deprecated_pkg)
        module = importlib.import_module(deprecated_mod)
        sys.modules.setdefault(legacy_pkg, package)
        sys.modules.setdefault(legacy_mod, module)

    def _append_ctrl_world_dependency_path(self):
        """Append optional Ctrl-World dependency paths without shadowing this venv."""
        dependency_paths = []
        configured_path = self.ctrl_world_cfg.get("python_site_packages", None)
        if configured_path is not None:
            dependency_paths.append(configured_path)
        env_site_packages = os.environ.get("CTRL_WORLD_PYTHON_SITE_PACKAGES", None)
        if env_site_packages is not None:
            dependency_paths.append(env_site_packages)
        extra_pythonpath = os.environ.get("CTRL_WORLD_EXTRA_PYTHONPATH", None)
        if extra_pythonpath:
            dependency_paths.extend(extra_pythonpath.split(os.pathsep))
        diffsynth_path = os.environ.get("DIFFSYNTH_PATH", None)
        if diffsynth_path:
            dependency_paths.append(diffsynth_path)

        for dependency_path in dependency_paths:
            if not dependency_path:
                continue
            resolved_path = str(Path(dependency_path).expanduser().resolve())
            if not os.path.isdir(resolved_path):
                raise ValueError(
                    f"Ctrl-World dependency path does not exist: {resolved_path}"
                )
            if resolved_path not in sys.path:
                sys.path.append(resolved_path)

    def _import_ctrl_world_modules(self):
        """Dynamically import Ctrl-World modules from an external repo path."""
        self._install_diffusers_text_to_video_compat()
        self._append_ctrl_world_dependency_path()
        ctrl_world_repo_path = self.ctrl_world_cfg.get("ctrl_world_repo_path", None)
        if ctrl_world_repo_path is None:
            ctrl_world_repo_path = os.environ.get("CTRL_WORLD_PATH", None)

        if ctrl_world_repo_path is None:
            raise ValueError(
                "ctrl_world_repo_path is required in env.ctrl_world_cfg or CTRL_WORLD_PATH env variable"
            )

        ctrl_world_repo_path = str(Path(ctrl_world_repo_path).expanduser().resolve())
        if not os.path.isdir(ctrl_world_repo_path):
            raise ValueError(f"ctrl_world_repo_path does not exist: {ctrl_world_repo_path}")

        if ctrl_world_repo_path not in sys.path:
            sys.path.insert(0, ctrl_world_repo_path)

        try:
            from models.ctrl_world import CrtlWorld  # type: ignore
            from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import Ctrl-World modules. Please ensure Ctrl-World dependencies "
                "are installed and ctrl_world_repo_path points to the repository root."
            ) from e

        return CrtlWorld, CtrlWorldDiffusionPipeline

    def _build_ctrl_world_model(self):
        """Instantiate Ctrl-World and load checkpoint weights."""
        CrtlWorld, CtrlWorldDiffusionPipeline = self._import_ctrl_world_modules()
        self.ctrl_world_pipeline_cls = CtrlWorldDiffusionPipeline

        args = _CtrlWorldArgs(
            svd_model_path=self.ctrl_world_cfg.svd_model_path,
            clip_model_path=self.ctrl_world_cfg.clip_model_path,
            action_dim=self.action_dim,
            num_history=self.num_history,
            num_frames=self.ctrl_world_chunk,
            text_cond=self.text_cond,
        )

        model = CrtlWorld(args)

        ckpt_path = self.ctrl_world_cfg.ckpt_path
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Ctrl-World checkpoint path does not exist: {ckpt_path}")

        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(raw_ckpt)
        model.load_state_dict(state_dict, strict=True)
        return model

    def _extract_state_dict(self, ckpt_obj):
        """Extract a plain state_dict from supported checkpoint wrappers."""
        state_dict = ckpt_obj
        if isinstance(state_dict, dict):
            for key in ["state_dict", "model_state_dict", "model", "module"]:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        if not isinstance(state_dict, dict):
            raise ValueError("Invalid Ctrl-World checkpoint format.")

        if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        return state_dict

    def _load_reward_model(self):
        """Load a reward model for rewards or an explicit diagnostic."""
        routed_reward_cfg = self.ctrl_world_cfg.get("reward_models", None)
        if routed_reward_cfg is not None:
            from rlinf.rewards.resnet_reward_model import (
                ColorRoutedResnetRewModel,
            )

            checkpoints = {}
            checkpoint_sha256 = {}
            reward_cfg = self.ctrl_world_cfg.get("reward_model", {})
            require_camera_metadata = _as_bool(
                reward_cfg.get("require_camera_metadata", False)
            )
            for color, model_cfg in routed_reward_cfg.items():
                checkpoint_path = Path(
                    os.path.expanduser(str(model_cfg.from_pretrained))
                )
                if checkpoint_path.is_dir():
                    checkpoint_path = checkpoint_path / str(
                        model_cfg.get("artifact_name", "resnet_rm.pth")
                    )
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Routed reward-model checkpoint does not exist: {checkpoint_path}"
                    )
                if require_camera_metadata:
                    metadata_path = checkpoint_path.parent / "run_config.json"
                    if not metadata_path.is_file():
                        raise FileNotFoundError(
                            "Routed reward model requires camera metadata, but "
                            f"{metadata_path} does not exist"
                        )
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid reward-model metadata {metadata_path}: {exc}"
                        ) from exc
                    metadata_camera_key = str(metadata.get("camera_key", "")).strip()
                    if metadata_camera_key != self.reward_model_camera_key:
                        raise ValueError(
                            f"Routed reward model {color!r} was trained for "
                            f"camera_key={metadata_camera_key!r}, but runtime is "
                            f"configured for {self.reward_model_camera_key!r}"
                        )
                checkpoints[str(color)] = checkpoint_path
                digest = str(model_cfg.get("sha256", "")).strip()
                if not digest:
                    raise ValueError(
                        f"Routed reward model {color!r} requires configured sha256."
                    )
                checkpoint_sha256[str(color)] = digest
            return ColorRoutedResnetRewModel(
                checkpoints=checkpoints,
                color_ranges=self.ctrl_world_cfg.get(
                    "reward_model_episode_color_ranges", None
                ),
                checkpoint_sha256=checkpoint_sha256,
            )

        reward_cfg = self.ctrl_world_cfg.get(
            "reward_model", self.cfg.get("reward_model", None)
        )
        diagnostic_only = bool(
            self.ctrl_world_cfg.get("reward_model_diagnostic_only", False)
        )
        if (
            "reward_source" in self.cfg
            and self.reward_source != "success_model"
            and not diagnostic_only
        ):
            return None
        if reward_cfg is None or reward_cfg.get("type", None) in {None, "none", "None"}:
            return None

        if reward_cfg.type == "ColorRoutedResnetRewModel":
            from rlinf.rewards.resnet_reward_model import (
                ColorRoutedResnetRewModel,
            )

            rew_model = ColorRoutedResnetRewModel(
                checkpoints=reward_cfg.checkpoints,
                color_ranges=reward_cfg.get("episode_color_ranges", None),
                checkpoint_sha256=reward_cfg.get("checkpoint_sha256", None),
            )
            return rew_model

        checkpoint_path = os.path.expanduser(str(reward_cfg.from_pretrained))
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(
                checkpoint_path, str(reward_cfg.get("artifact_name", "resnet_rm.pth"))
            )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Reward-model checkpoint does not exist: {checkpoint_path}"
            )

        if reward_cfg.type == "ResnetRewModel":
            if str(reward_cfg.get("loader", "auto")) == "rlinf_local":
                from rlinf.rewards.resnet_reward_model import ResnetRewModel
            else:
                try:
                    from diffsynth.models.reward_model import ResnetRewModel
                except ModuleNotFoundError as exc:
                    if exc.name != "diffsynth":
                        raise
                    from rlinf.rewards.resnet_reward_model import ResnetRewModel

            rew_model = ResnetRewModel(checkpoint_path)
        elif reward_cfg.type == "TaskEmbedResnetRewModel":
            try:
                from diffsynth.models.reward_model import TaskEmbedResnetRewModel
            except ModuleNotFoundError as exc:
                if exc.name == "diffsynth":
                    raise ModuleNotFoundError(
                        "TaskEmbedResnetRewModel requires diffsynth; only "
                        "ResnetRewModel has a repo-local fallback."
                    ) from exc
                raise

            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=checkpoint_path,
                task_suite_name=self.cfg.task_suite_name,
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_cfg.type}")

        rew_model.eval()
        for parameter in rew_model.parameters():
            parameter.requires_grad_(False)
        return rew_model

    def _load_action_stats(self):
        """Load action quantile statistics for normalization to [-1, 1]."""
        stats_path = self.ctrl_world_cfg.get("data_stat_path", None)
        if stats_path is None or not os.path.exists(stats_path):
            raise ValueError(f"Ctrl-World data_stat_path does not exist: {stats_path}")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        q01_key = "state_01"
        q99_key = "state_99"
        if q01_key not in stats or q99_key not in stats:
            raise ValueError(
                f"Expected {q01_key!r} and {q99_key!r} in {stats_path}, "
                f"got keys: {list(stats.keys())}"
            )

        q01 = np.asarray(stats[q01_key], dtype=np.float32)
        q99 = np.asarray(stats[q99_key], dtype=np.float32)
        return {"q01": q01, "q99": q99}

    def _load_learned_action_to_eef_adapter(self) -> None:
        if self.action_to_eef_adapter == "joint_learned_fk":
            adapter_path = self.ctrl_world_cfg.get(
                "joint_action_to_state_adapter_path",
                self.ctrl_world_cfg.get(
                    "action_to_eef_adapter_path",
                    self.ctrl_world_cfg.get("eef_adapter_path", None),
                ),
            )
            adapter_path_key = "joint_action_to_state_adapter_path"
        else:
            adapter_path = self.ctrl_world_cfg.get(
                "action_to_eef_adapter_path",
                self.ctrl_world_cfg.get("eef_adapter_path", None),
            )
            adapter_path_key = "action_to_eef_adapter_path"

        if not adapter_path:
            raise ValueError(
                f"{self.action_to_eef_adapter!r} requires {adapter_path_key}."
            )

        ckpt = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
        config_dict = ckpt.get("config", {})
        config = EEFAdapterConfig(**config_dict)

        model = ActionToEEFPoseAdapter(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval().to(self.device)
        self.learned_eef_adapter = model

        stats = ckpt.get("stats", {})

        required_stats = [
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
            "delta_mean",
            "delta_std",
        ]
        missing = [name for name in required_stats if name not in stats]
        if missing:
            raise ValueError(
                f"Learned EEF adapter checkpoint is missing stats: {missing}"
            )

        def _stat_tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(
                stats[name], device=self.device, dtype=torch.float32
            ).view(1, -1)

        self.eef_adapter_state_mean = _stat_tensor("state_mean")
        self.eef_adapter_state_std = torch.clamp(_stat_tensor("state_std"), min=1e-6)
        self.eef_adapter_action_mean = _stat_tensor("action_mean")
        self.eef_adapter_action_std = torch.clamp(_stat_tensor("action_std"), min=1e-6)
        self.eef_adapter_delta_mean = _stat_tensor("delta_mean")
        self.eef_adapter_delta_std = torch.clamp(_stat_tensor("delta_std"), min=1e-6)

    def _normalize_action(self, actions: np.ndarray) -> np.ndarray:
        """Normalize raw actions with dataset q01/q99 stats and clip to [-1, 1]."""
        q01 = self.action_stats["q01"]
        q99 = self.action_stats["q99"]

        action_dim = min(actions.shape[-1], q01.shape[-1], q99.shape[-1])
        actions_norm = actions.copy()
        actions_norm[..., :action_dim] = (
            2
            * (
                (actions_norm[..., :action_dim] - q01[:action_dim])
                / (q99[:action_dim] - q01[:action_dim] + 1e-8)
            )
            - 1
        )
        return np.clip(actions_norm, -1.0, 1.0)

    def _apply_adapter_gripper_hard_clamp(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optionally make LIBERO gripper conditions follow the command sign.

        Ctrl-World's state-conditioned LIBERO model was trained on absolute
        gripper opening, where low values are closed and high values are open.
        The policy emits a command sign, so this debug/rollout option can force
        the adapter output to the dataset close/open range before normalization.
        """
        if not self.adapter_gripper_hard_clamp or states.shape[-1] < 7 or actions.shape[-1] < 7:
            return states

        q01 = torch.as_tensor(
            self.action_stats["q01"], device=states.device, dtype=states.dtype
        )
        q99 = torch.as_tensor(
            self.action_stats["q99"], device=states.device, dtype=states.dtype
        )
        if q01.numel() >= 7 and q99.numel() >= 7:
            close_value = q01[-1]
            open_value = q99[-1]
        else:
            close_value = torch.as_tensor(
                self.adapter_gripper_close, device=states.device, dtype=states.dtype
            )
            open_value = torch.as_tensor(
                self.adapter_gripper_open, device=states.device, dtype=states.dtype
            )

        gripper = states[..., 6:7].clamp(
            min=torch.minimum(close_value, open_value),
            max=torch.maximum(close_value, open_value),
        )
        if self.adapter_gripper_hard_mode == "clip_only":
            states = states.clone()
            states[..., 6:7] = gripper
            return states
        threshold = self.adapter_gripper_hard_threshold
        if self.adapter_gripper_hard_mode == "policy_delta_state":
            if current_state is None:
                raise RuntimeError(
                    "adapter_gripper_hard_mode='policy_delta_state' requires current_state"
                )
            grippers = []
            current_gripper = current_state[:, 6:7].to(device=states.device, dtype=states.dtype)
            current_gripper = current_gripper.clamp(
                min=torch.minimum(close_value, open_value),
                max=torch.maximum(close_value, open_value),
            )
            for step in range(actions.shape[1]):
                gripper_score = current_gripper + actions[:, step, -1:].to(states.dtype)
                next_gripper = torch.where(
                    gripper_score > threshold,
                    close_value.expand_as(current_gripper),
                    torch.where(
                        gripper_score < -threshold,
                        open_value.expand_as(current_gripper),
                        current_gripper,
                    ),
                )
                grippers.append(next_gripper)
                current_gripper = next_gripper
            gripper = torch.stack(grippers, dim=1)
        else:
            gripper_cmd = actions[..., -1:]
            gripper = torch.where(gripper_cmd > threshold, close_value.expand_as(gripper), gripper)
            gripper = torch.where(gripper_cmd < -threshold, open_value.expand_as(gripper), gripper)

        states = states.clone()
        states[..., 6:7] = gripper
        return states

    def _align_adapter_states_for_ctrl_world(
        self,
        current_state: torch.Tensor,
        future_states: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split adapter-predicted states into Ctrl-World conditions and policy state.

        The adapter predicts states after each policy action, i.e. `[s_{t+1}, ..., s_{t+H}]`.
        Ctrl-World was trained with the current frame included in the frame-level condition,
        so the generated chunk should be conditioned by `[s_t, ..., s_{t+H-1}]`.
        The policy-side state cache advances to the last frame conditioned in this
        Ctrl-World call, so it stays aligned with the last generated latent.

        Example with num_history=6 and chunk=7:
        - `action_history` already carries 6 past conditions before this function.
        - `future_states` from the adapter is
          `[s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4}, s_{t+5}, s_{t+6}, s_{t+7}]`.
        - This function returns the 7 current-chunk Ctrl-World conditions
          `[s_t, s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4}, s_{t+5}, s_{t+6}]`.
        - The full Ctrl-World action/state condition length is therefore
          `6 past + 7 current-chunk = 13`; `s_{t+7}` is not fed in this call.
        """
        if future_states.ndim != 3:
            raise ValueError(
                f"Expected future_states to have shape [B, H, D], got {future_states.shape}"
            )
        future_states = self._maybe_resample_adapter_states(current_state, future_states)
        if future_states.shape[1] != self.ctrl_world_chunk:
            raise ValueError(
                f"Expected {self.ctrl_world_chunk} adapter states, got {future_states.shape[1]}"
            )

        condition_dim = min(current_state.shape[-1], future_states.shape[-1])
        current_condition = current_state[:, :condition_dim].to(
            device=future_states.device, dtype=future_states.dtype
        )
        full_ctrl_world_condition = torch.cat(
            [current_condition[:, None, :], future_states[:, :, :condition_dim]], dim=1
        )
        ctrl_world_condition = full_ctrl_world_condition[:, :-1, :]
        if self.ctrl_world_internal_rollout:
            next_policy_state = full_ctrl_world_condition[:, -1, :]
            self._last_ctrl_world_endpoint_condition_np = (
                full_ctrl_world_condition.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        else:
            next_policy_state = ctrl_world_condition[:, -1, :]
            self._last_ctrl_world_endpoint_condition_np = None
        return (
            ctrl_world_condition.detach().cpu().numpy().astype(np.float32, copy=False),
            next_policy_state.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def _maybe_resample_adapter_states(
        self, current_state: torch.Tensor, future_states: torch.Tensor
    ) -> torch.Tensor:
        if not self.action_fps_resample:
            return future_states
        return resample_eef_state_sequence_by_fps(
            current_state,
            future_states,
            source_fps=self.cosmos_action_fps,
            target_fps=self.ctrl_condition_fps,
            target_steps=self.ctrl_world_chunk,
        )

    def _policy_frame_indices(self, device=None) -> torch.Tensor:
        device = self.device if device is None else device
        if self.ctrl_world_chunk == self.chunk and not self.action_fps_resample:
            return torch.arange(self.chunk, device=device, dtype=torch.long)
        policy_times = (
            torch.arange(1, self.chunk + 1, device=device, dtype=torch.float32)
            / float(self.cosmos_action_fps)
        )
        indices = torch.round(policy_times * float(self.ctrl_condition_fps)).to(torch.long) - 1
        return torch.clamp(indices, min=0, max=self.ctrl_world_chunk - 1)

    def _select_policy_aligned_sequence(self, sequence: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return sequence.index_select(dim, self._policy_frame_indices(sequence.device))

    def _latest_policy_aligned_obs(self, obs: torch.Tensor) -> torch.Tensor:
        latest_ctrl = obs.permute(0, 3, 1, 2, 4, 5)[:, -self.ctrl_world_chunk :, :, :, :, :]
        return self._select_policy_aligned_sequence(latest_ctrl, dim=1)

    def _convert_policy_actions_to_ctrl_world(
        self, actions_np: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert policy outputs to Ctrl-World conditions and optional next state."""
        if self.policy_action_type not in {"delta_eef", "absolute_eef", "joint_absolute"}:
            raise ValueError(
                f"Unsupported policy_action_type: {self.policy_action_type}. "
                "Expected one of {'delta_eef', 'absolute_eef', 'joint_absolute'}."
            )

        if self.policy_action_type == "joint_absolute":
            if self.joint_action_adapter != "fk":
                raise ValueError(
                    "policy_action_type='joint_absolute' requires joint_action_adapter='fk'"
                )
            if self.current_states is None:
                raise RuntimeError("current_states must be initialized before joint conversion")
            current_state = state_to_ctrl_world_eef_state(
                self.current_states.to(device=self.device, dtype=torch.float32)
            )
            joint_actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.float32)
            if self.action_to_eef_adapter == "joint_learned_fk":
                if self.current_policy_states is None:
                    raise RuntimeError("current_policy_states must be initialized before joint learned adapter conversion")
                if self.learned_eef_adapter is None:
                    raise RuntimeError("learned_eef_adapter is not loaded")
                current_joint_state = self._align_policy_state_dim(
                    self.current_policy_states
                )[:, : self.eef_adapter_state_mean.shape[-1]]
                joint_actions_for_adapter = joint_actions[
                    ..., : self.eef_adapter_action_mean.shape[-1]
                ]
                current_joint_state_norm = (
                    current_joint_state - self.eef_adapter_state_mean
                ) / self.eef_adapter_state_std
                joint_actions_norm = (
                    joint_actions_for_adapter - self.eef_adapter_action_mean[:, None, :]
                ) / self.eef_adapter_action_std[:, None, :]
                with torch.no_grad():
                    pred_joint_delta_norm = self.learned_eef_adapter(
                        current_joint_state_norm, joint_actions_norm
                    )
                    pred_joint_delta = (
                        pred_joint_delta_norm * self.eef_adapter_delta_std[:, None, :]
                        + self.eef_adapter_delta_mean[:, None, :]
                    )
                    future_joint_states = current_joint_state[:, None, :] + pred_joint_delta
                    eef_states = ur5_joint_actions_to_eef_states(
                        future_joint_states,
                        branch_reference=current_state,
                        tcp_xyz=self.joint_fk_tcp_xyz,
                        tcp_rotvec=self.joint_fk_tcp_rotvec,
                    )
                    if self.joint_fk_to_ctrl_alignment is None:
                        raise RuntimeError("joint FK alignment must be initialized before rollout")
                    eef_states = apply_eef_rigid_alignment(
                        eef_states,
                        self.joint_fk_to_ctrl_alignment,
                        branch_reference=current_state,
                    )
                    eef_states = self._apply_adapter_gripper_hard_clamp(
                        eef_states, joint_actions_for_adapter, current_state
                    )
                ctrl_world_condition_np, next_policy_eef_np = (
                    self._align_adapter_states_for_ctrl_world(current_state, eef_states)
                )
                policy_timeline_joint_states = future_joint_states
                if self.action_fps_resample:
                    policy_timeline_joint_states = resample_joint_state_sequence_by_fps(
                        current_joint_state,
                        future_joint_states,
                        source_fps=self.cosmos_action_fps,
                        target_fps=self.ctrl_condition_fps,
                        target_steps=self.ctrl_world_chunk,
                    )
                full_policy_timeline = torch.cat(
                    [current_joint_state[:, None, :], policy_timeline_joint_states],
                    dim=1,
                )
                next_policy_joint = (
                    full_policy_timeline[:, -1, :]
                    if self.ctrl_world_internal_rollout
                    else full_policy_timeline[:, -2, :]
                )
                self.current_policy_states = next_policy_joint.detach().to(
                    device=self.device, dtype=torch.float32
                )
                return ctrl_world_condition_np, next_policy_eef_np

            eef_actions = ur5_joint_actions_to_eef_states(
                joint_actions,
                branch_reference=current_state,
                tcp_xyz=self.joint_fk_tcp_xyz,
                tcp_rotvec=self.joint_fk_tcp_rotvec,
            )
            if self.joint_fk_to_ctrl_alignment is None:
                raise RuntimeError("joint FK alignment must be initialized before rollout")
            eef_actions = apply_eef_rigid_alignment(
                eef_actions,
                self.joint_fk_to_ctrl_alignment,
                branch_reference=current_state,
            )
            actions_np = eef_actions.detach().cpu().numpy().astype(np.float32, copy=False)
            if self.action_to_eef_adapter == "none":
                return self._align_adapter_states_for_ctrl_world(current_state, eef_actions)

        # Ctrl-World is state-conditioned, while the policy usually emits delta
        # EEF actions. The adapter converts each action chunk into the EEF state
        # trajectory used as Ctrl-World conditioning.
        return self._action_to_eef_adapter_dispatch[self.action_to_eef_adapter](actions_np)

    def _convert_via_analytic(
        self, actions_np: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.current_states is None:
            raise RuntimeError("current_states must be initialized before analytic adapter conversion")
        actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.float32)
        states = analytic_libero_delta_actions_to_eef_states(
            self.current_states.to(device=self.device, dtype=torch.float32),
            actions,
            position_scale=self.adapter_position_scale,
            rotation_scale=self.adapter_rotation_scale,
            gripper_open=self.adapter_gripper_open,
            gripper_close=self.adapter_gripper_close,
        )
        current_state = state_to_ctrl_world_eef_state(
            self.current_states.to(device=self.device, dtype=torch.float32)
        )
        states = self._apply_adapter_gripper_hard_clamp(states, actions, current_state)
        return self._align_adapter_states_for_ctrl_world(current_state, states)

    def _convert_via_analytic_ur5_rot6d(
        self, actions_np: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.current_states is None:
            raise RuntimeError("current_states must be initialized before analytic_ur5_rot6d adapter conversion")
        # SE(3) composition must stay in float32 even when the caller uses
        # autocast for diffusion inference. Casting inputs alone is insufficient:
        # matmul/einsum would round every incremental rotation to bfloat16,
        # accumulating centimetres of position drift over a demonstration.
        # Keep FPS resampling in the same full-precision region.
        with torch.autocast(device_type=self.device.type, enabled=False):
            actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.float32)
            states = analytic_ur5_rot6d_delta_actions_to_eef_states(
                self.current_states.to(device=self.device, dtype=torch.float32),
                actions,
                translation_gain=self.adapter_translation_gain,
                rotation_gain=self.adapter_rotation_gain,
                translation_frame=self.adapter_translation_frame,
                translation_sign=self.adapter_translation_sign,
                rotation_mode=self.adapter_rotation_mode,
            )
            current_state = state_to_ctrl_world_eef_state(
                self.current_states.to(device=self.device, dtype=torch.float32)
            )
            states = self._apply_adapter_gripper_hard_clamp(states, actions, current_state)
            return self._align_adapter_states_for_ctrl_world(current_state, states)


    def _convert_via_learned(
        self, actions_np: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.current_states is None:
            raise RuntimeError("current_states must be initialized before learned adapter conversion")
        if self.learned_eef_adapter is None:
            raise RuntimeError("learned_eef_adapter is not loaded")
        actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.float32)
        current_state = state_to_ctrl_world_eef_state(
            self.current_states.to(device=self.device, dtype=torch.float32)
        )
        current_state = current_state[:, : self.eef_adapter_state_mean.shape[-1]]
        actions = actions[..., : self.eef_adapter_action_mean.shape[-1]]

        # The learned adapter was trained with normalized current states,
        # normalized policy actions, and normalized EEF deltas.
        current_state_norm = (
            current_state - self.eef_adapter_state_mean
        ) / self.eef_adapter_state_std
        actions_norm = (
            actions - self.eef_adapter_action_mean[:, None, :]
        ) / self.eef_adapter_action_std[:, None, :]
        with torch.no_grad():
            pred_delta_norm = self.learned_eef_adapter(current_state_norm, actions_norm)
            pred_delta = (
                pred_delta_norm * self.eef_adapter_delta_std[:, None, :]
                + self.eef_adapter_delta_mean[:, None, :]
            )
            # The checkpoint predicts deltas from the current EEF state;
            # convert those deltas into absolute EEF states for Ctrl-World.
            states = compose_learned_eef_delta_to_states(current_state, pred_delta)
            states = self._apply_adapter_gripper_hard_clamp(states, actions, current_state)
        return self._align_adapter_states_for_ctrl_world(current_state, states)

    def _convert_via_none(
        self, actions_np: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.policy_action_type != "absolute_eef":
            raise ValueError(
                "Ctrl-World state conditioning requires an action_to_eef_adapter "
                "for non-absolute EEF policy actions."
            )
        return actions_np.astype(np.float32, copy=True), None

    def _condition_from_current_states(self, current_states: torch.Tensor) -> torch.Tensor:
        return state_to_ctrl_world_eef_state(current_states)[:, : self.action_dim]

    def _set_current_states_from_ctrl_world_condition(self, state_condition: torch.Tensor) -> None:
        """Update policy-side state cache from a 7D Ctrl-World EEF state condition."""
        if self.current_states is None:
            self.current_states = torch.zeros(
                (state_condition.shape[0], self.action_dim),
                device=self.device,
                dtype=torch.float32,
            )
        pose_dim = min(6, self.current_states.shape[1], state_condition.shape[1])
        self.current_states[:, :pose_dim] = state_condition[:, :pose_dim]
        if state_condition.shape[1] >= 7:
            gripper = state_condition[:, 6:7]
            if self.current_states.shape[1] >= 7:
                self.current_states[:, 6:7] = gripper
        if self.policy_action_type != "joint_absolute":
            self.current_policy_states = self.current_states.detach().clone()

    def _align_policy_state_dim(self, states: torch.Tensor) -> torch.Tensor:
        """Return policy-visible states with exactly `action_dim` features."""
        states = states.to(device=self.device, dtype=torch.float32)
        if states.shape[-1] == self.action_dim:
            return states
        if states.shape[-1] > self.action_dim:
            return states[..., : self.action_dim].contiguous()
        pad_width = self.action_dim - states.shape[-1]
        return F.pad(states, (0, pad_width))

    def _build_initial_action_history(
        self, current_states: torch.Tensor, num_reset_envs: int, history_length: int | None = None
    ) -> torch.Tensor:
        """Initialize action history in the same action convention as Ctrl-World training."""
        init_actions = torch.zeros(
            (num_reset_envs, self.action_dim), device=self.device, dtype=torch.float32
        )

        init_condition = self._condition_from_current_states(current_states)
        state_action_dim = min(init_condition.shape[1], self.action_dim)
        init_actions[:, :state_action_dim] = init_condition[:, :state_action_dim]

        history_length = self.num_history if history_length is None else int(history_length)
        return init_actions.unsqueeze(1).repeat(1, history_length, 1)

    def _init_metrics(self):
        self.elapsed_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.success_already_emitted = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.elapsed_steps[mask] = 0
            self.success_already_emitted[mask] = False
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.elapsed_steps[:] = 0
            self.success_already_emitted[:] = False
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        # The success-model source updates success_once from its explicit latch.
        # Dense legacy/video signals are never reinterpreted as success when
        # terminations are ignored.
        if getattr(self, "reward_source", "legacy_reward_model") == "success_model":
            episode_info["success_once"] = self.success_once.clone()
        elif not self.ignore_terminations:
            if isinstance(terminations, torch.Tensor):
                self.success_once = self.success_once | terminations
            else:
                terminations_tensor = torch.tensor(
                    terminations, device=self.device, dtype=torch.bool
                )
                self.success_once = self.success_once | terminations_tensor
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.to(torch.float32).clone()
        episode_info["reward"] = episode_info["return"] / torch.clamp(
            episode_info["episode_len"], min=1.0
        )
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Convert absolute rewards to relative rewards according to the config."""
        if getattr(self, "reward_source", "legacy_reward_model") == "success_model":
            return chunk_rewards
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        """Estimate success from the maximum frame reward within the chunk."""
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]
        success_estimated = max_reward_in_chunk >= success_threshold
        return success_estimated.to(self.device)

    def _build_chunk_terminations(self, chunk_rewards):
        """Build terminations, short-circuiting before success estimation."""
        terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        if getattr(self, "reward_source", "legacy_reward_model") == "success_model":
            return terminations
        if self.ignore_terminations:
            return terminations

        # Legacy opt-in path for configurations with a calibrated success
        # reward. Cosmos 7D/10D train and eval keep ignore_terminations=True,
        # so they never call the reward-based success estimator.
        terminations[:, -1] = self._estimate_success_from_rewards(chunk_rewards)
        return terminations

    def update_reset_state_ids(self):
        """Sample one reset episode id per group and replicate it within the group."""
        if self._cross_rank_metadata is not None and not self.cfg.is_eval:
            self._update_cross_rank_reset_state_ids()
            return
        if self.specific_reset_id is not None:
            ids = self._normalize_specific_reset_ids()
            if len(ids) > 1:
                # A list of held-out episode ids, e.g. a fixed eval split.
                # Cycle through it if num_group != len(list).
                reset_state_ids = torch.tensor(
                    [ids[i % len(ids)] for i in range(self.num_group)],
                    dtype=torch.long,
                )
            else:
                reset_state_ids = torch.full(
                    (self.num_group,),
                    ids[0],
                    dtype=torch.long,
                )
        else:
            allowed_ids = self._allowed_reset_episode_ids()
            if not allowed_ids:
                raise ValueError(
                    "reset allowlist is empty after exclusions; nothing remains "
                    f"for split {self.reset_split!r}."
                )
            num_allowed = len(allowed_ids)

            if (
                self.cfg.is_eval and not self.random_reset_state_ids
            ) or self.use_ordered_reset_state_ids:
                stride = self.total_num_processes * self.num_group
                start = (
                    self._ordered_reset_cursor * stride
                    + self.seed_offset * self.num_group
                )
                raw_ids = (
                    torch.arange(self.num_group, dtype=torch.long) + start
                ) % num_allowed
                self._ordered_reset_cursor += 1
            else:
                raw_ids = torch.randint(
                    low=0,
                    high=num_allowed,
                    size=(self.num_group,),
                    generator=self._generator,
                )
            reset_state_ids = torch.tensor(
                [allowed_ids[i] for i in raw_ids.tolist()], dtype=torch.long
            )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _sample_vae_latent_distribution(
        self, distribution, *, block_batch_size: int
    ) -> torch.Tensor:
        """Sample VAE latents with shared noise inside each GRPO group.

        latent_dist.sample() uses independent noise for every batch row.
        That made identical reset images produce different hidden Ctrl-World
        states before the controlled diffusion generator was even used.  A
        per-view encode concatenates camera views as contiguous batch blocks,
        so preserve that layout while sharing one epsilon per local group.
        """
        if (
            self._fixed_denoise_seeds is None
            and not self.common_random_numbers_within_group
        ):
            return distribution.sample()

        mean = distribution.mean
        std = distribution.std
        block_batch_size = int(block_batch_size)
        if (
            block_batch_size <= 0
            or block_batch_size % self.group_size != 0
            or int(mean.shape[0]) % block_batch_size != 0
        ):
            return distribution.sample()

        num_blocks = int(mean.shape[0]) // block_batch_size
        num_groups = block_batch_size // self.group_size
        noise = torch.empty_like(mean)
        for block_id in range(num_blocks):
            block_start = block_id * block_batch_size
            for group_id in range(num_groups):
                row_start = block_start + group_id * self.group_size
                generator = torch.Generator(device=mean.device)
                if self._fixed_denoise_seeds is not None:
                    eval_seed = self._fixed_denoise_seeds[
                        group_id * self.group_size
                    ]
                    seed = derive_stable_seed(
                        "post_update_eval_ctrl_world_vae",
                        int(eval_seed),
                        block_id,
                    )
                elif self._cross_rank_metadata is None:
                    seed = (
                        self.common_noise_seed
                        + int(self.seed_offset) * 1_000_000_000
                        + int(self._common_noise_rollout_index) * 1_000_000
                        + block_id * 10_000
                        + group_id
                        + 900_000_000
                    )
                else:
                    seed = derive_stable_seed(
                        "ctrl_world_vae",
                        int(
                            self._cross_rank_metadata["ctrl_world_noise_seed"][
                                group_id * self.group_size
                            ].item()
                        ),
                        block_id,
                    )
                generator.manual_seed(seed)
                epsilon = torch.randn(
                    mean[row_start].shape,
                    generator=generator,
                    dtype=mean.dtype,
                    device=mean.device,
                )
                noise[row_start : row_start + self.group_size] = epsilon
        return mean + std * noise

    def _encode_full_image_to_latent(self, full_images: torch.Tensor) -> torch.Tensor:
        """Encode full vertically stacked three-view images into VAE latents."""
        if self.strict_native_resolution and tuple(full_images.shape[-3:]) != (
            3, *self.full_image_size
        ):
            raise ValueError(
                "Native Ctrl-World encoder expected [B,3,1440,640], got "
                f"{tuple(full_images.shape)}"
            )
        with torch.no_grad():
            if self.per_view_vae_codec:
                # Match rollout_interact_cosmos_actions.py: encode each camera
                # view independently, then stack the three view latents along
                # height for the Ctrl-World diffusion model.
                view_images = torch.cat(
                    torch.chunk(full_images, chunks=3, dim=2), dim=0
                )
                view_distribution = self.model.pipeline.vae.encode(
                    view_images
                ).latent_dist
                view_latents = self._sample_vae_latent_distribution(
                    view_distribution, block_batch_size=int(full_images.shape[0])
                ).mul_(self.model.pipeline.vae.config.scaling_factor)
                batch_size = full_images.shape[0]
                grouped = view_latents.reshape(
                    3, batch_size, *view_latents.shape[1:]
                ).permute(1, 0, 2, 3, 4)
                latents = torch.cat(list(grouped.unbind(dim=1)), dim=-2)
            else:
                distribution = self.model.pipeline.vae.encode(
                    full_images
                ).latent_dist
                latents = self._sample_vae_latent_distribution(
                    distribution, block_batch_size=int(full_images.shape[0])
                ).mul_(self.model.pipeline.vae.config.scaling_factor)
        if self.strict_native_resolution and tuple(latents.shape[-3:]) != (
            4, *self.expected_latent_size
        ):
            raise ValueError(
                "Native Ctrl-World VAE expected [B,4,180,80], got "
                f"{tuple(latents.shape)}"
            )
        if self.strict_native_resolution and tuple(latents.shape[-3:]) != (
            4, *self.expected_latent_size
        ):
            raise ValueError(
                "Native Ctrl-World VAE expected [B,4,180,80], got "
                f"{tuple(latents.shape)}"
            )
        return latents

    def _decode_latents_to_views(self, latents: torch.Tensor):
        """Decode latent videos and split the vertically stacked result into views."""
        # Latent shape: [B, T, 4, H', W']
        if self.strict_native_resolution and tuple(latents.shape[-3:]) != (
            4, *self.expected_latent_size
        ):
            raise ValueError(
                "Native Ctrl-World decoder expected [B,T,4,180,80], got "
                f"{tuple(latents.shape)}"
            )
        bsz, t = latents.shape[:2]
        if self.per_view_vae_codec:
            # Match the standalone runner's einops rearrange: each view is an
            # independent VAE sample during decoding.
            latent_views = torch.chunk(latents, chunks=3, dim=-2)
            flat_latents = torch.cat(latent_views, dim=0).flatten(0, 1)
        else:
            flat_latents = latents.flatten(0, 1)

        decoded_videos = []
        decode_kwargs = {}
        for i in range(0, flat_latents.shape[0], self.decode_chunk_size):
            chunk = (
                flat_latents[i : i + self.decode_chunk_size]
                / self.model.pipeline.vae.config.scaling_factor
            )
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_videos.append(self.model.pipeline.vae.decode(chunk, **decode_kwargs).sample)

        videos = torch.cat(decoded_videos, dim=0)
        if self.per_view_vae_codec:
            view_videos = videos.reshape(3, bsz, t, *videos.shape[1:])
            return tuple(view_videos.unbind(dim=0))
        videos = videos.reshape(bsz, t, *videos.shape[1:])  # [B, T, 3, H, W]
        if self.strict_native_resolution and tuple(videos.shape[-3:]) != (
            3, *self.full_image_size
        ):
            raise ValueError(
                "Native Ctrl-World decode expected [B,T,3,1440,640], got "
                f"{tuple(videos.shape)}"
            )

        # Split vertically stacked images into three views.
        views = torch.chunk(videos, chunks=3, dim=3)
        if len(views) != 3:
            raise ValueError(
                f"Expected 3 views after split, got {len(views)} with video shape {videos.shape}"
            )
        if self.strict_native_resolution:
            for view_index, view in enumerate(views):
                if tuple(view.shape[-2:]) != self.image_size:
                    raise ValueError(
                        f"Native decoded view {view_index} expected {self.image_size}, "
                        f"got {tuple(view.shape[-2:])}"
                    )
        return views

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        env_idx: Optional[Union[list[int], np.ndarray, torch.Tensor]] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Reset environment state from dataset initial frames.

        The reset initializes:
        - current image observations;
        - current latent and latent history;
        - action history, initialized from initial EEF poses when available;
        - task descriptions used for text conditioning.
        """
        self.onload()

        if env_idx is None:
            self._common_noise_call_index = 0
            self._common_noise_rollout_index += 1
            target_env_idx = torch.arange(self.num_envs, device=self.device)
        else:
            target_env_idx = torch.as_tensor(
                env_idx, device=self.device, dtype=torch.long
            ).reshape(-1)
            if target_env_idx.numel() == 0:
                return self._wrap_obs(), {}

        if self.is_start and env_idx is None:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False
        num_reset_envs = int(target_env_idx.numel())
        if len(self.dataset) < num_reset_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_reset_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)
            allowed_episode_ids = self._allowed_reset_episode_ids()
            if len(allowed_episode_ids) < num_reset_envs:
                raise ValueError(
                    "Not enough allowed reset episodes. Found "
                    f"{len(allowed_episode_ids)}, need {num_reset_envs}."
                )
            episode_indices = np.random.choice(
                allowed_episode_ids, size=num_reset_envs, replace=False
            )
        else:
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()
            episode_indices = np.asarray(episode_indices)
            if episode_indices.shape[0] != num_reset_envs:
                raise ValueError(
                    f"Expected {num_reset_envs} episode indices, got {episode_indices.shape[0]}"
                )
            invalid_episode_ids = sorted(
                set(int(value) for value in episode_indices)
                - set(self._allowed_reset_episode_ids())
            )
            if invalid_episode_ids:
                raise ValueError(
                    "reset attempted episodes outside reset_episode_ids: "
                    f"{invalid_episode_ids}; split={self.reset_split!r}"
                )

        main_imgs = []
        wrist_imgs = []
        extra_view_imgs = []
        raw_policy_main_imgs = []
        raw_policy_wrist_imgs = []
        raw_policy_extra_view_imgs = []
        full_imgs = []
        task_descriptions = []
        init_ee_poses = []
        init_joint_poses = []

        for episode_idx in episode_indices:
            episode_data = self.dataset[int(episode_idx)]
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]
            prompt_override = self.ctrl_world_cfg.get("prompt_override", None)
            if prompt_override is None or not str(prompt_override).strip():
                task_desc = str(episode_data.get("task", ""))
            else:
                task_desc = str(prompt_override).strip()
            task_descriptions.append(task_desc)

            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            default_image = first_frame["image"]
            view_tensors = [
                first_frame.get("main_image_1", default_image),
                first_frame.get("main_image_2", default_image),
                first_frame.get("wrist_image", default_image),
            ]

            if self.use_raw_reset_policy_views:
                # Preserve the dataset-resolution [C,H,W] images before the
                # Ctrl-World-specific resize and in-place [-1,1]
                # normalization below. These are exposed only on the full
                # reset observation, so subsequent chunks continue to use the
                # latest generated Ctrl-World views.
                source_raw_views = [view.detach().clone() for view in view_tensors]
                raw_views = [
                    source_raw_views[source_index]
                    for source_index in self.model_camera_ids
                ]
                raw_policy_main_imgs.append(raw_views[self.main_view_index])
                raw_policy_wrist_imgs.append(raw_views[self.wrist_view_index])
                raw_policy_extra_view_imgs.append(raw_views[self.extra_view_index])

            normalized_source_views = []
            for view_index, view_tensor in enumerate(view_tensors):
                if self.strict_native_resolution and tuple(view_tensor.shape[1:]) != self.image_size:
                    raise ValueError(
                        f"Native reset view {view_index} must be {self.image_size} without resize, "
                        f"got {tuple(view_tensor.shape[1:])}"
                    )
                if view_tensor.shape[1:] != self.image_size:
                    view_tensor = F.interpolate(
                        view_tensor.unsqueeze(0),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                normalized_source_views.append(self.trans_norm(view_tensor))

            # The dataset remains in physical disk order. Only this opt-in
            # permutation changes the Ctrl-World model order; identity keeps
            # every retained rollout byte-for-byte compatible.
            normalized_views = [
                normalized_source_views[source_index]
                for source_index in self.model_camera_ids
            ]

            full_img = torch.cat(normalized_views, dim=1)
            selected_main_view = normalized_views[self.main_view_index]
            selected_wrist_view = normalized_views[self.wrist_view_index]
            selected_extra_view = normalized_views[self.extra_view_index]

            main_imgs.append(selected_main_view)
            wrist_imgs.append(selected_wrist_view)
            extra_view_imgs.append(selected_extra_view)
            full_imgs.append(full_img)

            if "observation.state" in first_frame:
                init_pose = first_frame["observation.state"].detach().cpu().numpy()
                image_state_representation = self.initial_image_state_representation
                if image_state_representation == "auto":
                    image_state_representation = (
                        "joint" if self.joint_action_adapter == "fk" else "eef"
                    )
                if self.joint_action_adapter == "fk" and image_state_representation == "joint":
                    # This dataset's "observation.state" is raw joint angles
                    # (six joints + gripper), not an EEF pose -- FK-convert it
                    # before it is used as the EEF state seed for the
                    # analytic adapters and Ctrl-World conditioning. Feeding
                    # raw joint angles straight through as if they were
                    # [x,y,z,rx,ry,rz,gripper] made the position channel
                    # 10-30x out of the real EEF workspace range, which
                    # `_normalize_action`'s q01/q99 clipping immediately
                    # saturates to +/-1 for nearly the whole rollout --
                    # freezing Ctrl-World's positional conditioning
                    # regardless of the policy's real per-step deltas.
                    init_pose_t = torch.as_tensor(
                        init_pose, dtype=torch.float32
                    ).reshape(1, 1, -1)
                    init_pose = (
                        ur5_joint_actions_to_eef_states(
                            init_pose_t,
                            tcp_xyz=self.joint_fk_tcp_xyz,
                            tcp_rotvec=self.joint_fk_tcp_rotvec,
                        )[0, 0]
                        .cpu()
                        .numpy()
                    )
                init_ee_poses.append(init_pose)
            else:
                init_ee_poses.append(None)
            init_joint_poses.append(self._read_initial_joint_state(int(episode_idx)))

        stacked_main = torch.stack(main_imgs, dim=0).to(self.device)  # [B,3,H,W]
        stacked_main = stacked_main.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )
        stacked_wrist = torch.stack(wrist_imgs, dim=0).to(self.device)
        stacked_wrist = stacked_wrist.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )
        stacked_extra_view = torch.stack(extra_view_imgs, dim=0).to(self.device)
        stacked_extra_view = stacked_extra_view.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, 1, self.condition_frame_length, 1, 1
        )

        full_images = torch.stack(full_imgs, dim=0).to(self.device, self.inference_dtype)
        current_latent = self._encode_full_image_to_latent(full_images)

        history_latents = current_latent.unsqueeze(1).repeat(
            1, self.num_history, 1, 1, 1
        )
        history_latent_bank = current_latent.unsqueeze(1).repeat(
            1, self.ctrl_world_history_bank_size, 1, 1, 1
        )

        init_states = []
        state_dim = 0
        for init_ee_pose in init_ee_poses:
            if init_ee_pose is None:
                init_state = np.zeros(7, dtype=np.float32)
            else:
                init_state = np.asarray(init_ee_pose, dtype=np.float32).reshape(-1)
                if init_state.shape[0] < 7:
                    init_state = np.pad(init_state, (0, 7 - init_state.shape[0]))
            state_dim = max(state_dim, init_state.shape[0])
            init_states.append(init_state)
        state_dim = self.action_dim
        init_states = [
            np.pad(state, (0, state_dim - state.shape[0])) if state.shape[0] < state_dim else state[:state_dim]
            for state in init_states
        ]
        current_states = self._align_policy_state_dim(
            torch.from_numpy(np.stack(init_states, axis=0))
        )
        if any(joint_pose is not None for joint_pose in init_joint_poses):
            joint_states = []
            for joint_pose, eef_state in zip(init_joint_poses, init_states, strict=True):
                if joint_pose is None:
                    joint_state = np.asarray(eef_state, dtype=np.float32)
                else:
                    joint_state = np.asarray(joint_pose, dtype=np.float32).reshape(-1)
                if joint_state.shape[0] < self.action_dim:
                    joint_state = np.pad(joint_state, (0, self.action_dim - joint_state.shape[0]))
                joint_states.append(joint_state[: self.action_dim])
            current_policy_states = self._align_policy_state_dim(
                torch.from_numpy(np.stack(joint_states, axis=0))
            )
        else:
            current_policy_states = current_states
        joint_fk_to_ctrl_alignment = None
        if self.joint_action_adapter == "fk":
            if any(joint_pose is None for joint_pose in init_joint_poses):
                raise ValueError(
                    "joint_action_adapter='fk' requires an aligned raw-joint "
                    "initial_joint_state_path for every reset episode"
                )
            fk_current = ur5_joint_actions_to_eef_states(
                current_policy_states[:, None, :],
                tcp_xyz=self.joint_fk_tcp_xyz,
                tcp_rotvec=self.joint_fk_tcp_rotvec,
            )[:, 0, :]
            joint_fk_to_ctrl_alignment = compute_eef_rigid_alignment(
                fk_current, current_states
            )
        action_history = self._build_initial_action_history(current_states, num_reset_envs)
        action_history_bank = self._build_initial_action_history(
            current_states, num_reset_envs, history_length=self.ctrl_world_history_bank_size
        )

        if env_idx is None or self.current_obs is None:
            self.current_obs = stacked_main
            self.current_wrist_obs = stacked_wrist
            self.current_extra_view_obs = stacked_extra_view
            self.current_latent = current_latent
            self.history_latents = history_latents
            self.history_latent_bank = history_latent_bank
            self.current_states = current_states
            self.current_policy_states = current_policy_states
            self.joint_fk_to_ctrl_alignment = joint_fk_to_ctrl_alignment
            self.action_history = action_history
            self.action_history_bank = action_history_bank
            self.task_descriptions = task_descriptions
            self.init_ee_poses = init_ee_poses
        else:
            self.current_obs[target_env_idx] = stacked_main
            self.current_wrist_obs[target_env_idx] = stacked_wrist
            self.current_extra_view_obs[target_env_idx] = stacked_extra_view
            self.current_latent[target_env_idx] = current_latent
            self.history_latents[target_env_idx] = history_latents
            if self.history_latent_bank is not None:
                self.history_latent_bank[target_env_idx] = history_latent_bank
            self.current_states[target_env_idx] = current_states
            self.current_policy_states[target_env_idx] = current_policy_states
            if joint_fk_to_ctrl_alignment is not None:
                self.joint_fk_to_ctrl_alignment[target_env_idx] = joint_fk_to_ctrl_alignment
            self.action_history[target_env_idx] = action_history
            if self.action_history_bank is not None:
                self.action_history_bank[target_env_idx] = action_history_bank
            target_env_idx_list = target_env_idx.tolist()
            for local_i, global_i in enumerate(target_env_idx_list):
                self.task_descriptions[global_i] = task_descriptions[local_i]
                self.init_ee_poses[global_i] = init_ee_poses[local_i]

        self._reset_metrics(target_env_idx)

        extracted_obs = self._wrap_obs()
        if self.use_raw_reset_policy_views and env_idx is None:
            extracted_obs.update(
                {
                    "policy_main_images": torch.stack(raw_policy_main_imgs, dim=0).to(
                        self.device
                    ),
                    "policy_wrist_images": torch.stack(raw_policy_wrist_imgs, dim=0).to(
                        self.device
                    ),
                    "policy_extra_view_images": torch.stack(
                        raw_policy_extra_view_imgs, dim=0
                    ).to(self.device),
                }
            )
        infos = {}
        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        """CtrlWorldEnv does not support single-step `step`; use `chunk_step`."""
        raise NotImplementedError(
            "step in CtrlWorldEnv is not implemented, use chunk_step instead"
        )

    @torch.no_grad()
    def _infer_next_chunk_rewards(self):
        """Score generated frames from the classifier's configured camera view."""
        reward_cfg = self.ctrl_world_cfg.get(
            "reward_model", self.cfg.get("reward_model", None)
        )
        native_frame_source = bool(
            reward_cfg is not None
            and str(reward_cfg.get("frame_source", "policy_aligned"))
            == "ctrl_world_native"
        )
        frames_per_env = self.ctrl_world_chunk if native_frame_source else self.chunk
        if self.reward_model is None:
            return torch.zeros(
                (self.num_envs, frames_per_env),
                dtype=torch.float32,
                device=self.device,
            )

        color_routed = callable(
            getattr(self.reward_model, "color_for_episode", None)
        )
        reward_observation = self._reward_model_observation()
        if native_frame_source:
            extract_chunk_obs = reward_observation.permute(0, 3, 1, 2, 4, 5)[
                :, -self.ctrl_world_chunk :, :, 0
            ]
        else:
            extract_chunk_obs = self._latest_policy_aligned_obs(reward_observation)
        extract_chunk_obs = extract_chunk_obs.reshape(
            self.num_envs * frames_per_env, 3, 1, *self.image_size
        )
        extract_chunk_obs = extract_chunk_obs.squeeze(2).to(
            device=self.device, dtype=torch.float32
        )
        input_size = tuple(
            int(value)
            for value in reward_cfg.get("input_size", (224, 224))
        )
        input_range = str(reward_cfg.get("input_range", "minus_one_one"))
        if input_range != "minus_one_one":
            raise ValueError(
                "Ctrl-World classifier frames are normalized to [-1,1]; "
                f"unsupported success_model input_range={input_range!r}"
            )
        extract_chunk_obs = F.interpolate(
            extract_chunk_obs,
            size=input_size,
            mode="bilinear",
            align_corners=False,
            antialias=_as_bool(reward_cfg.get("resize_antialias", False)),
        )

        inference_batch_size = int(
            reward_cfg.get("inference_batch_size", self.num_envs * self.chunk)
        )
        if inference_batch_size <= 0:
            raise ValueError("success_model inference_batch_size must be positive")
        outputs = []
        routed_episode_ids = None
        if color_routed:
            routed_episode_ids = self.reset_state_ids.detach().cpu().repeat_interleave(
                frames_per_env
            )
        for start in range(0, extract_chunk_obs.shape[0], inference_batch_size):
            images = extract_chunk_obs[start : start + inference_batch_size]
            if color_routed:
                output = self.reward_model.predict_rew(
                    images,
                    routed_episode_ids[start : start + images.shape[0]],
                )
            elif reward_cfg.type == "ResnetRewModel":
                output = self.reward_model.predict_rew(images)
            elif reward_cfg.type == "TaskEmbedResnetRewModel":
                flat_instructions = [
                    self.task_descriptions[index // frames_per_env]
                    for index in range(
                        start,
                        min(start + inference_batch_size, extract_chunk_obs.shape[0]),
                    )
                ]
                output = self.reward_model.predict_rew(images, flat_instructions)
            else:
                raise ValueError(f"Unknown reward model type: {reward_cfg.type}")
            outputs.append(
                output.to(device=self.device, dtype=torch.float32).reshape(-1)
            )
        return torch.cat(outputs, dim=0).reshape(self.num_envs, frames_per_env)

    def _reward_model_observation(self):
        """Return the observation cache selected for classifier inference."""
        view_index = int(self.reward_model_view_index)
        if view_index == self.main_view_index:
            observation = self.current_obs
        elif view_index == self.wrist_view_index:
            observation = self.current_wrist_obs
        elif view_index == self.extra_view_index:
            observation = self.current_extra_view_obs
        else:
            raise RuntimeError(
                "Classifier view index does not match any Ctrl-World view: "
                f"{view_index}"
            )
        if observation is None:
            raise RuntimeError(
                "Classifier observation cache is unavailable for "
                f"view_index={view_index}, camera_key={self.reward_model_camera_key!r}"
            )
        return observation

    def _first_success_pulse(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Emit one reward at the first threshold crossing in each episode."""
        reward_cfg = self.ctrl_world_cfg.reward_model
        threshold = float(reward_cfg.get("threshold", 0.9))
        reward_value = float(reward_cfg.get("reward_value", 1.0))
        crossings = probabilities >= threshold
        eligible = crossings & ~self.success_already_emitted[:, None]
        has_crossing = eligible.any(dim=1)
        first_index = eligible.to(torch.int64).argmax(dim=1)
        pulses = torch.zeros_like(probabilities, dtype=torch.float32)
        env_indices = torch.nonzero(has_crossing, as_tuple=False).squeeze(-1)
        if env_indices.numel() > 0:
            pulses[env_indices, first_index[env_indices]] = reward_value
            self.success_already_emitted[env_indices] = True
            self.success_once[env_indices] = True
        return pulses

    def _append_latent_history(self, latents: torch.Tensor) -> None:
        """Append generated latent frames to the rolling Ctrl-World history."""
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B,T,C,H,W], got {tuple(latents.shape)}")
        latents = latents.detach().to(self.device, self.inference_dtype)
        self.current_latent = latents[:, -1]
        self.history_latents = torch.cat([self.history_latents, latents], dim=1)[
            :, -self.num_history :
        ]

    def _append_action_history(self, state_conditions: torch.Tensor) -> None:
        """Append Ctrl-World state/action condition frames to rolling history."""
        if state_conditions.ndim != 3:
            raise ValueError(
                f"state_conditions must be [B,T,D], got {tuple(state_conditions.shape)}"
            )
        state_conditions = state_conditions.detach().to(self.device, torch.float32)
        if state_conditions.shape[-1] != self.action_history.shape[-1]:
            padded = torch.zeros(
                (*state_conditions.shape[:-1], self.action_history.shape[-1]),
                dtype=state_conditions.dtype,
                device=state_conditions.device,
            )
            dim = min(state_conditions.shape[-1], padded.shape[-1])
            padded[..., :dim] = state_conditions[..., :dim]
            state_conditions = padded
        self.action_history = torch.cat([self.action_history, state_conditions], dim=1)[
            :, -self.num_history :
        ]

    def _select_sparse_window_history(self) -> None:
        """Select user-style sparse window history into Ctrl-World inputs."""
        if self.history_latent_bank is None or self.action_history_bank is None:
            return
        indices = torch.tensor(
            [int(idx) for idx in self.ctrl_world_history_idx],
            dtype=torch.long,
            device=self.device,
        )
        bank_len = int(self.history_latent_bank.shape[1])
        indices = torch.where(indices < 0, indices + bank_len, indices)
        indices = torch.clamp(indices, min=0, max=bank_len - 1)
        self.history_latents = self.history_latent_bank.index_select(1, indices)
        self.action_history = self.action_history_bank.index_select(1, indices)

    def _append_sparse_window_history(
        self, endpoint_latent: torch.Tensor, endpoint_condition: torch.Tensor
    ) -> None:
        """Append one generated window endpoint to the sparse history bank."""
        endpoint_latent = endpoint_latent.detach().to(self.device, self.inference_dtype)
        endpoint_condition = endpoint_condition.detach().to(self.device, torch.float32)
        if endpoint_latent.ndim != 5 or endpoint_latent.shape[1] != 1:
            raise ValueError(
                f"endpoint_latent must be [B,1,C,H,W], got {tuple(endpoint_latent.shape)}"
            )
        if endpoint_condition.ndim != 3 or endpoint_condition.shape[1] != 1:
            raise ValueError(
                "endpoint_condition must be [B,1,D], "
                f"got {tuple(endpoint_condition.shape)}"
            )
        if endpoint_condition.shape[-1] != self.action_dim:
            padded = torch.zeros(
                (*endpoint_condition.shape[:-1], self.action_dim),
                dtype=endpoint_condition.dtype,
                device=endpoint_condition.device,
            )
            dim = min(endpoint_condition.shape[-1], self.action_dim)
            padded[..., :dim] = endpoint_condition[..., :dim]
            endpoint_condition = padded
        self.current_latent = endpoint_latent[:, -1]
        if self.history_latent_bank is None:
            self.history_latent_bank = endpoint_latent.repeat(
                1, self.ctrl_world_history_bank_size, 1, 1, 1
            )
        else:
            self.history_latent_bank = torch.cat(
                [self.history_latent_bank, endpoint_latent], dim=1
            )[:, -self.ctrl_world_history_bank_size :]
        if self.action_history_bank is None:
            self.action_history_bank = endpoint_condition.repeat(
                1, self.ctrl_world_history_bank_size, 1
            )
        else:
            self.action_history_bank = torch.cat(
                [self.action_history_bank, endpoint_condition], dim=1
            )[:, -self.ctrl_world_history_bank_size :]
        self._select_sparse_window_history()

    def _ctrl_world_group_generators(self) -> list[torch.Generator] | None:
        if self._fixed_denoise_seeds is not None:
            if len(self._fixed_denoise_seeds) != self.num_envs:
                raise ValueError(
                    "fixed per-env denoise seeds must match num_envs: "
                    f"{len(self._fixed_denoise_seeds)} != {self.num_envs}"
                )
            call_index = int(self._common_noise_call_index)
            self._common_noise_call_index += 1
            generators = []
            derived_seeds = []
            for base_seed in self._fixed_denoise_seeds:
                seed = derive_stable_seed(
                    "post_update_eval_ctrl_world_denoise",
                    int(base_seed),
                    call_index,
                )
                derived_seeds.append(int(seed))
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                generators.append(generator)
            seed_log_entry = {
                "call_index": call_index,
                "derived_seeds": tuple(derived_seeds),
            }
            if (
                not self._post_update_eval_denoise_seed_log
                or self._post_update_eval_denoise_seed_log[-1]["call_index"]
                != call_index
            ):
                self._post_update_eval_denoise_seed_log.append(seed_log_entry)
            elif self._post_update_eval_denoise_seed_log[-1] != seed_log_entry:
                raise RuntimeError(
                    "post-update eval retry derived different Ctrl-World seeds"
                )
            return generators
        if self.fixed_denoise_seed is not None:
            generators = []
            for _ in range(self.num_envs):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.fixed_denoise_seed)
                generators.append(generator)
            return generators
        if not self.common_random_numbers_within_group:
            return None
        call_index = int(self._common_noise_call_index)
        self._common_noise_call_index += 1
        generators = []
        for env_id in range(self.num_envs):
            group_id = env_id // self.group_size
            if self._cross_rank_metadata is None:
                seed = (
                    self.common_noise_seed
                    + int(self.seed_offset) * 1_000_000_000
                    + int(self._common_noise_rollout_index) * 1_000_000
                    + group_id * 10_000
                    + call_index
                )
            else:
                seed = derive_stable_seed(
                    "ctrl_world_denoise",
                    int(
                        self._cross_rank_metadata["ctrl_world_noise_seed"][
                            group_id * self.group_size
                        ].item()
                    ),
                    call_index,
                )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            generators.append(generator)
        return generators

    def _run_ctrl_world_denoise(
        self, ctrl_world_condition_np: np.ndarray
    ) -> tuple[torch.Tensor, float, int]:
        """Run one Ctrl-World diffusion call for a state-condition window."""
        num_frames = int(ctrl_world_condition_np.shape[1])
        if num_frames <= 0:
            raise ValueError("ctrl_world_condition_np must contain at least one frame")
        action_cond_raw = np.concatenate(
            [self.action_history.detach().cpu().numpy(), ctrl_world_condition_np], axis=1
        )
        action_cond_norm = self._normalize_action(action_cond_raw)
        action_cond = (
            torch.from_numpy(action_cond_norm)
            .to(self.device)
            .to(self.inference_dtype)
        )

        with torch.no_grad():
            if self.text_cond:
                text_token = self.model.action_encoder(
                    action_cond,
                    self.task_descriptions,
                    self.model.tokenizer,
                    self.model.text_encoder,
                )
            else:
                text_token = self.model.action_encoder(action_cond)

            denoise_start = time.monotonic()
            max_retries = int(
                self.ctrl_world_cfg.get("rollout_retry_max_retries", 0)
            )
            if max_retries < 0:
                raise ValueError("rollout_retry_max_retries must be non-negative")
            retry_count = 0
            original_noise_call_index = int(self._common_noise_call_index)
            while True:
                self._common_noise_call_index = original_noise_call_index
                try:
                    _, pred_latents = self.ctrl_world_pipeline_cls.__call__(
                        self.model.pipeline,
                        image=self.current_latent.to(self.inference_dtype),
                        text=text_token,
                        width=self.image_size[1],
                        height=self.full_image_size[0],
                        num_frames=num_frames,
                        history=self.history_latents.to(self.inference_dtype),
                        num_inference_steps=self.num_inference_steps,
                        decode_chunk_size=self.decode_chunk_size,
                        max_guidance_scale=self.guidance_scale,
                        fps=self.fps,
                        motion_bucket_id=self.motion_bucket_id,
                        mask=None,
                        output_type="latent",
                        return_dict=False,
                        frame_level_cond=self.frame_level_cond,
                        his_cond_zero=self.his_cond_zero,
                        generator=self._ctrl_world_group_generators(),
                    )
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if (
                        retry_count >= max_retries
                        or "out of memory" in message
                        or "invalid" in message
                        or "shape" in message
                    ):
                        raise
                    retry_count += 1
            denoise_elapsed = time.monotonic() - denoise_start

        pred_latents = pred_latents.to(self.device, self.inference_dtype)
        if pred_latents.shape[1] != num_frames:
            raise RuntimeError(
                f"Ctrl-World returned {pred_latents.shape[1]} frames for request of {num_frames}"
            )
        return pred_latents, denoise_elapsed, retry_count

    def _infer_next_chunk_frames(self, actions):
        """Generate the next latent/video chunk conditioned on action history."""
        num_envs = self.num_envs
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        if actions_np.shape != (num_envs, self.chunk, self.policy_action_dim):
            raise ValueError(
                f"Expected actions shape {(num_envs, self.chunk, self.policy_action_dim)}, got {actions_np.shape}"
            )

        ctrl_world_actions_np, next_policy_state_np = (
            self._convert_policy_actions_to_ctrl_world(actions_np)
        )
        if ctrl_world_actions_np.shape[1] != self.ctrl_world_chunk:
            raise ValueError(
                f"Expected {self.ctrl_world_chunk} Ctrl-World conditions, "
                f"got {ctrl_world_actions_np.shape[1]}"
            )

        prev_current_latent = self.current_latent.detach()
        rollout_retry_count = 0

        if self.ctrl_world_internal_rollout:
            emitted_latents = []
            num_windows = 0
            total_elapsed = 0.0
            total_frames_requested = 0
            total_conditions = ctrl_world_actions_np.shape[1]
            endpoint_conditions_np = getattr(self, "_last_ctrl_world_endpoint_condition_np", None)
            condition_source_np = ctrl_world_actions_np
            has_endpoint_condition = False
            if (
                endpoint_conditions_np is not None
                and endpoint_conditions_np.shape[0] == ctrl_world_actions_np.shape[0]
                and endpoint_conditions_np.shape[1] == total_conditions + 1
            ):
                condition_source_np = endpoint_conditions_np
                has_endpoint_condition = True

            window_plan = _plan_ctrl_world_windows(
                total_conditions,
                self.ctrl_world_window_frames,
                self.ctrl_world_window_emit_frames,
                use_lookahead_latent=self.ctrl_world_window_use_lookahead_latent,
            )
            for emitted, window_frames, emit_frames in window_plan:
                window_condition_np = condition_source_np[
                    :, emitted : emitted + window_frames, :
                ]
                if window_condition_np.shape[1] < window_frames:
                    pad_frames = window_frames - window_condition_np.shape[1]
                    window_condition_np = np.concatenate(
                        [
                            window_condition_np,
                            np.repeat(
                                window_condition_np[:, -1:, :],
                                pad_frames,
                                axis=1,
                            ),
                        ],
                        axis=1,
                    )
                if self.ctrl_world_window_history_mode == "sparse_window":
                    self._select_sparse_window_history()
                pred_window, denoise_elapsed, window_retry_count = self._run_ctrl_world_denoise(
                    window_condition_np
                )
                rollout_retry_count += window_retry_count
                total_elapsed += denoise_elapsed
                total_frames_requested += window_frames
                num_windows += 1

                if self.ctrl_world_window_use_lookahead_latent:
                    # This checkpoint predicts [x_t, x_{t+1}, ..., x_{t+4}]
                    # in a five-frame window. Match the proven independent
                    # rollout exactly: publish prediction[0:4], but retain
                    # prediction[4] as the private endpoint used to seed the
                    # next window. The next window reconstructs that endpoint
                    # as its new prediction[0], so no published frame is
                    # duplicated.
                    if pred_window.shape[1] < emit_frames + 1:
                        raise RuntimeError(
                            "Ctrl-World lookahead prediction is missing its "
                            f"boundary frame: shape={tuple(pred_window.shape)}, "
                            f"emit_frames={emit_frames}"
                        )
                emitted_latents.append(pred_window[:, :emit_frames])
                if self.ctrl_world_window_history_mode == "sparse_window":
                    endpoint_idx = (
                        emit_frames
                        if self.ctrl_world_window_use_lookahead_latent
                        else max(emit_frames - 1, 0)
                    )
                    self._append_sparse_window_history(
                        pred_window[:, endpoint_idx : endpoint_idx + 1],
                        torch.from_numpy(window_condition_np[:, endpoint_idx : endpoint_idx + 1, :]),
                    )
                else:
                    if self.ctrl_world_window_use_lookahead_latent:
                        # The rolling history contains only frames preceding
                        # current_latent. Append [x_t, ..., x_{t+3}] and keep
                        # x_{t+4} exclusively as the next current boundary.
                        self._append_latent_history(
                            pred_window[:, :emit_frames]
                        )
                        self.current_latent = pred_window[:, emit_frames].detach().to(
                            self.device, self.inference_dtype
                        )
                    else:
                        self._append_latent_history(
                            pred_window[:, :emit_frames]
                        )
                    self._append_action_history(
                        torch.from_numpy(
                            window_condition_np[:, :emit_frames, :]
                        )
                    )

            pred_latents = torch.cat(emitted_latents, dim=1)
            if pred_latents.shape[1] != self.ctrl_world_chunk:
                raise RuntimeError(
                    f"Internal rollout emitted {pred_latents.shape[1]} frames, "
                    f"expected {self.ctrl_world_chunk}"
                )
            denoise_elapsed = total_elapsed
            frames_for_log = self.ctrl_world_chunk
        else:
            pred_latents, denoise_elapsed, rollout_retry_count = self._run_ctrl_world_denoise(
                ctrl_world_actions_np
            )
            num_windows = 1
            total_frames_requested = self.ctrl_world_chunk
            frames_for_log = self.ctrl_world_chunk
            self._append_latent_history(pred_latents)
            self._append_action_history(torch.from_numpy(ctrl_world_actions_np))

        self._ctrl_world_denoise_batches = getattr(
            self, "_ctrl_world_denoise_batches", 0
        ) + num_windows
        self.latest_rollout_retry_count = int(rollout_retry_count)
        log_interval = int(os.environ.get("RLINF_DENOISE_LOG_INTERVAL", "1"))
        if (
            log_interval > 0
            and self._ctrl_world_denoise_batches % log_interval == 0
            and getattr(self.worker_info, "rank", 0) == 0
        ):
            try:
                if torch.is_tensor(self.elapsed_steps):
                    chunk_index = int(self.elapsed_steps.min().item()) // max(self.chunk, 1) + 1
                else:
                    chunk_index = int(self.elapsed_steps) // max(self.chunk, 1) + 1
                if self.ctrl_world_internal_rollout:
                    get_logger().info(
                        "[Denoise][ctrl-world] chunk %s | batch=%s | frames=%s | "
                        "windows=%s | requested_frames=%s | denoise_steps=%s | "
                        "elapsed %.1fs | avg %.1fs/env",
                        chunk_index,
                        num_envs,
                        frames_for_log,
                        num_windows,
                        total_frames_requested,
                        self.num_inference_steps,
                        denoise_elapsed,
                        denoise_elapsed / max(num_envs, 1),
                    )
                else:
                    get_logger().info(
                        "[Denoise][ctrl-world] chunk %s | batch=%s | frames=%s | "
                        "denoise_steps=%s | elapsed %.1fs | avg %.1fs/env",
                        chunk_index,
                        num_envs,
                        self.ctrl_world_chunk,
                        self.num_inference_steps,
                        denoise_elapsed,
                        denoise_elapsed / max(num_envs, 1),
                    )
            except Exception:
                pass

        prev_step_latents = torch.cat(
            [prev_current_latent.unsqueeze(1), pred_latents[:, :-1]], dim=1
        )
        latent_motion = (pred_latents - prev_step_latents).pow(2).mean(
            dim=(2, 3, 4)
        ).sqrt()
        self.latest_latent_motion_ctrl_world = latent_motion.detach().to(torch.float32)
        self.latest_latent_motion = self._select_policy_aligned_sequence(
            self.latest_latent_motion_ctrl_world, dim=1
        )

        # Align the policy-side state cache with the last frame emitted for this chunk.
        last_condition = torch.from_numpy(ctrl_world_actions_np[:, -1, :]).to(
            self.device, torch.float32
        )
        next_policy_state = last_condition
        if next_policy_state_np is not None:
            next_policy_state = torch.from_numpy(next_policy_state_np).to(
                self.device, torch.float32
            )
        self._set_current_states_from_ctrl_world_condition(next_policy_state)

        # Decode the generated chunk and append it to the observation history.
        views = self._decode_latents_to_views(pred_latents)

        main_video = views[self.main_view_index].permute(0, 2, 1, 3, 4)  # [B,3,T,H,W]
        wrist_video = views[self.wrist_view_index].permute(0, 2, 1, 3, 4)
        extra_view_video = views[self.extra_view_index].permute(0, 2, 1, 3, 4)

        x_main = main_video.unsqueeze(2)   # [B,3,1,T,H,W]
        x_wrist = wrist_video.unsqueeze(2) # [B,3,1,T,H,W]
        x_extra = extra_view_video.unsqueeze(2)  # [B,3,1,T,H,W]

        self.current_obs = torch.cat([self.current_obs, x_main], dim=3)
        self.current_wrist_obs = torch.cat([self.current_wrist_obs, x_wrist], dim=3)
        self.current_extra_view_obs = torch.cat(
            [self.current_extra_view_obs, x_extra], dim=3
        )

        max_frames = self.condition_frame_length + self.ctrl_world_chunk * 2
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]
            self.current_wrist_obs = self.current_wrist_obs[:, :, :, -max_frames:, :, :]
            self.current_extra_view_obs = self.current_extra_view_obs[
                :, :, :, -max_frames:, :, :
            ]

    def _wrap_obs(self):
        """Convert internal normalized tensors to uint8 observations for the policy."""
        num_envs = self.num_envs

        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs

        main_last = self.current_obs[:, :, 0, -1, :, :]   # [B,3,H,W]
        wrist_last = self.current_wrist_obs[:, :, 0, -1, :, :] if self.current_wrist_obs is not None else None
        extra_view_last = (
            self.current_extra_view_obs[:, :, 0, -1, :, :]
            if self.current_extra_view_obs is not None
            else None
        )

        if wrist_last is not None and extra_view_last is not None:
            # The policy receives a 3-view concat composite built from each
            # view's native resolution (cosmos_backend.py's
            # _stitch_three_camera_views_for_policy_input). Letterbox-padding each view to
            # `policy_image_size` independently *before* concatenation would
            # bake a mismatched black border into every tile -- e.g. Ctrl-
            # World's native (192, 320) padded to a (480, 640) target adds
            # large bars since the aspect ratios differ a lot, and those
            # bars end up baked into the middle of the final composite, not
            # just its outer edge. Keep native resolution here; the concat
            # function is the only place a resize should happen.
            main_image = self._raw_image_to_uint8(main_last)
            wrist_image = self._raw_image_to_uint8(wrist_last)
            extra_view_image = self._raw_image_to_uint8(extra_view_last)
        else:
            main_image = self._policy_image_to_uint8(main_last)
            wrist_image = (
                self._policy_image_to_uint8(wrist_last) if wrist_last is not None else None
            )
            extra_view_image = (
                self._policy_image_to_uint8(extra_view_last)
                if extra_view_last is not None
                else None
            )

        policy_states = self.current_policy_states
        if policy_states is None:
            policy_states = self.current_states
        if policy_states is None:
            states = torch.zeros(
                (num_envs, self.action_dim), device=self.device, dtype=torch.float32
            )
        else:
            states = self._align_policy_state_dim(policy_states)
            self.current_policy_states = states

        obs = {
            "main_images": main_image,
            "wrist_images": wrist_image,
            "extra_view_images": extra_view_image,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _resize_policy_image_with_pad(self, image: torch.Tensor) -> torch.Tensor:
        """Resize normalized `[B, 3, H, W]` policy images without changing aspect ratio."""
        if self.policy_image_size is None:
            return image
        target_h, target_w = self.policy_image_size
        if image.shape[-2:] == (target_h, target_w):
            return image

        cur_h, cur_w = image.shape[-2:]
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = max(1, int(cur_h / ratio))
        resized_w = max(1, int(cur_w / ratio))
        image = F.interpolate(
            image,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )

        pad_h0, remainder_h = divmod(target_h - resized_h, 2)
        pad_w0, remainder_w = divmod(target_w - resized_w, 2)
        pad_h1 = pad_h0 + remainder_h
        pad_w1 = pad_w0 + remainder_w
        return F.pad(image, (pad_w0, pad_w1, pad_h0, pad_h1), value=-1.0)

    def _policy_image_to_uint8(self, image: torch.Tensor) -> torch.Tensor:
        image = self._resize_policy_image_with_pad(image)
        return self._raw_image_to_uint8(image)

    def _raw_image_to_uint8(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a normalized `[-1,1]` `[B,3,H,W]` image to uint8 `[B,H,W,3]`
        without any resize/letterbox -- used for multiview tiles that get
        concatenated downstream, where padding each tile independently would
        bake mismatched black borders into the composite."""
        image = image.permute(0, 2, 3, 1)
        image = (image + 1.0) / 2.0 * 255.0
        return torch.clamp(image, 0, 255).to(torch.uint8)

    def _get_latest_chunk_main_images(self) -> torch.Tensor:
        """Return the latest chunk of main-view images as uint8 `[B, chunk, H, W, C]`."""
        latest_chunk = self._latest_policy_aligned_obs(self.current_obs)[:, :, :, 0]  # [B,T,3,H,W]
        latest_chunk = latest_chunk.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _get_latest_ctrl_world_chunk_images(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the latest raw Ctrl-World chunk as uint8 frames."""
        latest_chunk = obs.permute(0, 3, 1, 2, 4, 5)[
            :, -self.ctrl_world_chunk :, :, 0
        ]  # [B,T,3,H,W]
        latest_chunk = latest_chunk.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _get_latest_ctrl_world_main_images(self) -> torch.Tensor:
        return self._get_latest_ctrl_world_chunk_images(self.current_obs)

    def _get_latest_ctrl_world_wrist_images(self) -> Optional[torch.Tensor]:
        if self.current_wrist_obs is None:
            return None
        return self._get_latest_ctrl_world_chunk_images(self.current_wrist_obs)

    def _get_latest_ctrl_world_extra_view_images(self) -> Optional[torch.Tensor]:
        if self.current_extra_view_obs is None:
            return None
        return self._get_latest_ctrl_world_chunk_images(self.current_extra_view_obs)

    def _get_latest_chunk_wrist_images(self) -> Optional[torch.Tensor]:
        """Return the latest chunk of wrist-view images as uint8 `[B, chunk, H, W, C]`."""
        if self.current_wrist_obs is None:
            return None
        latest_chunk = self._latest_policy_aligned_obs(self.current_wrist_obs)[:, :, :, 0]  # [B,T,3,H,W]
        latest_chunk = latest_chunk.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _get_latest_chunk_extra_view_images(self) -> Optional[torch.Tensor]:
        """Return the latest chunk of extra/side-view images as uint8 `[B, chunk, H, W, C]`."""
        if self.current_extra_view_obs is None:
            return None
        latest_chunk = self._latest_policy_aligned_obs(self.current_extra_view_obs)[:, :, :, 0]  # [B,T,3,H,W]
        latest_chunk = latest_chunk.permute(0, 1, 3, 4, 2)  # [B,T,H,W,C]
        latest_chunk = (latest_chunk + 1.0) / 2.0 * 255.0
        return torch.clamp(latest_chunk, 0, 255).to(torch.uint8)

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Auto-reset done environments while preserving terminal observations/info."""
        final_obs = extracted_obs
        final_info = infos
        done_env_idx = torch.where(dones)[0]
        extracted_obs, infos = self.reset(env_idx=done_env_idx)
        for key in (
            "chunk_raw_rewards",
            "latent_motion",
            "success_frame_time_idx",
            "success_frame_images",
            "success_frame_raw_images",
            "success_frame_raw_tensors",
            "success_frame_meta",
            "success_frame_wrist_images",
            "ctrl_world_video_chunk",
            "ctrl_world_video_chunk_wrist",
            "ctrl_world_video_chunk_extra",
            "ctrl_world_video_chunk_raw",
            "ctrl_world_video_chunk_raw_wrist",
            "ctrl_world_video_chunk_raw_extra",
        ):
            if key in final_info:
                infos[key] = final_info[key]

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Chunk-based environment stepping interface used by the RL loop.

        Args:
            policy_output_action: Tensor/ndarray with shape `[B, chunk, action_dim]`.

        Returns:
            Vector-env-style chunk tuple:
            `([obs], rewards, terminations, truncations, [infos])`.
        """
        self.onload()
        if isinstance(policy_output_action, torch.Tensor):
            policy_actions_for_video = policy_output_action.detach().to(
                device=self.device, dtype=torch.float32
            )
        else:
            policy_actions_for_video = torch.as_tensor(
                policy_output_action, device=self.device, dtype=torch.float32
            )
        effective_action = policy_output_action

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._infer_next_chunk_frames(effective_action)

        # One environment step corresponds to a full chunk, not a single frame.
        self.elapsed_steps += self.chunk

        extracted_obs = self._wrap_obs()

        # Score the latest generated chunk frame by frame, then convert to training rewards.
        chunk_probabilities = self._infer_next_chunk_rewards()
        if chunk_probabilities.shape[1] == self.chunk:
            chunk_rewards = (
                self._first_success_pulse(chunk_probabilities)
                if getattr(self, "reward_source", "legacy_reward_model")
                == "success_model"
                else chunk_probabilities
            )
        elif _as_bool(
            self.ctrl_world_cfg.get("reward_model_diagnostic_only", False)
        ):
            # Native-FPS classifier probabilities are diagnostic metadata only;
            # keep the environment reward width on the policy action chunk.
            chunk_rewards = torch.zeros(
                (self.num_envs, self.chunk),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise ValueError(
                "Non-diagnostic reward model output must match the policy chunk: "
                f"{chunk_probabilities.shape[1]} != {self.chunk}"
            )
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        # Respect ignore_terminations before consulting any reward-based success
        # heuristic. Cosmos 7D/10D train and eval use this mode because the
        # current reward is a dense training signal, not a success classifier.
        raw_chunk_terminations = self._build_chunk_terminations(chunk_rewards)

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        latest_chunk_main_images = self._get_latest_chunk_main_images()
        latest_chunk_wrist_images = self._get_latest_chunk_wrist_images()
        latest_ctrl_world_main_images = self._get_latest_ctrl_world_main_images()
        latest_ctrl_world_wrist_images = self._get_latest_ctrl_world_wrist_images()
        latest_ctrl_world_extra_images = self._get_latest_ctrl_world_extra_view_images()

        # Success-frame artifacts only make sense for the legacy opt-in path.
        # The Cosmos 7D/10D ignore-terminations path does not even compute an
        # argmax/threshold that could be mistaken for success classification.
        success_frame_info = {}
        if not self.ignore_terminations:
            success_frame_time_idx = chunk_rewards.argmax(dim=1)
            latest_chunk_main_tensors = self._latest_policy_aligned_obs(
                self.current_obs
            )[:, :, :, 0].contiguous()
            success_frame_images = latest_chunk_main_images[
                torch.arange(self.num_envs, device=self.device), success_frame_time_idx
            ]
            success_frame_raw_tensors = latest_chunk_main_tensors[
                torch.arange(self.num_envs, device=self.device), success_frame_time_idx
            ].detach().to(torch.float32).cpu()
            success_frame_raw_images = success_frame_raw_tensors.permute(0, 2, 3, 1)
            success_frame_raw_images = torch.clamp(
                (success_frame_raw_images + 1.0) / 2.0 * 255.0, 0, 255
            ).to(torch.uint8)
            success_frame_meta = []
            success_threshold = float(
                getattr(self.cfg, "success_reward_threshold", 0.9)
            )
            chunk_rewards_cpu = chunk_rewards.detach().to(torch.float32).cpu()
            success_frame_time_idx_cpu = success_frame_time_idx.detach().cpu()
            estimated_success_cpu = past_terminations.detach().cpu()
            elapsed_steps_cpu = self.elapsed_steps.detach().cpu()
            for env_id in range(self.num_envs):
                frame_idx = int(success_frame_time_idx_cpu[env_id].item())
                chunk_rewards_env = chunk_rewards_cpu[env_id]
                success_frame_meta.append(
                    {
                        "env_id": env_id,
                        "elapsed_steps": int(elapsed_steps_cpu[env_id].item()),
                        "success_frame_time_idx": frame_idx,
                        "selected_frame_reward": float(chunk_rewards_env[frame_idx].item()),
                        "max_reward_in_chunk": float(chunk_rewards_env.max().item()),
                        "chunk_raw_rewards": [
                            float(reward_value)
                            for reward_value in chunk_rewards_env.tolist()
                        ],
                        "success_reward_threshold": success_threshold,
                        "success_estimated": bool(estimated_success_cpu[env_id].item()),
                    }
                )
            success_frame_wrist_images = None
            if latest_chunk_wrist_images is not None:
                success_frame_wrist_images = latest_chunk_wrist_images[
                    torch.arange(self.num_envs, device=self.device),
                    success_frame_time_idx,
                ]
            success_frame_info = {
                "success_frame_time_idx": success_frame_time_idx,
                "success_frame_images": success_frame_images,
                "success_frame_raw_images": success_frame_raw_images,
                "success_frame_raw_tensors": success_frame_raw_tensors,
                "success_frame_meta": success_frame_meta,
                "success_frame_wrist_images": success_frame_wrist_images,
            }

        # The current chunk contributes the sum of all frame rewards to trajectory return.
        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, {}
        )
        infos["policy_action"] = policy_actions_for_video
        infos["policy_action_last"] = policy_actions_for_video[:, -1, :]
        infos["reset_state_ids"] = self.reset_state_ids.clone()
        infos["episode/reset_episode"] = self.reset_state_ids.clone()
        color_for_episode = getattr(self.reward_model, "color_for_episode", None)
        if callable(color_for_episode):
            color_order = tuple(getattr(self.reward_model, "color_ranges", {}).keys())
            color_ids = [
                color_order.index(color_for_episode(int(episode_id)))
                for episode_id in self.reset_state_ids.detach().cpu().tolist()
            ]
            infos["episode/color_id"] = torch.tensor(
                color_ids, dtype=torch.int64, device=self.device
            )
        infos["chunk_raw_rewards"] = chunk_probabilities
        infos["rollout_retry_count"] = torch.full(
            (self.num_envs, 1),
            int(getattr(self, "latest_rollout_retry_count", 0)),
            dtype=torch.int64,
            device=self.device,
        )
        if getattr(self, "reward_source", "legacy_reward_model") == "success_model":
            reward_cfg = self.ctrl_world_cfg.reward_model
            pulse_count = chunk_rewards.gt(0).sum().to(torch.float32)
            infos["reward/success_model/probability_min"] = (
                chunk_probabilities.min().reshape(1)
            )
            infos["reward/success_model/probability_mean"] = (
                chunk_probabilities.mean().reshape(1)
            )
            infos["reward/success_model/probability_max"] = (
                chunk_probabilities.max().reshape(1)
            )
            infos["reward/success_model/threshold"] = torch.tensor(
                [float(reward_cfg.get("threshold", 0.9))], device=self.device
            )
            infos["reward/success_model/pulse_count"] = pulse_count.reshape(1)
            infos["reward/success_model/pulse_rate"] = (
                pulse_count / float(self.num_envs)
            ).reshape(1)
            infos["episode/success_once"] = self.success_once.to(torch.float32)
        infos["latent_motion"] = self.latest_latent_motion
        infos.update(success_frame_info)
        infos["ctrl_world_video_chunk"] = latest_chunk_main_images
        infos["ctrl_world_video_chunk_wrist"] = latest_chunk_wrist_images
        infos["ctrl_world_video_chunk_extra"] = self._get_latest_chunk_extra_view_images()
        infos["ctrl_world_video_chunk_raw"] = latest_ctrl_world_main_images
        infos["ctrl_world_video_chunk_raw_wrist"] = latest_ctrl_world_wrist_images
        infos["ctrl_world_video_chunk_raw_extra"] = latest_ctrl_world_extra_images

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones, extracted_obs, infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        """Move large models and state to CPU to reduce GPU memory usage."""
        if self._is_offloaded:
            return

        self.model = self.model.to("cpu")
        if self.reward_model is not None:
            self.reward_model = self.reward_model.to("cpu")

        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_wrist_obs = recursive_to_device(self.current_wrist_obs, "cpu")
        self.current_extra_view_obs = recursive_to_device(self.current_extra_view_obs, "cpu")
        self.current_latent = recursive_to_device(self.current_latent, "cpu")
        self.history_latents = recursive_to_device(self.history_latents, "cpu")
        self.history_latent_bank = recursive_to_device(self.history_latent_bank, "cpu")
        self.current_states = recursive_to_device(self.current_states, "cpu")
        self.current_policy_states = recursive_to_device(self.current_policy_states, "cpu")
        self.joint_fk_to_ctrl_alignment = recursive_to_device(
            self.joint_fk_to_ctrl_alignment, "cpu"
        )
        self.action_history = self.action_history.cpu()
        if self.action_history_bank is not None:
            self.action_history_bank = self.action_history_bank.cpu()

        self.elapsed_steps = self.elapsed_steps.cpu()
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        self.success_already_emitted = self.success_already_emitted.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        trim_cpu_allocator()
        self._is_offloaded = True

    def onload(self):
        """Move state and models back to the target device before inference/stepping."""
        if not self._is_offloaded:
            return

        self.model = self.model.to(self.device, self.inference_dtype)
        if self.reward_model is not None:
            self.reward_model = self.reward_model.to(self.device)

        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_wrist_obs = recursive_to_device(self.current_wrist_obs, self.device)
        self.current_extra_view_obs = recursive_to_device(
            self.current_extra_view_obs, self.device
        )
        self.current_latent = recursive_to_device(self.current_latent, self.device)
        self.history_latents = recursive_to_device(self.history_latents, self.device)
        self.history_latent_bank = recursive_to_device(self.history_latent_bank, self.device)
        self.current_states = recursive_to_device(self.current_states, self.device)
        self.current_policy_states = recursive_to_device(
            self.current_policy_states, self.device
        )
        self.joint_fk_to_ctrl_alignment = recursive_to_device(
            self.joint_fk_to_ctrl_alignment, self.device
        )
        self.action_history = self.action_history.to(self.device)
        if self.action_history_bank is not None:
            self.action_history_bank = self.action_history_bank.to(self.device)

        self.elapsed_steps = self.elapsed_steps.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        self.success_already_emitted = self.success_already_emitted.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)

        # Moving the full Ctrl-World model back to GPU frees its CPU tensor
        # storage, but glibc normally keeps those pages in the process RSS.
        # Release them before the rollout begins so four Env workers do not
        # retain four allocator high-water marks inside the node cgroup.
        gc.collect()
        torch.cuda.empty_cache()
        trim_cpu_allocator()
        self._is_offloaded = False

    def get_state(self) -> bytes:
        """Serialize environment state for checkpointing or migration."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "current_wrist_obs": recursive_to_device(self.current_wrist_obs, "cpu")
            if self.current_wrist_obs is not None
            else None,
            "current_extra_view_obs": recursive_to_device(
                self.current_extra_view_obs, "cpu"
            )
            if self.current_extra_view_obs is not None
            else None,
            "current_latent": recursive_to_device(self.current_latent, "cpu")
            if self.current_latent is not None
            else None,
            "history_latents": recursive_to_device(self.history_latents, "cpu")
            if self.history_latents is not None
            else None,
            "history_latent_bank": recursive_to_device(self.history_latent_bank, "cpu")
            if self.history_latent_bank is not None
            else None,
            "current_states": recursive_to_device(self.current_states, "cpu")
            if self.current_states is not None
            else None,
            "current_policy_states": recursive_to_device(
                self.current_policy_states, "cpu"
            )
            if self.current_policy_states is not None
            else None,
            "joint_fk_to_ctrl_alignment": recursive_to_device(
                self.joint_fk_to_ctrl_alignment, "cpu"
            )
            if self.joint_fk_to_ctrl_alignment is not None
            else None,
            "action_history": self.action_history.cpu(),
            "action_history_bank": self.action_history_bank.cpu()
            if self.action_history_bank is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps.cpu(),
            "prev_step_reward": self.prev_step_reward.cpu(),
            "success_already_emitted": self.success_already_emitted.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
            "ordered_reset_cursor": self._ordered_reset_cursor,
            "common_noise_call_index": self._common_noise_call_index,
            "common_noise_rollout_index": self._common_noise_rollout_index,
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Restore serialized environment state produced by `get_state`."""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.current_wrist_obs = (
            recursive_to_device(state["current_wrist_obs"], self.device)
            if state["current_wrist_obs"] is not None
            else None
        )
        self.current_extra_view_obs = (
            recursive_to_device(state.get("current_extra_view_obs"), self.device)
            if state.get("current_extra_view_obs") is not None
            else None
        )
        self.current_latent = (
            recursive_to_device(state["current_latent"], self.device)
            if state["current_latent"] is not None
            else None
        )
        self.history_latents = (
            recursive_to_device(state["history_latents"], self.device)
            if state["history_latents"] is not None
            else None
        )
        self.history_latent_bank = (
            recursive_to_device(state.get("history_latent_bank"), self.device)
            if state.get("history_latent_bank") is not None
            else None
        )
        self.current_states = (
            recursive_to_device(state["current_states"], self.device)
            if state.get("current_states") is not None
            else None
        )
        if self.current_states is not None:
            self.current_states = self._align_policy_state_dim(self.current_states)
        self.current_policy_states = (
            recursive_to_device(state.get("current_policy_states"), self.device)
            if state.get("current_policy_states") is not None
            else self.current_states
        )
        self.joint_fk_to_ctrl_alignment = (
            recursive_to_device(state.get("joint_fk_to_ctrl_alignment"), self.device)
            if state.get("joint_fk_to_ctrl_alignment") is not None
            else None
        )
        self.action_history = state["action_history"].to(self.device)
        self.action_history_bank = (
            state.get("action_history_bank").to(self.device)
            if state.get("action_history_bank") is not None
            else None
        )
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"].to(self.device)
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self.success_already_emitted = state.get(
            "success_already_emitted", torch.zeros(self.num_envs, dtype=torch.bool)
        ).to(self.device)
        self._is_start = state["_is_start"]
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])
        self._ordered_reset_cursor = int(state.get("ordered_reset_cursor", 0))
        self._common_noise_call_index = int(
            state.get("common_noise_call_index", 0)
        )
        self._common_noise_rollout_index = int(
            state.get("common_noise_rollout_index", -1)
        )

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)


class CosmosSelfFeedbackCtrlWorldEnv(CtrlWorldEnv):
    """Ctrl-World comparison env with Cosmos's imagined frame as policy feedback.

    Ctrl-World state, latent, and history still advance independently through
    :class:`CtrlWorldEnv`. Only the observation returned to the policy is
    augmented with the last frame of Cosmos's just-generated video.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feedback_cfg = self.cfg.get("self_feedback", {})
        if not _as_bool(feedback_cfg.get("enabled", False)):
            raise ValueError("cosmos_self_ctrl_world requires self_feedback.enabled=true")
        if str(feedback_cfg.get("source", "")) != "imagined_video_chunk":
            raise ValueError(
                "cosmos_self_ctrl_world requires "
                "self_feedback.source='imagined_video_chunk'"
            )
        if self.auto_reset:
            raise ValueError("cosmos_self_ctrl_world requires auto_reset=false")
        self.self_feedback_frame_index = int(feedback_cfg.get("frame_index", -1))
        self.self_feedback_strict = _as_bool(feedback_cfg.get("strict", True))
        target_size = feedback_cfg.get("target_size", None)
        if target_size is not None and (
            len(target_size) != 2 or any(int(value) <= 0 for value in target_size)
        ):
            raise ValueError(
                "self_feedback.target_size must be [height, width] with positive values"
            )
        self.self_feedback_target_size = (
            tuple(int(value) for value in target_size)
            if target_size is not None
            else None
        )
        self._pending_policy_feedback = None
        self._policy_composite_images = None
        self._self_feedback_chunk_index = 0

    def set_policy_feedback(self, imagined_video_chunk: torch.Tensor) -> None:
        """Stage the current Cosmos video for use by the next policy chunk."""
        if not isinstance(imagined_video_chunk, torch.Tensor):
            raise TypeError("imagined_video_chunk must be a torch.Tensor")
        expected_shape = "[B,T,H,W,3]"
        if imagined_video_chunk.ndim != 5 or imagined_video_chunk.shape[-1] != 3:
            raise ValueError(
                f"imagined_video_chunk must be {expected_shape}, got "
                f"{tuple(imagined_video_chunk.shape)}"
            )
        if imagined_video_chunk.dtype != torch.uint8:
            raise TypeError(
                "imagined_video_chunk must be uint8, got "
                f"{imagined_video_chunk.dtype}"
            )
        if imagined_video_chunk.shape[0] != self.num_envs:
            raise ValueError(
                "imagined_video_chunk batch must equal num_envs: "
                f"{imagined_video_chunk.shape[0]} != {self.num_envs}"
            )
        if imagined_video_chunk.shape[1] != self.policy_chunk:
            raise ValueError(
                "imagined_video_chunk time dimension must equal the Cosmos chunk: "
                f"{imagined_video_chunk.shape[1]} != {self.policy_chunk}"
            )
        if imagined_video_chunk.shape[1] == 0:
            raise ValueError("imagined_video_chunk must contain at least one frame")
        frame_index = self.self_feedback_frame_index
        if not -imagined_video_chunk.shape[1] <= frame_index < imagined_video_chunk.shape[1]:
            raise IndexError(
                f"self_feedback.frame_index={frame_index} is out of range for "
                f"T={imagined_video_chunk.shape[1]}"
            )
        generated_size = tuple(int(value) for value in imagined_video_chunk.shape[2:4])
        if (
            self.self_feedback_strict
            and self.self_feedback_target_size is not None
            and generated_size != self.self_feedback_target_size
        ):
            raise ValueError(
                "Strict Cosmos self-feedback requires generated frames to match "
                "target_size exactly before Ctrl-World advances: "
                f"generated={generated_size}, target={self.self_feedback_target_size}."
            )
        self._pending_policy_feedback = imagined_video_chunk.detach().cpu().contiguous()

    def reset(self, *args, **kwargs):
        self._pending_policy_feedback = None
        self._policy_composite_images = None
        self._self_feedback_chunk_index = 0
        return super().reset(*args, **kwargs)

    @staticmethod
    def _last_frame_mse(cosmos_image: torch.Tensor, ctrl_video: torch.Tensor) -> torch.Tensor:
        ctrl_last = ctrl_video[:, -1]
        if ctrl_last.ndim != 4:
            raise ValueError(f"Ctrl-World video must expose 4D frames, got {ctrl_last.shape}")
        if ctrl_last.shape[-1] in (1, 3):
            ctrl_last = ctrl_last.permute(0, 3, 1, 2)
        cosmos_chw = cosmos_image.permute(0, 3, 1, 2).to(torch.float32)
        ctrl_chw = ctrl_last.to(
            device=cosmos_chw.device, dtype=torch.float32
        )
        ctrl_chw = F.interpolate(
            ctrl_chw, size=cosmos_chw.shape[-2:], mode="bilinear", align_corners=False
        )
        return ((cosmos_chw - ctrl_chw) / 255.0).square().mean()

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        if self._pending_policy_feedback is None:
            raise RuntimeError(
                "cosmos_self_ctrl_world requires set_policy_feedback() before every chunk_step()"
            )
        feedback = self._pending_policy_feedback
        result = super().chunk_step(policy_output_action)
        obs_list, rewards, terminations, truncations, infos_list = result
        feedback_image = feedback[:, self.self_feedback_frame_index].contiguous()
        if (
            self.self_feedback_target_size is not None
            and tuple(feedback_image.shape[1:3]) != self.self_feedback_target_size
        ):
            if self.self_feedback_strict:
                raise ValueError(
                    "Strict Cosmos self-feedback requires the generated frame "
                    "to match target_size exactly: "
                    f"generated={tuple(feedback_image.shape[1:3])}, "
                    f"target={self.self_feedback_target_size}."
                )
            feedback_chw = feedback_image.permute(0, 3, 1, 2).to(torch.float32)
            feedback_image = F.interpolate(
                feedback_chw,
                size=self.self_feedback_target_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            feedback_image = (
                feedback_image.round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        obs_list[-1]["policy_composite_images"] = feedback_image
        infos = infos_list[-1]
        ctrl_video = infos.get("ctrl_world_video_chunk")
        if not isinstance(ctrl_video, torch.Tensor):
            raise RuntimeError("Ctrl-World comparison video is missing from chunk infos")
        values = feedback_image.to(torch.float32)
        infos["self_feedback/cosmos_input_min"] = values.min().reshape(1)
        infos["self_feedback/cosmos_input_max"] = values.max().reshape(1)
        infos["self_feedback/cosmos_input_checksum"] = values.sum().reshape(1)
        infos["self_feedback/cosmos_vs_ctrl_last_frame_mse"] = self._last_frame_mse(
            feedback_image, ctrl_video
        ).cpu().reshape(1)
        infos["self_feedback/chunk_index"] = torch.tensor(
            [self._self_feedback_chunk_index], dtype=torch.float32
        )
        self._policy_composite_images = feedback_image
        self._pending_policy_feedback = None
        self._self_feedback_chunk_index += 1
        return obs_list, rewards, terminations, truncations, infos_list
