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

"""Cosmos training-mode registry and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_ONLY_GRPO = "action_only_grpo"
DEFAULT_COSMOS_TRAINING_MODE = ACTION_ONLY_GRPO


@dataclass(frozen=True)
class CosmosTrainingModeSpec:
    """Configuration contract for a Cosmos training mode."""

    name: str
    enabled: bool
    trainable_paths: tuple[str, ...]
    logprob_sources: tuple[str, ...]
    reward_sources: tuple[str, ...]
    replay_payload_keys: tuple[str, ...]
    loss_terms: tuple[str, ...]
    disabled_reason: str | None = None


ACTION_REPLAY_KEYS = (
    "action",
    "model_action",
    "action_chains",
    "action_denoise_timesteps",
    "action_denoise_indices",
    "action_sigmas",
    "action_mask",
    "action_condition_main_images",
    "action_condition_states",
)


_PLACEHOLDER_REASON = (
    "This Cosmos training mode is not yet implemented. Select "
    "cosmos.training_mode=action_only_grpo for the current action-only GRPO "
    "path."
)


COSMOS_TRAINING_MODE_REGISTRY: dict[str, CosmosTrainingModeSpec] = {
    ACTION_ONLY_GRPO: CosmosTrainingModeSpec(
        name=ACTION_ONLY_GRPO,
        enabled=True,
        trainable_paths=(
            "action_proj_in",
            "action_proj_out",
            "action_modality_embed",
            "moe_gen_ln_mlp",
        ),
        logprob_sources=("action_chain_replay",),
        reward_sources=("video_similarity", "success_model"),
        replay_payload_keys=ACTION_REPLAY_KEYS,
        loss_terms=("action_grpo",),
    ),
    "action_only_grpo_mixed_reward": CosmosTrainingModeSpec(
        name="action_only_grpo_mixed_reward",
        enabled=False,
        trainable_paths=("llm2action", "action_lora"),
        logprob_sources=("action_chain_replay",),
        reward_sources=("video_similarity", "success_or_progress"),
        replay_payload_keys=ACTION_REPLAY_KEYS,
        loss_terms=("action_grpo",),
        disabled_reason=_PLACEHOLDER_REASON,
    ),
    "inverse_dynamics_sft_or_fm": CosmosTrainingModeSpec(
        name="inverse_dynamics_sft_or_fm",
        enabled=False,
        trainable_paths=("inverse_dynamics", "llm2action"),
        logprob_sources=("supervised_or_flow_matching",),
        reward_sources=(),
        replay_payload_keys=(),
        loss_terms=("sft_or_fm",),
        disabled_reason=_PLACEHOLDER_REASON,
    ),
    "eval_or_replay_only": CosmosTrainingModeSpec(
        name="eval_or_replay_only",
        enabled=False,
        trainable_paths=(),
        logprob_sources=("action_chain_replay",),
        reward_sources=("video_similarity",),
        replay_payload_keys=ACTION_REPLAY_KEYS,
        loss_terms=(),
        disabled_reason=_PLACEHOLDER_REASON,
    ),
    "video_only_grpo": CosmosTrainingModeSpec(
        name="video_only_grpo",
        enabled=False,
        trainable_paths=("video", "forward_dynamics"),
        logprob_sources=("video_chain_replay",),
        reward_sources=("video_similarity",),
        replay_payload_keys=("video_chains", "video_denoise_timesteps", "video_sigmas"),
        loss_terms=("video_grpo",),
        disabled_reason=_PLACEHOLDER_REASON,
    ),
    "joint_action_video_grpo": CosmosTrainingModeSpec(
        name="joint_action_video_grpo",
        enabled=False,
        trainable_paths=("llm2action", "action_lora", "video", "forward_dynamics"),
        logprob_sources=("action_chain_replay", "video_chain_replay"),
        reward_sources=("video_similarity",),
        replay_payload_keys=ACTION_REPLAY_KEYS
        + ("video_chains", "video_denoise_timesteps", "video_sigmas"),
        loss_terms=("action_grpo", "video_grpo"),
        disabled_reason=_PLACEHOLDER_REASON,
    ),
}


def get_cosmos_training_mode_spec(mode: str) -> CosmosTrainingModeSpec:
    """Return a registered Cosmos training-mode spec or fail explicitly."""

    try:
        return COSMOS_TRAINING_MODE_REGISTRY[str(mode)]
    except KeyError as exc:
        supported = ", ".join(sorted(COSMOS_TRAINING_MODE_REGISTRY))
        raise ValueError(
            f"Unsupported cosmos.training_mode={mode!r}. Supported modes: {supported}."
        ) from exc


def validate_cosmos_training_mode(mode: str | None) -> CosmosTrainingModeSpec:
    """Validate the selected Cosmos training mode."""

    selected_mode = mode or DEFAULT_COSMOS_TRAINING_MODE
    spec = get_cosmos_training_mode_spec(selected_mode)
    if not spec.enabled:
        reason = spec.disabled_reason or "Mode is registered but not enabled."
        raise NotImplementedError(
            f"cosmos.training_mode={selected_mode!r} is not enabled. {reason}"
        )
    return spec


def cfg_get_training_mode(cosmos_cfg: Any) -> str | None:
    """Read training_mode from DictConfig, dict, or object-like configs."""

    if cosmos_cfg is None:
        return None
    if hasattr(cosmos_cfg, "get"):
        return cosmos_cfg.get("training_mode", None)
    return getattr(cosmos_cfg, "training_mode", None)
