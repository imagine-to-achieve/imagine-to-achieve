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

import torch

from rlinf.models.embodiment.cosmos.fpo import (
    COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD,
    resolve_cosmos_replay_objective,
)
from rlinf.models.embodiment.cosmos.logprob_replay import (
    validate_action_chain_replay_tensors,
)


REQUIRED_COSMOS_FORWARD_INPUT_KEYS = (
    "action",
    "model_action",
    "action_chains",
    "action_denoise_timesteps",
    "action_sigmas",
    "action_mask",
    "action_condition_main_images",
    "action_condition_states",
)
REQUIRED_COSMOS_FPO_FORWARD_INPUT_KEYS = (
    "action",
    "model_action",
    "imagined_video_chunk",
    "action_sampling_seed",
    "fpo_clean_joint_latent",
    "fpo_clean_action_normalized",
    "fpo_base_times",
    "fpo_sigmas",
    "fpo_timesteps",
    "fpo_noise_seeds",
    "action_condition_main_images",
    "action_condition_states",
)


def validate_cosmos_fpo_forward_inputs(
    forward_inputs: dict[str, torch.Tensor],
) -> None:
    """Validate the fixed-pair joint-context FPO replay tensors."""

    for key in (
        "fpo_clean_joint_latent",
        "fpo_clean_action_normalized",
        "fpo_base_times",
        "fpo_sigmas",
        "fpo_timesteps",
        "fpo_noise_seeds",
    ):
        if forward_inputs[key].ndim != 2:
            raise ValueError(
                f"Cosmos FPO field {key!r} must have two dimensions, got "
                f"{tuple(forward_inputs[key].shape)}."
            )

    metadata_shape = tuple(forward_inputs["fpo_base_times"].shape)
    if metadata_shape[1] <= 0:
        raise ValueError("Cosmos FPO replay requires at least one MC pair.")
    for key in ("fpo_sigmas", "fpo_timesteps", "fpo_noise_seeds"):
        if tuple(forward_inputs[key].shape) != metadata_shape:
            raise ValueError(
                "Cosmos FPO metadata shapes differ: "
                f"expected {metadata_shape}, got "
                f"{tuple(forward_inputs[key].shape)} for {key!r}."
            )
    if forward_inputs["fpo_noise_seeds"].dtype != torch.int64:
        raise TypeError("Cosmos FPO noise seeds must use torch.int64.")

    finite_fields = (
        "fpo_clean_joint_latent",
        "fpo_clean_action_normalized",
        "fpo_base_times",
        "fpo_sigmas",
        "fpo_timesteps",
    )
    for key in finite_fields:
        if not torch.isfinite(forward_inputs[key]).all():
            raise ValueError(f"Cosmos FPO field {key!r} contains NaN or Inf.")
    for key in ("fpo_base_times", "fpo_sigmas"):
        value = forward_inputs[key]
        if not ((value > 0.0) & (value < 1.0)).all():
            raise ValueError(
                f"Cosmos FPO field {key!r} must lie strictly inside (0, 1)."
            )
    if not (forward_inputs["fpo_timesteps"] > 0.0).all():
        raise ValueError("Cosmos FPO timesteps must be positive.")
    sampling_seed = forward_inputs["action_sampling_seed"]
    if sampling_seed.dtype != torch.int64:
        raise TypeError("Cosmos action_sampling_seed must use torch.int64.")


def validate_cosmos_forward_inputs(
    forward_inputs: dict[str, torch.Tensor],
    *,
    required_keys: tuple[str, ...] | None = None,
    replay_objective: str | None = None,
) -> None:
    """Validate one explicit flat Cosmos replay payload contract."""

    if not isinstance(forward_inputs, dict):
        raise TypeError(
            "Cosmos forward_inputs must be a flat dict[str, torch.Tensor], "
            f"got {type(forward_inputs).__name__}."
        )

    objective = (
        resolve_cosmos_replay_objective(replay_objective)
        if replay_objective is not None
        else None
    )
    if required_keys is None:
        if (
            objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            or (
                objective is None
                and "fpo_clean_joint_latent" in forward_inputs
            )
        ):
            required_keys = REQUIRED_COSMOS_FPO_FORWARD_INPUT_KEYS
        else:
            required_keys = REQUIRED_COSMOS_FORWARD_INPUT_KEYS

    missing_keys = [key for key in required_keys if key not in forward_inputs]
    if missing_keys:
        raise KeyError(
            "Cosmos forward_inputs missing required replay keys: "
            f"{missing_keys}. Available keys: {sorted(forward_inputs.keys())}"
        )

    batch_size = None
    for key, value in forward_inputs.items():
        if not torch.is_tensor(value):
            raise TypeError(
                "Cosmos forward_inputs must stay flat with tensor values; "
                f"key '{key}' has {type(value).__name__}."
            )
        if value.dim() == 0:
            raise ValueError(
                f"Cosmos forward_inputs['{key}'] must include a batch dimension."
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif int(value.shape[0]) != batch_size:
            raise ValueError(
                "Cosmos forward_inputs batch dimensions differ: "
                f"expected {batch_size}, got {value.shape[0]} for key '{key}'."
            )

    if (
        objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
        or "fpo_clean_joint_latent" in forward_inputs
    ):
        validate_cosmos_fpo_forward_inputs(forward_inputs)
        return
    validate_action_chain_replay_tensors(
        action_chains=forward_inputs["action_chains"],
        action_denoise_timesteps=forward_inputs["action_denoise_timesteps"],
        action_sigmas=forward_inputs["action_sigmas"],
        action_mask=forward_inputs["action_mask"],
    )
    if "action_denoise_indices" in forward_inputs:
        indices = forward_inputs["action_denoise_indices"]
        num_steps = int(forward_inputs["action_chains"].shape[1]) - 1
        if indices.shape != (batch_size, 1):
            raise ValueError(
                "forward_inputs['action_denoise_indices'] must have shape [B, 1], "
                f"got {tuple(indices.shape)}."
            )
        if bool(((indices < 0) | (indices >= num_steps)).any().item()):
            raise ValueError(
                "forward_inputs['action_denoise_indices'] must be within "
                f"[0, {num_steps})."
            )
