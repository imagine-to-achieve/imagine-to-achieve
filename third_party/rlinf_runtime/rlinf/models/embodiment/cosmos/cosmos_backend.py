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

"""Lazy boundary for native Cosmos3 inference."""

from __future__ import annotations

import base64
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
import fnmatch
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from rlinf.algorithms.cross_rank_grpo import (
    derive_stable_seed,
    map_semantic_seed_to_uint32,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.cosmos.action_adapter import (
    validate_direct_action_adapter,
)
from rlinf.models.embodiment.cosmos.camera_layout import (
    stitch_cosmos_three_view_chw,
)
from rlinf.models.embodiment.cosmos.fpo import (
    COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD,
    build_fpo_joint_noise_mask,
    build_fpo_mc_metadata,
    extract_fpo_normalized_action,
    fpo_action_epsilon_mse,
    fpo_action_velocity_mse,
    resolve_cosmos_replay_objective,
    seeded_fpo_epsilon,
)
from rlinf.models.embodiment.cosmos.logprob_replay import gaussian_logprob
from rlinf.models.embodiment.cosmos.replay_contract import (
    validate_cosmos_forward_inputs,
)
from rlinf.utils.logging import get_logger, quiet_third_party_progress_bars

# cosmos-framework's own loguru setup (cosmos_framework/utils/log.py) reads
# LOGURU_LEVEL at import time. Setting it here (rather than only via the
# Hydra/Ray runtime_env env_vars) guarantees it lands in os.environ before
# this process's *first* `import cosmos_framework...` happens, regardless of
# whether Ray's runtime_env env var propagation timing lines up with that
# first import. Set LOGURU_LEVEL=INFO (or unset) in the job env to restore
# cosmos-framework's per-request action-server logs for debugging.
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

quiet_third_party_progress_bars()


_SUPPORTED_COSMOS_FRAMEWORK_VERSION = "1.2.2"
_NUMPY_RANDOM_STATE_MAX_SEED = (1 << 32) - 1
_TORCH_DISTRIBUTED_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")


def _ensure_cosmos_dcp_hybrid_process_group() -> bool:
    """Route DCP object collectives over Gloo in distributed Cosmos workers.

    Native rollout workers construct the Cosmos service before any other RLinf
    component initializes torch.distributed.  If Cosmos is allowed to perform
    that initialization implicitly, its DCP load uses a plain NCCL default
    group and ``gather_object`` fails once the ranks span physical nodes.  The
    actor and Edge rollout paths already use this same hybrid backend.

    Local/unit-test construction has no distributed rendezvous environment and
    intentionally remains a no-op.
    """

    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return False
    if not all(os.environ.get(key) for key in _TORCH_DISTRIBUTED_ENV_KEYS):
        return False
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")
    return True


def _cosmos_numpy_noise_seed(namespace: str, seed: int) -> int:
    """Map an int63 experiment seed into pinned Cosmos' uint32 RNG domain."""
    mapped = map_semantic_seed_to_uint32(namespace, int(seed))
    assert mapped <= _NUMPY_RANDOM_STATE_MAX_SEED
    return mapped


def _cosmos_framework_source_version(model_module: Any) -> str | None:
    """Read the source checkout version without importing packaging helpers."""
    module_file = getattr(model_module, "__file__", None)
    if module_file is None:
        return None
    for parent in Path(module_file).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        return str(project.get("version")) if project.get("version") else None
    return None


def _validate_cosmos_framework_version(model_module: Any) -> None:
    """Fail closed when the compatibility hook sees an unknown Cosmos source."""
    module_name = str(getattr(model_module, "__name__", ""))
    if not module_name.startswith("cosmos_framework."):
        # Unit-test fakes deliberately live outside the third-party namespace.
        return
    version = _cosmos_framework_source_version(model_module)
    if version != _SUPPORTED_COSMOS_FRAMEWORK_VERSION:
        raise RuntimeError(
            "Unsupported Cosmos framework version for split modality seeds: "
            f"expected {_SUPPORTED_COSMOS_FRAMEWORK_VERSION}, found {version!r}"
        )


@contextmanager
def _split_cosmos_modality_noise_seeds(
    model: Any,
    *,
    vision_seed: int,
    action_seed: int,
    sampler_name: str,
):
    """Adapt the pinned single-seed inference API at its noise creation point."""
    if str(sampler_name).lower() != "unipc":
        raise ValueError(
            "cross-rank Cosmos modality seeds currently require sampler=unipc"
        )
    prepare = getattr(model, "_prepare_inference_data", None)
    if prepare is None:
        raise RuntimeError(
            "Pinned Cosmos model has no _prepare_inference_data compatibility hook"
        )
    parameters = tuple(inspect.signature(prepare).parameters)
    if parameters != ("data_batch", "seed", "has_negative_prompt"):
        raise RuntimeError(
            "Unsupported pinned Cosmos _prepare_inference_data signature: "
            f"{parameters}"
        )
    if bool(getattr(getattr(model, "config", None), "sound_gen", False)):
        raise ValueError("cross-rank modality seeds do not support sound generation")

    model_module = importlib.import_module(type(model).__module__)
    _validate_cosmos_framework_version(model_module)
    misc = getattr(model_module, "misc", None)
    original = getattr(misc, "arch_invariant_rand", None)
    if original is None:
        raise RuntimeError(
            "Pinned Cosmos module no longer exposes misc.arch_invariant_rand"
        )
    seen = {"vision": 0, "action": 0}
    if int(vision_seed) == int(action_seed):
        # Native joint video/action exploration is keyed by one semantic
        # seed.  Use one identical framework seed for both modality tensors;
        # shape differences still produce their own Gaussian arrays, while
        # the seed manifest exactly describes the joint sample.
        joint_seed = _cosmos_numpy_noise_seed(
            "cosmos_joint_initial_noise", vision_seed
        )
        compatible_seeds = {"vision": joint_seed, "action": joint_seed}
    else:
        compatible_seeds = {
            "vision": _cosmos_numpy_noise_seed(
                "cosmos_vision_initial_noise", vision_seed
            ),
            "action": _cosmos_numpy_noise_seed(
                "cosmos_action_initial_noise", action_seed
            ),
        }

    def seeded_arch_invariant_rand(shape, dtype, device, seed):
        shape = tuple(shape)
        if len(shape) in (4, 5):
            effective_seed = compatible_seeds["vision"]
            seen["vision"] += 1
        elif len(shape) in (2, 3):
            effective_seed = compatible_seeds["action"]
            seen["action"] += 1
        else:
            raise RuntimeError(
                "Unsupported Cosmos modality noise shape under split seed adapter: "
                f"{shape}"
            )
        return original(shape, dtype, device, effective_seed)

    misc.arch_invariant_rand = seeded_arch_invariant_rand
    try:
        yield
        if seen["vision"] == 0 or seen["action"] == 0:
            raise RuntimeError(
                "Cosmos split seed adapter did not observe both vision and action "
                f"noise creation: {seen}"
            )
    finally:
        misc.arch_invariant_rand = original


def _quiet_third_party_training_logs() -> None:
    if os.environ.get("RLINF_QUIET_THIRD_PARTY_LOGS", "1") in ("0", "false", "False"):
        return
    level_name = os.environ.get("LOGURU_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    for logger_name in (
        "cosmos_framework",
        "multistorageclient",
        "diffusers",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(level)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level=level_name)
    except Exception:
        pass


_quiet_third_party_training_logs()


COSMOS_FRAMEWORK_IMPORTS = (
    "cosmos_framework",
    "cosmos_framework.scripts.action_policy_server_libero",
    "cosmos_framework.model.vfm.omni_mot_model",
)
COSMOS_FRAMEWORK_IMPORT_ALTERNATIVES = {
    "cosmos_framework.model.vfm.omni_mot_model":
        "cosmos_framework.model.generator.omni_mot_model",
}


class CosmosNativeBackendUnavailable(RuntimeError):
    """Raised when native Cosmos3 prerequisites are not available."""


@dataclass(frozen=True)
class CosmosActionContract:
    """Checkpoint-specific action and conditioning contract.

    ``action_dim`` alone cannot identify a Cosmos policy: the UR5 joint model
    and an EEF model both use seven channels but have different units,
    conditioning and post-processing requirements.
    """

    name: str
    action_dim: int
    action_state_rows: int = 0
    domain_name: str | None = None
    camera_layout: str = "main_top"
    normalization: str = "auto"
    training_transform_compatible: bool = False
    requires_joint_domain_patch: bool = False
    resolution_tier: str | int | None = None


_DROID_CONCAT_VIEW_PROMPT_METADATA = (
    "This video contains concatenated views from multiple camera perspectives. "
    "The top row is the wrist view. The bottom row contains two horizontally "
    "concatenated external views: front view on the left and right-side view on "
    "the right."
)

_DROID_TRAINING_ADDITIONAL_VIEW_DESCRIPTION = (
    "The top row is the wrist view. The bottom row contains two horizontally "
    "concatenated external views: front view on the left and right-side view on "
    "the right."
)


def _append_droid_concat_view_prompt_metadata(prompt: str) -> str:
    """Match the 7D UR5 training transform's DROID viewpoint text.

    The joint checkpoint was trained with ``append_viewpoint_info=true``.
    Its dataset appended this exact camera-layout description before FPS and
    resolution metadata. Keep this separate from the legacy 10D path.
    """
    prompt = str(prompt).rstrip()
    if _DROID_CONCAT_VIEW_PROMPT_METADATA in prompt:
        return prompt
    if not prompt:
        return _DROID_CONCAT_VIEW_PROMPT_METADATA
    separator = " " if prompt.endswith(".") else ". "
    return prompt + separator + _DROID_CONCAT_VIEW_PROMPT_METADATA


def _select_policy_input_views(
    env_obs: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Select optional policy-resolution views without changing legacy inputs.

    Ctrl-World keeps its internal observations at the world model's native
    resolution. A caller may additionally expose the original reset frames as
    ``policy_*_images`` so Cosmos can condition its first chunk on the same
    full-resolution images used during SFT. The three override views form one
    atomic contract: partially overriding the camera tuple would silently mix
    resolutions and camera sources.
    """
    policy_composite = env_obs.get("policy_composite_images")
    if torch.is_tensor(policy_composite):
        if policy_composite.ndim != 4 or policy_composite.shape[-1] != 3:
            raise ValueError(
                "policy_composite_images must be [B,H,W,3], got "
                f"{tuple(policy_composite.shape)}"
            )
        return policy_composite, None, None
    override_keys = (
        "policy_main_images",
        "policy_wrist_images",
        "policy_extra_view_images",
    )
    override_values = tuple(env_obs.get(key) for key in override_keys)
    override_present = tuple(torch.is_tensor(value) for value in override_values)
    if any(override_present):
        if not all(override_present):
            missing = [
                key
                for key, present in zip(override_keys, override_present, strict=True)
                if not present
            ]
            raise ValueError(
                "Policy-resolution camera overrides must provide all three views; "
                f"missing {missing}."
            )
        return override_values
    return (
        env_obs.get("main_images"),
        env_obs.get("wrist_images"),
        env_obs.get("extra_view_images"),
    )


def _build_policy_condition_video(
    image_chw_uint8: torch.Tensor,
    *,
    t_frames: int,
    zero_future_frames: bool,
) -> torch.Tensor:
    """Build the policy video tensor with explicit future-frame semantics."""
    if not zero_future_frames:
        return image_chw_uint8.unsqueeze(1).repeat(1, t_frames, 1, 1)
    video = torch.zeros(
        (image_chw_uint8.shape[0], t_frames, *image_chw_uint8.shape[1:]),
        dtype=image_chw_uint8.dtype,
        device=image_chw_uint8.device,
    )
    video[:, 0] = image_chw_uint8
    return video


def _augment_prompt_like_action_training_transform(
    prompt: str,
    *,
    t_frames: int,
    fps: int,
    padded_height: int,
    padded_width: int,
    append_duration_fps: bool,
    append_resolution_info: bool,
) -> str:
    """Mirror the SFT ActionTransformPipeline metadata formatting exactly."""
    if append_duration_fps:
        # DurationFPSTextTimeStamps truncates to whole seconds before applying
        # its one-decimal display format.
        duration = int(t_frames / fps)
        separator = " " if prompt.rstrip().endswith(".") else ". "
        prompt += separator + (
            f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        )
    if append_resolution_info:
        separator = " " if prompt.rstrip().endswith(".") else ". "
        prompt += separator + (
            f"This video is of {padded_height}x{padded_width} resolution."
        )
    return prompt


_COSMOS_ACTION_CONTRACTS = {
    "ur5_joint_absolute_7d": CosmosActionContract(
        name="ur5_joint_absolute_7d",
        action_dim=7,
        action_state_rows=1,
        domain_name="robomind-ur-joint",
        camera_layout="droid",
        normalization="none",
        training_transform_compatible=True,
        requires_joint_domain_patch=True,
        resolution_tier=720,
    ),
    "ur5_joint_absolute_7d_no_state": CosmosActionContract(
        name="ur5_joint_absolute_7d_no_state",
        action_dim=7,
        action_state_rows=0,
        domain_name="robomind-ur-joint",
        camera_layout="droid",
        normalization="none",
        training_transform_compatible=True,
        requires_joint_domain_patch=True,
        resolution_tier=720,
    ),
    "ur5_eef_relative_10d": CosmosActionContract(
        name="ur5_eef_relative_10d", action_dim=10
    ),
    "ur5_eef_relative_10d_droid_native": CosmosActionContract(
        name="ur5_eef_relative_10d_droid_native",
        action_dim=10,
        domain_name="robomind-ur",
        camera_layout="droid",
        training_transform_compatible=True,
        resolution_tier=720,
    ),
    "ur5_eef_relative_10d_edge4b": CosmosActionContract(
        name="ur5_eef_relative_10d_edge4b",
        action_dim=10,
        domain_name="robomind-ur",
        camera_layout="main_top",
        normalization="mixed_affine",
        training_transform_compatible=True,
        resolution_tier=480,
    ),
    "libero_delta_eef_7d": CosmosActionContract(
        name="libero_delta_eef_7d", action_dim=7
    ),
}


def resolve_cosmos_action_contract(
    action_representation: str, *, action_dim: int
) -> CosmosActionContract:
    """Resolve an explicit action contract and reject ambiguous dimensions."""

    try:
        contract = _COSMOS_ACTION_CONTRACTS[str(action_representation)]
    except KeyError as exc:
        supported = ", ".join(sorted(_COSMOS_ACTION_CONTRACTS))
        raise ValueError(
            "Unsupported cosmos.action_representation="
            f"{action_representation!r}. Supported explicit contracts: {supported}."
        ) from exc
    if contract.action_dim != int(action_dim):
        raise ValueError(
            f"Cosmos contract {contract.name!r} requires action_dim="
            f"{contract.action_dim}, got {action_dim}."
        )
    return contract


def _resolve_action_normalization_for_service(
    requested_normalization: str, action_stats_path: Path | None
) -> str:
    """Translate RLinf's explicit raw-action contract to the server API.

    The pinned Cosmos action server only accepts normalization modes that
    *invert* statistics (or ``auto``), while a 7D checkpoint trained on raw
    absolute joint targets intentionally has no statistics to invert. With
    ``action_stats_path=None`` the server skips its normalization loader and
    returns the sampled actions unchanged; ``auto`` is therefore the API-safe
    spelling of RLinf's semantic ``none`` in that narrow case.
    """
    requested_normalization = str(requested_normalization).lower()
    if requested_normalization == "qnorm":
        if action_stats_path is None:
            raise ValueError(
                "action_normalization='qnorm' requires q01/q99 action stats"
            )
        # RLinf/config names the checkpoint contract qnorm. The pinned Cosmos
        # ActionServerArgs calls the identical q01/q99 affine transform
        # ``quantile`` and deliberately rejects unknown Literal values.
        return "quantile"
    if requested_normalization != "none":
        return requested_normalization
    if action_stats_path is not None:
        raise ValueError(
            "action_normalization='none' requires action_stats_path=None; "
            "otherwise the Cosmos server would need to denormalize the action."
        )
    return "auto"


@dataclass
class CosmosRecordedActionChain:
    """Flattened native sampling chain captured from a Cosmos sampler call."""

    full_chains: torch.Tensor
    transition_means: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor



DEFAULT_NATIVE_TRAINABLE_PARAM_PATTERNS = (
    "action_proj_in.*",
    "action_proj_out.*",
    "action_modality_embed",
    "layers.*.input_layernorm_moe_gen.*",
    "layers.*.mlp_moe_gen.*",
    "layers.*.post_attention_layernorm_moe_gen.*",
)


DEFAULT_NATIVE_BLOCKED_TRAINABLE_PARAM_PATTERNS = (
    "*.self_attn.*",
    "*.input_layernorm.*",
    "*.mlp.*",
    "*.post_attention_layernorm.*",
    "*embed_tokens*",
    "*vae2llm*",
    "*llm2vae*",
    "proj_in.*",
    "proj_out.*",
    "*visual*",
    "*vision*",
    "*vae*",
)


NATIVE_TRAINABLE_STATE_METADATA_KEY = "__cosmos_native_trainable_metadata__"


def _as_tuple_of_str(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[]")


def _alias_trainable_pattern(pattern: str) -> tuple[str, ...]:
    stripped = pattern.strip()
    aliases = [stripped]
    prefix_aliases = (
        ("action_proj_in.", "action2llm."),
        ("action_proj_out.", "llm2action."),
        ("proj_in.", "vae2llm."),
        ("proj_out.", "llm2vae."),
    )
    for source, target in prefix_aliases:
        if stripped.startswith(source):
            aliases.append(target + stripped[len(source) :])
    return tuple(dict.fromkeys(aliases))


def _parameter_name_matches(name: str, pattern: str) -> bool:
    if _has_glob(pattern):
        return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(
            name, f"*.{pattern}"
        )
    return name == pattern or name.endswith(f".{pattern}") or pattern in name


def _matches_any_parameter_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        for alias in _alias_trainable_pattern(pattern):
            if _parameter_name_matches(name, alias):
                return True
    return False


class NativeTrainableParameterProxy(nn.Module):
    def __init__(self, named_params: list[tuple[str, nn.Parameter]]) -> None:
        super().__init__()
        self.param_names = tuple(name for name, _ in named_params)
        self.params = nn.ParameterList([param for _, param in named_params])

    def named_service_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return list(zip(self.param_names, self.params, strict=True))


class CosmosNativeActionChainRecorder:
    """Small Euler sampler that records native flattened latent chains.

    The recorder is intentionally separate from the native inference
    default path. It proves where native chains can be captured, but full
    actor replay still needs tensorized condition reconstruction.
    """

    def __init__(
        self,
        *,
        chain_sigma: float = 0.2,
        sigma_min: float = 1e-4,
        timestep_scale: float = 1000.0,
    ) -> None:
        self.chain_sigma = float(chain_sigma)
        self.sigma_min = float(sigma_min)
        self.timestep_scale = float(timestep_scale)
        self.record: CosmosRecordedActionChain | None = None

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        del shift
        input_was_list = isinstance(noise, list)
        current = [item.clone() for item in noise] if input_was_list else [noise.clone()]
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")

        batch_size = len(current)
        device = current[0].device
        dtype = current[0].dtype
        step_size = 1.0 / float(num_steps)
        sigmas = torch.full(
            (batch_size, num_steps),
            max(self.chain_sigma, self.sigma_min),
            dtype=dtype,
            device=device,
        )
        timesteps_1d = torch.linspace(
            self.timestep_scale,
            0.0,
            num_steps + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        timesteps = timesteps_1d.unsqueeze(0).expand(batch_size, -1).contiguous()

        generators = _native_chain_generators(seed, batch_size, device)
        chain_states = [torch.stack([item.detach().clone() for item in current], dim=0)]
        transition_means = []
        for step_idx in range(num_steps):
            timestep = timesteps_1d[step_idx].reshape(1, 1)
            velocities = velocity_fn(current, timestep)
            next_states = []
            mean_states = []
            for sample_idx, (state, velocity) in enumerate(zip(current, velocities, strict=True)):
                mean = state + step_size * velocity.to(device=state.device, dtype=state.dtype)
                sigma = sigmas[sample_idx, step_idx]
                if sigma.item() > 0.0:
                    noise_i = torch.randn(
                        state.shape,
                        generator=generators[sample_idx],
                        dtype=state.dtype,
                        device=state.device,
                    )
                else:
                    noise_i = torch.zeros_like(state)
                next_state = mean + sigma * noise_i
                mean_states.append(mean.detach().clone())
                next_states.append(next_state)
            transition_means.append(torch.stack(mean_states, dim=0))
            current = next_states
            chain_states.append(torch.stack([item.detach().clone() for item in current], dim=0))

        self.record = CosmosRecordedActionChain(
            full_chains=torch.stack(chain_states, dim=1).contiguous(),
            transition_means=torch.stack(transition_means, dim=1).contiguous(),
            timesteps=timesteps,
            sigmas=sigmas.contiguous(),
        )
        return current if input_was_list else current[0]


class CosmosMatchedActionEulerGaussianSampler:
    """Action-only stochastic Euler rollout with an exactly matched replay."""

    def __init__(
        self,
        *,
        num_action_chunks: int,
        raw_action_dim: int,
        max_action_dim: int,
        action_state_rows: int = 0,
        chain_sigma: float = 0.2,
        sigma_min: float = 1.0e-4,
        timestep_scale: float = 1000.0,
        transition_noise_seed: int | list[int] | None = None,
    ) -> None:
        self.num_action_chunks = int(num_action_chunks)
        self.raw_action_dim = int(raw_action_dim)
        self.max_action_dim = int(max_action_dim)
        self.action_state_rows = int(action_state_rows)
        self.chain_sigma = float(chain_sigma)
        self.sigma_min = float(sigma_min)
        self.timestep_scale = float(timestep_scale)
        self.transition_noise_seed = transition_noise_seed
        self.record: CosmosRecordedActionChain | None = None
        if min(self.num_action_chunks, self.raw_action_dim) <= 0:
            raise ValueError("Matched Euler action dimensions must be positive.")
        if self.max_action_dim < self.raw_action_dim:
            raise ValueError("max_action_dim must cover raw_action_dim.")
        if self.action_state_rows < 0:
            raise ValueError("action_state_rows must be non-negative.")
        if min(self.chain_sigma, self.sigma_min) <= 0.0:
            raise ValueError("Matched Euler sigma values must be positive.")

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        del seed
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        input_was_list = isinstance(noise, list)
        current = list(noise) if input_was_list else [noise]
        current = [value.clone() for value in current]
        batch_size = len(current)
        device, dtype = current[0].device, current[0].dtype
        rows = self.action_state_rows + self.num_action_chunks
        action_width = rows * self.max_action_dim
        action_start = current[0].numel() - action_width
        if action_start < 0:
            raise ValueError("Matched Euler latent is narrower than its action suffix.")
        if any(value.numel() != current[0].numel() for value in current):
            raise ValueError("Matched Euler samples must have equal flat widths.")

        sigma_grid = torch.linspace(
            1.0,
            1.0 / self.timestep_scale,
            num_steps,
            device=device,
            dtype=torch.float32,
        )
        shift = 1.0 if shift is None else float(shift)
        sigma_grid = shift * sigma_grid / (
            1.0 + (shift - 1.0) * sigma_grid
        )
        sigma_grid = torch.cat([sigma_grid, sigma_grid.new_zeros(1)])
        score_sigma = max(self.chain_sigma, self.sigma_min)
        generators = _native_chain_generators(
            self.transition_noise_seed, batch_size, device
        )
        states = [torch.stack([value.detach().clone() for value in current])]
        means, timesteps = [], []
        h = 1.0 / float(num_steps)

        for step_idx in range(num_steps):
            sigma_cur, sigma_next = sigma_grid[step_idx : step_idx + 2]
            timestep = (sigma_cur * self.timestep_scale).to(dtype=dtype)
            velocities = velocity_fn(current, timestep.reshape(1, 1))
            next_states, step_means = [], []
            for sample_idx, (state, velocity) in enumerate(
                zip(current, velocities, strict=True)
            ):
                velocity = velocity.to(device=state.device, dtype=state.dtype)
                full_mean = state + (
                    sigma_next.to(dtype=dtype) - sigma_cur.to(dtype=dtype)
                ) * velocity
                state_rows = state[action_start:].reshape(
                    rows, self.max_action_dim
                )
                velocity_rows = velocity[action_start:].reshape(
                    rows, self.max_action_dim
                )
                action_mean = (
                    state_rows[self.action_state_rows :, : self.raw_action_dim]
                    + h
                    * velocity_rows[
                        self.action_state_rows :, : self.raw_action_dim
                    ]
                )
                mean_rows = full_mean[action_start:].reshape(
                    rows, self.max_action_dim
                )
                mean_rows[
                    self.action_state_rows :, : self.raw_action_dim
                ] = action_mean
                next_state = full_mean.clone()
                next_rows = next_state[action_start:].reshape(
                    rows, self.max_action_dim
                )
                action_noise = torch.randn(
                    action_mean.shape,
                    generator=generators[sample_idx],
                    device=state.device,
                    dtype=state.dtype,
                )
                next_rows[
                    self.action_state_rows :, : self.raw_action_dim
                ] = action_mean + score_sigma * action_noise
                step_means.append(full_mean.detach().clone())
                next_states.append(next_state)
            means.append(torch.stack(step_means))
            timesteps.append(timestep.expand(batch_size).detach().clone())
            current = next_states
            states.append(
                torch.stack([value.detach().clone() for value in current])
            )

        self.record = CosmosRecordedActionChain(
            full_chains=torch.stack(states, dim=1).contiguous(),
            transition_means=torch.stack(means, dim=1).contiguous(),
            timesteps=torch.stack(timesteps, dim=1).contiguous(),
            sigmas=torch.full(
                (batch_size, num_steps),
                score_sigma,
                device=device,
                dtype=dtype,
            ),
        )
        return current if input_was_list else current[0]


class CosmosNativeActionTraceSampler:
    """Wrapper that records a native sampler trace without changing sampling.

    The wrapped Cosmos sampler still owns the denoising algorithm and return
    value. RLinf only wraps velocity_fn to copy the states and model
    velocities that the official sampler actually used.
    """

    def __init__(
        self,
        base_sampler: Any,
        *,
        chain_sigma: float = 0.2,
        sigma_min: float = 1e-4,
    ) -> None:
        if base_sampler is None:
            raise ValueError("CosmosNativeActionTraceSampler requires a base sampler.")
        self.base_sampler = base_sampler
        self.chain_sigma = float(chain_sigma)
        self.sigma_min = float(sigma_min)
        self.record: CosmosRecordedActionChain | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_sampler, name)

    @staticmethod
    def _as_list(value: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(value, list):
            return value
        return list(torch.unbind(value, dim=0))

    @staticmethod
    def _restore_type(
        values: list[torch.Tensor],
        template: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(template, list):
            return values
        return torch.stack(values, dim=0)

    @staticmethod
    def _stack(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([value.detach().clone() for value in values], dim=0)

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        *args,
        num_steps: int = 35,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        states = []
        means = []
        timesteps = []
        sigmas = []
        step_size = 1.0 / float(num_steps)

        def traced_velocity_fn(current, timestep):
            current_list = self._as_list(current)
            velocity = velocity_fn(current, timestep)
            velocity_list = self._as_list(velocity)
            states.append(self._stack(current_list))
            mean_list = [
                state + step_size * vel.to(device=state.device, dtype=state.dtype)
                for state, vel in zip(current_list, velocity_list, strict=True)
            ]
            means.append(self._stack(mean_list))
            batch_size = len(current_list)
            timestep_value = timestep.reshape(-1)[0].to(
                device=current_list[0].device, dtype=current_list[0].dtype
            )
            timesteps.append(timestep_value.expand(batch_size).detach().clone())
            sigma_value = max(self.chain_sigma, self.sigma_min)
            sigmas.append(
                torch.full(
                    (batch_size,),
                    sigma_value,
                    dtype=current_list[0].dtype,
                    device=current_list[0].device,
                )
            )
            return self._restore_type(velocity_list, velocity)

        result = self.base_sampler(
            traced_velocity_fn,
            noise,
            *args,
            num_steps=num_steps,
            **kwargs,
        )
        if not states:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 trace sampler did not observe any denoise steps."
            )
        result_states = self._stack(self._as_list(result))
        self.record = CosmosRecordedActionChain(
            full_chains=torch.cat(
                [torch.stack(states, dim=1), result_states[:, None, :]], dim=1
            ).contiguous(),
            transition_means=torch.stack(means, dim=1).contiguous(),
            timesteps=torch.stack(timesteps, dim=1).contiguous(),
            sigmas=torch.stack(sigmas, dim=1).contiguous(),
        )
        return result

class CosmosNativeFinalLatentSampler:
    """Record only the native sampler output while preserving UniPC exactly."""

    def __init__(self, base_sampler: Any) -> None:
        if base_sampler is None:
            raise ValueError(
                "CosmosNativeFinalLatentSampler requires a base sampler."
            )
        self.base_sampler = base_sampler
        self.final_latents: torch.Tensor | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_sampler, name)

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        result = self.base_sampler(velocity_fn, noise, *args, **kwargs)
        result_list = (
            result
            if isinstance(result, list)
            else list(torch.unbind(result, dim=0))
        )
        self.final_latents = torch.stack(
            [value.detach().clone() for value in result_list], dim=0
        ).contiguous()
        return result

class CosmosActionHeadFPOSampler:
    """Evaluate fixed joint-context MC probes and score the raw action head."""

    def __init__(
        self,
        *,
        clean_joint_latent: torch.Tensor,
        clean_action_normalized: torch.Tensor,
        condition_image: torch.Tensor,
        base_times: torch.Tensor,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        noise_seeds: torch.Tensor,
        num_action_chunks: int,
        raw_action_dim: int,
        max_action_dim: int,
        action_state_rows: int = 0,
        latent_downsample_factor: int = 16,
        vision_state_channels: int = 48,
        vision_condition_latent_frames: int = 1,
        score_parameterization: str = "velocity",
    ) -> None:
        self.clean_joint_latent = clean_joint_latent
        self.clean_action_normalized = clean_action_normalized
        self.condition_image = condition_image
        self.base_times = base_times
        self.sigmas = sigmas
        self.timesteps = timesteps
        self.noise_seeds = noise_seeds
        self.num_action_chunks = int(num_action_chunks)
        self.raw_action_dim = int(raw_action_dim)
        self.max_action_dim = int(max_action_dim)
        self.action_state_rows = int(action_state_rows)
        self.latent_downsample_factor = int(latent_downsample_factor)
        self.vision_state_channels = int(vision_state_channels)
        self.vision_condition_latent_frames = int(
            vision_condition_latent_frames
        )
        self.score_parameterization = str(score_parameterization)
        self.pair_losses: torch.Tensor | None = None
        self.logprobs: torch.Tensor | None = None

        if self.score_parameterization not in {"velocity", "epsilon"}:
            raise ValueError(
                "FPO score_parameterization must be velocity or epsilon."
            )
        if self.clean_joint_latent.ndim != 2:
            raise ValueError(
                "FPO clean joint latent must have shape [B, width], got "
                f"{tuple(self.clean_joint_latent.shape)}."
            )
        batch_size = int(self.clean_joint_latent.shape[0])
        metadata_shape = None
        for name, value in (
            ("fpo_base_times", self.base_times),
            ("fpo_sigmas", self.sigmas),
            ("fpo_timesteps", self.timesteps),
            ("fpo_noise_seeds", self.noise_seeds),
        ):
            if value.ndim != 2 or int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{name} must have shape [B, Nmc], got {tuple(value.shape)}."
                )
            if metadata_shape is None:
                metadata_shape = tuple(value.shape)
            elif tuple(value.shape) != metadata_shape:
                raise ValueError(
                    "FPO MC metadata shapes differ: "
                    f"expected {metadata_shape}, got {tuple(value.shape)} "
                    f"for {name}."
                )
        expected_action_width = self.num_action_chunks * self.raw_action_dim
        if tuple(self.clean_action_normalized.shape) != (
            batch_size,
            expected_action_width,
        ):
            raise ValueError(
                "FPO normalized clean action must have shape "
                f"[{batch_size}, {expected_action_width}], got "
                f"{tuple(self.clean_action_normalized.shape)}."
            )
        if metadata_shape is None or metadata_shape[1] <= 0:
            raise ValueError("FPO requires at least one MC pair.")

    @staticmethod
    def _restore_type(
        values: torch.Tensor,
        template: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(template, list):
            return list(torch.unbind(values, dim=0))
        return values

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        del shift, seed
        noise_list = (
            noise if isinstance(noise, list) else list(torch.unbind(noise))
        )
        batch_size = len(noise_list)
        num_mc_samples = int(self.sigmas.shape[1])
        if batch_size != int(self.clean_joint_latent.shape[0]):
            raise ValueError(
                "FPO sampler batch does not match saved clean joint latents: "
                f"{batch_size} != {self.clean_joint_latent.shape[0]}."
            )
        if batch_size != 1:
            raise ValueError(
                "Native Cosmos velocity_fn accepts one shared scalar timestep; "
                "RLinf FPO replay therefore evaluates one saved sample per call."
            )
        if int(num_steps) != num_mc_samples:
            raise ValueError(
                "FPO sampler num_steps is the number of MC probes and must "
                f"match saved metadata: {num_steps} != {num_mc_samples}."
            )

        reference = noise_list[0]
        clean = self.clean_joint_latent.to(
            device=reference.device, dtype=reference.dtype
        )
        expected_action = self.clean_action_normalized.to(
            device=reference.device, dtype=reference.dtype
        )
        extracted_action = extract_fpo_normalized_action(
            clean,
            num_action_chunks=self.num_action_chunks,
            raw_action_dim=self.raw_action_dim,
            max_action_dim=self.max_action_dim,
            action_state_rows=self.action_state_rows,
        )
        if not torch.equal(extracted_action, expected_action):
            raise ValueError(
                "FPO normalized action does not exactly match the raw action "
                "suffix of the saved final joint latent."
            )
        joint_noise_mask = build_fpo_joint_noise_mask(
            clean,
            self.condition_image,
            num_action_chunks=self.num_action_chunks,
            raw_action_dim=self.raw_action_dim,
            max_action_dim=self.max_action_dim,
            action_state_rows=self.action_state_rows,
            latent_downsample_factor=self.latent_downsample_factor,
            vision_state_channels=self.vision_state_channels,
            vision_condition_latent_frames=self.vision_condition_latent_frames,
        )

        pair_losses = []
        for mc_index in range(num_mc_samples):
            epsilon = seeded_fpo_epsilon(
                clean,
                self.noise_seeds[:, mc_index].to(device=reference.device),
            )
            sigma = self.sigmas[:, mc_index : mc_index + 1].to(
                device=reference.device, dtype=reference.dtype
            )
            noised_joint = (
                clean + joint_noise_mask * sigma * (epsilon - clean)
            )
            timestep = self.timesteps[0, mc_index].to(
                device=reference.device, dtype=torch.float32
            ).reshape(1, 1)
            predicted = velocity_fn(
                list(torch.unbind(noised_joint, dim=0)), timestep
            )
            predicted_joint = (
                torch.stack(predicted, dim=0)
                if isinstance(predicted, list)
                else predicted
            )
            score_loss_fn = (
                fpo_action_epsilon_mse
                if self.score_parameterization == "epsilon"
                else fpo_action_velocity_mse
            )
            score_kwargs = (
                {"sigma": sigma}
                if self.score_parameterization == "epsilon"
                else {}
            )
            pair_losses.append(
                score_loss_fn(
                    predicted_joint,
                    epsilon,
                    clean,
                    **score_kwargs,
                    num_action_chunks=self.num_action_chunks,
                    raw_action_dim=self.raw_action_dim,
                    max_action_dim=self.max_action_dim,
                    action_state_rows=self.action_state_rows,
                )
            )

        self.pair_losses = torch.stack(pair_losses, dim=1).contiguous()
        scalar_score = -self.pair_losses.mean(dim=1, keepdim=True)
        action_width = self.num_action_chunks * self.raw_action_dim
        # Embodied chunk-level PPO reshapes log-scores by raw action width
        # and sums them. Uniformly distributing the scalar preserves exactly
        # exp(mean(old_loss - new_loss)) after that existing reduction.
        self.logprobs = scalar_score.expand(
            -1, action_width
        ).contiguous() / float(action_width)
        return self._restore_type(clean, noise)


def _native_chain_generators(
    seed: int | list[int] | None, batch_size: int, device: torch.device
) -> list[torch.Generator | None]:
    if seed is None:
        return [None] * batch_size
    seeds = [seed] * batch_size if isinstance(seed, int) else list(seed)
    if len(seeds) != batch_size:
        raise ValueError(f"seed length {len(seeds)} must match batch size {batch_size}.")
    generators = []
    for seed_i in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed_i))
        generators.append(generator)
    return generators


def derive_cosmos_rollout_seed(
    *,
    base_seed: int,
    global_step: int,
    distributed_rank: int,
    rollout_call_index: int,
    sample_index: int,
) -> int:
    """Derive a stable signed-int64 seed for one policy sample."""

    payload = (
        f"{int(base_seed)}:{int(global_step)}:{int(distributed_rank)}:"
        f"{int(rollout_call_index)}:{int(sample_index)}"
    ).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"rlinf-c3").digest()
    return int.from_bytes(digest, byteorder="little", signed=True)


def _cosmos_sampler_seed(seed: int) -> int:
    """Map the persisted int64 seed to Cosmos' NumPy RandomState domain."""

    return int(seed) & 0xFFFFFFFF



ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE = "native_unipc_surrogate"
ROLLOUT_TRANSITION_MATCHED_ACTION_EULER_GAUSSIAN = (
    "matched_action_euler_gaussian"
)


def derive_matched_transition_noise_seed(sampling_seed: int) -> int:
    """Derive an independent, persisted transition-noise seed."""

    digest = hashlib.blake2b(
        str(int(sampling_seed)).encode("ascii"),
        digest_size=8,
        person=b"rlinf-e4",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=True)


def _effective_rollout_transition_mode(
    cosmos_cfg: Any,
    *,
    rollout_mode: str,
    chain_logprob: bool,
) -> str:
    configured = str(
        _cfg_get(
            cosmos_cfg,
            "rollout_transition_mode",
            ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
        )
    ).strip().lower()
    supported = {
        ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
        ROLLOUT_TRANSITION_MATCHED_ACTION_EULER_GAUSSIAN,
    }
    if configured not in supported:
        raise ValueError(
            "Unsupported cosmos.rollout_transition_mode="
            f"{configured!r}; expected one of {sorted(supported)}."
        )
    if rollout_mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported Cosmos rollout mode {rollout_mode!r}.")
    if not chain_logprob or rollout_mode == "eval":
        return ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE
    return configured


def _build_rollout_chain_sampler(
    *,
    base_sampler: Any,
    replay_objective: str,
    transition_mode: str,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int,
    chain_sigma: float,
    sigma_min: float,
    timestep_scale: float,
    transition_noise_seed: int,
) -> (
    CosmosNativeActionTraceSampler
    | CosmosMatchedActionEulerGaussianSampler
    | CosmosNativeFinalLatentSampler
):
    objective = resolve_cosmos_replay_objective(replay_objective)
    if objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD:
        if transition_mode != ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE:
            raise ValueError(
                "FPO requires native UniPC behavior rollouts; "
                f"got rollout_transition_mode={transition_mode!r}."
            )
        return CosmosNativeFinalLatentSampler(base_sampler)
    if transition_mode == ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE:
        return CosmosNativeActionTraceSampler(
            base_sampler,
            chain_sigma=chain_sigma,
            sigma_min=sigma_min,
        )
    if transition_mode == ROLLOUT_TRANSITION_MATCHED_ACTION_EULER_GAUSSIAN:
        return CosmosMatchedActionEulerGaussianSampler(
            num_action_chunks=num_action_chunks,
            raw_action_dim=raw_action_dim,
            max_action_dim=max_action_dim,
            action_state_rows=action_state_rows,
            chain_sigma=chain_sigma,
            sigma_min=sigma_min,
            timestep_scale=timestep_scale,
            transition_noise_seed=transition_noise_seed,
        )
    raise AssertionError(f"Unhandled transition mode {transition_mode!r}.")


def _current_distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def extract_native_action_replay_tensors(
    record: CosmosRecordedActionChain,
    *,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int = 0,
) -> dict[str, torch.Tensor]:
    """Extract normal detached tensors from a flattened native rollout chain.

    Native rollout uses ``no_grad`` because the same FSDP2 model is replayed
    with autograd later. Keep this defensive conversion for records produced
    by older workers (or third-party samplers) under ``inference_mode``:
    inference tensors cannot later be saved by autograd even when they do not
    require gradients. Cloning here preserves sampled values while making the
    recorded constants safe for replay.
    """

    with torch.inference_mode(False), torch.no_grad():
        full_chains = record.full_chains.detach().clone()
        transition_means = record.transition_means.detach().clone()
        timesteps = record.timesteps.detach().clone()
        sigmas = record.sigmas.detach().clone()

    action_rows = int(num_action_chunks) + int(action_state_rows)
    action_flat_width = action_rows * int(max_action_dim)
    if full_chains.shape[-1] < action_flat_width:
        raise ValueError(
            "recorded native chain is narrower than the configured action slice: "
            f"{full_chains.shape[-1]} < {action_flat_width}."
        )
    action_start = full_chains.shape[-1] - action_flat_width
    action_model_chains = full_chains[..., action_start:].reshape(
        full_chains.shape[0],
        full_chains.shape[1],
        action_rows,
        int(max_action_dim),
    )
    action_model_means = transition_means[..., action_start:].reshape(
        transition_means.shape[0],
        transition_means.shape[1],
        action_rows,
        int(max_action_dim),
    )
    action_chains = action_model_chains[
        ..., action_state_rows:, : int(raw_action_dim)
    ].reshape(full_chains.shape[0], full_chains.shape[1], -1)
    action_transition_means = action_model_means[
        ..., action_state_rows:, : int(raw_action_dim)
    ].reshape(transition_means.shape[0], transition_means.shape[1], -1)
    action_mask = torch.ones(
        action_chains.shape[0],
        action_chains.shape[2],
        dtype=action_chains.dtype,
        device=action_chains.device,
    )
    return {
        "action_chains": action_chains.contiguous(),
        "action_transition_means": action_transition_means.contiguous(),
        "action_denoise_timesteps": timesteps.contiguous(),
        "action_sigmas": sigmas.contiguous(),
        "action_mask": action_mask.contiguous(),
        "native_full_chains": full_chains.contiguous(),
    }


def build_fpo_rollout_replay_tensors(
    final_latents: torch.Tensor,
    *,
    sampling_seed: int,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int,
    num_mc_samples: int,
    time_distribution: str,
    training_shift: float,
    timestep_scale: float,
) -> dict[str, torch.Tensor]:
    """Freeze one native UniPC result and deterministic FPO MC probes."""

    clean_joint_latent = final_latents.detach().clone().contiguous()
    clean_action_normalized = extract_fpo_normalized_action(
        clean_joint_latent,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
    )
    metadata = build_fpo_mc_metadata(
        sampling_seed,
        num_mc_samples=num_mc_samples,
        time_distribution=time_distribution,
        training_shift=training_shift,
        timestep_scale=timestep_scale,
        device=clean_joint_latent.device,
    )
    if int(clean_joint_latent.shape[0]) != 1:
        raise ValueError(
            "FPO rollout capture currently requires one native policy sample "
            f"per call, got batch={clean_joint_latent.shape[0]}."
        )
    return {
        "fpo_clean_joint_latent": clean_joint_latent,
        "fpo_clean_action_normalized": clean_action_normalized,
        **metadata,
    }


def _finalize_cosmos_rollout_replay(
    recorder: Any,
    *,
    replay_objective: str,
    sampling_seed: int,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int,
    sigma_min: float,
    num_mc_samples: int,
    time_distribution: str,
    training_shift: float,
    timestep_scale: float,
    logprob_mode: str = "joint_sum",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Convert an objective-specific rollout recorder into flat replay data."""

    objective = resolve_cosmos_replay_objective(replay_objective)
    if objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD:
        final_latents = getattr(recorder, "final_latents", None)
        if not torch.is_tensor(final_latents):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 FPO capture did not record a final joint latent."
            )
        replay_tensors = build_fpo_rollout_replay_tensors(
            final_latents,
            sampling_seed=sampling_seed,
            num_action_chunks=num_action_chunks,
            raw_action_dim=raw_action_dim,
            max_action_dim=max_action_dim,
            action_state_rows=action_state_rows,
            num_mc_samples=num_mc_samples,
            time_distribution=time_distribution,
            training_shift=training_shift,
            timestep_scale=timestep_scale,
        )
        prev_logprobs = torch.zeros(
            (
                final_latents.shape[0],
                int(num_action_chunks) * int(raw_action_dim),
            ),
            device=final_latents.device,
            dtype=torch.float32,
        )
    else:
        record = getattr(recorder, "record", None)
        if record is None:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 action-chain capture did not record a sampler chain."
            )
        replay_tensors = extract_native_action_replay_tensors(
            record,
            num_action_chunks=num_action_chunks,
            raw_action_dim=raw_action_dim,
            max_action_dim=max_action_dim,
            action_state_rows=action_state_rows,
        )
        prev_logprobs = compute_native_recorded_action_logprobs(
            replay_tensors,
            sigma_min=sigma_min,
            logprob_mode=logprob_mode,
        )
    replay_tensors = {
        key: value.detach().clone().contiguous()
        for key, value in replay_tensors.items()
    }
    return replay_tensors, prev_logprobs.detach().clone().contiguous()


def _materialize_normal_replay_tensors(value: Any) -> Any:
    """Clone inference tensors into ordinary tensors without detaching graphs.

    Cosmos rollout creates immutable inference tensors. Actor replay can feed
    those constants into modules with trainable parameters; autograd then
    rejects saving them for backward. Only inference tensors are cloned.
    Ordinary tensors, including differentiable denoising states, stay intact.
    """

    if torch.is_tensor(value):
        if not torch.is_inference(value):
            return value
        with torch.inference_mode(False), torch.no_grad():
            return value.detach().clone()
    if isinstance(value, list):
        converted = [_materialize_normal_replay_tensors(item) for item in value]
        return (
            value
            if all(new is old for new, old in zip(converted, value))
            else converted
        )
    if isinstance(value, tuple):
        converted = tuple(
            _materialize_normal_replay_tensors(item) for item in value
        )
        if all(new is old for new, old in zip(converted, value)):
            return value
        return type(value)(*converted) if hasattr(value, "_fields") else converted
    if isinstance(value, dict):
        converted = {
            key: _materialize_normal_replay_tensors(item)
            for key, item in value.items()
        }
        if all(converted[key] is item for key, item in value.items()):
            return value
        return converted
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _materialize_normal_replay_tensors(
                getattr(value, field.name)
            )
            for field in fields(value)
        }
        if all(updates[name] is getattr(value, name) for name in updates):
            return value
        return replace(value, **updates)
    return value


def _resize_native_actor_replay_chains(
    full_chains: torch.Tensor,
    *,
    source_image_size: tuple[int, int],
    target_image_size: tuple[int, int],
    num_action_chunks: int,
    max_action_dim: int,
    action_state_rows: int = 0,
    latent_downsample_factor: int = 16,
) -> torch.Tensor:
    """Resize only the vision prefix of a saved native replay chain.

    Cosmos flattens each sampler state as [vision | action]. Actor replay may
    use a smaller image than rollout to reduce activation memory, but its saved
    vision states must then use the corresponding latent grid. Treat the
    combined channel/time axes as independent planes, resize only their spatial
    dimensions, and retain the sampled action suffix exactly.
    """

    if full_chains.ndim != 3:
        raise ValueError(
            "native full chains must have shape [batch, steps, width], got "
            f"{tuple(full_chains.shape)}."
        )
    source_h, source_w = (int(value) for value in source_image_size)
    target_h, target_w = (int(value) for value in target_image_size)
    factor = int(latent_downsample_factor)
    if factor <= 0:
        raise ValueError(f"latent_downsample_factor must be positive, got {factor}.")
    for label, height, width in (
        ("source", source_h, source_w),
        ("target", target_h, target_w),
    ):
        if height <= 0 or width <= 0 or height % factor or width % factor:
            raise ValueError(
                f"{label} actor replay image size {(height, width)} must contain "
                f"positive dimensions divisible by {factor}."
            )

    action_flat_width = (
        int(num_action_chunks) + int(action_state_rows)
    ) * int(max_action_dim)
    vision_flat_width = int(full_chains.shape[-1]) - action_flat_width
    source_latent_h = source_h // factor
    source_latent_w = source_w // factor
    source_spatial = source_latent_h * source_latent_w
    if vision_flat_width <= 0 or vision_flat_width % source_spatial:
        raise ValueError(
            "native replay vision width is incompatible with its source latent grid: "
            f"vision_width={vision_flat_width}, grid="
            f"{source_latent_h}x{source_latent_w}."
        )

    target_latent_h = target_h // factor
    target_latent_w = target_w // factor
    vision = full_chains[..., :vision_flat_width]
    action = full_chains[..., vision_flat_width:]
    planes = vision_flat_width // source_spatial
    vision_planes = vision.reshape(
        full_chains.shape[0] * full_chains.shape[1] * planes,
        1,
        source_latent_h,
        source_latent_w,
    )
    resized = F.interpolate(
        vision_planes.float(),
        size=(target_latent_h, target_latent_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).to(dtype=full_chains.dtype)
    resized = resized.reshape(full_chains.shape[0], full_chains.shape[1], -1)
    return torch.cat([resized, action], dim=-1).contiguous()


def _resize_native_actor_condition_image(
    image: torch.Tensor, target_image_size: tuple[int, int]
) -> torch.Tensor:
    """Resize a batched CHW or HWC replay condition without changing layout."""

    if image.ndim != 4:
        raise ValueError(
            "actor replay condition image must be batched CHW or HWC, got "
            f"{tuple(image.shape)}."
        )
    target_h, target_w = (int(value) for value in target_image_size)
    input_is_hwc = image.shape[-1] in (1, 3, 4)
    if input_is_hwc:
        image_chw = image.permute(0, 3, 1, 2)
    elif image.shape[1] in (1, 3, 4):
        image_chw = image
    else:
        raise ValueError(
            "actor replay condition image has no recognizable channel axis: "
            f"{tuple(image.shape)}."
        )
    resized = F.interpolate(
        image_chw.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if not image.dtype.is_floating_point:
        resized = resized.round().clamp(0, 255)
    resized = resized.to(dtype=image.dtype)
    if input_is_hwc:
        resized = resized.permute(0, 2, 3, 1)
    return resized.contiguous()


NATIVE_CHAIN_LOGPROB_MODES = ("joint_sum", "joint_mean", "single_step")
NATIVE_CHAIN_LOGPROB_ALIASES = {
    "flow-noise": "joint_mean",
    "flow_noise": "joint_mean",
    "flow-sde": "single_step",
    "flow_sde": "single_step",
}


def _normalize_native_chain_logprob_mode(mode: str | None) -> str:
    normalized = str(mode or "joint_sum").strip().lower()
    normalized = NATIVE_CHAIN_LOGPROB_ALIASES.get(normalized, normalized)
    if normalized not in NATIVE_CHAIN_LOGPROB_MODES:
        supported = ", ".join(NATIVE_CHAIN_LOGPROB_MODES)
        raise ValueError(
            f"Unsupported Cosmos native chain_logprob_mode={mode!r}; "
            f"expected one of: {supported}."
        )
    return normalized


def _validate_native_denoise_indices(
    indices: torch.Tensor,
    *,
    batch_size: int,
    num_steps: int,
) -> torch.Tensor:
    if indices.shape != (batch_size, 1):
        raise ValueError(
            "action_denoise_indices must have shape [B, 1], got "
            f"{tuple(indices.shape)} for B={batch_size}."
        )
    indices = indices.to(dtype=torch.long)
    if bool(((indices < 0) | (indices >= num_steps)).any().item()):
        raise ValueError(
            "action_denoise_indices must be within [0, num_steps), got "
            f"min={int(indices.min().item())}, max={int(indices.max().item())}, "
            f"num_steps={num_steps}."
        )
    return indices


def _reduce_native_chain_logprobs(
    transition_logprobs: torch.Tensor,
    initial_logprobs: torch.Tensor,
    *,
    mode: str,
    action_denoise_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = _normalize_native_chain_logprob_mode(mode)
    if mode == "joint_sum":
        return transition_logprobs.sum(dim=1) + initial_logprobs
    if mode == "joint_mean":
        # Match RLinf OpenPI joint_logprob=True: average the initial-noise
        # logprob and all denoising-transition logprobs before PPO aggregation.
        return torch.cat(
            [initial_logprobs[:, None, :], transition_logprobs], dim=1
        ).mean(dim=1)

    if action_denoise_indices is None:
        raise ValueError("single_step logprob mode requires action_denoise_indices.")
    indices = _validate_native_denoise_indices(
        action_denoise_indices,
        batch_size=int(transition_logprobs.shape[0]),
        num_steps=int(transition_logprobs.shape[1]),
    ).to(device=transition_logprobs.device)
    gather_indices = indices[:, :, None].expand(-1, -1, transition_logprobs.shape[-1])
    return transition_logprobs.gather(dim=1, index=gather_indices).squeeze(1)


def compute_native_recorded_action_logprobs(
    action_replay_tensors: dict[str, torch.Tensor],
    *,
    sigma_min: float = 1e-4,
    logprob_mode: str = "joint_sum",
) -> torch.Tensor:
    """Compute rollout-time logprobs from recorded native transition means."""

    logprob_mode = _normalize_native_chain_logprob_mode(logprob_mode)

    transition_logprobs = gaussian_logprob(
        action_replay_tensors["action_chains"][:, 1:, :],
        action_replay_tensors["action_transition_means"],
        action_replay_tensors["action_sigmas"][:, :, None],
        mask=action_replay_tensors["action_mask"][:, None, :],
        sigma_min=sigma_min,
    )
    initial_logprobs = gaussian_logprob(
        action_replay_tensors["action_chains"][:, 0, :],
        torch.zeros_like(action_replay_tensors["action_chains"][:, 0, :]),
        torch.ones_like(action_replay_tensors["action_chains"][:, 0, :]),
        mask=action_replay_tensors["action_mask"],
        sigma_min=sigma_min,
    )
    if (
        logprob_mode == "single_step"
        and "action_denoise_indices" not in action_replay_tensors
    ):
        # RLinf's non-joint mode samples one shared denoise index for a rollout
        # batch, then stores it so old/new policy replay uses the exact same
        # transition. Most Cosmos rollout requests have B=1, but sharing the
        # index also preserves the current replay sampler's shared timestep.
        batch_size, num_steps = transition_logprobs.shape[:2]
        selected = torch.randint(
            num_steps,
            (1, 1),
            device=transition_logprobs.device,
        ).expand(batch_size, -1).clone()
        action_replay_tensors["action_denoise_indices"] = selected.contiguous()
    return _reduce_native_chain_logprobs(
        transition_logprobs,
        initial_logprobs,
        mode=logprob_mode,
        action_denoise_indices=action_replay_tensors.get("action_denoise_indices"),
    )


class CosmosNativeActionReplaySampler:
    """Sampler shim that replays saved native chains under current weights."""

    def __init__(
        self,
        *,
        full_chains: torch.Tensor,
        action_chains: torch.Tensor,
        action_denoise_timesteps: torch.Tensor,
        action_sigmas: torch.Tensor,
        action_mask: torch.Tensor,
        num_action_chunks: int,
        raw_action_dim: int,
        max_action_dim: int,
        action_state_rows: int = 0,
        sigma_min: float = 1e-4,
        action_denoise_indices: torch.Tensor | None = None,
        logprob_mode: str = "joint_sum",
        checkpoint_velocity_calls: bool = False,
        checkpoint_velocity_context_factory: Any = None,
    ) -> None:
        self.full_chains = full_chains
        self.action_chains = action_chains
        self.action_denoise_timesteps = action_denoise_timesteps
        self.action_sigmas = action_sigmas
        self.action_mask = action_mask
        self.action_denoise_indices = action_denoise_indices
        self.logprob_mode = _normalize_native_chain_logprob_mode(logprob_mode)
        self.checkpoint_velocity_calls = bool(checkpoint_velocity_calls)
        self.checkpoint_velocity_context_factory = (
            checkpoint_velocity_context_factory
        )
        self.num_action_chunks = int(num_action_chunks)
        self.raw_action_dim = int(raw_action_dim)
        self.max_action_dim = int(max_action_dim)
        self.action_state_rows = int(action_state_rows)
        self.sigma_min = float(sigma_min)
        self.logprobs: torch.Tensor | None = None
        self.action_transition_means: torch.Tensor | None = None
        self.transition_logprobs: torch.Tensor | None = None
        self.initial_logprobs: torch.Tensor | None = None

    def __call__(
        self,
        velocity_fn,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        del shift, seed
        input_was_list = isinstance(noise, list)
        noise_list = noise if input_was_list else [noise]
        batch_size = len(noise_list)
        expected_steps = int(self.full_chains.shape[1]) - 1
        if int(num_steps) != expected_steps:
            raise ValueError(
                "native replay num_steps must match saved chain transitions: "
                f"{num_steps} != {expected_steps}."
            )
        if batch_size != int(self.full_chains.shape[0]):
            raise ValueError(
                "native replay batch size must match saved full chains: "
                f"{batch_size} != {self.full_chains.shape[0]}."
            )

        action_flat_width = (self.num_action_chunks + self.action_state_rows) * self.max_action_dim
        action_start = int(self.full_chains.shape[-1]) - action_flat_width
        if action_start < 0:
            raise ValueError(
                "native replay full chain is narrower than the configured action slice: "
                f"{self.full_chains.shape[-1]} < {action_flat_width}."
            )

        if self.logprob_mode == "single_step":
            if self.action_denoise_indices is None:
                raise ValueError(
                    "single_step native replay requires saved action_denoise_indices."
                )
            selected_indices = _validate_native_denoise_indices(
                self.action_denoise_indices,
                batch_size=batch_size,
                num_steps=expected_steps,
            )
            if not bool((selected_indices == selected_indices[0, 0]).all().item()):
                raise ValueError(
                    "native replay currently requires one shared denoise index "
                    "per batch."
                )
            replay_step_indices = [int(selected_indices[0, 0].item())]
        else:
            replay_step_indices = list(range(expected_steps))

        transition_means = []
        for step_idx in replay_step_indices:
            current = [
                self.full_chains[sample_idx, step_idx].to(
                    device=noise_list[sample_idx].device,
                    dtype=noise_list[sample_idx].dtype,
                )
                for sample_idx in range(batch_size)
            ]
            timestep = self.action_denoise_timesteps[0, step_idx].reshape(1, 1).to(
                device=current[0].device, dtype=current[0].dtype
            )
            if self.checkpoint_velocity_calls and torch.is_grad_enabled():
                # Each saved replay state is independent: the current policy is
                # evaluated at all denoise states only to form their transition
                # logprobs. Checkpoint the whole velocity call so activations
                # from all denoise steps do not coexist until backward. The
                # The reentrant variant avoids nesting two non-reentrant
                # saved-tensor-hook stacks around Cosmos FSDP2. Give the
                # timestep a harmless grad edge because reentrant checkpoint
                # requires at least one grad-requiring input.
                def checkpointed_velocity(*tensor_args):
                    context = (
                        self.checkpoint_velocity_context_factory()
                        if self.checkpoint_velocity_context_factory is not None
                        else nullcontext()
                    )
                    with context:
                        outputs = velocity_fn(
                            list(tensor_args[:-1]), tensor_args[-1]
                        )
                    return tuple(outputs)

                velocities = list(
                    activation_checkpoint(
                        checkpointed_velocity,
                        *current,
                        timestep.detach().requires_grad_(True),
                        use_reentrant=True,
                        preserve_rng_state=True,
                    )
                )
            else:
                velocities = velocity_fn(current, timestep)
            means = []
            for sample_idx, (state, velocity) in enumerate(
                zip(current, velocities, strict=True)
            ):
                step_size = 1.0 / float(expected_steps)
                full_mean = state + step_size * velocity.to(
                    device=state.device, dtype=state.dtype
                )
                action_mean = full_mean[action_start:].reshape(
                    self.num_action_chunks + self.action_state_rows, self.max_action_dim
                )[self.action_state_rows :, : self.raw_action_dim]
                means.append(action_mean.reshape(-1))
            transition_means.append(torch.stack(means, dim=0))

        action_transition_means = torch.stack(transition_means, dim=1)
        action_targets = self.action_chains[
            :, [step_idx + 1 for step_idx in replay_step_indices], :
        ]
        action_sigmas = self.action_sigmas[:, replay_step_indices]
        transition_logprobs = gaussian_logprob(
            action_targets.to(action_transition_means.device),
            action_transition_means,
            action_sigmas[:, :, None].to(action_transition_means.device),
            mask=self.action_mask[:, None, :].to(action_transition_means.device),
            sigma_min=self.sigma_min,
        )
        initial_logprobs = gaussian_logprob(
            self.action_chains[:, 0, :].to(action_transition_means.device),
            torch.zeros_like(self.action_chains[:, 0, :]).to(action_transition_means.device),
            torch.ones_like(self.action_chains[:, 0, :]).to(action_transition_means.device),
            mask=self.action_mask.to(action_transition_means.device),
            sigma_min=self.sigma_min,
        )
        # Preserve the native replay diagnostics exposed by
        # _default_forward_single. These tensors are also useful for parity
        # checks, but they must not be required only by the new FPO path.
        self.action_transition_means = action_transition_means
        self.transition_logprobs = transition_logprobs
        self.initial_logprobs = initial_logprobs
        if self.logprob_mode == "single_step":
            reduction_indices = torch.zeros(
                batch_size,
                1,
                dtype=torch.long,
                device=transition_logprobs.device,
            )
        else:
            reduction_indices = None
        self.logprobs = _reduce_native_chain_logprobs(
            transition_logprobs,
            initial_logprobs,
            mode=self.logprob_mode,
            action_denoise_indices=reduction_indices,
        ).contiguous()

        final = [
            self.full_chains[sample_idx, -1].to(
                device=noise_list[sample_idx].device, dtype=noise_list[sample_idx].dtype
            )
            for sample_idx in range(batch_size)
        ]
        return final if input_was_list else final[0]


@dataclass(frozen=True)
class CosmosNativeBackendProbe:
    """Read-only readiness report for the native Cosmos3 backend."""

    in_slurm_allocation: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_error: str | None
    missing_imports: tuple[str, ...]
    checkpoint_path: str | None
    checkpoint_path_exists: bool
    checkpoint_name: str | None
    hf_token_available: bool

    @property
    def framework_available(self) -> bool:
        return not self.missing_imports

    @property
    def checkpoint_available(self) -> bool:
        if self.checkpoint_path:
            return self.checkpoint_path_exists
        return bool(self.checkpoint_name and self.hf_token_available)

    @property
    def ready(self) -> bool:
        return (
            self.in_slurm_allocation
            and self.cuda_available
            and self.framework_available
            and self.checkpoint_available
        )

    def blocker_reason(self) -> str | None:
        blockers = []
        if not self.in_slurm_allocation:
            blockers.append("not running inside a Slurm GPU allocation")
        if not self.cuda_available:
            detail = f": {self.cuda_error}" if self.cuda_error else ""
            blockers.append(f"torch CUDA is unavailable{detail}")
        if self.missing_imports:
            blockers.append(
                "missing cosmos_framework imports: "
                + ", ".join(self.missing_imports)
            )
        if not self.checkpoint_available:
            blockers.append(
                "missing local checkpoint path or HF token-backed checkpoint name"
            )
        if not blockers:
            return None
        return "; ".join(blockers)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg: Any, key: str, default: bool = False) -> bool:
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _expand_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path_text = str(path).strip()
    if not path_text or path_text.lower() in {"none", "null"}:
        return None
    return Path(os.path.expandvars(path_text)).expanduser()


def create_cosmos_qwen_tokenizer_processor_compat(
    pretrained_model_name: str, config_variant: str | None = None
):
    """Load a local Qwen bundle across Cosmos/Transformers API versions."""

    del config_variant
    tokenizer_dir = _expand_path(pretrained_model_name)
    if tokenizer_dir is None:
        raise ValueError("A local Qwen tokenizer directory is required.")
    vocab_file = tokenizer_dir / "vocab.json"
    merges_file = tokenizer_dir / "merges.txt"
    for required in (vocab_file, merges_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    reasoner = importlib.import_module(
        "cosmos_framework.configs.base.defaults.reasoner"
    )
    tokenizer = reasoner.Qwen2Tokenizer.from_pretrained(
        str(tokenizer_dir),
        vocab_file=str(vocab_file),
        merges_file=str(merges_file),
    )
    return reasoner.LLMTokenizerProcessor(tokenizer)


_COSMOS_CONFIG_TARGET_REWRITES = (
    ("cosmos_framework.model.vfm.", "cosmos_framework.model.generator."),
    (
        "cosmos_framework.configs.base.defaults.vlm.",
        "cosmos_framework.configs.base.defaults.reasoner.",
    ),
    (
        "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/",
        "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/",
    ),
)


def _upgrade_legacy_generator_config_schema(source: str) -> str:
    """Add current inference-only defaults absent from legacy VFM YAML."""
    rewritten = source
    if "\n    enable_input_bias:" not in rewritten:
        rewritten = rewritten.replace(
            "\n    exclude_reasoner_weights_from_checkpoint:",
            "\n    enable_input_bias: true"
            "\n    exclude_reasoner_weights_from_checkpoint:",
            1,
        )
    if "\n      attention_io_layout:" not in rewritten:
        rewritten = rewritten.replace(
            "\n    parallelism:\n",
            "\n    parallelism:\n      attention_io_layout: sequence_sharded\n",
            1,
        )
    required = (
        "\n    enable_input_bias: true",
        "\n      attention_io_layout: sequence_sharded",
    )
    missing = [entry.strip() for entry in required if entry not in rewritten]
    if missing:
        raise CosmosNativeBackendUnavailable(
            "Legacy Cosmos config could not be upgraded to the current model schema; "
            f"missing {missing}."
        )
    return rewritten


def _native_checkpoint_config_compat_path(
    config_path: Path,
    output_dir: Path,
    framework_path: str | Path | None = None,
) -> Path:
    """Rewrite legacy VFM model targets for current Cosmos inference only."""

    source = config_path.read_text()
    legacy_config = any(
        legacy_prefix in source
        for legacy_prefix, _ in _COSMOS_CONFIG_TARGET_REWRITES
    )
    try:
        legacy_runtime_available = (
            importlib.util.find_spec(
                "cosmos_framework.model.vfm.omni_mot_model"
            )
            is not None
        )
    except ModuleNotFoundError:
        legacy_runtime_available = False
    if legacy_runtime_available:
        return config_path

    processor_path = _expand_path(os.environ.get("COSMOS3_VLM_PROCESSOR_PATH"))
    # Current-format B configs still use the same download-capable reasoner
    # tokenizer factory. Rewrite those too when a local processor is supplied.
    if not legacy_config and processor_path is None:
        return config_path

    rewritten = source
    if legacy_config:
        rewritten = _upgrade_legacy_generator_config_schema(rewritten)
    for legacy_prefix, current_prefix in _COSMOS_CONFIG_TARGET_REWRITES:
        rewritten = rewritten.replace(legacy_prefix, current_prefix)

    selected_framework_path = (
        _expand_path(str(framework_path))
        if framework_path
        else _expand_path(os.environ.get("COSMOS3_FRAMEWORK_PATH"))
    )
    if selected_framework_path is not None:
        framework_json_prefix = selected_framework_path / "cosmos_framework"
        rewritten = rewritten.replace(
            "json_file: cosmos_framework/",
            f"json_file: {framework_json_prefix}/",
        )

    if processor_path is not None:
        required_tokenizer_files = (
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        )
        missing = [
            name for name in required_tokenizer_files
            if not (processor_path / name).is_file()
        ]
        if missing:
            raise CosmosNativeBackendUnavailable(
                "COSMOS3_VLM_PROCESSOR_PATH is missing tokenizer files: "
                + ", ".join(missing)
            )
        rewritten = rewritten.replace(
            "pretrained_model_name: Qwen/Qwen3-VL-8B-Instruct",
            f"pretrained_model_name: {processor_path}",
        )
        rewritten = rewritten.replace(
            "cosmos_framework.configs.base.defaults.reasoner."
            "create_qwen2_tokenizer_with_download",
            "rlinf.models.embodiment.cosmos.cosmos_backend."
            "create_cosmos_qwen_tokenizer_processor_compat",
        )
    if rewritten == source:
        return config_path

    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    compat_dir = output_dir / "compat_configs"
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_path = compat_dir / f"{config_path.stem}_generator_rank_{rank}.yaml"
    compat_path.write_text(rewritten)
    return compat_path


def _maybe_add_framework_path(framework_path: str | None) -> None:
    if not framework_path:
        return
    expanded_path = _expand_path(framework_path)
    if expanded_path is None or not expanded_path.exists():
        return
    expanded_path = expanded_path.resolve()

    # Python cannot safely switch an imported top-level package to another
    # checkout. A and B may use distinct frameworks only because RLinf runs
    # them in separate Ray worker processes; fail immediately if a smoke test
    # or future refactor accidentally co-loads both versions.
    loaded = sys.modules.get("cosmos_framework")
    if loaded is not None:
        locations = []
        module_file = getattr(loaded, "__file__", None)
        if module_file:
            locations.append(Path(module_file).resolve())
        locations.extend(
            Path(location).resolve()
            for location in (getattr(loaded, "__path__", None) or ())
        )
        package_root = expanded_path / "cosmos_framework"
        if locations and not any(
            location == package_root or package_root in location.parents
            for location in locations
        ):
            raise CosmosNativeBackendUnavailable(
                "cosmos_framework is already imported from a different checkout: "
                f"loaded={locations}, requested={expanded_path}. Run A and B in "
                "separate processes."
            )

    path_text = str(expanded_path)
    sys.path[:] = [entry for entry in sys.path if entry != path_text]
    sys.path.insert(0, path_text)


def _import_cosmos_module(*module_names: str):
    """Import the first module path available across Cosmos3 releases."""

    # Respect modules already loaded by the host process (and unit-test
    # dependency injection) before probing another release layout.
    for module_name in reversed(module_names):
        if module_name in sys.modules:
            return sys.modules[module_name]

    first_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Fall back only when this candidate path itself is absent. A
            # dependency missing from inside an existing module is a real
            # environment error and must not be hidden by trying another API.
            if not (
                exc.name == module_name
                or (exc.name is not None and module_name.startswith(f"{exc.name}."))
            ):
                raise
            if first_error is None:
                first_error = exc
    assert first_error is not None
    raise first_error


def _missing_imports() -> tuple[str, ...]:
    missing = []
    for module_name in COSMOS_FRAMEWORK_IMPORTS:
        candidates = (module_name,)
        alternative = COSMOS_FRAMEWORK_IMPORT_ALTERNATIVES.get(module_name)
        if alternative is not None:
            candidates += (alternative,)
        available = False
        for candidate in candidates:
            try:
                available = importlib.util.find_spec(candidate) is not None
            except (ModuleNotFoundError, ValueError):
                available = False
            if available:
                break
        if not available:
            missing.append(" | ".join(candidates))
    return tuple(missing)


def _cuda_state() -> tuple[bool, int, str | None]:
    try:
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    except RuntimeError as exc:
        return False, 0, str(exc)
    return bool(cuda_available), int(cuda_device_count), None


def _clear_cosmos_lazy_config_resolvers() -> None:
    from omegaconf import OmegaConf

    for resolver_name in ("add", "subtract"):
        if OmegaConf.has_resolver(resolver_name):
            OmegaConf.clear_resolver(resolver_name)


def _patch_typing_override_for_py311() -> None:
    import typing

    if hasattr(typing, "override"):
        return
    from typing_extensions import override

    typing.override = override


def _patch_cosmos_ur5_joint_domain() -> None:
    """Register the domain contract used by the UR5 joint SFT checkpoint.

    The checkpoint was trained with the companion cosmos-framework patch that
    aliases ``robomind-ur-joint`` to UR domain id 13 while changing only its
    raw action width from the legacy EEF model's 10 channels to 7 joint
    channels.  The pinned inference framework predates that small registry
    addition, so reproduce it at the integration boundary instead of changing
    the legacy ``robomind-ur`` entry or silently routing 7D data through a 10D
    contract.
    """

    domain_utils = _import_cosmos_module(
        "cosmos_framework.data.generator.action.domain_utils",
        "cosmos_framework.data.vfm.action.domain_utils",
    )
    domain_ids = domain_utils.EMBODIMENT_TO_DOMAIN_ID
    raw_dims = domain_utils.EMBODIMENT_TO_RAW_ACTION_DIM
    domain_name = "robomind-ur-joint"

    legacy_domain_id = domain_ids.get("robomind-ur")
    if legacy_domain_id != 13:
        raise CosmosNativeBackendUnavailable(
            "The UR5 joint checkpoint requires Cosmos domain id 13, but the "
            f"pinned framework reports robomind-ur={legacy_domain_id!r}."
        )

    existing_domain_id = domain_ids.get(domain_name)
    if existing_domain_id not in (None, 13):
        raise CosmosNativeBackendUnavailable(
            f"Conflicting {domain_name!r} domain id: {existing_domain_id!r}."
        )
    existing_raw_dim = raw_dims.get(domain_name)
    if existing_raw_dim not in (None, 7):
        raise CosmosNativeBackendUnavailable(
            f"Conflicting {domain_name!r} raw action dim: {existing_raw_dim!r}."
        )

    domain_ids[domain_name] = 13
    raw_dims[domain_name] = 7


def _patch_cosmos_checkpoint_tokenizer_factory() -> None:
    """Allow local checkpoint processors, including processor-less SFT exports."""

    legacy_vlm_module = "cosmos_framework.configs.base.defaults.vlm"
    try:
        vlm_defaults = importlib.import_module(legacy_vlm_module)
    except ModuleNotFoundError as exc:
        if not (
            exc.name == legacy_vlm_module
            or (exc.name is not None and legacy_vlm_module.startswith(f"{exc.name}."))
        ):
            raise
        # Cosmos 1.2.2 moved this factory to defaults.reasoner and its native
        # checkpoint loader already handles the current tokenizer contract.
        return
    processors = importlib.import_module("cosmos_framework.data.vfm.processors")
    original = vlm_defaults.create_qwen2_tokenizer_with_download
    if getattr(original, "_rlinf_accepts_tokenizer_type", False):
        return

    def create_qwen2_tokenizer_with_download(
        pretrained_model_name: str | None = None,
        config_variant: str | None = None,
        tokenizer_type: str | None = None,
        **kwargs,
    ):
        fallback = os.environ.get("COSMOS3_VLM_PROCESSOR_PATH")
        if tokenizer_type is not None:
            tokenizer_path = Path(tokenizer_type).expanduser()
            # A normal Cosmos HF export bundles the Qwen processor beside the
            # model shards. The 7D SFT exporter only wrote model shards, while
            # retaining the same Qwen3-VL-8B architecture. In that narrow
            # case, use the separately configured, compatible local processor
            # bundle rather than treating the incomplete export directory as a
            # Transformers repository or downloading anything at runtime.
            if tokenizer_path.is_dir() and not (
                tokenizer_path / "preprocessor_config.json"
            ).is_file():
                if not fallback:
                    raise CosmosNativeBackendUnavailable(
                        "Cosmos checkpoint has no bundled VLM processor. Set "
                        "COSMOS3_VLM_PROCESSOR_PATH to a compatible local "
                        "Qwen3-VL processor bundle."
                    )
                tokenizer_path = Path(fallback).expanduser()
                if not (tokenizer_path / "preprocessor_config.json").is_file():
                    raise CosmosNativeBackendUnavailable(
                        "COSMOS3_VLM_PROCESSOR_PATH does not contain "
                        f"preprocessor_config.json: {tokenizer_path}"
                    )
            return processors.build_processor(
                str(tokenizer_path) if tokenizer_path.is_dir() else tokenizer_type,
                config_variant=config_variant,
                **kwargs,
            )
        # Native DCP checkpoints load their training YAML directly, whose VLM
        # tokenizer node names the upstream Qwen repository. Prefer the same
        # explicitly configured local processor bundle used by HF exports so
        # compute-node startup stays offline and deterministic.
        if fallback:
            fallback_path = Path(fallback).expanduser()
            if not (fallback_path / "preprocessor_config.json").is_file():
                raise CosmosNativeBackendUnavailable(
                    "COSMOS3_VLM_PROCESSOR_PATH does not contain "
                    f"preprocessor_config.json: {fallback_path}"
                )
            return processors.build_processor(
                str(fallback_path),
                config_variant=config_variant,
                **kwargs,
            )
        if pretrained_model_name is None or config_variant is None:
            raise TypeError(
                "pretrained_model_name and config_variant are required when "
                "tokenizer_type is not provided"
            )
        return original(pretrained_model_name, config_variant)

    create_qwen2_tokenizer_with_download._rlinf_accepts_tokenizer_type = True
    vlm_defaults.create_qwen2_tokenizer_with_download = (
        create_qwen2_tokenizer_with_download
    )


def _patch_cosmos_single_rank_sync_model_states() -> None:
    """Skip redundant Cosmos model-state sync for native single-rank smoke."""

    distributed_utils = importlib.import_module("cosmos_framework.utils.distributed")
    original = distributed_utils.sync_model_states
    if getattr(original, "_rlinf_single_rank_noop", False):
        return

    def sync_model_states(model, *args, **kwargs):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        if dist.get_world_size() <= 1:
            return None
        return original(model, *args, **kwargs)

    sync_model_states._rlinf_single_rank_noop = True
    distributed_utils.sync_model_states = sync_model_states
    for module_name in (
        "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16",
        "cosmos_framework.model.generator.tokenizers.wan2pt1_vae_4x8x8",
        "cosmos_framework.model.generator.tokenizers.dc_ae.dc_ae_4x32x32",
        "cosmos_framework.model.generator.tokenizers.uniae.noncausal_4x16x16",
        "cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16",
        "cosmos_framework.model.vfm.tokenizers.wan2pt1_vae_4x8x8",
        "cosmos_framework.model.vfm.tokenizers.dc_ae.dc_ae_4x32x32",
        "cosmos_framework.model.vfm.tokenizers.uniae.noncausal_4x16x16",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "sync_model_states"):
            module.sync_model_states = sync_model_states


def _patch_python313_pathlib_pickle_compat() -> None:
    """Let Python <=3.12 unpickle DCP metadata written by Python 3.13."""
    import pathlib
    import types

    module_name = "pathlib._local"
    if module_name in sys.modules:
        return
    compat = types.ModuleType(module_name)
    for name in (
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ):
        setattr(compat, name, getattr(pathlib, name))
    sys.modules[module_name] = compat
    setattr(pathlib, "_local", compat)


def _patch_torch_dcp_single_rank_load() -> None:
    """Avoid NCCL collectives for single-rank native inference checkpoint load."""

    _patch_python313_pathlib_pickle_compat()
    dcp = importlib.import_module("torch.distributed.checkpoint")
    original = dcp.load
    if getattr(original, "_rlinf_single_rank_no_dist", False):
        return

    def load(*args, **kwargs):
        import torch.distributed as dist

        if (
            "no_dist" not in kwargs
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() <= 1
        ):
            kwargs["no_dist"] = True
        return original(*args, **kwargs)

    load._rlinf_single_rank_no_dist = True
    dcp.load = load


def _patch_action_server_guardrail_args(libero_server: Any) -> None:
    args_cls = libero_server.ActionServerArgs
    if getattr(args_cls, "_rlinf_guardrails_patch", False):
        return

    class RlinfActionServerArgs(args_cls):
        guardrails: bool | None = None
        offload_guardrail_models: bool | None = None
        internal_fsdp_shard: bool = False
        internal_fsdp_node_local: bool = False
        use_torch_compile: bool | None = None

        def build_setup_overrides(self):
            base = super().build_setup_overrides()
            if self.guardrails is not None:
                base.guardrails = bool(self.guardrails)
            if self.offload_guardrail_models is not None:
                base.offload_guardrail_models = bool(self.offload_guardrail_models)
            if self.use_torch_compile is not None:
                base.use_torch_compile = bool(self.use_torch_compile)
            # RLinf assigns a different rollout sample to every distributed
            # rank. Cosmos context parallelism and CFG parallelism instead
            # assume participating ranks are cooperating on the same sample.
            # If those dimensions are auto-selected from WORLD_SIZE, rollout
            # and replay can exchange data between unrelated trajectories; a
            # replay of the unchanged policy then incorrectly produces ratios
            # far from one. Keep these dimensions local for both execution
            # modes. internal_fsdp_shard may still use FSDP data parallel
            # sharding, which does support distinct per-rank samples.
            base.cp_size = 1
            base.cfgp_size = 1

            # Without native internal FSDP, keep the whole service model local
            # as well. RLinf outer FSDP1 cannot re-shard tensors after Cosmos
            # FSDP2 fully_shard() has already converted them to DTensors.
            if self.internal_fsdp_node_local and not self.internal_fsdp_shard:
                raise ValueError(
                    "internal_fsdp_node_local requires internal_fsdp_shard=true"
                )
            if not self.internal_fsdp_shard:
                base.dp_replicate_size = 1
                base.dp_shard_size = 1
            elif self.internal_fsdp_node_local:
                world_size = int(os.environ.get("WORLD_SIZE", "0"))
                local_world_size = int(
                    os.environ.get("NODE_LOCAL_WORLD_SIZE")
                    or os.environ.get("LOCAL_WORLD_SIZE", "0")
                )
                if world_size < 1 or local_world_size < 1:
                    raise ValueError(
                        "internal_fsdp_node_local requires positive WORLD_SIZE "
                        "and NODE_LOCAL_WORLD_SIZE (or LOCAL_WORLD_SIZE)"
                    )
                if world_size % local_world_size != 0:
                    raise ValueError(
                        "WORLD_SIZE must be divisible by the node-local world "
                        "size for internal_fsdp_node_local: "
                        f"WORLD_SIZE={world_size}, "
                        f"local_world_size={local_world_size}"
                    )
                # Cosmos constructs a 2-D (dp_replicate, dp_shard) DeviceMesh.
                # Sharding over the contiguous node-local dimension keeps the
                # large per-block parameter all-gathers within one node, while
                # HSDP synchronizes gradients over the replica dimension.
                base.dp_replicate_size = world_size // local_world_size
                base.dp_shard_size = local_world_size
            return base

    RlinfActionServerArgs.__name__ = args_cls.__name__
    RlinfActionServerArgs.__qualname__ = args_cls.__qualname__
    RlinfActionServerArgs._rlinf_guardrails_patch = True
    libero_server.ActionServerArgs = RlinfActionServerArgs


def _write_formal_hsdp_mesh_evidence(service: Any) -> None:
    """Persist the runtime Cosmos HSDP mesh for formal-run validation.

    The formal launcher opts in with an explicit, job-specific path.  Writing
    from distributed rank zero after ``ActionModelService`` construction makes
    the artifact evidence of the resolved ``ParallelDims`` and constructed
    ``DeviceMesh``, rather than merely a copy of the requested configuration.
    """

    raw_path = os.environ.get("COSMOS_FORMAL_HSDP_EVIDENCE_PATH")
    if not raw_path:
        return

    rank_text = os.environ.get("RANK")
    if rank_text is not None:
        rank = int(rank_text)
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    else:
        rank = 0
    if rank != 0:
        return

    model = getattr(service, "model", None)
    parallel_dims = getattr(model, "parallel_dims", None)
    if parallel_dims is None:
        raise RuntimeError("formal HSDP evidence requires runtime ParallelDims")
    dp_mesh = getattr(parallel_dims, "dp_mesh", None)
    mesh_tensor = getattr(dp_mesh, "mesh", None)
    if mesh_tensor is None:
        raise RuntimeError("formal HSDP evidence requires a constructed dp_mesh")

    world_size = int(parallel_dims.world_size)
    dp_replicate = int(parallel_dims.dp_replicate)
    dp_shard = int(parallel_dims.dp_shard)
    mesh_shape = tuple(int(dimension) for dimension in mesh_tensor.shape)
    expected_shape = (dp_replicate, dp_shard)
    if mesh_shape != expected_shape:
        raise RuntimeError(
            "runtime HSDP mesh shape mismatch: "
            f"actual={mesh_shape}, expected={expected_shape}"
        )

    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.rank{rank}.pid{os.getpid()}.tmp")
    temporary.write_text(
        "\n".join(
            (
                f"job={os.environ.get('SLURM_JOB_ID', 'unknown')}",
                f"runtime_rank={rank}",
                "resolved from runtime ActionModelService.model.parallel_dims",
                "dp_replicate was resolved based on "
                f"world_size {world_size} // dp_shard {dp_shard}",
                "Building 2-D device mesh with "
                f"['dp_replicate', 'dp_shard'], [{dp_replicate}, {dp_shard}]",
                f"runtime_dp_mesh_shape={list(mesh_shape)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _patch_cosmos_cpu_offload_materialization(
    omni_mot_module: Any, *, enabled: bool
) -> None:
    """Materialize FSDP2 meta shards on CPU when native offload is active.

    Pinned Cosmos builds ``OmniMoTModel`` on ``meta``, applies FSDP2, and
    then unconditionally materializes the network on its module-level CUDA
    device.  That ordering is valid without offload, but FSDP2 rejects the
    resulting CUDA storage when ``CPUOffloadPolicy`` owns CPU shards.  Keep
    Cosmos's runtime device unchanged and override only the short
    materialization window, then explicitly run the initialization that the
    original CUDA-only branch skipped.  DCP subsequently overwrites these
    initialized parameter values; non-persistent rotary buffers still need
    this initialization pass.
    """
    model_cls = getattr(omni_mot_module, "OmniMoTModel", None)
    if model_cls is None:
        if enabled and str(getattr(omni_mot_module, "__name__", "")).startswith(
            "cosmos_framework."
        ):
            raise RuntimeError(
                "Pinned Cosmos module no longer exposes OmniMoTModel for "
                "CPU-offload materialization"
            )
        return

    build_net = getattr(model_cls, "build_net", None)
    if build_net is None:
        if enabled:
            raise RuntimeError(
                "Pinned Cosmos OmniMoTModel no longer exposes build_net for "
                "CPU-offload materialization"
            )
        return
    original_build_net = getattr(
        build_net, "_rlinf_original_build_net", build_net
    )
    if not enabled:
        model_cls.build_net = original_build_net
        return

    device_enum = getattr(omni_mot_module, "Device", None)
    if device_enum is None or not hasattr(device_enum, "CPU") or not hasattr(
        device_enum, "CUDA"
    ):
        raise RuntimeError(
            "Pinned Cosmos device enum is incompatible with CPU materialization"
        )
    if not hasattr(omni_mot_module, "DEVICE"):
        raise RuntimeError(
            "Pinned Cosmos module has no DEVICE global for CPU materialization"
        )

    def build_net_with_cpu_offload_materialization(self, *args, **kwargs):
        original_device = omni_mot_module.DEVICE
        if original_device != device_enum.CUDA:
            return original_build_net(self, *args, **kwargs)

        omni_mot_module.DEVICE = device_enum.CPU
        try:
            net = original_build_net(self, *args, **kwargs)
            net.init_weights(buffer_device=device_enum.CPU)
            if getattr(self.config, "lora_enabled", False):
                self._init_lora_weights_post_materialization(net)
            return net
        finally:
            omni_mot_module.DEVICE = original_device

    build_net_with_cpu_offload_materialization._rlinf_original_build_net = (
        original_build_net
    )
    model_cls.build_net = build_net_with_cpu_offload_materialization


def _patch_cosmos_pre_fsdp_hooks(
    *,
    apply_trainable_freeze: bool,
    trainable_param_patterns: tuple[str, ...] = (),
    blocked_trainable_param_patterns: tuple[str, ...] = (),
    activation_checkpointing_mode: str | None = None,
    cpu_offload: bool = False,
) -> None:
    """Run RLinf-specific setup BEFORE Cosmos's internal fully_shard() runs,
    by intercepting the one function both concerns need to reach ahead of
    that call: ``parallelize_vfm_network``.

    Three independent fixes share this compatibility hook purely
    because Cosmos's own model construction gives us no other hook before
    fully_shard() wraps the model:

    1. Trainable-param freeze (``apply_trainable_freeze``). Freeze/unfreeze
       the trainable action subset before wrapping, instead of after (which
       is what configure_native_trainable_parameters does). Cosmos's own
       LoRA injector (inject_lora_pre_fsdp, cosmos_framework/utils/vfm/lora.py)
       documents exactly this ordering requirement: selecting which
       parameters are trainable must happen before FSDP2 wraps the model,
       because FSDP2 builds its own per-parameter bookkeeping for the
       autograd graph at wrap time. Flipping requires_grad afterward (as
       configure_native_trainable_parameters does) can leave the DTensor
       wrapper's flag out of sync with that internal state, producing a loss
       with no grad_fn at backward() even though the parameter reports
       requires_grad=True.

    2. Activation-checkpointing override (``activation_checkpointing_mode``).
       ``cosmos_framework/inference/model.py``'s ``Cosmos3OmniModel.__init__``
       unconditionally sets ``model_dict.config.activation_checkpointing.mode
       = "none"`` before instantiating the model, on the assumption that this
       class is only ever used for inference. RLinf's native backend reuses
       the same inference model class to run a genuinely differentiable
       30-step diffusion chain replay for chain-logprob GRPO (see
       ``chain_logprob`` in ``default_forward``/``_default_forward_single``
       below), so that inference-only assumption does not hold here: every
       transformer-block activation across all 30 steps is kept live for
       backward with no recompute, which is the dominant contributor to the
       actor's ~64-68GB peak footprint documented in this pipeline's OOM
       investigation. ``parallelize_vfm_network`` receives the (already
       forced-to-"none") ``ac_config`` as a kwarg and forwards it unchanged
       into ``parallelize_unified_mot.apply_ac``, which wraps each
       transformer block in ``torch.utils.checkpoint`` purely based on
       ``ac_config.mode`` -- independent of FSDP/parallel_dims, so this
       override applies whether or not ``internal_fsdp_shard`` is enabled.
       Overriding the mode back to a real value here (default "full", the
       same default ``ActivationCheckpointingConfig`` itself ships) trades
       recompute for memory on the differentiable path without touching
       Cosmos's own architecture.

    3. CPU-offload materialization (``cpu_offload``). Install the native
       FSDP2 offload policy at every Cosmos shard site, then make
       ``OmniMoTModel.build_net`` materialize its already-sharded meta
       parameters on CPU. Runtime inference and replay still use CUDA via
       FSDP2's pre-forward movement; only construction is redirected.

    OmniMoTModel.build_net (omni_mot_model.py) calls
    ``parallelize_vfm_network(net, ...)`` exactly where its own LoRA
    injection ("Inject LoRA BEFORE FSDP wrap, while still on meta device")
    would otherwise go, and that single call fans out into BOTH fully_shard()
    sites that matter for concern 1: parallelize_vfm_network's own top-level
    fully_shard(module=model, ...) (which shards whatever lives directly on
    the VFM network, e.g. action_proj_in/out, action_modality_embed) and,
    nested inside it, parallelize_unified_mot's apply_fsdp() (which shards
    each transformer layer, including the moe_gen sublayers). We can't
    change Cosmos's own model construction to accept RLinf's
    trainable_param_patterns or activation-checkpointing preference
    directly, so we intercept parallelize_vfm_network as bound in
    omni_mot_model's own namespace (it's imported there via ``from ...
    import parallelize_vfm_network``, so patching the defining module's
    attribute would not affect this call site) and apply the pre-shard fixes
    immediately before it runs, covering both fully_shard() calls and the
    AC-mode override in one place. Both pre-shard fixes are applied by this single
    wrapper (rather than two independently-installed patches of the same
    function) because each wrapper installation unwraps back to the true
    original via ``_rlinf_original_parallelize_vfm_network`` before
    re-wrapping -- two separate monkeypatch calls targeting the same
    function would each discard the other's behavior instead of composing.
    """
    omni_mot_module = _import_cosmos_module(
        "cosmos_framework.model.generator.omni_mot_model",
        "cosmos_framework.model.vfm.omni_mot_model",
    )
    original_parallelize_vfm_network = getattr(
        omni_mot_module.parallelize_vfm_network,
        "_rlinf_original_parallelize_vfm_network",
        omni_mot_module.parallelize_vfm_network,
    )

    # Cosmos imports FSDP2's fully_shard directly into each parallelization
    # module. RLinf's outer actor.fsdp_config therefore cannot affect these
    # calls. Install a narrowly-scoped default offload policy on the three
    # modules that own the active VFM/VLM shard sites.
    parallelize_modules = (
        _import_cosmos_module(
            "cosmos_framework.model.generator.mot.parallelize_vfm_network",
            "cosmos_framework.model.vfm.mot.parallelize_vfm_network",
        ),
        _import_cosmos_module(
            "cosmos_framework.model.generator.mot.parallelize_unified_mot",
            "cosmos_framework.model.vfm.mot.parallelize_unified_mot",
        ),
        _import_cosmos_module(
            "cosmos_framework.model.generator.parallelize_vlm",
            "cosmos_framework.model.vfm.parallelize_vlm",
        ),
    )
    for parallelize_module in parallelize_modules:
        fully_shard_fn = parallelize_module.fully_shard
        original_fully_shard = getattr(
            fully_shard_fn, "_rlinf_original_fully_shard", fully_shard_fn
        )
        if cpu_offload:
            from torch.distributed.fsdp import CPUOffloadPolicy

            offload_policy = CPUOffloadPolicy(pin_memory=True)

            def fully_shard_with_cpu_offload(
                *args,
                _original=original_fully_shard,
                _offload_policy=offload_policy,
                **kwargs,
            ):
                kwargs.setdefault("offload_policy", _offload_policy)
                return _original(*args, **kwargs)

            fully_shard_with_cpu_offload._rlinf_original_fully_shard = (
                original_fully_shard
            )
            parallelize_module.fully_shard = fully_shard_with_cpu_offload
        else:
            parallelize_module.fully_shard = original_fully_shard

    def parallelize_vfm_network_with_rlinf_hooks(model, *args, **kwargs):
        if apply_trainable_freeze:
            for name, param in model.named_parameters(remove_duplicate=False):
                requested = _matches_any_parameter_pattern(
                    name, trainable_param_patterns
                )
                blocked = _matches_any_parameter_pattern(
                    name, blocked_trainable_param_patterns
                )
                param.requires_grad_(requested and not blocked)
        if activation_checkpointing_mode is not None:
            ac_config = kwargs.get("ac_config")
            if ac_config is not None:
                ac_config.mode = activation_checkpointing_mode
        return original_parallelize_vfm_network(model, *args, **kwargs)

    parallelize_vfm_network_with_rlinf_hooks._rlinf_original_parallelize_vfm_network = (
        original_parallelize_vfm_network
    )
    omni_mot_module.parallelize_vfm_network = (
        parallelize_vfm_network_with_rlinf_hooks
    )
    _patch_cosmos_cpu_offload_materialization(
        omni_mot_module, enabled=cpu_offload
    )


def _patch_cosmos_cfg_branch_checkpointing(*, enabled: bool) -> None:
    """Checkpoint each complete differentiable CFG velocity branch.

    Cosmos evaluates the conditional and unconditional branches sequentially
    when classifier-free guidance is enabled. During differentiable action
    replay, both branch graphs otherwise stay live until the sampler loss is
    backpropagated through the full denoising chain. Transformer-block
    activation checkpointing reduces each layer's footprint, but still saves
    one checkpoint input per layer, branch, and denoising step.

    A reentrant checkpoint around the complete velocity call retains only
    the branch boundary and recomputes the branch during backward. The wrapper
    is deliberately inactive under inference/no-grad, so rollout sampling is
    byte-for-byte the upstream path. CFG parallelism is also left to upstream:
    checkpointing a branch that contains point-to-point collectives requires a
    separate distributed protocol and is not used by the current FSDP run.
    """

    if not enabled:
        return

    omni_mot_module = importlib.import_module(
        "cosmos_framework.model.vfm.omni_mot_model"
    )
    model_cls = omni_mot_module.OmniMoTModel
    original = getattr(
        model_cls._run_classifier_free_guidance,
        "_rlinf_original_run_classifier_free_guidance",
        model_cls._run_classifier_free_guidance,
    )

    def run_classifier_free_guidance_with_checkpointing(
        model_self,
        cond_tokens,
        uncond_tokens,
        skip_text_tokens_for_cfg,
        single_velocity_fn,
    ):
        parallel_dims = getattr(model_self, "parallel_dims", None)
        cfgp_enabled = bool(
            parallel_dims is not None
            and getattr(parallel_dims, "cfgp_enabled", False)
        )
        if not torch.is_grad_enabled() or cfgp_enabled:
            return original(
                model_self,
                cond_tokens,
                uncond_tokens,
                skip_text_tokens_for_cfg,
                single_velocity_fn,
            )

        def checkpoint_branch(tokens, skip_text_tokens):
            # Cosmos's transformer blocks already use non-reentrant activation
            # checkpoints. A non-reentrant outer checkpoint stages the inner
            # checkpoint as a Dynamo higher-order op during backward, which
            # cannot trace FSDP2 DTensors with symbolic token shapes. Reentrant
            # outer recompute is a regular eager forward and avoids that path.
            flat_tokens, token_spec = tree_flatten(tokens)
            anchor = next(
                (leaf for leaf in flat_tokens if torch.is_tensor(leaf)), None
            )
            anchor_device = (
                anchor.device
                if anchor is not None
                else torch.device("cpu")
            )
            grad_anchor = torch.zeros(
                (), device=anchor_device, dtype=torch.float32, requires_grad=True
            )

            def branch_forward(_grad_anchor, *flat_token_leaves):
                del _grad_anchor
                rebuilt_tokens = tree_unflatten(
                    list(flat_token_leaves), token_spec
                )
                return tuple(single_velocity_fn(rebuilt_tokens, skip_text_tokens))

            # Reentrant checkpointing only discovers top-level tensor inputs.
            # Flattening preserves gradients for tensors nested inside Cosmos's
            # token containers; the anchor covers parameter-only branches.
            outputs = torch_checkpoint(
                branch_forward,
                grad_anchor,
                *flat_tokens,
                use_reentrant=True,
                preserve_rng_state=True,
            )
            if torch.is_tensor(outputs):
                return [outputs]
            return list(outputs)

        return (
            checkpoint_branch(cond_tokens, False),
            checkpoint_branch(uncond_tokens, skip_text_tokens_for_cfg),
        )

    run_classifier_free_guidance_with_checkpointing._rlinf_original_run_classifier_free_guidance = (  # noqa: E501
        original
    )
    model_cls._run_classifier_free_guidance = (
        run_classifier_free_guidance_with_checkpointing
    )


def _add_action_processing_record(
    batch: dict[str, Any], raw_action_dim: int | None
) -> None:
    """Add Cosmos action unpadding metadata when the action server omits it."""

    if "action_processing_record" in batch:
        return
    if raw_action_dim is None:
        return

    action_processing = _import_cosmos_module(
        "cosmos_framework.data.generator.action.action_processing",
        "cosmos_framework.data.vfm.action.action_processing",
    )
    record = action_processing.ActionProcessingRecord(
        raw_action_dim=int(raw_action_dim),
        action_normalizer=None,
    )
    batch_size = len(batch.get("mode", [None]))
    batch.update(
        action_processing.make_batched_action_processing_fields(
            record,
            batch_size,
        )
    )


def _patch_action_service_action_processing_record(service: Any) -> None:
    """Inject the inverse action record required by Cosmos native generation."""

    original = service.model.generate_samples_from_batch
    if getattr(original, "_rlinf_adds_action_processing_record", False):
        return

    def generate_samples_from_batch(batch, *args, **kwargs):
        _add_action_processing_record(batch, service.raw_action_dim)
        return original(batch, *args, **kwargs)

    generate_samples_from_batch._rlinf_adds_action_processing_record = True
    generate_samples_from_batch._rlinf_original_generate_samples_from_batch = original
    service.model.generate_samples_from_batch = generate_samples_from_batch


@contextmanager
def _force_full_vision_encode_during_grad(model: Any):
    """Disable Cosmos' inference-only prefix encode for differentiable replay."""
    name = "_encode_vision_x0_tokens"
    original = getattr(model, name, None)
    if original is None:
        yield
        return
    instance_dict = vars(model)
    had_instance_override = name in instance_dict
    previous_override = instance_dict.get(name)

    def full_encode(raw_state_vision, num_vision_items_per_sample, _indexes):
        return original(raw_state_vision, num_vision_items_per_sample, None)

    setattr(model, name, full_encode)
    try:
        yield
    finally:
        if had_instance_override:
            setattr(model, name, previous_override)
        else:
            delattr(model, name)



@contextmanager
def _native_service_training_mode(model: Any):
    """Use training semantics for differentiable replay and restore afterwards."""
    if not callable(getattr(model, "train", None)):
        yield
        return
    was_training = bool(getattr(model, "training", False))
    model.train(True)
    try:
        yield
    finally:
        model.train(was_training)


@contextmanager
def _normal_replay_network_inputs(model: Any):
    """Materialize ordinary constants at every differentiable module boundary."""

    net = getattr(model, "net", None)
    if not isinstance(net, nn.Module):
        yield
        return

    def normal_input_hook(_module, args, kwargs):
        return (
            _materialize_normal_replay_tensors(args),
            _materialize_normal_replay_tensors(kwargs),
        )

    handles = [
        module.register_forward_pre_hook(
            normal_input_hook, with_kwargs=True
        )
        for module in net.modules()
    ]
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def _native_replay_velocity_context(model: Any):
    # Checkpoint recomputation happens after the original generation call
    # has returned, so it must restore every model/input context used by
    # the differentiable forward. In particular, Cosmos model construction
    # leaves some constants as inference tensors; without the input hooks,
    # FSDP2 recompute reaches mixed Tensor/DTensor operators.
    with (
        _native_service_training_mode(model),
        _force_full_vision_encode_during_grad(model),
        _normal_replay_network_inputs(model),
    ):
        yield


def _generate_samples_from_batch_with_grad(
    service: Any,
    batch: dict[str, Any],
    *,
    track_grad: bool = True,
    **kwargs,
):
    """Call Cosmos generation without its inference-only no-grad wrapper.

    Two independent things disable autograd for this call unless undone:

    1. ``generate_samples_from_batch`` itself is ``@torch.no_grad()``-decorated;
       stripping it via ``__wrapped__`` (below) avoids entering a *new*
       no-grad scope for this call.
    2. Cosmos's own process-wide setup (``cosmos_framework.inference.common
       .init``) calls the bare ``torch.set_grad_enabled(False)`` — not a
       context manager, a permanent flip — whenever ``training=False``,
       which is always the case for this "action server" inference service.
       That leaves gradients globally disabled for the rest of the
       process, so (1) alone is not enough: stripping the local decorator
       just avoids adding a second no-grad layer on top of one that was
       never lifted. ``torch.enable_grad()`` explicitly re-enables
       gradients for this call regardless of that ambient state.
    """

    _add_action_processing_record(batch, getattr(service, "raw_action_dim", None))
    generate = service.model.generate_samples_from_batch
    original = getattr(
        generate, "_rlinf_original_generate_samples_from_batch", generate
    )
    if not track_grad:
        with torch.no_grad():
            return original(batch, **kwargs)

    undecorated = getattr(original, "__wrapped__", None)
    with torch.enable_grad(), _native_replay_velocity_context(service.model):
        if undecorated is None:
            return original(batch, **kwargs)
        model_self = getattr(original, "__self__", service.model)
        return undecorated(model_self, batch, **kwargs)


def _predict_policy_with_recorded_action_chain(
    service: Any,
    request: dict[str, Any],
    *,
    replay_objective: str | None = None,
    sampling_seed: int = 0,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    chain_sigma: float = 0.2,
    sigma_min: float = 1e-4,
    timestep_scale: float = 1000.0,
    transition_mode: str = ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
    transition_noise_seed: int = 0,
    action_state_rows: int = 0,
    fpo_num_mc_samples: int = 4,
    fpo_time_distribution: str = "waver",
    fpo_training_shift: float = 10.0,
    logprob_mode: str = "joint_sum",
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    """Run the native policy service while injecting an objective recorder."""

    recorder = _build_rollout_chain_sampler(
        base_sampler=getattr(service.model, "sampler", None),
        replay_objective=replay_objective,
        transition_mode=transition_mode,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
        chain_sigma=chain_sigma,
        sigma_min=sigma_min,
        timestep_scale=timestep_scale,
        transition_noise_seed=transition_noise_seed,
    )
    original = service.model.generate_samples_from_batch

    def generate_samples_from_batch_with_recorder(batch, *args, **kwargs):
        kwargs["sampler"] = recorder
        return original(batch, *args, **kwargs)

    service.model.generate_samples_from_batch = generate_samples_from_batch_with_recorder
    try:
        response = service.predict_policy(request)
    finally:
        service.model.generate_samples_from_batch = original

    replay_tensors, prev_logprobs = _finalize_cosmos_rollout_replay(
        recorder,
        replay_objective=replay_objective,
        sampling_seed=sampling_seed,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
        sigma_min=sigma_min,
        num_mc_samples=fpo_num_mc_samples,
        time_distribution=fpo_time_distribution,
        training_shift=fpo_training_shift,
        timestep_scale=timestep_scale,
        logprob_mode=logprob_mode,
    )
    return response, replay_tensors, prev_logprobs


def _image_tensor_to_chw_uint8(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(
            f"Expected a single image tensor with 3 dimensions, got {tuple(image.shape)}."
        )
    image_cpu = image.detach().cpu()
    if int(image_cpu.shape[0]) == 3:
        image_chw = image_cpu
    elif int(image_cpu.shape[-1]) == 3:
        image_chw = image_cpu.permute(2, 0, 1)
    else:
        raise ValueError(
            "Native Cosmos3 inference expects RGB images in CHW or HWC layout."
        )

    if image_chw.is_floating_point():
        image_float = image_chw.float()
        if image_float.numel() and image_float.min().item() < 0.0:
            image_float = (image_float + 1.0) / 2.0
        if image_float.numel() and image_float.max().item() <= 1.0:
            image_float = image_float * 255.0
        return image_float.clamp(0.0, 255.0).round().to(torch.uint8).contiguous()
    return image_chw.clamp(0, 255).to(torch.uint8).contiguous()


def _image_tensor_to_chw_float01(image: torch.Tensor) -> torch.Tensor:
    """Convert a single RGB image to CHW float in [0, 1].

    The UR5 action-training dataset decodes video frames as float tensors and
    performs the DROID-layout resize before its final uint8 conversion. Keep
    that operation order separate from the legacy inference path, which
    historically quantized each camera tile first.
    """
    if image.dim() != 3:
        raise ValueError(
            f"Expected a single image tensor with 3 dimensions, got {tuple(image.shape)}."
        )
    image_cpu = image.detach().cpu()
    if int(image_cpu.shape[0]) == 3:
        image_chw = image_cpu
    elif int(image_cpu.shape[-1]) == 3:
        image_chw = image_cpu.permute(2, 0, 1)
    else:
        raise ValueError("Native Cosmos3 inference expects RGB images in CHW or HWC layout.")

    image_float = image_chw.float()
    if image_float.numel():
        if image_float.min().item() < 0.0:
            image_float = (image_float + 1.0) / 2.0
        elif image_float.max().item() > 1.0:
            image_float = image_float / 255.0
    return image_float.clamp(0.0, 1.0).contiguous()


def _stitch_three_camera_views_for_policy_input(
    main_image: torch.Tensor,
    side_image_a: torch.Tensor,
    side_image_b: torch.Tensor,
    *,
    upscale_to_main_height: int | None = 480,
    real_camera_aspect_ratio: float | None = 480 / 640,
    layout: str = "main_top",
    training_transform_compatible: bool = False,
    expected_source_view_size: tuple[int, int] | None = None,
    expected_composite_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Concatenate three single-env views into the 3-camera training layout.

    Mirrors cosmos-framework's ``UR5EEFLeRobotDataset._load_concat_video``:
    the main (front) view keeps its native size on top, while the two side
    views are each scaled to half height/width and placed side-by-side on
    the bottom. Checkpoints trained on this layout (e.g. the UR5-EEF
    close-desktop policy) see a badly out-of-distribution image if only a
    single view is sent.

    Ctrl-World's own per-view working resolution (e.g. 192x320, aspect
    0.6 H/W) has a different aspect ratio than the real camera used at
    training time (480x640, aspect 0.75 H/W). Applying the "main full +
    halved sides" recipe to tiles at the wrong per-view aspect ratio
    produces an overall composite with the wrong OVERALL aspect ratio too
    (measured: 0.9 instead of the training target 1.125) -- not because
    this function's own recipe stretches anything, but because it's
    faithfully building the right *shape* out of the wrong *proportioned*
    ingredients. Correct each tile to `real_camera_aspect_ratio` first (a
    real resize of Ctrl-World's own rendered content, since there's no way
    to recover the true undistorted proportions Ctrl-World never rendered)
    so the assembled composite's shape matches what the policy was
    actually trained on. Set to ``None`` to skip (e.g. callers already
    feeding real, full-resolution camera frames, whose tiles are already
    at the training aspect ratio).

    Ctrl-World's own generated views are also much smaller (its native
    working resolution) than the real cameras used at training time, so
    the resulting composite would look soft/blocky if left at that native
    size. Once the composite is correctly assembled (at native resolution,
    so no independent per-tile padding can bake in mismatched black bars --
    see `_wrap_obs`), upscale the WHOLE composite uniformly (same factor
    for both dimensions, so no distortion) so its main-view portion matches
    `upscale_to_main_height`. Set to ``None`` to skip.

    Args:
        main_image, side_image_a, side_image_b: single-env HWC uint8 images
            (as returned by ``env_obs['main_images'][i]`` etc).

    Returns:
        HWC uint8 concatenated image.
    """
    if training_transform_compatible:
        # Match UR5EEFLeRobotDataset._apply_view_layout exactly: resize decoded
        # float [0,1] frames, concatenate, then quantize once at the end.
        main_chw = _image_tensor_to_chw_float01(main_image)
        side_a_chw = _image_tensor_to_chw_float01(side_image_a)
        side_b_chw = _image_tensor_to_chw_float01(side_image_b)
    else:
        # Preserve the established 10D inference behavior byte-for-byte.
        main_chw = _image_tensor_to_chw_uint8(main_image).float()
        side_a_chw = _image_tensor_to_chw_uint8(side_image_a).float()
        side_b_chw = _image_tensor_to_chw_uint8(side_image_b).float()

    if expected_source_view_size is not None:
        if real_camera_aspect_ratio is not None or upscale_to_main_height is not None:
            raise ValueError(
                "Strict native camera input forbids legacy aspect correction and composite upscaling"
            )
        for view_name, view in (
            ("front", main_chw),
            ("wrist", side_a_chw),
            ("right", side_b_chw),
        ):
            if tuple(view.shape[-2:]) != expected_source_view_size:
                raise ValueError(
                    f"Native {view_name} view expected {expected_source_view_size}, "
                    f"got {tuple(view.shape[-2:])}"
                )

    if real_camera_aspect_ratio is not None:
        _, cur_h, cur_w = main_chw.shape
        cur_ratio = cur_h / cur_w
        if abs(cur_ratio - real_camera_aspect_ratio) > 1e-3:
            new_w = max(1, round(cur_h / real_camera_aspect_ratio))
            main_chw, side_a_chw, side_b_chw = (
                F.interpolate(
                    tensor.unsqueeze(0),
                    size=(cur_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for tensor in (main_chw, side_a_chw, side_b_chw)
            )

    if layout == "droid":
        # UR5 joint training used wrist=D435 above front=D405_0 and
        # right=D405_1, matching the Cosmos DROID recipe exactly.
        main_chw, side_a_chw = side_a_chw, main_chw
    elif layout != "main_top":
        raise ValueError(
            f"Unsupported Cosmos camera layout {layout!r}; expected main_top or droid."
        )

    _, h_main, w_main = main_chw.shape
    half_h, half_w = h_main // 2, w_main // 2
    side_a_resized = F.interpolate(
        side_a_chw.unsqueeze(0),
        size=(half_h, half_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    side_b_resized = F.interpolate(
        side_b_chw.unsqueeze(0),
        size=(half_h, half_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    bottom = torch.cat([side_a_resized, side_b_resized], dim=-1)
    concat_chw = torch.cat([main_chw, bottom], dim=-2)

    if upscale_to_main_height is not None and h_main != upscale_to_main_height:
        scale = upscale_to_main_height / h_main
        target_h = round(concat_chw.shape[-2] * scale)
        target_w = round(concat_chw.shape[-1] * scale)
        concat_chw = F.interpolate(
            concat_chw.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if expected_composite_size is not None and tuple(concat_chw.shape[-2:]) != expected_composite_size:
        raise ValueError(
            f"Native DROID composite expected {expected_composite_size}, "
            f"got {tuple(concat_chw.shape[-2:])}"
        )

    if training_transform_compatible:
        return (
            (concat_chw * 255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
        )
    return concat_chw.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


def _dump_policy_input_audit(
    image_hwc_uint8: torch.Tensor,
    *,
    cosmos_cfg: Any,
    request_id: int,
    source_shapes: list[tuple[int, ...]],
    source_kind: str,
    strict_native_resolution: bool,
    reflection_padding_target: tuple[int, int] | None = None,
) -> Path:
    """Persist the initial 7D composite that is handed to Cosmos inference."""
    from PIL import Image

    output_dir = _resolve_native_output_dir(cosmos_cfg) / "input_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    # Keep the established audit filename for existing 7D consumers. The JSON
    # metadata records the actual source/action contract for native 10D too.
    stem = f"rank_{rank}_request_{request_id:04d}_joint7d_training_composite"
    image_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"
    image_np = image_hwc_uint8.detach().cpu().numpy()
    Image.fromarray(image_np).save(image_path)
    composite_h, composite_w = image_np.shape[:2]
    padding_metadata = None
    if reflection_padding_target is not None:
        target_h, target_w = reflection_padding_target
        padding_metadata = {
            "performed_by": "Cosmos3 action service",
            "target_shape": [target_h, target_w],
            "content_shape": [composite_h, composite_w],
            "padding_right": target_w - composite_w,
            "padding_bottom": target_h - composite_h,
            "mode": "reflect",
        }
    metadata_path.write_text(
        json.dumps(
            {
                "camera_order": ["wrist:d435", "front:d405_0", "right:d405_1"],
                "layout": "wrist top; front bottom-left; right bottom-right",
                "source_shapes": [list(shape) for shape in source_shapes],
                "source_kind": source_kind,
                "strict_native_resolution": strict_native_resolution,
                "legacy_aspect_correction": False if strict_native_resolution else None,
                "legacy_composite_upscale": False if strict_native_resolution else None,
                "composite_shape": list(image_np.shape),
                "cosmos_reflection_padding": padding_metadata,
                "pixel_min": int(image_np.min()),
                "pixel_max": int(image_np.max()),
                "stitch_transform": "training_float_resize_then_uint8",
            },
            indent=2,
        )
        + "\n"
    )
    get_logger().info("Saved Cosmos 7D policy-input audit to %s", image_path)
    return image_path


def _build_native_policy_batch(
    service: Any,
    *,
    image: torch.Tensor,
    prompt: str,
    domain_name: str,
    image_size: int,
    condition_state: torch.Tensor | None = None,
    action_state_rows: int = 0,
    training_transform_compatible: bool = False,
    resolution_tier: str | int | None = None,
) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    libero_server = importlib.import_module(
        "cosmos_framework.scripts.action_policy_server_libero"
    )

    img_chw_uint8 = _image_tensor_to_chw_uint8(image)
    img_h, img_w = img_chw_uint8.shape[-2:]
    if img_h != image_size:
        scale = image_size / img_h
        new_w = int(round(img_w * scale))
        hwc = img_chw_uint8.permute(1, 2, 0).cpu().numpy()
        resized = Image.fromarray(hwc).resize(
            (new_w, image_size), resample=Image.Resampling.BILINEAR
        )
        arr = np.asarray(resized, dtype=np.uint8).copy()
        img_chw_uint8 = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    t_frames = service.cfg.action_chunk_size + 1
    _, final_h, final_w = img_chw_uint8.shape
    video_c_t_h_w_uint8 = _build_policy_condition_video(
        img_chw_uint8,
        t_frames=t_frames,
        zero_future_frames=training_transform_compatible,
    )

    resolution = (
        libero_server.get_vision_data_resolution((final_h, final_w))
        if resolution_tier is None
        else str(resolution_tier)
    )
    target_w, target_h = libero_server.find_closest_target_size(
        final_h, final_w, resolution
    )
    pad_dict: dict[str, Any] = {"video": video_c_t_h_w_uint8}
    libero_server.reflection_pad_to_target(
        pad_dict, ["video"], True, target_w, target_h
    )
    video_padded = pad_dict["video"]
    padded_image_size = pad_dict["image_size"]

    action_rows = int(service.cfg.action_chunk_size) + int(action_state_rows)
    action_t_d = torch.zeros(
        (action_rows, service.cfg.max_action_dim),
        dtype=torch.float32,
    )
    if action_state_rows:
        if condition_state is None:
            raise ValueError("State-conditioned Cosmos batch requires condition_state.")
        state = torch.as_tensor(condition_state, dtype=torch.float32).reshape(-1)
        if state.numel() != int(service.raw_action_dim):
            raise ValueError(
                "Cosmos condition_state width must match raw_action_dim, got "
                f"{state.numel()} and {service.raw_action_dim}."
            )
        action_t_d[0, : state.numel()] = state
    input_video_key = getattr(service.model, "input_video_key", None)
    if input_video_key is None:
        input_video_key = getattr(service.model, "config", None).input_video_key

    sequence_plan = libero_server.build_sequence_plan_from_mode(
        mode="policy",
        video_length=service.cfg.action_chunk_size + 1,
        action_length=action_rows,
        has_text=True,
    )
    if training_transform_compatible:
        augmented_prompt = _augment_prompt_like_action_training_transform(
            prompt,
            t_frames=t_frames,
            fps=service.cfg.fps,
            padded_height=target_h,
            padded_width=target_w,
            append_duration_fps=service.append_duration_fps,
            append_resolution_info=service.append_resolution_info,
        )
    else:
        # Preserve the pinned server's historical 10D request behavior.
        augmented_prompt = libero_server._augment_prompt_with_metadata(
            prompt,
            t_frames=t_frames,
            fps=service.cfg.fps,
            height=final_h,
            width=final_w,
            append_duration_fps=service.append_duration_fps,
            append_resolution_info=service.append_resolution_info,
        )

    return {
        input_video_key: [[video_padded]],
        "raw_action_dim": [torch.tensor(service.raw_action_dim, dtype=torch.long)],
        "action": [[action_t_d]],
        "mode": ["policy"],
        "ai_caption": [augmented_prompt],
        "prompt": [augmented_prompt],
        "conditioning_fps": [torch.tensor(service.cfg.fps, dtype=torch.long)],
        "image_size": padded_image_size.unsqueeze(0).to(device="cuda"),
        "domain_id": [
            torch.tensor(
                libero_server.get_domain_id(domain_name),
                dtype=torch.long,
            )
        ],
        "sequence_plan": [sequence_plan],
    }


def _build_action_training_transform_policy_batch(
    *,
    transform: Any,
    batch_builder: Any,
    image: torch.Tensor,
    prompt: str,
    domain_id: int,
    action_chunk_size: int,
    raw_action_dim: int,
    conditioning_fps: int,
    resolution_tier: str | int,
    condition_state: torch.Tensor | None = None,
    action_state_rows: int = 0,
) -> dict[str, Any]:
    """Build a 7D policy batch through the checkpoint SFT transform.

    The DCP joint checkpoints were trained through ActionTransformPipeline.
    Reusing that transform here keeps spatial padding, caption metadata,
    tokenization, sequence planning, and action preprocessing identical to SFT.
    Legacy 10D checkpoints intentionally continue to use the native batch.
    """

    image_chw_uint8 = _image_tensor_to_chw_uint8(image)
    t_frames = int(action_chunk_size) + 1
    video = _build_policy_condition_video(
        image_chw_uint8,
        t_frames=t_frames,
        zero_future_frames=True,
    )
    action_rows = int(action_chunk_size) + int(action_state_rows)
    action = torch.zeros((action_rows, int(raw_action_dim)), dtype=torch.float32)
    if action_state_rows:
        if condition_state is None:
            raise ValueError("State-conditioned Cosmos batch requires condition_state.")
        state = torch.as_tensor(condition_state, dtype=torch.float32).reshape(-1).cpu()
        if state.numel() != int(raw_action_dim):
            raise ValueError(
                "Cosmos condition_state width must match raw_action_dim, got "
                f"{state.numel()} and {raw_action_dim}."
            )
        action[0] = state

    sample = {
        "ai_caption": str(prompt),
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(int(conditioning_fps), dtype=torch.long),
        "mode": "policy",
        "domain_id": torch.tensor(int(domain_id), dtype=torch.long),
        "viewpoint": "concat_view",
        "additional_view_description": _DROID_TRAINING_ADDITIONAL_VIEW_DESCRIPTION,
    }
    transformed = transform(sample, resolution_tier)
    return batch_builder(transformed)


def _remove_native_policy_reflection_padding(
    decoded_video: torch.Tensor,
    image_size: torch.Tensor | list[torch.Tensor] | None,
) -> torch.Tensor:
    """Crop decoded Cosmos video back to its pre-padding content size.

    Native action inference reflection-pads the DROID composite to the model's
    square resolution. Match the official 10D service by removing that padding
    before exposing the imagined video to RLinf's comparison renderer.
    """
    libero_server = importlib.import_module(
        "cosmos_framework.scripts.action_policy_server_libero"
    )
    return libero_server.remove_reflection_padding(decoded_video, image_size)


def probe_cosmos_native_backend(cfg: Any) -> CosmosNativeBackendProbe:
    """Probe native backend readiness without loading checkpoints."""

    cosmos_cfg = _cfg_get(cfg, "cosmos", {})
    _maybe_add_framework_path(_cfg_get(cosmos_cfg, "framework_path"))
    in_slurm_allocation = bool(os.environ.get("SLURM_JOB_ID"))
    if in_slurm_allocation:
        cuda_available, cuda_device_count, cuda_error = _cuda_state()
    else:
        cuda_available, cuda_device_count = False, 0
        cuda_error = "not probed outside Slurm GPU allocation"
    checkpoint_path = _cfg_get(cosmos_cfg, "checkpoint_path", None) or _cfg_get(
        cfg, "model_path", None
    )
    checkpoint_path_exists = False
    if checkpoint_path:
        path = _expand_path(str(checkpoint_path))
        assert path is not None
        checkpoint_path_exists = path.exists()
    checkpoint_name = _cfg_get(cosmos_cfg, "checkpoint_name", None)

    return CosmosNativeBackendProbe(
        in_slurm_allocation=in_slurm_allocation,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_error=cuda_error,
        missing_imports=_missing_imports(),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        checkpoint_path_exists=checkpoint_path_exists,
        checkpoint_name=str(checkpoint_name) if checkpoint_name else None,
        hf_token_available=bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
    )


def _require_direct_action_shape(
    *,
    raw_action_dim: Any,
    action_dim: int,
    action_chunk_size: Any,
    num_action_chunks: int,
) -> None:
    validate_direct_action_adapter(
        raw_action_dim=raw_action_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        num_action_chunks=num_action_chunks,
    )
    if int(action_chunk_size) % 4 != 0:
        raise ValueError(
            "cosmos.action_chunk_size must be a multiple of 4 for native "
            "Cosmos3 inference because the generated video length is "
            "action_chunk_size + 1 and the Wan VAE requires 4n+1 frames."
        )


def _slice_batched_tensors(
    values: dict[str, torch.Tensor], sample_idx: int
) -> dict[str, torch.Tensor]:
    return {
        key: value[sample_idx : sample_idx + 1] for key, value in values.items()
    }


def _slice_env_obs(env_obs: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    sliced = {}
    for key, value in env_obs.items():
        if torch.is_tensor(value):
            sliced[key] = value[sample_idx : sample_idx + 1]
        elif isinstance(value, list):
            sliced[key] = value[sample_idx : sample_idx + 1]
        elif isinstance(value, tuple):
            sliced[key] = value[sample_idx : sample_idx + 1]
        else:
            sliced[key] = value
    return sliced


def _prompt_from_env_obs(env_obs: dict[str, Any], fallback: str) -> str:
    descriptions = env_obs.get("task_descriptions")
    if isinstance(descriptions, (list, tuple)) and descriptions:
        candidate = descriptions[0]
    else:
        candidate = descriptions
    if candidate is not None:
        prompt = str(candidate).strip()
        if prompt:
            return prompt
    return str(fallback)


def _merge_tensor_dicts(
    dicts: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {
        key: torch.cat([item[key] for item in dicts], dim=0).contiguous()
        for key in keys
    }


class CosmosNativeInferencePolicy(nn.Module, BasePolicy):
    """Native Cosmos3 inference policy boundary.

    This class intentionally exposes native readiness without silently
    substituting placeholder tensors. Native action logprob replay remains opt-in.
    """

    def __init__(self, cfg: Any, torch_dtype=None):
        # Must precede eager_load_trainable_state -> ActionModelService ->
        # dcp.load().  Actor processes are already initialized by
        # FSDPModelManager; Native rollout processes reach this call first.
        _ensure_cosmos_dcp_hybrid_process_group()
        super().__init__()
        self.cfg = cfg
        self.action_dim = int(cfg.action_dim)
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.torch_dtype = torch_dtype
        self.cosmos_cfg = cfg.get("cosmos", {})
        self.action_representation = _cfg_get(
            self.cosmos_cfg, "action_representation", None
        )
        if not self.action_representation:
            raise ValueError(
                "cosmos.action_representation is required for native backend."
            )
        self.action_contract = resolve_cosmos_action_contract(
            str(self.action_representation), action_dim=self.action_dim
        )
        self.action_state_rows = self.action_contract.action_state_rows
        self.max_action_dim = _cfg_get(self.cosmos_cfg, "max_action_dim", None)
        if self.max_action_dim is None:
            raise ValueError(
                "cosmos.max_action_dim is required for native backend; it is "
                "the model-internal padded action width, not a silent action "
                "representation conversion."
            )
        if int(self.max_action_dim) < self.action_dim:
            raise ValueError(
                "cosmos.max_action_dim must be greater than or equal to "
                f"action_dim, got {self.max_action_dim} and {self.action_dim}."
            )
        _require_direct_action_shape(
            raw_action_dim=_cfg_get(self.cosmos_cfg, "raw_action_dim", None),
            action_dim=self.action_dim,
            action_chunk_size=_cfg_get(self.cosmos_cfg, "action_chunk_size", None),
            num_action_chunks=self.num_action_chunks,
        )
        self.probe = probe_cosmos_native_backend(cfg)
        patterns = _cfg_get(self.cosmos_cfg, "trainable_param_patterns", None)
        self.trainable_param_patterns = _as_tuple_of_str(
            patterns, DEFAULT_NATIVE_TRAINABLE_PARAM_PATTERNS
        )
        if not self.trainable_param_patterns:
            raise ValueError("cosmos.trainable_param_patterns must not be empty.")
        blocked_patterns = _cfg_get(
            self.cosmos_cfg, "blocked_trainable_param_patterns", None
        )
        self.blocked_trainable_param_patterns = _as_tuple_of_str(
            blocked_patterns, DEFAULT_NATIVE_BLOCKED_TRAINABLE_PARAM_PATTERNS
        )
        self._service = None
        self._joint_action_training_transform = None
        self._joint_action_training_dataset_cfg = None
        self._logged_chain_logprob_semantics = False
        object.__setattr__(self, "_native_service_model", None)
        self.native_trainable_proxy = None
        self._native_rollout_service_evicted = False
        self.actor_replay_step_checkpointing = bool(
            _cfg_get(self.cosmos_cfg, "actor_replay_step_checkpointing", False)
        )
        if self.actor_replay_step_checkpointing:
            get_logger().info(
                "[Checkpoint][cosmos] actor replay denoise-step checkpointing enabled"
            )
        eager_trainable_state = _cfg_get(
            self.cosmos_cfg, "eager_load_trainable_state", False
        )
        if isinstance(eager_trainable_state, str):
            eager_trainable_state = eager_trainable_state.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if eager_trainable_state:
            self.ensure_native_trainable_state()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support {forward_type}.")

    def _get_joint_action_training_transform(self) -> tuple[Any, Any, dict[str, Any]]:
        if (
            self._joint_action_training_transform is not None
            and self._joint_action_training_dataset_cfg is not None
        ):
            robolab = importlib.import_module(
                "cosmos_framework.scripts.action_policy_server_robolab"
            )
            return (
                self._joint_action_training_transform,
                robolab,
                self._joint_action_training_dataset_cfg,
            )

        training_config_path = _expand_path(
            _cfg_get(self.cosmos_cfg, "training_config_path", None)
        ) or _expand_path(
            _cfg_get(self.cosmos_cfg, "checkpoint_config_path", None)
        )
        if training_config_path is None:
            raise CosmosNativeBackendUnavailable(
                "Training-transform-compatible Cosmos checkpoints require "
                "cosmos.training_config_path or cosmos.checkpoint_config_path."
            )

        import yaml
        inference_training_config_path = _native_checkpoint_config_compat_path(
            training_config_path,
            _resolve_native_output_dir(self.cosmos_cfg),
            _cfg_get(self.cosmos_cfg, "framework_path", None),
        )

        raw_config = yaml.safe_load(inference_training_config_path.read_text())
        try:
            datasets = raw_config["dataloader_train"]["dataloader"]["datasets"]
            dataset_cfg = next(iter(datasets.values()))["dataset"]
        except (KeyError, StopIteration, TypeError) as exc:
            raise CosmosNativeBackendUnavailable(
                "Checkpoint config does not contain the action SFT dataset transform "
                f"at dataloader_train.dataloader.datasets: {training_config_path}"
            ) from exc

        raw_use_state = dataset_cfg.get("use_state", False)
        use_state = (
            raw_use_state.lower() == "true"
            if isinstance(raw_use_state, str)
            else bool(raw_use_state)
        )
        if use_state != bool(self.action_state_rows):
            raise CosmosNativeBackendUnavailable(
                "Checkpoint use_state does not match the selected action contract: "
                f"{use_state} vs action_state_rows={self.action_state_rows}."
            )
        chunk_length = int(dataset_cfg.get("chunk_length", self.num_action_chunks))
        if chunk_length != self.num_action_chunks:
            raise CosmosNativeBackendUnavailable(
                "Checkpoint chunk_length does not match actor.model.num_action_chunks: "
                f"{chunk_length} vs {self.num_action_chunks}."
            )
        max_action_dim = int(dataset_cfg.get("max_action_dim", self.max_action_dim))
        if max_action_dim != int(self.max_action_dim):
            raise CosmosNativeBackendUnavailable(
                "Checkpoint max_action_dim does not match runtime configuration: "
                f"{max_action_dim} vs {self.max_action_dim}."
            )

        robolab = importlib.import_module(
            "cosmos_framework.scripts.action_policy_server_robolab"
        )
        dataset_target = str(dataset_cfg.get("_target_", ""))
        format_prompt_as_json = bool(
            _cfg_get(
                self.cosmos_cfg,
                "format_prompt_as_json",
                dataset_target.endswith(
                    "ur5_close_laptop_policy_dataset.get_ur5_close_laptop_policy_sft_dataset"
                ),
            )
        )
        transform = robolab.ActionTransformPipeline(
            tokenizer_config=dataset_cfg.get("tokenizer_config"),
            cfg_dropout_rate=0.0,
            max_action_dim=max_action_dim,
            append_viewpoint_info=bool(
                dataset_cfg.get("append_viewpoint_info", True)
            ),
            append_duration_fps_timestamps=bool(
                dataset_cfg.get("append_duration_fps_timestamps", True)
            ),
            append_resolution_info=bool(
                dataset_cfg.get("append_resolution_info", True)
            ),
            append_idle_frames=False,
            format_prompt_as_json=format_prompt_as_json,
        )
        self._joint_action_training_transform = transform
        self._joint_action_training_dataset_cfg = dataset_cfg
        return transform, robolab, dataset_cfg

    def _build_joint_action_training_batch(
        self,
        *,
        service: Any,
        image: torch.Tensor,
        prompt: str,
        domain_name: str,
        condition_state: torch.Tensor | None,
        resolution_tier: str | int | None = None,
    ) -> dict[str, Any]:
        transform, robolab, dataset_cfg = self._get_joint_action_training_transform()
        checkpoint_fps = int(float(dataset_cfg.get("fps", service.cfg.fps)))
        runtime_fps = int(service.cfg.fps)
        if checkpoint_fps != runtime_fps:
            raise CosmosNativeBackendUnavailable(
                "Checkpoint fps does not match the runtime Cosmos fps: "
                f"{checkpoint_fps} vs {runtime_fps}."
            )
        resolution = (
            dataset_cfg.get("resolution", self.action_contract.resolution_tier)
            if resolution_tier is None
            else resolution_tier
        )
        return _build_action_training_transform_policy_batch(
            transform=transform,
            batch_builder=robolab._build_data_batch_from_sample,
            image=image,
            prompt=prompt,
            domain_id=robolab.get_domain_id(domain_name),
            action_chunk_size=self.num_action_chunks,
            raw_action_dim=self.action_dim,
            conditioning_fps=runtime_fps,
            resolution_tier=resolution,
            condition_state=condition_state,
            action_state_rows=self.action_state_rows,
        )

    def _default_forward_single(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        track_grad: bool = True,
    ) -> dict[str, torch.Tensor]:
        service = self._get_service()
        replay_objective = self._replay_objective()
        if (
            bool(_cfg_get(self.cosmos_cfg, "replay_eval_mode", False))
            or replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
        ):
            service.model.eval()

        sampling_seed_tensor = forward_inputs.get("action_sampling_seed")
        sampling_seed = (
            int(sampling_seed_tensor.reshape(-1)[0].item())
            if torch.is_tensor(sampling_seed_tensor)
            else int(_cfg_get(self.cosmos_cfg, "seed", 0))
        )
        image = forward_inputs["action_condition_main_images"][0]
        domain_name = (
            _cfg_get(self.cosmos_cfg, "domain_name", None)
            or self.action_contract.domain_name
        )
        if not domain_name:
            raise ValueError("cosmos.domain_name is required for native backend.")
        prompt = _cfg_get(
            self.cosmos_cfg,
            "prompt",
            "Predict the next robot action chunk from the current observation.",
        )
        condition_state = (
            forward_inputs["action_condition_states"][0]
            if self.action_state_rows
            else None
        )
        configured_training_transform = bool(
            _cfg_get(self.cosmos_cfg, "training_transform_compatible", False)
        )
        actor_replay_size = _cfg_get(self.cosmos_cfg, "actor_replay_size", None)
        replay_resolution_tier = (
            _cfg_get(self.cosmos_cfg, "actor_replay_resolution_tier", None)
            if actor_replay_size is not None
            else None
        )
        if self.action_contract.training_transform_compatible:
            batch = self._build_joint_action_training_batch(
                service=service,
                image=image,
                prompt=str(prompt),
                domain_name=str(domain_name),
                condition_state=condition_state,
                resolution_tier=replay_resolution_tier,
            )
        else:
            if configured_training_transform:
                prompt = _append_droid_concat_view_prompt_metadata(str(prompt))
            image_size = int(_image_tensor_to_chw_uint8(image).shape[-2])
            resolution_tier = replay_resolution_tier
            if resolution_tier is None:
                resolution_tier = _cfg_get(
                    self.cosmos_cfg,
                    "resolution_tier",
                    self.action_contract.resolution_tier,
                )
            batch = _build_native_policy_batch(
                service,
                image=image,
                prompt=str(prompt),
                domain_name=str(domain_name),
                image_size=image_size,
                condition_state=condition_state,
                action_state_rows=self.action_state_rows,
                training_transform_compatible=configured_training_transform,
                resolution_tier=resolution_tier,
            )

        if replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD:
            sampler = CosmosActionHeadFPOSampler(
                clean_joint_latent=forward_inputs["fpo_clean_joint_latent"],
                clean_action_normalized=forward_inputs[
                    "fpo_clean_action_normalized"
                ],
                condition_image=forward_inputs["action_condition_main_images"],
                base_times=forward_inputs["fpo_base_times"],
                sigmas=forward_inputs["fpo_sigmas"],
                timesteps=forward_inputs["fpo_timesteps"],
                noise_seeds=forward_inputs["fpo_noise_seeds"],
                num_action_chunks=self.num_action_chunks,
                raw_action_dim=self.action_dim,
                max_action_dim=int(self.max_action_dim),
                action_state_rows=self.action_state_rows,
                latent_downsample_factor=int(
                    _cfg_get(self.cosmos_cfg, "latent_downsample_factor", 16)
                ),
                vision_state_channels=int(
                    _cfg_get(self.cosmos_cfg, "fpo_vision_state_channels", 48)
                ),
                vision_condition_latent_frames=int(
                    _cfg_get(
                        self.cosmos_cfg,
                        "fpo_vision_condition_latent_frames",
                        1,
                    )
                ),
                score_parameterization=str(
                    _cfg_get(
                        self.cosmos_cfg,
                        "fpo_score_parameterization",
                        "velocity",
                    )
                ),
            )
            replay_num_steps = int(forward_inputs["fpo_sigmas"].shape[1])
        else:
            sampler = CosmosNativeActionReplaySampler(
                full_chains=forward_inputs["native_full_chains"],
                action_chains=forward_inputs["action_chains"],
                action_denoise_timesteps=forward_inputs[
                    "action_denoise_timesteps"
                ],
                action_sigmas=forward_inputs["action_sigmas"],
                action_mask=forward_inputs["action_mask"],
                num_action_chunks=self.num_action_chunks,
                raw_action_dim=self.action_dim,
                max_action_dim=int(self.max_action_dim),
                action_state_rows=self.action_state_rows,
                sigma_min=float(
                    _cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)
                ),
                action_denoise_indices=forward_inputs.get(
                    "action_denoise_indices"
                ),
                logprob_mode=str(
                    _cfg_get(self.cosmos_cfg, "chain_logprob_mode", "joint_sum")
                ),
                checkpoint_velocity_calls=self.actor_replay_step_checkpointing,
                checkpoint_velocity_context_factory=lambda: (
                    _native_replay_velocity_context(service.model)
                ),
            )
            replay_num_steps = int(_cfg_get(self.cosmos_cfg, "num_steps", 2))

        with service._lock:
            _generate_samples_from_batch_with_grad(
                service,
                batch,
                track_grad=track_grad,
                guidance=float(_cfg_get(self.cosmos_cfg, "guidance", 1.0)),
                seed=[_cosmos_sampler_seed(sampling_seed)],
                num_steps=replay_num_steps,
                shift=float(_cfg_get(self.cosmos_cfg, "shift", 5.0)),
                has_negative_prompt=False,
                sampler=sampler,
            )

        if replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD:
            replay_outputs = {
                "logprobs": sampler.logprobs,
                "fpo_pair_losses": sampler.pair_losses,
            }
        else:
            replay_outputs = {
                "logprobs": sampler.logprobs,
                "replay_action_transition_means": (
                    sampler.action_transition_means
                ),
                "replay_transition_logprobs": sampler.transition_logprobs,
                "replay_initial_logprobs": sampler.initial_logprobs,
            }
        missing_outputs = [
            key for key, value in replay_outputs.items() if value is None
        ]
        if missing_outputs:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 replay sampler did not compute required "
                f"outputs: {missing_outputs}."
            )
        logprobs = replay_outputs["logprobs"]
        if track_grad and not logprobs.requires_grad:
            trainable = [
                name
                for name, param in service.model.named_parameters()
                if param.requires_grad
            ]
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 replay produced detached action scores before "
                "the PPO loss: "
                f"grad_enabled={torch.is_grad_enabled()}, "
                f"trainable_parameter_count={len(trainable)}, "
                f"trainable_examples={trainable[:8]}."
            )
        return replay_outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor] | None = None,
        compute_logprobs: bool = True,
        track_grad: bool = True,
        return_replay_diagnostics: bool = False,
        **kwargs,
    ):
        if not compute_logprobs:
            return {}
        if forward_inputs is None:
            raise KeyError("Native Cosmos replay requires forward_inputs.")

        replay_objective = self._replay_objective()
        validate_cosmos_forward_inputs(
            forward_inputs,
            replay_objective=replay_objective,
        )
        required_replay_key = (
            "fpo_clean_joint_latent"
            if replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
            else "native_full_chains"
        )
        if required_replay_key not in forward_inputs:
            raise NotImplementedError(
                "Native Cosmos3 current-weight replay is missing its "
                f"objective-specific field: {required_replay_key}."
            )
        if not self.probe.ready:
            reason = self.probe.blocker_reason()
            raise CosmosNativeBackendUnavailable(
                f"Native Cosmos3 replay is unavailable: {reason}"
            )

        self.ensure_native_trainable_state()
        forward_inputs = _materialize_normal_replay_tensors(forward_inputs)
        batch_size = int(forward_inputs[required_replay_key].shape[0])
        sample_outputs: list[dict[str, torch.Tensor]] = []
        grad_context = torch.enable_grad if track_grad else torch.no_grad
        with grad_context():
            for sample_idx in range(batch_size):
                sample_outputs.append(
                    self._default_forward_single(
                        _slice_batched_tensors(forward_inputs, sample_idx),
                        track_grad=track_grad,
                    )
                )
            concat_context = torch.enable_grad if track_grad else torch.no_grad
            with concat_context():
                output = {
                    "logprobs": torch.cat(
                        [item["logprobs"] for item in sample_outputs], dim=0
                    ).contiguous()
                }
                if return_replay_diagnostics:
                    diagnostic_keys = (
                        ("fpo_pair_losses",)
                        if replay_objective
                        == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
                        else (
                            "replay_action_transition_means",
                            "replay_transition_logprobs",
                            "replay_initial_logprobs",
                        )
                    )
                    for key in diagnostic_keys:
                        output[key] = torch.cat(
                            [item[key] for item in sample_outputs], dim=0
                        ).contiguous()
            if track_grad and not output["logprobs"].requires_grad:
                raise CosmosNativeBackendUnavailable(
                    "Native Cosmos3 batch replay detached action scores during "
                    "aggregation despite differentiable per-sample replay."
                )
            return output

    def _replay_objective(self) -> str:
        return resolve_cosmos_replay_objective(
            _cfg_get(self.cosmos_cfg, "replay_objective", None)
        )

    def _finalize_rollout_recorder(
        self,
        recorder: Any,
        *,
        sampling_seed: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return _finalize_cosmos_rollout_replay(
            recorder,
            replay_objective=self._replay_objective(),
            sampling_seed=sampling_seed,
            num_action_chunks=self.num_action_chunks,
            raw_action_dim=self.action_dim,
            max_action_dim=int(self.max_action_dim),
            action_state_rows=self.action_state_rows,
            sigma_min=float(
                _cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)
            ),
            num_mc_samples=int(
                _cfg_get(self.cosmos_cfg, "fpo_num_mc_samples", 4)
            ),
            time_distribution=str(
                _cfg_get(self.cosmos_cfg, "fpo_time_distribution", "waver")
            ),
            training_shift=float(
                _cfg_get(self.cosmos_cfg, "fpo_training_shift", 10.0)
            ),
            timestep_scale=float(
                _cfg_get(self.cosmos_cfg, "fpo_timestep_scale", 1000.0)
            ),
            logprob_mode=str(
                _cfg_get(self.cosmos_cfg, "chain_logprob_mode", "joint_sum")
            ),
        )

    def _sample_joint_policy(
        self,
        *,
        service: Any,
        image: torch.Tensor,
        prompt: str,
        domain_name: str,
        condition_state: torch.Tensor | None,
        chain_logprob: bool,
        vision_seed: int,
        action_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        """Sample a training-compatible checkpoint with its exact SFT contract."""

        batch = self._build_joint_action_training_batch(
            service=service,
            image=image,
            prompt=prompt,
            domain_name=domain_name,
            condition_state=condition_state,
        )
        recorder = None
        original = service.model.generate_samples_from_batch
        if chain_logprob:
            recorder = _build_rollout_chain_sampler(
                base_sampler=getattr(service.model, "sampler", None),
                replay_objective=self._replay_objective(),
                transition_mode=ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
                num_action_chunks=self.num_action_chunks,
                raw_action_dim=self.action_dim,
                max_action_dim=int(self.max_action_dim),
                action_state_rows=self.action_state_rows,
                chain_sigma=float(_cfg_get(self.cosmos_cfg, "chain_sigma", 0.2)),
                sigma_min=float(_cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)),
                timestep_scale=float(_cfg_get(self.cosmos_cfg, "timestep_scale", 1000.0)),
                transition_noise_seed=derive_matched_transition_noise_seed(
                    action_seed
                ),
            )

            def generate_with_recorder(batch, *args, **kwargs):
                kwargs["sampler"] = recorder
                return original(batch, *args, **kwargs)

            service.model.generate_samples_from_batch = generate_with_recorder
        try:
            with service._lock:
                # Do not use inference_mode on a model that will subsequently
                # run differentiable FSDP2 replay. FSDP2 lazily creates and
                # caches each all-gather output and unsharded Parameter on its
                # first forward. If that allocation happens in inference_mode,
                # the cached frozen weights remain inference tensors; a later
                # autograd replay then fails as soon as (for example) RMSNorm
                # needs to save such a weight to differentiate its input.
                # no_grad has identical rollout/autograd semantics here while
                # keeping those persistent FSDP buffers ordinary tensors.
                with torch.no_grad():
                    if vision_seed == action_seed:
                        # Cross-rank exploration seeds are auditable int63 values,
                        # while pinned Cosmos ultimately constructs NumPy
                        # RandomState instances. Use the exact uint32 value
                        # recorded in the seed manifest for the native joint
                        # vision/action sample.
                        api_seed = _cosmos_numpy_noise_seed(
                            "cosmos_joint_initial_noise", vision_seed
                        )
                        seed_context = nullcontext()
                    else:
                        # The adapter replaces both modality noise seeds at the
                        # creation point. The public API seed still has to be a
                        # legal uint32 in case pinned Cosmos validates it first.
                        api_seed = _cosmos_numpy_noise_seed(
                            "cosmos_vision_initial_noise", vision_seed
                        )
                        seed_context = _split_cosmos_modality_noise_seeds(
                            service.model,
                            vision_seed=vision_seed,
                            action_seed=action_seed,
                            sampler_name=str(
                                _cfg_get(self.cosmos_cfg, "sampler", "unipc")
                            ),
                        )
                    with seed_context:
                        samples = service.model.generate_samples_from_batch(
                            batch,
                            guidance=float(
                                _cfg_get(self.cosmos_cfg, "guidance", 1.0)
                            ),
                            seed=[api_seed],
                            num_steps=int(
                                _cfg_get(self.cosmos_cfg, "num_steps", 2)
                            ),
                            shift=float(
                                _cfg_get(self.cosmos_cfg, "shift", 5.0)
                            ),
                            has_negative_prompt=False,
                        )
        finally:
            service.model.generate_samples_from_batch = original

        action = samples["action"][0].float().squeeze(0)
        start = self.action_state_rows
        stop = start + self.num_action_chunks
        if action.ndim != 2 or action.shape[0] < stop or action.shape[1] < self.action_dim:
            raise CosmosNativeBackendUnavailable(
                "Training-compatible Cosmos response has invalid action shape "
                f"{tuple(action.shape)}; expected at least ({stop}, {self.action_dim})."
            )
        # The SFT transform trains every real action channel in normalized
        # model space. The joint-policy path must therefore invert those
        # statistics before handing actions to the environment. Bound the
        # diffusion sample to the transform's training support first;
        # otherwise model-space extrapolation becomes a many-centimetre EEF
        # delta after affine inversion.
        normalized_actions = action[start:stop, : self.action_dim].clamp(-1.0, 1.0)
        actions = service._denormalize_action(normalized_actions).unsqueeze(0).to(image.device)
        decoded = service.model.decode(samples["vision"][0]).squeeze(0)
        decoded = _remove_native_policy_reflection_padding(
            decoded, batch.get("image_size")
        )
        frames = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        frames = frames.permute(1, 2, 3, 0)
        video = frames[1 : 1 + self.num_action_chunks].unsqueeze(0).to(image.device)
        if video.shape[1] != self.num_action_chunks:
            raise CosmosNativeBackendUnavailable(
                "Training-compatible Cosmos response returned fewer video frames than actions."
            )

        replay_tensors = None
        prev_logprobs = None
        if recorder is not None:
            replay_tensors, prev_logprobs = self._finalize_rollout_recorder(
                recorder,
                sampling_seed=action_seed,
            )
        return actions, video, replay_tensors, prev_logprobs

    def _sample_training_transform_eef_policy(
        self,
        *,
        service: Any,
        image: torch.Tensor,
        prompt: str,
        domain_name: str,
        chain_logprob: bool,
        vision_seed: int,
        action_seed: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        """Sample a 10D EEF policy with the SFT/Diffusers input transform.

        The pinned HTTP-style action service predates the qnorm-c32 training
        transform: it repeats the initial image into every future frame and
        omits concat-view prompt metadata. The standalone two-way Diffusers
        runner instead uses ActionTransformPipeline (first frame followed by
        zeros, DROID concat metadata, padded resolution metadata). Keep this
        behavior opt-in so historical rollout launchers remain unchanged.

        When chain_logprob is enabled, record the exact stochastic action
        diffusion chain so GRPO can replay it with current trainable weights.
        """
        libero_server = importlib.import_module(
            "cosmos_framework.scripts.action_policy_server_libero"
        )
        prompt = _append_droid_concat_view_prompt_metadata(prompt)
        batch = _build_native_policy_batch(
            service,
            image=image,
            prompt=prompt,
            domain_name=domain_name,
            image_size=int(_image_tensor_to_chw_uint8(image).shape[-2]),
            training_transform_compatible=True,
            resolution_tier=_cfg_get(self.cosmos_cfg, "resolution_tier", None),
        )

        recorder = None
        original = service.model.generate_samples_from_batch
        if chain_logprob:
            recorder = _build_rollout_chain_sampler(
                base_sampler=getattr(service.model, "sampler", None),
                replay_objective=self._replay_objective(),
                transition_mode=ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
                num_action_chunks=self.num_action_chunks,
                raw_action_dim=self.action_dim,
                max_action_dim=int(self.max_action_dim),
                action_state_rows=self.action_state_rows,
                chain_sigma=float(_cfg_get(self.cosmos_cfg, "chain_sigma", 0.2)),
                sigma_min=float(_cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)),
                timestep_scale=float(_cfg_get(self.cosmos_cfg, "timestep_scale", 1000.0)),
                transition_noise_seed=derive_matched_transition_noise_seed(
                    action_seed
                ),
            )

            def generate_with_recorder(batch, *args, **kwargs):
                kwargs["sampler"] = recorder
                return original(batch, *args, **kwargs)

            service.model.generate_samples_from_batch = generate_with_recorder
        try:
            with service._lock:
                # See _sample_joint_policy: a shared training actor must not
                # prime FSDP2's persistent full-parameter buffers inside
                # inference_mode before its differentiable replay.
                with torch.no_grad():
                    seed_context = (
                        _split_cosmos_modality_noise_seeds(
                            service.model,
                            vision_seed=vision_seed,
                            action_seed=action_seed,
                            sampler_name=str(
                                _cfg_get(self.cosmos_cfg, "sampler", "unipc")
                            ),
                        )
                        if vision_seed != action_seed
                        else nullcontext()
                    )
                    with seed_context:
                        samples = service.model.generate_samples_from_batch(
                            batch,
                            guidance=float(
                                _cfg_get(self.cosmos_cfg, "guidance", 1.0)
                            ),
                            seed=[int(vision_seed)],
                            num_steps=int(
                                _cfg_get(self.cosmos_cfg, "num_steps", 2)
                            ),
                            shift=float(
                                _cfg_get(self.cosmos_cfg, "shift", 5.0)
                            ),
                            has_negative_prompt=False,
                        )
                    pred_action = samples["action"][0].float().squeeze(0)
                    pred_action = service._denormalize_action(
                        pred_action.clamp(-1.0, 1.0)
                    )
                    decoded = service.model.decode(samples["vision"][0]).squeeze(0)
                    decoded = libero_server.remove_reflection_padding(
                        decoded, batch["image_size"]
                    )
        finally:
            service.model.generate_samples_from_batch = original

        stop = self.num_action_chunks
        if pred_action.ndim != 2 or pred_action.shape[0] < stop:
            raise CosmosNativeBackendUnavailable(
                "Training-transform Cosmos response has invalid action shape "
                f"{tuple(pred_action.shape)}; expected at least ({stop}, {self.action_dim})."
            )
        actions = pred_action[:stop, : self.action_dim].unsqueeze(0).to(image.device)

        pil_frames = libero_server._video_tensor_to_pil_images(decoded)
        predicted_frames = pil_frames[1 : 1 + self.num_action_chunks]
        if len(predicted_frames) != self.num_action_chunks:
            raise CosmosNativeBackendUnavailable(
                "Training-transform Cosmos response returned fewer predicted "
                "video frames than actions."
            )
        import numpy as np

        video = torch.stack(
            [
                torch.from_numpy(np.asarray(frame, dtype=np.uint8).copy())
                for frame in predicted_frames
            ],
            dim=0,
        ).unsqueeze(0).to(image.device)

        replay_tensors = None
        prev_logprobs = None
        if recorder is not None:
            replay_tensors, prev_logprobs = self._finalize_rollout_recorder(
                recorder,
                sampling_seed=action_seed,
            )
        return actions, video, replay_tensors, prev_logprobs


    def _predict_action_single(self, env_obs: dict[str, Any], request_id: int):
        policy_override_keys = (
            "policy_main_images",
            "policy_wrist_images",
            "policy_extra_view_images",
        )
        uses_raw_policy_views = all(
            torch.is_tensor(env_obs.get(key)) for key in policy_override_keys
        )
        image, wrist_image, extra_view_image = _select_policy_input_views(env_obs)
        assert torch.is_tensor(image)
        training_transform_compatible = (
            self.action_contract.training_transform_compatible
            or bool(
                _cfg_get(
                    self.cosmos_cfg,
                    "training_transform_compatible",
                    False,
                )
            )
        )

        # Checkpoints trained on a 3-camera concatenated layout (main view on
        # top, two side views scaled to half size and placed side-by-side on
        # the bottom -- see cosmos-framework's UR5EEFLeRobotDataset) go
        # badly out-of-distribution if only the single main view is sent.
        # When the env provides the other two views, reproduce that layout.
        if torch.is_tensor(wrist_image) and torch.is_tensor(extra_view_image):
            source_shapes = [
                tuple(image[0].shape),
                tuple(wrist_image[0].shape),
                tuple(extra_view_image[0].shape),
            ]
            strict_source_size_cfg = _cfg_get(
                self.cosmos_cfg, "strict_source_view_size", None
            )
            strict_source_size = (
                tuple(int(value) for value in strict_source_size_cfg)
                if strict_source_size_cfg is not None
                else None
            )
            expected_composite_cfg = _cfg_get(
                self.cosmos_cfg, "expected_composite_size", None
            )
            expected_composite_size = (
                tuple(int(value) for value in expected_composite_cfg)
                if expected_composite_cfg is not None
                else None
            )
            image = _stitch_three_camera_views_for_policy_input(
                image[0],
                wrist_image[0],
                extra_view_image[0],
                upscale_to_main_height=None if strict_source_size is not None else 480,
                real_camera_aspect_ratio=None if strict_source_size is not None else 480 / 640,
                layout=str(
                    _cfg_get(
                        self.cosmos_cfg,
                        "camera_layout",
                        self.action_contract.camera_layout,
                    )
                ),
                training_transform_compatible=training_transform_compatible,
                expected_source_view_size=strict_source_size,
                expected_composite_size=expected_composite_size,
            ).unsqueeze(0)
            if (
                training_transform_compatible
                and bool(_cfg_get(self.cosmos_cfg, "input_audit", False))
            ):
                reflection_target_cfg = _cfg_get(
                    self.cosmos_cfg, "reflection_padding_target", None
                )
                reflection_target = (
                    tuple(int(value) for value in reflection_target_cfg)
                    if reflection_target_cfg is not None
                    else None
                )
                _dump_policy_input_audit(
                    image[0],
                    cosmos_cfg=self.cosmos_cfg,
                    request_id=request_id,
                    source_shapes=source_shapes,
                    source_kind="raw_reset" if uses_raw_policy_views else "ctrl_world_generated",
                    strict_native_resolution=strict_source_size is not None,
                    reflection_padding_target=reflection_target,
                )

        chain_logprob = bool(_cfg_get(self.cosmos_cfg, "chain_logprob", False))
        replay_objective = self._replay_objective()
        transition_mode = ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE

        service = self._get_service()
        image_b64, image_size = _image_tensor_to_png_request(image[0])
        domain_name = _cfg_get(self.cosmos_cfg, "domain_name", None) or self.action_contract.domain_name
        if not domain_name:
            raise ValueError("cosmos.domain_name is required for native backend.")
        fallback_prompt = _cfg_get(
            self.cosmos_cfg,
            "prompt",
            "Predict the next robot action chunk from the current observation.",
        )
        if bool(_cfg_get(self.cosmos_cfg, "force_config_prompt", False)):
            prompt = str(fallback_prompt)
        else:
            prompt = _prompt_from_env_obs(env_obs, str(fallback_prompt))
        request = {
            "image": image_b64,
            "prompt": prompt,
            "domain_name": str(domain_name),
            "image_size": image_size,
            "request_id": int(request_id),
        }

        replay_tensors = None
        prev_logprobs = None
        base_seed = int(_cfg_get(self.cosmos_cfg, "seed", 0))
        member_stride = int(
            _cfg_get(self.cosmos_cfg, "group_member_seed_stride", 0)
        )
        call_stride = int(
            _cfg_get(self.cosmos_cfg, "sampling_call_seed_stride", 0)
        )
        call_index = int(getattr(self, "_cosmos_denoise_batches", 0))
        legacy_sampling_seed = (
            base_seed
            + (int(request_id) - 1) * member_stride
            + call_index * call_stride
        )
        explicit_vision_seed = env_obs.get("vision_noise_seed")
        explicit_action_seed = env_obs.get("action_noise_seed")
        if (explicit_vision_seed is None) != (explicit_action_seed is None):
            raise ValueError(
                "Cosmos requires vision_noise_seed and action_noise_seed together"
            )
        if explicit_vision_seed is None:
            vision_seed = legacy_sampling_seed
            action_seed = legacy_sampling_seed
            split_modality_seeds = False
        else:
            vision_seed = int(torch.as_tensor(explicit_vision_seed).reshape(-1)[0])
            action_seed = int(torch.as_tensor(explicit_action_seed).reshape(-1)[0])
            split_modality_seeds = True
        request["seed"] = _cosmos_sampler_seed(action_seed)
        max_rollout_retries = int(
            _cfg_get(self.cosmos_cfg, "rollout_retry_max_retries", 0)
        )
        if max_rollout_retries < 0:
            raise ValueError("cosmos.rollout_retry_max_retries must be non-negative")
        rollout_retry_count = 0

        def sample_with_seed_reuse(sample_fn):
            nonlocal rollout_retry_count
            while True:
                try:
                    return sample_fn()
                except Exception as exc:
                    message = str(exc).lower()
                    retryable = isinstance(
                        exc,
                        (
                            TimeoutError,
                            ConnectionError,
                            OSError,
                            CosmosNativeBackendUnavailable,
                        ),
                    ) or any(
                        token in message
                        for token in ("temporarily unavailable", "timed out", "resource busy")
                    )
                    if (
                        not retryable
                        or "out of memory" in message
                        or rollout_retry_count >= max_rollout_retries
                    ):
                        raise
                    rollout_retry_count += 1
        if self.action_contract.training_transform_compatible:
            condition_state = None
            if self.action_state_rows:
                condition_states = env_obs.get("states")
                if not torch.is_tensor(condition_states):
                    raise ValueError(
                        "State-conditioned 7D Cosmos profile requires "
                        "env_obs['states'] joint state."
                    )
                condition_state = condition_states[0]
            actions, imagined_video_chunk, replay_tensors, prev_logprobs = (
                sample_with_seed_reuse(lambda: self._sample_joint_policy(
                    service=service,
                    image=image[0],
                    prompt=prompt,
                    domain_name=str(domain_name),
                    condition_state=condition_state,
                    chain_logprob=chain_logprob,
                    vision_seed=vision_seed,
                    action_seed=action_seed,
                ))
            )
        elif training_transform_compatible:
            actions, imagined_video_chunk, replay_tensors, prev_logprobs = (
                sample_with_seed_reuse(lambda: self._sample_training_transform_eef_policy(
                    service=service,
                    image=image[0],
                    prompt=prompt,
                    domain_name=str(domain_name),
                    chain_logprob=chain_logprob,
                    vision_seed=vision_seed,
                    action_seed=action_seed,
                ))
            )
        else:
            if split_modality_seeds:
                raise ValueError(
                    "Explicit vision/action seeds require a training-compatible "
                    "Cosmos sampling path"
                )
            if chain_logprob:
                response, replay_tensors, prev_logprobs = (
                    _predict_policy_with_recorded_action_chain(
                        service,
                        request,
                        replay_objective=replay_objective,
                        sampling_seed=action_seed,
                        num_action_chunks=self.num_action_chunks,
                        raw_action_dim=self.action_dim,
                        max_action_dim=int(self.max_action_dim),
                        chain_sigma=float(_cfg_get(self.cosmos_cfg, "chain_sigma", 0.2)),
                        sigma_min=float(_cfg_get(self.cosmos_cfg, "chain_sigma_min", 1e-4)),
                        timestep_scale=float(_cfg_get(self.cosmos_cfg, "timestep_scale", 1000.0)),
                        transition_mode=ROLLOUT_TRANSITION_NATIVE_UNIPC_SURROGATE,
                        transition_noise_seed=derive_matched_transition_noise_seed(action_seed),
                        action_state_rows=self.action_state_rows,
                        fpo_num_mc_samples=int(_cfg_get(self.cosmos_cfg, "fpo_num_mc_samples", 4)),
                        fpo_time_distribution=str(_cfg_get(self.cosmos_cfg, "fpo_time_distribution", "waver")),
                        fpo_training_shift=float(_cfg_get(self.cosmos_cfg, "fpo_training_shift", 10.0)),
                        logprob_mode=str(
                            _cfg_get(self.cosmos_cfg, "chain_logprob_mode", "joint_sum")
                        ),
                    )
                )
            else:
                response = service.predict_policy(request)
            if not training_transform_compatible:
                actions = _actions_from_native_response(
                    response=response,
                    action_dim=self.action_dim,
                    num_action_chunks=self.num_action_chunks,
                    device=image.device,
                )
                imagined_video_chunk = _video_from_native_response(
                    response=response,
                    num_action_chunks=self.num_action_chunks,
                    device=image.device,
                )

        actor_condition_image = image.detach().clone()
        actor_replay_size_cfg = _cfg_get(
            self.cosmos_cfg, "actor_replay_size", None
        )
        if chain_logprob and actor_replay_size_cfg is not None:
            if len(actor_replay_size_cfg) != 2:
                raise ValueError(
                    "cosmos.actor_replay_size must be [height, width], got "
                    f"{actor_replay_size_cfg}."
                )
            target_replay_size = tuple(
                int(value) for value in actor_replay_size_cfg
            )
            if (
                imagined_video_chunk.ndim == 5
                and imagined_video_chunk.shape[-1] in (1, 3, 4)
            ):
                # The native request transform may crop the condition before
                # sampling, so the recorded latent grid follows the generated
                # video rather than the pre-transform condition tensor.
                source_replay_size = tuple(
                    int(value) for value in imagined_video_chunk.shape[2:4]
                )
            else:
                source_chw = _image_tensor_to_chw_uint8(image[0])
                source_replay_size = tuple(
                    int(value) for value in source_chw.shape[-2:]
                )
            if target_replay_size != source_replay_size:
                assert replay_tensors is not None
                replay_tensors = dict(replay_tensors)
                replay_latent_key = (
                    "fpo_clean_joint_latent"
                    if replay_objective
                    == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD
                    else "native_full_chains"
                )
                replay_latent = replay_tensors[replay_latent_key]
                if replay_latent_key == "fpo_clean_joint_latent":
                    replay_latent = replay_latent[:, None, :]
                resized_replay_latent = _resize_native_actor_replay_chains(
                    replay_latent,
                    source_image_size=source_replay_size,
                    target_image_size=target_replay_size,
                    num_action_chunks=self.num_action_chunks,
                    max_action_dim=int(self.max_action_dim),
                    action_state_rows=self.action_state_rows,
                    latent_downsample_factor=int(
                        _cfg_get(
                            self.cosmos_cfg,
                            "latent_downsample_factor",
                            16,
                        )
                    ),
                )
                if replay_latent_key == "fpo_clean_joint_latent":
                    resized_replay_latent = resized_replay_latent[:, 0, :]
                    resized_action = extract_fpo_normalized_action(
                        resized_replay_latent,
                        num_action_chunks=self.num_action_chunks,
                        raw_action_dim=self.action_dim,
                        max_action_dim=int(self.max_action_dim),
                        action_state_rows=self.action_state_rows,
                    )
                    if not torch.equal(
                        resized_action,
                        replay_tensors["fpo_clean_action_normalized"],
                    ):
                        raise RuntimeError(
                            "FPO actor replay resize changed the action suffix."
                        )
                replay_tensors[replay_latent_key] = (
                    resized_replay_latent.contiguous()
                )
            condition_chw = _image_tensor_to_chw_uint8(actor_condition_image[0])
            if tuple(condition_chw.shape[-2:]) != target_replay_size:
                actor_condition_image = _resize_native_actor_condition_image(
                    actor_condition_image, target_replay_size
                )

        forward_inputs = {
            "action": actions.reshape(actions.shape[0], -1),
            "model_action": actions.reshape(actions.shape[0], -1),
            "imagined_video_chunk": imagined_video_chunk,
            # Save the effective seed so artifacts can verify that group
            # members received distinct initial action latent noise.
            "action_sampling_seed": torch.full(
                (actions.shape[0], 1),
                int(action_seed),
                dtype=torch.int64,
                device=actions.device,
            ),
            "vision_sampling_seed": torch.full(
                (actions.shape[0], 1),
                int(vision_seed),
                dtype=torch.int64,
                device=actions.device,
            ),
            "split_modality_seeds": torch.full(
                (actions.shape[0], 1),
                split_modality_seeds,
                dtype=torch.bool,
                device=actions.device,
            ),
            "rollout_retry_count": torch.full(
                (actions.shape[0], 1),
                int(rollout_retry_count),
                dtype=torch.int64,
                device=actions.device,
            ),
        }
        result = {"forward_inputs": forward_inputs}
        if chain_logprob:
            assert replay_tensors is not None
            assert prev_logprobs is not None
            states = env_obs.get("states")
            if torch.is_tensor(states):
                condition_states = states.detach().clone().to(device=image.device)
            else:
                condition_states = torch.zeros(
                    actions.shape[0],
                    1,
                    dtype=actions.dtype,
                    device=image.device,
                )
            forward_inputs.update(
                {
                    **replay_tensors,
                    "action_condition_main_images": actor_condition_image,
                    "action_condition_states": condition_states.contiguous(),
                    "action_transition_mode_id": torch.full(
                        (actions.shape[0], 1),
                        int(
                            transition_mode
                            == ROLLOUT_TRANSITION_MATCHED_ACTION_EULER_GAUSSIAN
                        ),
                        dtype=torch.int64,
                        device=actions.device,
                    ),
                    "matched_transition_noise_seed": torch.full(
                        (actions.shape[0], 1),
                        int(
                            derive_matched_transition_noise_seed(action_seed)
                            if transition_mode
                            == ROLLOUT_TRANSITION_MATCHED_ACTION_EULER_GAUSSIAN
                            else -1
                        ),
                        dtype=torch.int64,
                        device=actions.device,
                    ),
                }
            )
            validate_cosmos_forward_inputs(
                forward_inputs,
                replay_objective=replay_objective,
            )
            result["prev_logprobs"] = prev_logprobs.to(device=image.device)
        return actions, result

    def set_global_step(self, global_step: int) -> None:
        self._global_step = int(global_step)
        self._rollout_call_index = 0

    @torch.no_grad()
    def predict_action_batch(self, env_obs, **kwargs):
        if not self.probe.ready:
            reason = self.probe.blocker_reason()
            raise CosmosNativeBackendUnavailable(
                f"Native Cosmos3 inference is unavailable: {reason}"
            )

        if not isinstance(env_obs, dict):
            raise ValueError("Native Cosmos3 inference expects env_obs to be a dict.")
        image = env_obs.get("main_images")
        if image is None:
            raise ValueError("Native Cosmos3 inference requires env_obs['main_images'].")
        if not torch.is_tensor(image):
            raise ValueError("Native Cosmos3 env_obs['main_images'] must be a tensor.")

        actions_list = []
        result_list = []
        batch_size = int(image.shape[0])
        replay_objective = self._replay_objective()
        if (
            bool(_cfg_get(self.cosmos_cfg, "chain_logprob", False))
            and not self._logged_chain_logprob_semantics
        ):
            if replay_objective == COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD:
                get_logger().warning(
                    "Cosmos replay semantics=fpo_action_head: rollout remains "
                    "native UniPC; PPO scores fixed joint-context MC probes "
                    "using action-head velocity loss, not a policy likelihood."
                )
            self._logged_chain_logprob_semantics = True
        denoise_start = time.monotonic()
        for sample_idx in range(batch_size):
            actions_i, result_i = self._predict_action_single(
                _slice_env_obs(env_obs, sample_idx),
                request_id=sample_idx + 1,
            )
            actions_list.append(actions_i)
            result_list.append(result_i)

        denoise_elapsed = time.monotonic() - denoise_start
        self._cosmos_denoise_batches = getattr(self, "_cosmos_denoise_batches", 0) + 1
        log_interval = int(os.environ.get("RLINF_DENOISE_LOG_INTERVAL", "1"))
        if log_interval > 0 and self._cosmos_denoise_batches % log_interval == 0:
            try:
                avg_request = denoise_elapsed / max(batch_size, 1)
                get_logger().info(
                    "[Denoise][cosmos] batch %s | requests=%s | action_steps=%s | "
                    "denoise_steps=%s | elapsed %.1fs | avg %.1fs/request",
                    self._cosmos_denoise_batches,
                    batch_size,
                    self.num_action_chunks,
                    int(_cfg_get(self.cosmos_cfg, "num_steps", 2)),
                    denoise_elapsed,
                    avg_request,
                )
            except Exception:
                pass

        actions = torch.cat(actions_list, dim=0).contiguous()
        forward_inputs = _merge_tensor_dicts(
            [result["forward_inputs"] for result in result_list]
        )
        result = {"forward_inputs": forward_inputs}
        if all("prev_logprobs" in result_i for result_i in result_list):
            result["prev_logprobs"] = torch.cat(
                [result_i["prev_logprobs"] for result_i in result_list], dim=0
            ).contiguous()
            validate_cosmos_forward_inputs(
                forward_inputs,
                replay_objective=replay_objective,
            )
        return actions, result

    def ensure_native_trainable_state(self) -> nn.Module:
        service = self._get_service()
        service_model = getattr(service, "model", None)
        if not isinstance(service_model, nn.Module):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos service model does not expose torch parameters."
            )
        current_model = getattr(self, "_native_service_model", None)
        if current_model is not service_model or self.native_trainable_proxy is None:
            object.__setattr__(self, "_native_service_model", service_model)
            self.configure_native_trainable_parameters(service_model)
        return service_model

    def release_rollout_resources(self) -> bool:
        """Evict the unregistered native service model between RL phases."""
        service_model = getattr(self, "_native_service_model", None)
        service = self._service
        if service_model is None:
            if service is None:
                return False
            candidate = getattr(service, "model", None)
            if not isinstance(candidate, nn.Module):
                return False
            service_model = self.ensure_native_trainable_state()

        if self.native_trainable_proxy is None:
            self.configure_native_trainable_parameters(service_model)

        # ActionModelService aliases the FSDP2 model both directly and through
        # its inference pipe. Clear both aliases so the frozen CUDA backbone is
        # not retained after rollout; the registered action proxy stays alive.
        if service is not None:
            if getattr(service, "model", None) is service_model:
                service.model = None
            pipe = getattr(service, "pipe", None)
            if pipe is not None and getattr(pipe, "model", None) is service_model:
                pipe.model = None

        self._service = None
        object.__setattr__(self, "_native_service_model", None)
        self._native_rollout_service_evicted = True
        return True

    def restore_rollout_resources(self) -> bool:
        """Rebuild an evicted service and restore the latest action weights."""
        if not self._native_rollout_service_evicted:
            return False

        proxy = self.native_trainable_proxy
        if not isinstance(proxy, NativeTrainableParameterProxy):
            raise CosmosNativeBackendUnavailable(
                "Cannot restore an evicted Cosmos service without its action proxy."
            )
        named_params = proxy.named_service_parameters()
        names = [name for name, _ in named_params]
        action_state: dict[str, Any] = {
            NATIVE_TRAINABLE_STATE_METADATA_KEY: self._native_trainable_metadata(
                names=names
            )
        }
        action_state.update(
            {
                name: (
                    param.detach().full_tensor().cpu().clone()
                    if isinstance(param, DTensor)
                    else param.detach().cpu().clone()
                )
                for name, param in named_params
            }
        )

        self._get_service()
        self.ensure_native_trainable_state()
        self.load_native_trainable_action_state_dict(action_state, strict=True)
        self._native_rollout_service_evicted = False
        return True

    def configure_native_trainable_parameters(
        self, service_model: nn.Module | None = None
    ) -> list[nn.Parameter]:
        if service_model is None:
            service_model = getattr(self, "_native_service_model", None)
        if service_model is None:
            service_model = getattr(self._get_service(), "model", None)
        if not isinstance(service_model, nn.Module):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos service model does not expose torch parameters."
            )

        selected_named_params = []
        blocked_matches = []
        for name, param in service_model.named_parameters():
            requested = _matches_any_parameter_pattern(
                name, self.trainable_param_patterns
            )
            blocked = _matches_any_parameter_pattern(
                name, self.blocked_trainable_param_patterns
            )
            should_train = requested and not blocked
            param.requires_grad_(should_train)
            if should_train:
                selected_named_params.append((name, param))
            elif requested and blocked:
                blocked_matches.append(name)

        if not selected_named_params:
            patterns = ", ".join(self.trainable_param_patterns)
            blocked_detail = ""
            if blocked_matches:
                blocked_detail = (
                    " Matched parameters were blocked by "
                    f"cosmos.blocked_trainable_param_patterns: {blocked_matches}."
                )
            raise ValueError(
                "No native Cosmos action parameters matched "
                f"cosmos.trainable_param_patterns={patterns!r}." + blocked_detail
            )

        self.native_trainable_proxy = NativeTrainableParameterProxy(
            selected_named_params
        )
        return [param for _, param in selected_named_params]

    def _native_trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        self.ensure_native_trainable_state()
        proxy = self.native_trainable_proxy
        if not isinstance(proxy, NativeTrainableParameterProxy):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos trainable parameter proxy is not initialized."
            )
        return proxy.named_service_parameters()

    def validate_native_trainable_bindings(
        self, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        """Ensure service, proxy, and optimizer still share canonical parameters."""

        service_model = self.ensure_native_trainable_state()
        proxy = self.native_trainable_proxy
        if not isinstance(proxy, NativeTrainableParameterProxy):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos trainable parameter proxy is not initialized."
            )

        service_params = dict(service_model.named_parameters())
        proxy_params = proxy.named_service_parameters()
        missing = [name for name, _ in proxy_params if name not in service_params]
        replaced = [
            name
            for name, param in proxy_params
            if name in service_params and param is not service_params[name]
        ]
        if missing or replaced:
            raise RuntimeError(
                "Native Cosmos trainable parameter binding changed: "
                f"missing_service_params={missing}, replaced_proxy_params={replaced}. "
                "A state-dict or device-migration operation replaced a canonical "
                "service parameter."
            )

        if optimizer is None:
            return
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        detached = [
            name for name, param in proxy_params if id(param) not in optimizer_param_ids
        ]
        if detached:
            raise RuntimeError(
                "Native Cosmos optimizer is detached from live service parameters: "
                f"{detached}."
            )

    def prepare_native_cpu_offload_training_storage(
        self, optimizer: torch.optim.Optimizer | None = None
    ) -> dict[str, Any]:
        """Materialize action shards on CPU before the first FSDP2 replay.

        Cosmos internally applies FSDP2 with ``CPUOffloadPolicy`` while RLinf
        registers the selected action parameters through an outer proxy. A
        full-tensor rollout weight materialization can leave those proxy
        parameters on CUDA before FSDP2 has run its first lazy initialization.
        FSDP2 then rejects the first replay because CPU-offloaded sharded
        parameters must start on CPU.

        Moving the proxy is safe only before the owning FSDP2 groups complete
        lazy initialization: their first ``reset_sharded_param()`` call will
        adopt and pin the new CPU local storage. Moving it after that point
        would detach FSDP2's cached sharded storage from AdamW, so fail loudly
        instead of attempting a late repair.
        """

        self.validate_native_trainable_bindings(optimizer)
        if not _cfg_bool(self.cosmos_cfg, "fsdp_cpu_offload", False):
            return {"enabled": False, "moved_parameter_count": 0}

        service_model = self.ensure_native_trainable_state()
        proxy = self.native_trainable_proxy
        if not isinstance(proxy, NativeTrainableParameterProxy):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos action proxy is unavailable for CPU-offload preparation."
            )

        named_params = proxy.named_service_parameters()
        non_cpu = [
            (name, str(param.device))
            for name, param in named_params
            if param.device.type != "cpu"
        ]
        if non_cpu:
            initialized_groups: list[str] = []
            for module_name, module in service_model.named_modules():
                get_state = getattr(module, "_get_fsdp_state", None)
                if not callable(get_state):
                    continue
                state = get_state()
                param_group = getattr(state, "_fsdp_param_group", None)
                if param_group is not None and bool(
                    getattr(param_group, "_reset_sharded_params", False)
                ):
                    initialized_groups.append(module_name or "<root>")
            if initialized_groups:
                raise RuntimeError(
                    "Native Cosmos action shards left CPU after FSDP2 lazy init; "
                    "a proxy-only device migration would detach optimizer storage. "
                    f"non_cpu={non_cpu}, initialized_groups={initialized_groups[:8]}"
                )

            # ParameterList._apply preserves Parameter identity in the pinned
            # PyTorch build. The checks below make that an enforced contract.
            proxy.to(device="cpu")
            self.validate_native_trainable_bindings(optimizer)

        remaining = [
            (name, str(param.device))
            for name, param in proxy.named_service_parameters()
            if param.device.type != "cpu"
        ]
        if remaining:
            raise RuntimeError(
                "Native Cosmos CPU-offload preparation left action shards off CPU: "
                f"{remaining}"
            )
        return {
            "enabled": True,
            "moved_parameter_count": len(non_cpu),
            "moved_parameters": [name for name, _ in non_cpu],
        }

    @staticmethod
    def _materialize_native_parameter(
        param: nn.Parameter, *, cpu_offload: bool
    ) -> torch.Tensor:
        tensor = (
            param.detach().full_tensor()
            if isinstance(param, DTensor)
            else param.detach()
        )
        if cpu_offload:
            tensor = tensor.cpu()
        return tensor.clone()

    def native_rollout_state_dict(
        self, *, cpu_offload: bool = False
    ) -> dict[str, torch.Tensor]:
        """Build a root-compatible full state dict without DCP proxy mutation.

        The generic distributed-checkpoint materializer may replace registered
        parameters while building a full state dict. Native Cosmos exposes its
        internally-sharded service parameters through a proxy, so replacement
        detaches AdamW from the parameters used by replay. Materialize the
        canonical parameters directly and preserve the ordinary root keys
        expected by rollout workers.
        """

        self.validate_native_trainable_bindings()
        proxy = self.native_trainable_proxy
        assert isinstance(proxy, NativeTrainableParameterProxy)
        return {
            f"native_trainable_proxy.params.{index}": (
                self._materialize_native_parameter(
                    param, cpu_offload=cpu_offload
                )
            )
            for index, param in enumerate(proxy.params)
        }

    def _native_trainable_metadata(
        self, *, names: list[str] | None = None
    ) -> dict[str, Any]:
        if names is None:
            names = [name for name, _ in self._native_trainable_named_parameters()]
        return {
            "training_mode": _cfg_get(
                self.cosmos_cfg, "training_mode", "action_only_grpo"
            ),
            "trainable_param_patterns": list(self.trainable_param_patterns),
            "blocked_trainable_param_patterns": list(
                self.blocked_trainable_param_patterns
            ),
            "matched_param_names": names,
            "action_dim": self.action_dim,
            "num_action_chunks": self.num_action_chunks,
            "raw_action_dim": int(
                _cfg_get(self.cosmos_cfg, "raw_action_dim", self.action_dim)
            ),
            "action_chunk_size": int(
                _cfg_get(self.cosmos_cfg, "action_chunk_size", self.num_action_chunks)
            ),
            "action_representation": str(self.action_representation),
        }

    def native_trainable_action_state_dict(self) -> dict[str, Any]:
        """Build a CPU state dict of the trainable action parameters.

        When the backbone is internally FSDP2-sharded (native strategy),
        trainable parameters are DTensor-backed; ``.full_tensor()`` gathers
        each parameter's shards into a full local tensor before moving it to
        CPU. That gather is a collective, so this method must be called on
        every rank of the parameter's mesh, not just rank 0.
        """
        state_dict: dict[str, Any] = {
            NATIVE_TRAINABLE_STATE_METADATA_KEY: self._native_trainable_metadata()
        }
        state_dict.update(
            {
                name: self._materialize_native_parameter(
                    param, cpu_offload=True
                )
                for name, param in self._native_trainable_named_parameters()
            }
        )
        return state_dict

    def load_native_trainable_action_state_dict(
        self, state_dict: dict[str, Any], *, strict: bool = True
    ) -> None:
        trainable = dict(self._native_trainable_named_parameters())
        metadata = state_dict.get(NATIVE_TRAINABLE_STATE_METADATA_KEY)
        tensor_state = {
            name: tensor
            for name, tensor in state_dict.items()
            if name != NATIVE_TRAINABLE_STATE_METADATA_KEY
        }
        expected_metadata = self._native_trainable_metadata()
        if strict:
            if metadata != expected_metadata:
                raise KeyError(
                    "Native Cosmos action state metadata mismatch: "
                    f"expected={expected_metadata}, found={metadata}."
                )
        missing = sorted(set(trainable) - set(tensor_state))
        unexpected = sorted(set(tensor_state) - set(trainable))
        if strict and (missing or unexpected):
            raise KeyError(
                "Native Cosmos action state mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        with torch.no_grad():
            for name, tensor in tensor_state.items():
                if name not in trainable:
                    continue
                dst = trainable[name]
                if isinstance(dst, DTensor):
                    # A DTensor destination rejects a plain-tensor source
                    # (`aten.copy_.default got mixed torch.Tensor and
                    # DTensor`); re-shard the incoming full tensor to match
                    # dst's mesh/placements first. The full action state was
                    # materialized on every rank before rebuilding the native
                    # service, so shard locally instead of running a redundant
                    # rank-0 scatter. Besides avoiding extra communication,
                    # this keeps restore independent from other NCCL groups
                    # created by the collocated rollout/env workers.
                    src = distribute_tensor(
                        tensor.to(dtype=dst.dtype),
                        dst.device_mesh,
                        dst.placements,
                        src_data_rank=None,
                    )
                else:
                    src = tensor.to(device=dst.device, dtype=dst.dtype)
                dst.copy_(src)

    def save_native_trainable_action_state(self, path: str | Path) -> None:
        torch.save(self.native_trainable_action_state_dict(), _expand_path(path))

    def load_native_trainable_action_state(
        self, path: str | Path, *, strict: bool = True
    ) -> None:
        state_dict = torch.load(
            _expand_path(path), map_location="cpu", weights_only=False
        )
        self.load_native_trainable_action_state_dict(state_dict, strict=strict)

    def _get_service(self):
        if self._service is not None:
            return self._service

        # Also cover a lazily constructed policy whose distributed rendezvous
        # environment became available after __init__.
        _ensure_cosmos_dcp_hybrid_process_group()

        try:
            _clear_cosmos_lazy_config_resolvers()
            _patch_typing_override_for_py311()
            configured_vlm_processor = _expand_path(
                _cfg_get(self.cosmos_cfg, "vlm_processor_path", None)
            )
            if configured_vlm_processor is not None:
                # Ray actors consume the strict config path, not a missing login env var.
                os.environ["COSMOS3_VLM_PROCESSOR_PATH"] = str(configured_vlm_processor)
            if self.action_contract.requires_joint_domain_patch:
                _patch_cosmos_ur5_joint_domain()
            _patch_cosmos_checkpoint_tokenizer_factory()
            _patch_cosmos_single_rank_sync_model_states()
            _patch_torch_dcp_single_rank_load()
            activation_checkpointing_mode = _cfg_get(
                self.cosmos_cfg, "activation_checkpointing", "full"
            )
            if activation_checkpointing_mode == "none":
                activation_checkpointing_mode = None
            _patch_cosmos_cfg_branch_checkpointing(
                enabled=_cfg_bool(
                    self.cosmos_cfg, "cfg_branch_checkpointing", False
                )
            )
            _patch_cosmos_pre_fsdp_hooks(
                apply_trainable_freeze=_cfg_bool(
                    self.cosmos_cfg, "internal_fsdp_shard", False
                ),
                trainable_param_patterns=self.trainable_param_patterns,
                blocked_trainable_param_patterns=self.blocked_trainable_param_patterns,
                activation_checkpointing_mode=activation_checkpointing_mode,
                cpu_offload=_cfg_bool(
                    self.cosmos_cfg, "fsdp_cpu_offload", False
                ),
            )
            libero_server = importlib.import_module(
                "cosmos_framework.scripts.action_policy_server_libero"
            )
            _quiet_third_party_training_logs()
            _patch_action_server_guardrail_args(libero_server)
            common_args = importlib.import_module(
                "cosmos_framework.inference.common.args"
            )
        except ModuleNotFoundError as exc:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: "
                f"missing native dependency {exc.name}"
            ) from exc

        checkpoint_path = self.probe.checkpoint_path or self.probe.checkpoint_name
        if not checkpoint_path:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: no checkpoint configured"
            )

        guardrails_enabled = _cfg_bool(self.cosmos_cfg, "guardrails", True)
        if guardrails_enabled and not _guardrail_setup_authorized(self.cosmos_cfg):
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: HF credentials or "
                "cosmos.allow_guardrail_download=true is required before "
                "Cosmos guardrail checkpoint setup."
            )

        output_dir = _resolve_native_output_dir(self.cosmos_cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_kwargs = {
            "checkpoint_path": checkpoint_path,
            "vlm_processor_from_checkpoint": bool(
                _cfg_get(self.cosmos_cfg, "vlm_processor_from_checkpoint", True)
            ),
        }
        checkpoint_config_path = _cfg_get(
            self.cosmos_cfg, "checkpoint_config_path", None
        ) or _cfg_get(self.cosmos_cfg, "config_file", None)
        if checkpoint_config_path:
            config_path = _expand_path(checkpoint_config_path)
            assert config_path is not None
            config_path = _native_checkpoint_config_compat_path(
                config_path,
                output_dir,
                _cfg_get(self.cosmos_cfg, "framework_path", None),
            )
            checkpoint_kwargs["config_file"] = str(config_path)
        checkpoint = common_args.CheckpointOverrides(**checkpoint_kwargs)
        max_action_dim = int(self.max_action_dim)
        action_stats_path = _expand_path(
            _cfg_get(self.cosmos_cfg, "action_stats_path", None)
        )
        requested_normalization = str(
            _cfg_get(self.cosmos_cfg, "action_normalization", "auto")
        )
        service_normalization = _resolve_action_normalization_for_service(
            requested_normalization, action_stats_path
        )
        action_server_kwargs = dict(
            checkpoint=checkpoint,
            output_dir=output_dir,
            sampler=str(_cfg_get(self.cosmos_cfg, "sampler", "unipc")),
            seed=int(_cfg_get(self.cosmos_cfg, "seed", 0)),
            guidance=float(_cfg_get(self.cosmos_cfg, "guidance", 1.0)),
            num_steps=int(_cfg_get(self.cosmos_cfg, "num_steps", 2)),
            fps=int(_cfg_get(self.cosmos_cfg, "fps", 5)),
            action_chunk_size=self.num_action_chunks,
            max_action_dim=int(max_action_dim),
            raw_action_dim=self.action_dim,
            dump_dir=None,
            action_stats_path=action_stats_path,
            action_normalization=service_normalization,
            guardrails=guardrails_enabled,
            offload_guardrail_models=_cfg_bool(
                self.cosmos_cfg, "offload_guardrail_models", False
            ),
            internal_fsdp_shard=_cfg_bool(
                self.cosmos_cfg, "internal_fsdp_shard", False
            ),
            internal_fsdp_node_local=_cfg_bool(
                self.cosmos_cfg, "internal_fsdp_node_local", False
            ),
            # The native service defaults to inference-oriented torch.compile.
            # Current-weight chain replay must build a backward graph.
            use_torch_compile=_cfg_bool(
                self.cosmos_cfg, "use_torch_compile", False
            ),
            host="127.0.0.1",
            port=0,
            run_validation=False,
        )
        supported_server_fields = getattr(
            libero_server.ActionServerArgs, "model_fields", None
        )
        if supported_server_fields is not None:
            action_server_kwargs = {
                key: value
                for key, value in action_server_kwargs.items()
                if key in supported_server_fields
            }
        args = libero_server.ActionServerArgs(**action_server_kwargs)
        try:
            self._service = libero_server.ActionModelService(args)
            _write_formal_hsdp_mesh_evidence(self._service)
            # ActionModelService construction runs cosmos_framework's own
            # process-wide setup, which permanently disables autograd via
            # a bare torch.set_grad_enabled(False) (not a context manager)
            # since this "action server" is built for inference and always
            # sees training=False. That leaves gradients disabled for the
            # rest of this process's lifetime, silently zeroing grad_fn on
            # every later computation (including RLinf's own loss/backward
            # pipeline) unless something re-enables it. Rollout's own
            # inference entry point (predict_action_batch) is independently
            # @torch.no_grad()-decorated, so restoring the global default
            # here is safe even for a process that also does pure inference.
            torch.set_grad_enabled(True)
            _patch_action_service_action_processing_record(self._service)
        except ModuleNotFoundError as exc:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: "
                f"missing native dependency {exc.name}"
            ) from exc
        except FileNotFoundError as exc:
            missing = exc.filename or str(exc)
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: "
                f"missing native executable {missing}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise CosmosNativeBackendUnavailable(
                "Native Cosmos3 inference is unavailable: "
                f"{_called_process_error_message(exc)}"
            ) from exc
        return self._service


def _resolve_native_output_dir(cosmos_cfg: Any) -> Path:
    configured = _cfg_get(cosmos_cfg, "output_dir", None)
    if configured:
        path = _expand_path(configured)
        assert path is not None
        return path
    return Path.cwd() / "logs" / "cosmos_native_inference_smoke_output"


def _called_process_output_tail(value: Any, *, max_lines: int = 40) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _called_process_error_message(exc: subprocess.CalledProcessError) -> str:
    parts = [f"native setup command failed with exit code {exc.returncode}"]
    if exc.cmd:
        parts.append(f"command={exc.cmd!r}")
    stdout_tail = _called_process_output_tail(getattr(exc, "stdout", None))
    stderr_tail = _called_process_output_tail(getattr(exc, "stderr", None))
    if stdout_tail:
        parts.append(f"stdout tail:\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"stderr tail:\n{stderr_tail}")
    return "; ".join(parts)


def _guardrail_setup_authorized(cosmos_cfg: Any) -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    allow_download = _cfg_get(cosmos_cfg, "allow_guardrail_download", False)
    if isinstance(allow_download, str):
        return allow_download.lower() in {"1", "true", "yes", "on"}
    return bool(allow_download)


def _image_tensor_to_png_request(image: torch.Tensor) -> tuple[str, int]:
    from PIL import Image

    image_chw = _image_tensor_to_chw_uint8(image)
    image_hwc = image_chw.permute(1, 2, 0)
    array = image_hwc.contiguous().numpy()
    pil_image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), int(array.shape[0])


def _actions_from_native_response(
    *,
    response: dict[str, Any],
    action_dim: int,
    num_action_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    if "action" not in response:
        raise CosmosNativeBackendUnavailable(
            "Native Cosmos3 inference response did not include an action chunk."
        )
    actions = torch.as_tensor(response["action"], dtype=torch.float32, device=device)
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    if actions.shape[:2] != (1, num_action_chunks):
        raise CosmosNativeBackendUnavailable(
            "Native Cosmos3 inference returned action shape "
            f"{tuple(actions.shape)}, expected (1, {num_action_chunks}, {action_dim})."
        )
    if int(actions.shape[-1]) != action_dim:
        raise NotImplementedError(
            "Native Cosmos3 inference returned explicit raw action dimension "
            f"{int(actions.shape[-1])}, but Ctrl-World action_dim={action_dim}. "
            "Native inference does not silently convert action representations."
        )
    if not torch.isfinite(actions).all():
        raise CosmosNativeBackendUnavailable(
            "Native Cosmos3 inference returned non-finite action values."
        )
    return actions


def _video_from_native_response(
    *,
    response: dict[str, Any],
    num_action_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    video = response.get("video")
    if not video:
        raise CosmosNativeBackendUnavailable(
            "Native Cosmos3 inference response did not include rollout video frames."
        )
    if len(video) < num_action_chunks:
        raise CosmosNativeBackendUnavailable(
            "Native Cosmos3 inference returned fewer video frames than the action chunk: "
            f"{len(video)} < {num_action_chunks}."
        )

    frames = []
    for encoded_frame in video[:num_action_chunks]:
        with Image.open(io.BytesIO(base64.b64decode(encoded_frame))) as frame:
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
        frames.append(torch.from_numpy(array))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device)
