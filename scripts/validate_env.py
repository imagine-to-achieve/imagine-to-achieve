"""Validate the aarch64 runtime along the native Cosmos actor import path."""

from __future__ import annotations

import importlib
import platform

import ray
import torch
import transformer_engine
import transformer_engine_torch


print("core imports passed", flush=True)
backend = importlib.import_module("rlinf.models.embodiment.cosmos.cosmos_backend")
print("RLinf Cosmos backend import passed", flush=True)

# Mirror CosmosNativeInferencePolicy._get_service() through the dependency-
# loading and compatibility-patch stage, stopping before checkpoint/model load.
backend._clear_cosmos_lazy_config_resolvers()
backend._patch_typing_override_for_py311()
backend._patch_cosmos_checkpoint_tokenizer_factory()
print("Cosmos VLM config import passed", flush=True)
backend._patch_cosmos_single_rank_sync_model_states()
backend._patch_torch_dcp_single_rank_load()
backend._patch_cosmos_cfg_branch_checkpointing(enabled=False)
backend._patch_cosmos_pre_fsdp_hooks(
    apply_trainable_freeze=False,
    activation_checkpointing_mode=None,
    cpu_offload=False,
)
libero_server = importlib.import_module(
    "cosmos_framework.scripts.action_policy_server_libero"
)
backend._patch_action_server_guardrail_args(libero_server)
importlib.import_module("cosmos_framework.inference.common.args")
print("native Cosmos service dependency path passed", flush=True)
robolab = importlib.import_module(
    "cosmos_framework.scripts.action_policy_server_robolab"
)
robolab.ActionTransformPipeline(
    tokenizer_config=None,
    cfg_dropout_rate=0.0,
    max_action_dim=64,
    append_viewpoint_info=True,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    append_idle_frames=False,
    format_prompt_as_json=False,
)
print("RoboLab rollout transform dependency path passed", flush=True)

assert platform.machine() == "aarch64"
assert torch.version.cuda == "13.0"
print(
    "environment validation passed",
    torch.__version__,
    ray.__version__,
    transformer_engine.__version__,
    transformer_engine_torch.__file__,
)
