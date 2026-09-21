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

import dataclasses
import importlib.util
import json
import logging
import math
import os
from dataclasses import asdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.algorithms.cross_rank_grpo import (
    validate_cross_rank_group_configuration,
)
from rlinf.envs import SupportedEnvType

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

logging.getLogger().setLevel(logging.INFO)


def _make_cluster(*args, **kwargs):
    from rlinf.scheduler.cluster import Cluster

    return Cluster(*args, **kwargs)


def _get_placement_helpers():
    from rlinf.utils.placement import (
        HybridComponentPlacement,
        ModelParallelComponentPlacement,
        PlacementMode,
    )

    return HybridComponentPlacement, ModelParallelComponentPlacement, PlacementMode


class SupportedModel(Enum):
    # Reasoning models
    QWEN2_5 = ("qwen2.5", "reasoning")
    QWEN2_5_VL = ("qwen2.5_vl", "reasoning")
    QWEN3 = ("qwen3", "reasoning")
    QWEN3_MOE = ("qwen3_moe", "reasoning")

    # Embodied models
    OPENVLA = ("openvla", "embodied")
    OPENVLA_OFT = ("openvla_oft", "embodied")
    OPENPI = ("openpi", "embodied")
    COSMOS = ("cosmos", "embodied")
    MLP_POLICY = ("mlp_policy", "embodied")
    GR00T = ("gr00t", "embodied")
    DEXBOTIC_PI = ("dexbotic_pi", "embodied")
    CNN_POLICY = ("cnn_policy", "embodied")
    FLOW_POLICY = ("flow_policy", "embodied")
    CMA_POLICY = ("cma", "embodied")
    LINGBOTVLA = ("lingbotvla", "embodied")

    # Sft models
    QWEN2_5_VL_SFT = ("qwen2.5_vl", "sft")
    QWEN3_VL_SFT = ("qwen3_vl", "sft")
    QWEN3_VL_MOE_SFT = ("qwen3_vl_moe", "sft")

    def __new__(cls, value, category):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.category = category
        return obj


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


SUPPORTED_ROLLOUT_BACKENDS = ["sglang", "vllm"]
SUPPORTED_TASK_TYPE = [
    "embodied",
    "reasoning",
    "reasoning_eval",
    "coding_online_rl",
    "sft",
]
SUPPORTED_TRAINING_BACKENDS = ["megatron", "fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None, "null"]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


@torch.jit.script
def gelu_impl(x):
    """
    OpenAI's gelu implementation.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except Exception:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def activation_to_func(
    activation: str, openai_gelu: bool = False, onnx_safe: bool = False
) -> Callable:
    """
    Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.

    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        "gelu",
        "geglu",
        "reglu",
        "swiglu",
        "squared-relu",
        "fast-geglu",
        "fast-swiglu",
        "fast-reglu",
        "approx-gelu",
    ]

    if activation not in supported_activations:
        raise ValueError(
            f"Unsupported activation {activation}. Supported activations: {supported_activations} "
        )

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == "squared-relu":
        activation_func = squared_relu

    return activation_func


def validate_rollout_cfg(cfg, algorithm_cfg):
    assert get_supported_model(cfg.model.model_type)

    def validate_sglang_cfg(cfg):
        assert cfg is not None, (
            "sglang config must be specified if rollout_backend is sglang."
        )
        cfg.attention_backend = cfg.get("attention_backend", "triton")
        cfg.decode_log_interval = cfg.get("decode_log_interval", 500000)
        cfg.use_torch_compile = cfg.get("use_torch_compile", False)
        cfg.torch_compile_max_bs = cfg.get("torch_compile_max_bs", 128)
        return cfg

    def validate_vllm_cfg(cfg):
        assert cfg is not None, (
            "vllm config must be specified if rollout_backend is vllm."
        )
        cfg.attention_backend = cfg.get("attention_backend", "FLASH_ATTN")
        cfg.enable_chunked_prefill = cfg.get("enable_chunked_prefill", True)
        cfg.enable_prefix_caching = cfg.get("enable_prefix_caching", True)
        cfg.enable_flash_infer_sampler = cfg.get("enable_flash_infer_sampler", True)
        cfg.max_num_batched_tokens = cfg.get("max_num_batched_tokens", None)
        cfg.torch_profiler_dir = cfg.get("torch_profiler_dir", None)
        return cfg

    with open_dict(cfg):
        cfg.gpu_memory_utilization = cfg.get("gpu_memory_utilization", 0.65)
        assert cfg.model.model_path is not None, (
            "rollout.model.model_path must be specified for rollout."
        )

        cfg.disable_log_stats = cfg.get("disable_log_stats", False)
        cfg.detokenize = cfg.get("detokenize", False)
        cfg.rollout_backend = cfg.get("rollout_backend", "sglang")
        assert cfg.rollout_backend in SUPPORTED_ROLLOUT_BACKENDS, (
            f"rollout_backend must be one of {SUPPORTED_ROLLOUT_BACKENDS}."
        )
        cfg.return_logprobs = cfg.return_logprobs or algorithm_cfg.get(
            "importance_sampling_fix", False
        )
        cfg.sglang = validate_sglang_cfg(cfg.sglang)
        cfg.vllm = validate_vllm_cfg(cfg.vllm)

    return cfg


def validate_model_cfg_by_hf_config(cfg, hf_model_path):
    # validate by hf config
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    if "Qwen2ForCausalLM" in hf_config.architectures:
        qkv_bias = True
    else:
        qkv_bias = getattr(hf_config, "attention_bias", False)

    if (
        "Qwen3ForCausalLM" in hf_config.architectures
        or "Qwen3MoeForCausalLM" in hf_config.architectures
    ):
        qk_layernorm = True
    else:
        qk_layernorm = getattr(cfg.model, "qk_layernorm", False)

    with open_dict(cfg):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict):
            rtype = rs.get("type", "")
            if rtype in {"linear", "dynamic", "ntk", "yarn"}:
                f = rs.get("factor")
                if f is not None:
                    cfg.model.seq_len_interpolation_factor = float(f)
            else:
                # mrope
                cfg.model.seq_len_interpolation_factor = None
        cfg.model.padded_vocab_size = hf_config.vocab_size
        cfg.model.max_position_embeddings = hf_config.max_position_embeddings
        cfg.model.rotary_base = hf_config.rope_theta
        cfg.model.share_embeddings_and_output_weights = getattr(
            hf_config, "tie_word_embeddings", False
        )
        cfg.model.num_layers = hf_config.num_hidden_layers
        cfg.model.hidden_size = hf_config.hidden_size
        cfg.model.num_attention_heads = hf_config.num_attention_heads
        cfg.model.num_query_groups = hf_config.num_key_value_heads
        cfg.model.ffn_hidden_size = hf_config.intermediate_size
        cfg.model.attention_dropout = hf_config.attention_dropout
        cfg.model.hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)
        cfg.model.add_qkv_bias = qkv_bias
        cfg.model.qk_layernorm = qk_layernorm
        cfg.model.layernorm_epsilon = hf_config.rms_norm_eps
        cfg.model.head_dim = getattr(
            hf_config,
            "head_dim",
            cfg.model.hidden_size // cfg.model.num_attention_heads,
        )
        if cfg.model.head_dim is not None:
            cfg.model.kv_channels = cfg.model.head_dim

        # MoE model
        cfg.model.num_moe_experts = getattr(hf_config, "num_experts", None)
        cfg.model.num_experts = getattr(hf_config, "num_experts", None)
        cfg.model.moe_ffn_hidden_size = getattr(
            hf_config, "moe_intermediate_size", None
        )
        cfg.model.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 2)

    return cfg


def validate_fsdp_cfg(cfg: DictConfig) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        """Validate AMP configuration and ensure mutual exclusivity with FSDP mixed_precision."""

        param_dtype = config.mixed_precision.param_dtype
        reduce_dtype = config.mixed_precision.reduce_dtype
        buffer_dtype = config.mixed_precision.buffer_dtype

        all_none = param_dtype is None and reduce_dtype is None and buffer_dtype is None

        all_fp32 = (
            param_dtype == "fp32" and reduce_dtype == "fp32" and buffer_dtype == "fp32"
        )

        use_fsdp_mixed_precision = not (all_none or all_fp32)

        amp_autocast = config.get("amp_autocast", {})
        config.amp_autocast = {
            "enabled": amp_autocast.get("enabled", False),
            "precision": amp_autocast.get("precision", "bf16"),
        }

        grad_scaler = config.get("grad_scaler", {})
        config.grad_scaler = {
            "enabled": grad_scaler.get("enabled", False),
            "init_scale": grad_scaler.get("init_scale", None),
            "growth_interval": grad_scaler.get("growth_interval", None),
        }

        if "amp" in config:
            logging.warning(
                "fsdp_config.amp is no longer supported, use fsdp_config.amp_autocast and fsdp_config.grad_scaler instead"
            )

        if config.amp_autocast.enabled and use_fsdp_mixed_precision:
            assert False, (
                "amp_autocast should not be enabled when fsdp mixed_precision is enabled"
            )
        assert config.amp_autocast.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp_autocast.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )
        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", None
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", None
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", None
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

    return cfg


def validate_megatron_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.mcore_gpt = cfg.get("mcore_gpt", True)
        spec_name = cfg.get("spec_name", "local_gpt")
        cfg.spec_name = spec_name

        # Pad the vocab size to be divisible by this value.
        cfg.model.make_vocab_size_divisible_by = cfg.model.get(
            "make_vocab_size_divisible_by", 8
        )
        cfg.use_torch_fsdp2 = False

        # training args for megatron
        cfg.megatron.load = cfg.model.get("megatron_checkpoint", None)
        use_hf_ckpt = cfg.megatron.get("use_hf_ckpt", False)
        if cfg.megatron.load is None:
            assert use_hf_ckpt, (
                "model.megatron_checkpoint is required if use_hf_ckpt is False"
            )
        else:
            assert not use_hf_ckpt, (
                "model.megatron_checkpoint should be None if use_hf_ckpt is True"
            )
        cfg.megatron.pretrained_checkpoint = cfg.get("pretrained_checkpoint", None)
        cfg.megatron.save = None
        cfg.megatron.micro_batch_size = cfg.get("micro_batch_size", 1)
        cfg.megatron.global_batch_size = cfg.get("global_batch_size", 1)
        cfg.megatron.tp_comm_overlap_cfg = cfg.megatron.get("tp_comm_overlap_cfg", None)
        cfg.megatron.decoder_tp_comm_overlap = cfg.megatron.get(
            "decoder_tp_comm_overlap", False
        )
        cfg.megatron.timing_log_level = cfg.megatron.get(
            "timing_log_level", 0
        )  # choices=range(0,3)
        cfg.megatron.timing_log_option = cfg.megatron.get(
            "timing_log_option", "minmax"
        )  # choices=['max', 'minmax', 'all']

        # Megatron >= 0.12.0
        cfg.megatron.init_model_with_meta_device = cfg.megatron.get(
            "init_model_with_meta_device", False
        )
        cfg.megatron.use_torch_fsdp2 = cfg.megatron.get("use_torch_fsdp2", False)
        cfg.megatron.use_custom_fsdp = cfg.megatron.get("use_custom_fsdp", False)
        cfg.megatron.check_for_large_grads = cfg.megatron.get(
            "check_for_large_grads", False
        )
        cfg.megatron.ddp_num_buckets = cfg.megatron.get("ddp_num_buckets", None)
        cfg.megatron.ddp_pad_buckets_for_high_nccl_busbw = cfg.megatron.get(
            "ddp_pad_buckets_for_high_nccl_busbw", False
        )
        cfg.megatron.enable_gloo_process_groups = cfg.megatron.get(
            "enable_gloo_process_groups", True
        )

        # ddp config
        cfg.megatron.check_for_nan_in_loss_and_grad = cfg.megatron.get(
            "check_for_nan_in_loss_and_grad", False
        )
        cfg.megatron.ddp_bucket_size = cfg.megatron.get("ddp_bucket_size", None)
        cfg.megatron.ddp_average_in_collective = cfg.megatron.get(
            "ddp_average_in_collective", False
        )
        cfg.megatron.accumulate_allreduce_grads_in_fp32 = cfg.megatron.get(
            "accumulate_allreduce_grads_in_fp32", True
        )

        # profiler
        cfg.megatron.use_profiler = cfg.megatron.get("use_profiler", False)
        if cfg.megatron.use_profiler:
            cfg.megatron.profiler.schedule_warmup = cfg.megatron.profiler.get(
                "schedule_warmup", 3
            )
            cfg.megatron.profiler.schedule_active = cfg.megatron.profiler.get(
                "schedule_active", 1
            )

        # distributed
        # If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.
        cfg.megatron.use_tp_pp_dp_mapping = cfg.megatron.get(
            "use_tp_pp_dp_mapping", False
        )
        # Which backend to use for distributed training. Support 'nccl' and 'gloo'
        cfg.megatron.distributed_backend = cfg.megatron.get(
            "distributed_backend", "nccl"
        )
        cfg.megatron.distributed_timeout_minutes = cfg.megatron.get(
            "distributed_timeout_minutes", 10
        )
        cfg.megatron.num_distributed_optimizer_instances = cfg.megatron.get(
            "num_distributed_optimizer_instances", 1
        )
        cfg.megatron.nccl_communicator_config_path = cfg.megatron.get(
            "nccl_communicator_config_path", None
        )
        cfg.megatron.encoder_tensor_model_parallel_size = cfg.megatron.get(
            "encoder_tensor_model_parallel_size", 0
        )
        cfg.megatron.encoder_pipeline_model_parallel_size = cfg.megatron.get(
            "encoder_pipeline_model_parallel_size", 0
        )

        # checkpoint
        cfg.megatron.rerun_mode = cfg.megatron.get(
            "rerun_mode", "disabled"
        )  # choices=['disabled', 'validate_results', 'report_stats']
        cfg.megatron.error_injection_rate = cfg.megatron.get(
            "error_injection_rate", 0
        )  # Rate at which to inject unexpected results, e.g. 1000 means once every 1000 result validations
        cfg.megatron.error_injection_type = cfg.megatron.get(
            "error_injection_type", "transient_error"
        )  # choices=['correct_result', 'transient_error', 'persistent_error']

        cfg.megatron.moe_use_upcycling = cfg.megatron.get("moe_use_upcycling", False)
        cfg.megatron.async_save = cfg.megatron.get("async_save", False)
        cfg.megatron.use_dist_ckpt = cfg.megatron.get("use_dist_ckpt", False)
        cfg.megatron.no_load_optim = cfg.megatron.get("no_load_optim", False)
        cfg.megatron.no_load_rng = cfg.megatron.get("no_load_rng", False)
        cfg.megatron.no_save_optim = cfg.megatron.get("no_save_optim", False)
        cfg.megatron.no_save_rng = cfg.megatron.get("no_save_rng", False)
        cfg.megatron.ckpt_fully_parallel_save = cfg.megatron.get(
            "ckpt_fully_parallel_save", False
        )
        cfg.megatron.ckpt_format = cfg.megatron.get("ckpt_format", "torch")
        cfg.megatron.ckpt_convert_format = cfg.megatron.get(
            "ckpt_convert_format", None
        )  # choices=[None, 'torch', 'torch_dist', 'zarr']
        cfg.megatron.auto_detect_ckpt_format = cfg.megatron.get(
            "auto_detect_ckpt_format", False
        )
        cfg.megatron.non_persistent_save_interval = cfg.megatron.get(
            "non_persistent_save_interval", None
        )
        cfg.megatron.non_persistent_ckpt_type = cfg.megatron.get(
            "non_persistent_ckpt_type", None
        )
        cfg.megatron.non_persistent_local_ckpt_dir = cfg.megatron.get(
            "non_persistent_local_ckpt_dir", None
        )
        cfg.megatron.non_persistent_global_ckpt_dir = cfg.megatron.get(
            "non_persistent_global_ckpt_dir", None
        )
        cfg.megatron.non_persistent_local_ckpt_algo = cfg.megatron.get(
            "non_persistent_local_ckpt_algo", "fully_parallel"
        )  # choices=['fully_parallel', 'atomic']
        cfg.megatron.ckpt_convert_update_legacy_dist_opt_format = cfg.megatron.get(
            "ckpt_convert_update_legacy_dist_opt_format", False
        )
        cfg.megatron.finetune = cfg.megatron.get("finetune", False)
        cfg.megatron.ckpt_assume_constant_structure = cfg.megatron.get(
            "ckpt_assume_constant_structure", False
        )
        cfg.megatron.log_progress = cfg.megatron.get("log_progress", False)
        cfg.megatron.exit_on_missing_checkpoint = cfg.megatron.get(
            "exit_on_missing_checkpoint", True
        )
        cfg.megatron.retro_add_retriever = cfg.megatron.get(
            "retro_add_retriever", False
        )
        cfg.megatron.data_parallel_random_init = cfg.megatron.get(
            "data_parallel_random_init", False
        )
        cfg.megatron.use_tokenizer_model_from_checkpoint_args = cfg.megatron.get(
            "use_tokenizer_model_from_checkpoint_args", False
        )

        # cfg.model
        assert (
            cfg.model.get("precision", None) is not None
            and torch_dtype_from_precision(cfg.model.precision) is not None
        ), "model.precision is required"

        cfg.model.tensor_model_parallel_size = cfg.model.get(
            "tensor_model_parallel_size", 1
        )
        cfg.model.pipeline_model_parallel_size = cfg.model.get(
            "pipeline_model_parallel_size", 1
        )
        cfg.model.virtual_pipeline_model_parallel_size = cfg.model.get(
            "virtual_pipeline_model_parallel_size", None
        )
        cfg.model.pipeline_model_parallel_split_rank = cfg.model.get(
            "pipeline_model_parallel_split_rank", None
        )
        cfg.model.context_parallel_size = cfg.model.get("context_parallel_size", 1)

        cfg.model.expert_model_parallel_size = cfg.model.get(
            "expert_model_parallel_size", 1
        )

        cfg.model.expert_tensor_parallel_size = cfg.model.get(
            "expert_tensor_parallel_size", None
        )

        from rlinf.hybrid_engines.megatron.megatron_model_manager import HAVE_FUSCO

        if HAVE_FUSCO:
            assert (
                cfg.model.moe_token_dispatcher_type == "alltoall"
                and cfg.model.expert_model_parallel_size > 1
                and cfg.model.expert_tensor_parallel_size == 1
                and not cfg.model.variable_seq_lengths
            ), (
                f"FUSCO support detected. to enable FUSCO, moe_token_dispatcher_type must be 'alltoall', expert_model_parallel_size must be greater than 1, expert_tensor_parallel_size must be 1, and variable_seq_lengths must be False. get value ({cfg.model.moe_token_dispatcher_type}, {cfg.model.expert_model_parallel_size}, {cfg.model.expert_tensor_parallel_size}, {cfg.model.variable_seq_lengths})"
            )

        cfg.model.moe_grouped_gemm = cfg.model.get("moe_grouped_gemm", None)
        assert cfg.model.moe_grouped_gemm in [None, "te"], (
            f"grouped_gemm type only avail in [null, te]. get value ({cfg.model.moe_grouped_gemm})"
        )

        if cfg.model.expert_tensor_parallel_size is not None:
            assert (
                cfg.model.expert_tensor_parallel_size
                <= cfg.model.tensor_model_parallel_size
            ), (
                f"expert_tensor_parallel_size ({cfg.model.expert_tensor_parallel_size}) must be less than or equal to tensor_model_parallel_size ({cfg.model.tensor_model_parallel_size})"
            )

        cfg.model.position_embedding_type = cfg.model.get(
            "position_embedding_type", "learned_absolute"
        )
        cfg.model.rotary_percentage = cfg.model.get("rotary_percentage", 1.0)
        cfg.model.seq_len_interpolation_factor = cfg.model.get(
            "seq_len_interpolation_factor", None
        )
        cfg.model.rotary_base = cfg.model.get("rotary_base", 10000)
        cfg.model.share_embeddings_and_output_weights = cfg.model.get(
            "share_embeddings_and_output_weights", False
        )

        cfg.model.gradient_accumulation_fusion = cfg.model.get(
            "gradient_accumulation_fusion", True
        )
        cfg.model.masked_softmax_fusion = cfg.model.get("masked_softmax_fusion", True)
        cfg.model.persist_layer_norm = cfg.model.get("persist_layer_norm", True)

        cfg.model.padded_vocab_size = cfg.model.get("padded_vocab_size", None)
        cfg.model.use_cpu_initialization = cfg.model.get(
            "use_cpu_initialization", False
        )
        cfg.model.add_position_embedding = cfg.model.get("add_position_embedding", True)

        cfg.model.variable_seq_lengths = cfg.model.get("variable_seq_lengths", True)
        cfg.model.add_bias_linear = cfg.model.get("add_bias_linear", False)

        # optimizer config
        if cfg.optim.get("fp16", None) is None:
            cfg.optim.fp16 = (
                torch_dtype_from_precision(cfg.model.precision) == torch.float16
            )
        if cfg.optim.get("bf16", None) is None:
            cfg.optim.bf16 = (
                torch_dtype_from_precision(cfg.model.precision) == torch.bfloat16
            )
        cfg.optim.weight_decay = cfg.optim.get("weight_decay", 0.01)
        cfg.optim.overlap_param_gather_with_optimizer_step = cfg.optim.get(
            "overlap_param_gather_with_optimizer_step", False
        )
        cfg.optim.optimizer_cpu_offload = cfg.optim.get("optimizer_cpu_offload", False)
        cfg.optim.optimizer_offload_fraction = cfg.optim.get(
            "optimizer_offload_fraction", 0.0
        )
        cfg.optim.use_precision_aware_optimizer = cfg.optim.get(
            "use_precision_aware_optimizer", False
        )

        # learning rate
        cfg.lr_sched.lr = cfg.optim.get("lr", None)
        cfg.lr_sched.min_lr = cfg.lr_sched.get("min_lr", 0.0)
        # lr_decay_style choices=['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD']
        cfg.lr_sched.lr_decay_style = cfg.lr_sched.get("lr_decay_style", "constant")
        # weight_decay_incr_style: Weight decay increment function. choices=['constant', 'linear', 'cosine']
        cfg.lr_sched.weight_decay_incr_style = cfg.lr_sched.get(
            "weight_decay_incr_style", "constant"
        )
        # lr_wsd_decay_style choices=['exponential', 'linear', 'cosine']
        cfg.lr_sched.lr_wsd_decay_style = cfg.lr_sched.get(
            "lr_wsd_decay_style", "exponential"
        )

        # TODO fix this
        cfg.megatron.train_iters = 100000
        # lr_decay_iters: number of iterations to decay learning rate over, defaults to train_iters
        cfg.lr_sched.lr_decay_iters = cfg.lr_sched.get("lr_decay_iters", None)
        cfg.lr_sched.lr_wsd_decay_iters = cfg.lr_sched.get("lr_wsd_decay_iters", None)
        cfg.lr_sched.lr_warmup_init = cfg.lr_sched.get("lr_warmup_init", 0.0)
        cfg.lr_sched.lr_warmup_iters = cfg.lr_sched.get("lr_warmup_iters", 0)
        cfg.lr_sched.lr_warmup_fraction = cfg.lr_sched.get("lr_warmup_fraction", None)
        cfg.lr_sched.use_checkpoint_opt_param_scheduler = cfg.lr_sched.get(
            "use_checkpoint_opt_param_scheduler", True
        )
        cfg.lr_sched.override_opt_param_scheduler = cfg.lr_sched.get(
            "override_opt_param_scheduler", False
        )

        if cfg.lr_sched.lr_decay_style == "constant":
            assert cfg.lr_sched.get("start_weight_decay") is None
            assert cfg.lr_sched.get("end_weight_decay") is None
            cfg.lr_sched.start_weight_decay = cfg.optim.weight_decay
            cfg.lr_sched.end_weight_decay = cfg.optim.weight_decay
        else:
            if not hasattr(cfg.lr_sched, "start_weight_decay"):
                raise ValueError(
                    "Error: 'start_weight_decay' is missing from 'cfg.lr_sched'"
                )
            if not hasattr(cfg.lr_sched, "end_weight_decay"):
                raise ValueError(
                    "Error: 'end_weight_decay' is missing from 'cfg.lr_sched'"
                )
            assert cfg.lr_sched.start_weight_decay is not None
            assert cfg.lr_sched.end_weight_decay is not None

        # TODO. Following args are needed for AUTO mode now, but will be removed in the future.
        cfg.megatron.transformer_impl = getattr(
            cfg.megatron, "transformer_impl", "transformer_engine"
        )
        cfg.megatron.swiglu = cfg.model.activation in ["swiglu", "fast-swiglu"]
        cfg.megatron.untie_embeddings_and_output_weights = (
            not cfg.model.share_embeddings_and_output_weights
        )
        # In RLinf, padded_vocab_size is set to hf_config.vocab_size, so make_vocab_size_divisible_by=1
        cfg.megatron.make_vocab_size_divisible_by = 1
        if cfg.model.normalization == "rmsnorm":
            cfg.megatron.normalization = "RMSNorm"

    return cfg


def _validate_cosmos_forward_dynamics_env_contract(actor_model, env_cfg, label):
    """Fail before launch when policy A and frozen model B cannot interoperate."""
    if str(env_cfg.get("env_type", "")) != "cosmos_forward_dynamics_wm":
        return
    fd_cfg = env_cfg.get("forward_dynamics_cfg", None)
    if fd_cfg is None or fd_cfg.get("model", None) is None:
        raise ValueError(f"env.{label}.forward_dynamics_cfg.model is required.")
    fd_model = fd_cfg.model
    actor_cosmos = actor_model.cosmos
    fd_cosmos = fd_model.cosmos

    actor_framework = str(actor_cosmos.get("framework_path", "") or "")
    fd_framework = str(fd_cosmos.get("framework_path", "") or "")
    for owner, framework in (
        ("Policy A", actor_framework),
        ("Forward-dynamics B", fd_framework),
    ):
        framework_path = (
            os.path.realpath(os.path.expanduser(framework)) if framework else ""
        )
        package_path = os.path.join(framework_path, "cosmos_framework")
        if not framework_path or not os.path.isdir(package_path):
            raise ValueError(
                f"{owner} framework_path must contain cosmos_framework: "
                f"{framework or '<unset>'}"
            )

    contracts = {
        "action_dim": (actor_model.action_dim, fd_model.action_dim),
        "action_chunk_size": (
            actor_model.num_action_chunks,
            fd_model.num_action_chunks,
        ),
        "cosmos.action_chunk_size": (
            actor_cosmos.action_chunk_size,
            fd_cosmos.action_chunk_size,
        ),
        "action_representation": (
            actor_cosmos.action_representation,
            fd_cosmos.action_representation,
        ),
        "domain_name": (actor_cosmos.domain_name, fd_cosmos.domain_name),
        "raw_action_dim": (actor_cosmos.raw_action_dim, fd_cosmos.raw_action_dim),
        "max_action_dim": (actor_cosmos.max_action_dim, fd_cosmos.max_action_dim),
        "fps": (actor_cosmos.fps, fd_cosmos.fps),
        "resolution_tier": (
            actor_cosmos.resolution_tier,
            fd_cosmos.resolution_tier,
        ),
        "prompt": (actor_cosmos.prompt, fd_cfg.get("prompt", fd_cosmos.prompt)),
        "cosmos.prompt": (actor_cosmos.prompt, fd_cosmos.prompt),
        "camera_layout": (
            actor_cosmos.get("camera_layout", "main_top"),
            fd_cfg.get("camera_layout", "main_top"),
        ),
    }
    mismatches = [
        f"{name}: A={a_value!r}, B={b_value!r}"
        for name, (a_value, b_value) in contracts.items()
        if str(a_value) != str(b_value)
    ]
    if mismatches:
        raise ValueError(
            f"Cosmos A/B contract mismatch in env.{label}: " + "; ".join(mismatches)
        )

    view_indices = [
        int(fd_cfg.get("main_view_index", 0)),
        int(fd_cfg.get("wrist_view_index", 2)),
        int(fd_cfg.get("extra_view_index", 1)),
    ]
    if sorted(view_indices) != [0, 1, 2]:
        raise ValueError(
            f"env.{label} forward-dynamics view indices must be a permutation "
            f"of [0, 1, 2], got {view_indices}."
        )
    sampler = str(fd_cosmos.get("sampler", "unipc")).lower()
    if sampler not in {"unipc", "edm"}:
        raise ValueError(
            f"env.{label} forward-dynamics sampler must be unipc or edm, "
            f"got {sampler!r}."
        )
    try:
        sampling_values = {
            "num_steps": float(fd_cosmos.get("num_steps", 10)),
            "guidance": float(fd_cosmos.get("guidance", 3.0)),
            "shift": float(fd_cosmos.get("shift", 10.0)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"env.{label} forward-dynamics num_steps, guidance, and shift "
            "must be numeric."
        ) from exc
    invalid_sampling_values = {
        name: value
        for name, value in sampling_values.items()
        if not math.isfinite(value) or value <= 0
    }
    if invalid_sampling_values:
        raise ValueError(
            f"env.{label} forward-dynamics sampling values must be finite and "
            f"positive, got {invalid_sampling_values}."
        )
    if not sampling_values["num_steps"].is_integer():
        raise ValueError(
            f"env.{label} forward-dynamics num_steps must be an integer, got "
            f"{sampling_values['num_steps']}."
        )

    checkpoint = str(fd_cosmos.get("checkpoint_path", "") or fd_model.get("model_path", ""))
    if not checkpoint:
        raise ValueError("Frozen Cosmos B checkpoint_path must be explicitly configured.")
    checkpoint_path = os.path.expanduser(checkpoint)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Frozen Cosmos B checkpoint is not readable: {checkpoint}")
    if os.path.isdir(checkpoint_path):
        metadata_path = os.path.join(checkpoint_path, "model", ".metadata")
        if not os.path.isfile(metadata_path):
            raise ValueError(
                "Frozen Cosmos B DCP checkpoint is missing model/.metadata: "
                f"{checkpoint}"
            )

    config_path = str(fd_cosmos.get("checkpoint_config_path", "") or "")
    if not config_path or not os.path.isfile(os.path.expanduser(config_path)):
        raise ValueError(
            "Frozen Cosmos B checkpoint_config_path must name a readable file: "
            f"{config_path or '<unset>'}"
        )
    stats_path = str(
        fd_cosmos.get("forward_dynamics_action_stats_path", "") or ""
    )
    if not stats_path or not os.path.isfile(os.path.expanduser(stats_path)):
        raise ValueError(
            "Frozen Cosmos B forward_dynamics_action_stats_path must name a "
            f"readable file: {stats_path or '<unset>'}"
        )
    try:
        with open(os.path.expanduser(stats_path), encoding="utf-8") as handle:
            stats = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Frozen Cosmos B action stats are invalid: {stats_path}"
        ) from exc
    if "effective_low" in stats and "effective_high" in stats:
        low, high = stats["effective_low"], stats["effective_high"]
    else:
        ranges = stats.get("global_raw", stats.get("global", stats))
        low = ranges.get("q01", ranges.get("min"))
        high = ranges.get("q99", ranges.get("max"))
    try:
        low_values = [float(value) for value in low]
        high_values = [float(value) for value in high]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frozen Cosmos B action stats have no numeric low/high range: {stats_path}"
        ) from exc
    expected_action_dim = int(fd_model.action_dim)
    if len(low_values) != expected_action_dim or len(high_values) != expected_action_dim:
        raise ValueError(
            "Frozen Cosmos B action stats width does not match action_dim: "
            f"low={len(low_values)}, high={len(high_values)}, "
            f"action_dim={expected_action_dim}."
        )
    if not all(
        math.isfinite(low_value)
        and math.isfinite(high_value)
        and high_value > low_value
        for low_value, high_value in zip(low_values, high_values, strict=True)
    ):
        raise ValueError(
            f"Frozen Cosmos B action stats contain non-finite or empty ranges: {stats_path}"
        )
    stats_action_dim = stats.get("action_dim")
    if stats_action_dim is not None and int(stats_action_dim) != expected_action_dim:
        raise ValueError(
            "Frozen Cosmos B stats action_dim does not match the model: "
            f"{stats_action_dim} vs {expected_action_dim}."
        )
    stats_fps = stats.get("fps")
    if stats_fps is not None and not math.isclose(
        float(stats_fps), float(fd_cosmos.fps), rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(
            f"Frozen Cosmos B stats FPS does not match the model: "
            f"{stats_fps} vs {fd_cosmos.fps}."
        )

    actor_checkpoint = str(
        actor_cosmos.get("checkpoint_path", "") or actor_model.get("model_path", "")
    )
    if (
        actor_checkpoint
        and os.path.realpath(os.path.expanduser(actor_checkpoint))
        == os.path.realpath(os.path.expanduser(checkpoint))
    ):
        raise ValueError("Policy A and forward-dynamics B must use different checkpoints.")



def _validate_cosmos_ctrl_world_contract(cfg) -> None:
    """Fail before model loading when strict native train/eval contracts drift."""

    actor_model = cfg.actor.model
    cosmos_cfg = actor_model.get("cosmos", {})
    strict_source_cfg = cosmos_cfg.get("strict_source_view_size", None)
    if strict_source_cfg is None:
        return
    strict_source_size = tuple(int(value) for value in strict_source_cfg)
    if len(strict_source_size) != 2 or min(strict_source_size) <= 0:
        raise ValueError(
            "actor.model.cosmos.strict_source_view_size must be two positive integers"
        )

    modes = [
        mode
        for mode in ("train", "eval")
        if str(cfg.env[mode].get("env_type", ""))
        in {"ctrl_world_wm", "cosmos_self_ctrl_world"}
    ]
    if not modes:
        return

    actor_chunk = int(actor_model.num_action_chunks)
    actor_fps = float(cosmos_cfg.get("fps", 0.0))
    expected_composite_cfg = cosmos_cfg.get("expected_composite_size", None)
    expected_composite_size = (
        tuple(int(value) for value in expected_composite_cfg)
        if expected_composite_cfg is not None
        else None
    )
    resolved_contracts = {}
    for mode in modes:
        env_cfg = cfg.env[mode]
        ctrl_node = env_cfg.ctrl_world_cfg
        ctrl_cfg = (
            OmegaConf.to_container(ctrl_node, resolve=True)
            if OmegaConf.is_config(ctrl_node)
            else dict(ctrl_node)
        )
        resolved_contracts[mode] = ctrl_cfg

        ctrl_chunk = int(ctrl_cfg.get("chunk", -1))
        if ctrl_chunk != actor_chunk:
            raise ValueError(
                f"env.{mode}.ctrl_world_cfg.chunk must match "
                f"actor.model.num_action_chunks: {ctrl_chunk} != {actor_chunk}"
            )
        action_fps = float(ctrl_cfg.get("cosmos_action_fps", 0.0))
        if actor_fps <= 0.0 or not math.isclose(
            action_fps, actor_fps, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ValueError(
                f"env.{mode}.ctrl_world_cfg.cosmos_action_fps must match "
                f"actor.model.cosmos.fps: {action_fps} != {actor_fps}"
            )

        use_raw_views = ctrl_cfg.get("use_raw_reset_policy_views", False) is True
        ctrl_image_size = tuple(int(value) for value in ctrl_cfg.get("image_size", []))
        if not use_raw_views and ctrl_image_size != strict_source_size:
            raise ValueError(
                f"env.{mode} cannot satisfy strict Cosmos source views "
                f"{strict_source_size}: use_raw_reset_policy_views is false and "
                f"Ctrl-World emits {ctrl_image_size}. Preserve the dataset-resolution "
                "reset views or use an explicitly compatible native image_size."
            )

        condition_fps = float(ctrl_cfg.get("ctrl_condition_fps", action_fps))
        resample_actions = ctrl_cfg.get("action_fps_resample", False) is True
        expected_ctrl_chunk = (
            int(round(ctrl_chunk * condition_fps / action_fps))
            if resample_actions
            else ctrl_chunk
        )
        actual_ctrl_chunk = int(
            ctrl_cfg.get("ctrl_world_chunk", expected_ctrl_chunk)
        )
        if actual_ctrl_chunk != expected_ctrl_chunk:
            raise ValueError(
                f"env.{mode}.ctrl_world_cfg.ctrl_world_chunk is inconsistent with "
                f"the action FPS contract: {actual_ctrl_chunk} != {expected_ctrl_chunk}"
            )
        if int(ctrl_cfg.get("decode_chunk_size", 0)) <= 0:
            raise ValueError(
                f"env.{mode}.ctrl_world_cfg.decode_chunk_size must be positive"
            )

        if expected_composite_size is not None:
            feedback_size = tuple(
                int(value)
                for value in env_cfg.get("self_feedback", {}).get("target_size", [])
            )
            if feedback_size != expected_composite_size:
                raise ValueError(
                    f"env.{mode}.self_feedback.target_size must match the Cosmos "
                    f"composite size: {feedback_size} != {expected_composite_size}"
                )

    if set(resolved_contracts) == {"train", "eval"}:
        shared_fields = (
            "action_dim",
            "policy_action_dim",
            "num_history",
            "chunk",
            "ctrl_world_chunk",
            "decode_chunk_size",
            "image_size",
            "policy_image_size",
            "use_raw_reset_policy_views",
            "per_view_vae_codec",
            "prompt_override",
            "fps",
            "cosmos_action_fps",
            "ctrl_condition_fps",
            "action_fps_resample",
            "ctrl_world_internal_rollout",
            "ctrl_world_window_frames",
            "ctrl_world_window_emit_frames",
            "ctrl_world_window_use_lookahead_latent",
            "ctrl_world_window_history_mode",
            "ctrl_world_history_idx",
            "ctrl_world_history_bank_size",
            "action_to_eef_adapter",
            "adapter_translation_gain",
            "adapter_rotation_gain",
            "adapter_translation_frame",
            "adapter_translation_sign",
            "adapter_rotation_mode",
            "policy_gripper_type",
            "adapter_gripper_hard_clamp",
            "adapter_gripper_hard_mode",
            "model_camera_ids",
            "main_view_index",
            "wrist_view_index",
            "fixed_denoise_seed",
            "rollout_retry_max_retries",
            "common_random_numbers_within_group",
            "common_noise_seed",
            "per_member_vision_seed",
            "reward_model_diagnostic_only",
        )
        train_contract = resolved_contracts["train"]
        eval_contract = resolved_contracts["eval"]
        mismatches = [
            f"{field}: train={train_contract.get(field)!r}, "
            f"eval={eval_contract.get(field)!r}"
            for field in shared_fields
            if (field in train_contract) != (field in eval_contract)
            or train_contract.get(field) != eval_contract.get(field)
        ]
        if mismatches:
            raise ValueError(
                "strict native Ctrl-World train/eval contract mismatch: "
                + "; ".join(mismatches)
            )


def _validate_cosmos_reward_source(cfg) -> None:
    reward_cfg = cfg.get("reward", {})
    if "source" not in reward_cfg:
        return
    source = str(reward_cfg.get("source", "video_similarity"))
    allowed_sources = {"video_similarity", "success_model"}
    if source not in allowed_sources:
        raise ValueError(
            f"reward.source must be one of {sorted(allowed_sources)}, got {source!r}"
        )

    training_source = str(
        reward_cfg.get("training_source", "continuous_combined")
    )
    allowed_training_sources = {
        "continuous_combined",
        "trajectory_mse",
        "terminal_goal_mse",
        "success_binary",
    }
    if training_source not in allowed_training_sources:
        raise ValueError(
            "reward.training_source must be one of "
            f"{sorted(allowed_training_sources)}, got {training_source!r}"
        )
    classifier_cfg = reward_cfg.get("success_classifier", {})
    if training_source == "success_binary":
        if not bool(classifier_cfg.get("enabled", False)):
            raise ValueError("success_binary requires success_classifier.enabled=true")
        if bool(classifier_cfg.get("diagnostic_only", True)):
            raise ValueError("success_binary requires diagnostic_only=false")
        if not bool(classifier_cfg.get("replace_training_reward", False)):
            raise ValueError("success_binary requires replace_training_reward=true")
        if float(classifier_cfg.get("reward_value", 1.0)) != 1.0:
            raise ValueError("success_binary reward_value must be exactly 1.0")
    elif bool(classifier_cfg.get("replace_training_reward", False)):
        raise ValueError(
            "replace_training_reward=true requires training_source=success_binary"
        )

    ctrl_world_modes = []
    for mode in ("train", "eval"):
        env_cfg = cfg.env[mode]
        env_type = str(env_cfg.get("env_type", ""))
        if env_type in {"ctrl_world_wm", "cosmos_self_ctrl_world"}:
            ctrl_world_modes.append(mode)

    camera_keys = list(cfg.env.train.get("initial_image_camera_keys", []))
    is_close_laptop = camera_keys[:3] == [
        "observation.images.d405_0_rgb",
        "observation.images.d405_1_rgb",
        "observation.images.d435_rgb",
    ]
    if is_close_laptop:
        for mode in ctrl_world_modes:
            ctrl_cfg = cfg.env[mode].ctrl_world_cfg
            mapping = (
                int(ctrl_cfg.get("main_view_index", -1)),
                int(ctrl_cfg.get("wrist_view_index", -1)),
            )
            if mapping != (0, 2):
                raise ValueError(
                    f"env.{mode} uses invalid close-laptop camera mapping {mapping}; "
                    "expected main_view_index=0 and wrist_view_index=2"
                )
        actor_layout = str(cfg.actor.model.get("cosmos", {}).get("camera_layout", ""))
        if ctrl_world_modes and actor_layout != "droid":
            raise ValueError(
                "close-laptop Ctrl-World policies require "
                "actor.model.cosmos.camera_layout=droid"
            )

    if source == "video_similarity":
        if not reward_cfg.get("video_similarity", {}).get("enabled", False):
            raise ValueError("video_similarity reward source requires video_similarity.enabled=true")
        return

    if len(ctrl_world_modes) != 2:
        raise ValueError(
            "success_model is supported only for Ctrl-World-backed Native, "
            "Edge4B, and self-feedback environments"
        )
    success_cfg = reward_cfg.get("success_model", {})
    if success_cfg.get("type") != "ResnetRewModel":
        raise ValueError("success_model.type must be ResnetRewModel")
    if success_cfg.get("camera_key") != "observation.images.d405_0_rgb":
        raise ValueError("success_model.camera_key must be observation.images.d405_0_rgb")
    if int(success_cfg.get("ctrl_world_view_index", -1)) != 0:
        raise ValueError("success_model.ctrl_world_view_index must be 0")
    if success_cfg.get("input_view") != "main":
        raise ValueError("success_model.input_view must be main")
    if success_cfg.get("signal") != "first_success":
        raise ValueError("success_model.signal must be first_success")
    if list(success_cfg.get("input_size", [])) != [224, 224]:
        raise ValueError("success_model.input_size must be [224, 224]")
    threshold = float(success_cfg.get("threshold", 0.9))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("success_model.threshold must be in [0, 1]")
    if int(success_cfg.get("inference_batch_size", 0)) <= 0:
        raise ValueError("success_model.inference_batch_size must be positive")
    checkpoint_path = os.path.expanduser(str(success_cfg.get("from_pretrained", "")))
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(
            checkpoint_path, str(success_cfg.get("artifact_name", "resnet_rm.pth"))
        )
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"success_model checkpoint does not exist: {checkpoint_path}")


def _validate_embodied_eval_parallelism(
    cfg: DictConfig, env_world_size: int, stage_num: int
) -> None:
    """Validate eval topology, including duck's sparse one-episode workers.

    The normal evaluator requires an equal, nonempty local batch on every
    environment process. Duck post-update evaluation intentionally assigns its
    fixed Cosmos validation episodes unevenly (including 40 episodes across 32
    workers), so only that explicitly enabled single-stage diagnostic path may
    use a non-divisible eval count.
    """
    total_num_envs = int(cfg.env.eval.total_num_envs)
    env_world_size = int(env_world_size)
    stage_num = int(stage_num)
    assert total_num_envs > 0, (
        "Total number of parallel environments for evaluation must be greater than 0"
    )

    duck_cfg = cfg.get("duck", {})
    duck_eval_cfg = duck_cfg.get("evaluation", {})
    generic_eval_cfg = cfg.get("post_update_evaluation", {})
    sparse_eval_enabled = (
        bool(generic_eval_cfg.get("enabled", False))
        or (
            bool(duck_cfg.get("enabled", False))
            and bool(duck_eval_cfg.get("enabled", False))
        )
    )
    uneven_sparse_eval = (
        sparse_eval_enabled and total_num_envs % env_world_size != 0
    )
    if uneven_sparse_eval:
        assert stage_num == 1, (
            "Uneven sparse evaluation requires rollout.pipeline_stage_num == 1"
        )
        assert total_num_envs > 0, (
            "Sparse evaluation must contain at least one validation episode"
        )
        assert int(cfg.env.eval.group_size) == 1, (
            "Uneven sparse evaluation requires env.eval.group_size == 1"
        )
        return

    assert total_num_envs % env_world_size == 0, (
        "Total number of parallel environments for evaluation must be divisible "
        "by the number of environment processes"
    )
    assert total_num_envs % env_world_size % stage_num == 0, (
        "Total number of parallel environments for evaluation must be divisible "
        "by the number of environment processes and the number of pipeline stages"
    )
    assert total_num_envs // env_world_size // stage_num > 0, (
        "env.eval.total_num_envs // env_world_size // "
        "rollout.pipeline_stage_num must be greater than 0"
    )
    assert (
        total_num_envs // env_world_size // stage_num % cfg.env.eval.group_size == 0
    ), (
        "env.eval.total_num_envs // env_world_size // "
        "rollout.pipeline_stage_num must be divisible by the group size"
    )


def validate_embodied_cfg(cfg):
    assert get_supported_model(cfg.actor.model.model_type).category == "embodied", (
        f"Model type: '{cfg.actor.model.model_type}' is not an embodied model. "
        f"Supported embodied models: {[e.value for e in SupportedModel if e.category == 'embodied']}."
    )
    if get_supported_model(cfg.actor.model.model_type) == SupportedModel.COSMOS:
        validate_cosmos_actor_model_cfg(cfg.actor.model)
        _validate_cosmos_ctrl_world_contract(cfg)
        _validate_cosmos_reward_source(cfg)
        self_feedback_modes = [
            mode for mode in ("train", "eval") if cfg.env[mode].env_type == "cosmos_self_ctrl_world"
        ]
        if self_feedback_modes:
            cosmos_cfg = cfg.actor.model.cosmos
            if cosmos_cfg.get("action_representation") not in {
                "ur5_eef_relative_10d",
                "ur5_eef_relative_10d_droid_native",
            }:
                raise ValueError(
                    "cosmos_self_ctrl_world requires a supported 10D relative EEF "
                    "action representation"
                )
            if int(cfg.actor.model.action_dim) != 10:
                raise ValueError("cosmos_self_ctrl_world requires action_dim=10")
            if (
                str(cfg.reward.get("source", "video_similarity")) == "video_similarity"
                and not cfg.reward.get("video_similarity", {}).get("enabled", False)
            ):
                raise ValueError("video_similarity source requires video_similarity.enabled=true")
            for mode in self_feedback_modes:
                env_cfg = cfg.env[mode]
                feedback_cfg = env_cfg.get("self_feedback", {})
                if env_cfg.get("auto_reset", True):
                    raise ValueError(f"env.{mode}.auto_reset must be false")
                if not feedback_cfg.get("enabled", False):
                    raise ValueError(f"env.{mode}.self_feedback.enabled must be true")
                if feedback_cfg.get("source") != "imagined_video_chunk":
                    raise ValueError(f"env.{mode}.self_feedback.source is invalid")
        internal_fsdp_shard = cfg.actor.model.cosmos.get("internal_fsdp_shard", False)
        fsdp_strategy = str(cfg.actor.fsdp_config.get("strategy", "fsdp2")).lower()
        if internal_fsdp_shard and fsdp_strategy != "native":
            raise ValueError(
                "actor.model.cosmos.internal_fsdp_shard=True lets the Cosmos "
                "backbone shard itself internally via FSDP2, which produces "
                "DTensor parameters that actor.fsdp_config.strategy="
                f"{fsdp_strategy!r} (RLinf's own FSDP wrap) cannot re-shard. "
                "Set actor.fsdp_config.strategy=native to skip RLinf's own "
                "wrap for this model, or leave internal_fsdp_shard=False."
            )
        _validate_cosmos_forward_dynamics_env_contract(
            cfg.actor.model, cfg.env.train, "train"
        )
        _validate_cosmos_forward_dynamics_env_contract(
            cfg.actor.model, cfg.env.eval, "eval"
        )

    # NOTE: Currently we only support actor_critic as PPO algorithm loss, and only support value_head as critic model.
    # This will be updated in the future to support more algorithms and critic models.
    # Check that actor_critic loss requires value_head
    if (
        cfg.algorithm.loss_type == "actor_critic"
        or cfg.algorithm.loss_type == "decoupled_actor_critic"
    ):
        add_value_head = cfg.actor.model.get("add_value_head", False)
        assert add_value_head, (
            f"When using PPO algorithm (algorithm.loss_type='actor_critic'), "
            f"actor.model.add_value_head must be True. "
            f"Current value: {add_value_head}"
        )

    # process num-envs
    HybridComponentPlacement, _, _ = _get_placement_helpers()
    component_placement = HybridComponentPlacement(cfg, _make_cluster())
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")
    actor_world_size = component_placement.get_world_size("actor")

    cross_rank_group = cfg.algorithm.get("cross_rank_group", None)
    cross_rank_enabled = bool(
        cross_rank_group is not None
        and cross_rank_group.get("enabled", False)
    )
    if cross_rank_enabled:
        members_per_rank = int(cross_rank_group.get("members_per_rank", 0))
        group_size = int(cfg.algorithm.group_size)
        local_env_batch = cfg.env.train.total_num_envs // env_world_size // stage_num
        configured_runner_mode = str(
            cfg.runner.get("mode", "synchronous")
        ).lower()
        runner_mode = (
            "synchronous"
            if configured_runner_mode in ("sync", "synchronous")
            else configured_runner_mode
        )
        if bool(cfg.runner.get("async", False)) or any(
            key in cfg.algorithm
            for key in ("staleness_threshold", "behave_weight_threshold")
        ):
            runner_mode = "asynchronous"
        validate_cross_rank_group_configuration(
            runner_mode=runner_mode,
            task_type=str(cfg.runner.task_type),
            adv_type=str(cfg.algorithm.adv_type),
            actor_backend=str(cfg.actor.training_backend),
            num_nodes=int(cfg.cluster.num_nodes),
            pipeline_stage_num=int(stage_num),
            rollout_epoch=int(cfg.algorithm.get("rollout_epoch", 1)),
            group_size=group_size,
            members_per_rank=members_per_rank,
            rank_layout=str(cross_rank_group.get("rank_layout", "contiguous")),
            gather_backend=str(cross_rank_group.get("gather_backend", "actor_world")),
            env_world_size=int(env_world_size),
            actor_world_size=int(actor_world_size),
            total_num_envs=int(cfg.env.train.total_num_envs),
            local_env_batch=int(local_env_batch),
            local_group_size=int(cfg.env.train.group_size),
        )
        if bool(cross_rank_group.get("semantic_topology_v2", False)):
            logical_rounds = int(
                cross_rank_group.get("logical_rounds_per_update", 0)
            )
            waves_per_round = int(
                cross_rank_group.get("waves_per_logical_round", 0)
            )
            slots_per_round = int(
                cross_rank_group.get("group_slots_per_logical_round", 0)
            )
            groups_per_wave = (
                int(env_world_size) * members_per_rank // group_size
            )
            expected_rollout_epochs = logical_rounds * waves_per_round
            if int(cfg.algorithm.rollout_epoch) != expected_rollout_epochs:
                raise ValueError(
                    "semantic topology rollout_epoch mismatch: "
                    f"{cfg.algorithm.rollout_epoch} != {logical_rounds} * "
                    f"{waves_per_round}"
                )
            if groups_per_wave * waves_per_round != slots_per_round:
                raise ValueError(
                    "semantic topology group slots mismatch: "
                    f"{groups_per_wave} groups/wave * {waves_per_round} waves "
                    f"!= {slots_per_round} slots/logical round"
                )
            expected_trajectories = int(
                cross_rank_group.get(
                    "expected_trajectories_per_update",
                    int(cfg.env.train.total_num_envs)
                    * int(cfg.algorithm.rollout_epoch),
                )
            )
            actual_trajectories = (
                int(cfg.env.train.total_num_envs)
                * int(cfg.algorithm.rollout_epoch)
            )
            if actual_trajectories != expected_trajectories:
                raise ValueError(
                    "semantic topology trajectory count mismatch: "
                    f"{actual_trajectories} != {expected_trajectories}"
                )

    if cfg.runner.val_check_interval > 0 or cfg.runner.only_eval:
        _validate_embodied_eval_parallelism(cfg, env_world_size, stage_num)
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    if not cfg.runner.only_eval:
        assert cfg.env.train.total_num_envs > 0, (
            "Total number of parallel environments for training must be greater than 0"
        )
        assert cfg.env.train.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes"
        )
        assert cfg.env.train.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.train.total_num_envs // env_world_size // stage_num > 0, (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.train.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.train.group_size
            == 0
        ), (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.train.max_steps_per_rollout_epoch
            % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    with open_dict(cfg):
        weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        assert weight_sync_interval > 0, "weight_sync_interval must be greater than 0"
        cfg.runner.weight_sync_interval = weight_sync_interval
        if (
            SupportedEnvType(cfg.env.train.env_type) == SupportedEnvType.MANISKILL
            or SupportedEnvType(cfg.env.eval.env_type) == SupportedEnvType.MANISKILL
        ):

            def get_robot_control_mode(robot: str):
                if robot == "panda-qpos":
                    return "pd_joint_delta_pos"
                elif robot == "panda-ee-dpos":
                    return "pd_ee_delta_pos"
                elif robot == "panda-ee-target-dpos":  # for GSEnv
                    return "pd_ee_target_delta_pose"
                elif "google_robot_static" in robot:
                    return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                elif "panda" in robot:
                    return "pd_ee_body_target_delta_pose_real_root_frame"
                else:
                    raise NotImplementedError(f"Robot {robot} not supported")

            cfg.env.train.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )
            cfg.env.eval.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )
        elif (
            SupportedEnvType(cfg.env.train.env_type) == SupportedEnvType.BEHAVIOR
            or SupportedEnvType(cfg.env.eval.env_type) == SupportedEnvType.BEHAVIOR
        ):
            import omnigibson as og

            assert cfg.env.train.base_config_name == "r1pro_behavior", (
                f"Only r1pro_behavior is supported for omnigibson, got {cfg.env.train.base_config_name}"
            )
            # Load the pre-selected configuration and set the online_sampling flag
            config_filename = os.path.join(
                og.example_config_path, "r1pro_behavior.yaml"
            )
            omnigibson_cfg = yaml.load(
                open(config_filename, "r"), Loader=yaml.FullLoader
            )
            omnigibson_cfg = OmegaConf.create(omnigibson_cfg)
            with open_dict(omnigibson_cfg):
                omnigibson_cfg.robots[0].obs_modalities = ["rgb", "depth", "proprio"]
            cfg.env.train.omnigibson_cfg = omnigibson_cfg
            cfg.env.eval.omnigibson_cfg = omnigibson_cfg

    return cfg


def validate_cosmos_actor_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate Cosmos-specific actor config and fill native-training-scale defaults."""

    from rlinf.models.embodiment.cosmos.training_modes import (
        DEFAULT_COSMOS_TRAINING_MODE,
        cfg_get_training_mode,
        validate_cosmos_training_mode,
    )

    cosmos_cfg = model_cfg.get("cosmos", None)
    if cosmos_cfg is None:
        raise ValueError("actor.model.cosmos is required when model_type='cosmos'.")
    with open_dict(cosmos_cfg):
        if cfg_get_training_mode(cosmos_cfg) is None:
            cosmos_cfg.training_mode = DEFAULT_COSMOS_TRAINING_MODE
    validate_cosmos_training_mode(cfg_get_training_mode(cosmos_cfg))
    return model_cfg


def validate_sft_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.actor.get("global_batch_size", None) is not None, (
        "the actor.global_batch_size is not set"
    )
    assert cfg.actor.get("micro_batch_size", None) is not None, (
        "the actor.micro_batch_size is not set"
    )

    with open_dict(cfg):
        if cfg.data.get("train_data_paths", None) is None:
            # if train_data_paths is None, the code will just eval the model
            assert cfg.data.get("eval_data_paths", None) is not None, (
                "the data.train_data_paths is None, so data.eval_data_paths is required"
            )
        elif cfg.data.get("eval_data_paths", None) is not None:
            # set the val_check_interval to max_epochs
            if cfg.runner.get("val_check_interval", None) is None:
                cfg.runner.val_check_interval = cfg.runner.max_epochs
        else:
            # set the val_check_interval to -1 if there is no eval data
            cfg.runner.val_check_interval = -1
    return cfg


def validate_reasoning_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    if cfg.algorithm.recompute_logprobs and cfg.rollout.return_logprobs:
        assert cfg.algorithm.get("importance_sampling_fix", False), (
            "Importance sampling fix must be enabled if both `algorithm.recompute_logprobs` and `rollout.return_logprobs` are True."
        )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        if hasattr(cfg, "critic"):
            cfg.critic.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
            cfg.critic.global_batch_size = (
                cfg.data.rollout_batch_size
                * cfg.algorithm.group_size
                // cfg.algorithm.n_minibatches
            )
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_reasoning_eval_cfg(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )
        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_coding_online_rl_cfg(cfg: DictConfig) -> DictConfig:
    assert (
        get_supported_model(cfg.rollout.model.model_type) == SupportedModel.QWEN2_5
    ), f"Model type {cfg.rollout.model.model_type} is not supported"

    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    if cfg.algorithm.recompute_logprobs and cfg.rollout.return_logprobs:
        assert cfg.algorithm.get("importance_sampling_fix", False), (
            "Importance sampling fix must be enabled if both `algorithm.recompute_logprobs` and `rollout.return_logprobs` are True."
        )

    assert cfg.algorithm.recompute_logprobs, (
        "Online coding task must use recompute_logprobs"
    )

    assert cfg.actor.training_backend == "megatron", (
        "Online coding task must use megatron training backend"
    )

    _, ModelParallelComponentPlacement, PlacementMode = _get_placement_helpers()
    cluster = _make_cluster()
    component_placement = ModelParallelComponentPlacement(cfg, cluster)
    assert component_placement.placement_mode == PlacementMode.DISAGGREGATED, (
        "Online coding task must use disaggregated placement mode"
    )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.runner.per_worker_log = cfg.runner.get("per_worker_log", False)
        cfg.runner.per_worker_log_path = None
        if cfg.runner.per_worker_log:
            cfg.runner.per_worker_log_path = os.path.join(
                cfg.runner.logger.log_path, "worker_logs"
            )

    # Init cluster
    _make_cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    if cfg.runner.task_type == "embodied":
        cfg = validate_embodied_cfg(cfg)
    elif cfg.runner.task_type == "reasoning":
        cfg = validate_reasoning_cfg(cfg)
    elif cfg.runner.task_type == "coding_online_rl":
        cfg = validate_coding_online_rl_cfg(cfg)
    elif cfg.runner.task_type == "reasoning_eval":
        cfg = validate_reasoning_eval_cfg(cfg)
        return cfg
    elif cfg.runner.task_type == "sft":
        cfg = validate_sft_cfg(cfg)

    if cfg.algorithm.adv_type in ("grpo", "grpo_dynamic", "reinpp_baseline"):
        assert cfg.algorithm.group_size > 1

    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (
        f"Unsupported training_backend {cfg.actor.training_backend}. Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
    )

    if cfg.actor.training_backend == "megatron":
        cfg.actor = validate_megatron_cfg(cfg.actor)
        cfg.actor = validate_model_cfg_by_hf_config(
            cfg.actor, cfg.rollout.model.model_path
        )
        # TODO. Need actually pad padded_vocab_size.
        assert (
            cfg.actor.model.padded_vocab_size
            % cfg.actor.model.tensor_model_parallel_size
            == 0
        ), (
            f"padded_vocab_size ({cfg.actor.model.padded_vocab_size}) must be divisible by tensor_model_parallel_size ({cfg.actor.model.tensor_model_parallel_size})"
        )
    elif cfg.actor.training_backend == "fsdp":
        HybridComponentPlacement, _, _ = _get_placement_helpers()
        component_placement = HybridComponentPlacement(cfg, _make_cluster())
        actor_world_size = component_placement.get_world_size("actor")
        assert (
            cfg.actor.global_batch_size
            % (cfg.actor.micro_batch_size * actor_world_size)
            == 0
        ), (
            f"actor.global_batch_size ({cfg.actor.global_batch_size}) must be divisible by (actor.micro_batch_size ({cfg.actor.micro_batch_size}) * actor_world_size ({actor_world_size}))"
        )
        cfg.actor = validate_fsdp_cfg(cfg.actor)

    if cfg.critic.use_critic_model and cfg.critic.training_backend == "megatron":
        cfg.critic = validate_megatron_cfg(cfg.critic)
        cfg.critic = validate_model_cfg_by_hf_config(
            cfg.critic, cfg.rollout.model.model_path
        )
    elif cfg.critic.use_critic_model and cfg.critic.training_backend == "fsdp":
        cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)


def build_transformer_config(cfg) -> "TransformerConfig":
    """
    Builds the megatron core transformer config for the model.
    For attributes in the RLinf model config that are the same
    as the megatron core TransformerConfig, we will use the value from the RLinf model config.
    For attributes in TransformerConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    # get model parallel configs
    model_parallel_config = _build_model_parallel_config(cfg)

    # create a dictionary copy of the model config
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # create a dict to store the transformer config arguments
    transformer_config_dict = {}

    num_layers = cfg.get("num_layers", 1)
    if num_layers % cfg.get("pipeline_model_parallel_size", 1) != 0:
        raise ValueError(
            f"num_layers ({cfg.num_layers}) should be divisible by "
            f"pipeline_model_parallel_size ({cfg.get('pipeline_model_parallel_size', 1)})"
        )

    add_bias_linear = cfg.get("add_bias_linear", True)
    add_qkv_bias = cfg.get("add_qkv_bias", False)

    activation = cfg.get("activation", "gelu")
    gated_linear_unit = activation.endswith("glu")
    # TODO: need to check which activation functions are supported in mcore
    activation_func = activation_to_func(
        activation, openai_gelu=cfg.get("openai_gelu", False)
    )

    normalization = cfg.get("normalization", "layernorm").lower()
    layernorm_zero_centered_gamma = cfg.get(
        "normalization", "layernorm"
    ) == "layernorm1p" or cfg.get("layernorm_zero_centered_gamma", False)
    if normalization == "layernorm":
        normalization = "LayerNorm"
    elif normalization == "rmsnorm":
        normalization = "RMSNorm"
    elif normalization == "layernorm1p":
        normalization = "LayerNorm"
        layernorm_zero_centered_gamma = True
    else:
        logging.warning(
            f"The normalization type: {normalization} might not be supported in megatron core."
            f"Supported types are LayerNorm and RMSNorm."
        )

    tp_comm_overlap = cfg.get("tp_comm_overlap", False)

    if not cfg.get("fp8", False):
        fp8 = None
    elif cfg.get("fp8_e4m3", False):
        fp8 = "e4m3"
    elif cfg.get("fp8_hybrid", False):
        fp8 = "hybrid"
    else:
        raise ValueError(
            "fp8 enabled but fp8_format (fp8_e4m3 | fp8_hybrid) is not set."
        )

    init_method_std = cfg.get("init_method_std", 0.02)
    # default used in mcore
    init_method = init_method_normal(init_method_std)

    output_layer_init_method = init_method

    use_scaled_init_method = cfg.get("use_scaled_init_method", True)
    if use_scaled_init_method:
        output_layer_init_method = scaled_init_method_normal(
            init_method_std, num_layers=num_layers
        )

    attention_softmax_in_fp32 = cfg.get("attention_softmax_in_fp32", True)
    apply_query_key_layer_scaling = cfg.get("apply_query_key_layer_scaling", False)

    rotary_interleaved = cfg.get("rotary_interleaved", False)

    if apply_query_key_layer_scaling:
        if model_parallel_config.fp16:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
        else:
            logging.warning(
                "apply_query_key_layer_scaling is only enabled when using FP16, setting it to False "
                "and setting NVTE_APPLY_QK_LAYER_SCALING=0"
            )
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "0"
            apply_query_key_layer_scaling = False

    if apply_query_key_layer_scaling:
        attention_softmax_in_fp32 = True

    bias_activation_fusion = cfg.get("bias_activation_fusion", True)

    bias_dropout_fusion = cfg.get("bias_dropout_fusion", True)

    apply_rope_fusion = cfg.get("apply_rope_fusion", False)

    # TODO: need to check if recompute APIs are matching up properly
    recompute_granularity = cfg.get("recompute_granularity", None)
    recompute_method = cfg.get("recompute_method", None)
    recompute_num_layers = cfg.get("recompute_num_layers", None)

    tp_only_amax_red = cfg.get("tp_only_amax_red", False)

    if cfg.get("enable_cuda_graph", False):
        if importlib.util.find_spec("transformer_engine") is None:
            raise ImportError(
                "Can not import transformer_engine, which is required for cudagraphs."
            )
        assert cfg.get("use_te_rng_tracker", False), (
            "Transformer engine's RNG tracker is required for cudagraphs, this can be enabled with \
            'use_te_rng_tracker=True'."
        )

    # any configs that are not in the RLinf model config will be added here
    config_mapping = {
        "apply_query_key_layer_scaling": apply_query_key_layer_scaling,
        "apply_residual_connection_post_layernorm": False,  # we don't use this in NeMo
        "add_bias_linear": add_bias_linear,
        "add_qkv_bias": add_qkv_bias,
        "gated_linear_unit": gated_linear_unit,
        "activation_func": activation_func,
        "normalization": normalization,
        "layernorm_zero_centered_gamma": layernorm_zero_centered_gamma,
        "init_method": init_method,
        "output_layer_init_method": output_layer_init_method,
        "attention_softmax_in_fp32": attention_softmax_in_fp32,
        "bias_activation_fusion": bias_activation_fusion,
        "bias_dropout_fusion": bias_dropout_fusion,
        "apply_rope_fusion": apply_rope_fusion,
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "distribute_saved_activations": False,  # not currently used in NeMo
        "fp8": fp8,
        "tp_comm_overlap": tp_comm_overlap,
        "rotary_interleaved": rotary_interleaved,
        "deallocate_pipeline_outputs": False,
        "tp_only_amax_red": tp_only_amax_red,
        "qk_layernorm": cfg.get("qk_layernorm", False),
        "kv_channels": cfg.get("head_dim", None),
        # MoE related
        "num_moe_experts": cfg.get("num_moe_experts", None),
        "moe_ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None),
        # now the sequential mlp should ffn hidden size == moe_ffn_hidden_size
        "ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None)
        or cfg.get("ffn_hidden_size", None),
        "moe_router_load_balancing_type": cfg.get(
            "moe_router_load_balancing_type", "aux_loss"
        ),
        "moe_router_topk": cfg.get("moe_router_topk", 2),
        "moe_grouped_gemm": cfg.get("moe_grouped_gemm", False),
        "moe_aux_loss_coeff": cfg.get(
            "moe_aux_loss_coeff", 0
        ),  # 1e-2 would be a good start value for load balance loss.
        "moe_z_loss_coeff": cfg.get(
            "moe_z_loss_coeff", None
        ),  # 1e-3 would be a good start value for z-loss
        "moe_input_jitter_eps": cfg.get("moe_input_jitter_eps", None),
        "moe_token_dropping": cfg.get("moe_token_dropping", False),
        "enable_cuda_graph": cfg.get("enable_cuda_graph", False),
    }

    # populate the transformer config dict
    for field in dataclasses.fields(TransformerConfig):
        # config mapping has second highest priority
        if field.name in config_mapping:
            transformer_config_dict[field.name] = config_mapping[field.name]
        # then config
        elif field.name in cfg:
            transformer_config_dict[field.name] = cfg[field.name]
        # then model parallel config
        elif field in dataclasses.fields(model_parallel_config):
            transformer_config_dict[field.name] = getattr(
                model_parallel_config, field.name
            )

    transformer_config = TransformerConfig(**transformer_config_dict)

    # pass mcore customization configs directly to mcore
    mcore_customization_config_dict = cfg.get("mcore_customization_config", {})
    for key, value in mcore_customization_config_dict.items():
        setattr(transformer_config, key, value)

    return transformer_config


def _build_model_parallel_config(cfg: DictConfig) -> "ModelParallelConfig":
    """
    For attributes in the RLinf model config that are the same as the
    megatron core ModelParallelConfig we will use the value from the RLinf config.
    For attributes in ModelParallelConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.training.global_vars import get_timers
    # cfg = OmegaConf.to_container(cfg, resolve=True)

    # dtype used in p2p communication
    if cfg.get("precision", None) is None:
        raise f"precision not found in {cfg}."
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    params_dtype = (
        torch_dtype if torch_dtype in [torch.bfloat16, torch.float16] else torch.float32
    )
    pipeline_dtype = cfg.get("pipeline_dtype", params_dtype)
    autocast_dtype = cfg.get("autocast_dtype", params_dtype)

    timers = get_timers()
    # maps NeMo model configs to ModelParallelConfig from megatron core
    config_mapping = {
        "perform_initialization": True,  # initailize weights when constructing the module
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
        "params_dtype": params_dtype,
        "timers": timers,
        "async_tensor_model_parallel_allreduce": False,  # Deprecated in megatron
        "pipeline_dtype": pipeline_dtype,
        "grad_scale_func": None,
        "enable_autocast": False,  # torch_dtype in [torch.bfloat16, torch.float16],
        "autocast_dtype": autocast_dtype,
        "num_microbatches_with_partial_activation_checkpoints": cfg.get(
            "num_microbatches_with_partial_activation_checkpoints", None
        ),
        "batch_p2p_sync": True,  # call torch.cuda.synchronize() after batch isend/rcv
        "use_ring_exchange_p2p": False,
        "deallocate_pipeline_outputs": False,
        "no_sync_func": None,  # set dynamically during training
        "grad_sync_func": None,  # set dynamically during training
        "param_sync_func": None,  # set dynamically during training
        "tp_comm_overlap": cfg.get("tp_comm_overlap", False),
        "tp_comm_bootstrap_backend": cfg.get("tp_comm_bootstrap_backend", "nccl"),
    }

    # instantitate ModelParallelConfig from this dict
    mp_config_dict = {}

    for field in dataclasses.fields(ModelParallelConfig):
        # model config has priority
        if field.name in cfg:
            mp_config_dict[field.name] = cfg[field.name]
        # then config_mapping
        elif field.name in config_mapping:
            mp_config_dict[field.name] = config_mapping[field.name]

    model_parallel_config = ModelParallelConfig(**mp_config_dict)

    return model_parallel_config
