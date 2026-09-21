"""Strict YAML configuration schema with opt-in compatibility defaults.

Every required dataclass field must be present in YAML. This is intentional: the old
launch path mixed Hydra inheritance, shell variables, and runtime edits, which
made the effective training contract impossible to audit reliably.
"""

from __future__ import annotations

import dataclasses
import math
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates its declared contract."""


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    seed: int
    dtype: str
    max_updates: int
    segment_updates: int
    continuous_stress: bool
    output_dir: str
    resume: bool
    require_real_assets: bool
    deterministic: bool
    minimum_free_disk_gib: int
    minimum_free_inodes: int
    max_gpu_reserved_fraction: float


@dataclass(frozen=True)
class SlurmConfig:
    account: str
    partition: str
    nodes: int
    gpus_per_node: int
    ntasks_per_node: int
    cpus_per_task: int
    memory: str
    time_limit: str
    signal_seconds: int


@dataclass(frozen=True)
class BatchConfig:
    parallel_envs: int
    rollout_epochs: int
    trajectories_per_update: int
    chunks_per_trajectory: int
    global_minibatch_chunks: int
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    expected_optimizer_steps: int
    logical_rounds_per_update: int
    group_slots_per_logical_round: int
    waves_per_logical_round: int


@dataclass(frozen=True)
class TaskConfig:
    profile: str
    mode: str
    base_prompt: str
    allowed_variants: tuple[str, ...]
    active_variants: tuple[str, ...]
    train_episodes_per_variant: int
    eval_episodes_per_variant: int
    append_viewpoint_info: bool
    append_duration_fps: bool
    append_resolution_info: bool
    rollout_cfg_dropout_rate: float
    camera_labels: tuple[str, str, str]
    split_manifest_path: Optional[str]
    variant_prompts: dict[str, str]


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    advantage: str
    group_size: int
    gamma: float
    gae_lambda: Optional[float]
    clip_ratio_low: float
    clip_ratio_high: float
    normalize_advantages: bool
    logprob_level: str


@dataclass(frozen=True)
class TrajectoryRecordsConfig:
    enabled: bool
    mode: str
    auto_analyze: bool
    require_plots: bool



@dataclass(frozen=True)
class FPOConfig:
    replay_objective: str
    ratio_granularity: str
    score_parameterization: str
    num_mc_samples: int
    time_distribution: str
    training_shift: float
    timestep_scale: float
    vision_state_channels: int
    vision_condition_latent_frames: int
    latent_downsample_factor: int
    freeze_video_path: bool
    train_action_path_only: bool


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    clip_grad_norm: float


@dataclass(frozen=True)
class CosmosConfig:
    framework_version: str
    data_parallel_shard_degree: int
    checkpoint_path: Optional[str]
    checkpoint_config_path: Optional[str]
    action_stats_path: Optional[str]
    vlm_processor_path: Optional[str]
    checkpoint_target_rewrite: bool
    strict_load: bool
    action_dim: int
    max_action_dim: int
    action_representation: str
    action_normalization: str
    action_chunk: int
    action_fps: float
    sampler: str
    sampler_steps: int
    guidance: float
    resolution_tier: int


@dataclass(frozen=True)
class CtrlWorldArchitectureConfig:
    sample_size: Optional[int]
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    addition_time_embed_dim: int
    projection_class_embeddings_input_dim: int
    layers_per_block: int
    cross_attention_dim: int
    transformer_layers_per_block: int
    num_attention_heads: tuple[int, ...]
    num_frames: int


@dataclass(frozen=True)
class CtrlWorldRuntimeConfig:
    action_dim: int
    policy_action_dim: int
    num_history: int
    action_encoder_hidden_size: int
    text_cond: bool
    frame_level_cond: bool
    his_cond_zero: bool
    policy_chunk: int
    policy_action_fps: float
    condition_fps: float
    resample_actions: bool
    resampled_chunk: int
    internal_rollout: bool
    window_frames: int
    window_emit_frames: int
    use_lookahead_latent: bool
    history_mode: str
    history_indices: tuple[int, ...]
    history_bank_size: int
    num_inference_steps: int
    guidance_scale: float
    decode_chunk_size: int
    per_view_image_size: tuple[int, int]
    per_view_vae_codec: bool
    action_to_eef_adapter: str


@dataclass(frozen=True)
class CtrlWorldConfig:
    architecture: CtrlWorldArchitectureConfig
    runtime: CtrlWorldRuntimeConfig


@dataclass(frozen=True)
class VideoRewardConfig:
    enabled: bool
    metric: str
    alignment_mode: str
    camera_layout: str
    size: tuple[int, int]
    range_policy: str
    scale: float
    # Physical camera names; null preserves the historical canvas pixel average.
    view_weights: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class TerminalRewardConfig:
    enabled: bool
    same_episode: bool
    frame_index: int
    window_size: int
    size: tuple[int, int]
    range_policy: str
    view_weights: tuple[float, float, float]
    scale: float


@dataclass(frozen=True)
class SuccessClassifierConfig:
    enabled: bool
    diagnostic_only: bool
    replace_training_reward: bool
    threshold: float
    aggregation: str


@dataclass(frozen=True)
class RewardConfig:
    source: str
    video_similarity: VideoRewardConfig
    terminal_goal: TerminalRewardConfig
    success_classifier: SuccessClassifierConfig


@dataclass(frozen=True)
class AssetConfig:
    legacy_config_name: Optional[str]
    dataset_path: Optional[str]
    ctrl_world_checkpoint: Optional[str]
    ctrl_world_stats: Optional[str]
    svd_model_path: Optional[str]
    clip_model_path: Optional[str]
    reward_models: dict[str, str]
    expected_sha256: dict[str, str]


@dataclass(frozen=True)
class DistributedConfig:
    backend: str
    ray_object_store_bytes: int
    ray_ready_timeout_seconds: int
    collective_timeout_seconds: int
    heartbeat_interval_seconds: int
    actor_offload: bool
    rollout_offload: bool
    replay_cpu_resident: bool
    activation_checkpointing: str
    fsdp_use_orig_params: bool


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool
    interval_updates: int
    episodes_per_variant: int
    save_video: bool
    fixed_seeds: bool = False
    before_training: bool = False


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool
    interval_updates: int
    keep_latest: int
    atomic: bool
    save_optimizer: bool
    emergency_on_signal: bool


@dataclass(frozen=True)
class TrainConfig:
    schema_version: int
    experiment_name: str
    runtime: RuntimeConfig
    slurm: SlurmConfig
    batch: BatchConfig
    task: TaskConfig
    algorithm: AlgorithmConfig
    trajectory_records: TrajectoryRecordsConfig
    fpo: FPOConfig
    optimizer: OptimizerConfig
    cosmos: CosmosConfig
    ctrl_world: CtrlWorldConfig
    reward: RewardConfig
    assets: AssetConfig
    distributed: DistributedConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig

    @property
    def world_size(self) -> int:
        return self.slurm.nodes * self.slurm.gpus_per_node

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"schema_version must be 1, got {self.schema_version}")
        if self.runtime.backend not in {"single", "ray_fsdp"}:
            errors.append("runtime.backend must be 'single' or 'ray_fsdp'")
        if self.distributed.backend != self.runtime.backend:
            errors.append("distributed.backend must equal runtime.backend")
        if self.algorithm.name != "fpo_grpo":
            errors.append("only algorithm.name=fpo_grpo is supported")
        if self.algorithm.advantage != "grpo_action_suffix":
            errors.append("only advantage=grpo_action_suffix is supported")
        if self.algorithm.gae_lambda is not None:
            errors.append("gae_lambda must be null because GRPO suffix does not use GAE")
        if self.fpo.ratio_granularity not in {"per_action", "per_mc"}:
            errors.append("fpo.ratio_granularity must be per_action or per_mc")
        if self.fpo.score_parameterization not in {"velocity", "epsilon"}:
            errors.append("fpo.score_parameterization must be velocity or epsilon")
        if self.trajectory_records.mode not in {"disabled", "lightweight_analysis"}:
            errors.append(
                "trajectory_records.mode must be disabled or lightweight_analysis"
            )
        if self.trajectory_records.enabled != (
            self.trajectory_records.mode == "lightweight_analysis"
        ):
            errors.append(
                "trajectory_records.enabled must match mode=lightweight_analysis"
            )
        if self.trajectory_records.auto_analyze and not self.trajectory_records.enabled:
            errors.append(
                "trajectory_records.auto_analyze requires trajectory recording"
            )
        if self.algorithm.group_size < 2:
            errors.append("algorithm.group_size must be >= 2")
        if self.batch.trajectories_per_update % self.algorithm.group_size:
            errors.append("trajectories_per_update must be divisible by group_size")
        if self.batch.parallel_envs * self.batch.rollout_epochs != self.batch.trajectories_per_update:
            errors.append("parallel_envs * rollout_epochs must equal trajectories_per_update")
        # Semantic group slots describe global GRPO groups, not env ranks.
        members_per_rank = 4
        if self.runtime.backend == "ray_fsdp":
            if self.algorithm.group_size % members_per_rank:
                errors.append("algorithm.group_size must be divisible by four members per rank")
            else:
                groups_per_wave = self.world_size * members_per_rank // self.algorithm.group_size
                if self.batch.rollout_epochs != (
                    self.batch.logical_rounds_per_update * self.batch.waves_per_logical_round
                ):
                    errors.append(
                        "rollout_epochs must equal logical_rounds_per_update * "
                        "waves_per_logical_round"
                    )
                if self.batch.group_slots_per_logical_round != (
                    groups_per_wave * self.batch.waves_per_logical_round
                ):
                    errors.append(
                        "group_slots_per_logical_round must equal groups_per_wave * "
                        "waves_per_logical_round"
                    )
                semantic_trajectories = (
                    self.batch.logical_rounds_per_update
                    * self.batch.group_slots_per_logical_round
                    * self.algorithm.group_size
                )
                if semantic_trajectories != self.batch.trajectories_per_update:
                    errors.append("semantic topology must reproduce trajectories_per_update")
        if self.batch.chunks_per_trajectory <= 0:
            errors.append("chunks_per_trajectory must be positive")
        expected_minibatch = (
            self.batch.micro_batch_size_per_gpu
            * self.world_size
            * self.batch.gradient_accumulation_steps
        )
        if self.batch.global_minibatch_chunks != expected_minibatch:
            errors.append(
                "global_minibatch_chunks must equal micro_batch_size_per_gpu "
                "* world_size * gradient_accumulation_steps"
            )
        flattened = self.batch.trajectories_per_update * self.batch.chunks_per_trajectory
        if flattened % self.batch.global_minibatch_chunks:
            errors.append("flattened rollout chunks must divide global_minibatch_chunks exactly")
        elif flattened // self.batch.global_minibatch_chunks != self.batch.expected_optimizer_steps:
            errors.append("expected_optimizer_steps does not match flattened rollout/minibatch")
        if self.cosmos.data_parallel_shard_degree <= 0:
            errors.append("cosmos.data_parallel_shard_degree must be positive")
        elif (
            self.runtime.backend == "ray_fsdp"
            and self.world_size % self.cosmos.data_parallel_shard_degree
        ):
            errors.append(
                "world_size must be divisible by cosmos.data_parallel_shard_degree"
            )
        ctrl_runtime = self.ctrl_world.runtime
        if self.cosmos.action_chunk != 32 or ctrl_runtime.policy_chunk != 32:
            errors.append("Cosmos and Ctrl-World policy chunks must both be 32")
        if ctrl_runtime.policy_action_fps <= 0 or ctrl_runtime.condition_fps <= 0:
            errors.append("Ctrl-World policy_action_fps and condition_fps must be positive")
        else:
            expected_resampled_chunk = (
                round(
                    ctrl_runtime.policy_chunk
                    * ctrl_runtime.condition_fps
                    / ctrl_runtime.policy_action_fps
                )
                if ctrl_runtime.resample_actions
                else ctrl_runtime.policy_chunk
            )
            if ctrl_runtime.resampled_chunk != expected_resampled_chunk:
                errors.append(
                    "Ctrl-World resampled_chunk must match policy_chunk and the FPS ratio"
                )
        if ctrl_runtime.decode_chunk_size <= 0:
            errors.append("Ctrl-World decode_chunk_size must be positive")
        if (
            len(ctrl_runtime.per_view_image_size) != 2
            or min(ctrl_runtime.per_view_image_size) <= 0
        ):
            errors.append("Ctrl-World per_view_image_size must contain two positive values")
        if ctrl_runtime.history_mode not in {"dense", "sparse_window"}:
            errors.append(
                "Ctrl-World history_mode must be dense or sparse_window"
            )
        if ctrl_runtime.num_history <= 0:
            errors.append("Ctrl-World num_history must be positive")
        elif len(ctrl_runtime.history_indices) != ctrl_runtime.num_history:
            errors.append("Ctrl-World history_indices length must match num_history")
        required_history_bank = max(
            ctrl_runtime.num_history,
            max(
                (abs(value) for value in ctrl_runtime.history_indices if value < 0),
                default=0,
            ),
        )
        if ctrl_runtime.history_bank_size < required_history_bank:
            errors.append("Ctrl-World history_bank_size is too small for history_indices")
        if ctrl_runtime.internal_rollout and not (
            ctrl_runtime.window_frames > 0
            and 1 <= ctrl_runtime.window_emit_frames <= ctrl_runtime.window_frames
        ):
            errors.append(
                "Ctrl-World internal rollout requires valid window_frames/window_emit_frames"
            )
        if (
            ctrl_runtime.internal_rollout
            and ctrl_runtime.use_lookahead_latent
            and ctrl_runtime.window_emit_frames >= ctrl_runtime.window_frames
        ):
            errors.append(
                "Ctrl-World lookahead rollout requires window_emit_frames "
                "< window_frames"
            )
        if self.cosmos.action_dim != 10 or self.ctrl_world.runtime.action_dim != 7:
            errors.append("selected checkpoints require Cosmos 10D and Ctrl-World 7D actions")
        if self.runtime.backend == "ray_fsdp" and self.cosmos.action_representation not in {
            "ur5_eef_relative_10d",
            "ur5_eef_relative_10d_droid_native",
        }:
            # Reject aliases that the real Cosmos backend cannot load.
            errors.append("real training requires a canonical Cosmos 10D relative EEF contract")
        if self.task.profile not in {"synthetic", "close", "duck", "nest_four_cups", "push_t"}:
            errors.append(
                "task.profile must be synthetic, close, duck, nest_four_cups, or push_t"
            )
        if not self.task.active_variants:
            errors.append("task.active_variants must not be empty")
        unknown_variants = set(self.task.active_variants) - set(self.task.allowed_variants)
        if unknown_variants:
            errors.append(f"unsupported active_variants: {sorted(unknown_variants)}")
        missing_prompts = set(self.task.active_variants) - set(self.task.variant_prompts)
        if missing_prompts:
            errors.append(f"missing variant prompts: {sorted(missing_prompts)}")
        if len(set(self.task.active_variants)) != len(self.task.active_variants):
            errors.append("task.active_variants must not contain duplicates")
        groups_per_update = self.batch.trajectories_per_update // self.algorithm.group_size
        if groups_per_update % len(self.task.active_variants):
            errors.append("GRPO groups per update must be divisible by active_variants")
        elif self.task.profile == "duck":
            groups_per_color = groups_per_update // len(self.task.active_variants)
            if groups_per_color % self.batch.logical_rounds_per_update:
                errors.append(
                    "Duck groups per color must be divisible by logical_rounds_per_update"
                )
        if self.task.profile == "close" and self.task.base_prompt != "Close the laptop":
            errors.append("close base_prompt must exactly match the SFT dataset: 'Close the laptop'")
        if (
            self.task.profile == "nest_four_cups"
            and self.task.base_prompt != "Nest the four cups into one stack"
        ):
            errors.append(
                "nest_four_cups base_prompt must exactly match the SFT dataset: "
                "'Nest the four cups into one stack'"
            )
        if self.task.profile in {"duck", "nest_four_cups", "push_t"} and not self.task.split_manifest_path:
            errors.append(f"{self.task.profile} requires task.split_manifest_path")
        if (
            self.evaluation.enabled
            and self.evaluation.episodes_per_variant
            != self.task.eval_episodes_per_variant
        ):
            errors.append(
                "evaluation.episodes_per_variant must equal "
                "task.eval_episodes_per_variant"
            )
        if self.task.rollout_cfg_dropout_rate != 0.0:
            errors.append("rollout_cfg_dropout_rate must be 0.0; SFT-only CFG dropout is not replayed")
        weights = self.reward.video_similarity.view_weights
        if weights is not None:
            if set(weights) != {"main", "wrist", "side"}:
                errors.append("video_similarity.view_weights must name main, wrist, side")
            if any(not math.isfinite(v) or v < 0 for v in weights.values()):
                errors.append("video_similarity.view_weights must be finite and nonnegative")
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                errors.append("video_similarity.view_weights must sum to 1")
            if self.reward.video_similarity.camera_layout not in {"droid", "main_top"}:
                errors.append("weighted video reward requires droid or main_top layout")
        if self.evaluation.before_training and not self.evaluation.enabled:
            errors.append("evaluation.before_training requires evaluation.enabled")
        if abs(sum(self.reward.terminal_goal.view_weights) - 1.0) > 1e-6:
            errors.append("terminal_goal.view_weights must sum to 1")
        allowed_reward_sources = {
            "ctrl_world_aligned_mse_plus_terminal",
            "trajectory_mse",
            "terminal_goal_mse",
            "success_binary",
        }
        if self.reward.source not in allowed_reward_sources:
            errors.append(
                "reward.source must be one of "
                f"{sorted(allowed_reward_sources)}, got {self.reward.source!r}"
            )
        full_diagnostic_ablation = self.reward.source in {
            "trajectory_mse",
            "terminal_goal_mse",
            "success_binary",
        }
        if full_diagnostic_ablation and not self.reward.video_similarity.enabled:
            errors.append(
                "video_similarity must remain enabled so every reward ablation "
                "records the same trajectory-MSE diagnostics"
            )
        if full_diagnostic_ablation and not self.reward.terminal_goal.enabled:
            errors.append(
                "terminal_goal must remain enabled so every reward ablation "
                "records the same terminal-MSE diagnostics"
            )
        if full_diagnostic_ablation and not self.reward.success_classifier.enabled:
            errors.append(
                "success_classifier must remain enabled so every reward ablation "
                "records the same terminal-success diagnostics"
            )
        success_is_training_reward = self.reward.source == "success_binary"
        if success_is_training_reward:
            if self.reward.success_classifier.diagnostic_only:
                errors.append(
                    "success_binary requires success_classifier.diagnostic_only=false"
                )
            if not self.reward.success_classifier.replace_training_reward:
                errors.append(
                    "success_binary requires "
                    "success_classifier.replace_training_reward=true"
                )
        else:
            if self.reward.success_classifier.replace_training_reward:
                errors.append(
                    "success_classifier.replace_training_reward is only valid "
                    "for reward.source=success_binary"
                )
            if not self.reward.success_classifier.diagnostic_only:
                errors.append(
                    "non-success reward sources require "
                    "success_classifier.diagnostic_only=true"
                )
        if self.runtime.segment_updates <= 0 or self.runtime.max_updates <= 0:
            errors.append("max_updates and segment_updates must be positive")
        if not 0.0 < self.runtime.max_gpu_reserved_fraction <= 1.0:
            errors.append("max_gpu_reserved_fraction must be in (0,1]")
        if self.runtime.require_real_assets:
            if not self.assets.legacy_config_name:
                errors.append("assets.legacy_config_name is required for real training")
            for label, value in {
                "cosmos.checkpoint_path": self.cosmos.checkpoint_path,
                "cosmos.checkpoint_config_path": self.cosmos.checkpoint_config_path,
                "cosmos.action_stats_path": self.cosmos.action_stats_path,
                "assets.dataset_path": self.assets.dataset_path,
                "assets.ctrl_world_checkpoint": self.assets.ctrl_world_checkpoint,
                "assets.ctrl_world_stats": self.assets.ctrl_world_stats,
                "assets.svd_model_path": self.assets.svd_model_path,
                "assets.clip_model_path": self.assets.clip_model_path,
            }.items():
                if value is None or not str(value).strip():
                    errors.append(f"{label} is required for real training")
        if errors:
            raise ConfigError("Invalid training configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin is Union or (union_type is not None and origin is union_type):
        if value is None and type(None) in args:
            return None
        candidates = [item for item in args if item is not type(None)]
        if len(candidates) != 1:
            raise ConfigError(f"{path}: unsupported union annotation {annotation!r}")
        return _coerce(candidates[0], value, path)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        return _strict_dataclass(annotation, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected list/tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item, f"{path}[]") for item in value)
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} entries, got {len(value)}")
        return tuple(_coerce(kind, item, f"{path}[{idx}]") for idx, (kind, item) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping")
        key_type, value_type = args
        return {
            _coerce(key_type, key, f"{path}.<key>"): _coerce(value_type, item, f"{path}.{key}")
            for key, item in value.items()
        }
    if annotation is Any:
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected string")
        return value
    if not isinstance(value, annotation):
        raise ConfigError(f"{path}: expected {annotation}, got {type(value).__name__}")
    return value


def _strict_dataclass(cls: type[T], data: dict[str, Any], path: str) -> T:
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    missing = sorted(
        name for name, field in fields.items()
        if name not in data and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {unknown}")
    if missing:
        raise ConfigError(f"{path}: missing keys: {missing}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _coerce(hints[name], data[name], f"{path}.{name}")
        for name in fields if name in data
    }
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    """Load, strictly type-check, and semantically validate one YAML profile."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Top-level YAML value must be a mapping")
    config = _strict_dataclass(TrainConfig, raw, "config")
    config.validate()
    return config


def write_resolved_config(config: TrainConfig, path: str | Path) -> None:
    """Atomically persist the immutable, fully resolved run configuration."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(output)
