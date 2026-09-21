"""Bridge strict standalone profiles into the vendored Ray/FSDP runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rlinf_modified.config import TrainConfig, write_resolved_config
from rlinf_modified.preflight import repository_root, run_preflight
from rlinf_modified.tasks import build_task_spec


def _list(values: tuple[Any, ...]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _runtime_training_reward_source(source: str) -> str:
    mapping = {
        "ctrl_world_aligned_mse_plus_terminal": "continuous_combined",
        "trajectory_mse": "trajectory_mse",
        "terminal_goal_mse": "terminal_goal_mse",
        "success_binary": "success_binary",
    }
    try:
        return mapping[str(source)]
    except KeyError as error:
        raise ValueError(f"unsupported standalone reward source: {source!r}") from error


def _paired_ctrl_world_overrides(cfg: TrainConfig) -> list[str]:
    """Project one standalone Ctrl-World contract onto train and eval.

    The vendored Hydra configs inherit train and eval from separate nodes. A
    production-only override on just one node can therefore survive config
    composition and fail only when periodic evaluation first resets. Keep
    every policy-facing/runtime-shaping value represented by the standalone
    schema paired here so both modes consume the same model contract.
    """

    runtime = cfg.ctrl_world.runtime
    values = {
        "action_dim": str(runtime.action_dim),
        "policy_action_dim": str(runtime.policy_action_dim),
        "num_history": str(runtime.num_history),
        "text_cond": str(runtime.text_cond).lower(),
        "frame_level_cond": str(runtime.frame_level_cond).lower(),
        "his_cond_zero": str(runtime.his_cond_zero).lower(),
        "chunk": str(runtime.policy_chunk),
        "cosmos_action_fps": str(runtime.policy_action_fps),
        "ctrl_condition_fps": str(runtime.condition_fps),
        "action_fps_resample": str(runtime.resample_actions).lower(),
        "ctrl_world_chunk": str(runtime.resampled_chunk),
        "ctrl_world_internal_rollout": str(runtime.internal_rollout).lower(),
        "ctrl_world_window_frames": str(runtime.window_frames),
        "ctrl_world_window_emit_frames": str(runtime.window_emit_frames),
        "ctrl_world_window_use_lookahead_latent": str(
            runtime.use_lookahead_latent
        ).lower(),
        "ctrl_world_window_history_mode": runtime.history_mode,
        "ctrl_world_history_idx": _list(runtime.history_indices),
        "ctrl_world_history_bank_size": str(runtime.history_bank_size),
        "num_inference_steps": str(runtime.num_inference_steps),
        "guidance_scale": str(runtime.guidance_scale),
        "decode_chunk_size": str(runtime.decode_chunk_size),
        "image_size": _list(runtime.per_view_image_size),
        "policy_image_size": _list(runtime.per_view_image_size),
        "per_view_vae_codec": str(runtime.per_view_vae_codec).lower(),
        "action_to_eef_adapter": runtime.action_to_eef_adapter,
        # Native DROID checkpoints enforce the unresized 480x640 three-view
        # reset contract. Ctrl-World itself remains at image_size.
        "use_raw_reset_policy_views": str(
            cfg.cosmos.action_representation
            == "ur5_eef_relative_10d_droid_native"
        ).lower(),
    }
    if cfg.task.profile == "duck":
        # Frozen Duck Ctrl-World checkpoint contract, verified against its
        # training manifest: model order is wrist/front/side, Cosmos deltas
        # are already metric, and the canonical profile owns history policy.
        # The inherited fixed seed recreated seed 42 for every five-frame
        # denoise call. Null selects the existing group-CRN stream, whose call
        # index advances between windows while retaining fair noise within a
        # GRPO group.
        values.update(
            {
                "model_camera_ids": "[2,0,1]",
                "model_view_order": "[wrist:d435,front:d405,right:d405_1]",
                "main_view_index": "1",
                "wrist_view_index": "0",
                "side_view_index": "2",
                "adapter_translation_gain": "1.0",
                "fixed_denoise_seed": "null",
            }
        )
    elif cfg.task.profile == "push_t":
        # Push-T is a single-task UR5 EEF profile. Reuse the validated
        # [wrist, front, right] Ctrl-World ordering and sparse-window rollout,
        # but use one plain d405 ResNet model instead of Duck routing.
        reward_model = Path(
            cfg.assets.reward_models[cfg.task.active_variants[0]]
        ).expanduser().resolve()
        values.update(
            {
                "model_camera_ids": "[2,0,1]",
                "model_view_order": "[wrist:d435,front:d405,right:d405_1]",
                "main_view_index": "1",
                "wrist_view_index": "0",
                "side_view_index": "2",
                "adapter_translation_gain": "1.0",
                "adapter_rotation_gain": "1.0",
                "fixed_denoise_seed": "null",
                "prompt_override": json.dumps(cfg.task.base_prompt),
                "reward_models": "null",
                "reward_model.type": "ResnetRewModel",
                "reward_model.loader": "rlinf_local",
                "reward_model.from_pretrained": str(reward_model),
                "reward_model.artifact_name": reward_model.name,
                "reward_model.camera_key": "observation.images.d405_rgb",
                "reward_model.ctrl_world_view_index": "1",
                "reward_model.input_view": "main",
                "reward_model.input_size": "[224,224]",
                "reward_model.input_range": "minus_one_one",
                "reward_model.resize_antialias": "true",
                "reward_model.frame_source": "ctrl_world_native",
                "reward_model.threshold": str(cfg.reward.success_classifier.threshold),
                "reward_model.signal": "probability",
                "reward_model.inference_batch_size": "64",
            }
        )
    return [
        f"++env.{mode}.ctrl_world_cfg.{key}={value}"
        for mode in ("train", "eval")
        for key, value in values.items()
    ]


def _shard_aligned_eval_episode_ids(
    episode_ids: tuple[int, ...], shard_degree: int
) -> tuple[int, ...]:
    """Pad internal eval execution without changing the official episode set."""
    official = tuple(sorted(int(value) for value in episode_ids))
    if not official or len(set(official)) != len(official):
        raise ValueError("official evaluation episode IDs must be non-empty and unique")
    if shard_degree <= 0:
        raise ValueError("Cosmos data-parallel shard degree must be positive")
    padding = (-len(official)) % shard_degree
    # Red Duck has 10 official episodes; pad execution to 12 so
    # ranks 8--11 enter Cosmos collectives together, then discard the duplicates.
    return official + tuple(
        official[index % len(official)] for index in range(padding)
    )



class RealTrainer:
    """Execute the proven embodied runner from local vendored sources only."""

    def __init__(self, config: TrainConfig) -> None:
        if config.runtime.backend != "ray_fsdp":
            raise ValueError("RealTrainer requires runtime.backend=ray_fsdp")
        self.config = config
        self.root = repository_root()
        self.run_dir = Path(config.runtime.output_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(config, self.run_dir / "resolved_config.yaml")

    def preflight(self) -> dict[str, Any]:
        return run_preflight(self.config, run_dir=self.run_dir)

    def _overrides(self, normalized_checkpoint_config: str) -> list[str]:
        cfg = self.config
        training_reward_source = _runtime_training_reward_source(cfg.reward.source)
        terminal_reference_mode = (
            "same_episode_last_frame"
            if cfg.reward.terminal_goal.same_episode
            else "fixed"
        )
        vendor_cosmos = self.root / "third_party"
        vendor_ctrl_world = self.root / "third_party" / "ctrl_world"
        segment_end = int(os.environ.get("RLINF_SEGMENT_END", cfg.runtime.max_updates))
        segment_start = int(os.environ.get("RLINF_SEGMENT_START", "0"))
        rollout_steps = cfg.batch.chunks_per_trajectory * cfg.cosmos.action_chunk
        if not 0 <= segment_start < segment_end <= cfg.runtime.max_updates:
            raise ValueError(
                f"invalid segment bounds: start={segment_start}, end={segment_end}, "
                f"max={cfg.runtime.max_updates}"
            )
        overrides = [
            f"runner.max_epochs={cfg.runtime.max_updates}",
            f"runner.max_steps={cfg.runtime.max_updates}",
            # Intermediate segments stop recoverably and never publish global SUCCESS.
            f"++runner.stop_after_updates={segment_end if segment_end < cfg.runtime.max_updates else 0}",
            f"runner.save_interval={cfg.checkpoint.interval_updates}",
            # Save at the configured interval, but reserve "milestone" retention
            # for the final update so checkpoint.keep_latest controls recoveries.
            f"runner.checkpoint_milestone_interval={cfg.runtime.max_updates}",
            f"++runner.checkpoint_keep_latest_recovery={cfg.checkpoint.keep_latest}",
            f"runner.val_check_interval={cfg.evaluation.interval_updates if cfg.evaluation.enabled else -1}",
            f"runner.logger.log_path={self.run_dir.parent}",
            f"runner.logger.experiment_name={self.run_dir.name}",
            f"cluster.num_nodes={cfg.slurm.nodes}",
            f"algorithm.group_size={cfg.algorithm.group_size}",
            f"algorithm.gamma={cfg.algorithm.gamma}",
            f"algorithm.gae_lambda=null",
            f"algorithm.clip_ratio_low={cfg.algorithm.clip_ratio_low}",
            f"algorithm.clip_ratio_high={cfg.algorithm.clip_ratio_high}",
            f"++algorithm.fpo_ratio_granularity={cfg.fpo.ratio_granularity}",
            "++algorithm.diagnostics.fpo_alignment=true",
            f"algorithm.rollout_epoch={cfg.batch.rollout_epochs}",
            f"algorithm.trajectory_chunks={cfg.batch.chunks_per_trajectory}",
            f"++algorithm.cross_rank_group.chunks_per_trajectory={cfg.batch.chunks_per_trajectory}",
            # Lightweight mode records scalar analysis data without
            # retaining videos or requiring the excluded preparation-only audit suite.
            f"algorithm.trajectory_records.enabled={str(cfg.trajectory_records.enabled).lower()}",
            f"++algorithm.trajectory_records.provenance_mode={cfg.trajectory_records.mode}",
            f"algorithm.trajectory_records.output_dir={self.run_dir / 'trajectory_records'}",
            f"algorithm.trajectory_records.expected_trajectories={cfg.batch.trajectories_per_update}",
            f"algorithm.trajectory_records.trajectory_frames={rollout_steps}",
            f"algorithm.cross_rank_group.logical_rounds_per_update={cfg.batch.logical_rounds_per_update}",
            f"algorithm.cross_rank_group.group_slots_per_logical_round={cfg.batch.group_slots_per_logical_round}",
            f"algorithm.cross_rank_group.waves_per_logical_round={cfg.batch.waves_per_logical_round}",
            f"algorithm.cross_rank_group.expected_trajectories_per_update={cfg.batch.trajectories_per_update}",
            f"env.train.total_num_envs={cfg.batch.parallel_envs}",
            f"env.train.group_size={cfg.algorithm.group_size // 4}",
            # Disabled eval still constructs rollout metadata; keep horizons numeric.
            f"env.train.max_episode_steps={rollout_steps}",
            f"env.train.max_steps_per_rollout_epoch={rollout_steps}",
            f"env.eval.max_episode_steps={rollout_steps}",
            f"env.eval.max_steps_per_rollout_epoch={rollout_steps}",
            # Sparse eval must activate complete Cosmos HSDP shard groups.
            f"++rollout.sparse_eval_collective_group_size={cfg.cosmos.data_parallel_shard_degree}",
            f"actor.global_batch_size={cfg.batch.global_minibatch_chunks}",
            f"actor.micro_batch_size={cfg.batch.micro_batch_size_per_gpu}",
            f"exp.optim.global_batch_size={cfg.batch.global_minibatch_chunks}",
            f"exp.optim.micro_batch_size={cfg.batch.micro_batch_size_per_gpu}",
            f"exp.optim.lr={cfg.optimizer.lr}",
            f"++exp.optim.weight_decay={cfg.optimizer.weight_decay}",
            f"++exp.optim.clip_grad={cfg.optimizer.clip_grad_norm}",
            f"actor.optim.lr={cfg.optimizer.lr}",
            f"++actor.optim.adam_beta1={cfg.optimizer.beta1}",
            f"++actor.optim.adam_beta2={cfg.optimizer.beta2}",
            f"++actor.optim.adam_eps={cfg.optimizer.eps}",
            f"++actor.optim.weight_decay={cfg.optimizer.weight_decay}",
            f"++actor.optim.clip_grad={cfg.optimizer.clip_grad_norm}",
            f"actor.rollout_batch_cpu_resident={str(cfg.distributed.replay_cpu_resident).lower()}",
            f"actor.model.model_path={cfg.cosmos.checkpoint_path}",
            f"actor.model.cosmos.checkpoint_path={cfg.cosmos.checkpoint_path}",
            f"actor.model.cosmos.checkpoint_config_path={normalized_checkpoint_config}",
            # This compatibility-only field is absent from legacy Hydra schemas.
            f"++actor.model.cosmos.training_config_path={normalized_checkpoint_config}",
            f"actor.model.cosmos.action_stats_path={cfg.cosmos.action_stats_path}",
            f"++actor.model.cosmos.vlm_processor_path={cfg.cosmos.vlm_processor_path}",
            f"actor.model.cosmos.framework_path={vendor_cosmos}",
            f"actor.model.cosmos.output_dir={self.run_dir / 'cosmos_native'}",
            # Explicitly suppress per-request policy-input audit images for
            # runs configured without automatic analysis.
            f"actor.model.cosmos.input_audit={str(cfg.trajectory_records.auto_analyze).lower()}",
            f"actor.model.cosmos.prompt={json.dumps(cfg.task.base_prompt)}",
            f"actor.model.cosmos.action_representation={cfg.cosmos.action_representation}",
            f"actor.model.cosmos.action_normalization={cfg.cosmos.action_normalization}",
            f"actor.model.cosmos.raw_action_dim={cfg.cosmos.action_dim}",
            f"actor.model.cosmos.max_action_dim={cfg.cosmos.max_action_dim}",
            f"actor.model.cosmos.action_chunk_size={cfg.cosmos.action_chunk}",
            f"actor.model.cosmos.num_steps={cfg.cosmos.sampler_steps}",
            f"actor.model.cosmos.guidance={cfg.cosmos.guidance}",
            f"actor.model.cosmos.replay_objective={cfg.fpo.replay_objective}",
            f"++actor.model.cosmos.fpo_score_parameterization={cfg.fpo.score_parameterization}",
            f"actor.model.cosmos.fpo_num_mc_samples={cfg.fpo.num_mc_samples}",
            f"actor.model.cosmos.fpo_time_distribution={cfg.fpo.time_distribution}",
            f"actor.model.cosmos.fpo_training_shift={cfg.fpo.training_shift}",
            f"actor.model.cosmos.fpo_timestep_scale={cfg.fpo.timestep_scale}",
            f"actor.model.cosmos.fpo_vision_state_channels={cfg.fpo.vision_state_channels}",
            f"actor.model.cosmos.fpo_vision_condition_latent_frames={cfg.fpo.vision_condition_latent_frames}",
            f"actor.model.cosmos.latent_downsample_factor={cfg.fpo.latent_downsample_factor}",
            f"exp.paths.ctrl_world_repo_path={vendor_ctrl_world}",
            f"exp.paths.ctrl_world_ckpt_path={cfg.assets.ctrl_world_checkpoint}",
            f"exp.paths.ctrl_world_stat_path={cfg.assets.ctrl_world_stats}",
            f"exp.paths.initial_image_path={cfg.assets.dataset_path}",
            f"exp.paths.initial_joint_state_path={cfg.assets.dataset_path}",
            f"exp.paths.svd_path={cfg.assets.svd_model_path}",
            f"exp.paths.clip_path={cfg.assets.clip_model_path}",
            # Close's retained env schema derives these paths from
            # ctrl_world_repo_path; pin external assets instead of the vendored code tree.
            f"env.train.ctrl_world_svd_model_path={cfg.assets.svd_model_path}",
            f"env.train.ctrl_world_clip_model_path={cfg.assets.clip_model_path}",
            f"env.eval.ctrl_world_svd_model_path={cfg.assets.svd_model_path}",
            f"env.eval.ctrl_world_clip_model_path={cfg.assets.clip_model_path}",
            f"reward.video_similarity.enabled={str(cfg.reward.video_similarity.enabled).lower()}",
            f"reward.video_similarity.reward_scale={cfg.reward.video_similarity.scale}",
            f"reward.terminal_goal.enabled={str(cfg.reward.terminal_goal.enabled).lower()}",
            f"reward.terminal_goal.reward_scale={cfg.reward.terminal_goal.scale}",
            # The retained legacy runtime carries a task-specific goal dataset
            # path; always replace it with the standalone profile's dataset so
            # Push-T terminal references cannot accidentally come from Nest.
            f"reward.terminal_goal.dataset_path={cfg.assets.dataset_path}",
            f"reward.terminal_goal.window_size={cfg.reward.terminal_goal.window_size}",
            f"++reward.terminal_goal.reference_mode={terminal_reference_mode}",
            f"++reward.terminal_goal.strict_same_episode={str(cfg.reward.terminal_goal.same_episode).lower()}",
            f"reward.terminal_goal.dataset_frame_index={cfg.reward.terminal_goal.frame_index}",
            f"reward.terminal_goal.view_weights.main={cfg.reward.terminal_goal.view_weights[0]}",
            f"reward.terminal_goal.view_weights.wrist={cfg.reward.terminal_goal.view_weights[1]}",
            f"reward.terminal_goal.view_weights.extra={cfg.reward.terminal_goal.view_weights[2]}",
            # Keep the proven video-similarity execution path active for all
            # ablations so trajectory MSE, terminal MSE, and classifier output
            # are measured on identical rollouts. ``training_source`` selects
            # the sole component that reaches GRPO advantages.
            "reward.source=video_similarity",
            f"++reward.training_source={training_reward_source}",
            f"reward.success_classifier.diagnostic_only={str(cfg.reward.success_classifier.diagnostic_only).lower()}",
            f"reward.success_classifier.replace_training_reward={str(cfg.reward.success_classifier.replace_training_reward).lower()}",
            "++reward.success_classifier.reward_value=1.0",
            f"reward.success_classifier.threshold={cfg.reward.success_classifier.threshold}",
            f"++reward.success_classifier.aggregation={cfg.reward.success_classifier.aggregation}",
            "env.train.video_cfg.save_video=false",
            "env.train.video_cfg.save_all_trajectories=false",
            "env.train.video_cfg.stream_all_cosmos_comparison=false",
            "env.train.video_cfg.save_worst_rollout=false",
            f"env.eval.video_cfg.save_video={str(cfg.evaluation.save_video).lower()}",
            f"exp.ctrl_world.num_inference_steps={cfg.ctrl_world.runtime.num_inference_steps}",
        ]
        if cfg.reward.video_similarity.view_weights is not None:
            # Trajectory keys are physical. The terminal legacy tuple above
            # instead has wrist/main/side order.
            for physical, runtime_name in (("main", "main"), ("wrist", "wrist"), ("side", "extra")):
                overrides.append(
                    f"++reward.video_similarity.view_weights.{runtime_name}="
                    f"{cfg.reward.video_similarity.view_weights[physical]}"
                )
        if cfg.evaluation.enabled:
            for node in ("duck.evaluation", "post_update_evaluation"):
                overrides.extend([
                    f"++{node}.fixed_seeds={str(cfg.evaluation.fixed_seeds).lower()}",
                    f"++{node}.before_training={str(cfg.evaluation.before_training).lower()}",
                    f"++{node}.seed_base={cfg.runtime.seed}",
                ])
        overrides.extend(_paired_ctrl_world_overrides(cfg))
        # Training-time classifier and reset-scheduler settings must not depend on
        # whether post-update evaluation is enabled. Keeping these overrides in
        # the evaluation branch made evaluation=false fall back to the legacy
        # batch-512 defaults (8 groups per color) for a batch-128 run (2 per
        # color), which fails before the first rollout.
        for variant in cfg.task.active_variants:
            reward_model = Path(cfg.assets.reward_models[variant]).resolve()
            # Lightweight runs intentionally skip asset-integrity gates. Keep
            # the identity field when configured, but do not make a checksum
            # entry a prerequisite for launching the rollout.
            reward_sha256 = cfg.assets.expected_sha256.get(
                f"reward_model_{variant}", "unverified"
            )
            overrides.extend(
                [
                    f"++reward.success_models.{variant}.from_pretrained={reward_model}",
                    f"++reward.success_models.{variant}.artifact_name={reward_model.name}",
                    f"++reward.success_models.{variant}.sha256={reward_sha256}",
                ]
            )
        if cfg.task.profile in {"duck", "nest_four_cups", "push_t"}:
            groups_per_update = (
                cfg.batch.trajectories_per_update // cfg.algorithm.group_size
            )
            groups_per_color = groups_per_update // len(cfg.task.active_variants)
            task_spec = build_task_spec(cfg)
            training_ids = tuple(
                episode
                for variant in cfg.task.active_variants
                for episode in task_spec.train_episode_ids[variant]
            )
            evaluation_ids = tuple(
                episode
                for variant in cfg.task.active_variants
                for episode in task_spec.eval_episode_ids[variant]
            )
            overrides.extend(
                [
                    f"duck.colors={_list(cfg.task.active_variants)}",
                    f"duck.topology.nodes={cfg.slurm.nodes}",
                    f"duck.topology.gpus_per_node={cfg.slurm.gpus_per_node}",
                    f"duck.topology.total_gpus={cfg.world_size}",
                    # The retained manifest backend uses the legacy `duck`
                    # namespace for every episode-partitioned real task.
                    f"++runner.checkpoint_expected_distcp_shards={cfg.world_size}",
                    f"duck.training.groups_per_update={groups_per_update}",
                    f"duck.training.episodes_per_color_per_update={groups_per_color}",
                    f"algorithm.cross_rank_group.episodes_per_color_per_update={groups_per_color}",
                    f"algorithm.cross_rank_group.color_order={_list(cfg.task.active_variants)}",
                    f"duck.split_manifest={cfg.task.split_manifest_path}",
                    f"algorithm.trajectory_records.split_manifest={cfg.task.split_manifest_path}",
                    f"++env.train.episode_manifest_path={cfg.task.split_manifest_path}",
                    "++env.train.episode_manifest_split=training",
                    f"++env.train.exclude_reset_ids={_list(evaluation_ids)}",
                    f"++env.eval.episode_manifest_path={cfg.task.split_manifest_path}",
                    "++env.eval.episode_manifest_split=evaluation",
                    f"++env.eval.specific_reset_id={_list(evaluation_ids)}",
                ]
            )
        if cfg.task.profile == "duck":
            final_frame_dir = self.run_dir / "rollout_final_side_frames"
            validation_frame_dir = final_frame_dir / "validation"
            success_model_registry = Path(
                "/path/to/shared/checkpoint/"
                "reward_model/outputs/duck_threeway_soft_boundary_rm/"
                "model_registry.json"
            )
            overrides.extend(
                [
                    "++algorithm.trajectory_records.save_final_side_frames=true",
                    f"++algorithm.trajectory_records.expected_train_records_per_step={cfg.batch.trajectories_per_update}",
                    f"++algorithm.trajectory_records.expected_validation_records_per_step={cfg.task.eval_episodes_per_variant * len(cfg.task.active_variants)}",
                    "++algorithm.trajectory_records.final_side_frame_archive_format=npz_uint8_rgb_v1",
                    "++algorithm.trajectory_records.final_side_frame_sha256_scope=contiguous_uint8_rgb_bytes",
                    "++algorithm.trajectory_records.final_side_frame_camera_key=observation.images.d405_1_rgb",
                    "++algorithm.trajectory_records.final_side_frame_ctrl_world_view_index=2",
                    "++algorithm.trajectory_records.success_decision_rule=legacy_max_window",
                    "++algorithm.trajectory_records.success_decision_window_frames=4",
                    f"++algorithm.trajectory_records.success_decision_threshold={cfg.reward.success_classifier.threshold}",
                    "++algorithm.trajectory_records.success_decision_threshold_operator_override=true",
                    f"++algorithm.trajectory_records.success_model_registry_path={success_model_registry}",
                    "++algorithm.trajectory_records.success_model_registry_sha256=c1bcec36a5a75a9d83c84166b4f617c844219700b2a0b0238b230ac825639267",
                    "++algorithm.trajectory_records.success_model_registry_thresholds.brown=0.9369907379150391",
                    "++algorithm.trajectory_records.success_model_registry_thresholds.red=0.9986948370933533",
                    "++algorithm.trajectory_records.success_model_registry_thresholds.white=0.997776448726654",
                    "++algorithm.trajectory_records.success_model_registry_thresholds.yellow=0.9998925924301147",
                    "++algorithm.trajectory_records.save_mse_extreme_final_frames=true",
                    f"++algorithm.trajectory_records.final_side_frame_dir={final_frame_dir}",
                    "++env.train.ctrl_world_cfg.reward_model.camera_key=observation.images.d405_1_rgb",
                    "++env.train.ctrl_world_cfg.reward_model.ctrl_world_view_index=2",
                    "++env.train.ctrl_world_cfg.reward_model.require_camera_metadata=true",
                    "++env.eval.ctrl_world_cfg.reward_model.camera_key=observation.images.d405_1_rgb",
                    "++env.eval.ctrl_world_cfg.reward_model.ctrl_world_view_index=2",
                    "++env.eval.ctrl_world_cfg.reward_model.require_camera_metadata=true",
                    "++duck.evaluation.save_final_side_frames=true",
                    "++duck.evaluation.save_mse_extreme_final_frames=true",
                    f"++duck.evaluation.final_side_frame_dir={validation_frame_dir}",
                ]
            )
        if cfg.evaluation.enabled:
            task_spec = build_task_spec(cfg)
            evaluation_ids = tuple(
                episode
                for variant in cfg.task.active_variants
                for episode in task_spec.eval_episode_ids[variant]
            )
            execution_ids = _shard_aligned_eval_episode_ids(
                evaluation_ids,
                cfg.cosmos.data_parallel_shard_degree,
            )
            if cfg.task.profile in {"duck", "nest_four_cups", "push_t"}:
                overrides.extend(
                    [
                        f"duck.evaluation.episode_ids={_list(evaluation_ids)}",
                        f"duck.evaluation.expected_records_per_step={len(evaluation_ids)}",
                        f"++duck.evaluation.execution_episode_ids={_list(execution_ids)}",
                        f"++duck.evaluation.data_parallel_shard_degree={cfg.cosmos.data_parallel_shard_degree}",
                        f"duck.evaluation.interval_updates={cfg.evaluation.interval_updates}",
                        f"env.eval.total_num_envs={len(execution_ids)}",
                        f"exp.evaluation.total_num_envs={len(execution_ids)}",
                    ]
                )
            else:
                if cfg.task.profile != "close":
                    raise ValueError(
                        "post-update real evaluation is unsupported for profile "
                        f"{cfg.task.profile!r}"
                    )
                record_dir = self.run_dir / "evaluation_records"
                overrides.extend(
                    [
                        "++post_update_evaluation.enabled=true",
                        f"++post_update_evaluation.variants={_list(cfg.task.active_variants)}",
                        f"++post_update_evaluation.episode_ids={_list(evaluation_ids)}",
                        f"++post_update_evaluation.expected_records_per_step={len(evaluation_ids)}",
                        f"++post_update_evaluation.execution_episode_ids={_list(execution_ids)}",
                        f"++post_update_evaluation.data_parallel_shard_degree={cfg.cosmos.data_parallel_shard_degree}",
                        f"++post_update_evaluation.interval_updates={cfg.evaluation.interval_updates}",
                        f"++post_update_evaluation.seed_base={cfg.runtime.seed}",
                        f"++post_update_evaluation.record_dir={record_dir}",
                        f"env.eval.specific_reset_id={_list(execution_ids)}",
                        "env.eval.group_size=1",
                        f"env.eval.total_num_envs={len(execution_ids)}",
                        f"exp.evaluation.total_num_envs={len(execution_ids)}",
                    ]
                )
        else:
            overrides.extend(
                ["env.eval.total_num_envs=0", "exp.evaluation.total_num_envs=0"]
            )
        if segment_start:
            resume_dir = self.run_dir / "checkpoints" / f"global_step_{segment_start}"
            if not resume_dir.is_dir():
                raise FileNotFoundError(
                    f"segment {segment_start}:{segment_end} requires checkpoint {resume_dir}"
                )
            # Dependency success alone is insufficient; verify exact recovery state.
            overrides.append(f"runner.resume_dir={resume_dir}")
        return overrides

    def run(self) -> dict[str, Any]:
        report = self.preflight()
        entry = self.root / "third_party" / "rlinf_runtime" / "examples" / "embodiment" / "train_embodied_agent.py"
        config_dir = entry.parent / "config"
        command = [
            sys.executable,
            str(entry),
            "--config-path",
            str(config_dir),
            "--config-name",
            str(self.config.assets.legacy_config_name),
            *self._overrides(report["normalized_checkpoint_config"]),
        ]
        env = os.environ.copy()
        python_paths = [
            str(self.root / "third_party" / "rlinf_runtime"),
            str(self.root / "third_party"),
            str(self.root / "third_party" / "ctrl_world"),
            str(self.root / "src"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(python_paths + [env.get("PYTHONPATH", "")])
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        # Use the non-deprecated PyTorch 2.10 flight-recorder variable.
        env["TORCH_FR_BUFFER_SIZE"] = env.get("TORCH_FR_BUFFER_SIZE", "1048576")
        env["RLINF_NODE_RANK"] = env.get("RLINF_NODE_RANK", env.get("SLURM_PROCID", "0"))
        completed = subprocess.run(
            command,
            cwd=entry.parent,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            result = {"state": "failed", "returncode": completed.returncode}
            status_path = self.run_dir / "status.json"
            try:
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                status.update(result)
                status["failure_source"] = "vendored_training_process"
                temporary = status_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(status, indent=2) + "\n")
                temporary.replace(status_path)
            except (OSError, ValueError) as error:
                print(f"Could not persist failed training status: {error}", file=sys.stderr)
            (self.run_dir / "bridge_result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"vendored real training exited with code {completed.returncode}"
            )
        result: dict[str, Any] = {"state": "completed", "returncode": 0}
        if self.config.trajectory_records.auto_analyze:
            from rlinf_modified.analysis import analyze_run

            # Publish CSV/report/curves from each atomically
            # committed segment, so analysis never depends on ad-hoc log parsing.
            analysis = analyze_run(
                self.run_dir,
                require_plots=self.config.trajectory_records.require_plots,
            )
            result["analysis"] = {
                "complete_steps": analysis["complete_steps"],
                "output_dir": str(self.run_dir / "analysis"),
            }
        (self.run_dir / "bridge_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    def validate_composition(self) -> dict[str, Any]:
        """Resolve the complete Hydra job on an aarch64 node without starting Ray."""

        report = self.preflight()
        entry = (
            self.root
            / "third_party"
            / "rlinf_runtime"
            / "examples"
            / "embodiment"
            / "train_embodied_agent.py"
        )
        config_dir = entry.parent / "config"
        command = [
            sys.executable,
            str(entry),
            "--config-path",
            str(config_dir),
            "--config-name",
            str(self.config.assets.legacy_config_name),
            "--cfg",
            "job",
            "--resolve",
            *self._overrides(report["normalized_checkpoint_config"]),
        ]
        env = os.environ.copy()
        python_paths = [
            str(self.root / "third_party" / "rlinf_runtime"),
            str(self.root / "third_party"),
            str(self.root / "third_party" / "ctrl_world"),
            str(self.root / "src"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(
            python_paths + [env.get("PYTHONPATH", "")]
        )
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        completed = subprocess.run(
            command,
            cwd=entry.parent,
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Hydra configuration resolution exited with code {completed.returncode}"
            )
        return {
            "state": "valid",
            "backend": "ray_fsdp",
            "hydra_config": self.config.assets.legacy_config_name,
        }
