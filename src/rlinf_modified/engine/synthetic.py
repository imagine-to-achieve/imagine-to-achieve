"""End-to-end synthetic FPO/GRPO trainer using production data contracts."""

from __future__ import annotations

import gc
import json
import signal
from pathlib import Path
from typing import Any

import torch

from rlinf_modified.algorithms import (
    grpo_action_suffix_advantages,
    ppo_clipped_actor_loss,
)
from rlinf_modified.checkpoint import AtomicCheckpointer
from rlinf_modified.config import TrainConfig, write_resolved_config
from rlinf_modified.contracts import (
    BatchContract,
    CameraBundle,
    CheckpointState,
    ChunkResult,
    ReplayBatch,
    Trajectory,
)
from rlinf_modified.distributed.telemetry import append_jsonl, memory_snapshot, write_heartbeat
from rlinf_modified.models import SyntheticFlowPolicy, SyntheticWorld
from rlinf_modified.rewards import (
    aligned_video_similarity_reward,
    combine_chunk_rewards,
    terminal_goal_reward,
)
from rlinf_modified.tasks import build_task_spec


class SyntheticTrainer:
    """Run rollout, reward, GRPO, FPO replay, PPO update, and checkpoint."""

    def __init__(self, config: TrainConfig) -> None:
        if config.runtime.backend != "single":
            raise ValueError("SyntheticTrainer requires runtime.backend=single")
        self.config = config
        self.task = build_task_spec(config)
        self.run_dir = Path(config.runtime.output_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(config, self.run_dir / "resolved_config.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.runtime.seed)
        if config.runtime.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.runtime.seed)
        self.policy = SyntheticFlowPolicy(
            observation_dim=16,
            action_chunk=config.cosmos.action_chunk,
            action_dim=config.cosmos.action_dim,
        ).to(self.device)
        self.world = SyntheticWorld(image_size=config.reward.video_similarity.size)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        self.checkpointer = AtomicCheckpointer(self.run_dir)
        self.stop_requested = False
        self._install_signal_handlers()
        BatchContract(
            world_size=config.world_size,
            trajectories_per_update=config.batch.trajectories_per_update,
            chunks_per_trajectory=config.batch.chunks_per_trajectory,
            global_minibatch_chunks=config.batch.global_minibatch_chunks,
            micro_batch_size_per_gpu=config.batch.micro_batch_size_per_gpu,
            gradient_accumulation_steps=config.batch.gradient_accumulation_steps,
            expected_optimizer_steps=config.batch.expected_optimizer_steps,
        ).validate()

    def _install_signal_handlers(self) -> None:
        def request_stop(signum: int, _frame: Any) -> None:
            self.stop_requested = True
            append_jsonl(
                self.run_dir / "events.jsonl",
                {"event": "signal", "signal": signum},
            )

        signal.signal(signal.SIGTERM, request_stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, request_stop)

    def _variant_for_index(self, index: int) -> str:
        variants = self.task.active_variants
        group = index // self.config.algorithm.group_size
        return variants[group % len(variants)]

    @torch.no_grad()
    def _collect(self, update: int) -> Trajectory:
        config = self.config
        batch = config.batch.trajectories_per_update
        chunks: list[ChunkResult] = []
        dones = torch.zeros(
            config.batch.chunks_per_trajectory + 1,
            batch,
            dtype=torch.bool,
            device=self.device,
        )
        for step in range(config.batch.chunks_per_trajectory):
            variants = [self._variant_for_index(index) for index in range(batch)]
            observations = torch.randn(
                batch,
                16,
                generator=self.generator,
                device=self.device,
            )
            observations[:, 1] = torch.tensor(
                [SyntheticWorld.variant_code(item) for item in variants],
                device=self.device,
            )
            actions = self.policy.sample_actions(observations, generator=self.generator)
            fpo_noise = torch.randn(
                batch,
                config.fpo.num_mc_samples,
                config.cosmos.action_chunk,
                config.cosmos.action_dim,
                generator=self.generator,
                device=self.device,
            )
            old_scores = self.policy.fpo_score(observations, actions, fpo_noise)
            imagined_parts = []
            main_parts = []
            wrist_parts = []
            extra_parts = []
            goal_main_parts = []
            goal_wrist_parts = []
            goal_extra_parts = []
            for index, variant in enumerate(variants):
                imagined, world, goal = self.world.render(
                    observations[index : index + 1],
                    actions[index : index + 1],
                    variant=variant,
                    progress=(step + 1) / config.batch.chunks_per_trajectory,
                )
                imagined_parts.append(imagined)
                main_parts.append(world.main)
                wrist_parts.append(world.wrist)
                extra_parts.append(world.extra)
                goal_main_parts.append(goal.main)
                goal_wrist_parts.append(goal.wrist)
                goal_extra_parts.append(goal.extra)
            imagined = torch.cat(imagined_parts)
            world_bundle = CameraBundle(
                main=torch.cat(main_parts),
                wrist=torch.cat(wrist_parts),
                extra=torch.cat(extra_parts),
            )
            goal_bundle = CameraBundle(
                main=torch.cat(goal_main_parts),
                wrist=torch.cat(goal_wrist_parts),
                extra=torch.cat(goal_extra_parts),
            )
            frame_rewards = aligned_video_similarity_reward(
                imagined,
                world_bundle,
                size=config.reward.video_similarity.size,
                scale=config.reward.video_similarity.scale,
            )
            done = torch.full(
                (batch,),
                step == config.batch.chunks_per_trajectory - 1,
                dtype=torch.bool,
                device=self.device,
            )
            terminal = terminal_goal_reward(
                world_bundle,
                goal_bundle,
                window_size=config.reward.terminal_goal.window_size,
                size=config.reward.terminal_goal.size,
                view_weights=config.reward.terminal_goal.view_weights,
                scale=config.reward.terminal_goal.scale,
            )
            combined = combine_chunk_rewards(frame_rewards, terminal, done).sum(dim=1)
            dones[step + 1] = done
            chunk = ChunkResult(
                observations=observations.detach().cpu(),
                actions=actions.detach().cpu(),
                fpo_noise=fpo_noise.detach().cpu(),
                old_scores=old_scores.detach().cpu().float(),
                rewards=combined.detach().cpu().float(),
                imagined_video_chunk=imagined.detach().cpu(),
                world_video=world_bundle.detach_cpu(),
                done=done.detach().cpu(),
                variant="mixed" if len(set(variants)) > 1 else variants[0],
            )
            chunk.validate()
            chunks.append(chunk)
            # Full videos never survive beyond the reward/CPU transfer phase.
            del imagined, world_bundle, goal_bundle, frame_rewards, terminal, combined
        trajectory = Trajectory(chunks=tuple(chunks), dones=dones.detach().cpu())
        trajectory.validate(expected_chunks=config.batch.chunks_per_trajectory)
        return trajectory

    def _make_replay(self, trajectory: Trajectory) -> ReplayBatch:
        rewards = torch.stack([chunk.rewards for chunk in trajectory.chunks])
        loss_mask = torch.ones_like(rewards, dtype=torch.bool)
        advantages = grpo_action_suffix_advantages(
            rewards,
            loss_mask,
            trajectory.dones,
            group_size=self.config.algorithm.group_size,
            gamma=self.config.algorithm.gamma,
        )
        replay = ReplayBatch(
            observations=torch.cat([chunk.observations for chunk in trajectory.chunks]),
            actions=torch.cat([chunk.actions for chunk in trajectory.chunks]),
            fpo_noise=torch.cat([chunk.fpo_noise for chunk in trajectory.chunks]),
            old_scores=torch.cat([chunk.old_scores for chunk in trajectory.chunks]).float(),
            current_scores=torch.cat([chunk.old_scores for chunk in trajectory.chunks]).float(),
            advantages=advantages.reshape(-1).float(),
            loss_mask=loss_mask.reshape(-1),
            variants=tuple(chunk.variant for chunk in trajectory.chunks),
            semantics_version="fpo_action_head_chunk_v1",
        )
        replay.validate()
        return replay

    def _train_replay(self, replay: ReplayBatch, update: int) -> dict[str, float]:
        config = self.config
        self.policy.train()
        torch.set_grad_enabled(True)  # Inference init must not leak no-grad.
        observations = replay.observations.to(self.device)
        actions = replay.actions.to(self.device)
        fpo_noise = replay.fpo_noise.to(self.device)
        old_scores = replay.old_scores.to(self.device)
        advantages = replay.advantages.to(self.device)
        loss_mask = replay.loss_mask.to(self.device)
        with torch.no_grad():
            recomputed_old = self.policy.fpo_score(observations, actions, fpo_noise)
        if not torch.allclose(recomputed_old, old_scores, atol=1e-6, rtol=1e-5):
            # Validate rebased old scores before any optimizer step.
            difference = float((recomputed_old - old_scores).abs().max().cpu())
            raise ValueError(f"old FPO score replay mismatch before update: max_abs={difference}")

        total = observations.shape[0]
        global_batch = config.batch.global_minibatch_chunks
        micro = config.batch.micro_batch_size_per_gpu
        metrics: dict[str, float] = {}
        optimizer_steps = 0
        for start in range(0, total, global_batch):
            stop = start + global_batch
            self.optimizer.zero_grad(set_to_none=True)
            micro_steps = 0
            accumulated: dict[str, float] = {}
            for micro_start in range(start, stop, micro):
                selection = slice(micro_start, micro_start + micro)
                current = self.policy.fpo_score(
                    observations[selection], actions[selection], fpo_noise[selection]
                ).float()
                loss, local_metrics = ppo_clipped_actor_loss(
                    current,
                    old_scores[selection],
                    advantages[selection],
                    loss_mask[selection],
                    clip_ratio_low=config.algorithm.clip_ratio_low,
                    clip_ratio_high=config.algorithm.clip_ratio_high,
                )
                (loss / config.batch.gradient_accumulation_steps).backward()
                micro_steps += 1
                for name, value in local_metrics.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value.cpu())
            if micro_steps != config.batch.gradient_accumulation_steps:
                raise ValueError(
                    f"gradient accumulation mismatch: {micro_steps} != "
                    f"{config.batch.gradient_accumulation_steps}"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), config.optimizer.clip_grad_norm
            )
            self.optimizer.step()
            optimizer_steps += 1
            metrics = {
                name: value / micro_steps for name, value in accumulated.items()
            }
            metrics["grad_norm"] = float(grad_norm.detach().cpu())
        if optimizer_steps != config.batch.expected_optimizer_steps:
            raise ValueError(
                f"optimizer steps mismatch: {optimizer_steps} != "
                f"{config.batch.expected_optimizer_steps}"
            )
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["update"] = float(update)
        return metrics

    def run(self) -> dict[str, Any]:
        start_update = 0
        if self.config.runtime.resume:
            start_update, _ = self.checkpointer.load_latest(self.policy, self.optimizer)
        final_update = self.config.runtime.max_updates
        if not self.config.runtime.continuous_stress:
            final_update = min(final_update, start_update + self.config.runtime.segment_updates)
        self.checkpointer.write_status(
            CheckpointState.STAGING,
            last_completed_update=start_update,
            target_update=final_update,
        )
        for update in range(start_update + 1, final_update + 1):
            write_heartbeat(self.run_dir / "heartbeats", rank=0, phase="rollout", update=update)
            append_jsonl(
                self.run_dir / "telemetry.jsonl",
                {"phase": "before_rollout", "update": update, **memory_snapshot(self.run_dir)},
            )
            trajectory = self._collect(update)
            write_heartbeat(
                self.run_dir / "heartbeats",
                rank=0,
                phase="replay",
                update=update,
                shapes={"dones": tuple(trajectory.dones.shape)},
            )
            replay = self._make_replay(trajectory)
            metrics = self._train_replay(replay, update)
            append_jsonl(self.run_dir / "metrics.jsonl", metrics)
            if self.config.checkpoint.enabled and (
                update % self.config.checkpoint.interval_updates == 0
                or update == final_update
                or self.stop_requested
            ):
                self.checkpointer.save(
                    update=update,
                    model=self.policy,
                    optimizer=self.optimizer,
                    extra={"experiment_name": self.config.experiment_name},
                )
            del replay, trajectory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            append_jsonl(
                self.run_dir / "telemetry.jsonl",
                {"phase": "update_boundary", "update": update, **memory_snapshot(self.run_dir)},
            )
            if self.stop_requested:
                self.checkpointer.write_status(
                    CheckpointState.PREEMPTED,
                    last_completed_update=update,
                )
                return {"state": "preempted", "update": update}
        self.checkpointer.write_status(
            CheckpointState.COMPLETED,
            last_completed_update=final_update,
        )
        summary = {"state": "completed", "update": final_update}
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
