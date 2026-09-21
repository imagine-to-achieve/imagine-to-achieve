"""Typed contracts crossing rollout, reward, replay, and checkpoint boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # Python 3.9 login-node static checks; compute runtime is 3.11.
    class StrEnum(str, Enum):
        pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by preflight on login nodes
        raise RuntimeError("PyTorch is required for tensor contract validation") from exc
    return torch


@dataclass(frozen=True)
class BatchContract:
    world_size: int
    trajectories_per_update: int
    chunks_per_trajectory: int
    global_minibatch_chunks: int
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    expected_optimizer_steps: int

    def validate(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        expected = (
            self.micro_batch_size_per_gpu
            * self.world_size
            * self.gradient_accumulation_steps
        )
        if self.global_minibatch_chunks != expected:
            raise ValueError(
                "global minibatch mismatch: "
                f"{self.global_minibatch_chunks} != {self.micro_batch_size_per_gpu} "
                f"* {self.world_size} * {self.gradient_accumulation_steps}"
            )
        flattened = self.trajectories_per_update * self.chunks_per_trajectory
        if flattened % self.global_minibatch_chunks:
            raise ValueError("flattened rollout is not divisible by global minibatch")
        if flattened // self.global_minibatch_chunks != self.expected_optimizer_steps:
            raise ValueError("optimizer step count does not match the batch contract")


@dataclass(frozen=True)
class TaskSpec:
    profile: str
    mode: str
    prompt: str
    active_variants: tuple[str, ...]
    train_episode_ids: dict[str, tuple[int, ...]]
    eval_episode_ids: dict[str, tuple[int, ...]]


@dataclass
class CameraBundle:
    main: Any
    wrist: Any
    extra: Any
    layout: str = "droid"

    def validate(self, *, batch_size: int | None = None) -> None:
        torch = _torch()
        tensors = {"main": self.main, "wrist": self.wrist, "extra": self.extra}
        shapes: set[tuple[int, ...]] = set()
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"CameraBundle.{name} must be a torch.Tensor")
            if tensor.ndim not in {4, 5}:
                raise ValueError(f"CameraBundle.{name} must be BCHW or BTCHW")
            if batch_size is not None and tensor.shape[0] != batch_size:
                raise ValueError(f"CameraBundle.{name} batch mismatch")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"CameraBundle.{name} contains NaN or Inf")
            minimum = float(tensor.detach().amin().cpu())
            maximum = float(tensor.detach().amax().cpu())
            if minimum < -1.0001 or maximum > 1.0001:
                raise ValueError(
                    f"CameraBundle.{name} range [{minimum}, {maximum}] is outside [-1, 1]"
                )
            shapes.add(tuple(tensor.shape))
        if len(shapes) != 1:
            raise ValueError(f"CameraBundle views have different shapes: {sorted(shapes)}")
        if self.layout != "droid":
            raise ValueError(f"Unsupported camera layout: {self.layout}")

    def detach_cpu(self) -> "CameraBundle":
        return CameraBundle(
            main=self.main.detach().cpu().contiguous(),
            wrist=self.wrist.detach().cpu().contiguous(),
            extra=self.extra.detach().cpu().contiguous(),
            layout=self.layout,
        )


@dataclass
class ChunkResult:
    observations: Any
    actions: Any
    fpo_noise: Any
    old_scores: Any
    rewards: Any
    imagined_video_chunk: Any
    world_video: CameraBundle
    done: Any
    variant: str

    def validate(self) -> None:
        torch = _torch()
        if self.imagined_video_chunk is None:
            # Replay once omitted this field and failed late in actor replay.
            raise ValueError("imagined_video_chunk is required")
        for name in (
            "observations",
            "actions",
            "fpo_noise",
            "old_scores",
            "rewards",
            "imagined_video_chunk",
            "done",
        ):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"ChunkResult.{name} must be a torch.Tensor")
        if self.actions.ndim != 3:
            raise ValueError("actions must have shape [B, chunk, action_dim]")
        if self.old_scores.shape[0] != self.actions.shape[0]:
            raise ValueError("old_scores batch size must match actions")
        self.world_video.validate(batch_size=int(self.actions.shape[0]))


@dataclass
class Trajectory:
    chunks: tuple[ChunkResult, ...]
    dones: Any

    def validate(self, *, expected_chunks: int) -> None:
        torch = _torch()
        if len(self.chunks) != expected_chunks:
            raise ValueError(f"expected {expected_chunks} chunks, got {len(self.chunks)}")
        if not isinstance(self.dones, torch.Tensor) or self.dones.dtype != torch.bool:
            raise TypeError("dones must be a bool tensor")
        # T chunks require T+1 boundaries for suffix-return recursion.
        if self.dones.shape[0] != expected_chunks + 1:
            raise ValueError("dones must contain T+1 boundaries for T chunks")
        for chunk in self.chunks:
            chunk.validate()


@dataclass
class ReplayBatch:
    observations: Any
    actions: Any
    fpo_noise: Any
    old_scores: Any
    current_scores: Any
    advantages: Any
    loss_mask: Any
    variants: tuple[str, ...]
    semantics_version: str

    def validate(self) -> None:
        torch = _torch()
        if self.semantics_version != "fpo_action_head_chunk_v1":
            raise ValueError(f"unsupported score semantics: {self.semantics_version}")
        tensors = {
            "observations": self.observations,
            "fpo_noise": self.fpo_noise,
            "old_scores": self.old_scores,
            "current_scores": self.current_scores,
            "advantages": self.advantages,
            "loss_mask": self.loss_mask,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"ReplayBatch.{name} must be a tensor")
        if self.old_scores.shape != self.current_scores.shape:
            raise ValueError("old/current score shapes differ")
        if self.old_scores.shape != self.advantages.shape:
            raise ValueError("score/advantage shapes differ")
        if self.loss_mask.shape != self.old_scores.shape:
            raise ValueError("loss mask/score shapes differ")
        if self.old_scores.dtype != torch.float32 or self.current_scores.dtype != torch.float32:
            raise TypeError("FPO replay scores must be float32")
        if not torch.isfinite(self.old_scores).all() or not torch.isfinite(self.current_scores).all():
            raise ValueError("FPO replay scores contain NaN or Inf")


class CheckpointState(StrEnum):
    STAGING = "staging"
    COMPLETED = "completed"
    PREEMPTED = "preempted"
    FAILED = "failed"
