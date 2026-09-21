"""Small deterministic PyTorch backend exercising the production contracts."""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from rlinf_modified.contracts import CameraBundle
from rlinf_modified.rewards import droid_composite


class SyntheticFlowPolicy(nn.Module):
    """Tiny action-head flow model with a differentiable FPO surrogate score."""

    def __init__(self, observation_dim: int, action_chunk: int, action_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_chunk = action_chunk
        self.action_dim = action_dim
        hidden = 64
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_chunk * action_dim),
        )

    def predicted_velocity(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.network(observations.float())
        return output.reshape(-1, self.action_chunk, self.action_dim)

    @torch.no_grad()
    def sample_actions(
        self,
        observations: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        mean = torch.tanh(self.predicted_velocity(observations))
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        return (mean + 0.10 * noise).clamp(-1.0, 1.0)

    def fpo_score(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        fpo_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar negative velocity-CFM MSE per chunk sample."""

        if fpo_noise.ndim != 4:
            raise ValueError("fpo_noise must be [B,MC,chunk,action_dim]")
        if fpo_noise.shape[0] != actions.shape[0] or fpo_noise.shape[2:] != actions.shape[1:]:
            raise ValueError("FPO noise/action shapes are incompatible")
        predicted = self.predicted_velocity(observations).unsqueeze(1)
        target_velocity = fpo_noise.float() - actions.float().unsqueeze(1)
        per_mc = (predicted.float() - target_velocity).square().mean(dim=(2, 3))
        return -per_mc.mean(dim=1).float()


class SyntheticWorld:
    """Deterministic three-camera world used by CPU and single-GPU smoke tests."""

    def __init__(self, *, image_size: tuple[int, int], video_frames: int = 4) -> None:
        self.image_size = image_size
        self.video_frames = video_frames

    @staticmethod
    def variant_code(variant: str) -> float:
        digest = hashlib.sha256(variant.encode("utf-8")).digest()
        return (int.from_bytes(digest[:2], "little") / 65535.0) * 1.5 - 0.75

    def render(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        variant: str,
        progress: float,
    ) -> tuple[torch.Tensor, CameraBundle, CameraBundle]:
        batch = observations.shape[0]
        height, width = self.image_size
        base = torch.tanh(observations[:, :1] + self.variant_code(variant) + progress)
        base = base.reshape(batch, 1, 1, 1, 1)
        timeline = torch.linspace(
            -0.05,
            0.05,
            self.video_frames,
            device=observations.device,
            dtype=observations.dtype,
        ).reshape(1, self.video_frames, 1, 1, 1)

        def view(offset: float) -> torch.Tensor:
            return (base + timeline + offset).expand(
                batch, self.video_frames, 3, height, width
            ).clamp(-1.0, 1.0).contiguous()

        world = CameraBundle(main=view(0.02), wrist=view(-0.03), extra=view(0.05))
        goal_value = torch.full_like(world.main, 0.75)
        goal = CameraBundle(main=goal_value, wrist=goal_value.clone(), extra=goal_value.clone())
        action_error = actions.float().square().mean(dim=(1, 2)).reshape(batch, 1, 1, 1, 1)
        imagined = (droid_composite(world, self.image_size) - 0.05 * action_error).clamp(
            -1.0, 1.0
        )
        return imagined, world, goal

