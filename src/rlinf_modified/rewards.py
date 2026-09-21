"""Aligned three-camera video and terminal-goal rewards."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rlinf_modified.contracts import CameraBundle


def _btchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim != 5:
        raise ValueError("camera tensor must be BCHW or BTCHW")
    return tensor


def _resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, time, channels, height, width = video.shape
    resized = functional.interpolate(
        video.reshape(batch * time, channels, height, width).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, time, channels, *size)


def droid_composite(bundle: CameraBundle, size: tuple[int, int]) -> torch.Tensor:
    """Compose wrist on top and main/extra on the bottom like the SFT canvas."""

    bundle.validate()
    main = _btchw(bundle.main)
    wrist = _btchw(bundle.wrist)
    extra = _btchw(bundle.extra)
    height, width = size
    top_height = height // 2
    bottom_height = height - top_height
    wrist_top = _resize(wrist, (top_height, width))
    main_left = _resize(main, (bottom_height, width // 2))
    extra_right = _resize(extra, (bottom_height, width - width // 2))
    return torch.cat((wrist_top, torch.cat((main_left, extra_right), dim=-1)), dim=-2)


def aligned_video_similarity_reward(
    imagined_video: torch.Tensor,
    world_video: CameraBundle,
    *,
    size: tuple[int, int],
    scale: float,
) -> torch.Tensor:
    """Return per-frame negative MSE against the aligned DROID composite."""

    imagined = _btchw(imagined_video)
    reference = droid_composite(world_video, size)
    imagined = _resize(imagined, size)
    if imagined.shape != reference.shape:
        raise ValueError(
            f"imagined/reference video shapes differ: {imagined.shape} != {reference.shape}"
        )
    reward = -(imagined.float() - reference.float()).square().mean(dim=(2, 3, 4))
    reward = reward * float(scale)
    if not torch.isfinite(reward).all():
        raise ValueError("video similarity reward produced NaN or Inf")
    return reward


def terminal_goal_reward(
    world_video: CameraBundle,
    goal: CameraBundle,
    *,
    window_size: int,
    size: tuple[int, int],
    view_weights: tuple[float, float, float],
    scale: float,
) -> torch.Tensor:
    """Return one weighted negative terminal MSE per trajectory."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    world_video.validate()
    goal.validate()
    components = []
    for current, target, weight in zip(
        (world_video.main, world_video.wrist, world_video.extra),
        (goal.main, goal.wrist, goal.extra),
        view_weights,
    ):
        current_video = _resize(_btchw(current)[:, -window_size:], size)
        target_video = _resize(_btchw(target)[:, -window_size:], size)
        components.append(
            -float(weight)
            * (current_video.float() - target_video.float()).square().mean(dim=(1, 2, 3, 4))
        )
    result = sum(components) * float(scale)
    if not torch.isfinite(result).all():
        raise ValueError("terminal goal reward produced NaN or Inf")
    return result


def combine_chunk_rewards(
    video_rewards: torch.Tensor,
    terminal_rewards: torch.Tensor,
    done_mask: torch.Tensor,
) -> torch.Tensor:
    """Add terminal reward only to the last frame of completed trajectories."""

    if video_rewards.ndim != 2:
        raise ValueError("video_rewards must be [B,T]")
    if terminal_rewards.shape != (video_rewards.shape[0],):
        raise ValueError("terminal_rewards must have shape [B]")
    if done_mask.shape != (video_rewards.shape[0],):
        raise ValueError("done_mask must have shape [B]")
    combined = video_rewards.float().clone()
    combined[done_mask, -1] += terminal_rewards[done_mask]
    return combined

