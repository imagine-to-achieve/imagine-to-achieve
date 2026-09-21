"""Critic-free GRPO action-suffix advantages.

The selected algorithm never calls GAE.  Keeping this function separate makes
the configured gamma visible and prevents a critic-free GAE helper from
silently forcing gamma and lambda to one.
"""

from __future__ import annotations

import torch


def grpo_action_suffix_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    dones: torch.Tensor,
    *,
    group_size: int,
    gamma: float,
) -> torch.Tensor:
    """Return group-normalized discounted suffix returns for every chunk.

    Args:
        rewards: Tensor shaped ``[chunks, trajectories]``.
        loss_mask: Boolean tensor with the same shape.
        dones: Boolean tensor shaped ``[chunks + 1, trajectories]``.
        group_size: Number of trajectories normalized together.
        gamma: Discount applied between chunks.
    """

    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [T,B], got {tuple(rewards.shape)}")
    if loss_mask.shape != rewards.shape:
        raise ValueError("loss_mask shape must match rewards")
    if dones.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        # T rewards require T+1 done boundaries.
        raise ValueError("dones must have shape [T+1,B]")
    if group_size < 2 or rewards.shape[1] % group_size:
        raise ValueError("trajectory count must be divisible by group_size >= 2")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards contain NaN or Inf")

    running = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
    suffix_returns = torch.empty_like(rewards)
    for step in reversed(range(rewards.shape[0])):
        continuation = (~dones[step + 1]).to(rewards.dtype)
        running = rewards[step] + float(gamma) * running * continuation
        suffix_returns[step] = running

    num_groups = rewards.shape[1] // group_size
    grouped = suffix_returns.reshape(rewards.shape[0], num_groups, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True)
    advantages = ((grouped - mean) / (std + 1e-6)).reshape_as(rewards)
    return advantages.float() * loss_mask.to(dtype=torch.float32)

