"""Numerically stable PPO clipped actor loss used by FPO replay scores."""

from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(values.dtype)
    return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def ppo_clipped_actor_loss(
    current_scores: torch.Tensor,
    old_scores: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply PPO clipping to float32 FPO action-head surrogate scores."""

    if not (
        current_scores.shape == old_scores.shape == advantages.shape == loss_mask.shape
    ):
        raise ValueError("scores, advantages and loss_mask must have identical shapes")
    if current_scores.dtype != torch.float32 or old_scores.dtype != torch.float32:
        raise TypeError("current_scores and old_scores must be float32")
    if advantages.dtype != torch.float32:
        raise TypeError("advantages must be float32")
    if clip_ratio_low < 0 or clip_ratio_high < 0:
        raise ValueError("PPO clip ratios must be non-negative")

    log_ratio = current_scores - old_scores
    ratio = torch.where(loss_mask, torch.exp(log_ratio), torch.zeros_like(log_ratio))
    clipped = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * clipped
    element_loss = torch.maximum(unclipped_loss, clipped_loss)
    loss = _masked_mean(element_loss, loss_mask)

    clipped_mask = (unclipped_loss.detach() < clipped_loss.detach()) & loss_mask
    count = loss_mask.count_nonzero().clamp_min(1)
    metrics = {
        "policy_loss": loss.detach(),
        "ratio": _masked_mean(ratio.detach(), loss_mask),
        "ratio_abs": _masked_mean((ratio.detach() - 1.0).abs(), loss_mask),
        "approx_kl": -torch.where(loss_mask, log_ratio.detach(), 0.0).sum() / count,
        "clip_fraction": clipped_mask.count_nonzero().float() / count,
    }
    return loss, metrics

