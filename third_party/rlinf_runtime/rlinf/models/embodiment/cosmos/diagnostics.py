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

"""Cosmos GRPO logging and guardrail diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from rlinf.utils.utils import masked_mean


ACTION_RATIO_EXPLOSION_THRESHOLD = 10.0
ACTION_RATIO_EXPLOSION_PATIENCE = 2


def _broadcast_mask(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device=values.device, dtype=torch.bool)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    return mask


def _valid_values(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    mask = _broadcast_mask(mask, values)
    if mask is None:
        return values.reshape(-1)
    return values[mask]


def _safe_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    valid = _valid_values(values, mask)
    if valid.numel() == 0:
        return values.detach().new_zeros(())
    return valid.detach().float().mean()


def _safe_max(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    valid = _valid_values(values, mask)
    if valid.numel() == 0:
        return values.detach().new_zeros(())
    return valid.detach().float().max()


def validate_action_logprobs_finite(
    *,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> None:
    """Abort actor update when old or new action logprobs are non-finite."""

    if not torch.isfinite(old_logprobs).all():
        raise ValueError("Cosmos action old logprobs contain NaN or Inf; aborting update.")
    if not torch.isfinite(new_logprobs).all():
        raise ValueError("Cosmos action new logprobs contain NaN or Inf; aborting update.")


def compute_action_logprob_diagnostics(
    *,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    entropy: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action ratio/logprob aliases for RLinf metric logging."""

    logp_delta = new_logprobs - old_logprobs
    ratio = torch.exp(logp_delta)
    valid_ratio = _valid_values(ratio, loss_mask).detach().float()
    zero = ratio.detach().new_zeros(())
    ratio_std = valid_ratio.std(unbiased=False) if valid_ratio.numel() else zero
    ratio_min = valid_ratio.min() if valid_ratio.numel() else zero
    metrics = {
        "action/logp_old_mean": _safe_mean(old_logprobs, loss_mask),
        "action/logp_new_mean": _safe_mean(new_logprobs, loss_mask),
        "action/logp_delta_mean": _safe_mean(logp_delta, loss_mask),
        "action/ratio_mean": _safe_mean(ratio, loss_mask),
        "action/ratio_std": ratio_std,
        "action/ratio_min": ratio_min,
        "action/ratio_max": _safe_max(ratio, loss_mask),
    }
    if clip_ratio_low is not None and clip_ratio_high is not None:
        mask = _broadcast_mask(loss_mask, ratio)
        clipped = (ratio < (1.0 - float(clip_ratio_low))) | (
            ratio > (1.0 + float(clip_ratio_high))
        )
        if mask is not None:
            clipped = clipped & mask
            denom = mask.count_nonzero().clamp(min=1)
        else:
            denom = clipped.new_tensor(max(clipped.numel(), 1), dtype=torch.long)
        metrics["action/clip_fraction"] = clipped.float().sum() / denom.float()
    else:
        metrics["action/clip_fraction"] = ratio.detach().new_zeros(())
    if entropy is not None:
        metrics["action/entropy"] = masked_mean(entropy.detach(), loss_mask)
    return metrics


def compute_advantage_logprob_alignment(
    *,
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure whether the first PPO update raises log-probability for positive advantage."""
    delta = new_logprobs.detach() - old_logprobs.detach()
    adv = advantages.detach().to(device=delta.device, dtype=delta.dtype)
    while adv.dim() > delta.dim() and adv.shape[-1] == 1:
        adv = adv.squeeze(-1)
    while adv.dim() < delta.dim():
        adv = adv.unsqueeze(-1)
    adv = adv.expand_as(delta)
    mask = _broadcast_mask(loss_mask, delta)
    if mask is not None:
        delta = delta[mask]
        adv = adv[mask]
    else:
        delta = delta.reshape(-1)
        adv = adv.reshape(-1)
    zero = delta.new_zeros(())
    positive = adv > 0
    negative = adv < 0
    pos_mean = delta[positive].float().mean() if positive.any() else zero
    neg_mean = delta[negative].float().mean() if negative.any() else zero
    directional_mean = (adv.float() * delta.float()).mean()
    delta_std = delta.float().std(unbiased=False)
    centered_adv = adv.float() - adv.float().mean()
    centered_delta = delta.float() - delta.float().mean()
    denom = centered_adv.norm() * centered_delta.norm()
    corr = (centered_adv * centered_delta).sum() / denom if denom > 0 else zero
    nonzero = adv != 0
    alignment = (
        (torch.sign(adv[nonzero]) == torch.sign(delta[nonzero])).float().mean()
        if nonzero.any()
        else zero
    )
    return {
        "action/post_update_adv_positive_logp_delta_mean": pos_mean,
        "action/post_update_adv_negative_logp_delta_mean": neg_mean,
        "action/post_update_adv_directional_mean": directional_mean,
        "action/post_update_logp_delta_std": delta_std,
        "action/post_update_adv_logp_delta_corr": corr,
        "action/post_update_adv_alignment_fraction": alignment,
        "action/post_update_adv_positive_count": positive.float().sum(),
        "action/post_update_adv_negative_count": negative.float().sum(),
    }


def compute_chain_diagnostics(
    forward_inputs: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Compute replay-chain norm and sigma range diagnostics."""

    if not forward_inputs:
        return {}
    action_chains = forward_inputs.get("action_chains")
    action_sigmas = forward_inputs.get("action_sigmas")
    if not torch.is_tensor(action_chains) or not torch.is_tensor(action_sigmas):
        return {}
    return {
        "chain/x_norm_mean": action_chains.detach().float().norm(dim=-1).mean(),
        "chain/sigma_min_max/min": action_sigmas.detach().float().min(),
        "chain/sigma_min_max/max": action_sigmas.detach().float().max(),
    }


def is_action_parameter_name(name: str) -> bool:
    return any(
        token in name
        for token in (
            "action",
            "llm2action",
            "action2llm",
            "native_trainable_proxy",
            "mlp_moe_gen",
            "input_layernorm_moe_gen",
            "post_attention_layernorm_moe_gen",
        )
    )


def is_video_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("video", "vision", "vae", "decoder"))


def _named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    if hasattr(named_parameters, "named_parameters"):
        yield from named_parameters.named_parameters()
    else:
        yield from named_parameters


def _full_tensor_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    # A DTensor's .norm()/.item() silently returns the calling rank's local
    # partial value, not the true global norm, unless gathered first. This is
    # a collective and must be called on every rank of the tensor's mesh.
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _norm_from_tensors(tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not tensors:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.linalg.vector_norm(
        torch.stack(
            [_full_tensor_if_dtensor(tensor).detach().float().norm() for tensor in tensors]
        )
    )


def compute_gradient_diagnostics(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Split gradient norm diagnostics between action and frozen video paths."""

    action_grads = []
    video_grads = []
    fallback_device = torch.device("cpu")
    for name, param in _named_parameters(named_parameters):
        fallback_device = param.device
        if param.grad is None:
            continue
        if is_action_parameter_name(name):
            action_grads.append(param.grad)
        if is_video_parameter_name(name) and not param.requires_grad:
            video_grads.append(param.grad)
    return {
        "grad/action_norm": _norm_from_tensors(action_grads, fallback_device),
        "grad/video_norm_should_be_zero": _norm_from_tensors(video_grads, fallback_device),
    }


def compute_action_param_checksum(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
) -> torch.Tensor:
    """Return a deterministic floating checksum for trainable action parameters."""

    checksum = None
    fallback_device = torch.device("cpu")
    for name, param in _named_parameters(named_parameters):
        fallback_device = param.device
        if not is_action_parameter_name(name):
            continue
        value = param.detach().float().sum()
        checksum = value if checksum is None else checksum + value.to(checksum.device)
    if checksum is None:
        return torch.zeros((), dtype=torch.float32, device=fallback_device)
    return checksum.reshape(())


class ActionRatioExplosionGuard:
    """Track consecutive pre-clip action-ratio explosions."""

    def __init__(
        self,
        *,
        threshold: float = ACTION_RATIO_EXPLOSION_THRESHOLD,
        patience: int = ACTION_RATIO_EXPLOSION_PATIENCE,
    ) -> None:
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.count = 0

    def check(self, ratio_max: torch.Tensor | float) -> None:
        ratio_value = float(ratio_max.detach().cpu().item() if torch.is_tensor(ratio_max) else ratio_max)
        if ratio_value > self.threshold:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.patience:
            raise RuntimeError(
                "Cosmos action pre-clip ratio exceeded threshold repeatedly: "
                f"ratio_max={ratio_value:.6g}, threshold={self.threshold:.6g}, "
                f"patience={self.patience}."
            )


def expected_video_range_for_tensor(video: torch.Tensor) -> tuple[float, float]:
    """Return the expected normalized range for a video tensor boundary."""

    if torch.is_floating_point(video):
        return 0.0, 1.0
    return 0.0, 255.0


def find_video_range_warnings(
    *,
    imagined_video_chunk: torch.Tensor | None,
    ctrl_world_video_chunk: torch.Tensor | None,
) -> list[str]:
    """Return warnings for videos outside their expected raw boundary range."""

    warnings = []
    for name, video in (
        ("imagined_video_chunk", imagined_video_chunk),
        ("ctrl_world_video_chunk", ctrl_world_video_chunk),
    ):
        if not torch.is_tensor(video) or video.numel() == 0:
            continue
        expected_min, expected_max = expected_video_range_for_tensor(video)
        actual_min = float(video.detach().float().min().item())
        actual_max = float(video.detach().float().max().item())
        if actual_min < expected_min or actual_max > expected_max:
            warnings.append(
                f"{name} range [{actual_min}, {actual_max}] is outside expected "
                f"[{expected_min}, {expected_max}]."
            )
    return warnings


def warn_if_frozen_video_grads(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | torch.nn.Module,
) -> list[str]:
    """Return warnings for frozen video-path parameters with nonzero gradients."""

    warnings = []
    for name, param in _named_parameters(named_parameters):
        if not is_video_parameter_name(name) or param.requires_grad or param.grad is None:
            continue
        grad_norm = float(_full_tensor_if_dtensor(param.grad).detach().float().norm().item())
        if grad_norm > 0.0:
            warnings.append(
                f"Frozen video parameter {name!r} received grad norm {grad_norm:.6g}."
            )
    return warnings
