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

"""Action-chain logprob helpers for Cosmos replay."""

from __future__ import annotations

import math

import torch


def gaussian_logprob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """Return diagonal Gaussian logprob with optional element mask."""

    sigma = torch.clamp(sigma.to(device=sample.device, dtype=sample.dtype), min=sigma_min)
    mean = mean.to(device=sample.device, dtype=sample.dtype)
    logprobs = (
        -torch.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((sample - mean) / sigma).pow(2)
    )
    if mask is not None:
        logprobs = logprobs * mask.to(device=sample.device, dtype=sample.dtype)
    return logprobs


def make_fixed_sigma_schedule(
    *,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float | torch.Tensor = 0.2,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """Build a fixed replay sigma schedule shaped [B, J]."""

    if torch.is_tensor(sigma):
        sigma_tensor = sigma.to(device=device, dtype=dtype)
        if sigma_tensor.dim() == 0:
            sigma_tensor = sigma_tensor.expand(batch_size, num_steps)
        elif sigma_tensor.shape == (num_steps,):
            sigma_tensor = sigma_tensor.unsqueeze(0).expand(batch_size, -1)
        elif sigma_tensor.shape != (batch_size, num_steps):
            raise ValueError(
                "action sigma schedule must be scalar, [J], or [B, J], "
                f"got {tuple(sigma_tensor.shape)}."
            )
    else:
        sigma_tensor = torch.full(
            (batch_size, num_steps),
            float(sigma),
            dtype=dtype,
            device=device,
        )
    return torch.clamp(sigma_tensor, min=sigma_min).contiguous()


def validate_action_chain_replay_tensors(
    *,
    action_chains: torch.Tensor,
    action_denoise_timesteps: torch.Tensor,
    action_sigmas: torch.Tensor,
    action_mask: torch.Tensor,
) -> None:
    """Validate chain-logprob-replay tensor shapes."""

    if action_chains.dim() != 3:
        raise ValueError(
            "forward_inputs['action_chains'] must have shape [B, J + 1, D], "
            f"got {tuple(action_chains.shape)}."
        )
    batch_size, chain_len, action_width = action_chains.shape
    num_steps = chain_len - 1
    if num_steps <= 0:
        raise ValueError("forward_inputs['action_chains'] must contain at least one transition.")
    if action_denoise_timesteps.shape != (batch_size, num_steps):
        raise ValueError(
            "forward_inputs['action_denoise_timesteps'] must have shape [B, J], "
            f"got {tuple(action_denoise_timesteps.shape)} for B={batch_size}, J={num_steps}."
        )
    if action_sigmas.shape != (batch_size, num_steps):
        raise ValueError(
            "forward_inputs['action_sigmas'] must have shape [B, J], "
            f"got {tuple(action_sigmas.shape)} for B={batch_size}, J={num_steps}."
        )
    if action_mask.shape != (batch_size, action_width):
        raise ValueError(
            "forward_inputs['action_mask'] must have shape [B, D], "
            f"got {tuple(action_mask.shape)} for B={batch_size}, D={action_width}."
        )


def reduce_action_chain_logprob(
    transition_logprobs: torch.Tensor,
    *,
    action_mask: torch.Tensor,
    reduction: str = "action_dim",
) -> torch.Tensor:
    """Reduce per-transition logprobs [B, J, D] to RLinf-facing logprobs."""

    if transition_logprobs.dim() != 3:
        raise ValueError(
            "transition_logprobs must have shape [B, J, D], "
            f"got {tuple(transition_logprobs.shape)}."
        )
    if action_mask.shape != (
        transition_logprobs.shape[0],
        transition_logprobs.shape[2],
    ):
        raise ValueError(
            "action_mask must have shape [B, D], "
            f"got {tuple(action_mask.shape)} for logprobs {tuple(transition_logprobs.shape)}."
        )

    masked_logprobs = transition_logprobs * action_mask[:, None, :].to(
        device=transition_logprobs.device,
        dtype=transition_logprobs.dtype,
    )
    if reduction == "action_dim":
        return masked_logprobs.sum(dim=1)
    if reduction == "chunk":
        return masked_logprobs.sum(dim=(1, 2))
    raise ValueError(f"Unsupported action chain logprob reduction: {reduction!r}.")
