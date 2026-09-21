# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow Policy Optimization helpers for native Cosmos joint latents.

Cosmos samples one flattened ``[vision | action]`` rectified-flow state.  The
FPO objective used by RLinf deliberately keeps that joint model input (so the
network sees the same noising geometry as SFT) while reducing the CFM loss to
the raw action head.  It is therefore an action-head surrogate for the joint
policy, not an exact marginal action likelihood.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


COSMOS_REPLAY_OBJECTIVE_EULER_GAUSSIAN = "euler_gaussian_chain"
COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD = "fpo_action_head"
SUPPORTED_COSMOS_REPLAY_OBJECTIVES = frozenset(
    {
        COSMOS_REPLAY_OBJECTIVE_EULER_GAUSSIAN,
        COSMOS_REPLAY_OBJECTIVE_FPO_ACTION_HEAD,
    }
)


def resolve_cosmos_replay_objective(value: object | None) -> str:
    """Return a canonical objective name and reject silent fallbacks."""

    objective = str(
        COSMOS_REPLAY_OBJECTIVE_EULER_GAUSSIAN if value is None else value
    ).strip().lower()
    if objective not in SUPPORTED_COSMOS_REPLAY_OBJECTIVES:
        raise ValueError(
            f"Unsupported cosmos.replay_objective={objective!r}; expected one "
            f"of {sorted(SUPPORTED_COSMOS_REPLAY_OBJECTIVES)}."
        )
    return objective


@dataclass(frozen=True)
class FPOJointLayout:
    """Resolved flattened Cosmos joint-latent layout."""

    vision_width: int
    action_width: int
    vision_channels: int
    vision_latent_frames: int
    vision_latent_height: int
    vision_latent_width: int


def _image_height_width(image: torch.Tensor) -> tuple[int, int]:
    if image.ndim != 4:
        raise ValueError(
            "FPO condition image must be batched CHW or HWC, got "
            f"{tuple(image.shape)}."
        )
    if image.shape[-1] in (1, 3, 4):
        return int(image.shape[1]), int(image.shape[2])
    if image.shape[1] in (1, 3, 4):
        return int(image.shape[2]), int(image.shape[3])
    raise ValueError(
        "FPO condition image has no recognizable channel axis: "
        f"{tuple(image.shape)}."
    )


def resolve_fpo_joint_layout(
    clean_joint_latent: torch.Tensor,
    condition_image: torch.Tensor,
    *,
    num_action_chunks: int,
    max_action_dim: int,
    action_state_rows: int = 0,
    latent_downsample_factor: int = 16,
    vision_state_channels: int = 48,
) -> FPOJointLayout:
    """Resolve the native ``[C,T,H,W | action]`` flattening without guessing."""

    if clean_joint_latent.ndim != 2:
        raise ValueError(
            "FPO clean joint latent must have shape [B, width], got "
            f"{tuple(clean_joint_latent.shape)}."
        )
    if int(condition_image.shape[0]) != int(clean_joint_latent.shape[0]):
        raise ValueError(
            "FPO clean joint latent and condition image batch sizes differ: "
            f"{clean_joint_latent.shape[0]} != {condition_image.shape[0]}."
        )
    factor = int(latent_downsample_factor)
    channels = int(vision_state_channels)
    if factor <= 0 or channels <= 0:
        raise ValueError(
            "FPO latent_downsample_factor and vision_state_channels must be positive."
        )
    image_h, image_w = _image_height_width(condition_image)
    if image_h % factor or image_w % factor:
        raise ValueError(
            "FPO condition image dimensions must be divisible by the latent "
            f"downsample factor: image={(image_h, image_w)}, factor={factor}."
        )
    latent_h, latent_w = image_h // factor, image_w // factor
    action_rows = int(num_action_chunks) + int(action_state_rows)
    action_width = action_rows * int(max_action_dim)
    vision_width = int(clean_joint_latent.shape[-1]) - action_width
    spatial_width = latent_h * latent_w
    denominator = channels * spatial_width
    if action_rows <= 0 or int(max_action_dim) <= 0:
        raise ValueError("FPO action rows and max_action_dim must be positive.")
    if vision_width <= 0 or vision_width % denominator:
        raise ValueError(
            "FPO joint latent width is incompatible with the declared Cosmos "
            "vision layout: "
            f"joint_width={clean_joint_latent.shape[-1]}, "
            f"action_width={action_width}, channels={channels}, "
            f"latent_grid={latent_h}x{latent_w}."
        )
    latent_frames = vision_width // denominator
    return FPOJointLayout(
        vision_width=vision_width,
        action_width=action_width,
        vision_channels=channels,
        vision_latent_frames=latent_frames,
        vision_latent_height=latent_h,
        vision_latent_width=latent_w,
    )


def build_fpo_joint_noise_mask(
    clean_joint_latent: torch.Tensor,
    condition_image: torch.Tensor,
    *,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int = 0,
    latent_downsample_factor: int = 16,
    vision_state_channels: int = 48,
    vision_condition_latent_frames: int = 1,
) -> torch.Tensor:
    """Build the joint SFT-compatible noising mask.

    Future vision latents and raw generated action coordinates are one.  The
    observed vision prefix, optional state row, and padded action coordinates
    are zero and therefore remain clean during FPO re-noising.
    """

    layout = resolve_fpo_joint_layout(
        clean_joint_latent,
        condition_image,
        num_action_chunks=num_action_chunks,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
        latent_downsample_factor=latent_downsample_factor,
        vision_state_channels=vision_state_channels,
    )
    raw_dim = int(raw_action_dim)
    max_dim = int(max_action_dim)
    condition_frames = int(vision_condition_latent_frames)
    if raw_dim <= 0 or raw_dim > max_dim:
        raise ValueError(
            f"FPO raw_action_dim must lie in [1, {max_dim}], got {raw_dim}."
        )
    if condition_frames < 0 or condition_frames >= layout.vision_latent_frames:
        raise ValueError(
            "FPO vision_condition_latent_frames must leave at least one future "
            f"latent frame, got {condition_frames} of "
            f"{layout.vision_latent_frames}."
        )

    batch_size = int(clean_joint_latent.shape[0])
    mask = torch.zeros_like(clean_joint_latent)
    vision_mask = mask[:, : layout.vision_width].reshape(
        batch_size,
        layout.vision_channels,
        layout.vision_latent_frames,
        layout.vision_latent_height,
        layout.vision_latent_width,
    )
    vision_mask[:, :, condition_frames:, :, :] = 1
    action_rows = int(num_action_chunks) + int(action_state_rows)
    action_mask = mask[:, layout.vision_width :].reshape(
        batch_size, action_rows, max_dim
    )
    action_mask[
        :, int(action_state_rows) : int(action_state_rows) + int(num_action_chunks), :raw_dim
    ] = 1
    return mask.contiguous()


def extract_fpo_normalized_action(
    clean_joint_latent: torch.Tensor,
    *,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int = 0,
) -> torch.Tensor:
    """Extract normalized raw action coordinates from a final joint latent."""

    if clean_joint_latent.ndim != 2:
        raise ValueError(
            "FPO clean joint latent must have shape [B, width], got "
            f"{tuple(clean_joint_latent.shape)}."
        )
    action_rows = int(num_action_chunks) + int(action_state_rows)
    max_dim = int(max_action_dim)
    raw_dim = int(raw_action_dim)
    action_width = action_rows * max_dim
    if raw_dim <= 0 or raw_dim > max_dim:
        raise ValueError(
            f"FPO raw_action_dim must lie in [1, {max_dim}], got {raw_dim}."
        )
    if int(clean_joint_latent.shape[-1]) < action_width:
        raise ValueError(
            "FPO clean joint latent is narrower than the configured action "
            f"suffix: {clean_joint_latent.shape[-1]} < {action_width}."
        )
    action = clean_joint_latent[:, -action_width:].reshape(
        clean_joint_latent.shape[0], action_rows, max_dim
    )
    return action[
        :, int(action_state_rows) : int(action_state_rows) + int(num_action_chunks), :raw_dim
    ].reshape(clean_joint_latent.shape[0], -1).contiguous()


def fpo_action_velocity_mse(
    predicted_joint_velocity: torch.Tensor,
    epsilon_joint: torch.Tensor,
    clean_joint_latent: torch.Tensor,
    *,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int = 0,
) -> torch.Tensor:
    """Return one mean velocity-CFM loss per sample on raw action channels."""

    if predicted_joint_velocity.shape != clean_joint_latent.shape:
        raise ValueError(
            "FPO predicted velocity and clean joint latent shapes differ: "
            f"{tuple(predicted_joint_velocity.shape)} != "
            f"{tuple(clean_joint_latent.shape)}."
        )
    if epsilon_joint.shape != clean_joint_latent.shape:
        raise ValueError(
            "FPO epsilon and clean joint latent shapes differ: "
            f"{tuple(epsilon_joint.shape)} != {tuple(clean_joint_latent.shape)}."
        )
    pred_action = extract_fpo_normalized_action(
        predicted_joint_velocity,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
    )
    epsilon_action = extract_fpo_normalized_action(
        epsilon_joint,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
    )
    clean_action = extract_fpo_normalized_action(
        clean_joint_latent,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
    )
    target_action_velocity = epsilon_action.float() - clean_action.float()
    return (pred_action.float() - target_action_velocity).square().mean(dim=-1)


def fpo_action_epsilon_mse(
    predicted_joint_velocity: torch.Tensor,
    epsilon_joint: torch.Tensor,
    clean_joint_latent: torch.Tensor,
    sigma: torch.Tensor,
    *,
    num_action_chunks: int,
    raw_action_dim: int,
    max_action_dim: int,
    action_state_rows: int = 0,
) -> torch.Tensor:
    """Return epsilon-prediction MSE implied by the linear CFM velocity.

    For x_sigma = x_0 + sigma * (epsilon - x_0), the implied prediction is
    epsilon_hat = x_sigma + (1 - sigma) * v_hat. Consequently its error is
    exactly (1 - sigma) times the velocity error on every action coordinate.
    """

    sigma_per_sample = sigma.float().reshape(sigma.shape[0], -1)
    if sigma_per_sample.shape[1] != 1:
        raise ValueError(
            "FPO epsilon MSE requires one scalar sigma per sample, got "
            f"{tuple(sigma.shape)}."
        )
    velocity_mse = fpo_action_velocity_mse(
        predicted_joint_velocity,
        epsilon_joint,
        clean_joint_latent,
        num_action_chunks=num_action_chunks,
        raw_action_dim=raw_action_dim,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
    )
    return velocity_mse * (1.0 - sigma_per_sample[:, 0]).square()


def fpo_vision_velocity_mse(
    predicted_joint_velocity: torch.Tensor,
    epsilon_joint: torch.Tensor,
    clean_joint_latent: torch.Tensor,
    condition_image: torch.Tensor,
    *,
    num_action_chunks: int,
    max_action_dim: int,
    action_state_rows: int = 0,
    latent_downsample_factor: int = 16,
    vision_state_channels: int = 48,
    vision_condition_latent_frames: int = 1,
) -> torch.Tensor:
    """Return one velocity-CFM loss per sample on future vision latents.

    This is a diagnostic for the joint-policy/action-head surrogate gap. It
    deliberately excludes the observed vision prefix and every action
    coordinate, so adding it cannot alter the action-only FPO objective.
    """

    if predicted_joint_velocity.shape != clean_joint_latent.shape:
        raise ValueError(
            "FPO predicted velocity and clean joint latent shapes differ: "
            f"{tuple(predicted_joint_velocity.shape)} != "
            f"{tuple(clean_joint_latent.shape)}."
        )
    if epsilon_joint.shape != clean_joint_latent.shape:
        raise ValueError(
            "FPO epsilon and clean joint latent shapes differ: "
            f"{tuple(epsilon_joint.shape)} != {tuple(clean_joint_latent.shape)}."
        )
    layout = resolve_fpo_joint_layout(
        clean_joint_latent,
        condition_image,
        num_action_chunks=num_action_chunks,
        max_action_dim=max_action_dim,
        action_state_rows=action_state_rows,
        latent_downsample_factor=latent_downsample_factor,
        vision_state_channels=vision_state_channels,
    )
    condition_frames = int(vision_condition_latent_frames)
    if condition_frames < 0 or condition_frames >= layout.vision_latent_frames:
        raise ValueError(
            "FPO vision_condition_latent_frames must leave at least one future "
            f"latent frame, got {condition_frames} of "
            f"{layout.vision_latent_frames}."
        )
    batch_size = int(clean_joint_latent.shape[0])
    shape = (
        batch_size,
        layout.vision_channels,
        layout.vision_latent_frames,
        layout.vision_latent_height,
        layout.vision_latent_width,
    )
    predicted_vision = predicted_joint_velocity[:, : layout.vision_width].reshape(
        shape
    )
    target_vision = (
        epsilon_joint[:, : layout.vision_width].float()
        - clean_joint_latent[:, : layout.vision_width].float()
    ).reshape(shape)
    residual = (
        predicted_vision[:, :, condition_frames:].float()
        - target_vision[:, :, condition_frames:]
    )
    return residual.square().flatten(start_dim=1).mean(dim=1)


def _fpo_hash_pair(sampling_seed: int, mc_index: int) -> tuple[int, float]:
    payload = f"{int(sampling_seed)}:{int(mc_index)}".encode("ascii")
    digest = hashlib.blake2b(
        payload, digest_size=16, person=b"rlinf-fpo-v1"
    ).digest()
    noise_seed = int.from_bytes(digest[:8], byteorder="little", signed=True)
    uniform_bits = int.from_bytes(digest[8:], byteorder="little", signed=False)
    uniform = (uniform_bits + 0.5) / float(1 << 64)
    return noise_seed, uniform


def _sample_base_time(uniform: float, distribution: str) -> float:
    """Mirror Cosmos train-time sampling before scheduler shift."""

    if distribution == "uniform":
        sampled_train_time = uniform
    elif distribution == "waver":
        sampled_train_time = 1.0 - uniform - 1.29 * (
            math.cos(math.pi * 0.5 * uniform) ** 2 - 1.0 + uniform
        )
    else:
        raise ValueError(
            "FPO time distribution must be explicitly supported; expected "
            f"'uniform' or 'waver', got {distribution!r}."
        )
    # Cosmos training converts the sampler output with t = 1 - t_raw.
    return min(max(1.0 - sampled_train_time, 1.0e-6), 1.0 - 1.0e-6)


def build_fpo_mc_metadata(
    sampling_seed: int,
    *,
    num_mc_samples: int,
    time_distribution: str,
    training_shift: float,
    timestep_scale: float,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Build fixed MC probes shared by old and current policy evaluations."""

    nmc = int(num_mc_samples)
    shift = float(training_shift)
    scale = float(timestep_scale)
    distribution = str(time_distribution).strip().lower()
    if nmc <= 0:
        raise ValueError(f"FPO num_mc_samples must be positive, got {nmc}.")
    if shift <= 0.0 or scale <= 0.0:
        raise ValueError("FPO training_shift and timestep_scale must be positive.")

    seeds = []
    base_times = []
    for mc_index in range(nmc):
        noise_seed, uniform = _fpo_hash_pair(sampling_seed, mc_index)
        seeds.append(noise_seed)
        base_times.append(_sample_base_time(uniform, distribution))
    base = torch.tensor(base_times, dtype=torch.float32, device=device).unsqueeze(0)
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    return {
        "fpo_base_times": base.contiguous(),
        "fpo_sigmas": sigmas.contiguous(),
        "fpo_timesteps": (sigmas * scale).contiguous(),
        "fpo_noise_seeds": torch.tensor(
            seeds, dtype=torch.int64, device=device
        ).unsqueeze(0),
    }


def seeded_fpo_epsilon(
    reference: torch.Tensor, noise_seeds: torch.Tensor
) -> torch.Tensor:
    """Regenerate one joint epsilon per sample from persisted int64 seeds."""

    if reference.ndim != 2:
        raise ValueError(
            "FPO epsilon reference must have shape [B, width], got "
            f"{tuple(reference.shape)}."
        )
    seeds = noise_seeds.reshape(-1)
    if int(seeds.numel()) != int(reference.shape[0]):
        raise ValueError(
            "FPO noise seed count must match batch size: "
            f"{seeds.numel()} != {reference.shape[0]}."
        )
    samples = []
    for sample_idx in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seeds[sample_idx].item()))
        samples.append(
            torch.randn(
                reference[sample_idx].shape,
                generator=generator,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
    return torch.stack(samples, dim=0).contiguous()


def compute_fpo_causal_score_diagnostics(
    *,
    before_logprobs: torch.Tensor,
    after_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ratio_guard_threshold: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Measure one fixed-pair update in the GRPO-favoured direction.

    FPO uniformly encodes one scalar score over the action coordinates.  The
    existing chunk-level GRPO path sums those coordinates, so the exact local
    first-order improvement statistic is ``advantage * (score_after -
    score_before)``.  This helper intentionally accepts one advantage and one
    mask value per sample; P1 is a one-chunk causal gate, not a multi-chunk
    training metric.
    """

    if before_logprobs.shape != after_logprobs.shape:
        raise ValueError(
            "FPO P1 before/after logprob shapes differ: "
            f"{tuple(before_logprobs.shape)} != {tuple(after_logprobs.shape)}."
        )
    if before_logprobs.ndim < 2 or before_logprobs.shape[0] <= 0:
        raise ValueError(
            "FPO P1 logprobs must have shape [batch, ...] with a non-empty "
            f"batch, got {tuple(before_logprobs.shape)}."
        )
    batch_size = int(before_logprobs.shape[0])
    advantage = advantages.detach().float().reshape(-1)
    if int(advantage.numel()) != batch_size:
        raise ValueError(
            "FPO P1 requires exactly one chunk-level advantage per sample: "
            f"got {advantage.numel()} values for batch {batch_size}."
        )
    if loss_mask is None:
        valid = torch.ones(
            batch_size, dtype=torch.bool, device=before_logprobs.device
        )
    else:
        valid = loss_mask.detach().reshape(-1).to(
            device=before_logprobs.device, dtype=torch.bool
        )
        if int(valid.numel()) != batch_size:
            raise ValueError(
                "FPO P1 requires exactly one chunk-level loss mask per sample: "
                f"got {valid.numel()} values for batch {batch_size}."
            )
    valid_count = valid.count_nonzero()
    if int(valid_count.item()) == 0:
        raise ValueError("FPO P1 has no valid samples for its causal probe.")

    score_before = before_logprobs.detach().float().reshape(batch_size, -1).sum(-1)
    score_after = after_logprobs.detach().float().reshape(batch_size, -1).sum(-1)
    advantage = advantage.to(device=score_before.device)
    score_delta = score_after - score_before
    directional = advantage * score_delta
    valid_directional = directional[valid]
    valid_delta = score_delta[valid]
    finite = (
        torch.isfinite(score_before[valid]).all()
        & torch.isfinite(score_after[valid]).all()
        & torch.isfinite(advantage[valid]).all()
        & torch.isfinite(valid_delta).all()
        & torch.isfinite(valid_directional).all()
    )

    threshold = float(ratio_guard_threshold)
    if not math.isfinite(threshold) or threshold <= 1.0:
        raise ValueError(
            "FPO P1 ratio_guard_threshold must be finite and greater than 1, "
            f"got {threshold}."
        )
    log_threshold = math.log(threshold)
    ratio_guard_pass = finite & (valid_delta.max() <= log_threshold)
    reported_ratio = torch.exp(torch.clamp(valid_delta, min=-20.0, max=20.0))
    return {
        "score_before": score_before,
        "score_after": score_after,
        "score_delta": score_delta,
        "advantages": advantage,
        "valid_mask": valid,
        "valid_count": valid_count.to(dtype=torch.float32),
        "directional_sum": valid_directional.sum(),
        "directional_mean": valid_directional.mean(),
        "score_abs_delta_max": valid_delta.abs().max(),
        "log_ratio_abs_max": valid_delta.abs().max(),
        "ratio_min": reported_ratio.min(),
        "ratio_max": reported_ratio.max(),
        "all_finite": finite.to(dtype=torch.float32),
        "ratio_guard_pass": ratio_guard_pass.to(dtype=torch.float32),
    }
