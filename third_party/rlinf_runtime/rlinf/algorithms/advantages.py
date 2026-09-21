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

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    if kwargs.get("chunk_reward_log") is not None:
        kwargs["chunk_reward_log"]["grouped_std"] = grouped_reward_std.detach()

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    if loss_mask is None:
        # `rewards` here is already the per-episode collapsed score (one value
        # per trajectory, see `calculate_scores`), not the full [n_steps,
        # batch_size] time-expanded tensor. Broadcasting against `rewards`'
        # own shape only happens to be correct when n_steps == 1; use the
        # actual time-expanded shape so multi-chunk episodes broadcast the
        # per-episode advantage to every chunk-step instead of failing the
        # reshape in `postprocess_embodied_advantages_outputs`.
        loss_mask = torch.ones(
            kwargs["n_steps"], kwargs["batch_size"], device=rewards.device
        )

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_action_suffix")
def compute_grpo_action_suffix_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    dones: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    dynamic_gammas: Optional[torch.Tensor] = None,
    suffix_success_anchor_enabled: bool = False,
    suffix_success_threshold: float = 0.9,
    suffix_success_trace_decay: float = 0.95,
    suffix_gamma_min: float = 0.9,
    suffix_gamma_max: float = 1.0,
    latent_motion: Optional[torch.Tensor] = None,
    suffix_latent_gamma_enabled: bool = False,
    suffix_latent_gamma_beta: float = 0.02,
    suffix_latent_gamma_min: float = 0.98,
    suffix_latent_gamma_after_success: bool = True,
    **kwargs,
):
    """
    Compute a group-normalized suffix-return advantage at every action step.

    Args:
        rewards: Step rewards with shape [n_steps, bsz].
        loss_mask: Valid-step mask with shape [n_steps, bsz].
        group_size: Number of samples in each GRPO group.
        dones: Done flags with shape [n_steps + 1, bsz].

    Returns:
        Tuple[torch.Tensor, None]: Per-step advantages and no returns tensor.
    """
    if loss_mask is None:
        loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    if dones is None:
        dones = torch.zeros(
            rewards.shape[0] + 1,
            rewards.shape[1],
            dtype=torch.bool,
            device=rewards.device,
        )
    n_steps, batch_size = rewards.shape
    if group_size < 2:
        raise ValueError(f"GRPO group_size must be at least 2, got {group_size}")
    if batch_size % group_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by group_size {group_size}"
        )
    if suffix_gamma_min > suffix_gamma_max:
        raise ValueError(
            f"suffix_gamma_min ({suffix_gamma_min}) must be <= suffix_gamma_max ({suffix_gamma_max})"
        )

    running = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)

    valid_mask = loss_mask.to(torch.bool)
    success = (rewards > float(suffix_success_threshold)) & valid_mask
    if dynamic_gammas is not None:
        if dynamic_gammas.shape != rewards.shape:
            raise ValueError(
                "dynamic_gammas shape must match rewards shape: "
                f"{dynamic_gammas.shape} != {rewards.shape}"
            )
        gammas = dynamic_gammas.to(device=rewards.device, dtype=rewards.dtype)
    elif suffix_latent_gamma_enabled:
        if latent_motion is None:
            raise ValueError(
                "suffix_latent_gamma_enabled=True requires latent_motion in the rollout batch"
            )
        if latent_motion.shape != rewards.shape:
            raise ValueError(
                f"latent_motion shape {latent_motion.shape} must match rewards shape {rewards.shape}"
            )
        motion = latent_motion.to(device=rewards.device, dtype=rewards.dtype)
        mask_float = valid_mask.to(rewards.dtype)
        motion = motion * mask_float
        motion_mean = motion.sum(dim=0, keepdim=True) / mask_float.sum(
            dim=0, keepdim=True
        ).clamp_min(1.0)
        motion_norm = motion / motion_mean.clamp_min(1e-6)
        motion_excess = torch.relu(motion_norm - 1.0)
        if suffix_latent_gamma_after_success:
            success_gate = torch.empty_like(rewards)
            seen_success = torch.zeros(
                batch_size, dtype=rewards.dtype, device=rewards.device
            )
            for step in range(n_steps):
                seen_success = seen_success * (~dones[step]).to(rewards.dtype)
                seen_success = torch.where(
                    success[step],
                    torch.ones_like(seen_success),
                    seen_success,
                )
                success_gate[step] = seen_success
        else:
            success_gate = success.any(dim=0, keepdim=True).to(rewards.dtype)
            success_gate = success_gate.expand_as(rewards)
        gammas = 1.0 - float(suffix_latent_gamma_beta) * success_gate * motion_excess
        gammas = torch.clamp(
            gammas, min=float(suffix_latent_gamma_min), max=float(suffix_gamma_max)
        )
    elif suffix_success_anchor_enabled:
        future_success_trace = torch.zeros(
            batch_size, dtype=rewards.dtype, device=rewards.device
        )
        gammas = torch.empty_like(rewards)
        trace_decay = float(suffix_success_trace_decay)
        gamma_min = float(suffix_gamma_min)
        gamma_scale = float(suffix_gamma_max) - gamma_min
        for step in reversed(range(n_steps)):
            not_done_next = (~dones[step + 1]).to(rewards.dtype)
            future_success_trace = future_success_trace * not_done_next
            gammas[step] = gamma_min + gamma_scale * future_success_trace
            current_success = success[step].to(rewards.dtype)
            future_success_trace = torch.where(
                current_success > 0,
                torch.ones_like(future_success_trace),
                future_success_trace * trace_decay,
            )
    else:
        gammas = torch.full_like(rewards, float(gamma))

    suffix_returns = torch.empty_like(rewards)
    for step in reversed(range(n_steps)):
        gamma_t = gammas[step]
        running = rewards[step] + gamma_t * running * (~dones[step + 1]).to(
            rewards.dtype
        )
        suffix_returns[step] = running

    num_groups = batch_size // group_size
    grouped_returns = suffix_returns.reshape(n_steps, num_groups, group_size)
    grouped_mean = grouped_returns.mean(dim=-1, keepdim=True)
    grouped_std = grouped_returns.std(dim=-1, keepdim=True)
    if kwargs.get("chunk_reward_log") is not None:
        kwargs["chunk_reward_log"]["grouped_std"] = grouped_std.detach()
    advantages = ((grouped_returns - grouped_mean) / (grouped_std + 1e-6)).reshape(
        n_steps, batch_size
    )

    advantages = advantages * loss_mask.to(advantages.dtype)
    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None
