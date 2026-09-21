from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rlinf_modified.algorithms import grpo_action_suffix_advantages, ppo_clipped_actor_loss
from rlinf_modified.algorithms.fpo import (
    fpo_action_epsilon_mse,
    fpo_action_velocity_mse,
)
from rlinf_modified.contracts import CameraBundle, Trajectory
from rlinf_modified.rewards import combine_chunk_rewards, terminal_goal_reward
from rlinf.rewards.resnet_reward_model import (
    classify_terminal_probabilities,
    resolve_eval_success_model_provenance,
)


def test_grpo_suffix_and_ppo_clip_are_finite():
    rewards = torch.tensor([[1.0, 0.0, 2.0, 1.0], [0.5, 1.0, 0.0, 2.0]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    dones = torch.zeros(3, 4, dtype=torch.bool)
    dones[-1] = True
    advantages = grpo_action_suffix_advantages(rewards, mask, dones, group_size=2, gamma=0.95)
    current = torch.tensor([0.5, -0.5], dtype=torch.float32, requires_grad=True)
    old = torch.zeros(2, dtype=torch.float32)
    loss, metrics = ppo_clipped_actor_loss(
        current, old, advantages[0, :2], mask[0, :2], clip_ratio_low=0.2, clip_ratio_high=0.2
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())


def test_fpo_epsilon_mse_is_velocity_mse_times_one_minus_sigma_squared():
    clean = torch.zeros(2, 4)
    epsilon = torch.ones_like(clean)
    predicted_velocity = torch.zeros_like(clean)
    sigma = torch.tensor([[0.25], [0.75]])
    kwargs = {
        "num_action_chunks": 2,
        "raw_action_dim": 1,
        "max_action_dim": 2,
    }

    velocity_mse = fpo_action_velocity_mse(
        predicted_velocity,
        epsilon,
        clean,
        **kwargs,
    )
    epsilon_mse = fpo_action_epsilon_mse(
        predicted_velocity,
        epsilon,
        clean,
        sigma,
        **kwargs,
    )

    assert torch.allclose(velocity_mse, torch.ones(2))
    assert torch.allclose(epsilon_mse, torch.tensor([0.75**2, 0.25**2]))


def test_trajectory_requires_t_plus_one_done_boundaries():
    trajectory = Trajectory(chunks=(None,), dones=torch.zeros(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"T\+1"):
        trajectory.validate(expected_chunks=1)


def test_per_mc_ratio_differs_from_ratio_after_mc_loss_average():
    advantages = torch.ones(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    old_scores = torch.zeros(1, 2)
    new_scores = torch.tensor([[0.4, -0.4]], requires_grad=True)
    per_mc_loss, _ = ppo_clipped_actor_loss(
        new_scores,
        old_scores,
        advantages,
        mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    averaged_loss, _ = ppo_clipped_actor_loss(
        new_scores.mean(dim=1),
        old_scores.mean(dim=1),
        advantages.mean(dim=1),
        mask[:, 0],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    assert per_mc_loss.item() != pytest.approx(averaged_loss.item())


def test_three_camera_terminal_reward_and_final_addition():
    zeros = torch.zeros(2, 4, 3, 8, 8)
    ones = torch.ones_like(zeros)
    world = CameraBundle(main=zeros, wrist=zeros, extra=zeros)
    goal = CameraBundle(main=ones, wrist=ones, extra=ones)
    terminal = terminal_goal_reward(
        world,
        goal,
        window_size=4,
        size=(8, 8),
        view_weights=(2 / 3, 1 / 6, 1 / 6),
        scale=160.0,
    )
    combined = combine_chunk_rewards(torch.zeros(2, 3), terminal, torch.tensor([False, True]))
    assert combined[0].sum() == 0
    assert combined[1, -1] == pytest.approx(-160.0)


def test_native_terminal_classifier_appends_and_requires_actual_last_frame():
    probabilities = torch.ones(1, 64)
    probabilities[:, -1] = 0.1
    cfg = {
        "aggregation": "terminal_positive_ratio",
        "terminal_window_frames": 30,
        "frame_stride": 3,
        "probability_threshold": 0.5,
        "minimum_terminal_positive_ratio": 0.8,
        "requires_last_frame_positive": True,
    }

    failed = classify_terminal_probabilities(probabilities, cfg)
    assert failed["sample_indices"].tolist() == [
        34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 63
    ]
    assert failed["positive_ratio"].item() == pytest.approx(10 / 11)
    assert failed["success"].item() is False

    probabilities[:, -1] = 0.9
    passed = classify_terminal_probabilities(probabilities, cfg)
    assert passed["success"].item() is True


def test_single_task_eval_provenance_does_not_use_duck_color_ranges(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "checkpoint_best.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    digest = "9" * 64

    variant, provenance = resolve_eval_success_model_provenance(
        13,
        active_variants=["nest_four_cups"],
        runtime_provenance={},
        configured_models={
            "nest_four_cups": {
                "from_pretrained": str(checkpoint.parent),
                "artifact_name": checkpoint.name,
                "sha256": digest,
            }
        },
    )

    assert variant == "nest_four_cups"
    assert provenance == {"path": str(checkpoint.resolve()), "sha256": digest}


def test_multi_variant_eval_provenance_keeps_duck_episode_routing(tmp_path):
    checkpoint = tmp_path / "brown.pt"
    checkpoint.touch()
    digest = "a" * 64

    variant, provenance = resolve_eval_success_model_provenance(
        13,
        active_variants=["brown", "red", "white", "yellow"],
        runtime_provenance={
            "brown": {"path": str(checkpoint), "sha256": digest}
        },
        configured_models={},
    )

    assert variant == "brown"
    assert provenance["sha256"] == digest


def test_eval_provenance_fails_before_inference_without_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="no valid success-model SHA-256"):
        resolve_eval_success_model_provenance(
            13,
            active_variants=["nest_four_cups"],
            runtime_provenance={},
            configured_models={
                "nest_four_cups": {"from_pretrained": str(checkpoint)}
            },
        )
