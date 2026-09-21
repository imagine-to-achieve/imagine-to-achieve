from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.workers.actor import fsdp_actor_worker
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker


def _repeat(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64).reshape(1, -1).repeat(5, 1)


def test_lightweight_trajectory_records_do_not_require_audit_manifests(
    monkeypatch, tmp_path
):
    contract = SimpleNamespace(
        color_order=("brown", "red", "white", "yellow"),
        episodes_by_color={
            "brown": (0,),
            "red": (50,),
            "white": (100,),
            "yellow": (150,),
        },
        manifest_path="/frozen/duck_split_manifest.json",
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(
        fsdp_actor_worker,
        "load_duck_episode_split",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        fsdp_actor_worker,
        "assert_duck_episode_split_unchanged",
        lambda _contract: None,
    )

    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "trajectory_records": {
                    "enabled": True,
                    "provenance_mode": "lightweight_analysis",
                    "output_dir": str(tmp_path),
                    "expected_trajectories": 4,
                    "trajectory_frames": 160,
                    "split_manifest": "/frozen/duck_split_manifest.json",
                },
                "rollout_retry": {"max_retries": 1},
            },
            "reward": {"success_classifier": {"threshold": 0.5}},
            "actor": {"model": {"num_action_chunks": 32}},
            "duck": {"colors": ["red"]},
        }
    )
    actor._local_accelerator_rank = 0
    actor._rank = 0
    actor.global_step = 5
    actor.version = 5

    chunk_ids = torch.arange(5, dtype=torch.int64).reshape(5, 1).repeat(1, 4)
    action_seeds = chunk_ids + 100
    probabilities = torch.zeros(5, 4, 4, dtype=torch.float32)
    probabilities[-1, :, -1] = 0.75
    actor.rollout_batch = {
        "rollout_uid": _repeat([500, 501, 502, 503]),
        "global_group_id": _repeat([0, 0, 0, 0]),
        "group_member_id": _repeat([0, 1, 2, 3]),
        "source_env_rank": _repeat([0, 0, 0, 0]),
        "local_env_id": _repeat([0, 1, 2, 3]),
        "reset_seed": _repeat([42, 42, 42, 42]),
        "vision_noise_seed": action_seeds,
        "action_noise_seed": action_seeds,
        "ctrl_world_noise_seed": _repeat([123, 123, 123, 123]),
        "update_id": _repeat([5, 5, 5, 5]),
        "logical_round_id": _repeat([0, 0, 0, 0]),
        "physical_wave_id": _repeat([0, 0, 0, 0]),
        "group_slot": _repeat([0, 0, 0, 0]),
        "chunk_id": chunk_ids,
        "reset_episode": _repeat([50, 50, 50, 50]),
        "seed_nonce": chunk_ids,
        "shuffle_seed": _repeat([7, 7, 7, 7]),
        "retry_count": torch.zeros(5, 4, dtype=torch.int64),
        "video_similarity_rewards": torch.full((5, 4), -0.01),
        "terminal_goal_rewards": torch.full((5, 4), -0.02),
        "continuous_combined_rewards": torch.full((5, 4), -0.03),
        "reward_model_probabilities": probabilities,
        "rewards": torch.full((5, 4), -0.03),
        "advantages": torch.zeros(5, 4),
        "prev_logprobs": torch.full((5, 4), -0.1),
    }

    actor._initialize_trajectory_records()

    assert len(actor._trajectory_records) == 4
    for record in actor._trajectory_records.values():
        assert record["provenance_mode"] == "lightweight_analysis"
        assert record["checkpoint_checksum_manifest"] is None
        assert record["input_checksums"]["artifacts"] == {}
        assert record["color"] == "red"
        assert record["reset_episode"] == 50
        assert record["success_probability_max"] == 0.75
        assert record["training_reward_source"] == "continuous_combined"
        assert record["success_reward"] == 0.0



def test_terminal_max_window_success_binary_reward_is_emitted_only_at_end():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "cosmos"}},
            "reward": {
                "training_source": "success_binary",
                "success_classifier": {
                    "enabled": True,
                    "diagnostic_only": False,
                    "replace_training_reward": True,
                    "aggregation": "max",
                    "window_size": 4,
                    "threshold": 0.5,
                    "reward_value": 1.0,
                },
            },
        }
    )
    rewards = torch.zeros(2, 4, dtype=torch.float32)
    env_output = SimpleNamespace(
        dones=torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        )
    )
    probabilities = torch.tensor(
        [[0.10, 0.60, 0.20, 0.10], [0.49, 0.20, 0.10, 0.40]],
        dtype=torch.float32,
    )
    env_info = {"_reward_model_frame_probabilities": probabilities}

    recorded = worker._apply_success_classifier_diagnostic(
        rewards, env_output, env_info
    )
    binary = env_info.pop("_success_binary_rewards")

    assert torch.equal(recorded, probabilities)
    assert torch.equal(binary[:, :-1], torch.zeros(2, 3))
    assert torch.equal(binary[:, -1], torch.tensor([1.0, 0.0]))
    assert torch.equal(env_info["reward/success"], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        env_info["reward/success_probability"], torch.tensor([0.60, 0.49])
    )


def test_training_reward_selector_returns_exactly_one_component():
    worker = EnvWorker.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {"reward": {"training_source": "continuous_combined"}}
    )
    trajectory = torch.tensor([[1.0, 2.0]])
    terminal = torch.tensor([[0.0, 3.0]])
    continuous = trajectory + terminal
    success = torch.tensor([[0.0, 1.0]])
    expected = {
        "continuous_combined": continuous,
        "trajectory_mse": trajectory,
        "terminal_goal_mse": terminal,
        "success_binary": success,
    }
    for source, value in expected.items():
        worker.cfg.reward.training_source = source
        selected = worker._select_training_rewards(
            trajectory_rewards=trajectory,
            terminal_rewards=terminal,
            continuous_rewards=continuous,
            success_rewards=success,
        )
        assert torch.equal(selected, value)
