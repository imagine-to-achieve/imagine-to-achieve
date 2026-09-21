from __future__ import annotations

import ast
import dataclasses
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from rlinf.config import _validate_cosmos_ctrl_world_contract
from rlinf.envs.world_model.duck_episode_contract import (
    load_duck_episode_split,
    select_duck_episode_colors,
)
from rlinf.utils.comm_mapping import CommMapper
from rlinf_modified.config import ConfigError, load_config
from rlinf_modified.engine.real import (
    RealTrainer,
    _shard_aligned_eval_episode_ids,
)
from rlinf_modified.launch import _segments
from rlinf_modified.tasks import aggregate_sparse_evaluation, build_task_spec


ROOT = Path(__file__).resolve().parents[1]


def _strict_native_ctrl_world_cfg():
    ctrl_contract = {
        "action_dim": 7,
        "policy_action_dim": 10,
        "chunk": 32,
        "ctrl_world_chunk": 64,
        "decode_chunk_size": 7,
        "image_size": [192, 320],
        "policy_image_size": [192, 320],
        "use_raw_reset_policy_views": True,
        "per_view_vae_codec": True,
        "prompt_override": "close the laptop",
        "fps": 30,
        "cosmos_action_fps": 15,
        "ctrl_condition_fps": 30,
        "action_fps_resample": True,
        "ctrl_world_internal_rollout": True,
        "main_view_index": 0,
        "wrist_view_index": 2,
    }
    return OmegaConf.create(
        {
            "actor": {
                "model": {
                    "num_action_chunks": 32,
                    "cosmos": {
                        "strict_source_view_size": [480, 640],
                        "expected_composite_size": [720, 640],
                        "fps": 15,
                    },
                }
            },
            "env": {
                mode: {
                    "env_type": "cosmos_self_ctrl_world",
                    "self_feedback": {"target_size": [720, 640]},
                    "ctrl_world_cfg": dict(ctrl_contract),
                }
                for mode in ("train", "eval")
            },
        }
    )


def test_strict_native_ctrl_world_contract_accepts_paired_modes():
    _validate_cosmos_ctrl_world_contract(_strict_native_ctrl_world_cfg())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: setattr(
                cfg.env.eval.ctrl_world_cfg, "use_raw_reset_policy_views", False
            ),
            "cannot satisfy strict Cosmos source views",
        ),
        (
            lambda cfg: setattr(cfg.env.eval.ctrl_world_cfg, "chunk", 16),
            "chunk must match actor.model.num_action_chunks",
        ),
        (
            lambda cfg: setattr(cfg.env.eval.ctrl_world_cfg, "per_view_vae_codec", False),
            "train/eval contract mismatch",
        ),
    ],
)
def test_strict_native_ctrl_world_contract_rejects_eval_drift(mutation, message):
    cfg = _strict_native_ctrl_world_cfg()
    mutation(cfg)
    with pytest.raises(ValueError, match=message):
        _validate_cosmos_ctrl_world_contract(cfg)


@pytest.mark.parametrize(
    ("name", "world", "trajectories", "minibatch", "accumulation"),
    [
        ("close.yaml", 32, 512, 640, 20),
        ("duck.yaml", 32, 512, 640, 20),
        ("close_b128_mb32_u25.yaml", 32, 128, 160, 5),
        ("duck4_b512_mb128_u10.yaml", 32, 512, 640, 20),
        ("duck_red_b128_mb32_u10.yaml", 32, 128, 160, 5),
        ("duck4_b128_mb32_u50_traj_reward_fixediface_side_rm.yaml", 32, 128, 160, 5),
        ("duck4_b128_mb32_u50_terminal_mse_reward_fixediface_side_rm.yaml", 32, 128, 160, 5),
        ("duck4_b128_mb32_u50_combined_mse_reward_fixediface_side_rm.yaml", 32, 128, 160, 5),
        ("close_smoke_2n4g.yaml", 8, 32, 40, 5),
        ("duck_smoke_2n4g.yaml", 8, 64, 80, 10),
        ("nest_four_cups_32n4g_b512_u50.yaml", 128, 512, 1280, 10),
        ("nest_four_cups_32n4g_b512_recent_views_u50.yaml", 128, 512, 1280, 10),
        ("nest_four_cups_32n4g_b512_c21_u50.yaml", 128, 512, 2688, 21),
    ],
)
def test_batch_contract_profiles(name, world, trajectories, minibatch, accumulation):
    config = load_config(ROOT / "configs" / name)
    assert (config.world_size, config.batch.trajectories_per_update) == (world, trajectories)
    assert config.batch.global_minibatch_chunks == minibatch
    assert config.batch.gradient_accumulation_steps == accumulation
    assert config.cosmos.action_representation == "ur5_eef_relative_10d_droid_native"
    groups_per_wave = config.world_size * 4 // config.algorithm.group_size
    assert config.batch.rollout_epochs == (
        config.batch.logical_rounds_per_update * config.batch.waves_per_logical_round
    )
    assert config.batch.group_slots_per_logical_round == (
        groups_per_wave * config.batch.waves_per_logical_round
    )


@pytest.mark.parametrize(
    ("name", "source", "training_source"),
    [
        (
            "duck4_b128_mb32_u50_traj_reward_fixediface_side_rm.yaml",
            "trajectory_mse",
            "trajectory_mse",
        ),
        (
            "duck4_b128_mb32_u50_terminal_mse_reward_fixediface_side_rm.yaml",
            "terminal_goal_mse",
            "terminal_goal_mse",
        ),
        (
            "duck4_b128_mb32_u50_combined_mse_reward_fixediface_side_rm.yaml",
            "ctrl_world_aligned_mse_plus_terminal",
            "continuous_combined",
        ),
    ],
)
def test_clean_duck_u50_side_model_contract(
    tmp_path, name, source, training_source
):
    config = load_config(ROOT / "configs" / name)
    assert config.runtime.resume is False
    assert config.runtime.max_updates == 50
    assert config.runtime.segment_updates == 5
    assert config.slurm.nodes == 8
    assert config.slurm.gpus_per_node == 4
    assert config.batch.trajectories_per_update == 128
    assert config.batch.global_minibatch_chunks // config.batch.chunks_per_trajectory == 32
    assert config.reward.source == source
    assert config.ctrl_world.runtime.history_mode == "dense"
    assert config.ctrl_world.runtime.history_indices == (0, 1, 2, 3, 4, 5)
    assert config.ctrl_world.runtime.history_bank_size == 6
    expected_sha256 = {
        "brown": "04bfb6fcac04252f4dad4832d272194d8a6bd38513cf2521e873a32e7c4a72b8",
        "red": "f93c74954956cdb079a5f8e3136da1cdeffc59d274e5b97ed91deb226d2c361e",
        "white": "7650c52b7b6689991d06c416f01b6c9562c9573ce794d699a2d05f7536e018d4",
        "yellow": "17765fdeb74570bcd2c3402210c1df2e04b6548af983ee42a4f725f762916b98",
    }
    for color, digest in expected_sha256.items():
        model_path = Path(config.assets.reward_models[color])
        assert "duck_threeway_soft_boundary_rm" in model_path.parts
        assert config.assets.expected_sha256[f"reward_model_{color}"] == digest

    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime, output_dir=str(tmp_path / source)
        ),
    )
    overrides = RealTrainer(config)._overrides(
        "/tmp/normalized-duck-clean-config.yaml"
    )
    assert f"++reward.training_source={training_source}" in overrides
    assert not any(value.startswith("runner.resume_dir=") for value in overrides)
    assert "++algorithm.trajectory_records.expected_train_records_per_step=128" in overrides
    assert "++algorithm.trajectory_records.expected_validation_records_per_step=40" in overrides
    assert "++algorithm.trajectory_records.success_decision_threshold=0.5" in overrides
    assert "++env.train.ctrl_world_cfg.model_camera_ids=[2,0,1]" in overrides
    assert "++env.eval.ctrl_world_cfg.model_camera_ids=[2,0,1]" in overrides
    for mode in ("train", "eval"):
        prefix = f"++env.{mode}.ctrl_world_cfg"
        assert f"{prefix}.main_view_index=1" in overrides
        assert f"{prefix}.wrist_view_index=0" in overrides
        assert f"{prefix}.side_view_index=2" in overrides
        assert f"{prefix}.adapter_translation_gain=1.0" in overrides
        assert f"{prefix}.ctrl_world_window_history_mode=dense" in overrides
        assert (
            f"{prefix}.reward_model.camera_key="
            "observation.images.d405_1_rgb"
        ) in overrides
        assert f"{prefix}.reward_model.ctrl_world_view_index=2" in overrides


def test_duck_checkpoint20_side_eval_uses_audited_side_classifiers():
    config = load_config(
        ROOT / "configs" / "duck4_combined_mse_checkpoint20_side_eval_extremes.yaml"
    )
    assert config.experiment_name == (
        "duck4_combined_mse_checkpoint20_side_eval_extremes_8n4g"
    )
    assert config.assets.legacy_config_name.endswith("_eval_extremes_side")
    expected_models = {
        "brown": (
            "duck_0731_resnet_side_brown",
            "4d3562adaa0df893fb61dcfae7dc23d520806b058f2d6afd7085099b520d36b0",
        ),
        "red": (
            "duck_0731_resnet_side_red",
            "a5faf39630992a26589bfea6d177ce65a399e6622e39052d984e4139049aa7f3",
        ),
        "white": (
            "duck_0731_resnet_side_white",
            "42af102ccc10c16d22387df49a8f435f94f47a581a68e113d936ae59b5d845b6",
        ),
        "yellow": (
            "duck_0731_resnet_side_yellow",
            "a3a53ffa6c46abcc26aae8bcc4d1905cdb66bb5530b51ef972229bbc71505f8b",
        ),
    }
    for color, (directory_name, digest) in expected_models.items():
        model_path = Path(config.assets.reward_models[color])
        assert model_path.parent.name == directory_name
        assert model_path.name == "resnet_rm.pth"
        assert config.assets.expected_sha256[f"reward_model_{color}"] == digest

    overrides = RealTrainer(config)._overrides(
        "/tmp/duck-side-normalized-checkpoint-config.yaml"
    )
    for color, (_, digest) in expected_models.items():
        model_path = Path(config.assets.reward_models[color]).resolve()
        assert (
            f"++reward.success_models.{color}.from_pretrained={model_path}"
            in overrides
        )
        assert (
            f"++reward.success_models.{color}.artifact_name=resnet_rm.pth"
            in overrides
        )
        assert f"++reward.success_models.{color}.sha256={digest}" in overrides

    runtime_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "examples"
        / "embodiment"
        / "config"
        / (
            "ctrl_world_duck4color_grpo_cosmos_fpo_native_unipc_b512_"
            "cosmos_split_v2_eval_extremes_side.yaml"
        )
    )
    runtime = yaml.safe_load(runtime_path.read_text())
    for mode in ("train", "eval"):
        reward_model = runtime["env"][mode]["ctrl_world_cfg"]["reward_model"]
        assert reward_model["camera_key"] == "observation.images.d405_1_rgb"
        assert reward_model["ctrl_world_view_index"] == 2
        assert reward_model["require_camera_metadata"] is True


def test_reward_model_observation_routes_all_three_views_and_side_is_used():
    source = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "envs"
        / "world_model"
        / "world_model_ctrl_world_env.py"
    ).read_text()
    tree = ast.parse(source)
    route_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_reward_model_observation"
    )
    route_method.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(body=[route_method], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, "<reward-model-view-router>", "exec"), namespace)
    route = namespace["_reward_model_observation"]

    main, side, wrist = object(), object(), object()
    env = SimpleNamespace(
        main_view_index=0,
        extra_view_index=1,
        wrist_view_index=2,
        current_obs=main,
        current_extra_view_obs=side,
        current_wrist_obs=wrist,
        reward_model_camera_key="observation.images.d405_1_rgb",
    )
    for view_index, expected in ((0, main), (1, side), (2, wrist)):
        env.reward_model_view_index = view_index
        assert route(env) is expected

    env.reward_model_view_index = 1
    env.current_extra_view_obs = None
    with pytest.raises(RuntimeError, match="camera_key=.*d405_1_rgb"):
        route(env)

    inference_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_infer_next_chunk_rewards"
    )
    called_attributes = {
        node.attr for node in ast.walk(inference_method) if isinstance(node, ast.Attribute)
    }
    assert "_reward_model_observation" in called_attributes


def test_ctrl_world_lookahead_windows_preserve_boundary_and_emit_next_frames():
    source = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "envs"
        / "world_model"
        / "world_model_ctrl_world_env.py"
    ).read_text()
    tree = ast.parse(source)
    planner = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plan_ctrl_world_windows"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[planner], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, "<ctrl-world-window-plan>", "exec"), namespace)
    plan = namespace["_plan_ctrl_world_windows"]
    assert plan(10, 5, 4, use_lookahead_latent=True) == [
        (0, 5, 4),
        (4, 5, 4),
        (8, 5, 2),
    ]
    with pytest.raises(ValueError, match="non-emitted boundary frame"):
        plan(10, 5, 5, use_lookahead_latent=True)
    assert "pred_window[:, 1 : emit_frames + 1]" not in source
    assert "emitted_latents.append(pred_window[:, :emit_frames])" in source
    assert "self.current_latent = pred_window[:, emit_frames]" in source
    assert "pred_window[:, :emit_frames]" in source


def test_duck_one_step_uses_independent_ctrl_world_rollout_contract(tmp_path):
    config = load_config(
        ROOT
        / "configs"
        / "duck4_combined_mse_gs15_to16_principal_ctrl50_extreme_videos.yaml"
    )
    assert config.runtime.max_updates == 17
    assert config.runtime.require_real_assets is False
    assert config.evaluation.enabled is True
    assert config.trajectory_records.enabled is True
    assert config.trajectory_records.mode == "lightweight_analysis"
    assert config.ctrl_world.runtime.history_mode == "sparse_window"
    assert config.ctrl_world.runtime.history_indices == (-7, -6, -5, -4, -3, -2)
    assert config.ctrl_world.runtime.history_bank_size == 24

    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime, output_dir=str(tmp_path / "duck-independent-rollout")
        ),
    )
    overrides = RealTrainer(config)._overrides(
        "/tmp/normalized-duck-independent-rollout.yaml"
    )
    for mode in ("train", "eval"):
        prefix = f"++env.{mode}.ctrl_world_cfg"
        assert f"{prefix}.ctrl_world_window_history_mode=sparse_window" in overrides
        assert f"{prefix}.ctrl_world_history_idx=[-7,-6,-5,-4,-3,-2]" in overrides
        assert f"{prefix}.ctrl_world_history_bank_size=24" in overrides
        assert f"{prefix}.fixed_denoise_seed=null" in overrides
    assert "algorithm.trajectory_records.enabled=true" in overrides
    assert "++algorithm.trajectory_records.save_final_side_frames=true" in overrides
    assert "++duck.evaluation.enabled=false" not in overrides
    assert "++duck.analysis.enabled=false" not in overrides
    assert "duck.analysis.enabled=false" not in overrides
    assert "++post_update_evaluation.enabled=false" not in overrides


def test_unknown_config_key_fails_before_runtime(tmp_path):
    payload = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text())
    payload["runtime"]["typo_key"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("condition_fps", 0.0, "policy_action_fps and condition_fps"),
        ("resampled_chunk", 63, "resampled_chunk"),
        ("decode_chunk_size", 0, "decode_chunk_size"),
        ("history_indices", (0, -1), "history_indices length"),
        ("history_mode", "invalid", "history_mode"),
        ("window_emit_frames", 5, "lookahead rollout"),
    ],
)
def test_invalid_ctrl_world_runtime_contract_fails_before_ray(field, value, message):
    config = load_config(ROOT / "configs" / "close_smoke_2n4g.yaml")
    runtime = dataclasses.replace(config.ctrl_world.runtime, **{field: value})
    config = dataclasses.replace(
        config,
        ctrl_world=dataclasses.replace(config.ctrl_world, runtime=runtime),
    )
    with pytest.raises(ConfigError, match=message):
        config.validate()


def test_noncanonical_real_action_representation_fails_before_ray():
    config = load_config(ROOT / "configs" / "close_smoke_2n4g.yaml")
    config = dataclasses.replace(
        config,
        cosmos=dataclasses.replace(config.cosmos, action_representation="relative_eef"),
    )
    with pytest.raises(ConfigError, match="canonical Cosmos 10D relative EEF"):
        config.validate()


def test_arbitrary_duck_subset_is_single_source_of_truth():
    config = load_config(ROOT / "configs" / "duck.yaml")
    config = dataclasses.replace(
        config, task=dataclasses.replace(config.task, active_variants=("brown", "white"))
    )
    config.validate()
    spec = build_task_spec(config)
    assert spec.active_variants == ("brown", "white")
    assert all(len(spec.train_episode_ids[color]) == 40 for color in spec.active_variants)
    assert "red" not in spec.train_episode_ids


def test_nest_four_cups_uses_cosmos_80_20_split_and_requested_eval_contract():
    config = load_config(ROOT / "configs" / "nest_four_cups_32n4g_b512_u50.yaml")
    spec = build_task_spec(config)
    train = spec.train_episode_ids["nest_four_cups"]
    evaluation = spec.eval_episode_ids["nest_four_cups"]

    assert len(train) == 80
    assert len(evaluation) == 20
    assert set(train).isdisjoint(evaluation)
    assert set(train) | set(evaluation) == set(range(100))
    assert config.evaluation.interval_updates == 5
    assert config.evaluation.episodes_per_variant == 20
    assert config.evaluation.save_video
    assert config.cosmos.guidance == 1.0
    assert config.ctrl_world.runtime.guidance_scale == 1.0


def test_nest_four_cups_default_is_ten_chunks_with_four_optimizer_steps():
    config = load_config(ROOT / "configs" / "nest_four_cups_32n4g_b512_u50.yaml")
    assert config.batch.chunks_per_trajectory == 10
    assert config.batch.trajectories_per_update * config.batch.chunks_per_trajectory == 5120
    assert config.batch.global_minibatch_chunks == 1280
    assert config.batch.gradient_accumulation_steps == 10
    assert config.batch.expected_optimizer_steps == 4
    overrides = RealTrainer(config)._overrides("/tmp/normalized-cosmos-config.yaml")
    assert "algorithm.trajectory_chunks=10" in overrides
    assert "algorithm.trajectory_records.trajectory_frames=320" in overrides
    assert "env.train.max_episode_steps=320" in overrides
    assert "env.eval.max_episode_steps=320" in overrides


def test_success_failure_mse_extremes_select_each_global_category():
    records = [
        {"rank": 0, "env_id": 0, "success": True, "trajectory_mse": 0.4},
        {"rank": 1, "env_id": 0, "success": True, "trajectory_mse": 0.1},
        {"rank": 2, "env_id": 0, "success": False, "trajectory_mse": 0.2},
        {"rank": 3, "env_id": 0, "success": False, "trajectory_mse": 0.7},
    ]
    runner_source = (
        ROOT / "third_party" / "rlinf_runtime" / "rlinf" / "runners" / "embodied_runner.py"
    ).read_text()
    tree = ast.parse(runner_source)
    selector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_success_failure_mse_extremes"
    )
    selector.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[selector], type_ignores=[]))
    namespace = {}
    exec(compile(module, "<mse-selector>", "exec"), namespace)
    selected = namespace["_select_success_failure_mse_extremes"](records)
    assert selected == {
        "success_mse_min": records[1],
        "success_mse_max": records[0],
        "failure_mse_min": records[2],
        "failure_mse_max": records[3],
    }


def test_final_side_frames_are_one_lossless_indexed_archive_per_step(tmp_path):
    import hashlib as hashlib_module
    import math as math_module
    import os as os_module
    import tempfile as tempfile_module

    import numpy as np_module

    runner_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "runners"
        / "embodied_runner.py"
    )
    tree = ast.parse(runner_path.read_text())
    wanted = {
        "_select_success_failure_mse_extremes",
        "_atomic_write_npz",
        "_archive_final_side_frames",
    }
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {
        "hashlib": hashlib_module,
        "math": math_module,
        "np": np_module,
        "os": os_module,
        "Path": Path,
        "tempfile": tempfile_module,
    }
    exec(compile(module, "<final-side-frame-archive>", "exec"), namespace)

    records = []
    specs = (
        (0, 13, True, 0.4, 10),
        (1, 10, True, 0.1, 20),
        (2, 12, False, 0.2, 30),
        (3, 11, False, 0.7, 40),
    )
    for env_id, rollout_uid, success, mse, pixel in specs:
        frame = np_module.full((6, 8, 3), pixel, dtype=np_module.uint8)
        records.append(
            {
                "rank": env_id,
                "env_id": 0,
                "episode": 100 + env_id,
                "rollout_uid": rollout_uid,
                "group_id": env_id // 2,
                "member_id": env_id,
                "success": success,
                "trajectory_mse": mse,
                "reward": -mse,
                "_final_side_frame": frame,
                "final_side_frame_sha256": hashlib_module.sha256(
                    frame.tobytes(order="C")
                ).hexdigest(),
            }
        )
    artifact_cfg = {
        "save_final_side_frames": True,
        "save_mse_extreme_final_frames": True,
        "expected_train_records_per_step": 4,
        "final_side_frame_dir": str(tmp_path / "final_frames"),
    }
    runner = SimpleNamespace(
        cfg=SimpleNamespace(algorithm={"trajectory_records": artifact_cfg}),
        _run_root=lambda: tmp_path,
        _select_success_failure_mse_extremes=namespace[
            "_select_success_failure_mse_extremes"
        ],
        _atomic_write_npz=namespace["_atomic_write_npz"],
    )
    metrics = namespace["_archive_final_side_frames"](
        runner, records, step=7, split="training"
    )
    archive_path = (
        tmp_path
        / "final_frames"
        / "training"
        / "step_000007"
        / "final_side_frames.npz"
    )
    assert archive_path.is_file()
    assert metrics["rollout_final_side_frames_saved"] == 4.0
    assert metrics["mse_extreme_final_frames_saved"] == 4.0
    with np_module.load(archive_path, allow_pickle=False) as archive:
        assert archive["frames"].shape == (4, 6, 8, 3)
        assert archive["frames"].dtype == np_module.uint8
        assert archive["camera_key"].tolist() == [
            "observation.images.d405_1_rgb"
        ]
        assert archive["ctrl_world_view_index"].tolist() == [2]
        assert archive["extreme_categories"].tolist() == [
            "success_mse_min",
            "success_mse_max",
            "failure_mse_min",
            "failure_mse_max",
        ]
        assert archive["extreme_available"].tolist() == [True, True, True, True]
        assert archive["extreme_archive_index"].tolist() == [0, 3, 2, 1]
        assert archive["extreme_frames"][:, 0, 0, 0].tolist() == [20, 10, 30, 40]
    assert all("_final_side_frame" not in record for record in records)
    assert all(
        record["final_side_frame_archive_path"] == str(archive_path.resolve())
        for record in records
    )


def test_mse_extreme_finalizer_promotes_four_videos_and_writes_csv(tmp_path):
    import csv as csv_module
    import math as math_module
    import os as os_module
    import shutil as shutil_module

    runner_path = (
        ROOT / "third_party" / "rlinf_runtime" / "rlinf" / "runners" / "embodied_runner.py"
    )
    tree = ast.parse(runner_path.read_text())
    wanted = {
        "_select_success_failure_mse_extremes",
        "_finalize_success_failure_mse_extremes",
    }
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {
        "csv": csv_module,
        "math": math_module,
        "os": os_module,
        "shutil": shutil_module,
    }
    exec(compile(module, "<mse-finalizer>", "exec"), namespace)

    base_dir = tmp_path / "mse_extremes"
    video_cfg = {
        "save_success_failure_mse_extremes": True,
        "mse_extremes_base_dir": str(base_dir),
        "video_base_dir": str(tmp_path / "videos" / "train"),
    }
    cfg = SimpleNamespace(
        env=SimpleNamespace(train=SimpleNamespace(video_cfg=video_cfg)),
        runner=SimpleNamespace(
            logger=SimpleNamespace(log_path=str(tmp_path), experiment_name="test")
        ),
    )
    runner = SimpleNamespace(
        cfg=cfg,
        logger=SimpleNamespace(warning=lambda *args: None),
        _select_success_failure_mse_extremes=namespace[
            "_select_success_failure_mse_extremes"
        ],
    )
    records = [
        {
            "rank": 0,
            "env_id": 0,
            "success": True,
            "trajectory_mse": 0.4,
            "rollout_uid": 10,
            "group_id": 1,
            "member_id": 0,
        },
        {
            "rank": 1,
            "env_id": 0,
            "success": True,
            "trajectory_mse": 0.1,
            "rollout_uid": 11,
            "group_id": 1,
            "member_id": 1,
        },
        {
            "rank": 2,
            "env_id": 0,
            "success": False,
            "trajectory_mse": 0.2,
            "rollout_uid": 12,
            "group_id": 1,
            "member_id": 2,
        },
        {
            "rank": 3,
            "env_id": 0,
            "success": False,
            "trajectory_mse": 0.7,
            "rollout_uid": 13,
            "group_id": 1,
            "member_id": 3,
        },
    ]
    selected = runner._select_success_failure_mse_extremes(records)
    candidate_dir = tmp_path / "mse_extremes_candidates" / "step_000001"
    candidate_dir.mkdir(parents=True)
    for category, record in selected.items():
        candidate = candidate_dir / (
            f"{category}_rank_{record['rank']}_uid_{record['rollout_uid']}.mp4"
        )
        candidate.write_bytes(category.encode())

    metrics = namespace["_finalize_success_failure_mse_extremes"](runner, records, 1)
    assert metrics["mse_extremes_videos_saved"] == 4.0
    assert len(list(base_dir.glob("step_000001_*.mp4"))) == 4
    assert not candidate_dir.exists()
    with (base_dir / "mse_extremes_summary.csv").open(newline="") as csv_file:
        rows = list(csv_module.DictReader(csv_file))
    assert {row["category"] for row in rows} == set(selected)
    assert all(row["video_saved"] == "True" for row in rows)

    with pytest.raises(RuntimeError, match="Failed to promote success_mse_min"):
        namespace["_finalize_success_failure_mse_extremes"](runner, records, 2)


def test_eval_mse_extreme_finalizer_copies_four_audited_videos(tmp_path):
    import csv as csv_module
    import math as math_module
    import os as os_module
    import shutil as shutil_module

    runner_path = (
        ROOT / "third_party" / "rlinf_runtime" / "rlinf" / "runners" / "embodied_runner.py"
    )
    tree = ast.parse(runner_path.read_text())
    wanted = {
        "_select_success_failure_mse_extremes",
        "_finalize_post_update_eval_mse_extremes",
    }
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {
        "Path": Path,
        "csv": csv_module,
        "math": math_module,
        "os": os_module,
        "shutil": shutil_module,
    }
    exec(compile(module, "<eval-mse-finalizer>", "exec"), namespace)

    base_dir = tmp_path / "mse_extremes"
    evaluation_cfg = {
        "save_success_failure_mse_extremes": True,
        "require_all_mse_extreme_categories": True,
        "mse_extremes_base_dir": str(base_dir),
    }
    records = []
    for index, (success, mse) in enumerate(
        ((True, 0.4), (True, 0.1), (False, 0.2), (False, 0.7))
    ):
        source = tmp_path / f"source_{index}.mp4"
        source.write_bytes(f"video-{index}".encode())
        records.append(
            {
                "episode": 10 + index,
                "rank": index,
                "env_id": 0,
                "color": "brown",
                "success": success,
                "success_probability_max": 0.9 if success else 0.1,
                "trajectory_mse": mse,
                "comparison_video_path": str(source),
            }
        )

    runner = SimpleNamespace(
        global_step=20,
        _post_update_evaluation_cfg=lambda: evaluation_cfg,
        _select_success_failure_mse_extremes=namespace[
            "_select_success_failure_mse_extremes"
        ],
        _run_root=lambda: tmp_path,
        _atomic_write_json=lambda path, value: Path(path).write_text(
            yaml.safe_dump(value), encoding="utf-8"
        ),
    )
    metrics = namespace["_finalize_post_update_eval_mse_extremes"](
        runner, records
    )

    assert metrics["eval_mse_extremes_videos_saved"] == 4.0
    assert len(list(base_dir.glob("global_step_000020_*.mp4"))) == 4
    with (base_dir / "mse_extremes_summary.csv").open(newline="") as csv_file:
        rows = list(csv_module.DictReader(csv_file))
    assert {row["category"] for row in rows} == {
        "success_mse_min",
        "success_mse_max",
        "failure_mse_min",
        "failure_mse_max",
    }
    assert {int(row["episode"]) for row in rows} == {10, 11, 12, 13}


def test_only_eval_runs_one_post_update_eval_before_any_training_loop():
    source = (
        ROOT / "third_party" / "rlinf_runtime" / "rlinf" / "runners" / "embodied_runner.py"
    ).read_text()
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner"
    )
    run = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    only_eval_if = next(
        node
        for node in run.body
        if isinstance(node, ast.If) and "only_eval" in ast.unparse(node.test)
    )
    training_loop = next(node for node in run.body if isinstance(node, ast.For))
    assert only_eval_if.lineno < training_loop.lineno
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_evaluation_only"
        for node in ast.walk(only_eval_if)
    )

    eval_only = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_evaluation_only"
    )
    called_methods = {
        node.func.attr
        for node in ast.walk(eval_only)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_run_post_update_evaluation" in called_methods
    assert called_methods.isdisjoint({"interact", "generate", "run_training"})


def test_global_mse_extreme_mode_encodes_only_selected_rank_trajectories():
    worker_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "env"
        / "env_worker.py"
    )
    tree = ast.parse(worker_path.read_text())
    writer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "write_selected_mse_extreme_candidates"
    )
    assert not any(isinstance(node, ast.ExceptHandler) for node in ast.walk(writer))
    writer.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[]))
    namespace = {}
    exec(compile(module, "<global-mse-writer>", "exec"), namespace)

    encoded = []
    reset = []
    worker = SimpleNamespace(
        _rank=1,
        best_rollout_episode_rewards=[1.0],
        best_rollout_comparison_frames=[[object()]],
        _save_global_mse_extremes_only=lambda: True,
        _write_rollout_candidate_video=lambda **kwargs: encoded.append(kwargs),
        _reset_best_rollout_video_buffer=lambda: reset.append(True),
        log_on_first_rank=lambda _message: None,
    )
    selections = {
        "success_mse_min": {"rank": 0, "env_id": 0, "rollout_uid": 10},
        "success_mse_max": {"rank": 1, "env_id": 0, "rollout_uid": 11},
        "failure_mse_min": {"rank": 2, "env_id": 0, "rollout_uid": 12},
        "failure_mse_max": {"rank": 3, "env_id": 0, "rollout_uid": 13},
    }

    result = namespace["write_selected_mse_extreme_candidates"](
        worker, selections
    )

    assert result == {"mse_extreme_candidate_videos_written": 1}
    assert len(encoded) == 1
    assert encoded[0]["candidate_label"] == "success_mse_max"
    assert encoded[0]["rollout_uid"] == 11
    assert reset == [True]


def test_global_mse_extreme_mode_fails_if_selected_video_is_missing():
    runner_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "runners"
        / "embodied_runner.py"
    )
    tree = ast.parse(runner_path.read_text())
    writer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_write_selected_mse_extreme_candidates"
    )
    writer.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[]))
    namespace = {}
    exec(compile(module, "<global-mse-runner-writer>", "exec"), namespace)

    selected_record = {
        "rank": 0,
        "env_id": 0,
        "success": False,
        "trajectory_mse": 0.5,
    }
    runner = SimpleNamespace(
        cfg=SimpleNamespace(
            env=SimpleNamespace(
                train=SimpleNamespace(
                    video_cfg={"global_mse_extremes_only": True}
                )
            )
        ),
        _select_success_failure_mse_extremes=lambda _records: {
            "failure_mse_min": selected_record
        },
        env=SimpleNamespace(
            write_selected_mse_extreme_candidates=lambda _selections: SimpleNamespace(
                wait=lambda: [{"mse_extreme_candidate_videos_written": 0}]
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Expected 1 globally selected MSE-extreme candidate videos, wrote 0",
    ):
        namespace["_write_selected_mse_extreme_candidates"](
            runner, [selected_record]
        )


def test_env_metric_aggregation_rejects_string_metadata_with_metric_name():
    worker_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "env"
        / "env_worker.py"
    )
    tree = ast.parse(worker_path.read_text())
    concatenate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_concatenate_metric_lists"
    )
    concatenate.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[concatenate], type_ignores=[]))
    namespace = {"Any": object, "torch": torch}
    exec(compile(module, "<env-metric-aggregation>", "exec"), namespace)
    aggregate = namespace["_concatenate_metric_lists"]

    result = aggregate(
        {"reward/value": [torch.tensor(1.0), torch.tensor([2.0, 3.0])]},
        context="train rollout",
    )
    assert result["reward/value"].tolist() == [1.0, 2.0, 3.0]

    with pytest.raises(
        TypeError,
        match=r"train rollout metric 'reward/success_rule'.*found str",
    ):
        aggregate(
            {"reward/success_rule": ["terminal_positive_ratio"]},
            context="train rollout",
        )


def test_final_side_frame_metadata_survives_tensor_only_aggregation():
    import hashlib as hashlib_module

    import numpy as np_module

    worker_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "env"
        / "env_worker.py"
    )
    worker_tree = ast.parse(worker_path.read_text())
    worker_names = {
        "_encode_sha256_hex_batch",
        "_flush_best_rollout_video_candidate",
        "_concatenate_metric_lists",
    }
    worker_methods = {
        node.name: node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.FunctionDef) and node.name in worker_names
    }
    assert worker_methods.keys() == worker_names
    for method in worker_methods.values():
        method.decorator_list = []
    worker_module = ast.fix_missing_locations(
        ast.Module(
            body=[
                worker_methods["_encode_sha256_hex_batch"],
                worker_methods["_flush_best_rollout_video_candidate"],
                worker_methods["_concatenate_metric_lists"],
            ],
            type_ignores=[],
        )
    )
    worker_namespace = {"Any": object, "hashlib": hashlib_module, "torch": torch}
    exec(
        compile(worker_module, "<final-side-frame-worker-metadata>", "exec"),
        worker_namespace,
    )

    frames = torch.arange(2 * 3 * 4 * 3, dtype=torch.uint8).reshape(2, 3, 4, 3)
    digests = [
        hashlib_module.sha256(frame.numpy().tobytes(order="C")).hexdigest()
        for frame in frames
    ]
    worker = SimpleNamespace(
        best_rollout_episode_rewards=torch.tensor([1.0, 2.0]),
        best_rollout_chunk_rewards=[[1.0], [2.0]],
        best_rollout_chunk_mses=[[0.2], [0.7]],
        best_rollout_successes=torch.tensor([True, False]),
        best_rollout_final_side_frames=frames,
        best_rollout_final_side_frame_sha256=digests,
        best_rollout_final_side_frame_episodes=[41, 42],
        rollout_results=[],
        _save_success_failure_mse_extremes=lambda: False,
        _save_mse_extreme_final_frames=lambda: True,
        _save_rollout_final_side_frames=lambda: True,
        _cross_rank_group_cfg=lambda: None,
        _save_best_rollout_video=lambda: False,
        _save_worst_rollout_video=lambda: False,
        _save_global_mse_extremes_only=lambda: True,
        log_on_first_rank=lambda _message: None,
    )
    summary = worker_namespace["_flush_best_rollout_video_candidate"](worker)
    assert all(torch.is_tensor(value) for value in summary.values())
    assert summary["_best_rollout_final_side_frame_sha256"].shape == (2, 32)
    assert summary["_best_rollout_final_side_frame_sha256"].dtype == torch.uint8
    assert (
        summary["_best_rollout_final_side_frame_episode"].dtype == torch.int64
    )

    aggregated = worker_namespace["_concatenate_metric_lists"](
        {key: [value] for key, value in summary.items()},
        context="train rollout",
    )

    runner_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "runners"
        / "embodied_runner.py"
    )
    runner_tree = ast.parse(runner_path.read_text())
    pop_records = next(
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pop_best_rollout_records"
    )
    pop_records.decorator_list = []
    runner_module = ast.fix_missing_locations(
        ast.Module(body=[pop_records], type_ignores=[])
    )
    runner_namespace = {"hashlib": hashlib_module, "np": np_module}
    exec(
        compile(runner_module, "<final-side-frame-runner-metadata>", "exec"),
        runner_namespace,
    )
    records = runner_namespace["_pop_best_rollout_records"]([dict(aggregated)])
    assert len(records) == 2
    assert [record["episode"] for record in records] == [41, 42]
    assert [record["final_side_frame_sha256"] for record in records] == digests
    assert [record["success"] for record in records] == [True, False]
    assert np_module.array_equal(records[0]["_final_side_frame"], frames[0].numpy())

    aggregate = worker_namespace["_concatenate_metric_lists"]
    encode_sha256 = worker_namespace["_encode_sha256_hex_batch"]
    rank_results = []
    expected_digests = []
    for rank in range(32):
        rank_frames = torch.stack(
            [
                torch.full((3, 4, 3), rank * 4 + env_id, dtype=torch.uint8)
                for env_id in range(4)
            ]
        )
        rank_digests = [
            hashlib_module.sha256(frame.numpy().tobytes(order="C")).hexdigest()
            for frame in rank_frames
        ]
        expected_digests.extend(rank_digests)
        episodes = torch.arange(rank * 4, rank * 4 + 4, dtype=torch.int64)
        rank_summary = {
            "_best_rollout_episode_rewards": torch.arange(4, dtype=torch.float32),
            "_best_rollout_chunk_reward_max": torch.arange(4, dtype=torch.float32),
            "_best_rollout_chunk_reward_mean": torch.arange(4, dtype=torch.float32),
            "_best_rollout_trajectory_mse": torch.arange(4, dtype=torch.float32),
            "_best_rollout_success": episodes.remainder(2).eq(0),
            "_best_rollout_rollout_uid": episodes + 1000,
            "_best_rollout_global_group_id": episodes.div(
                4, rounding_mode="floor"
            ),
            "_best_rollout_group_member_id": episodes.remainder(4),
            "_best_rollout_reset_episode": episodes,
            "_best_rollout_final_side_frame": rank_frames,
            "_best_rollout_final_side_frame_sha256": encode_sha256(rank_digests),
            "_best_rollout_final_side_frame_episode": episodes,
        }
        rank_results.append(
            aggregate(
                {key: [value] for key, value in rank_summary.items()},
                context=f"train rollout rank {rank}",
            )
        )
    records_128 = runner_namespace["_pop_best_rollout_records"](rank_results)
    assert len(records_128) == 128
    assert [record["episode"] for record in records_128] == list(range(128))
    assert [
        record["final_side_frame_sha256"] for record in records_128
    ] == expected_digests
    assert [
        int(record["_final_side_frame"][0, 0, 0]) for record in records_128
    ] == list(range(128))


def test_success_rule_is_record_metadata_not_numeric_runtime_metric():
    env_worker = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "env"
        / "env_worker.py"
    ).read_text()
    actor_worker = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "actor"
        / "fsdp_actor_worker.py"
    ).read_text()
    assert 'env_info["reward/success_rule"]' not in env_worker
    assert 'record["success_decision_rule"] = decision["rule"]' in env_worker
    assert '"success_decision_rule": decision["rule"]' in actor_worker


def test_nest_four_cups_runtime_saves_four_mse_extreme_categories():
    runtime_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "examples"
        / "embodiment"
        / "config"
        / "ctrl_world_nest_four_cups_grpo_cosmos_fpo_native_unipc_b512.yaml"
    )
    runtime = yaml.safe_load(runtime_path.read_text())
    video_cfg = runtime["env"]["train"]["video_cfg"]
    assert video_cfg["save_success_failure_mse_extremes"] is True
    assert video_cfg["save_best_rollout"] is False
    assert video_cfg["save_worst_rollout"] is False
    assert runtime["env"]["train"]["max_episode_steps"] == 320
    assert runtime["actor"]["global_batch_size"] == 1280


def test_recent_views_rollout_is_opt_in_and_matches_standalone_contract():
    old_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "examples"
        / "embodiment"
        / "config"
        / "ctrl_world_nest_four_cups_grpo_cosmos_fpo_native_unipc_b512.yaml"
    )
    new_path = old_path.with_name(
        "ctrl_world_nest_four_cups_grpo_cosmos_fpo_native_unipc_b512_recent_views.yaml"
    )
    old_runtime = yaml.safe_load(old_path.read_text())
    new_runtime = yaml.safe_load(new_path.read_text())

    old_ctrl = old_runtime["env"]["train"]["ctrl_world_cfg"]
    new_ctrl = new_runtime["env"]["train"]["ctrl_world_cfg"]
    assert "model_camera_ids" not in old_ctrl
    assert old_ctrl["main_view_index"] == 0
    assert old_ctrl["wrist_view_index"] == 2
    assert new_ctrl["model_camera_ids"] == [2, 0, 1]
    assert new_ctrl["model_view_order"] == [
        "wrist:d435", "front:d405", "right:d405_1"
    ]
    assert new_ctrl["main_view_index"] == 1
    assert new_ctrl["wrist_view_index"] == 0
    assert new_ctrl["ctrl_world_history_idx"] == [-7, -6, -5, -4, -3, -2]
    assert old_ctrl["num_inference_steps"] == 20
    assert new_ctrl["num_inference_steps"] == 50
    assert "global_mse_extremes_only" not in old_runtime["env"]["train"]["video_cfg"]
    assert new_runtime["env"]["train"]["video_cfg"]["global_mse_extremes_only"] is True
    assert new_runtime["exp"]["ctrl_world"]["num_inference_steps"] == 50

    classifier = new_runtime["reward"]["success_classifier"]
    assert classifier["terminal_window_frames"] == 30
    assert classifier["frame_stride"] == 3
    assert classifier["minimum_terminal_positive_ratio"] == 0.8
    assert classifier["requires_last_frame_positive"] is True
    assert new_runtime["reward"]["success_model"]["frame_source"] == "ctrl_world_native"
    assert new_runtime["reward"]["success_model"]["artifact_name"] == "checkpoint_best.pt"
    assert new_runtime["actor"]["model"]["cosmos"]["camera_layout"] == "droid"
    assert new_runtime["env"]["train"]["self_feedback"] == {
        "enabled": True,
        "source": "imagined_video_chunk",
        "frame_index": -1,
    }

    config = load_config(
        ROOT / "configs" / "nest_four_cups_32n4g_b512_recent_views_u50.yaml"
    )
    overrides = RealTrainer(config)._overrides("/tmp/normalized-cosmos-config.yaml")
    assert config.ctrl_world.runtime.num_inference_steps == 50
    assert config.evaluation.save_video is False
    assert "exp.ctrl_world.num_inference_steps=50" in overrides
    assert "++env.train.ctrl_world_cfg.num_inference_steps=50" in overrides
    assert "++env.eval.ctrl_world_cfg.num_inference_steps=50" in overrides
    assert (
        "++env.train.ctrl_world_cfg.ctrl_world_history_idx=[-7,-6,-5,-4,-3,-2]"
        in overrides
    )
    assert "++reward.success_classifier.aggregation=terminal_positive_ratio" in overrides


def test_nest_four_cups_c21_keeps_four_optimizer_steps_and_672_step_horizon():
    config = load_config(ROOT / "configs" / "nest_four_cups_32n4g_b512_c21_u50.yaml")
    assert config.batch.chunks_per_trajectory == 21
    assert config.batch.trajectories_per_update * config.batch.chunks_per_trajectory == 10752
    assert config.batch.global_minibatch_chunks == 2688
    assert config.batch.gradient_accumulation_steps == 21
    assert config.batch.expected_optimizer_steps == 4
    overrides = RealTrainer(config)._overrides("/tmp/normalized-cosmos-config.yaml")
    assert "algorithm.trajectory_chunks=21" in overrides
    assert "algorithm.trajectory_records.trajectory_frames=672" in overrides
    assert "env.train.max_episode_steps=672" in overrides
    assert "env.eval.max_episode_steps=672" in overrides
    assert "env.train.video_cfg.save_best_rollout=false" not in overrides


def test_runtime_reset_contract_selects_red_from_validated_four_color_manifest():
    contract = load_duck_episode_split(
        str(ROOT / "configs" / "duck_split_manifest.json"),
        "training",
        verify_source_hashes=False,
    )
    episodes, pools, color_order = select_duck_episode_colors(contract, ["red"])
    assert color_order == ("red",)
    assert tuple(pools) == ("red",)
    assert episodes == contract.episodes_by_color["red"]
    assert len(episodes) == 40

    with pytest.raises(ValueError, match="absent from the validated manifest"):
        select_duck_episode_colors(contract, ["blue"])


def test_sparse_eval_accepts_empty_and_uneven_ranks():
    result = aggregate_sparse_evaluation(
        [[], [{"episode_id": 9}], [{"episode_id": 2}, {"episode_id": 5}], []]
    )
    assert [item["episode_id"] for item in result] == [2, 5, 9]


def test_sparse_eval_preserves_complete_four_rank_fsdp_shard_groups():
    # Ten eval episodes on 32 ranks must not activate only
    # ranks 8/9 of the [8 replicas, 4 shards] Cosmos HSDP mesh.
    official = tuple(range(90, 100))
    execution = _shard_aligned_eval_episode_ids(official, 4)
    assert execution == official + official[:2]
    active_world = CommMapper.get_collective_aligned_world_size(
        len(execution), 32, 4
    )
    assert active_world == 12
    sizes = [
        (
            CommMapper.get_rank_batch_size(len(execution), active_world, rank)
            if rank < active_world
            else 0
        )
        for rank in range(32)
    ]
    assert sizes == [1] * 12 + [0] * 20
    assert sum(sizes) == 12
    for shard_start in range(0, active_world, 4):
        assert all(sizes[rank] == 1 for rank in range(shard_start, shard_start + 4))


def test_rollout_sparse_eval_batch_deactivates_unused_ranks_when_evenly_divisible():
    source = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "rollout"
        / "hf"
        / "huggingface_worker.py"
    ).read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_sparse_eval_batch_size"
    )
    namespace = {"CommMapper": CommMapper}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
    get_batch_size = namespace["_get_sparse_eval_batch_size"]

    sizes = [get_batch_size(20, 20, rank, 1) for rank in range(128)]
    assert sizes == [1] * 20 + [0] * 108
    assert sum(sizes) == 20


def test_sparse_eval_worker_does_not_enter_global_barrier():
    # Inactive ranks used to time out in a global barrier
    # while active shard-aligned ranks were still executing held-out episodes.
    source = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "workers"
        / "rollout"
        / "hf"
        / "huggingface_worker.py"
    ).read_text()
    tree = ast.parse(source)
    evaluate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "evaluate"
    )
    barrier_calls = [
        node
        for node in ast.walk(evaluate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "barrier"
    ]
    assert barrier_calls == []
    assert "_synchronize_sparse_eval_workers" not in source


def test_production_submission_has_ten_dependency_segments():
    config = load_config(ROOT / "configs" / "close.yaml")
    assert _segments(config) == [(index, index + 5) for index in range(0, 50, 5)]


def test_nest_four_cups_submission_can_stop_at_25_and_resume_at_25():
    config = load_config(ROOT / "configs" / "nest_four_cups_32n4g_b512_u50.yaml")
    assert _segments(config, until_update=25) == [
        (index, index + 5) for index in range(0, 25, 5)
    ]
    assert _segments(config, start_update=25, until_update=50) == [
        (index, index + 5) for index in range(25, 50, 5)
    ]
    with pytest.raises(ValueError, match="align to segment width"):
        _segments(config, until_update=23)


def test_duck_smoke_evaluates_only_after_resumed_update_two():
    config = load_config(ROOT / "configs" / "duck_smoke_2n4g.yaml")
    # Update 1 must checkpoint and exit before the required final eval.
    assert config.runtime.segment_updates == 1
    assert config.evaluation.enabled
    assert config.evaluation.interval_updates == 2


def test_checkpoint_each_update_and_evaluate_at_update_two_are_independent():
    # Update 1 must save without eval; update 2 must do both.
    namespace = runpy.run_path(
        str(ROOT / "third_party" / "rlinf_runtime" / "rlinf" / "utils" / "runner_utils.py")
    )
    check_progress = namespace["check_progress"]
    assert check_progress(1, 2, 2, 1, 1.0) == (False, True, False)
    assert check_progress(2, 2, 2, 1, 1.0) == (True, True, True)


def test_vendored_algorithm_initializer_does_not_import_pruned_toolcall_stack():
    path = ROOT / "third_party/rlinf_runtime/rlinf/algorithms/__init__.py"
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "toolcall_parsers" not in imported


@pytest.mark.parametrize(
    "name",
    [
        "close_smoke_2n4g.yaml",
        "duck_smoke_2n4g.yaml",
        "nest_four_cups_32n4g_b512_u50.yaml",
    ],
)
def test_real_bridge_marks_compatibility_only_hydra_keys(tmp_path, name):
    config = load_config(ROOT / "configs" / name)
    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, output_dir=str(tmp_path / name)),
    )
    overrides = RealTrainer(config)._overrides("/tmp/normalized_cosmos_config.yaml")
    assert any(value.startswith("++actor.model.cosmos.training_config_path=") for value in overrides)
    assert any(value.startswith("++actor.model.cosmos.vlm_processor_path=") for value in overrides)
    assert any(value.startswith("actor.model.cosmos.output_dir=") for value in overrides)
    assert "++env.train.ctrl_world_cfg.guidance_scale=1.0" in overrides
    assert "++env.eval.ctrl_world_cfg.guidance_scale=1.0" in overrides
    for mode in ("train", "eval"):
        assert f"++env.{mode}.ctrl_world_cfg.chunk={config.ctrl_world.runtime.policy_chunk}" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.ctrl_world_chunk={config.ctrl_world.runtime.resampled_chunk}" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.decode_chunk_size={config.ctrl_world.runtime.decode_chunk_size}" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.cosmos_action_fps={config.ctrl_world.runtime.policy_action_fps}" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.per_view_vae_codec=true" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.use_raw_reset_policy_views=true" in overrides
    assert f"actor.model.cosmos.guidance={config.cosmos.guidance}" in overrides
    assert (
        f"env.eval.video_cfg.save_video={str(config.evaluation.save_video).lower()}"
        in overrides
    )
    assert f"++algorithm.fpo_ratio_granularity={config.fpo.ratio_granularity}" in overrides
    assert (
        f"++actor.model.cosmos.fpo_score_parameterization={config.fpo.score_parameterization}"
        in overrides
    )
    assert f"actor.optim.lr={config.optimizer.lr}" in overrides
    assert f"exp.optim.lr={config.optimizer.lr}" in overrides
    assert "reward.video_similarity.enabled=true" in overrides
    assert "reward.terminal_goal.enabled=true" in overrides
    rollout_steps = config.batch.chunks_per_trajectory * config.cosmos.action_chunk
    assert f"env.train.max_steps_per_rollout_epoch={rollout_steps}" in overrides
    assert f"env.eval.max_steps_per_rollout_epoch={rollout_steps}" in overrides
    assert (
        f"++rollout.sparse_eval_collective_group_size={config.cosmos.data_parallel_shard_degree}"
        in overrides
    )
    # Scalar records no longer depend on excluded audit manifests.
    assert "algorithm.trajectory_records.enabled=true" in overrides
    assert "++algorithm.trajectory_records.provenance_mode=lightweight_analysis" in overrides
    assert any(
        value.startswith("algorithm.trajectory_records.output_dir=") for value in overrides
    )
    # The vendored Ctrl-World code path is not an asset root.
    assert f"env.train.ctrl_world_svd_model_path={config.assets.svd_model_path}" in overrides
    assert f"env.train.ctrl_world_clip_model_path={config.assets.clip_model_path}" in overrides
    assert f"env.eval.ctrl_world_svd_model_path={config.assets.svd_model_path}" in overrides
    assert f"env.eval.ctrl_world_clip_model_path={config.assets.clip_model_path}" in overrides
    if config.task.profile in {"duck", "nest_four_cups"}:
        assert f"++runner.checkpoint_expected_distcp_shards={config.world_size}" in overrides
        groups_per_color = (
            config.batch.trajectories_per_update
            // config.algorithm.group_size
            // len(config.task.active_variants)
        )
        assert (
            f"algorithm.cross_rank_group.episodes_per_color_per_update={groups_per_color}"
            in overrides
        )
        assert (
            f"algorithm.trajectory_records.split_manifest={config.task.split_manifest_path}"
            in overrides
        )
        execution_override = next(
            value
            for value in overrides
            if value.startswith("++duck.evaluation.execution_episode_ids=")
        )
        execution_values = execution_override.split("=", 1)[1].strip("[]").split(",")
        official_count = (
            len(config.task.active_variants) * config.evaluation.episodes_per_variant
        )
        shard_degree = config.cosmos.data_parallel_shard_degree
        expected_count = official_count + (-official_count) % shard_degree
        assert len(execution_values) == expected_count
        assert f"env.eval.total_num_envs={expected_count}" in overrides
        assert f"exp.evaluation.total_num_envs={expected_count}" in overrides


def test_checkpoint_precedes_post_update_eval_in_runner():
    source = (ROOT / "third_party/rlinf_runtime/rlinf/runners/embodied_runner.py").read_text()
    progress = source.index("run_val, save_model, is_train_end = check_progress(")
    checkpoint = source.index("self._save_checkpoint()", progress)
    evaluation = source.index("if run_val:", progress)
    assert progress < checkpoint < evaluation


def test_resume_completes_missing_checkpoint_boundary_eval_before_training():
    source = (
        ROOT / "third_party/rlinf_runtime/rlinf/runners/embodied_runner.py"
    ).read_text()
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner"
    )
    run = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    resume_if = next(
        node
        for node in run.body
        if isinstance(node, ast.If)
        and "_resume_requires_post_update_eval" in ast.unparse(node.test)
    )
    training_loop = next(node for node in run.body if isinstance(node, ast.For))
    assert resume_if.lineno < training_loop.lineno
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_post_update_evaluation"
        for node in ast.walk(resume_if)
    )


def test_real_trainer_runs_strict_analysis_after_success(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs" / "close_smoke_2n4g.yaml")
    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, output_dir=str(tmp_path / "run")),
    )
    trainer = RealTrainer(config)
    monkeypatch.delenv("RLINF_SEGMENT_START", raising=False)
    monkeypatch.delenv("RLINF_SEGMENT_END", raising=False)
    monkeypatch.setattr(
        trainer,
        "preflight",
        lambda: {"normalized_checkpoint_config": "/tmp/normalized.yaml"},
    )
    monkeypatch.setattr(
        "rlinf_modified.engine.real.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    calls = []
    monkeypatch.setattr(
        "rlinf_modified.analysis.analyze_run",
        lambda run_dir, require_plots: calls.append((run_dir, require_plots))
        or {"complete_steps": [1]},
    )
    result = trainer.run()
    assert calls == [(tmp_path / "run", True)]
    assert result["analysis"]["complete_steps"] == [1]


def test_ray_daemon_inherits_vendored_runtime_before_start():
    launcher = (ROOT / "scripts/launch_ray_node.sh").read_text()
    export_index = launcher.index("third_party/rlinf_runtime")
    head_start_index = launcher.index("ray start --head")
    worker_start_index = launcher.index("ray start --address")
    assert export_index < head_start_index
    assert export_index < worker_start_index


def test_scratch_cleanup_stops_ray_before_bounded_retry():
    launcher = (ROOT / "scripts" / "launch_ray_node.sh").read_text()
    preserve_index = launcher.index('preserve_failure_diagnostics "${effective_exit_code}"')
    ray_stop_index = launcher.index("ray stop --force")
    retry_index = launcher.index("for _cleanup_attempt in {1..10}")
    remove_index = launcher.index('rm -rf -- "${SCRATCH_DIR}"')
    assert preserve_index < ray_stop_index < retry_index < remove_index
    # A stray argument on `done` (e.g. `done 5`) runs a command named "5", turning a
    # successful job into a spurious exit-127 failure.
    loop_terminators = [
        line.strip() for line in launcher.splitlines() if line.lstrip().startswith("done")
    ]
    assert loop_terminators
    assert set(loop_terminators) == {"done"}


def test_failed_allocation_preserves_each_nodes_ray_logs():
    launcher = (ROOT / "scripts" / "launch_ray_node.sh").read_text()
    assert '${OUTPUT_DIR}/diagnostics/job_${SLURM_JOB_ID}/node_${RLINF_NODE_RANK}' in launcher
    assert 'find "${ray_logs}" -maxdepth 1 -type f -print0' in launcher
    assert 'cp -a -- "${ray_log}" "${diagnostic_dir}/ray_logs/"' in launcher
    # Worker launchers normally exit zero after observing the head-node marker;
    # they must inherit a non-zero allocation result so their logs are retained.
    assert 'effective_exit_code="$(allocation_exit_code "${exit_code}")"' in launcher


def test_launcher_emits_full_errors_without_ctrl_world_progress_spam():
    launcher = (ROOT / "scripts" / "launch_ray_node.sh").read_text()
    assert "export PYTHONUNBUFFERED=1" in launcher
    assert "export HYDRA_FULL_ERROR=1" in launcher
    assert "export TQDM_DISABLE=1" in launcher


def test_launchers_use_pytorch_210_flight_recorder_variable():
    submit = (ROOT / "slurm" / "submit.sbatch").read_text()
    real_bridge = (ROOT / "src" / "rlinf_modified" / "engine" / "real.py").read_text()
    assert "TORCH_FR_BUFFER_SIZE" in submit
    assert "TORCH_FR_BUFFER_SIZE" in real_bridge
    assert "TORCH_NCCL_TRACE_BUFFER_SIZE" not in submit
    assert "TORCH_NCCL_TRACE_BUFFER_SIZE" not in real_bridge


def test_vendored_scheduler_does_not_set_deprecated_record_streams_variable():
    scheduler = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "rlinf"
        / "scheduler"
        / "hardware"
        / "accelerators"
        / "nvidia_gpu.py"
    ).read_text()
    assert 'env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"]' not in scheduler



def test_close_native_runtime_shares_train_eval_ctrl_world_contract():
    runtime_path = (
        ROOT
        / "third_party"
        / "rlinf_runtime"
        / "examples"
        / "embodiment"
        / "config"
        / "ctrl_world_close_laptop_grpo_cosmos_native_unipc_large.yaml"
    )
    runtime = yaml.safe_load(runtime_path.read_text())
    train_ctrl = runtime["env"]["train"]["ctrl_world_cfg"]
    eval_ctrl = runtime["env"]["eval"]["ctrl_world_cfg"]
    assert eval_ctrl == train_ctrl
    assert eval_ctrl["use_raw_reset_policy_views"] is True
    assert eval_ctrl["chunk"] == 32
    assert eval_ctrl["cosmos_action_fps"] == 15
    assert eval_ctrl["per_view_vae_codec"] is True


@pytest.mark.parametrize(
    ("name", "standalone_source", "runtime_source", "diagnostic_only", "replace_reward"),
    [
        (
            "close_b128_mb32_u15_traj_reward.yaml",
            "trajectory_mse",
            "trajectory_mse",
            True,
            False,
        ),
        (
            "close_b128_mb32_u15_terminal_mse_reward.yaml",
            "terminal_goal_mse",
            "terminal_goal_mse",
            True,
            False,
        ),
        (
            "close_b128_mb32_u15_success_reward.yaml",
            "success_binary",
            "success_binary",
            False,
            True,
        ),
    ],
)
def test_close_reward_ablation_contracts_and_sparse_eval(
    tmp_path,
    monkeypatch,
    name,
    standalone_source,
    runtime_source,
    diagnostic_only,
    replace_reward,
):
    config = load_config(ROOT / "configs" / name)
    assert config.runtime.max_updates == 15
    assert config.runtime.segment_updates == 5
    assert config.runtime.output_dir.startswith(
        "/path/to/rlinf_release/outputs/"
    )
    assert config.slurm.nodes == 8
    assert config.slurm.gpus_per_node == 4
    assert config.batch.trajectories_per_update == 128
    assert config.batch.global_minibatch_chunks == 160
    assert config.batch.gradient_accumulation_steps == 5
    assert config.batch.expected_optimizer_steps == 4
    assert config.reward.source == standalone_source
    assert config.reward.video_similarity.enabled
    assert config.reward.terminal_goal.enabled
    assert config.reward.terminal_goal.same_episode is True
    assert config.reward.terminal_goal.frame_index == -1
    assert config.reward.terminal_goal.window_size == 4
    assert config.reward.terminal_goal.view_weights == pytest.approx((2 / 3, 1 / 6, 1 / 6))
    assert config.reward.success_classifier.enabled
    assert config.reward.success_classifier.aggregation == "max"
    assert config.reward.success_classifier.diagnostic_only is diagnostic_only
    assert config.reward.success_classifier.replace_training_reward is replace_reward
    assert config.evaluation.enabled
    assert config.evaluation.interval_updates == 5
    assert config.evaluation.episodes_per_variant == 8
    assert config.evaluation.save_video is False
    assert config.checkpoint.enabled is True
    assert config.checkpoint.interval_updates == 5

    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime,
            output_dir=str(tmp_path / standalone_source),
        ),
    )
    monkeypatch.delenv("RLINF_SEGMENT_START", raising=False)
    monkeypatch.delenv("RLINF_SEGMENT_END", raising=False)
    overrides = RealTrainer(config)._overrides("/tmp/normalized-cosmos-config.yaml")
    assert "runner.save_interval=5" in overrides
    assert "runner.checkpoint_milestone_interval=15" in overrides
    assert "++runner.checkpoint_keep_latest_recovery=1" in overrides
    assert "reward.source=video_similarity" in overrides
    assert f"++reward.training_source={runtime_source}" in overrides
    assert (
        "reward.success_classifier.diagnostic_only="
        f"{str(diagnostic_only).lower()}"
    ) in overrides
    assert (
        "reward.success_classifier.replace_training_reward="
        f"{str(replace_reward).lower()}"
    ) in overrides
    assert "++reward.success_classifier.reward_value=1.0" in overrides
    assert "++reward.terminal_goal.reference_mode=same_episode_last_frame" in overrides
    assert "++reward.terminal_goal.strict_same_episode=true" in overrides
    assert "reward.terminal_goal.dataset_frame_index=-1" in overrides
    assert "reward.terminal_goal.view_weights.main=0.6666666666666666" in overrides
    assert "reward.terminal_goal.view_weights.wrist=0.16666666666666666" in overrides
    assert "reward.terminal_goal.view_weights.extra=0.16666666666666666" in overrides
    assert (
        "++post_update_evaluation.episode_ids=[6,7,14,20,26,31,32,37]"
        in overrides
    )
    assert "++post_update_evaluation.expected_records_per_step=8" in overrides
    assert "++post_update_evaluation.seed_base=42" in overrides
    assert "env.eval.specific_reset_id=[6,7,14,20,26,31,32,37]" in overrides
    assert "env.eval.total_num_envs=8" in overrides
    assert "exp.evaluation.total_num_envs=8" in overrides
    assert "env.eval.video_cfg.save_video=false" in overrides
    for mode in ("train", "eval"):
        assert f"++env.{mode}.ctrl_world_cfg.use_raw_reset_policy_views=true" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.chunk=32" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.ctrl_world_chunk=64" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.decode_chunk_size=7" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.cosmos_action_fps=15.0" in overrides
        assert f"++env.{mode}.ctrl_world_cfg.per_view_vae_codec=true" in overrides
    assert any(
        value.startswith("++reward.success_models.close.sha256=")
        for value in overrides
    )
