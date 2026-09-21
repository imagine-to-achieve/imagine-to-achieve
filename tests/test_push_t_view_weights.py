"""Numerical and runtime-contract checks for the Push-T camera ablation."""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.rewards.video_similarity import compute_video_similarity_reward, compute_terminal_goal_reward
from rlinf_modified.config import ConfigError, load_config
from rlinf_modified.engine.real import RealTrainer

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = {"wrist": 2 / 3, "main": 1 / 6, "extra": 1 / 6}
NO_WRIST = {"wrist": 0.0, "main": 0.5, "extra": 0.5}


def _reward(wrist, weights, mask=None):
    # Actual input is three physical views, exercising the production stitch.
    imagined = torch.zeros(2, 3, 3, 720, 640)
    main = torch.full((2, 3, 480, 640, 3), 0.2)
    side = torch.full_like(main, 0.4)
    wrist_video = torch.full_like(main, wrist)
    return compute_video_similarity_reward(
        imagined, main, size=(192, 320),
        ctrl_world_video_chunk_wrist=wrist_video,
        ctrl_world_video_chunk_extra=side,
        alignment_mode="aligned", camera_layout="droid",
        view_weights=weights, mask=mask, return_stats=True,
    )


def test_original_weights_reproduce_pixel_average_after_production_resize():
    raw, raw_stats = _reward(0.9, None)
    weighted, stats = _reward(0.9, ORIGINAL)
    torch.testing.assert_close(weighted, raw, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(stats["pixel_per_frame_mse"], raw_stats["per_frame_mse"])


def test_zero_wrist_is_invariant_to_wrist_error_with_real_view_alignment():
    # Use an exact-size composite to avoid cross-view interpolation at a border.
    imagined = torch.zeros(2, 3, 3, 192, 320)
    reference = torch.zeros_like(imagined)
    reference[:, :, :, 128:, :160] = 0.2
    reference[:, :, :, 128:, 160:] = 0.4
    kwargs = dict(camera_layout="droid", composite_view_stats=True, view_weights=NO_WRIST)
    first = compute_video_similarity_reward(imagined, reference, **kwargs)
    reference[:, :, :, :128] = 1.0
    second = compute_video_similarity_reward(imagined, reference, **kwargs)
    torch.testing.assert_close(first, torch.full((2, 3), -0.1))
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    original = compute_video_similarity_reward(imagined, reference, **{**kwargs, "view_weights": ORIGINAL})
    assert (original < second).all()
    aligned_low, _ = _reward(0.0, NO_WRIST)
    aligned_high, _ = _reward(1.0, NO_WRIST)
    torch.testing.assert_close(aligned_low, aligned_high, rtol=0, atol=0)


def test_weighted_mask_ignores_invalid_frames_and_handles_empty_mask():
    mask = torch.tensor([[True, False, True], [False, False, False]])
    rewards, stats = _reward(0.9, NO_WRIST, mask)
    assert (rewards[~mask] == 0).all()
    torch.testing.assert_close(stats["mse_mean"], -rewards[mask].mean())
    rewards, stats = _reward(0.9, NO_WRIST, torch.zeros_like(mask))
    assert (rewards == 0).all()
    assert stats["mse_mean"].item() == 0


@pytest.mark.parametrize("weights", [
    {"wrist": -0.1, "main": 0.6, "extra": 0.5},
    {"wrist": float("nan"), "main": 0.5, "extra": 0.5},
    {"wrist": 0.0, "main": 0.0, "extra": 0.0},
    {"wrist": 0.0, "main": 0.5, "side": 0.5},
])
def test_invalid_runtime_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="[Ww]eight"):
        _reward(0.9, weights)


def _method(relative, class_name, method_name, namespace):
    module = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    code = ast.Module(body=[method], type_ignores=[])
    exec(compile(code, relative, "exec"), namespace)
    return namespace[method_name]


def test_terminal_worker_remaps_legacy_goal_keys_to_physical_views():
    method = _method(
        "third_party/rlinf_runtime/rlinf/workers/env/env_worker.py", "EnvWorker",
        "_apply_terminal_goal_reward", {"torch": torch, "EnvOutput": Any, "Any": Any,
                                      "compute_terminal_goal_reward": compute_terminal_goal_reward},
    )
    cfg = OmegaConf.create({"reward": {"terminal_goal": {
        "view_weights": {"main": 0.0, "wrist": 0.5, "extra": 0.5},
        "ctrl_world_source_mapping": {"main": "wrist", "wrist": "main", "extra": "extra"},
        "reference_mode": "same_episode_last_frame", "window_size": 4,
        "size": [8, 12], "reward_scale": 416.0,
    }}})
    reference = {name: torch.zeros(1, 3, 8, 12) for name in ("main", "wrist", "extra")}
    reference["metadata"] = {"episode_index": 6, "frame_index": 365}
    dummy = SimpleNamespace(cfg=cfg, _terminal_goal_reward_enabled=lambda: True,
                            _get_terminal_goal_reference=lambda episode: reference)
    videos = {name: torch.full((2, 4, 8, 12, 3), level)
              for name, level in (("main", .2), ("wrist", .9), ("extra", .4))}
    rewards = torch.zeros(2, 32)
    dones = torch.zeros_like(rewards, dtype=torch.bool); dones[0, -1] = True
    info = {}
    combined, terminal = method(dummy, rewards, SimpleNamespace(dones=dones, reset_state_ids=torch.tensor([6, 0])), info,
                               ctrl_world_video_chunk=videos["main"],
                               ctrl_world_video_chunk_wrist=videos["wrist"],
                               ctrl_world_video_chunk_extra=videos["extra"])
    assert terminal[0, -1].item() == pytest.approx(-41.6, abs=1e-4)
    assert combined[1].count_nonzero() == 0
    for key, expected in (("main", .04), ("wrist", .81), ("extra", .16)):
        assert info[f"reward/terminal_goal_{key}_mse"].item() == pytest.approx(expected, abs=1e-6)


def test_fixed_eval_stream_includes_pretraining_and_keeps_legacy_default():
    helper = _method("third_party/rlinf_runtime/rlinf/runners/embodied_runner.py", "EmbodiedRunner",
                     "_post_update_eval_seed_base", {})
    for step in (0, 5, 10, 15):
        dummy = SimpleNamespace(global_step=step)
        cfg = {"seed_base": 42, "episode_ids": list(range(30)), "fixed_seeds": True}
        base = helper(dummy, cfg)
        assert [base + 30 * (step - 1) + i for i in range(30)] == list(range(42, 72))
        cfg["fixed_seeds"] = False
        assert helper(dummy, cfg) == 42


def test_ablation_profiles_only_differ_in_names_paths_and_both_view_weights(monkeypatch):
    configs = [load_config(ROOT / "configs" / f"push_t_combined_mse_fp32_h13_{arm}_15u_s5.yaml")
               for arm in ("original", "no_wrist")]
    left, right = [c.to_dict() for c in configs]
    for value in (left, right):
        value.pop("experiment_name"); value["runtime"].pop("output_dir")
        value["reward"]["video_similarity"].pop("view_weights")
        value["reward"]["terminal_goal"].pop("view_weights")
    assert left == right
    assert configs[0].batch.chunks_per_trajectory * 32 / 15 > 26.94
    for name in ("RLINF_SEGMENT_START", "RLINF_SEGMENT_END"):
        monkeypatch.delenv(name, raising=False)
    for cfg in configs:
        trainer = RealTrainer.__new__(RealTrainer)
        trainer.config, trainer.root, trainer.run_dir = cfg, ROOT, Path(cfg.runtime.output_dir)
        overrides = trainer._overrides("/tmp/normalized-cosmos-config.yaml")
        assert "env.train.max_episode_steps=416" in overrides
        assert "env.eval.max_episode_steps=416" in overrides
        assert "++algorithm.cross_rank_group.chunks_per_trajectory=13" in overrides
        assert "++duck.evaluation.fixed_seeds=true" in overrides
        assert "++duck.evaluation.before_training=true" in overrides
        for physical, runtime in (("wrist", "wrist"), ("main", "main"), ("side", "extra")):
            assert f"++reward.video_similarity.view_weights.{runtime}={cfg.reward.video_similarity.view_weights[physical]}" in overrides
        for name, value in zip(("main", "wrist", "extra"), cfg.reward.terminal_goal.view_weights):
            assert f"reward.terminal_goal.view_weights.{name}={value}" in overrides
    legacy = load_config(ROOT / "configs/push_t_combined_mse_ctrl50_8n4g_base50_s5.yaml")
    assert legacy.reward.video_similarity.view_weights is None
    assert legacy.evaluation.fixed_seeds is False


@pytest.mark.parametrize('matrix_dones', [False, True])
def test_view_sidecar_keeps_each_sample_and_terminal_identity(tmp_path, matrix_dones, monkeypatch):
    import json
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'node'))
    monkeypatch.setenv('SLURM_JOB_ID', 'testjob')
    method = _method('third_party/rlinf_runtime/rlinf/workers/env/env_worker.py', 'EnvWorker',
                     '_record_view_reward_diagnostics', {'torch': torch, 'Path': Path, 'json': json})
    cfg = OmegaConf.create({'algorithm': {'trajectory_records': {
        'enabled': True, 'output_dir': str(tmp_path / 'trajectory_records')
    }}})
    dummy = SimpleNamespace(cfg=cfg, global_step=2, _rank=3)
    done = torch.tensor([True, False])
    if matrix_dones:
        done = done[:, None].expand(-1, 32)
    env = SimpleNamespace(dones=done, reset_state_ids=torch.tensor([6, 0]))
    stats = {f'{key}_per_frame_mse': torch.full((2, 32), value)
             for key,value in (('main', .04), ('wrist', .81), ('extra', .16), ('pixel', .5))}
    stats['per_frame_mse'] = torch.full((2, 32), .1)
    info = {'reward/terminal_goal_main_mse': torch.tensor([.04])}
    method(dummy, stats, env, info, {'chunk_index': torch.tensor([12,12])})
    result = json.loads((tmp_path / 'reward_view_records/rank_003_job_testjob.jsonl').read_text())
    assert result['episode_ids'] == [6, 0]
    assert result['done'] == [True, False]
    assert result['metadata']['chunk_index'] == [12,12]
    assert result['trajectory_view_mse']['side'] == pytest.approx([.16,.16])
    assert result['terminal_view_mse_done']['main'] == pytest.approx([.04])
