"""Resolve and audit the actual Hydra contracts for both comparison arms."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from rlinf_modified.config import load_config
from rlinf_modified.engine.real import RealTrainer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/push_t_view_weight_ablation_fp32_h13_20260904'
os.environ.setdefault('VENV_PATH', str(ROOT / '.venv_aarch64'))
results = {}
for arm in ('original', 'no_wrist'):
    path = ROOT / 'configs' / f'push_t_combined_mse_fp32_h13_{arm}_15u_s5.yaml'
    cfg = load_config(path)
    trainer = RealTrainer(cfg)
    report = trainer.preflight()
    entry = ROOT / 'third_party/rlinf_runtime/examples/embodiment/train_embodied_agent.py'
    command = [sys.executable, str(entry), '--config-path', str(entry.parent / 'config'),
               '--config-name', cfg.assets.legacy_config_name, '--cfg', 'job', '--resolve',
               *trainer._overrides(report['normalized_checkpoint_config'])]
    result = subprocess.run(command, cwd=entry.parent, text=True, capture_output=True, check=False)
    (OUT / f'{arm}.composition.stderr.log').write_text(result.stderr)
    (OUT / f'{arm}.hydra.yaml').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{arm}: Hydra resolution failed: {result.stderr[-2500:]}')
    resolved = yaml.safe_load(result.stdout)
    for mode in ('train', 'eval'):
        assert resolved['env'][mode]['max_episode_steps'] == 416
        assert resolved['env'][mode]['max_steps_per_rollout_epoch'] == 416
        ctrl = resolved['env'][mode]['ctrl_world_cfg']
        assert (ctrl['main_view_index'], ctrl['wrist_view_index'], ctrl['side_view_index']) == (1, 0, 2)
        assert ctrl['adapter_translation_gain'] == ctrl['adapter_rotation_gain'] == 1.0
        assert ctrl['num_inference_steps'] == 50
    reward = resolved['reward']
    assert reward['training_source'] == 'continuous_combined'
    assert reward['video_similarity']['alignment_mode'] == 'aligned'
    assert reward['video_similarity']['camera_layout'] == 'droid'
    assert reward['terminal_goal']['ctrl_world_source_mapping'] == {'main':'wrist','wrist':'main','extra':'extra'}
    physical_trajectory = {('side' if k == 'extra' else k): v for k,v in reward['video_similarity']['view_weights'].items()}
    physical_terminal = {('side' if k == 'extra' else k): reward['terminal_goal']['view_weights'][goal]
                         for goal,k in reward['terminal_goal']['ctrl_world_source_mapping'].items()}
    assert physical_trajectory == physical_terminal == cfg.reward.video_similarity.view_weights
    assert reward['terminal_goal']['reward_scale'] == 416.0
    assert resolved['duck']['evaluation']['fixed_seeds'] and resolved['duck']['evaluation']['before_training']
    assert resolved['actor']['global_batch_size'] == 416
    assert resolved['algorithm']['trajectory_chunks'] == 13
    # Cross-rank seed schedules must use the same complete trajectory length.
    assert resolved['algorithm']['cross_rank_group']['chunks_per_trajectory'] == 13
    results[arm] = {'state':'passed', 'physical_trajectory_weights': physical_trajectory,
                    'physical_terminal_weights': physical_terminal,
                    'horizon_seconds': 416 / 15, 'validation_episodes': resolved['duck']['evaluation']['episode_ids'],
                    'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'record_dir': resolved['duck']['evaluation']['record_dir']}
    print(arm, 'actual Hydra contract passed', flush=True)
(OUT / 'composition_validation.json').write_text(json.dumps(results, indent=2)+'\n')
