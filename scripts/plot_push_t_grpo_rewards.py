"""Plot native GRPO reward inputs from committed rollout logs and records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from plot_push_t_weight_ablation_steps import ARMS, COLORS, LABELS, ROOT, DEFAULT, style, save_figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

FIELDS = ('trajectory_reward', 'lastframe_reward', 'combined_reward')
METRICS = ('rollout/episode_reward_mean', 'rollout/rewards')


def collect():
    snapshot = {'captured_at': datetime.now(timezone.utc).isoformat(), 'arms': {}}
    plotted = []
    for arm in ARMS:
        run = ROOT / 'outputs' / f'push_t_combined_mse_fp32_h13_{arm}_15u_s5'
        status_bytes = (run / 'status.json').read_bytes()
        status = json.loads(status_bytes)
        log_bytes = (run / 'metrics/rollout.csv').read_bytes()
        logs = {}
        for row in csv.DictReader(io.StringIO(log_bytes.decode())):
            if row['metric'] in METRICS:
                key = int(row['step'])
                metrics = logs.setdefault(key, {})
                value = float(row['value'])
                if row['metric'] in metrics and metrics[row['metric']] != value:
                    raise ValueError(f'Conflicting logged reward at {arm}, {key}')
                metrics[row['metric']] = value
        source = {'run_dir': str(run), 'status': status,
                  'status_sha256': hashlib.sha256(status_bytes).hexdigest(),
                  'rollout_csv_sha256': hashlib.sha256(log_bytes).hexdigest(),
                  'steps': [], 'excluded_steps': []}
        for step, metrics in sorted(logs.items()):
            if step > status['completed_updates'] or set(metrics) != set(METRICS):
                source['excluded_steps'].append(step)
                continue
            record_path = run / 'trajectory_records' / f'update_{step-1:04d}.jsonl'
            payload = record_path.read_bytes()
            rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
            assert len(rows) == 128, (arm, step, len(rows))
            assert all(r['valid'] is True and r['update'] == step - 1 for r in rows)
            assert {r['training_reward_source'] for r in rows} == {'continuous_combined'}
            assert {r['trajectory_frames'] for r in rows} == {416}
            assert len({r['rollout_uid'] for r in rows}) == 128
            groups = Counter(r['group_id'] for r in rows)
            assert len(groups) == 8 and set(groups.values()) == {16}
            assert all(len(r['chunk_rewards']) == 13 for r in rows)
            assert all(math.isfinite(float(r[k])) for r in rows for k in FIELDS)
            means = {key: fmean(r[key] for r in rows) for key in FIELDS}
            audit = {
                'max_record_component_sum_error': max(abs(r['trajectory_reward'] + r['lastframe_reward'] - r['combined_reward']) for r in rows),
                'max_record_chunk_sum_error': max(abs(sum(r['chunk_rewards']) - r['combined_reward']) for r in rows),
                'episode_mean_log_error': abs(means['combined_reward'] - metrics['rollout/episode_reward_mean']),
                'frame_mean_log_error': abs(means['combined_reward'] / 416 - metrics['rollout/rewards']),
            }
            assert max(audit.values()) < 1e-4, (arm, step, audit)
            selected = [{key: r[key] for key in ('update', 'rollout_uid', 'group_id', 'member_id',
                        'trajectory_frames', 'training_reward_source', 'chunk_rewards') + FIELDS} for r in rows]
            source['steps'].append({'step': step, 'records_path': str(record_path),
                'records_sha256': hashlib.sha256(payload).hexdigest(), 'logged_metrics': metrics,
                'audit': audit, 'records': selected})
            plotted.append({'arm': arm, 'training_round': step, 'policy_updates_before_sampling': step-1,
                'trajectories': len(rows), 'frames_per_trajectory': 416,
                'trajectory_reward_mean': means['trajectory_reward'],
                'terminal_reward_mean': means['lastframe_reward'],
                'combined_reward_mean': metrics['rollout/episode_reward_mean'],
                'combined_reward_mean_from_records': means['combined_reward'],
                'reward_per_frame_mean': metrics['rollout/rewards']})
        assert source['steps'], f'No committed reward points for {arm}'
        snapshot['arms'][arm] = source
    return snapshot, plotted


def main(out):
    snapshot, rows = collect()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'reward_source_snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    with (out / 'grpo_reward_steps.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    style()
    stamp = datetime.fromisoformat(snapshot['captured_at']).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
    fig.subplots_adjust(left=0.07, right=0.915, top=0.775, bottom=0.16, hspace=0.75, wspace=0.29)
    fig.text(0.07, 0.955, 'Push-T: the reward GRPO actually uses', fontsize=23, color='#172A42')
    fig.text(0.07, 0.910, f'Data as of {stamp} (UTC) | 128 trajectories per point | higher reward (closer to 0) is better', fontsize=11, color='#526176')
    max_step = max(r['training_round'] for r in rows)
    columns = [('trajectory_reward_mean', 'Trajectory reward ↑'), ('terminal_reward_mean', 'Terminal reward ↑'),
               ('combined_reward_mean', 'Total reward = trajectory + terminal ↑')]
    row_labels = {'original': 'Original weights: wrist 2/3, main / side 1/6 each',
                  'no_wrist': 'Zeroed wrist: wrist 0, main / side 0.5 each'}
    for i, arm in enumerate(ARMS):
        values = [r for r in rows if r['arm'] == arm]
        fig.text(0.07, 0.838 if i == 0 else 0.447, f"{row_labels[arm]} | {len(values)} rounds recorded", fontsize=13, color=COLORS[arm])
        for j, (key, title) in enumerate(columns):
            ax = axes[i, j]
            xs = [r['training_round'] for r in values]
            ys = [r[key] for r in values]
            ax.plot(xs, ys, color=COLORS[arm], marker='o', markersize=5.5, linewidth=2.2)
            ax.set_title(title, loc='left', fontsize=12.5, pad=10)
            ax.set_xlim(0.6, max_step + 0.6)
            ax.set_xticks(range(1, max_step + 1))
            ax.set_xlabel('Training round k')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.margins(y=0.30)
            ax.grid(axis='y', color='#E5EAF0', linewidth=0.8)
            ax.grid(axis='x', color='#F1F4F7', linewidth=0.6)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel('Mean cumulative reward per trajectory')
            for idx in sorted({0, len(xs)-1}):
                ax.annotate(f'{ys[idx]:.3f}', (xs[idx], ys[idx]), xytext=(0, 10),
                            textcoords='offset points', ha='center', fontsize=10, color=COLORS[arm])
            if j == 2:
                secondary = ax.secondary_yaxis('right', functions=(lambda y:y/416, lambda y:y*416))
                secondary.spines['right'].set_visible(True)
                secondary.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                secondary.yaxis.set_major_locator(MaxNLocator(5))
                secondary.set_ylabel('Mean per-frame reward (total reward / 416)', fontsize=10)
    fig.text(0.07, 0.087, 'Left axis: summed over 416 frames, then averaged over 128 trajectories; the total-reward right axis corresponds to rollout/rewards in the logs.', fontsize=10.5, color='#526176')
    fig.text(0.07, 0.052, 'Each arm keeps its actual training weights and each panel has its own y axis; absolute reward values should not be compared across arms. Curves are unsmoothed and uninterpolated.', fontsize=10.5, color='#526176')
    fig.text(0.07, 0.018, 'Point k is the sampled reward used for update k (the sampling policy has had k-1 updates); this shows the reward inputs, before discounted returns and within-group advantages are computed.', fontsize=10.5, color='#526176')
    save_figure(fig, out, 'grpo_reward_terms')
    max_audit = {key: max(s['audit'][key] for a in snapshot['arms'].values() for s in a['steps'])
                 for key in next(iter(snapshot['arms'].values()))['steps'][0]['audit']}
    manifest = {'captured_at': snapshot['captured_at'],
        'completed_rounds': {arm: [s['step'] for s in snapshot['arms'][arm]['steps']] for arm in ARMS},
        'source_fields': list(FIELDS), 'source_metrics': list(METRICS),
        'training_reward_source': 'continuous_combined',
        'reward_definition': 'Native training trajectory_reward + lastframe_reward; no cross-view rescoring.',
        'shown_quantity': 'Undiscounted raw reward input before suffix-return calculation and group normalization.',
        'training_x_definition': 'k is the batch used for update k; sampling policy has k-1 updates.',
        'camera_weights': {'original': {'wrist': 2/3, 'main': 1/6, 'side': 1/6},
                           'no_wrist': {'wrist': 0, 'main': 0.5, 'side': 0.5}},
        'aggregation': 'Sum in time, then mean over 128 trajectories.',
        'smoothing': False, 'imputed_steps': False, 'audit_max_abs_errors': max_audit}
    (out / 'plot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (out / 'README.md').write_text(
        '# GRPO reward curves\n\n'
        'All values come from the actual training records and keep each arm\'s own view weights. Each round has 128 complete trajectories of 416 frames.\n\n'
        'The trajectory term uses trajectory_reward and the terminal term uses lastframe_reward; each is first summed over time and then averaged over trajectories. '
        'The total reward is taken directly from rollout/episode_reward_mean and checked against the logged mean of combined_reward. '
        'The right axis gives the total reward divided by 416, checked against rollout/rewards.\n\n'
        'What is shown is the reward input before discounted returns and within-group advantages are computed; it is not the policy loss and not a normalized advantage. '
        'The current training_reward_source=continuous_combined; the success-classifier reward is not used in this run.\n\n'
        'The x axis k corresponds to the batch sampled for weight update k, with k-1 updates already applied to the sampling policy. '
        'Only rounds that are marked complete, logged, and have 128 valid trajectories are included. Sampled scenes may differ between rounds.\n\n'
        'The reward definition changes with the view weights, and each panel has its own y axis, so absolute values should not be compared across arms.\n\n'
        'grpo_reward_steps.csv contains the per-round plotted values; reward_source_snapshot.json stores the raw reward fields used, the logged values and source checksums; '
        'plot_manifest.json stores the timestamp, the plotting definitions and the numerical cross-check results.\n')
    print(json.dumps({'output': str(out), 'completed_rounds': manifest['completed_rounds'],
        'audit_max_abs_errors': max_audit,
        'first_and_latest': {arm: [r for r in rows if r['arm'] == arm][::max(1, len([r for r in rows if r['arm'] == arm])-1)] for arm in ARMS}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or DEFAULT / ('grpo_rewards_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    main(output.resolve())
