"""Refresh reward/success figures using only validated, completed training data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_push_t_grpo_rewards import main as reward_main
from plot_push_t_paired_evaluation import plot_paired_evaluation
from plot_push_t_weight_ablation_steps import ARMS, COLORS, DEFAULT, LABELS, ROOT, save_figure, style


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def line(ax, arm, x, y):
    ax.plot(x, y, color=COLORS[arm], linewidth=2.1,
            linestyle='-' if arm == 'original' else '--',
            marker='o' if arm == 'original' else 's',
            markersize=5.5 if arm == 'original' else 8,
            markerfacecolor=COLORS[arm] if arm == 'original' else 'none',
            markeredgewidth=1.5, label=LABELS[arm])
    ax.grid(axis='y', color='#E5EAF0'); ax.set_axisbelow(True)


def refresh(out):
    reward_main(out)
    source = json.loads((out / 'reward_source_snapshot.json').read_text())
    success_source = {'captured_at': source['captured_at'], 'training': [], 'evaluation': []}
    training, evaluation = [], []
    scene_seeds = None
    for arm in ARMS:
        info = source['arms'][arm]
        for entry in info['steps']:
            payload = Path(entry['records_path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry['records_sha256']
            rows = [json.loads(l) for l in payload.splitlines() if l.strip()]
            successes = sum(bool(r['model_success']) for r in rows)
            training.append({'arm': arm, 'training_round': entry['step'],
                             'policy_updates': entry['step'] - 1, 'n': len(rows),
                             'successes': successes, 'success_percent': 100 * successes / len(rows)})
            success_source['training'].append({'arm': arm, 'step': entry['step'],
                'records_path': entry['records_path'], 'sha256': entry['records_sha256'],
                'labels': [{'rollout_uid': r['rollout_uid'], 'model_success': r['model_success']} for r in rows]})
        run = Path(info['run_dir'])
        for p in sorted((run / 'evaluation_records').glob('*.jsonl')):
            payload = p.read_bytes()
            rows = [json.loads(l) for l in payload.splitlines() if l.strip()]
            if len(rows) != 30:
                continue
            steps = {r['global_step'] for r in rows}; assert len(steps) == 1
            step = steps.pop()
            if step > info['status']['completed_updates']:
                continue
            assert all(r['valid'] for r in rows) and len({r['episode'] for r in rows}) == 30
            seeds = {r['episode']: r['seed'] for r in rows}
            if scene_seeds is None:
                scene_seeds = seeds
            assert seeds == scene_seeds, 'Fixed evaluation scenes/seeds differ'
            total = fmean(r['combined_reward'] for r in rows)
            terminal = fmean(-416 * r['lastframe_mse'] for r in rows)
            successes = sum(bool(r['model_success']) for r in rows)
            evaluation.append({'arm': arm, 'policy_updates': step, 'n': 30,
                'successes': successes, 'success_percent': 100 * successes / 30,
                'combined_reward_mean': total, 'terminal_reward_mean': terminal,
                'trajectory_reward_mean': total - terminal})
            success_source['evaluation'].append({'arm': arm, 'policy_updates': step,
                'records_path': str(p), 'sha256': hashlib.sha256(payload).hexdigest(),
                'rows': [{k: r[k] for k in ('episode', 'seed', 'model_success', 'combined_reward')} for r in rows]})
    training.sort(key=lambda r: (r['arm'], r['training_round']))
    evaluation.sort(key=lambda r: (r['arm'], r['policy_updates']))
    write_csv(out / 'training_success_steps.csv', training)
    write_csv(out / 'fixed_evaluation_reward_success.csv', evaluation)
    (out / 'success_source_snapshot.json').write_text(json.dumps(success_source, ensure_ascii=False, indent=2) + '\n')
    style()
    stamp = datetime.fromisoformat(source['captured_at']).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    fig.subplots_adjust(left=.07, right=.98, top=.70, bottom=.23, wspace=.25)
    fig.text(.07, .94, 'Push-T: training rollout and fixed evaluation success rate', fontsize=21, color='#172A42')
    fig.text(.07, .874, f'Data as of {stamp} (UTC) | original weights {sum(r["arm"] == "original" for r in training)} rounds, zeroed wrist {sum(r["arm"] == "no_wrist" for r in training)} rounds', fontsize=11, color='#526176')
    for ax, data, xkey, title in ((axes[0], training, 'training_round', 'Training rollouts | 128 trajectories per point'),
                                 (axes[1], evaluation, 'policy_updates', 'Fixed-scene evaluation | 30 trajectories per point')):
        for arm in ARMS:
            vals = [r for r in data if r['arm'] == arm]
            line(ax, arm, [r[xkey] for r in vals], [r['success_percent'] for r in vals])
            if xkey == 'training_round':
                last = vals[-1]
                ax.annotate(f'{last["successes"]}/{last["n"]}', (last[xkey], last['success_percent']),
                            xytext=(0, 11), textcoords='offset points', ha='center', color=COLORS[arm], fontsize=10)
        xs = sorted({r[xkey] for r in data})
        ax.set_xticks(xs); ax.set_xlim(min(xs) - .4, max(xs) + .6)
        ax.set_ylabel('Success rate (%)'); ax.set_title(title, loc='left', pad=12)
        ax.set_xlabel('Training round k (sampled with the pre-update policy)' if xkey == 'training_round' else 'Weight updates completed at evaluation time')
        ax.set_ylim(-1, max(15, max(r['success_percent'] for r in data) + 8))
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        if xkey == 'policy_updates':
            for x in xs:
                vals = [r for r in data if r[xkey] == x]
                if len({r['successes'] for r in vals}) == 1:
                    ax.annotate(f'{vals[0]["successes"]}/30' + (' (both arms)' if len(vals) == 2 else ' (original weights)'),
                                (x, vals[0]['success_percent']), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10, color='#526176')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.065, .83), ncol=2, frameon=False)
    fig.text(.07, .12, 'Only completed observations are connected; no smoothing and no interpolation. Training rollout scenes vary by round, while the fixed evaluation reuses the same 30 scenes and seeds.', fontsize=10.5, color='#526176')
    fig.text(.07, .065, 'Success labels come from the world-model image classifier; they are a monitoring metric, whereas training here optimizes the trajectory and terminal rewards.', fontsize=10.5, color='#526176')
    save_figure(fig, out, 'success_rate_curves')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    fig.subplots_adjust(left=.07, right=.97, top=.76, bottom=.16, wspace=.28, hspace=.85)
    fig.text(.07, .94, 'Push-T: fixed-scene evaluation reward', fontsize=22, color='#172A42')
    fig.text(.07, .885, f'Data as of {stamp} (UTC) | same 30 scenes and seeds | higher reward is better', fontsize=11, color='#526176')
    cols = [('trajectory_reward_mean', 'Trajectory reward ↑'), ('terminal_reward_mean', 'Terminal reward ↑'), ('combined_reward_mean', 'Total reward ↑')]
    for i, arm in enumerate(ARMS):
        vals = [r for r in evaluation if r['arm'] == arm]
        fig.text(.07, .812 if i == 0 else .432, LABELS[arm], fontsize=13, color=COLORS[arm])
        for ax, (key, title) in zip(axes[i], cols):
            line(ax, arm, [r['policy_updates'] for r in vals], [r[key] for r in vals])
            ax.set_title(title, loc='left'); ax.set_xlabel('Completed weight updates')
            ax.set_xticks(sorted({r['policy_updates'] for r in evaluation})); ax.set_xlim(-.4, max(r['policy_updates'] for r in evaluation) + .6)
            ax.margins(y=.3); ax.yaxis.set_major_locator(MaxNLocator(4))
            last = vals[-1]
            ax.annotate(f'{last[key]:.3f}', (last['policy_updates'], last[key]), xytext=(0, 10), textcoords='offset points', ha='center', color=COLORS[arm], fontsize=10)
    fig.text(.07, .075, 'Each point is the mean cumulative reward over 30 complete trajectories; each arm keeps its own training weights and each panel has its own y axis.', fontsize=10.5, color='#526176')
    fig.text(.07, .031, 'The terminal term is restored with the actual factor 416; the trajectory term is the logged total reward minus the terminal term. Only completed evaluations are plotted; missing points are not filled in.', fontsize=10.5, color='#526176')
    save_figure(fig, out, 'fixed_evaluation_reward_terms')

    paired_done = plot_paired_evaluation(out)
    direction_path = ROOT / 'outputs/push_t_reward_direction_and_scale_20260906/direction_progress.json'
    direction_done = False
    if direction_path.exists():
        payload = direction_path.read_bytes(); result = json.loads(payload)
        if result.get('after_update_complete'):
            assert result['optimizer_steps'] == 4 and result['frozen_inputs_verified']
            (out / 'direction_source_snapshot.json').write_bytes(payload)
            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3))
            fig.subplots_adjust(left=.08, right=.97, top=.72, bottom=.24, wspace=.32)
            fig.text(.08, .94, 'Full 4-step update: fixed-batch direction check', fontsize=21, color='#172A42')
            fig.text(.08, .864, 'Same batch of 128 trajectories / 1664 chunks; advantages, old scores and MC noise held fixed; all 32 ranks completed', fontsize=10.5, color='#526176')
            for ax, key, title in ((axes[0], 'objective_improvement', 'Change in clipped surrogate objective ↑'),
                                   (axes[1], 'directional_mean', 'Change in mean A x score ↑')):
                vals = [result['no_update_control'][key], result['after_update'][key]]
                ax.bar([0, 1], vals, color=['#94A3B8', '#21836B'], width=.58)
                ax.set_xticks([0, 1], ['Replay without parameter update', 'After the full 4-step update'])
                ax.set_title(title, loc='left'); ax.set_ylim(0, max(vals) * 1.35)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(axis='y', color='#E5EAF0'); ax.set_axisbelow(True)
                for i, v in enumerate(vals):
                    ax.annotate(f'{v:+.3e}', (i, v), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=11)
            fig.text(.08, .13, 'This update raises the surrogate objective by clearly more than the numerical error of the no-update replay; it is the result of a single update.', fontsize=10.5, color='#526176')
            note = ('Paired evaluation completed: resampled reward and success rate are in the paired_policy_evaluation figure.' if paired_done
                    else 'This figure shows the optimization direction; before/after differences in behavioural reward and success rate require the paired evaluation to finish.')
            fig.text(.08, .069, note, fontsize=10.5, color='#526176')
            save_figure(fig, out, 'fixed_batch_update_direction')
            direction_done = True
    manifest_path = out / 'plot_manifest.json'; manifest = json.loads(manifest_path.read_text())
    manifest['fixed_evaluation_updates'] = {arm: [r['policy_updates'] for r in evaluation if r['arm'] == arm] for arm in ARMS}
    manifest['success_label_field'] = 'model_success'; manifest['direction_figure_available'] = direction_done
    manifest['paired_evaluation_figure_available'] = paired_done
    manifest['artifacts'] = [p.name for p in sorted(out.iterdir()) if p.suffix in ('.png', '.pdf', '.csv')]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with (out / 'README.md').open('a') as f:
        f.write('\nUpdated success-rate and fixed-evaluation reward figures: success_rate_curves.png/pdf and fixed_evaluation_reward_terms.png/pdf. The corresponding per-point data are in training_success_steps.csv and fixed_evaluation_reward_success.csv.\n')
        f.write('\nfixed_batch_update_direction.png/pdf is a standalone diagnostic of the update from 7 to 8 and must not be spliced onto the original training curves; its data are in direction_source_snapshot.json. Trajectory coefficients 0.1 and 0 have no completed rounds yet, so no curves are drawn for them.\n')
    if paired_done:
        with (out / 'README.md').open('a') as f:
            f.write('\nPaired evaluation completed: paired_policy_evaluation.png/pdf shows the resampled reward, the per-scene paired differences with a 95% interval, and the success rate for the standalone update from 7 to 8. These data are not spliced into the two original training curves. Per-scene values and cross-check records are in paired_policy_evaluation.csv and paired_evaluation_source_snapshot.json.\n')
    pointer = DEFAULT / 'grpo_rewards_latest'; temporary = DEFAULT / '.grpo_rewards_latest.tmp'
    assert not temporary.exists() and not temporary.is_symlink()
    if pointer.exists() and not pointer.is_symlink():
        raise RuntimeError('Latest pointer is an existing real directory; refusing to replace it')
    temporary.symlink_to(out.relative_to(DEFAULT), target_is_directory=True); temporary.replace(pointer)
    (DEFAULT / 'latest_reward_plots.json').write_text(json.dumps({'output_dir': str(out), 'manifest': str(manifest_path), 'captured_at': source['captured_at']}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'latest': str(pointer), 'training_rounds': {arm: sum(r['arm'] == arm for r in training) for arm in ARMS}, 'fixed_evaluation_updates': manifest['fixed_evaluation_updates'], 'artifacts': manifest['artifacts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); args = p.parse_args()
    out = args.output or DEFAULT / ('grpo_rewards_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    refresh(out.resolve())
