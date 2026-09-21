"""Plot paired, newly sampled policy rewards without mixing them into training."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from plot_push_t_weight_ablation_steps import ROOT, save_figure, style


def plot_paired_evaluation(out):
    source_dir = ROOT / 'outputs/push_t_reward_direction_and_scale_20260906'
    result_path = source_dir / 'direction_result.json'
    if not result_path.exists():
        return False
    payload = result_path.read_bytes()
    result = json.loads(payload)
    if result.get('state') != 'complete':
        return False
    paired = result['paired_evaluation']
    before_step, after_step = paired['policy_updates']
    run = ROOT / 'outputs/push_t_reward_direction_probe_from7_to8_20260906'
    snapshots = []
    records = []
    for step in (before_step, after_step):
        path = run / 'evaluation_records' / f'step_{step - 1}.jsonl'
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == 30 and len({row['episode'] for row in rows}) == 30
        assert all(row['valid'] and row['global_step'] == step for row in rows)
        snapshots.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        records.append({row['episode']: row for row in rows})
    before, after = records
    assert before.keys() == after.keys()
    pairs = []
    for episode in sorted(before):
        b, a = before[episode], after[episode]
        assert b['seed'] == a['seed']
        assert all(b['rng_manifest'][key] == a['rng_manifest'][key]
                   for key in ('cosmos', 'ctrl_world'))
        pairs.append({'episode': episode, 'seed': b['seed'],
                      'reward_before': b['combined_reward'],
                      'reward_after': a['combined_reward'],
                      'reward_delta': a['combined_reward'] - b['combined_reward'],
                      'success_before': int(b['model_success']),
                      'success_after': int(a['model_success'])})
    b = np.array([row['reward_before'] for row in pairs])
    a = np.array([row['reward_after'] for row in pairs])
    delta = a - b
    mean = float(delta.mean())
    ci = np.quantile(np.random.default_rng(20260906).choice(
        delta, size=(10000, len(delta)), replace=True).mean(axis=1), [.025, .975])
    assert np.isfinite(np.r_[b, a, delta, ci]).all()
    assert abs(mean - paired['reward_delta_mean']) < 1e-12
    assert np.allclose(ci, paired['reward_delta_bootstrap95'], atol=1e-12, rtol=0)
    assert abs(b.mean() - paired['reward_before_mean']) < 1e-12
    assert abs(a.mean() - paired['reward_after_mean']) < 1e-12
    successes = [sum(row[key] for row in pairs)
                 for key in ('success_before', 'success_after')]
    assert successes == [paired['successes_before'], paired['successes_after']]
    with (out / 'paired_policy_evaluation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    (out / 'paired_evaluation_source_snapshot.json').write_text(json.dumps({
        'result_path': str(result_path), 'result_sha256': hashlib.sha256(payload).hexdigest(),
        'records': snapshots, 'actual_rng_streams_match': True,
        'policy_updates': [before_step, after_step], 'result': paired,
        'positive_reward_deltas': int((delta > 0).sum()),
        'negative_reward_deltas': int((delta < 0).sum())}, indent=2) + '\n')
    style()
    colors = ['#2563B5', '#7C3AED']
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2))
    fig.subplots_adjust(left=.065, right=.97, top=.73, bottom=.27, wspace=.43)
    fig.text(.065, .94, f'Updates {before_step} to {after_step}: resampled reward and success rate',
             fontsize=22, color='#172A42')
    fig.text(.065, .864, '30 identical scenes | paired Cosmos / Ctrl-World random streams verified | 4 optimizer steps | 416 frames per trajectory',
             fontsize=11, color='#526176')
    ax = axes[0]
    for bi, ai in zip(b, a):
        ax.plot([0, 1], [bi, ai], color='#CBD5E1', alpha=.65, linewidth=.85, zorder=1)
    ax.scatter(np.zeros(len(b)), b, s=18, color=colors[0], alpha=.70, zorder=2)
    ax.scatter(np.ones(len(a)), a, s=18, color=colors[1], alpha=.70, zorder=2)
    ax.plot([0, 1], [b.mean(), a.mean()], color='#172A42', linewidth=2.7,
            marker='D', markersize=7, label='Mean reward', zorder=4)
    ax.set_title('Reward on the same scene ↑', loc='left', pad=13)
    ax.set_xticks([0, 1], [f'After update {before_step}', f'After update {after_step}'])
    ax.set_xlim(-.3, 1.3)
    span = float(np.ptp(np.r_[b, a])) or 1.
    ax.set_ylim(min(b.min(), a.min()) - .12 * span, max(b.max(), a.max()) + .12 * span)
    ax.set_ylabel('Total reward per trajectory')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(.5, -.24, f'Mean {b.mean():.4f} -> {a.mean():.4f}',
            transform=ax.transAxes, ha='center', color='#334155', fontsize=11)

    ax = axes[1]
    jitter = ((np.arange(len(delta)) * 7) % len(delta) / (len(delta) - 1) - .5) * .32
    point_colors = np.where(delta > 0, '#21836B', '#D76122')
    ax.scatter(delta, 1 + jitter, s=25, c=point_colors, alpha=.78, zorder=3)
    ax.errorbar(mean, 0, xerr=np.array([[mean - ci[0]], [ci[1] - mean]]),
                color='#172A42', marker='D', markersize=7, capsize=7, linewidth=2.3, zorder=4)
    ax.axvline(0, color='#94A3B8', linestyle='--', linewidth=1)
    ax.set_title('Reward change: after minus before', loc='left', pad=13)
    ax.set_yticks([0, 1], ['Mean and\n95% interval', 'Per scene'])
    ax.set_ylim(-.6, 1.6)
    ax.set_xlabel('Δ reward (positive means improvement)')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.margins(x=.12)
    ax.text(.5, -.24, f'Δ = {mean:+.4f}; 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]',
            transform=ax.transAxes, ha='center', color='#334155', fontsize=10.5)
    ax.text(.98, .92, f'{int((delta > 0).sum())} improved / {int((delta < 0).sum())} degraded',
            transform=ax.transAxes, ha='right', color='#526176', fontsize=10)

    ax = axes[2]
    percentages = np.array(successes) / len(pairs) * 100
    ax.bar([0, 1], percentages, color=colors, width=.44, alpha=.8)
    ax.scatter([0, 1], percentages, color=colors, s=60, marker='s', zorder=4)
    ax.set_title('Success rate on fixed scenes ↑', loc='left', pad=13)
    ax.set_xticks([0, 1], [f'After update {before_step}', f'After update {after_step}'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('Success rate (%)')
    ax.set_ylim(-1, max(15, float(percentages.max()) + 10))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    for i, (n, pct) in enumerate(zip(successes, percentages)):
        ax.annotate(f'{n}/{len(pairs)}', (i, pct), xytext=(0, 13),
                    textcoords='offset points', ha='center', color=colors[i], fontsize=13)
    if successes == [0, 0]:
        ax.text(.5, .57, 'Neither evaluation was scored as a success', transform=ax.transAxes,
                ha='center', color='#526176', fontsize=12)
    ax.text(.5, -.24, 'Success labels come from the world-model image classifier',
            transform=ax.transAxes, ha='center', color='#526176', fontsize=10.5)
    for ax in axes:
        ax.grid(axis='y', color='#E5EAF0'); ax.set_axisbelow(True)
    conclusion = ('The interval of the mean difference spans 0: no clear reward improvement or degradation is detected in this run.'
                  if ci[0] <= 0 <= ci[1] else
                  'The error bar is the 95% interval of the mean reward change over these paired scenes.')
    fig.text(.065, .10, conclusion, fontsize=12, color='#172A42')
    fig.text(.065, .038, 'Light grey lines connect the same scene; error bars are computed by resampling scenes 10,000 times. This figure is a standalone diagnostic and its conclusion is limited to this one update and paired evaluation.',
             fontsize=10.5, color='#526176')
    save_figure(fig, out, 'paired_policy_evaluation')
    return True
