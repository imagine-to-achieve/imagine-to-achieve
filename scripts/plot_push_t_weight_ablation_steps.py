"""Plot committed training batches and the actual fixed evaluation checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'outputs/push_t_view_weight_ablation_fp32_h13_20260904'
ARMS = ('original', 'no_wrist')
COLORS = {'original': '#2563B5', 'no_wrist': '#D76122'}
LABELS = {'original': 'Original weights', 'no_wrist': 'Zeroed wrist'}


def style():
    plt.rcParams.update({'font.family': ['DejaVu Sans'], 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#B7C0CC', 'axes.labelcolor': '#334155',
        'xtick.color': '#475569', 'ytick.color': '#475569',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def records(summary, dataset):
    result = {}
    for arm in ARMS:
        source = summary['arms'][arm]['training' if dataset == 'train' else 'evaluation']
        rows = []
        for entry in source:
            if dataset == 'train':
                assert entry['n'] == 128 and entry['complete_per_view_records'] == 128
                policy_step = int(entry['updates_before_rollout'])
                training_step = policy_step + 1
            else:
                assert entry['n'] == 30
                policy_step = int(entry['updates'])
                training_step = None
            row = {'arm': arm, 'dataset': dataset, 'training_update': training_step,
                   'policy_updates_before_sampling': policy_step, 'n': entry['n'],
                   'successes': entry['successes'], 'success_percent': 100*entry['success_rate'],
                   'trajectory_main_side_mse': entry['trajectory_rescore_no_wrist'],
                   'terminal_main_side_mse': entry['terminal_rescore_no_wrist'],
                   'trajectory_own_weight_mse': entry['training_weight_trajectory_mse'],
                   'terminal_own_weight_mse': entry['training_weight_terminal_mse'],
                   'trajectory_original_weight_mse': entry['trajectory_rescore_original'],
                   'terminal_original_weight_mse': entry['terminal_rescore_original']}
            assert 0 <= row['successes'] <= row['n']
            assert abs(row['success_percent'] - 100*row['successes']/row['n']) < 1e-10
            assert all(math.isfinite(value) and value >= 0 for key,value in row.items() if key.endswith('_mse'))
            rows.append(row)
        result[arm] = sorted(rows, key=lambda r:r['policy_updates_before_sampling'])
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw(ax, rows, xkey, ykey, *, fixed=False):
    for arm in ARMS:
        data = rows[arm]
        x = [r[xkey] for r in data]; y = [r[ykey] for r in data]
        ax.plot(x, y, color=COLORS[arm], linewidth=2.2,
                linestyle='-' if arm == 'original' else '--',
                marker='o' if arm == 'original' else 's',
                markersize=5.5 if arm == 'original' else 8.0,
                markerfacecolor=COLORS[arm] if arm == 'original' else 'none',
                markeredgewidth=1.5, label=LABELS[arm], zorder=3 if arm == 'original' else 4)
    xs = sorted({r[xkey] for arm in ARMS for r in rows[arm]})
    if xs:
        ax.set_xlim(min(xs)-0.35, max(xs)+0.4)
        ax.set_xticks(xs if fixed else list(range(int(min(xs)), int(max(xs))+1)))
    ax.grid(axis='y', color='#E5EAF0', linewidth=0.8, zorder=0)
    ax.grid(axis='x', color='#F1F4F7', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Completed weight updates' if fixed else 'Training round k')
    if ykey == 'success_percent':
        maximum = max((r[ykey] for arm in ARMS for r in rows[arm]), default=0)
        top = max(35, math.ceil(maximum/10)*10+5)
        ax.set_ylim(-1.3, min(101.3, top))
        ax.set_yticks(list(range(0, min(101, top+1), 10)))
        ax.set_ylabel('Success rate (%)')
        if fixed:
            by_step = {arm:{r[xkey]:r for r in rows[arm]} for arm in ARMS}
            for x in xs:
                a = by_step['original'].get(x); b = by_step['no_wrist'].get(x)
                if a and b and a['successes'] == b['successes']:
                    ax.annotate(f"{a['successes']}/{a['n']} (both arms)", (x,a[ykey]),
                                xytext=(0,10), textcoords='offset points', ha='center',
                                color='#475569', fontsize=10)
    else:
        ax.set_ylabel('MSE')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.margins(y=0.16)


def save_figure(fig, out, name):
    fig.savefig(out/f'{name}.png', dpi=190)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)


def main(source, output):
    payload = source.read_bytes(); summary = json.loads(payload)
    if summary['integrity_errors']:
        raise ValueError(f"Unresolved data integrity errors: {summary['integrity_errors']}")
    output.mkdir(parents=True, exist_ok=True)
    (output/'source_summary.json').write_bytes(payload)
    train = records(summary, 'train'); evaluation = records(summary, 'eval')
    write_csv(output/'training_steps.csv', [r for arm in ARMS for r in train[arm]])
    write_csv(output/'fixed_evaluation_steps.csv', [r for arm in ARMS for r in evaluation[arm]])
    style()
    stamp = datetime.fromisoformat(summary['generated_at']).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')
    counts = '; '.join(f'{LABELS[arm]} {len(train[arm])} rounds' for arm in ARMS)
    panels = [('trajectory_main_side_mse','Trajectory MSE ↓'),
              ('terminal_main_side_mse','Terminal MSE ↓'), ('success_percent','Success rate ↑')]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.13, hspace=0.70, wspace=0.29)
    fig.text(0.07,0.961,'Push-T: per-step MSE and success rate', fontsize=23, color='#172A42')
    fig.text(0.07,0.925,f'Data as of {stamp} (UTC) | {counts}', fontsize=11, color='#526176')
    for i,(key,title) in enumerate(panels):
        draw(axes[0,i],train,'training_update',key)
        draw(axes[1,i],evaluation,'policy_updates_before_sampling',key,fixed=True)
        axes[0,i].set_title(title,loc='left',pad=12)
        axes[1,i].set_title(title,loc='left',pad=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper left',
               bbox_to_anchor=(0.063,0.903), ncol=2, frameon=False, columnspacing=3)
    fig.text(0.07,0.849,'Training rollouts | 128 trajectories per round, raw per-round means',fontsize=13,color='#172A42')
    fig.text(0.07,0.444,'Fixed evaluation | 30 trajectories each, only actual evaluation points shown',fontsize=13,color='#172A42')
    fig.text(0.07,0.066,'MSE is rescored uniformly as main x 0.5 + side x 0.5; trajectory and terminal terms are shown separately.',fontsize=10.5,color='#526176')
    fig.text(0.07,0.032,'Training point k comes from the policy before update k (k-1 updates applied); no smoothing, no interpolation. Success labels come from the world-model image classifier.',fontsize=10.5,color='#526176')
    save_figure(fig,output,'per_step_mse_success')

    fig, axes = plt.subplots(1,3,figsize=(16.5,4.7))
    fig.subplots_adjust(left=0.07,right=0.98,top=0.66,bottom=0.23,wspace=0.29)
    fig.text(0.07,0.93,'Raw training records: per-step curves under each arm\'s own reward weights',fontsize=21,color='#172A42')
    fig.text(0.07,0.852,'Each arm uses its own reward weights, so the view weighting behind the MSE values differs.',fontsize=11,color='#526176')
    keys = [('trajectory_own_weight_mse','Trajectory MSE ↓'),('terminal_own_weight_mse','Terminal MSE ↓'),('success_percent','Success rate ↑')]
    for ax,(key,title) in zip(axes,keys):
        draw(ax,train,'training_update',key);ax.set_title(title,loc='left',pad=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(0.063,0.814),ncol=2,frameon=False,columnspacing=3)
    fig.text(0.07,0.107,'Original weights: wrist 2/3, main/side 1/6 each; zeroed wrist: wrist 0, main/side 0.5 each.',fontsize=10.5,color='#526176')
    fig.text(0.07,0.047,f'128 trajectories per round; round k is sampled with the pre-update policy; no smoothing. Data as of {stamp} (UTC).',fontsize=10.5,color='#526176')
    save_figure(fig,output,'training_own_reward_weights')

    manifest = {'generated_at':datetime.now(timezone.utc).isoformat(),
                'source_snapshot_time':summary['generated_at'],'source_path':str(source),
                'source_sha256':hashlib.sha256(payload).hexdigest(),
                'training_rounds':{arm:[r['training_update'] for r in train[arm]] for arm in ARMS},
                'fixed_evaluation_updates':{arm:[r['policy_updates_before_sampling'] for r in evaluation[arm]] for arm in ARMS},
                'main_figure_weights':{'wrist':0,'main':0.5,'side':0.5},
                'training_x_definition':'k is the rollout batch used for optimizer update k; behavior policy has k-1 completed updates.',
                'evaluation_x_definition':'Number of completed updates when the fixed evaluation was run.',
                'smoothing':False,'missing_steps_imputed':False,
                'integrity_errors':summary['integrity_errors']}
    (output/'plot_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'README.md').write_text('# Per-step curves\n\n'
        'The top row of the main figure is per-round training rollouts (128 trajectories per point); the bottom row is the fixed evaluation (30 per point). MSE in the main figure is rescored for both arms as main/side 0.5 each.\n\n'
        'Training point k corresponds to the batch sampled for update k, with k-1 updates already applied to the sampling policy; the x axis of the fixed evaluation is the number of updates completed at evaluation time.\n\n'
        'Every recorded point is plotted directly, without smoothing and without filling in rounds that have not finished. Connecting lines join observed points only; the actual evaluation points for each arm are listed in plot_manifest.json.\n\n'
        'training_own_reward_weights.png/pdf keeps the MSE under each arm\'s original training weights, where the view weighting differs.\n\n'
        'training_steps.csv and fixed_evaluation_steps.csv contain all plotted values; source_summary.json and plot_manifest.json pin this data snapshot and its provenance.\n')
    print(json.dumps({'output':str(output),'training_rounds':manifest['training_rounds'],
                      'fixed_evaluation_updates':manifest['fixed_evaluation_updates'],
                      'artifacts':[p.name for p in sorted(output.iterdir())]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=DEFAULT/'comparison_results.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or DEFAULT/('curves_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    main(args.source.resolve(),out.resolve())
