"""Compare fixed-stream held-out outcomes and rescore both arms identically.

Safe to rerun while training: only committed trajectory/evaluation files count.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/push_t_view_weight_ablation_fp32_h13_20260904'
WEIGHTS = {'original': {'wrist': 2/3, 'main': 1/6, 'side': 1/6},
           'no_wrist': {'wrist': 0., 'main': .5, 'side': .5}}


def mean(values):
    return statistics.fmean(values) if values else None


def read_jsonl(path, partial=False):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not partial:
                raise
    return rows


def scalar(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def score(views, arm):
    return sum(views[name] * weight for name, weight in WEIGHTS[arm].items())


def wilson(successes, n):
    if n == 0:
        return None
    z = 1.95996398454
    p = successes/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def submitted_job_ranges(out, arm):
    path = out/'submission_records.json'
    if not path.exists():
        return None
    return {str(row['job_id']): (int(row['segment_start']), int(row['segment_end']))
            for row in json.loads(path.read_text()) if row['arm'] == arm}


def sidecars(run, job_ranges=None):
    samples = defaultdict(dict)
    for path in sorted((run/'reward_view_records').glob('rank_*.jsonl')):
        match = re.fullmatch(r'rank_\d+_job_(.+)\.jsonl', path.name)
        job = match.group(1) if match else None
        if job_ranges is not None and job not in job_ranges:
            continue
        for batch in read_jsonl(path, partial=True):
            if job_ranges is not None:
                start, end = job_ranges[job]
                if not start <= int(batch['global_step']) < end:
                    continue
            metadata = batch['metadata']
            offset = 0
            for i, raw_uid in enumerate(metadata['rollout_uid']):
                uid = int(scalar(raw_uid))
                chunk = int(scalar(metadata['chunk_id'][i]))
                row = {'trajectory': {name: values[i] for name,values in batch['trajectory_view_mse'].items()}}
                if batch['done'][i]:
                    row['terminal'] = {name: values[offset] for name,values in batch['terminal_view_mse_done'].items()}
                    offset += 1
                samples[(int(batch['global_step']),uid)][chunk] = row
    return samples


def summarize(out):
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(),
               'target_updates': 15, 'trajectory_seconds': 416/15,
               'weights_physical': WEIGHTS, 'arms': {}, 'paired_evaluation': [], 'integrity_errors': []}
    eval_rows = []; train_rows = []; evaluations = {}
    for arm in WEIGHTS:
        run = ROOT/'outputs'/f'push_t_combined_mse_fp32_h13_{arm}_15u_s5'
        state_path = run/'status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'state':'not_started'}
        values = {'run_dir':str(run), 'runtime_status':state, 'evaluation':[], 'training':[]}
        summary['arms'][arm] = values
        paths = sorted((run/'evaluation_records').glob('step_*.jsonl'))
        baseline = run/'evaluation_records/baseline.jsonl'
        if baseline.exists():
            paths.insert(0, baseline)
        for path in paths:
            records = read_jsonl(path)
            if len(records) != 30 or not all(r.get('valid') for r in records):
                summary['integrity_errors'].append(f'{arm}: invalid evaluation {path.name}')
                continue
            steps = {int(r['global_step']) for r in records}
            if len(steps) != 1:
                raise ValueError(f'Mixed evaluation steps in {path}')
            step = steps.pop()
            ordered = sorted(records, key=lambda r:r['episode'])
            if [r['seed'] for r in ordered] != list(range(42,72)):
                summary['integrity_errors'].append(f'{arm}: evaluation seeds drift at {step}')
            successes = sum(bool(r['model_success']) for r in records)
            entry = {'arm':arm, 'updates':step, 'n':30, 'successes':successes,
                     'success_rate':successes/30, 'success_ci95':wilson(successes,30),
                     'training_weight_trajectory_mse':mean([r['trajectory_mse'] for r in records]),
                     'training_weight_terminal_mse':mean([r['lastframe_mse'] for r in records])}
            for term, key in [('trajectory','trajectory_view_mse'),('terminal','terminal_view_mse')]:
                if not all(set(r.get(key,{})) == {'main','wrist','side'} for r in records):
                    summary['integrity_errors'].append(f'{arm}: missing {term} per-view metrics at {step}')
                    continue
                for camera in ('main','wrist','side'):
                    entry[f'{term}_{camera}_mse'] = mean([r[key][camera] for r in records])
                for scoring in WEIGHTS:
                    entry[f'{term}_rescore_{scoring}'] = mean([score(r[key], scoring) for r in records])
            values['evaluation'].append(entry); eval_rows.append(entry)
            evaluations[(arm,step)] = {r['episode']:r for r in records}
        job_ranges = submitted_job_ranges(out, arm)
        values['sidecar_job_ranges'] = job_ranges
        samples = sidecars(run, job_ranges)
        for path in sorted((run/'trajectory_records').glob('update_*.jsonl')):
            records = read_jsonl(path)
            if len(records) != 128 or not all(r.get('valid') for r in records):
                summary['integrity_errors'].append(f'{arm}: invalid training batch {path.name}')
                continue
            update = int(records[0]['update'])
            successes = [r for r in records if bool(r['model_success'])]
            entry = {'arm':arm,'updates_before_rollout':update,'n':len(records),
                     'successes':len(successes),'success_rate':len(successes)/len(records),
                     # This is the full-return diagnostic, not the suffix-return
                     # chunk advantage used by the optimizer.
                     'success_negative_advantage':sum(r['normalized_advantage']<0 for r in successes),
                     'success_negative_mean_chunk_advantage':sum(mean(r['chunk_advantages'])<0 for r in successes),
                     'success_negative_chunk_fraction':mean([float(a<0) for r in successes for a in r['chunk_advantages']]),
                     'success_mean_chunk_advantage':mean([a for r in successes for a in r['chunk_advantages']]),
                     'training_weight_trajectory_mse':mean([r['trajectory_mse'] for r in records]),
                     'training_weight_terminal_mse':mean([r['lastframe_mse'] for r in records])}
            joined = []
            for r in records:
                chunks = samples.get((update,int(r['rollout_uid'])),{})
                if set(chunks) != set(range(13)) or 'terminal' not in chunks[12]:
                    continue
                joined.append({'trajectory':{name:mean([chunks[k]['trajectory'][name] for k in range(13)]) for name in WEIGHTS['original']},
                               'terminal':chunks[12]['terminal'], 'row':r})
            entry['complete_per_view_records'] = len(joined)
            if len(joined) != 128:
                summary['integrity_errors'].append(f'{arm}: {update} has only {len(joined)}/128 per-view records')
            if joined:
                for term in ('trajectory','terminal'):
                    for scoring in WEIGHTS:
                        entry[f'{term}_rescore_{scoring}'] = mean([score(r[term], scoring) for r in joined])
                errors = [abs(score(j['trajectory'],arm)-j['row']['trajectory_mse']) for j in joined]
                errors += [abs(score(j['terminal'],arm)-j['row']['lastframe_mse']) for j in joined]
                entry['max_mse_recompute_error'] = max(errors)
                if max(errors)>1e-5:
                    summary['integrity_errors'].append(f'{arm}: MSE logging mismatch at {update}')
                # Pairwise ranking is computed only inside a shared reset group.
                for scoring in WEIGHTS:
                    concordant=0.; pairs=0
                    groups=defaultdict(list)
                    for row in joined:
                        groups[row['row']['group_id']].append(row)
                    for group in groups.values():
                        positive=[r for r in group if r['row']['model_success']]
                        negative=[r for r in group if not r['row']['model_success']]
                        for a in positive:
                            for b in negative:
                                sa=score(a['trajectory'],scoring)+score(a['terminal'],scoring)
                                sb=score(b['trajectory'],scoring)+score(b['terminal'],scoring)
                                concordant += float(sa<sb)+.5*float(sa==sb); pairs+=1
                    entry[f'within_group_auc_{scoring}'] = concordant/pairs if pairs else None
                    entry['within_group_success_failure_pairs'] = pairs
            values['training'].append(entry); train_rows.append(entry)
    common = sorted({s for a,s in evaluations if a=='original'} & {s for a,s in evaluations if a=='no_wrist'})
    for step in common:
        a=evaluations[('original',step)]; b=evaluations[('no_wrist',step)]
        if set(a)!=set(b):
            summary['integrity_errors'].append(f'Evaluation episode sets differ at {step}'); continue
        discordant=[0,0]
        noise_equal=True
        for ep in a:
            if a[ep]['model_success'] and not b[ep]['model_success']: discordant[0]+=1
            if b[ep]['model_success'] and not a[ep]['model_success']: discordant[1]+=1
            for engine in ('cosmos','ctrl_world'):
                if a[ep]['rng_manifest'][engine] != b[ep]['rng_manifest'][engine]: noise_equal=False
        if not noise_equal:
            summary['integrity_errors'].append(f'Actual evaluation noise differs between arms at {step}')
        n=sum(discordant)
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(discordant)+1))/2**n) if n else 1.
        summary['paired_evaluation'].append({'updates':step,'original_only_success':discordant[0],
            'no_wrist_only_success':discordant[1], 'success_difference_pp':100*(discordant[1]-discordant[0])/30,
            'mcnemar_exact_p':p,'actual_noise_manifests_equal':noise_equal})
    summary['all_expected_records_present'] = all(
        all((arm, step) in evaluations for step in (0, 5, 10, 15))
        and len(summary['arms'][arm]['training']) == 15 for arm in WEIGHTS
    )
    summary['complete'] = summary['all_expected_records_present'] and not summary['integrity_errors']
    summary['baseline_comparability'] = None
    if ('original', 0) in evaluations and ('no_wrist', 0) in evaluations:
        a = evaluations[('original', 0)]; b = evaluations[('no_wrist', 0)]
        if set(a) == set(b):
            errors = [abs(a[ep][key][view] - b[ep][key][view])
                      for ep in a for key in ('trajectory_view_mse', 'terminal_view_mse')
                      for view in ('main', 'wrist', 'side')
                      if view in a[ep].get(key, {}) and view in b[ep].get(key, {})]
            summary['baseline_comparability'] = {
                'success_disagreements': sum(a[ep]['model_success'] != b[ep]['model_success'] for ep in a),
                'max_raw_view_mse_difference': max(errors) if errors else None,
            }
    for arm in WEIGHTS:
        if (arm, 0) not in evaluations:
            continue
        initial = evaluations[(arm, 0)]
        for (current_arm, step), records in evaluations.items():
            if current_arm != arm or step == 0:
                continue
            if set(records) != set(initial) or any(
                records[ep]['rng_manifest'][engine] != initial[ep]['rng_manifest'][engine]
                for ep in initial for engine in ('cosmos', 'ctrl_world')
            ):
                summary['integrity_errors'].append(f'{arm}: actual evaluation noise changed from baseline at {step}')
                summary['complete'] = False
    for filename,rows in [('evaluation_comparison.csv',eval_rows),('training_comparison.csv',train_rows)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (out/filename).open('w') as handle:
            writer=csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out/'comparison_results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    failed_arms = [arm for arm, data in summary['arms'].items()
                   if data['runtime_status'].get('state') == 'failed']
    state_text = ('All 15 updates and the final paired evaluation are available.' if summary['complete'] else
                  ('Run interrupted (' + ', '.join(failed_arms) + '); no conclusion about the training effect can be drawn yet.'
                   if failed_arms else 'Run in progress; no final conclusion can be drawn yet.'))
    text=['# Push-T camera reward weight comparison', '',
          'Status: '+state_text, '',
          'Both arms use FP32 pose repair, 13 chunks (27.73 s), and the same SFT initialization and training parameters.',
          'Trajectory and terminal terms both compare the original weights (wrist/main/side=2/3,1/6,1/6) against (0,0.5,0.5).',
          'The held-out evaluation uses a fixed set of 30 episodes; the actual random streams are included in the paired check.', '',
          '| Updates | Arm | Successes / 30 | Success rate | main/side trajectory MSE | main/side terminal MSE |',
          '|---:|---|---:|---:|---:|---:|']
    for r in sorted(eval_rows,key=lambda r:(r['updates'],r['arm'])):
        text.append(f"| {r['updates']} | {r['arm']} | {r['successes']}/30 | {r['success_rate']:.1%} | {r.get('trajectory_rescore_no_wrist',float('nan')):.6f} | {r.get('terminal_rescore_no_wrist',float('nan')):.6f} |")
    text += ['', 'The MSE columns score both arms under the same main/side definition, so changing the weights cannot by itself produce an apparent improvement. Rescored original weights and per-view data are in the CSV/JSON.',
             'Success rates come from a fixed world-model image classifier and are not equivalent to real-robot success rates. A single training seed and 30 paired samples provide only preliminary ablation evidence.',
             'In the training CSV, success_negative_advantage is a group-normalized diagnostic of the total reward; success_negative_mean_chunk_advantage, success_negative_chunk_fraction and success_mean_chunk_advantage separately describe the per-chunk advantages actually used for optimization. The sign of the advantage is a reward-alignment diagnostic and does not on its own demonstrate a change in success rate after a policy update.']
    if summary['baseline_comparability']:
        baseline_check = summary['baseline_comparability']
        text += ['', f"Success-label disagreements between the two arms before training: {baseline_check['success_disagreements']}/30; largest per-view MSE difference: {baseline_check['max_raw_view_mse_difference']}."]
    if summary['complete']:
        index = {(row['arm'], row['updates']): row for row in eval_rows}
        gain = {arm: index[(arm,15)]['successes'] - index[(arm,0)]['successes'] for arm in WEIGHTS}
        pair = next(row for row in summary['paired_evaluation'] if row['updates'] == 15)
        text += ['', f"After 15 updates, the change in successes relative to the initial SFT checkpoint is {gain['original']:+d}/30 for the original weights and {gain['no_wrist']:+d}/30 for the zeroed wrist.",
                 f"The final success-rate difference (zeroed wrist minus original weights) is {pair['success_difference_pp']:+.2f} percentage points; paired disagreements are {pair['original_only_success']} won only by the original weights and {pair['no_wrist_only_success']} won only by the zeroed wrist, exact McNemar p={pair['mcnemar_exact_p']:.4g}."]
        if pair['success_difference_pp'] > 0 and gain['no_wrist'] > 0:
            text += ['This single training seed provides preliminary support for down-weighting the wrist view; the visual terminal states still need to be inspected and more seeds run, so this result alone cannot establish a stable effect.']
        else:
            text += ['In this round the zeroed-wrist arm did not show a success-rate gain both relative to its own initialization and relative to the original-weight arm. Raw per-view MSE, advantages on successful trajectories and within-group ranking should be read together to decide whether the limitation lies in reward alignment or in the policy update.']
    if summary['integrity_errors']:
        text += ['', 'Outstanding integrity issues:']+['- '+e for e in summary['integrity_errors']]
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    if eval_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for arm in WEIGHTS:
            rows=sorted(summary['arms'][arm]['evaluation'],key=lambda r:r['updates'])
            if not rows: continue
            x=[r['updates'] for r in rows]
            axes[0].plot(x,[100*r['success_rate'] for r in rows],marker='o',label=arm)
            for ax,key in zip(axes[1:],('trajectory_rescore_no_wrist','terminal_rescore_no_wrist')):
                if all(key in r for r in rows): ax.plot(x,[r[key] for r in rows],marker='o',label=arm)
        for ax,title in zip(axes,('Fixed held-out success (%)','Main/side trajectory MSE','Main/side terminal MSE')):
            ax.set_title(title); ax.set_xlabel('Completed updates'); ax.grid(alpha=.2)
        axes[0].set_ylim(0,100); axes[0].legend(); fig.tight_layout()
        fig.savefig(out/'comparison_curves.png',dpi=180); fig.savefig(out/'comparison_curves.pdf'); plt.close(fig)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=summarize(args.output)
    print(json.dumps({'complete':result['complete'],'integrity_errors':result['integrity_errors'],
                      'evaluations':{k:len(v['evaluation']) for k,v in result['arms'].items()}},indent=2))
