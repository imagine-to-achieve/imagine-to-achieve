"""Refresh four Push-T training curves from completed rollout batches (no hashes)."""
import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/push_t_runtime_repair_20260907'
SPECS = [
    ('original', 'Original camera weights + combined reward', 'push_t_combined_mse_fp32_h13_original_15u_s5', 'continuous_combined', 1, 1),
    ('traj01', 'Wrist 0: trajectory 0.1 + terminal', 'push_t_no_wrist_traj01_fp32_h13_15u_20260906', 'continuous_combined', 1, 1),
    ('terminal_only', 'Wrist 0: terminal reward only', 'push_t_no_wrist_traj0_fp32_h13_15u_20260906', 'continuous_combined', 0, 1),
    ('trajectory_only', 'Wrist 0: trajectory reward only', 'push_t_no_wrist_traj_only_fp32_h13_15u_20260906', 'trajectory_mse', 1, 0),
]


def slope(values):
    center = (len(values) + 1) / 2
    average = mean(values)
    return sum((i-center)*(v-average) for i,v in enumerate(values,1)) / sum((i-center)**2 for i in range(1,len(values)+1))


def collect(out, now):
    data = {'checked_at_utc':now.isoformat(), 'runs':{}, 'method':'Completed training batches; 128 trajectories per batch. Changing initial scenes. First/last 5-batch means and descriptive slopes; no significance claim.'}
    for arm,label,name,reward_source,traj_active,term_active in SPECS:
        run=ROOT/'outputs'/name
        status=json.loads((run/'status.json').read_text()); n=status['completed_updates']
        logs=[json.loads(l) for l in (run/'metrics_history.jsonl').read_text().splitlines() if l.strip()]
        metrics={int(r['step']):r['metrics'] for r in logs if int(r['step'])<=n}
        assert set(metrics)==set(range(1,n+1)),(arm,'incomplete metrics')
        points=[]; records_info=[]
        for step in range(1,n+1):
            source=run/'trajectory_records'/f'update_{step-1:04d}.jsonl'
            rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
            assert len(rows)==128 and len({r['rollout_uid'] for r in rows})==128
            assert all(r['valid'] and r['update']==step-1 and r['trajectory_frames']==416 and r['training_reward_source']==reward_source for r in rows),(arm,step)
            reward=mean(r['combined_reward'] for r in rows)
            successes=sum(bool(r['model_success']) for r in rows); logged=metrics[step]
            assert abs(reward-float(logged['rollout/episode_reward_mean']))<1e-4
            assert abs(successes-float(logged['rollout/success_trajectory_count']))<1e-6
            assert abs(successes/128-float(logged['env/cosmos/success_rate']))<1e-6
            assert max(abs(sum(r['chunk_rewards'])-r['combined_reward']) for r in rows)<1e-4
            assert max(abs(traj_active*r['trajectory_reward']+term_active*r['lastframe_reward']-r['combined_reward']) for r in rows)<1e-4
            assert math.isfinite(reward)
            points.append({'step':step,'reward':float(logged['rollout/episode_reward_mean']),'success_rate':successes/128})
            records_info.append({'step':step,'source':str(source),'trajectories':128,'successes':successes})
        rewards=[p['reward'] for p in points]; success=[p['success_rate'] for p in points]
        data['runs'][arm]={'label':label,'steps':n,'max_updates':status['max_updates'],'state':status['state'],'source':str(run/'metrics_history.jsonl'),'reward_source':reward_source,'points':points,'records':records_info,
            'first5_reward':mean(rewards[:5]),'last5_reward':mean(rewards[-5:]),'first5_success':mean(success[:5]),'last5_success':mean(success[-5:]),'reward_slope_per_step':slope(rewards),'success_slope_percentage_points_per_step':100*slope(success)}
        (out/(arm+'_status.json')).write_text(json.dumps(status,indent=2)+'\n')
        with (out/(arm+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['step','reward','success_rate']);w.writeheader();w.writerows(points)
        with (out/(arm+'.dat')).open('w') as f:
            for i,p in enumerate(points):
                moving=[f'{mean(rewards[i-4:i+1]):.12g}',f'{100*mean(success[i-4:i+1]):.12g}'] if i>=4 else ['NaN','NaN']
                f.write(' '.join([str(p['step']),f'{p["reward"]:.12g}',f'{100*p["success_rate"]:.12g}',*moving])+'\n')
    (out/'training_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return data


def quoted(text):
    return json.dumps(str(text), ensure_ascii=False)


def draw(out,data):
    stamp=datetime.fromisoformat(data['checked_at_utc']).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')
    commands=['set encoding utf8', 'set datafile missing "NaN"', 'set border 3 lc rgb "#A9B7C8"', 'set tics out nomirror textcolor rgb "#526176"', 'set grid ytics lc rgb "#E5EAF0"', 'set key bottom left opaque box lc rgb "#FFFFFF"', 'set xlabel "Training step (sampled batch)"', 'set style fill solid 1.0']
    artifacts=[]
    for group in ['all_four', *data['runs']]:
        arms=list(data['runs']) if group=='all_four' else [group]
        for extension in ('png','pdf'):
            font=16 if extension=='png' else 10
            if extension=='png':
                commands.append(f'set terminal pngcairo enhanced size {1900 if len(arms)>1 else 1700},{2300 if len(arms)>1 else 680} font "DejaVu Sans,{font}"')
            else:
                commands.append(f'set terminal pdfcairo enhanced size 13in,{16 if len(arms)>1 else 5.2}in font "DejaVu Sans,{font}"')
            filename=out/(group+'.'+extension); artifacts.append(filename.name)
            commands += ['set output '+quoted(filename),f'set multiplot layout {len(arms)},2 rowsfirst margins 0.105,0.965,{0.08 if len(arms)>1 else 0.29},{0.88 if len(arms)>1 else 0.74} spacing 0.135,{0.068 if len(arms)>1 else 0.02}',
                'set label 101 "Push-T: optimized reward and training success rate" at screen 0.105,0.97 front font "DejaVu Sans,'+str(font+5)+'"',
                'set label 102 '+quoted('Updated to '+stamp+' (UTC) | 128 trajectories per point; higher reward is better')+' at screen 0.105,0.935 front',
                'set label 103 "Light: raw per-step values; dark: trailing 5-step mean. Each arm keeps its own reward definition and y range." at screen 0.105,'+('0.034' if len(arms)>1 else '0.08')+' front',
                'set label 104 "Point k is sampled by the policy after k-1 updates; initial scenes vary with step. Success is judged by the world-model image classifier." at screen 0.105,'+('0.014' if len(arms)>1 else '0.035')+' front']
            for i,arm in enumerate(arms):
                run=data['runs'][arm];n=run['steps'];values=[p['reward'] for p in run['points']]
                commands.append('set xlabel '+quoted('Training step (sampled batch)' if i==len(arms)-1 else ''))
                ticks=sorted(set(([1]+list(range(5,n+1,5))+[n]) if n>20 else list(range(1,n+1,2))+[n]))
                lo,hi=min(values),max(values);pad=max(.025,(hi-lo)*.20)
                commands += [f'set xrange [0.5:{n+.5}]','set xtics ('+','.join(map(str,ticks))+')',
                    'set format y "%.2f"',f'set yrange [{lo-pad}:{hi+pad}]', 'set ytics autofreq',
                    'set ylabel "Cumulative reward per trajectory"',
                    'set title '+quoted(f'{run["label"]} | {n}/{run["max_updates"]} steps\nOptimized reward ↑'),
                    'plot '+quoted(out/(arm+'.dat'))+' using 1:2 with linespoints lc rgb "#AAC4F2" lw 1.5 pt 7 ps 0.6 title "Per-step", "" using 1:4 with lines lc rgb "#2563EB" lw 3 title "Trailing 5-step mean"']
                if i==0:commands.extend(['unset label 101','unset label 102','unset label 103','unset label 104'])
                commands += ['set yrange [0:40]', 'set ytics 0,10,40', 'set format y "%.0f%%"', 'set ylabel "Training success rate"',
                    'set title '+quoted(f'{run["label"]} | {n}/{run["max_updates"]} steps\nTraining success rate ↑'),
                    'plot '+quoted(out/(arm+'.dat'))+' using 1:3 with linespoints lc rgb "#A3CEC1" lw 1.5 pt 7 ps 0.6 title "Per-step", "" using 1:5 with lines lc rgb "#07866C" lw 3 title "Trailing 5-step mean"']
            commands += ['unset multiplot','unset output']
    script=out/'plot.gp';script.write_text('\n'.join(commands)+'\n')
    result=subprocess.run(['gnuplot',str(script)],text=True,capture_output=True,check=True)
    if result.stderr:print(result.stderr)
    manifest={'captured_at_utc':data['checked_at_utc'],'reward':'rollout/episode_reward_mean, checked against native combined_reward and active reward components','success':'model_success count / 128, checked against training log','x':'Batch used for update k, sampled after k-1 completed policy updates','smoothing':'Trailing 5-step mean; raw values also shown. No missing points filled.','data_snapshot':str(out/'training_data.json'),'runs':{a:{'steps':r['steps'],'source':r['source']} for a,r in data['runs'].items()},'artifacts':artifacts}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    text=['# Push-T: latest training curves for the four arms','',f'Data as of {stamp} (UTC). 128 trajectories of 416 frames per step.','',
        'The reward is the mean cumulative per-trajectory reward actually used for training, where higher is better; the success rate comes from model_success. The terminal-only arm uses the combined-reward configuration with a trajectory coefficient of 0; the trajectory-only arm does not count the terminal diagnostic reward.',
        'Light lines are raw per-step values, dark lines are the trailing 5-step mean, which starts at step 5. Point k is the batch sampled before weight update k, with k-1 updates already applied. Initial scenes vary by round; these are descriptive training trends.','',
        '| Arm | Steps completed | Reward first 5 -> last 5 mean | Success first 5 -> last 5 mean |','|---|---:|---:|---:|']
    for r in data['runs'].values():
        text.append(f'| {r["label"]} | {r["steps"]}/{r["max_updates"]} | {r["first5_reward"]:.3f} → {r["last5_reward"]:.3f} | {100*r["first5_success"]:.2f}% → {100*r["last5_success"]:.2f}% |')
    text += ['','all_four.png/pdf is the overview of all four arms; each arm also has its own PNG/PDF and per-step CSV. training_data.json stores the plotted values and their provenance, and plot.gp can redraw this snapshot.','']
    (out/'README.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    now=datetime.now(timezone.utc)
    out=args.output or PARENT/('four_training_curves_'+now.strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    data=collect(out,now);draw(out,data)
    print(json.dumps({'output':str(out),'steps':{a:r['steps'] for a,r in data['runs'].items()}},ensure_ascii=False),flush=True)
    Path('/tmp/push_t_refresh_output_20260909.txt').write_text(str(out)+'\n')

if __name__=='__main__':main()
