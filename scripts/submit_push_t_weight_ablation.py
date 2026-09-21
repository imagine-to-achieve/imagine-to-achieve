"""Submit the paired three-segment experiments only after actual config validation."""
from __future__ import annotations
import argparse
import json
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from rlinf_modified.config import load_config
from rlinf_modified.launch import _segments
from check_push_t_storage_quota import require_storage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/push_t_view_weight_ablation_fp32_h13_20260904'
parser=argparse.ArgumentParser(); parser.add_argument('--submit',action='store_true'); parser.add_argument('--retry-failed',action='store_true')
parser.add_argument('--allow-expired-disk-pool-after-write-probe',action='store_true')
args=parser.parse_args()
if args.submit:
    for arm in ('original','no_wrist'):
        require_storage(ROOT/'outputs'/f'push_t_combined_mse_fp32_h13_{arm}_15u_s5',allow_expired_disk_pool_after_write_probe=args.allow_expired_disk_pool_after_write_probe)
validation=json.loads((OUT/'composition_validation.json').read_text())
assert all(validation[arm]['state']=='passed' for arm in ('original','no_wrist'))
previous = OUT/'submission_records.json'
if previous.exists():
    if not args.retry_failed:
        if args.submit:
            raise RuntimeError('Submission record exists; use --retry-failed only for an inactive failed attempt.')
    else:
        prior = json.loads(previous.read_text())
        job_ids = ','.join(row['job_id'] for row in prior)
        active = subprocess.run(['squeue','-h','-j',job_ids,'-o','%i|%T'],check=True,text=True,capture_output=True).stdout.strip()
        if active:
            raise RuntimeError('Previous jobs are still active: '+active)
        accounting = subprocess.run(['sacct','-X','-n','-P','-j',job_ids,'--format=JobIDRaw,State,ExitCode'],check=True,text=True,capture_output=True).stdout
        states = {line.split('|')[0]:line.split('|')[1] for line in accounting.splitlines() if '|' in line}
        if not all(states.get(row['job_id'],'').startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')) for row in prior):
            raise RuntimeError('Previous attempt is not completely failed/inactive: '+repr(states))
        for arm in ('original','no_wrist'):
            run = ROOT/'outputs'/f'push_t_combined_mse_fp32_h13_{arm}_15u_s5'
            if list((run/'trajectory_records').glob('update_*.jsonl')):
                raise RuntimeError('Completed updates exist; resume their checkpoint instead of restarting.')
        if args.submit:
            archive = OUT/'submission_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            archive.mkdir(parents=True)
            shutil.copy2(previous,archive/'submission_records.json')
            (archive/'sacct.txt').write_text(accounting)
records=[]
for arm in ('original','no_wrist'):
    path=ROOT/'configs'/f'push_t_combined_mse_fp32_h13_{arm}_15u_s5.yaml'
    cfg=load_config(path); log=Path(cfg.runtime.output_dir)/'slurm'; log.mkdir(parents=True,exist_ok=True)
    dependency=None
    for start,end in _segments(cfg):
        command=['sbatch','--parsable',f'--account={cfg.slurm.account}',f'--partition={cfg.slurm.partition}',
                 f'--nodes={cfg.slurm.nodes}',f'--ntasks-per-node={cfg.slurm.ntasks_per_node}',
                 f'--cpus-per-task={cfg.slurm.cpus_per_task}',f'--gpus-per-node={cfg.slurm.gpus_per_node}',
                 f'--mem={cfg.slurm.memory}',f'--time={cfg.slurm.time_limit}',
                 f'--signal=B:USR1@{cfg.slurm.signal_seconds}',f'--job-name={cfg.experiment_name}-{start}-{end}',
                 f'--output={log}/%x-%j.out',
                 f'--export=ALL,RLINF_CONFIG={path},RLINF_SEGMENT_START={start},RLINF_SEGMENT_END={end},RLINF_ALLOW_EXPIRED_DISK_POOL_AFTER_WRITE_PROBE={int(args.allow_expired_disk_pool_after_write_probe)}']
        if dependency: command.append(f'--dependency=afterok:{dependency}')
        command += [str(ROOT/'slurm/train_push_t_view_weights.sbatch'),'--config',str(path)]
        if args.submit:
            job=subprocess.run(command,check=True,text=True,capture_output=True).stdout.strip().split(';')[0]
            if not job.isdigit(): raise RuntimeError('Unexpected sbatch response '+repr(job))
        else: job=f'DRYRUN_{arm}_{end}'
        records.append({'arm':arm,'segment_start':start,'segment_end':end,'job_id':job,'argv':command,'explicit_expired_disk_pool_exception':args.allow_expired_disk_pool_after_write_probe})
        dependency=job
        if args.submit:
            temporary=OUT/'submission_records.json.tmp'
            temporary.write_text(json.dumps(records,indent=2)+'\n'); temporary.replace(OUT/'submission_records.json')
        print(arm,start,end,job,flush=True)
if not args.submit:
    (OUT/'submission_preview.json').write_text(json.dumps(records,indent=2)+'\n')
