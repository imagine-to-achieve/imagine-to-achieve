"""Check global AND actual OST-pool quotas before allocating training jobs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import importlib.util
import tempfile
import os
from pathlib import Path
import re
import subprocess


def parse_quota(output):
    for line in output.splitlines():
        fields=line.split()
        if len(fields)!=8:
            continue
        try:
            numbers={i:int(fields[i].rstrip('*')) for i in (0,1,2,4,5,6)}
        except ValueError:
            continue
        return {'blocks':{'used':numbers[0],'soft':numbers[1],'hard':numbers[2],'grace':fields[3]},
                'files':{'used':numbers[4],'soft':numbers[5],'hard':numbers[6],'grace':fields[7]}}
    raise ValueError('Cannot parse quota report; refusing to assume storage is writable.')


def quota_violations(quota):
    failures=[]
    for resource,values in quota.items():
        used,soft,hard,grace=(values[k] for k in ('used','soft','hard','grace'))
        if hard and used>=hard:
            failures.append(f'{resource}: usage {used} reaches hard limit {hard}')
        elif soft and used>soft and grace=='expired':
            failures.append(f'{resource}: usage {used} exceeds soft limit {soft} and grace expired')
    return failures


def read_command(args):
    return subprocess.run(args,check=True,text=True,capture_output=True,timeout=20).stdout


def probe_output_writes(path):
    module_path=Path(__file__).resolve().parents[1]/'third_party/rlinf_runtime/rlinf/utils/diagnostic_io.py'
    spec=importlib.util.spec_from_file_location('push_t_diagnostic_io',module_path)
    diagnostic=importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnostic)
    results=[]
    for relative in ('trajectory_records/.rank_shards','checkpoints'):
        target=Path(path)/relative
        result=diagnostic.probe_directory(target,workers=4)
        with tempfile.TemporaryDirectory(prefix='.quota_override_probe_',dir=target) as temporary:
            probe=Path(temporary)/'atomic.bin'
            diagnostic.atomic_write(probe,b'a'*(1024*1024))
            replacement=b'b'*(1024*1024)
            diagnostic.atomic_write(probe,replacement)
            if probe.read_bytes()!=replacement:
                raise RuntimeError('Actual output write probe failed readback verification.')
        results.append({'directory':str(target),'append_fsync_probe':result,'atomic_replace_verified_bytes':len(replacement)})
    return results


def apply_write_probe_exception(report):
    waived=[]
    for entry in report['checks']:
        if entry['identity']!='project' or entry['pool']!='disk':
            continue
        blocks=entry['quota']['blocks']
        if (blocks['soft'] and blocks['used']>blocks['soft'] and blocks['grace']=='expired'
                and (not blocks['hard'] or blocks['used']<blocks['hard'])):
            waived += [f"project {entry['id']}, disk: {v}" for v in quota_violations({'blocks':blocks})]
    report['original_violations']=list(report['violations'])
    report['waived_violations']=waived
    report['violations']=[v for v in report['violations'] if v not in waived]
    report['explicit_write_probe_exception']=True
    if waived and not report['violations']:
        try:
            report['live_output_write_probes']=probe_output_writes(report['path'])
        except Exception as error:
            report['violations'].append('Actual output write probe failed: '+str(error))
    report['allowed']=not report['violations']
    return report


def check_storage(path, *, allow_expired_disk_pool_after_write_probe=False):
    path=Path(path).resolve()
    existing=path
    while not existing.exists() and existing!=existing.parent:
        existing=existing.parent
    if not existing.is_dir():
        existing=existing.parent
    project=read_command(['lfs','project','-d',str(existing)]).split()[0]
    if not project.isdigit():
        raise ValueError('Cannot identify Lustre project ID.')
    layout=read_command(['lfs','getstripe','-d',str(existing)])
    pools=sorted(set(re.findall(r'\b(?:lmm_)?pool:\s*(\S+)',layout)))
    if not pools:
        raise ValueError('Cannot identify output OST pool; refusing a global-quota-only check.')
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'path':str(path),'project_id':int(project),
            'pools':pools,'checks':[],'violations':[]}
    identities=[('project','-p',project),('user','-u',str(os.getuid())),('group','-g',str(existing.stat().st_gid))]
    for identity,flag,number in identities:
        for pool in [None,*pools]:
            args=['lfs','quota','-q',flag,number]
            if pool is not None: args+=['--pool',pool]
            args.append(str(existing))
            raw=read_command(args); parsed=parse_quota(raw)
            entry={'identity':identity,'id':int(number),'pool':pool or 'global','quota':parsed}
            report['checks'].append(entry)
            report['violations'] += [f'{identity} {number}, {pool or "global"}: {v}' for v in quota_violations(parsed)]
    report['allowed']=not report['violations']
    if allow_expired_disk_pool_after_write_probe and not report['allowed']:
        return apply_write_probe_exception(report)
    return report


def require_storage(path, *, allow_expired_disk_pool_after_write_probe=False):
    report=check_storage(path,allow_expired_disk_pool_after_write_probe=allow_expired_disk_pool_after_write_probe)
    if not report['allowed']:
        raise RuntimeError('Training submission blocked by storage quota: '+'; '.join(report['violations']))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--path',type=Path,required=True); parser.add_argument('--report',type=Path)
    parser.add_argument('--allow-expired-disk-pool-after-write-probe',action='store_true',help='Explicit exception for project disk block soft-grace warnings, requiring live writes in both output subdirectories.')
    args=parser.parse_args(); result=check_storage(args.path,allow_expired_disk_pool_after_write_probe=args.allow_expired_disk_pool_after_write_probe)
    if args.report:
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['allowed'] else 2)
