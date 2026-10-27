"""Reject partial/sampled/relaxed results before the main promotion gate."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import CASES,CONFIG,REVISION,write


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def validate_case(value,ep,bs):
    reports,repeats=value['reports'],value['repeat_equal']
    if len(reports)!=ep or repeats!=[True]*ep:raise ValueError('Missing rank/repeat mismatch')
    for rank,report in enumerate(reports):
        if (report['rank'],report['checked_token_count'],report['total_tokens'],report['check_tokens'],
                report['elements'])!=(rank,bs,bs,0,bs*CONFIG['model']['hidden']):
            raise ValueError('Incomplete or sampled Golden')
        if (report['diff_thd'],report['pct_thd'],report['max_diff'])!=(0.01,0.05,0.1):
            raise ValueError('Golden tolerance changed')
        if not all(report.get(k) is True for k in ('passed','counts_equal','cumulative_counts_equal','finite')):
            raise ValueError(f'rank={rank}: accuracy/counts failed')
        if report['bad_elements']>0.05*report['elements'] or not 0<=report['max_mixed_error']<=0.1:
            raise ValueError('Invalid passing result')
        if not all(math.isfinite(report[k]) for k in ('max_mixed_error','max_abs_error','relative_rmse')):
            raise ValueError('Nonfinite error report')
    return dict(bs=bs,rank_count=ep,checked_elements=ep*bs*CONFIG['model']['hidden'],
        max_mixed_error=max(r['max_mixed_error'] for r in reports),
        max_relative_rmse=max(r['relative_rmse'] for r in reports),
        max_bad_fraction=max(r['bad_elements']/r['elements'] for r in reports),passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    manifest=load(a.root/'experiment.json');ep=manifest['ep']
    if manifest['status']!='collected_not_audited' or manifest['exit_codes']!=[0]*(ep//8):
        raise ValueError('Incomplete/failed workers')
    events=[json.loads(line) for line in (a.root/'events.jsonl').read_text().splitlines()]
    if len([e for e in events if e['stage']=='released'])!=ep//8:raise ValueError('Missing full-cohort release')
    if manifest['candidate']!=REVISION or manifest['check_tokens']!=0:raise ValueError('Wrong candidate or sample scope')
    reference={};local_reports={};native=None;records=[]
    for rank in range(ep):
        folder=a.root/f'node{rank//8}'/manifest['label']/f'rank_{rank}'
        identity=load(folder/'identity.json');complete=load(folder/'complete.json')
        if (identity['rank'],identity['ep'],identity['revision'],identity['config'],identity['capacity'],
                identity['timer'],identity['worker_sha256'])!=(rank,ep,REVISION,CONFIG,4096,False,manifest['hashes']['worker.py']):
            raise ValueError(f'rank={rank}: wrong identity')
        if native is None:native=identity['native']
        if native!=identity['native'] or complete!=dict(passed=True,cases=len(CASES),check_tokens=0,ep=ep,rank=rank):
            raise ValueError('Incomplete cases/native mismatch')
        for routing,bs in CASES:
            key=(routing,bs);value=load(folder/f'{routing}-bs{bs}-accuracy.json')
            spec=load(folder/f'{routing}-bs{bs}-contract.json')
            if (spec['world'],spec['bs'],spec['routing'],spec['check_tokens'],spec['model'],spec['seed'],spec['input'])!=\
                    (ep,bs,routing,0,CONFIG['model'],CONFIG['seed'],'fp8'):
                raise ValueError('Wrong Golden numerical/input contract')
            if value.get('report_scope')=='local':
                if len(value['reports'])!=1 or value['reports'][0]['rank']!=rank or value['repeat_equal']!=[True]*ep:
                    raise ValueError('Wrong local rank/repeat scope')
                local_reports.setdefault(key,{})[rank]=value['reports'][0]
                if key not in reference:reference[key]=(None,spec)
                elif reference[key]!=(None,spec):raise ValueError('Per-rank contracts disagree')
            elif key not in reference:reference[key]=(value,spec)
            elif reference[key]!=(value,spec):raise ValueError('Per-rank reports/contracts disagree')
    for routing,bs in CASES:
        key=(routing,bs);value,spec=reference[key]
        if key in local_reports:
            if set(local_reports[key])!=set(range(ep)):raise ValueError('Missing local rank reports')
            value=dict(reports=[local_reports[key][r] for r in range(ep)],repeat_equal=[True]*ep)
        records.append(dict(routing=routing,reference_sha256=spec['reference_sha256'],**validate_case(value,ep,bs)))
    result=dict(status='full_golden_pass',ep=ep,revision=REVISION,native=native,
        placement=dict(rack=manifest['rack'],nodes=manifest['nodes']),check_tokens=0,cases=records,
        tolerance=dict(diff_thd=0.01,pct_thd=0.05,max_diff=0.1),
        limitations=['Quantization-aware NumPy reference, not bitwise Cube/exp emulation.',
                    'No EP64/128 or shared expert validation.'])
    write(a.root/'summary.json',result)
    write(a.root/'evidence-sha256.json',{str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.root.rglob('*') if f.is_file() and f.name!='evidence-sha256.json'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
