"""审计实际rank CSV/退出码/输入输出；缺测保持pending，不继承历史成绩。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def audit_run(root,label,ep,registry,timer_requested):
    saved=load(root/label/'performance.json')
    method=label.split('-')[1]
    bs=int(label.split('-')[0][2:])
    if (saved['ep'],saved['bs'],saved['timer'],saved['model']['num_experts'])!=(ep,bs,False,384):
        raise ValueError(f'{label}: wrong measurement domain')
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    exits=[e for e in events if e['stage']=='node_exit' and e['label']==label]
    if len(exits)!=ep//8 or any(e['exit']!=0 for e in exits):
        raise ValueError(f'{label}: incomplete launch')
    traced=timer_requested and label.endswith('-a')
    reports=[('profiler',saved)]
    on=load(root/label/'timer_on_performance.json') if traced else None
    if traced:
        if on['timer_ranks']!=[0] or not on['all_measured_invocations_instrumented']:
            raise ValueError('Wrong timer overhead scope')
        reports.append(('timer_profiler',on))
    inputs=[];outputs=[];csv_count=0
    for kind,report in reports:
        samples=[]
        for rank in range(ep):
            folder=root/f'node{rank//8}/runs/{label}/rank_{rank}'
            identity=load(folder/'identity.json')
            if identity['native']!=registry[method]['native'] or identity['rank']!=rank:
                raise ValueError(f'{label} rank={rank}: wrong native identity')
            files=list((folder/kind).rglob('kernel_details.csv'))
            if len(files)!=1:raise ValueError(f'{label} rank={rank}: missing CSV')
            with files[0].open(encoding='utf-8-sig',newline='') as f:
                rows=[r for r in csv.DictReader(f) if 'megamoe' in r['Name'].lower().replace('_','')]
            rows.sort(key=lambda r:float(r['Start Time(us)']))
            if len(rows)!=20:raise ValueError(f'{label} rank={rank}: expected20 kernels')
            samples.append([float(r['Duration(us)']) for r in rows[-10:]])
            csv_count+=1
            if kind=='profiler':
                inputs.append(load(folder/'inputs.json'));outputs.append(load(folder/'output.json'))
                if not outputs[-1]['finite']:raise ValueError('Nonfinite output')
                if traced and load(folder/'timer_output.json')!=outputs[-1]:raise ValueError('Timer output mismatch')
        lo=statistics.mean(min(r[i] for r in samples) for i in range(10))
        hi=statistics.mean(max(r[i] for r in samples) for i in range(10))
        if abs(lo-report['mean_us'])>0.00011 or abs(hi-report['rank_max_mean_us'])>0.00011:
            raise ValueError(f'{label}: CSV aggregation mismatch')
    trace=None
    if traced:
        trace_dir=root/f'node0/runs/{label}/rank_0/timer'
        meta=load(trace_dir/'metadata.json')
        native=load(trace_dir/'native_events.json')
        if meta['dynamic_groups']!=32 or meta['dynamic_aiv0']!=1 or not native:
            raise ValueError('Trace coverage mismatch')
        if any(e['rank']!=0 or not 0<e['start_cycle']<e['end_cycle'] for e in native):
            raise ValueError('Invalid native interval')
        if not (trace_dir/'timer_trace.json').is_file():raise ValueError('Missing visual trace')
        trace=dict(json=str(trace_dir/'timer_trace.json'),events=meta['event_count'])
    record=dict(label=label,method=method,mean_us=saved['mean_us'],rank_max_mean_us=saved['rank_max_mean_us'],
        on_us=on['mean_us'] if on else None,overhead_pct=100*(on['mean_us']/saved['mean_us']-1) if on else None,
        csv_count=csv_count,trace=trace)
    return record,inputs,outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--partial',action='store_true')
    a=p.parse_args();manifest=load(a.root/'experiment.json');ep=manifest['ep']
    cases=[]
    for bs in manifest['bs_values']:
        labels=[f'bs{bs}-{job}' for job in manifest['schedule']]
        if not all(label in manifest['completed'] for label in labels):
            cases.append(dict(ep=ep,bs=bs,status='pending'));continue
        records=[];inputs=None;outputs=None
        for label in labels:
            record,ins,outs=audit_run(a.root,label,ep,manifest['packages'],manifest['timer_requested'])
            if inputs is None:inputs,outputs=ins,outs
            if inputs!=ins or outputs!=outs:raise ValueError(f'EP{ep} BS{bs}: control/candidate fingerprint mismatch')
            records.append(record)
        methods={}
        for method in ('control','combined'):
            runs=[r for r in records if r['method']==method]
            methods[method]=dict(mean_us=statistics.mean(r['mean_us'] for r in runs),
                rank_max_mean_us=statistics.mean(r['rank_max_mean_us'] for r in runs),starts=2,runs=runs)
        cases.append(dict(ep=ep,bs=bs,status='audited_complete',methods=methods,
            reduction_pct=100*(1-methods['combined']['mean_us']/methods['control']['mean_us']),
            output_hashes_equal=True))
    complete=all(c['status']=='audited_complete' for c in cases)
    if not complete and not a.partial:raise ValueError('Matrix incomplete; use --partial without claiming completion')
    result=dict(ep=ep,status='audited_complete' if complete else 'partial',cases=cases,
        scope=manifest['scope'],accuracy=manifest['accuracy'],packages=manifest['packages'],
        placement=dict(rack=manifest['rack'],nodes=manifest['nodes']),
        caveats=['No independent Golden.','rank-min is not global slowest-rank latency.',
                 'ISSUE is not hardware completion; timer perturbation can include measurement noise.'])
    (a.root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if complete:
        manifest['status']='audited_complete'
        (a.root/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        hashes={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.root.rglob('*')
                if f.is_file() and f.name!='manifest.json'}
        (a.root/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
