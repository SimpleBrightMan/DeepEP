"""Collect one unified binary; never synthesize per-shape winners from other branches."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

root=Path(__file__).resolve().parent
for phase in ('build','campaign'):
    if (root/f'{phase}.rc').read_text().strip()!='0':raise ValueError(f'Incomplete {phase}')
identity=json.loads((root/'identity.json').read_text())
packages={'control':root.parent/'aiv1-async-1020-v2/async-package/deep_ep_experimental/megamoe',
          'adaptive':root/'cv-package/deep_ep_experimental/megamoe'}
shas={m:hashlib.sha256((p/'libmegamoe_kernel.so').read_bytes()).hexdigest() for m,p in packages.items()}
exts={m:hashlib.sha256(next(p.glob('_C*.so')).read_bytes()).hexdigest() for m,p in packages.items()}
model=dict(hidden=7168,intermediate_hidden=3072,num_experts=384,num_topk=6,num_shared_experts=0)
result=dict(ep=8,bs=512,identity=identity,kernel_shas=shas,extension_shas=exts,model=model,methods={},screening=[],routes=[],
    scope='full communication; timer off; warm10/sample20/tail10; per-round rank-min',
    accuracy='8-rank finite and bitwise differential; independent Golden not run',
    policy='EP8 only; paired weights; expected tokens/expert <=128 -> CV; otherwise GM; one binary')


def validate(label,method,bs,routing='balanced'):
    folder=root/label
    report=json.loads((folder/'performance.json').read_text())
    diff=json.loads((folder/'regression.json').read_text())
    if (report['ep'],report['bs'],report['timer'],report['rounds'],report['recorded_rounds'],report['sha256'],report['extension_sha256'])!=(8,bs,False,10,20,shas[method],exts[method]):
        raise ValueError(f'{label}: binary/config mismatch')
    if not diff['passed'] or diff['ranks']!=[True]*8:raise ValueError(f'{label}: differential failed')
    for rank in range(8):
        manifest=json.loads((folder/f'setup/rank_{rank}/manifest.json').read_text())
        if (manifest['ep_size'],manifest['rank'],manifest['routing'],manifest['input'],manifest['measurement'])!=(8,rank,routing,'fp8','full_moe'):
            raise ValueError(f'{label}: effective routing/rank mismatch')
        if manifest['config']['bs_values']!=[bs] or manifest['config']['model']!=model:raise ValueError('Model mismatch')
    csvs=[p for p in folder.rglob('kernel_details.csv') if 'timer_on' not in p.parts]
    if len(csvs)!=8:raise ValueError(f'{label}: missing off rank CSVs')
    return dict(label=label,mean_us=report['mean_us'],rank_max_mean_us=report['rank_max_mean_us'])


for method in ('control','adaptive'):
    runs=[validate(f'{method}-{suffix}',method,512) for suffix in ('a','b')]
    result['methods'][method]=dict(runs=runs,mean_us=statistics.mean(r['mean_us'] for r in runs),
        rank_max_mean_us=statistics.mean(r['rank_max_mean_us'] for r in runs))
result['reduction_pct']=100*(1-result['methods']['adaptive']['mean_us']/result['methods']['control']['mean_us'])
for bs in (32,96,256,1024,2048,4096):
    records={m:validate(f'bs{bs}-{m}',m,bs) for m in packages}
    result['screening'].append(dict(bs=bs,methods=records,reduction_pct=100*(1-records['adaptive']['mean_us']/records['control']['mean_us']),scope='one launch each'))
for routing in ('random','skewed'):
    result['routes'].append(dict(routing=routing,bs=96,record=validate(f'{routing}-regression','adaptive',96,routing),
        scope='bitwise to prior frozen parent; no same-window performance control'))
validate('adaptive-trace','adaptive',512)
folder=root/'adaptive-trace'
events=json.loads((folder/'timer_all_events.json').read_text())
origin=min(e['start_cycle'] for e in events if e['name']=='KERNEL_TIMING' and e['core_type']=='AIC')
trace=[]
for gid in range(32):
    for role,sub,offset in (('AIC',0,0),('AIV',0,1),('AIV',1,2)):
        trace.append(dict(ph='M',name='thread_sort_index',pid=0,tid=f'group_{gid}_{role.lower()}_{sub}',args={'sort_index':gid*3+offset}))
for e in events:
    if e['name']=='KERNEL_TIMING' or any(s in e['name'] for s in ('WAIT','SYNC','DRAIN')):continue
    trace.append(dict(ph='X',name=e['name'],pid=0,tid=f"group_{e['group_id']}_{e['core_type'].lower()}_{e['sub_id']}",
        ts=(e['start_cycle']-origin)/1000,dur=(e['end_cycle']-e['start_cycle'])/1000,
        args={k:e[k] for k in ('group_id','core_type','sub_id')}))
if len({e['tid'] for e in trace if e['ph']=='X'})!=96:raise ValueError('Incomplete unified passive trace')
(folder/'timer_trace.json').write_text(json.dumps(trace,separators=(',',':')))
totals=collections.defaultdict(float)
for e in events:
    if e['name']!='KERNEL_TIMING':totals[e['name']]+=(e['end_cycle']-e['start_cycle'])/1000/32
result['trace']=dict(path='adaptive-trace/timer_trace.json',stage_mean_us=dict(totals),
    overhead=json.loads((folder/'timer_overhead.json').read_text()))
(root/'summary.json').write_text(json.dumps(result,indent=2))
sources=set()
for p in root.iterdir():
    if p.is_file() and p.suffix in ('.json','.rc','.log','.py','.sh') and p.name not in ('manifest.json','collect.log'):sources.add((p,p.name))
for folder in root.iterdir():
    if not folder.is_dir() or folder.name in ('cv-source','cv-package','native-build','native'):continue
    selected=set(folder.glob('*.json'))
    for name in ('kernel_details.csv','kernel_samples.json','op_summary*.csv','sample.json','manifest.json','resources.json'):selected.update(folder.rglob(name))
    for p in selected:
        if p.is_file():sources.add((p,str(p.relative_to(root))))
for method,package in packages.items():
    for p in [package/'libmegamoe_kernel.so',*package.glob('_C*.so')]:sources.add((p,f'native-{method}/{p.name}'))
for name in ('native/CMakeLists.txt','native-build/compile_commands.json'):sources.add((root/name,name))
for name in ('ours.py','check_label.py'):sources.add((root.parent/name,'harness/'+name))
for routing in ('random','skewed'):
    base=root.parent/f'ds-cv-stage-1020-v1/{routing}-bs96-control'
    for p in [base/'performance.json',*base.glob('output-rank*.json')]:sources.add((p,f'route-reference/{routing}/{p.name}'))
manifest=[dict(path=name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in sorted(sources,key=lambda x:x[1])]
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for p,name in sorted(sources,key=lambda x:x[1]):archive.add(p,arcname=name)
    archive.add(root/'manifest.json',arcname='manifest.json')
print('COLLECT_COMPLETE',len(manifest),'files',sum(x['path'].endswith('/kernel_details.csv') for x in manifest),'kernel CSVs',flush=True)
print('METHODS',json.dumps(result['methods']),'REDUCTION',result['reduction_pct'],flush=True)
print('SCREENING',json.dumps(result['screening']),flush=True)
print('TRACE_OVERHEAD',result['trace']['overhead']['overhead_pct'],flush=True)
