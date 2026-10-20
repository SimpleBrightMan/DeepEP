"""Checkpoint completed nodes; pending/failed nodes remain explicit, never fabricated."""
import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import tarfile

p=argparse.ArgumentParser()
p.add_argument('checkpoint')
a=p.parse_args()
if not a.checkpoint.replace('-','').isalnum(): raise ValueError('Invalid checkpoint')
root=Path(__file__).resolve().parent
sources=set()
summary=dict(scope='EP8 BS512 balanced full communication; timer-off; warm10/sample20/tail10; per-round rank-min',
    accuracy='8-rank finite/bitwise differential within each mathematical implementation; independent Golden not run',
    nodes=[],incomplete=[],auxiliary=[])
expected_model=dict(hidden=7168,intermediate_hidden=3072,num_experts=384,num_topk=6,num_shared_experts=0)

def add(path,name=None):
    if path.is_file(): sources.add((path,name or str(path.relative_to(root))))

def validate(node,label,method,identity,bs=512):
    folder=node/label
    report=json.loads((folder/'performance.json').read_text())
    diff=json.loads((folder/'regression.json').read_text())
    if not diff['passed'] or diff['ranks']!=[True]*8: raise ValueError(f'{folder}: output regression')
    if (report['ep'],report['bs'],report['timer'],report['recorded_rounds'],report['rounds'])!=(8,bs,False,20,10):
        raise ValueError('Measurement contract mismatch')
    if method=='ds':
        r=report['identity']
        if (r['header_sha256'],r['simd_sha256'])!=('f274c707897a0e2b716019817324588abe0fb136c495cc0a7b3c80a5830d76da',
            'c8fcd758cf864c68982af8679b14c0f71eddf95b6719f5439c6d083e92729ec3'):
            raise ValueError('DS identity mismatch')
        if r['routing']!='balanced' or r['shared']!=0 or r['seed']!=20260922: raise ValueError('DS config mismatch')
    else:
        package=Path(report['package'])/'deep_ep_experimental/megamoe'
        for field,name in [('sha256','libmegamoe_kernel.so'),('extension_sha256',Path(report['extension']).name)]:
            if report[field]!=hashlib.sha256((package/name).read_bytes()).hexdigest(): raise ValueError('Actual SO mismatch')
            if method=='candidate' and report[field]!=identity['libraries'][name]: raise ValueError('Candidate identity mismatch')
        for rank in range(8):
            m=json.loads((folder/f'setup/rank_{rank}/manifest.json').read_text())
            if (m['ep_size'],m['rank'],m['routing'],m['measurement'],m['input'])!=(8,rank,'balanced','full_moe','fp8'):
                raise ValueError('Effective rank config mismatch')
            if m['config']['model']!=expected_model or m['config']['bs_values']!=[bs]: raise ValueError('Model mismatch')
    csvs=[f for f in folder.rglob('kernel_details.csv') if 'timer_on' not in f.parts]
    if len(csvs)!=8: raise ValueError('Missing rank CSVs')
    return dict(label=str(folder.relative_to(root)),mean_us=report['mean_us'],rank_max_mean_us=report['rank_max_mean_us'])

def export_trace(trace):
    raw=json.loads((trace/'timer_all_events.json').read_text())
    events=[dict(e,name=e['family'],group_id=e['core']//3,
                 core_type='AIC' if e['core']%3==0 else 'AIV',sub_id=max(0,e['core']%3-1))
            if 'family' in e else e for e in raw]
    origin=min(e['start_cycle'] for e in events if e['name']=='KERNEL_TIMING' and e['core_type']=='AIC')
    chrome=[]
    for group in range(32):
        for role,sub,off in [('AIC',0,0),('AIV',0,1),('AIV',1,2)]:
            chrome.append(dict(ph='M',name='thread_sort_index',pid=0,tid=f'group_{group}_{role.lower()}_{sub}',args={'sort_index':group*3+off}))
    totals=collections.defaultdict(float)
    for e in events:
        if e['name']=='KERNEL_TIMING':continue
        totals[e['name']]+=(e['end_cycle']-e['start_cycle'])/1000/32
        if any(s in e['name'] for s in ('WAIT','SYNC','DRAIN')):continue
        chrome.append(dict(ph='X',name=e['name'],pid=0,tid=f"group_{e['group_id']}_{e['core_type'].lower()}_{e['sub_id']}",
            ts=(e['start_cycle']-origin)/1000,dur=(e['end_cycle']-e['start_cycle'])/1000,
            args={k:e[k] for k in ('group_id','core_type','sub_id')}))
    if len({e['tid'] for e in chrome if e['ph']=='X'})!=96:raise ValueError('Incomplete passive trace')
    (trace/'timer_trace.json').write_text(json.dumps(chrome,separators=(',',':')))
    return dict(path=str((trace/'timer_trace.json').relative_to(root)),stage_mean_us=dict(totals),
        overhead=json.loads((trace/'timer_overhead.json').read_text())['overhead_pct'])

for node in sorted(root.iterdir()):
    if not node.is_dir() or node.name in ('native','screen','trace-pair'): continue
    rc=node/'campaign.rc'
    if not rc.exists() or rc.read_text().strip()!='0':
        summary['incomplete'].append(dict(name=node.name,build_rc=(node/'build.rc').read_text().strip() if (node/'build.rc').exists() else None,
            campaign_rc=rc.read_text().strip() if rc.exists() else None))
    else:
        identity=json.loads((node/'identity.json').read_text())
        with tarfile.open(root/f'{node.name}.tar') as bundle: revision=bundle.pax_headers['comment']
        if not revision.startswith(identity['revision']): raise ValueError('Archive identity mismatch')
        for name,sha in identity['source_hashes'].items():
            if hashlib.sha256((node/'source'/name).read_bytes()).hexdigest()!=sha: raise ValueError('Built source changed')
        methods={}
        for method in ('l0','parent','candidate','ds'):
            if not (node/f'{method}-a/performance.json').exists():
                if method=='parent': continue
                raise ValueError(f'Missing {method}')
            runs=[validate(node,f'{method}-{side}',method,identity) for side in ('a','b')]
            methods[method]=dict(runs=runs,mean_us=statistics.mean(r['mean_us'] for r in runs),
                rank_max_mean_us=statistics.mean(r['rank_max_mean_us'] for r in runs))
        parent=methods.get('parent',methods['l0'])
        item=dict(name=node.name,revision=revision,kernel_sha256=identity['libraries']['libmegamoe_kernel.so'],methods=methods,
            parent_reduction_pct=100*(1-methods['candidate']['mean_us']/parent['mean_us']),
            l0_reduction_pct=100*(1-methods['candidate']['mean_us']/methods['l0']['mean_us']),
            ds_increase_pct=100*(methods['candidate']['mean_us']/methods['ds']['mean_us']-1))
        item['trace']=export_trace(node/'candidate-a')
        summary['nodes'].append(item)
        package=node/'package/deep_ep_experimental/megamoe'
        for f in [package/'libmegamoe_kernel.so',*package.glob('_C*.so')]: add(f,f'{node.name}/native/{f.name}')
    build_rc=(node/'build.rc').read_text().strip() if (node/'build.rc').exists() else None
    if not rc.exists() and build_rc in (None,'0'):
        continue  # 在途日志/产物不能进入已冻结清单，避免manifest与归档时刻不一致。
    for f in node.iterdir():
        if f.is_file() and f.suffix in ('.json','.log','.rc'): add(f)
    for folder in node.iterdir():
        if not folder.is_dir() or folder.name in ('native-build','source','package'): continue
        for f in folder.rglob('*'):
            if f.is_file() and (f.suffix=='.json' or f.name in ('kernel_details.csv','metric_summary.db')): add(f)
    add(node/'native-build/compile_commands.json')

aux=root/'trace-pair'
if (aux/'campaign.rc').exists() and (aux/'campaign.rc').read_text().strip()=='0':
    identity=json.loads((root/'l6/identity.json').read_text())
    methods={}
    for name,method in [('l6','candidate'),('ds','ds')]:
        record=validate(aux,name,method,identity,bs=1024)
        record['trace']=export_trace(aux/name)
        if len(list((aux/name).rglob('kernel_details.csv')))!=16:raise ValueError('Auxiliary on/off CSV missing')
        methods[name]=record
    summary['auxiliary'].append(dict(ep=8,bs=1024,methods=methods,
        scope='L6/DS timer-off plus separate full passive timer; one launch each, not a new optimization'))
    for f in aux.rglob('*'):
        if f.is_file() and (f.suffix in ('.json','.log','.rc') or f.name=='kernel_details.csv'):add(f)
    add(root/'trace-pair-controller.log')

stable={n['name'] for n in summary['nodes']} | {n['name'] for n in summary['incomplete']
    if n['campaign_rc'] is not None or n['build_rc'] not in (None,'0')}
for f in root.iterdir():
    if f.is_file() and (f.suffix in ('.sh','.py','.tar') or
            (f.suffix=='.log' and any(f.name==name+'-controller.log' for name in stable))): add(f)
add(root/'native/CMakeLists.txt')
output=root/f'{a.checkpoint}-summary.json'
output.write_text(json.dumps(summary,indent=2));add(output)
manifest=[dict(path=name,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    for f,name in sorted(sources,key=lambda x:x[1])]
manifest_path=root/f'{a.checkpoint}-manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
archive_path=root/f'{a.checkpoint}-evidence.tar.gz'
if archive_path.exists(): raise ValueError('Never overwrite a checkpoint')
with tarfile.open(archive_path,'w:gz') as bundle:
    for f,name in sorted(sources,key=lambda x:x[1]): bundle.add(f,arcname=name)
    bundle.add(manifest_path,arcname='manifest.json')
print('CHECKPOINT_COMPLETE',len(manifest),'files',sum(x['path'].endswith('/kernel_details.csv') for x in manifest),'kernel CSVs')
for node in summary['nodes']:
    print('NODE',node['name'],{m:round(r['mean_us'],4) for m,r in node['methods'].items()},
        'parent_reduction_pct',node['parent_reduction_pct'],'l0_reduction_pct',node['l0_reduction_pct'])
print('INCOMPLETE',summary['incomplete'])
