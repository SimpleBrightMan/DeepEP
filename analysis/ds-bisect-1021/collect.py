"""Archive only verified completed runs, including negative results and actual native/JIT files."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('checkpoint')
a=p.parse_args();root=a.campaign.resolve()
if not a.checkpoint.replace('-','').isalnum():raise ValueError('Invalid checkpoint')
files={}
def add(f,name=None):
    if f.is_file():files[name or str(f.relative_to(root))]=f
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def run(folder,bs):
    r=json.loads((folder/'performance.json').read_text())
    if (r['ep'],r['bs'],r['timer'],r['recorded_rounds'],r['rounds'])!=(8,bs,False,20,10):
        raise ValueError(f'Measurement contract mismatch: {folder}')
    diff=json.loads((folder/'regression.json').read_text())
    if not diff['passed'] or diff['ranks']!=[True]*8:raise ValueError(f'Output regression: {folder}')
    csvs=[f for f in folder.rglob('kernel_details.csv') if 'timer_on' not in f.parts]
    if len(csvs)!=8:raise ValueError(f'Missing rank CSVs: {folder}')
    return dict(label=str(folder.relative_to(root)),mean_us=r['mean_us'],rank_max_mean_us=r['rank_max_mean_us'],
                samples_us=r.get('rounds_us',r.get('samples_us')),identity=r.get('identity'),
                kernel_sha256=r.get('sha256'),extension_sha256=r.get('extension_sha256'))

def method(folder,bs,label):
    runs=[run(folder/f'bs{bs}-{label}-{suffix}',bs) for suffix in ('a','b')]
    return dict(runs=runs,**{key:statistics.mean(r[key] for r in runs) for key in ('mean_us','rank_max_mean_us')})

def trace(folder):
    raw=json.loads((folder/'timer_all_events.json').read_text())
    events=[dict(e,name=e['family'],group_id=e['core']//3,core_type='AIC' if e['core']%3==0 else 'AIV',
                 sub_id=max(0,e['core']%3-1)) if 'family' in e else e for e in raw]
    required={('AIC',0,'AIC_GMM1_ISSUE'),('AIC',0,'AIC_GMM2_ISSUE'),('AIV',0,'AIV_GMM1_PROLOGUE'),
              ('AIV',0,'AIV_GMM2_PROLOGUE'),('AIV',1,'AIV_DISPATCH_ISSUE'),
              ('AIV',1,'AIV_SWIGLU_ISSUE'),('AIV',1,'AIV_COMBINE_ISSUE')}
    aliases={'AIV_GMM1_PROLOGUE_ISSUE':'AIV_GMM1_PROLOGUE','AIV_GMM2_PROLOGUE_ISSUE':'AIV_GMM2_PROLOGUE'}
    for group in range(32):
        seen={(e['core_type'],e['sub_id'],aliases.get(e['name'],e['name'])) for e in events if e['group_id']==group}
        if not required<=seen:raise ValueError(f'Sparse dynamic trace group={group}: {folder}')
    origin=min(e['start_cycle'] for e in events if e['name']=='KERNEL_TIMING' and e['core_type']=='AIC')
    chrome=[dict(ph='M',name='thread_sort_index',pid=0,tid=f'group_{g}_{role}_{sub}',args=dict(sort_index=g*3+offset))
            for g in range(32) for role,sub,offset in [('aic',0,0),('aiv',0,1),('aiv',1,2)]]
    for e in events:
        if e['name']=='KERNEL_TIMING' or any(s in e['name'] for s in ('WAIT','SYNC','DRAIN')):continue
        chrome.append(dict(ph='X',name=e['name'],pid=0,tid=f"group_{e['group_id']}_{e['core_type'].lower()}_{e['sub_id']}",
            ts=(e['start_cycle']-origin)/1000,dur=(e['end_cycle']-e['start_cycle'])/1000,
            args={k:e[k] for k in ('group_id','core_type','sub_id')}))
    target=folder/'timer_trace.json';target.write_text(json.dumps(chrome,separators=(',',':')))
    overhead=json.loads((folder/'timer_overhead.json').read_text())
    if len(list(folder.rglob('kernel_details.csv')))!=16:raise ValueError(f'Missing off/on CSV pair: {folder}')
    return dict(path=str(target.relative_to(root)),groups=32,tracks=96,overhead_pct=overhead['overhead_pct'])

summary=dict(scope='EP8 BS512/1024 full communication; timer-off; warm10/sample20/tail10; per-round rank-min',
             accuracy='8-rank finite/bitwise differential within each implementation; no independent Golden',
             endpoint=[],nodes=[],incomplete=[])
stable={'d0'}
endpoint=root/'endpoint'
if (endpoint/'campaign.rc').is_file() and (endpoint/'campaign.rc').read_text().strip()=='0':
    stable.add('endpoint')
    for bs in (512,1024):
        methods={name:method(endpoint,bs,name) for name in ('l6','l12','reference','endpoint')}
        summary['endpoint'].append(dict(bs=bs,methods=methods,trace=trace(endpoint/f'bs{bs}-endpoint-a')))
for name in ('c1','c1-r1','c2','c3','c4','c5','d1','d2','d3'):
    node=root/name
    if not node.is_dir():continue
    rc=(node/'campaign.rc').read_text().strip() if (node/'campaign.rc').exists() else None
    brc=(node/'build.rc').read_text().strip() if (node/'build.rc').exists() else None
    if rc is not None or brc not in (None,'0'):stable.add(name)
    if rc!='0':
        summary['incomplete'].append(dict(name=name,campaign_rc=rc,
            build_rc=(node/'build.rc').read_text().strip() if (node/'build.rc').exists() else None))
        continue
    identity=json.loads((node/'identity.json').read_text())
    rows=[]
    for bs in (512,1024):
        names=('parent','candidate','ds') if name.startswith('c') else ('parent','candidate')
        methods={m:method(node,bs,m) for m in names}
        if name.startswith('c'):
            if any(r['kernel_sha256']!=identity['libraries']['libmegamoe_kernel.so'] for r in methods['candidate']['runs']):
                raise ValueError('Actual measured kernel differs from node identity')
        rows.append(dict(bs=bs,methods=methods,trace=trace(node/f'bs{bs}-candidate-a'),
            parent_reduction_pct=100*(1-methods['candidate']['mean_us']/methods['parent']['mean_us'])))
    summary['nodes'].append(dict(name=name,identity=identity,cases=rows))
for node in root.iterdir():
    if node.is_dir() and node.name in stable:
        for f in node.rglob('*'):
            rel=f.relative_to(node)
            if 'source' in rel.parts or 'package' in rel.parts:continue
            if f.suffix in ('.json','.log','.rc') or f.name in ('kernel_details.csv','kernel.asc','kernel.o'):
                add(f)
        package=node/'package'
        for f in package.rglob('*.so'):add(f,f'{node.name}/native/{f.name}')
screen=root/'screen'
if (screen/'campaign.rc').exists():
    screen_rc=(screen/'campaign.rc').read_text().strip()
    cases=[];routing=[]
    for bs in (32,96,256,512,1024,2048,4096):
        folder=screen/f'bs{bs}'
        if not all((folder/m/'regression.json').exists() for m in ('l6','candidate','ds')):continue
        cases.append(dict(bs=bs,methods={m:run(folder/m,bs) for m in ('l6','candidate','ds')}))
    for mode in ('random','skewed'):
        folder=screen/mode
        if all((folder/m/'regression.json').exists() for m in ('l6','candidate')):
            routing.append(dict(bs=96,routing=mode,methods={m:run(folder/m,96) for m in ('l6','candidate')}))
    if screen_rc=='0' and (len(cases)!=7 or len(routing)!=2):raise ValueError('Incomplete successful screening')
    expected=json.loads((root/'c4/identity.json').read_text())['libraries']['libmegamoe_kernel.so']
    if any(c['methods']['candidate']['kernel_sha256']!=expected for c in cases+routing):raise ValueError('Screen changed native candidate')
    summary['screening']=dict(campaign_rc=screen_rc,cases=cases,routing=routing,
        scope='one launch/method/shape; same C4 binary; no per-shape splice; screening not stability proof')
    for f in screen.rglob('*'):
        if f.suffix in ('.json','.log','.rc') or f.name=='kernel_details.csv':add(f)
    stable.add('screen')
for f in root.iterdir():
    if f.suffix in ('.sh','.py','.tar') or (f.suffix=='.log' and any(f.name.startswith(n+'-') for n in stable)):add(f)
report=root/f'{a.checkpoint}-summary.json'
if report.exists():raise ValueError('Checkpoint already exists; do not overwrite')
report.write_text(json.dumps(summary,indent=2));add(report)
manifest=[dict(path=name,size=f.stat().st_size,sha256=sha(f)) for name,f in sorted(files.items())]
mf=root/f'{a.checkpoint}-manifest.json';mf.write_text(json.dumps(manifest,indent=2))
archive=root/f'{a.checkpoint}-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as bundle:
    for name,f in sorted(files.items()):bundle.add(f,arcname=name,recursive=False)
    bundle.add(mf,arcname='manifest.json',recursive=False)
print('CHECKPOINT_COMPLETE',a.checkpoint,'nodes=',len(summary['nodes']),'endpoint=',len(summary['endpoint']),
      'files=',len(files),'csv=',sum(e['path'].endswith('kernel_details.csv') for e in manifest),
      'bytes=',archive.stat().st_size,'sha256=',sha(archive),flush=True)
