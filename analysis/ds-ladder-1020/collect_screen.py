"""Freeze same-binary shape/routing screening separately from repeated BS512 ablations."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
out=root/'screen'
archive=root/'screen-evidence.tar.gz'
if archive.exists():raise ValueError('Never overwrite screening archive')
if (out/'campaign.rc').read_text().strip()!='0':
    raise ValueError('Screening incomplete; do not publish partial shapes as completed')
identities={name:json.loads((root/name/'identity.json').read_text()) for name in ('l6','l11')}
model=dict(hidden=7168,intermediate_hidden=3072,num_experts=384,num_topk=6,num_shared_experts=0)


def run(folder,method,bs,routing='balanced',trace=False):
    report=json.loads((folder/'performance.json').read_text())
    if (report['ep'],report['bs'],report['timer'],report['recorded_rounds'],report['rounds'])!=(8,bs,False,20,10):
        raise ValueError(f'{folder}: measurement mismatch')
    if method!='l0':
        diff=json.loads((folder/'regression.json').read_text())
        if not diff['passed'] or diff['ranks']!=[True]*8:
            raise ValueError(f'{folder}: output differential failed')
    for rank in range(8):
        result=json.loads((folder/f'output-rank{rank}.json').read_text())
        if not result['finite'] or result['rank']!=rank or result['shape']!=[bs,7168]:
            raise ValueError(f'{folder}: invalid output')
    if method=='ds':
        identity=report['identity']
        if (identity['header_sha256'],identity['simd_sha256'])!=(
            'f274c707897a0e2b716019817324588abe0fb136c495cc0a7b3c80a5830d76da',
            'c8fcd758cf864c68982af8679b14c0f71eddf95b6719f5439c6d083e92729ec3'):
            raise ValueError('DS frozen source mismatch')
    else:
        for rank in range(8):
            m=json.loads((folder/f'setup/rank_{rank}/manifest.json').read_text())
            if (m['ep_size'],m['rank'],m['routing'],m['measurement'],m['input'])!=(8,rank,routing,'full_moe','fp8'):
                raise ValueError('Effective rank configuration mismatch')
            if m['config']['model']!=model or m['config']['bs_values']!=[bs]:
                raise ValueError('Screening model mismatch')
        if method in identities:
            identity=identities[method]
            if report['sha256']!=identity['libraries']['libmegamoe_kernel.so']:
                raise ValueError('Candidate kernel mismatch')
            if report['extension_sha256']!=identity['libraries'][Path(report['extension']).name]:
                raise ValueError('Candidate host extension mismatch')
    csvs=list(folder.rglob('kernel_details.csv'))
    off=[f for f in csvs if 'timer_on' not in f.parts]
    if len(off)!=8 or (trace and len(csvs)!=16):
        raise ValueError(f'{folder}: missing paired rank CSV')
    return dict(label=str(folder.relative_to(root)),mean_us=report['mean_us'],
                rank_max_mean_us=report['rank_max_mean_us'])


cases=[]
for bs in (32,96,256,1024,2048,4096):
    methods={name:run(out/f'bs{bs}'/name,name,bs,trace=name=='l11' and bs==1024)
             for name in ('l0','l6','l11','ds')}
    cases.append(dict(bs=bs,methods=methods,scope='同窗口各一次启动，仅筛选'))
routes=[]
for routing in ('random','skewed'):
    routes.append(dict(bs=96,routing=routing,methods={name:run(out/routing/name,name,96,routing)
        for name in ('l0','l6','l11')}))
ds_trace=run(out/'ds-trace-bs512','ds',512,trace=True)
for folder in (out/'bs1024/l11',out/'ds-trace-bs512'):
    raw=json.loads((folder/'timer_all_events.json').read_text())
    events=[dict(e,name=e['family'],group_id=e['core']//3,
                 core_type='AIC' if e['core']%3==0 else 'AIV',sub_id=max(0,e['core']%3-1))
            if 'family' in e else e for e in raw]
    origin=min(e['start_cycle'] for e in events if e['name']=='KERNEL_TIMING' and e['core_type']=='AIC')
    chrome=[]
    for group in range(32):
        for role,sub,offset in [('AIC',0,0),('AIV',0,1),('AIV',1,2)]:
            chrome.append(dict(ph='M',name='thread_sort_index',pid=0,tid=f'group_{group}_{role.lower()}_{sub}',args={'sort_index':group*3+offset}))
    for e in events:
        if e['name']=='KERNEL_TIMING' or any(x in e['name'] for x in ('WAIT','SYNC','DRAIN')):continue
        chrome.append(dict(ph='X',name=e['name'],pid=0,tid=f'group_{e["group_id"]}_{e["core_type"].lower()}_{e["sub_id"]}',
                           ts=(e['start_cycle']-origin)/1000,dur=(e['end_cycle']-e['start_cycle'])/1000,
                           args={k:e[k] for k in ('group_id','core_type','sub_id')}))
    if len({e['tid'] for e in chrome if e['ph']=='X'})!=96:raise ValueError('Incomplete screening trace')
    (folder/'timer_trace.json').write_text(json.dumps(chrome,separators=(',',':')))
summary=dict(ep=8,model=model,cases=cases,routing_regression=routes,identities=identities,
    ds_trace=ds_trace,accuracy='Finite and within-implementation bitwise differential, no independent Golden',
    note='BS512 is in the repeated ladder checkpoints. Other six shapes use one launch per method, not a stable-gain claim.')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
files=[]
files.append((Path(__file__),'harness/collect_screen.py'))
files.append((root/'screen.sh','harness/screen.sh'))
for f in out.rglob('*'):
    if f.is_file() and f.name!='manifest.json' and (f.suffix in ('.json','.log','.rc') or f.name=='kernel_details.csv'):
        files.append((f,str(f.relative_to(out))))
# 保留实际调用脚本和冻结DS源，不能只靠可变服务器路径重现。
for source in ('ours.py','check_label.py','dashboard_ds_runner_1020.py','sota_ds_trace_1020.py'):
    files.append((root.parent/source,'harness/'+source))
for source in ('deep_gemm/include/deep_gemm/mega_moe.hpp','deep_gemm/include/deep_gemm/simd/mega_moe.hpp'):
    files.append((root.parent/'candidate-dashboard-1020'/source,'ds-source/'+source))
cache=root.parent/'cache_dashboard_ds_optimized_1020'
for f in cache.rglob('*'):
    if f.is_file() and f.suffix!='.lock':files.append((f,'ds-jit-cache/'+str(f.relative_to(cache))))
manifest=[dict(path=rel,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f,rel in sorted(files)]
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(archive,'w:gz') as bundle:
    for f,rel in sorted(files):bundle.add(f,arcname=rel)
    bundle.add(out/'manifest.json',arcname='manifest.json')
print('SCREEN_ARCHIVE_COMPLETE',len(manifest),'files')
for case in cases:print('SHAPE',case['bs'],{k:round(v['mean_us'],4) for k,v in case['methods'].items()})
for case in routes:print('ROUTING',case['routing'],{k:round(v['mean_us'],4) for k,v in case['methods'].items()})
