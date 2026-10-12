"""Collect completed native screens, deduplicating snapshots, not counting samples."""
import hashlib
import json
from pathlib import Path

W=Path('C:/Project/Infra/ops-transformer-hs')
HERE=Path(__file__).resolve().parent
OLD=W/'.codex_worktrees/megamoe-ep8-parity-1012/analysis'
sources=list(OLD.rglob('*.json'))+list((W/'outputs/ep8-parity-1012').rglob('*.json'))+[W/'outputs/round47-final.json']
found={}
for path in sources:
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value,dict):
        continue
    for item in value.get('variants',[]):
        if not isinstance(item,dict) or item.get('status')!='screened' or len(item.get('reports',[]))!=5:
            continue
        if not item['id'].startswith('e'):
            continue
        previous=found.get(item['id'])
        if previous is None or item.get('finished',0)>previous[0].get('finished',0):
            found[item['id']]=(item,path)
rows=[]
for name,(item,path) in sorted(found.items()):
    reports={r['bs']:r['report'] for r in item['reports']}
    assert set(reports)=={32,96,256,1024,4096},name
    assert item.get('accuracy_returncode')==0 and item.get('perf_returncode')==0,name
    for bs,r in reports.items():
        assert r['aggregation']=='mean_of_per_round_min_rank' and r['measurement']=='full_moe',(name,bs)
    rows.append(dict(id=name,parent=item.get('parent'),hypothesis=item.get('hypothesis'),
        native_commit=item.get('native_commit'),wheel_sha256=item.get('wheel_sha256'),
        means={str(bs):r['mean_us'] for bs,r in reports.items()},
        source=str(path.relative_to(W)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        accuracy_verdicts=item.get('accuracy_verdicts'),reports=item['reports']))
target=HERE/'evidence/experiment-catalog.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('Completed distinct native screens:',len(rows))
