"""96 tracks alone are insufficient: fixed boundary events can hide sparse dynamic traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

REQUIRED=(('AIC',0,'AIC_GMM1_ISSUE'),('AIC',0,'AIC_GMM2_ISSUE'),
          ('AIV',0,'AIV_GMM1_PROLOGUE'),('AIV',0,'AIV_GMM2_PROLOGUE'),
          ('AIV',1,'AIV_DISPATCH_ISSUE'),('AIV',1,'AIV_SWIGLU_ISSUE'),('AIV',1,'AIV_COMBINE_ISSUE'))
ALIASES={'AIV_GMM1_PROLOGUE_ISSUE':'AIV_GMM1_PROLOGUE','AIV_GMM2_PROLOGUE_ISSUE':'AIV_GMM2_PROLOGUE'}


def validate(events,groups=32):
    counts=Counter()
    for e in events:
        if 'family' in e:
            group=e['core']//3;role='AIC' if e['core']%3==0 else 'AIV';sub=max(0,e['core']%3-1);name=e['family']
        else:
            group,role,sub,name=e['group_id'],e['core_type'],e['sub_id'],e['name']
        if not 0<=group<groups:raise ValueError(f'Unexpected group={group}; expected 0..{groups-1}')
        counts[group,role,sub,ALIASES.get(name,name)]+=1
    missing=[f'g{group}/{role}{sub}/{name}' for group in range(groups) for role,sub,name in REQUIRED
             if counts[group,role,sub,name]==0]
    if missing:raise ValueError('Sparse dynamic coverage: '+', '.join(missing[:8]))
    return dict(groups=groups,dynamic_groups=groups,tracks=groups*3,
                minimum_dynamic_regions=min(counts[group,role,sub,name] for group in range(groups)
                                            for role,sub,name in REQUIRED))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folders',nargs='+',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    reports=[]
    for folder in a.folders:
        for f in sorted(folder.rglob('timer_all_events.json')):
            raw=f.read_bytes()
            reports.append(dict(path=str(f.resolve()),sha256=hashlib.sha256(raw).hexdigest(),**validate(json.loads(raw))))
    if not reports:raise ValueError('No trace files found')
    if a.output:a.output.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('FULL_DYNAMIC_TRACE_COVERAGE_PASS',len(reports),'traces; every group includes all seven required role/stage families')
