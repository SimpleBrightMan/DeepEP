"""Compare role-local passive regions; ISSUE is not hardware completion."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def summarize(path):
    raw = json.loads(Path(path).read_text())
    events = [dict(e, name=e['family'], group_id=e['core']//3,
                   core_type='AIC' if e['core']%3 == 0 else 'AIV', sub_id=max(0,e['core']%3-1))
              if 'family' in e else e for e in raw]
    totals = collections.defaultdict(float)
    rows = []
    for group in range(32):
        aic = [e for e in events if e['group_id']==group and e['core_type']=='AIC']
        kernel = next(e for e in aic if e['name']=='KERNEL_TIMING')
        g1 = [e for e in aic if e['name']=='AIC_GMM1_ISSUE']
        g2 = [e for e in aic if e['name']=='AIC_GMM2_ISSUE']
        first, last = min(e['start_cycle'] for e in g1), max(e['end_cycle'] for e in g2)
        t1 = sum(e['end_cycle']-e['start_cycle'] for e in g1)
        t2 = sum(e['end_cycle']-e['start_cycle'] for e in g2)
        rows.append(dict(group=group, head_us=(first-kernel['start_cycle'])/1000,
                         gmm1_issue_us=t1/1000,gmm2_issue_us=t2/1000,
                         outside_issue_us=(last-first-t1-t2)/1000,
                         submit_window_us=(last-first)/1000,
                         tail_after_last_issue_us=(kernel['end_cycle']-last)/1000))
    for e in events:
        if e['name']=='KERNEL_TIMING': continue
        role = 'AIC' if e['core_type']=='AIC' else f'AIV{e["sub_id"]}'
        totals[role+'/'+e['name']] += (e['end_cycle']-e['start_cycle'])/1000/32
    return dict(path=str(path),group_means={k:statistics.mean(r[k] for r in rows) for k in rows[0] if k!='group'},
                role_stage_mean_us=dict(totals),per_group=rows,
                caveat='Passive Scalar submission regions, not hardware active time. Overlapping stages must not be summed into E2E.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('events',nargs='+')
    p.add_argument('--brief',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    results=[summarize(f) for f in a.events]
    if a.brief:
        for r in results:r.pop('per_group')
    serialized=json.dumps(results,indent=2)
    if a.output:a.output.write_text(serialized,encoding='utf-8')
    print(serialized)
