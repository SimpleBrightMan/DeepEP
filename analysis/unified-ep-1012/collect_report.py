"""Build report inputs from measured records; no fabricated or missing results."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

W = Path('C:/Project/Infra/ops-transformer-hs')
HERE = Path(__file__).resolve().parent
OUT = W/'outputs/megamoe-best-unified-1012'
OLD = W/'.codex_worktrees/megamoe-ep8-parity-1012/analysis'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-pending',action='store_true')
    args=parser.parse_args()
    state = read(OUT/'campaign.json')
    assert state['status']=='complete' or args.allow_pending, state.get('error',state['status'])
    paired = read(OLD/'ep8-parity-recheck43-1012/results-final.json')
    ds8 = {bs: mean(r['report']['mean_us'] for run in paired['records'] if run['method']=='ds'
                    for r in run['reports'] if r['bs']==bs) for bs in (32,96,256,1024,4096)}
    with (W/'outputs/ep32-main-20261011/main_ep32.csv').open(encoding='utf-8-sig') as f:
        main32 = {int(r['bs_per_rank']): float(r['rank_min_mean_us']) for r in csv.DictReader(f)}
    main8 = dict(zip((32,96,256,1024,4096),(841.973,874.752,928.764,1260.373,4253.838)))
    rows=[]
    for ep in (8,32,128):
        for bs in (32,96,256,1024,4096):
            if not all((OUT/f'ep{ep}-perf{n}-bs{bs}-profiling.json').exists() for n in (1,2)):
                assert args.allow_pending
                rows.append(dict(ep=ep,bs=bs,run1=None,run2=None,mean_us=None,max_rank_us=None,spread=None,
                    baseline_us=read(W/f'outputs/ep128-rack05-07-20261011/bs-{bs}.json')['mean_us'] if ep==128 else None,
                    ds_us=read(W/f'outputs/ep128-deepseek-20261011/bs-{bs}.json')['mean_us'] if ep==128 else None,
                    gain_main=None,gain_ds=None,accuracy='用户取消精度；性能启动失败，待补测',reference_date='main/DS 10-11 historical',
                    source='pending',timer='pending'))
                continue
            reports=[read(OUT/f'ep{ep}-perf{n}-bs{bs}-profiling.json') for n in (1,2)]
            waived=any(s['ep']==ep and s['stage']=='accuracy' and s['status']=='cancelled_by_user' for s in state['stages'])
            if waived:
                accuracy_label='用户取消精度测试；本次未验证'
            else:
                accuracy=read(OUT/f'ep{ep}-accuracy-bs{bs}-accuracy.json')
                assert len(accuracy)==ep and all(r['passed'] and r['counts_equal'] for r in accuracy)
                accuracy_label=f'{ep}/{ep} ranks PASS, 16 tokens/rank'
            for r in reports:
                assert r['aggregation']=='mean_of_per_round_min_rank' and r['rounds']==10
                assert len(r['per_rank_rounds_us'])==ep and r['measurement']=='full_moe'
                assert all(len(rank)==10 for rank in r['per_rank_rounds_us'])
                selected=list(zip(*r['per_rank_rounds_us']))
                assert abs(mean(min(v) for v in selected)-r['mean_us'])<1e-6
                assert abs(mean(max(v) for v in selected)-r['rank_max_mean_us'])<1e-6
                assert abs(mean(r['rank_min_rounds_us'])-r['mean_us'])<1e-6
            baseline=main8[bs] if ep==8 else main32[bs] if ep==32 else read(W/f'outputs/ep128-rack05-07-20261011/bs-{bs}.json')['mean_us']
            ds=ds8[bs] if ep==8 else read(W/f'outputs/megamoe-three-versions-20261011/ep32-deepseek-bs-{bs}.json')['mean_us'] if ep==32 else read(W/f'outputs/ep128-deepseek-20261011/bs-{bs}.json')['mean_us']
            values=[r['mean_us'] for r in reports]
            rows.append(dict(ep=ep,bs=bs,run1=values[0],run2=values[1],mean_us=mean(values),
                max_rank_us=mean(r['rank_max_mean_us'] for r in reports),spread=max(values)-min(values),
                baseline_us=baseline,ds_us=ds,gain_main=1-mean(values)/baseline,gain_ds=1-mean(values)/ds,
                accuracy=accuracy_label,
                reference_date='main 10-12 screen; DS 10-12 3 pairs' if ep==8 else 'main/DS 10-11 historical',
                source=f'ep{ep}-perf[1,2]-bs{bs}-profiling.json',
                timer=f"{state['remote_root']}/ep{ep}-perf1/bs_{bs}/timer"))
    data=dict(plan=state['plan'],rows=rows,optimizations=read(HERE/'optimizations.json'),
              scope='2026-10-12; full communication; two timer-off repeats; historical comparisons not fresh paired ablations',
              campaign_sha256=hashlib.sha256((OUT/'campaign.json').read_bytes()).hexdigest(),
              finished=state.get('finished'),status=state['status'],pending_reason=state.get('error'),
              completed_cases=sum(r['mean_us'] is not None for r in rows))
    (OUT/'report-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'unified-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
