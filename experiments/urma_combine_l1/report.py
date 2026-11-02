"""Same-rank completion envelopes; never compare absolute clocks across ranks."""
import csv
import json
import statistics as st
from pathlib import Path


def analyze(output, devices, iters, group_size=4, ticks_per_us=1000, l1_enable=False):
    rank_rounds=[]
    for rank, device in enumerate(devices):
        buckets={i:[] for i in range(iters)}
        with (Path(output)/f'rank{rank}.csv').open() as stream:
            for raw in csv.DictReader(stream):
                row={k:(v if k=='kind' else int(v)) for k,v in raw.items()}
                if row['iteration'] not in buckets: raise ValueError('Unexpected iteration')
                if row['end_tick']<=row['start_tick'] or row['bytes']<=0: raise ValueError('Invalid interval')
                buckets[row['iteration']].append(row)
        rounds=[]
        for iteration, rows in buckets.items():
            comm=[r for r in rows if r['kind']=='combine']
            loads=[r for r in rows if r['kind']=='l1']
            if sorted(r['peer'] for r in comm)!=[p for p in range(len(devices)) if p!=rank]:
                raise ValueError('Missing/duplicate peer record')
            if sorted(r['worker'] for r in loads)!=(list(range(28)) if l1_enable else []):
                raise ValueError('Missing/duplicate AIC record')
            entry={}
            for name, selected in [('TOTAL',comm),
                ('INTRA',[r for r in comm if devices[r['peer']]//group_size==device//group_size]),
                ('CROSS',[r for r in comm if devices[r['peer']]//group_size!=device//group_size]),('L1',loads)]:
                if not selected:entry[name]=None;continue
                lo=min(r['start_tick'] for r in selected);hi=max(r['end_tick'] for r in selected)
                entry[name]={'bytes':sum(r['bytes'] for r in selected),'us':(hi-lo)/ticks_per_us}
            lo=min(r['start_tick'] for r in comm);hi=max(r['end_tick'] for r in comm)
            coverage=sum(max(0,min(hi,r['end_tick'])-max(lo,r['start_tick'])) for r in loads)/(28*(hi-lo))
            entry['aic_active_coverage']=coverage
            rounds.append(entry)
        rank_rounds.append(rounds)
    result={'classes':{},'rank_rounds':rank_rounds}
    for kind in ('TOTAL','INTRA','CROSS','L1'):
        ranks=[(rank,[r[kind] for r in rounds]) for rank,rounds in enumerate(rank_rounds) if rounds[0][kind]]
        if not ranks:result['classes'][kind]=None;continue
        gbps=[st.mean(r['bytes'] for r in rows)/st.mean(r['us'] for r in rows)/1000 for _,rows in ranks]
        result['classes'][kind]={
            'mean_gbps':st.mean(gbps),'rank_min_gbps':min(gbps),'rank_max_gbps':max(gbps),
            'round_max_mean_us':st.mean(max(rows[i]['us'] for _,rows in ranks) for i in range(iters)),
            'round_min_mean_us':st.mean(min(rows[i]['us'] for _,rows in ranks) for i in range(iters)),
            'mean_bytes_per_card':st.mean(r['bytes'] for _,rows in ranks for r in rows)}
    result['aic_active_coverage_mean']=st.mean(r['aic_active_coverage'] for rounds in rank_rounds for r in rounds)
    result['denominator']='sum(payload bytes)/rounds divided by mean same-rank max(end)-min(start); one direction'
    return result


def display(result,label):
    print(f'\n[{label}] COMPLETE Combine includes final quiet; no per-CV timer')
    for kind, row in result['classes'].items():
        if row is None:print(f'{kind}: N/A');continue
        print(f"{kind}: {row['mean_gbps']:.3f} GB/s/card rank-min/max="
              f"{row['rank_min_gbps']:.3f}/{row['rank_max_gbps']:.3f}; "
              f"round-max-mean={row['round_max_mean_us']:.3f}us "
              f"round-min-mean={row['round_min_mean_us']:.3f}us")
    print(f"AIC active coverage of Combine: {result['aic_active_coverage_mean']*100:.1f}% "
          '(mean 28-core interval overlap / Combine envelope)')
