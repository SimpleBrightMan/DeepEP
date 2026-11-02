"""Data-only communication completion plus coupled GMM/dequant intervals."""
import csv
import statistics as st
from pathlib import Path


def analyze(output,devices,iters,group_size=4,ticks_per_us=1000,compute_enable=False):
    rank_rounds=[]
    for rank,device in enumerate(devices):
        buckets={i:[] for i in range(iters)}
        with (Path(output)/f'rank{rank}.csv').open() as stream:
            for raw in csv.DictReader(stream):
                row={k:(v if k=='kind' else int(v)) for k,v in raw.items()}
                if row['iteration'] not in buckets or row['end_tick']<=row['start_tick'] or row['bytes']<0:
                    raise ValueError('Invalid interval/iteration')
                buckets[row['iteration']].append(row)
        rounds=[]
        for rows in buckets.values():
            kinds={k:[r for r in rows if r['kind']==k] for k in ('total','dispatch_submit','combine_shared_tail','gmm','dequant')}
            if sum(map(len,kinds.values()))!=len(rows):raise ValueError('Unknown record kind')
            for kind in ('total','dispatch_submit','combine_shared_tail'):
                if sorted(r['peer'] for r in kinds[kind])!=[p for p in range(len(devices)) if p!=rank]:
                    raise ValueError('Missing/duplicate peer record')
            for kind,first in [('gmm',0),('dequant',28)]:
                if sorted(r['worker'] for r in kinds[kind])!=(list(range(first,first+28)) if compute_enable else []):
                    raise ValueError('Missing/duplicate compute record')
            entry={}
            selections=[('GMM',kinds['gmm']),('DEQUANT',kinds['dequant'])]
            for prefix,kind in [('', 'total'),('COMBINE_', 'combine_shared_tail'),('DISPATCH_SUBMIT_', 'dispatch_submit')]:
                for cohort in ('TOTAL','INTRA','CROSS'):
                    selected=[r for r in kinds[kind] if cohort=='TOTAL' or
                        ((devices[r['peer']]//group_size==device//group_size)==(cohort=='INTRA'))]
                    selections.append((prefix+cohort,selected))
            for name,selected in selections:
                if not selected:entry[name]=None;continue
                lo=min(r['start_tick'] for r in selected);hi=max(r['end_tick'] for r in selected)
                entry[name]={'bytes':sum(r['bytes'] for r in selected),'us':(hi-lo)/ticks_per_us}
            for name,comm in [('total',kinds['total']),('combine',kinds['combine_shared_tail'])]:
                lo=min(r['start_tick'] for r in comm);hi=max(r['end_tick'] for r in comm)
                entry['coverage_'+name]=sum(max(0,min(hi,r['end_tick'])-max(lo,r['start_tick'])) for r in kinds['gmm'])/(28*(hi-lo))
            rounds.append(entry)
        rank_rounds.append(rounds)
    result={'classes':{},'rank_rounds':rank_rounds,
        'scope':'TOTAL is completed X+scale+Y; COMBINE shares final quiet with in-flight Dispatch; Dispatch submission is not completion'}
    for kind in rank_rounds[0][0]:
        if kind.startswith('coverage_'):continue
        ranks=[(r,[x[kind] for x in rounds]) for r,rounds in enumerate(rank_rounds) if rounds[0][kind]]
        if not ranks:result['classes'][kind]=None;continue
        completed_bytes=kind not in ('GMM','DEQUANT') and not kind.startswith('DISPATCH_SUBMIT_')
        bw=[st.mean(x['bytes'] for x in rows)/st.mean(x['us'] for x in rows)/1000 for _,rows in ranks]
        result['classes'][kind]={'mean_gbps':st.mean(bw) if completed_bytes else None,
            'rank_min_gbps':min(bw) if completed_bytes else None,'rank_max_gbps':max(bw) if completed_bytes else None,
            'round_max_mean_us':st.mean(max(rows[i]['us'] for _,rows in ranks) for i in range(iters)),
            'round_min_mean_us':st.mean(min(rows[i]['us'] for _,rows in ranks) for i in range(iters)),
            'mean_bytes_per_card':st.mean(x['bytes'] for _,rows in ranks for x in rows)}
    for name in ('total','combine'):
        result['gmm_coverage_'+name]=st.mean(r['coverage_'+name] for rounds in rank_rounds for r in rounds)
    return result


def display(result,label):
    print(f'\n[compute+dequant={label}] data-only; final quiet covers Dispatch+Combine')
    for kind,row in result['classes'].items():
        if row is None:print(f'{kind}: N/A');continue
        bw='' if row['mean_gbps'] is None else f"{row['mean_gbps']:.3f} GB/s/card rank-min/max={row['rank_min_gbps']:.3f}/{row['rank_max_gbps']:.3f}; "
        print(f"{kind}: {bw}round-max-mean={row['round_max_mean_us']:.3f}us round-min-mean={row['round_min_mean_us']:.3f}us")
    print('COMBINE_* includes any remaining Dispatch drain; DISPATCH_SUBMIT_* is issue time only, no completion bandwidth.')
    print(f"GMM active coverage: TOTAL={result['gmm_coverage_total']*100:.1f}% COMBINE={result['gmm_coverage_combine']*100:.1f}%")
