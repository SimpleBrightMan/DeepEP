"""Data-only communication completion plus coupled GMM/dequant intervals."""
import csv
import math
import statistics as st
from pathlib import Path


def analyze(output,devices,iters,group_size=4,ticks_per_us=1000,compute_enable=False,*,communication_enable=True,model=None,peak_tflops=756.0):
    if not devices or iters<1 or not math.isfinite(ticks_per_us) or ticks_per_us<=0:
        raise ValueError('Devices/iterations/tick frequency must be valid')
    if not compute_enable and not communication_enable:raise ValueError('No active workload')
    if not math.isfinite(peak_tflops) or peak_tflops<=0:raise ValueError('Peak TFLOPS must be positive and finite')
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
                if sorted(r['peer'] for r in kinds[kind])!=([p for p in range(len(devices)) if p!=rank] if communication_enable else []):
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
                if not comm:
                    entry['coverage_'+name]=None;continue
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
        values=[r['coverage_'+name] for rounds in rank_rounds for r in rounds if r['coverage_'+name] is not None]
        result['gmm_coverage_'+name]=st.mean(values) if values else None
    result['mfu']=calculate_mfu(result,model,peak_tflops) if compute_enable and model is not None else None
    result['communication_enabled']=communication_enable
    return result


def calculate_mfu(result,model,peak_tflops):
    """Balanced per-card effective FLOPs / per-round rank-max AIC envelope."""
    for key in ('bs','hidden','intermediate','topk'):
        if type(model.get(key)) is not int or model[key]<=0:raise ValueError('Invalid MFU model '+key)
    shared=model.get('shared_experts',0)
    si=model.get('shared_intermediate',model['intermediate'])
    if type(shared) is not int or shared not in (0,1) or type(si) is not int or si<=0:
        raise ValueError('Invalid shared MFU model')
    # Complete balanced traffic: receive rows per card = BS * TopK, not /EP.
    base=model['bs']*model['hidden']*(model['topk']*model['intermediate']+shared*si)
    gmm1=4*base;gmm2=2*base;flops=gmm1+gmm2
    aic=result['classes']['GMM']
    if not aic or aic['round_max_mean_us']<=0:raise ValueError('Missing AIC envelope')
    rank_means=[]
    for rounds in result['rank_rounds']:
        rank_means.append(st.mean(flops/(row['GMM']['us']*peak_tflops*1e6)*100 for row in rounds))
    return dict(percent=flops/(aic['round_max_mean_us']*peak_tflops*1e6)*100,
        flops_per_card=flops,gmm1_flops_per_card=gmm1,gmm2_flops_per_card=gmm2,peak_tflops=peak_tflops,
        aic_round_max_mean_us=aic['round_max_mean_us'],aic_round_min_mean_us=aic['round_min_mean_us'],
        rank_mean_percent=st.mean(rank_means),rank_min_percent=min(rank_means),rank_max_percent=max(rank_means),
        per_rank_percent=rank_means,model=dict(model),
        formula='6*BS*H*(TopK*I+S*Is)/(peak_TFLOPS*1e6*mean_round(max_rank(AIC_last_end-AIC_first_start)))*100',
        scope='Production comm-first GMM/dequant classes with inputs and GMM2 activations prefilled. Includes shared head/tail and intra-compute gaps; excludes Pack/SwiGLU/Reduce/Reorder and communication readiness. Not hardware Cube-only utilization.')


def display(result,label):
    print(f'\n[compute+dequant={label} communication={"on" if result.get("communication_enabled",True) else "off"}] prefilled independent GMM inputs')
    for kind,row in result['classes'].items():
        if row is None:print(f'{kind}: N/A');continue
        bw='' if row['mean_gbps'] is None else f"{row['mean_gbps']:.3f} GB/s/card rank-min/max={row['rank_min_gbps']:.3f}/{row['rank_max_gbps']:.3f}; "
        print(f"{kind}: {bw}round-max-mean={row['round_max_mean_us']:.3f}us round-min-mean={row['round_min_mean_us']:.3f}us")
    if result['mfu'] is not None:
        m=result['mfu']
        print(f"[MFU] {m['percent']:.3f}% = {m['flops_per_card']/1e12:.6f} TFLOP/card / {m['aic_round_max_mean_us']:.3f}us / {m['peak_tflops']:.3f} TFLOPS/card; AIC envelope, waits included")
        print(f"[MFU-RANKS] per-rank mean-of-round MFU min/mean/max={m['rank_min_percent']:.3f}/{m['rank_mean_percent']:.3f}/{m['rank_max_percent']:.3f}%")
    print('COMBINE_* includes any remaining Dispatch drain; DISPATCH_SUBMIT_* is issue time only, no completion bandwidth.')
    def coverage(value):return 'N/A' if value is None else f'{value*100:.1f}%'
    print(f"GMM active coverage: TOTAL={coverage(result['gmm_coverage_total'])} COMBINE={coverage(result['gmm_coverage_combine'])}")
