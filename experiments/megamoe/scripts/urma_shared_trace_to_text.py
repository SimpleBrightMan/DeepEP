#!/usr/bin/env python3
"""Compact shared URMA single-rank/allrank JSON; standard library, no Torch/NPU."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import urma_trace_to_text as base

STAGES = {
    "Dsubmit": {1, 2, 4, 48, 53}, "Csubmit": {10, 30, 49, 61},
    "Dcomplete": {43}, "Ccomplete": {42},
    "routedGMM": {15, 16, 19, 21, 27, 33, 34, 35, 36},
    "sharedGMM1": {55, 56}, "sharedGMM2": {58, 59},
    "allGMM": {15, 16, 19, 21, 27, 33, 34, 35, 36, 46, 55, 56, 58, 59},
    "dequant": {17, 20, 37, 38, 47, 55, 56, 58, 59},
    "pack": {0, 44, 54}, "activation": {23, 50, 57},
    "reduce": {25, 32, 51}, "sharedAdd": {60}, "sync": {24, 26, 45, 52},
}


def split_ranks(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        raise ValueError("Expected Chrome trace JSON, not summary.json")
    meta=doc.get("metadata", {})
    per_rank={m["rank"]: m for m in meta.get("per_rank", [])}
    grouped=defaultdict(list)
    for e in doc["traceEvents"]:
        if e.get("ph")=="X": grouped[e.get("pid")].append(e)
    if not grouped: raise ValueError("Empty trace")
    if len(grouped)>1 and not per_rank: raise ValueError("allrank needs metadata.per_rank; cannot invent rank-local spans")
    result={}
    for rank,rows in grouped.items():
        m=per_rank.get(rank, meta if len(grouped)==1 else None)
        if m is None: raise ValueError(f"Missing rank {rank} metadata")
        result[rank]=base.read_events(dict(metadata=m,traceEvents=rows))
    return result


def render(doc, *, source="trace.json", ranks="slowest", detail=False, cores=None, digits=2, top=5):
    data=split_ranks(doc)
    def fmt(x): return "-" if x is None else f"{x:.{digits}f}"
    def span(rank):
        meta,rows,_,_=data[rank]
        if meta.get("envelope_us") is not None:return base.number(meta["envelope_us"],"envelope_us")
        return base.stats(rows)["span"]
    def interval(rows):
        if not rows:return "-"
        s=base.stats(rows);return f"{fmt(s['first'])}:{fmt(s['last'])}({fmt(s['span'])})"
    def stage(rows,name):
        out=[e for e in rows if e['id'] in STAGES[name]]
        if name in ('allGMM','routedGMM','sharedGMM1','sharedGMM2'):out=[e for e in out if e['tid']%3==0]
        if name=='dequant':out=[e for e in out if e['tid']%3==1]
        return out
    if ranks=="slowest":selected=[max(sorted(data),key=span)]
    elif ranks=="all":selected=sorted(data)
    else:
        try:selected=sorted(set(map(int,ranks.split(','))))
        except ValueError:raise ValueError("--ranks must be slowest / all / 0,3,...")
        if not selected or any(r not in data for r in selected):raise ValueError("Requested rank missing")
    lines=[f"URMA_SHARED_TEXT_V1 source={json.dumps(str(source),ensure_ascii=False)}",
           f"ranks={','.join(map(str,sorted(data)))} selected={','.join(map(str,selected))} detail={int(detail)} digits={digits}",
           "NOTE us; rank clocks independent; preserve gaps; keep ALL source waits, quiet, drain and final sync.",
           "NOTE Dsubmit/Csubmit are SUBMISSION windows, not network service time. No ready-time inference from missing events.",
           "NOTE coarse/fine are separate invocations: do not align their timestamps or treat differences as one run.",
           "NOTE source-hidden waits cannot be recovered; coarse cannot resolve per-CV readiness. Rounded values only for display.",
           "[RANK] rank diagnostic_us source_status instrumentation source_events hidden_waits dropped"]
    for rank in sorted(data):
        m,rows,_,status=data[rank]
        lines.append(f"{rank} {fmt(span(rank))} {status} {m.get('instrumentation','?')} {len(rows)} {m.get('hidden_wait_count',0)} {m.get('dropped',0)}")
        if m.get('model') is not None:
            lines.append(f"MODEL rank={rank} bs={m.get('bs')} ep={m.get('world')} "+json.dumps(m['model'],ensure_ascii=False,separators=(',',':')))
        lines.append("BOUND rank="+str(rank)+" "+" ".join(f"{key}={interval(stage(rows,key))}" for key in STAGES))
    declared=doc.get('metadata',{}).get('rank_count')
    if declared is not None and declared!=len(data):lines.append(f"WARNING declared_rank_count={declared} actual={len(data)}")
    for rank in selected:
        meta,original,definitions,status=data[rank]
        rows=[e for e in original if cores is None or e['tid']//3 in cores]
        if not rows:raise ValueError("No events in selected cores")
        lines.append(f"[RANK_DETAIL] rank={rank} source_status={status} selected_events={len(rows)}")
        lines.append("[LEGEND] id name category")
        for id in sorted({e['id'] for e in rows}):lines.append(f"{id} {' '.join(definitions[id])}")
        lines.append("[FIELDS] p=peer c=CV f=fragment e=expert r=row l=rows m=tile_m n=tile_n w=columns s=sync_stage b=bytes")
        lanes=defaultdict(list);peers=defaultdict(list)
        for e in rows:
            lanes[e['tid']].append(e)
            if e['args'].get('peer') is not None:peers[e['args']['peer']].append(e)
        lines.append("[LANE] block/engine first:last(span) union_us gaps_us; op=id:count/sum_us")
        for tid,lane in sorted(lanes.items()):
            ss=base.stats(lane);ops=defaultdict(list)
            for e in lane:ops[e['id']].append(e)
            lines.append(f"{base.engine_label(tid)} {interval(lane)} union={fmt(ss['union'])} gaps={fmt(max(0,ss['span']-ss['union']))} "+
                         ' '.join(f"{id}:{len(es)}/{fmt(sum(e['dur'] for e in es))}" for id,es in sorted(ops.items())))
        lines.append("[PEER] p stage put_count payload_B first:last(span); longest gaps prev_end->next_start:us,next_cv")
        for peer,events in sorted(peers.items()):
            for name,ids in [('DX',{1}),('DS',{2}),('CY',{10})]:
                puts=sorted([e for e in events if e['id'] in ids],key=lambda e:e['ts'])
                if not puts:continue
                gaps=[(max(0,b['ts']-a['ts']-a['dur']),a['ts']+a['dur'],b['ts'],b['args'].get('cv')) for a,b in zip(puts,puts[1:])]
                lines.append(f"{peer} {name} {len(puts)} {sum(e['args'].get('bytes',0) or 0 for e in puts)} {interval(puts)} "+
                             ' '.join(f"{fmt(a)}->{fmt(b)}:{fmt(g)},c{cv}" for g,a,b,cv in sorted(gaps,reverse=True)[:top]))
        if detail:
            lines.append("[EVENTS] grouped by lane; start_us duration_us id fields (waits kept)")
            for tid,lane in sorted(lanes.items()):
                lines.append('@ '+base.engine_label(tid))
                for e in sorted(lane,key=lambda x:x['ts']):
                    lines.append(f"{fmt(e['ts'])} {fmt(e['dur'])} {e['id']} {base.short_args(e)}".rstrip())
                    sources=e['args'].get('source_records')
                    if sources:
                        clock=meta.get('clock_mhz')
                        origin=e['args'].get('start_tick')
                        if clock and origin is not None:
                            parts=[f"{part['event_id']}:{fmt(e['ts']+(int(part['start_tick'])-int(origin))/clock)}:{fmt((int(part['end_tick'])-int(part['start_tick']))/clock)}" for part in sources]
                            lines.append(' src(id:start:dur)='+' '.join(parts))
                        else:lines.append(' src_ticks='+json.dumps(sources,separators=(',',':')))
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('json',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--ranks',default='slowest',help='slowest (diagnostic span), all, or comma-separated ranks')
    p.add_argument('--detail',action='store_true',help='Append all selected events; waits are always kept')
    p.add_argument('--cores',type=base.parse_cores,help='Optional 0,1,15-17; all lanes by default')
    p.add_argument('--digits',type=int,choices=range(0,7),default=2)
    p.add_argument('--top',type=int,default=5)
    a=p.parse_args();out=a.output or a.json.with_suffix('.shared.txt')
    try:
        if a.top<0:raise ValueError('--top must be nonnegative')
        if out.resolve()==a.json.resolve() or (out.exists() and out.samefile(a.json)):raise ValueError('Cannot overwrite input JSON')
        text=render(json.loads(a.json.read_text(encoding='utf-8-sig')),source=a.json,ranks=a.ranks,detail=a.detail,cores=a.cores,digits=a.digits,top=a.top)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
    except (OSError,ValueError,KeyError,TypeError) as error:p.exit(2,f'Export failed: {error}\n')
    print(f'{out} ({len(text.encode("utf-8")):,} bytes)')


if __name__=='__main__':main()
