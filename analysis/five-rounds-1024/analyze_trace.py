"""按实际原生周期对比rank0完整被动流水；不将ISSUE或等待跨度冒充硬件利用率。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy
import statistics
import sys


def union_length(intervals):
    total=0;left=right=None
    for start,end in sorted(intervals):
        if right is None:left,right=start,end
        elif start>right:total+=right-left;left,right=start,end
        else:right=max(right,end)
    return total+(right-left if right is not None else 0)


def analyze(events,cycles_per_us):
    family=lambda e:e['name'].rsplit('_',1)[0] if e['name'].rsplit('_',1)[-1].isdigit() else e['name']
    by=defaultdict(list)
    for event in events:by[family(event)].append(event)
    kernels=by['KERNEL_TIMING']
    base=min(e['start_cycle'] for e in kernels);end=max(e['end_cycle'] for e in kernels)
    metrics=dict(native_kernel_envelope_us=(end-base)/cycles_per_us)
    windows=[];outside=[]
    for group in range(32):
        issue=[e for f in ('AIC_GMM1_ISSUE','AIC_GMM2_ISSUE') for e in by[f] if e['group_id']==group]
        if not issue:raise ValueError('Incomplete AIC group')
        span=max(e['end_cycle'] for e in issue)-min(e['start_cycle'] for e in issue)
        windows.append(span/cycles_per_us)
        outside.append((span-union_length([(e['start_cycle'],e['end_cycle']) for e in issue]))/cycles_per_us)
    metrics['aic_issue_window_mean_us']=statistics.mean(windows)
    metrics['aic_outside_issue_mean_us']=statistics.mean(outside)
    for f in ('AIC_GMM1_ISSUE','AIC_GMM2_ISSUE','AIC_GMM1_WAIT','AIC_GMM2_WAIT',
              'AIV_COMBINE_ISSUE','AIV_SWIGLU_ISSUE','AIV_DISPATCH_ISSUE'):
        group=by[f]
        durations=[(e['end_cycle']-e['start_cycle'])/cycles_per_us for e in group]
        metrics[f]=dict(count=len(group),sum_per_lane_us=sum(durations)/32,
            max_us=max(durations),last_end_relative_us=(max(e['end_cycle'] for e in group)-base)/cycles_per_us)
    metrics['combine_total_owned_rows']=sum(e['task']['row_count'] for e in by['AIV_COMBINE_ISSUE'])
    metrics['combine_max_callback_rows']=max(e['task']['row_count'] for e in by['AIV_COMBINE_ISSUE'])
    return metrics


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    summary=json.loads((a.root/'summary.json').read_text())
    if summary['status']!='audited_complete':raise ValueError('Unaudited trace experiment')
    protocol=runpy.run_path(str(Path(__file__).resolve().parents[2]/'experiments/megamoe/python/deep_ep_experimental/megamoe/timer/protocol.py'))
    result=dict(scope='rank0、全部32group；最后一次timer-on调用，不是20次性能均值',
        cycles_per_us=protocol['CYCLE_TO_US'],methods={})
    for method in ('control','combined'):
        folder=a.root/f'node0/runs/bs4096-{method}-a/rank_0/timer'
        meta=json.loads((folder/'metadata.json').read_text())
        events=json.loads((folder/'native_events.json').read_text())
        if meta['dynamic_groups']!=32 or meta['dynamic_aiv0']!=1 or len(events)!=meta['event_count']:
            raise ValueError('Wrong trace coverage')
        result['methods'][method]=analyze(events,protocol['CYCLE_TO_US'])
    control,candidate=(result['methods'][m] for m in ('control','combined'))
    if control['combine_total_owned_rows']!=candidate['combine_total_owned_rows'] or any(
        control[f]['count']!=candidate[f]['count'] for f in ('AIC_GMM1_ISSUE','AIC_GMM2_ISSUE')):
        raise ValueError('Trace work differs')
    rows=[('原生kernel包络',lambda x:x['native_kernel_envelope_us']),
          ('AIC提交窗口/核均值',lambda x:x['aic_issue_window_mean_us']),
          ('AIC窗口中ISSUE之外/核均值',lambda x:x['aic_outside_issue_mean_us']),
          ('GMM1 ISSUE累计/核均值',lambda x:x['AIC_GMM1_ISSUE']['sum_per_lane_us']),
          ('GMM2 ISSUE累计/核均值',lambda x:x['AIC_GMM2_ISSUE']['sum_per_lane_us']),
          ('GMM2 WAIT累计/核均值',lambda x:x['AIC_GMM2_WAIT']['sum_per_lane_us']),
          ('单次Combine ISSUE最长',lambda x:x['AIV_COMBINE_ISSUE']['max_us']),
          ('Dispatch最后提交结束相对起点',lambda x:x['AIV_DISPATCH_ISSUE']['last_end_relative_us'])]
    lines=['# R5被动流水诊断','',result['scope']+'。时间单位μs；每份JSON均有同启动同rank的off/on kernel_details.csv。',
        '', '|观测|冻结基线|4行配额|基线−候选|','|---|---:|---:|---:|']
    for title,get in rows:
        left,right=get(control),get(candidate)
        lines.append(f'|{title}|{left:.3f}|{right:.3f}|{left-right:+.3f}|')
    lines+=['',f'两版GMM1/GMM2提交各{control["AIC_GMM1_ISSUE"]["count"]}/{control["AIC_GMM2_ISSUE"]["count"]}次，Combine均发送{control["combine_total_owned_rows"]}行；原单次服务最多{control["combine_max_callback_rows"]}行，候选最多{candidate["combine_max_callback_rows"]}行。',
        '', '解释：GMM累计提交区间没有变短，而提交窗口中的区间外时间减少；最长Combine占用显著缩短，Dispatch更早推进完成。与“有界服务改善AIV1轮转、减少供数迟延/暴露等待”相容，不支持把收益解释为GEMM算术本身加速。',
        '', '限制：ISSUE不是硬件完成，区间之外不全是空泡；WAIT及各stage可重叠，不能相加还原端到端。最后一次rank0流水不是全rank逐轮均值；头部同步、其他rank及运行波动也影响两图差异。没有新增PMU，不能由此宣称MTE带宽提高。', '']
    (a.root/'trace_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.root/'TRACE_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:16]))


if __name__=='__main__':main()
