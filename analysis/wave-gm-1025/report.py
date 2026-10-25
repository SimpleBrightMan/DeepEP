"""汇总已审计的Wave+GM对照；不覆盖原完整矩阵，不将单点确认挑入最好成绩。"""
import argparse
import copy
import json
import runpy
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(root,ep,batches):
    summary=load(root/'summary.json');manifest=load(root/'experiment.json')
    if (summary['ep'],summary['status'],summary['execution_mode'],summary['buffer_capacity'])!=(ep,'audited_complete','continuous',4096):
        raise ValueError('Require audited complete continuous domain')
    expected=dict(control='4a0de8571b18d4dde6e55e03e895ea9f55cc0e1c',combined='095b4c7f088723fe52e5308abcd6f61477149ee1')
    if any(summary['packages'][m]['revision']!=r for m,r in expected.items()):raise ValueError('Wrong native identity')
    if [c['bs'] for c in summary['cases']]!=batches:raise ValueError('Missing BS')
    if manifest['schedule']!=['control-a','combined-a','combined-b','control-b']:raise ValueError('Not ABBA')
    if len(manifest['nodes'])!=ep//8:raise ValueError('Wrong physical node count')
    for case in summary['cases']:
        if not case['output_hashes_equal']:raise ValueError('Output difference')
        for method in case['methods'].values():
            if method['starts']!=2:raise ValueError('Missing repeat')
            for run in method['runs']:
                for rank in range(ep):
                    meta=load(root/f'node{rank//8}/runs/{run["label"]}/rank_{rank}/identity.json')
                    if meta['host_coordination']!='cpu_gloo':raise ValueError('Mixed host coordination')
    return summary


def phases(root,case,method):
    folder=root/f'node0/runs/bs{case}-{method}-a/rank_0/timer'
    events=load(folder/'native_events.json')
    start=min(e['start_cycle'] for e in events);end=max(e['end_cycle'] for e in events)
    first=min(e['start_cycle'] for e in events if e['name'].startswith('AIC_GMM1_ISSUE_'))
    last=max(e['end_cycle'] for e in events if e['name'].startswith('AIC_GMM2_ISSUE_'))
    return dict(head_us=(first-start)/1000,issue_window_us=(last-first)/1000,tail_us=(end-last)/1000,
                envelope_us=(end-start)/1000,json=str(folder/'timer_trace.json'))


def confirmation(root):
    """只审计成功退出的A轮；失败的B轮不能补成完整ABBA。"""
    manifest=load(root/'experiment.json')
    if (manifest['ep'],manifest['bs_values'],manifest['buffer_capacity'],manifest['status'])!=(32,[1024],4096,'failed'):
        raise ValueError('Unexpected confirmation domain/status')
    expected=['preflight','bs1024-control-a','bs1024-combined-a']
    if manifest['completed']!=expected:raise ValueError('Unexpected partial completion')
    revisions={'control':'4a0de8571b18d4dde6e55e03e895ea9f55cc0e1c',
               'combined':'095b4c7f088723fe52e5308abcd6f61477149ee1'}
    if any(manifest['packages'][m]['revision']!=r for m,r in revisions.items()):
        raise ValueError('Confirmation native identity mismatch')
    audit=runpy.run_path(str(Path(__file__).parents[1]/'five-rounds-1024/report.py'))['audit_run']
    records=[];fingerprints=None
    for label in expected[1:]:
        record,ins,outs=audit(root,label,32,manifest['packages'],True,'continuous',4096)
        if fingerprints is not None and fingerprints!=(ins,outs):raise ValueError('Confirmation fingerprint mismatch')
        fingerprints=(ins,outs);records.append(record)
    return dict(status='partial_ab_only',runs=records,output_hashes_equal=True,
                reduction_pct=100*(1-records[1]['mean_us']/records[0]['mean_us']),
                failure='Third start failed in CPU Gloo prepare before measured kernel; collective size mismatch 24 vs 4. Root cause unresolved.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ep8',type=Path,required=True);p.add_argument('--ep32',type=Path,required=True)
    p.add_argument('--confirm',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decision',required=True)
    a=p.parse_args();batches=[32,96,256,512,1024,2048,4096]
    full=[validate(a.ep8,8,batches),validate(a.ep32,32,batches)]
    confirm=confirmation(a.confirm)
    cases=[dict(copy.deepcopy(c),baseline_path='CV' if s['ep']==8 and c['bs']<=1024 else 'GM',candidate_path='GM')
           for s in full for c in s['cases']]
    phase_rows=[dict(ep=ep,bs=bs,**{m:phases(root,bs,m) for m in ('control','combined')})
                for ep,root in ((8,a.ep8),(32,a.ep32)) for bs in batches]
    records=[r for s in full for c in s['cases'] for m in c['methods'].values() for r in m['runs']]+confirm['runs']
    csv_count=sum(r['csv_count'] for r in records)
    trace_count=sum(bool(r['trace']) for r in records)
    if (csv_count,trace_count)!=(1808,30):raise ValueError('Evidence incomplete')
    max_overhead=max(r['overhead_pct'] for r in records if r['overhead_pct'] is not None)
    data=dict(status='audited_complete',branch='codex/megamoe-wave-gm-1025',revision='095b4c7f088723fe52e5308abcd6f61477149ee1',
        baseline_revision='4a0de8571b18d4dde6e55e03e895ea9f55cc0e1c',scope=full[0]['scope'],
        execution_mode='continuous',buffer_capacity=4096,host_coordination='cpu_gloo',accuracy='not_run_user_requested',
        cases=cases,confirmation=confirm,packages=full[0]['packages'],decision=a.decision,
        csv_count=csv_count,trace_count=trace_count,max_measured_timer_overhead_pct=max_overhead,
        caveats=['EP32 already GM in baseline; algorithm-level negative control, not identical machine code.',
                 'CPU Gloo only outside timed device kernels; Dispatch/Combine remains SHMEM/MTE.',
                 'ISSUE end is not hardware completion.','No independent Golden. EP64/128 untested.',
                 'Confirmation stays separate; do not replace full-matrix result with a selected best value.'])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'phase-comparison.json').write_text(json.dumps(phase_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Wave+GM 对照报告','',f'结论：{a.decision}','',
           '基线原生`4a0de857`；统一GM原生`095b4c7f`，分支`codex/megamoe-wave-gm-1025`。'
           '只关闭CV策略，不叠加提前归约，不改Wave预算、gate/up配对、lag或数学契约。', '',
           '完整通信、无共享专家、FP8/FP4、H7168/I3072/E384/TopK6、balanced。'
           'EP8单节点，EP32同柜四节点；每项同窗ABBA，各两次独立启动，连续七BS、capacity4096。'
           '主性能timer-off，warmup10/iters20取后10次逐轮rank-min均值，同时保留rank-max与每次启动。'
           '32rank/8rank输出指纹与重复结果一致，未跑独立Golden。'
           '隔离测试主机控制采用CPU Gloo，不建立HCCL；真正设备通信仍为SHMEM/MTE。','',
           '| EP | BS/卡 | 原路径 | 原版 μs | 统一GM μs | 降耗 |', '|---:|---:|---|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['ep']} | {c['bs']} | {c['baseline_path']} | {c['methods']['control']['mean_us']:.2f} | "
                     f"{c['methods']['combined']['mean_us']:.2f} | {c['reduction_pct']:+.2f}% |")
    lines+=['','## EP32 BS1024 确认试验','',
            f"仅完成一对AB、capacity仍为4096：原版{confirm['runs'][0]['mean_us']:.2f}μs，"
            f"统一GM {confirm['runs'][1]['mean_us']:.2f}μs，降耗{confirm['reduction_pct']:+.2f}%。"
            '第三次启动在CPU Gloo prepare阶段出现24/4字节collective尺寸不匹配，尚未进入性能kernel，根因未确定；后续停止。'
            '因此这不是完整ABBA，不能替代上方已完成的七BS矩阵，也不能宣布消除了BS1024回退风险。','',
            '## 如何理解','',
            '1. 真正的CV→GM切换仅为EP8 BS≤1024；EP8大BS、EP32原本已是GM。后者若变化，先查波动/编译产物，不归因于多了一次GM中转。',
            '2. 该消融复用整个现有GM后端：GM中转、64行激活量子、6个Dispatch/2个Combine槽；CV为32行量子、2/4槽，激活UB区域分别为144KiB和64KiB（不是整个UB总容量）。不是只改一条写回。',
            '3. CV减少GMM1结果GM读写，但占更多UB、改变credit与调度粒度；GM中转也可能被计算/供数重叠。不能只凭通路名称判断性能。',
            '4. 单一策略常量可消除运行时判断，源代码算法路径一致不代表机器码一致。未证明带宽bound或某个环节的唯一因果。',
            '5. 这版保留CV实现作消融，并没有清理全部CV死代码。若选GM作维护基线，清理需另做等价回归；EP64/128尚未验证，不外推。','',
            '## rank0 单次流水（a轮最后一次timer-on）','',
            '头部 / 首次GMM1到最后GMM2 ISSUE / 尾部，单位μs；ISSUE不是硬件完成，不能与正式timer-off均值混算。'
            '只保存了最后一次流水，个别调用有尾段长样本；不能用该样本的尾段差值解释多轮均值差距。','',
            '| EP / BS | 原版三段 | GM三段 |','|---|---|---|']
    for row in phase_rows:
        if row['bs'] not in (256,512,1024,4096):continue
        text=lambda v:' + '.join(f'{v[k]:.2f}' for k in ('head_us','issue_window_us','tail_us'))
        lines.append(f"| {row['ep']} / {row['bs']} | {text(row['control'])} | {text(row['combined'])} |")
    lines+=['',f'本地证据：{csv_count}份配对kernel_details.csv，{trace_count}份rank0完整被动流水（含确认试验）。'
            f'本次观测timer-on最大正向开销{max_overhead:.2f}%；负差值按波动处理，不解释为打点加速。'
            'timer-on/off每个shape单独记录，不预设开销永远低于某个比例。','']
    public='\n'.join(lines)
    (a.output/'PUBLIC_RESULTS.md').write_text(public,encoding='utf-8')
    local=[public,'## MindStudio 路径（每行一个反斜杠版本）','']
    for ep,root in ((8,a.ep8),(32,a.ep32)):
        for method in ('control','combined'):
            path=root/f'node0/runs/bs512-{method}-a/rank_0/timer/timer_trace.json'
            local += [f'EP{ep} BS512 {method}：','```text',str(path).replace('/','\\'),'```','']
    (a.output/'RESULTS.md').write_text('\n'.join(local),encoding='utf-8')
    print(public)


if __name__=='__main__':main()
