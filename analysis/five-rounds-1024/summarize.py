"""汇总独立对照，不跨节点拼接最佳值；保留缺测和失败。"""
import argparse
import json
from pathlib import Path

ROUNDS = {
    'r1': ('Combine一行配额', 'codex/megamoe-r1-combine-one-row-1024'),
    'r2': ('先提交激活再收FIX完成', 'codex/megamoe-r2-activation-before-fix-1024'),
    'r3': ('一轮AIC wave', 'codex/megamoe-r3-one-round-wave-1024'),
    'r4': ('三轮AIC wave', 'codex/megamoe-r4-three-round-wave-1024'),
    'r5': ('Combine四行配额', 'codex/megamoe-r5-combine-four-rows-1024'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args()
    result=dict(baseline_revision='904131cfa5c22e27eb8e278ec9019ab4dbb0392a',
        scope='完整通信kernel Duration；timer-off；warm10/20采后10；逐轮rank-min均值，同时保留rank-max',
        accuracy='not_run_user_requested',experiments=[],attempts=[],scans=[],
        ep128=dict(status='pending',note='尚未找到16节点同时空闲且软件一致的窗口，不继承历史性能'))
    for path in sorted(a.root.glob('scan-*.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        result['scans'].append(dict(file=path.name,time_utc=rows[0]['time_utc'],
            available_nodes=[r['rack']+'/'+r['node'] for r in rows if r['available']],
            authentication_failures=sum('Authentication' in r.get('error','') for r in rows)))
    cases={}
    for folder in sorted(a.root.iterdir()):
        manifest=folder/'experiment.json'
        if not folder.is_dir() or not manifest.is_file():continue
        m=json.loads(manifest.read_text())
        tag=folder.name.split('-')[0]
        if tag not in ROUNDS:raise ValueError('Unknown experiment '+folder.name)
        result['attempts'].append(dict(round=tag,ep=m['ep'],status=m['status'],folder=folder.name,
            finished_workers=[r['label'] for r in m['finished_workers']],completed=m['completed']))
        if not (folder/'summary.json').is_file():continue
        summary=json.loads((folder/'summary.json').read_text())
        if summary['status']!='audited_complete':continue
        if summary['packages']['control']['revision']!=result['baseline_revision']:
            raise ValueError('Wrong baseline '+folder.name)
        for case in summary['cases']:
            key=(tag,case['ep'],case['bs'])
            if key in cases:raise ValueError('Duplicate independent campaign; retain both, do not cherry-pick '+str(key))
            cases[key]=dict(round=tag,ep=case['ep'],bs=case['bs'],
                control=case['methods']['control'],candidate=case['methods']['combined'],
                reduction_pct=case['reduction_pct'],output_hashes_equal=case['output_hashes_equal'],
                source=folder.name+'/summary.json',placement=summary['placement'],
                revision=summary['packages']['combined']['revision'])
    for tag,(title,branch) in ROUNDS.items():
        result['experiments'].append(dict(id=tag,title=title,branch=branch,
            cases=[v for k,v in sorted(cases.items()) if k[0]==tag]))
    lines=['# 五轮优化实验（10月24日）','',
        '冻结对照为当前两轮wave＋AIV1完成等待 `904131c`。每轮独立对照，不把不同机器或不同分支的最小值拼成一个版本。',
        '',result['scope']+'；无共享专家，balanced seed20260922，H7168/I3072/E384/topk6。用户要求跳过独立Golden，输出指纹相等不等于精度验收。',
        '', '|轮次|EP|BS/卡|本轮基线 μs|候选 μs|降耗|rank-max 基线→候选 μs|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for e in result['experiments']:
        for c in e['cases']:
            a0,a1=c['control'],c['candidate']
            lines.append(f'|{e["id"].upper()} {e["title"]}|{c["ep"]}|{c["bs"]}|{a0["mean_us"]:.2f}|{a1["mean_us"]:.2f}|{c["reduction_pct"]:+.2f}%|{a0["rank_max_mean_us"]:.2f}→{a1["rank_max_mean_us"]:.2f}|')
    lines+=['','每个已审计case均为ABBA、两次独立启动/方法；原始CSV逐rank重算通过，输入/输出指纹一致。微小均值差不自动构成稳定收益。',
        '', '## 分支与缺测','']
    for e in result['experiments']:
        eps=sorted({c['ep'] for c in e['cases']})
        lines.append(f'- {e["id"].upper()}：`{e["branch"]}`；已审计EP：{eps or "无"}。')
    lines+=['','EP32/EP128未完整采齐的条目保持待测；端口占用、SHMEM初始化失败、外部作业入场均不计为优化失败或有效性能。',
        '不终止其他用户程序，不修改正式master或冻结基线。已完成尝试及资源扫描详见同目录 `five-rounds.json`。','']
    (a.root/'five-rounds.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('AUDITED_CASES',len(cases),'ATTEMPTS',len(result['attempts']),'SCANS',len(result['scans']))


if __name__=='__main__':main()
