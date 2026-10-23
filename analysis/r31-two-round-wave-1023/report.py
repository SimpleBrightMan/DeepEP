"""复核原始32rank CSV、输入输出及完整被动流水，不继承旧paired父版收益。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--harness', type=Path, required=True)
    a = p.parse_args()
    manifest = load(a.root/'experiment.json')
    if manifest['status'] != 'collected_not_yet_audited':
        raise ValueError('Only audit a completed fixed schedule')
    for file in ('report_trace.py','campaign.py'):
        if hashlib.sha256((a.harness/file).read_bytes()).hexdigest() != manifest['harness_hashes'][file]:
            raise ValueError('Frozen harness changed')
    sys.path.insert(0, str(a.harness))
    from report_trace import audit
    records = []
    inputs, outputs = None, None
    for job in manifest['jobs'].split(','):
        label = 'bs4096-' + job
        r = audit(a.root, label)
        current = [load(a.root/f'node{k//8}/runs/{label}/rank_{k}/output.json') for k in range(32)]
        if inputs is None:
            inputs, outputs = r['inputs'], current
        if inputs != r.pop('inputs') or current != outputs or not all(x['finite'] for x in current):
            raise ValueError('Inputs/outputs differ; not an independent Golden test')
        events = load(a.root/f'node0/runs/{label}/rank_0/timer/native_events.json')
        families = {}
        for e in events:
            families.setdefault(re.sub(r'_\d+$','',e['name']), []).append(e)
        g1 = families['AIC_GMM1_ISSUE']
        g2 = families['AIC_GMM2_ISSUE']
        waves = sorted({e['wave_id'] for e in g1})
        expected_waves = 16 if r['method'] == 'paired' else 32
        if len(waves) != expected_waves or len(g1) != 2304 or len(g2) != 2688:
            raise ValueError(f'{label}: wrong waves or GMM task count')
        tile_keys = {(e['task']['expert_id'],e['task']['row_start'],e['task']['n_tile']) for e in g1}
        g2_keys = {(e['task']['expert_id'],e['task']['row_start'],e['task']['n_tile']) for e in g2}
        if len(tile_keys) != len(g1) or len(g2_keys) != len(g2):
            raise ValueError('Duplicated GMM work')
        if ({sum(e['group_id']==k for e in g1) for k in range(32)} != {72}
                or {sum(e['group_id']==k for e in g2) for k in range(32)} != {84}):
            raise ValueError('Wrong per-AIC task assignment')
        last_g1 = max(e['end_cycle'] for e in g1)
        last_gmm = max(last_g1,max(e['end_cycle'] for e in g2))
        export_end = max(e['end_cycle'] for e in families['AIV_EXPORT_SYNC'])
        combine = families.get('AIV_COMBINE_ISSUE',families.get('AIV_COMBINE',[]))
        if not combine:
            raise ValueError('No Combine timeline')
        r.update(wave_count=len(waves),gmm1_task_count=len(g1),gmm2_task_count=len(g2),
            after_last_gmm1_to_export_us=(export_end-last_g1)/1000,
            after_last_gmm_to_export_us=(export_end-last_gmm)/1000,
            longest_combine_issue_us=max(e['end_cycle']-e['start_cycle'] for e in combine)/1000,
            rank_max_mean_us=load(a.root/label/'performance.json')['rank_max_mean_us'],
            raw_inputs_equal=True,output_hashes_equal=True)
        records.append(r)
    summary = {}
    for method in ('paired','paired2'):
        runs = [r for r in records if r['method'] == method]
        if len(runs) != 2:
            raise ValueError('ABBA requires two complete runs per method')
        summary[method] = dict(mean_us=statistics.mean(r['off_us'] for r in runs),
            rank_max_mean_us=statistics.mean(r['rank_max_mean_us'] for r in runs),
            starts=2,runs=[{k:r[k] for k in ('label','off_us','on_us','overhead_pct','rank_max_mean_us')} for r in runs])
    gain = 100*(1-summary['paired2']['mean_us']/summary['paired']['mean_us'])
    result = dict(ep=32,bs=4096,summary=summary,reduction_pct=gain,records=records,
        accuracy='not_run_user_requested',placement=dict(rack=manifest['rack'],nodes=manifest['nodes']),
        branches={m:r['revision'] for m,r in manifest['packages'].items()},
        scope='full communication kernel Duration; timer-off; last10/20; mean per-round rank-min',
        caveat='rank0 last-call ISSUE intervals are not hardware completion or additive critical-path times')
    (a.root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# R31两轮wave：真实EP32/BS4096', '', result['scope'],
        '原R31控制与候选共用同一设备库，仅host wave预算不同。未跑Golden；32rank输入/输出指纹一致且有限。', '',
        '|方法|两次timer-off μs|均值 μs|rank-max μs|', '|---|---|---:|---:|']
    for method, row in summary.items():
        times = ' / '.join(format(r['off_us'], '.3f') for r in row['runs'])
        lines.append(f"|{method}|{times}|{row['mean_us']:.3f}|{row['rank_max_mean_us']:.3f}|")
    lines += ['',f'延迟降低率：{gain:.3f}%；是R31本轮新对照，不继承旧paired父版9.10%。', '',
        '|启动|wave数|最后GMM1至EXPORT μs|最后GMM至EXPORT μs|timer扰动 %|', '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['label']}|{r['wave_count']}|{r['after_last_gmm1_to_export_us']:.3f}|{r['after_last_gmm_to_export_us']:.3f}|{r['overhead_pct']:.3f}|")
    lines += ['', '流水只解释rank0最后一次调用；ISSUE不是硬件完成，区间不可相加。只测一个shape，不外推其他EP/BS。']
    (a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['status'] = 'audited_complete'
    (a.root/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    files = {str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.root.rglob('*')
             if f.is_file() and f.name != 'manifest.json'}
    (a.root/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
