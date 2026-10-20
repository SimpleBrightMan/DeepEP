"""Archive the asynchronous lifecycle experiment, preserving ISSUE/completion distinction."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import struct
import tarfile

root = Path(__file__).resolve().parent
build = root.parent / 'aiv1-async-1020-v2'
for folder, phase in ((root, 'campaign'), (root, 'diagnose'), (root, 'regress'), (root, 'edges'), (root, 'skew_recheck')):
    if (folder / f'{phase}.rc').read_text().strip() != '0':
        raise ValueError(f'{folder.name}/{phase}: incomplete')
identity = json.loads((root / 'identity.json').read_text())
control = root.parent / 'aiv1-fair-1020/fair-package/deep_ep_experimental/megamoe'
control_sha = hashlib.sha256((control / 'libmegamoe_kernel.so').read_bytes()).hexdigest()
shas = {'control': control_sha, 'async': identity['kernel_sha256']}
extension_shas = {'control': hashlib.sha256(next(control.glob('_C*.so')).read_bytes()).hexdigest(),
                  'async': next(v for k,v in identity['libraries'].items() if k.startswith('_C'))}
result = dict(ep=8, bs=512, scope='full communication; timer off; warm10/sample20/tail10; mean of per-round rank-min',
              accuracy='8-rank finite and bitwise differential; independent Golden not run',
              identity=identity, kernel_shas=shas, extension_shas=extension_shas, methods={}, traces={}, pmu=[])

def validate_run(folder, method, bs, routing):
    report = json.loads((folder / 'performance.json').read_text())
    regression = json.loads((folder / 'regression.json').read_text())
    if (report['ep'], report['bs'], report['timer'], report['rounds'], report['sha256'], report['extension_sha256']) != (
            8, bs, False, 10, shas[method], extension_shas[method]):
        raise ValueError(f'{folder.name}: config/native identity mismatch')
    if not regression['passed'] or regression['ranks'] != [True]*8:
        raise ValueError(f'{folder.name}: output differential failed')
    for rank in range(8):
        manifest = json.loads((folder / f'setup/rank_{rank}/manifest.json').read_text())
        if (manifest['ep_size'], manifest['rank'], manifest['input'], manifest['routing'], manifest['measurement']) != (
                8, rank, 'fp8', routing, 'full_moe'):
            raise ValueError(f'{folder.name}: effective rank/routing contract mismatch')
        if manifest['config']['model'] != dict(hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=6, num_shared_experts=0):
            raise ValueError(f'{folder.name}: model contract mismatch')
    return report

result['shape_screening'] = []
for bs in (96, 1024, 4096):
    reports = {}
    for method in ('control', 'async'):
        folder = root / f'bs{bs}-{method}'
        report = validate_run(folder, method, bs, 'balanced')
        reports[method] = dict(mean_us=report['mean_us'], rank_max_mean_us=report['rank_max_mean_us'])
    result['shape_screening'].append(dict(bs=bs, methods=reports, scope='one launch each; regression screening, not ABBA promotion'))
result['routing_regression'] = []
for routing in ('random', 'skewed'):
    reports = {}
    for method in ('control', 'async'):
        report = validate_run(root / f'{routing}-{method}', method, 96, routing)
        reports[method] = dict(mean_us=report['mean_us'], rank_max_mean_us=report['rank_max_mean_us'])
    result['routing_regression'].append(dict(bs=96, routing=routing, methods=reports, scope='single launch; differential regression only'))
for method, prefix in (('control', 'control'), ('async', 'candidate')):
    runs = []
    for suffix in ('a', 'b'):
        label = f'{prefix}-{suffix}'
        report = validate_run(root / label, method, 512, 'balanced')
        runs.append(dict(label=label, mean_us=report['mean_us'], rank_max_mean_us=report['rank_max_mean_us']))
    result['methods'][method] = dict(runs=runs, mean_us=statistics.mean(r['mean_us'] for r in runs))
result['reduction_pct'] = 100 * (1 - result['methods']['async']['mean_us']/result['methods']['control']['mean_us'])
skew = {}
for method in ('control', 'async'):
    runs = []
    for label in (f'skewed-{method}', f'skewed-{method}-b'):
        report = validate_run(root / label, method, 96, 'skewed')
        runs.append(dict(label=label, mean_us=report['mean_us'], rank_max_mean_us=report['rank_max_mean_us']))
    skew[method] = dict(runs=runs, mean_us=statistics.mean(r['mean_us'] for r in runs))
result['skew_recheck'] = dict(bs=96, routing='skewed', methods=skew,
    reduction_pct=100*(1-skew['async']['mean_us']/skew['control']['mean_us']),
    caveat='extended A/B/B/A; Combine6->2 also changes below BS256; not an isolated lifecycle benefit')

for method in ('control', 'async'):
    folder = root / f'{method}-trace'
    events = json.loads((folder / 'timer_all_events.json').read_text())
    origin = min(e['start_cycle'] for e in events if e['name'] == 'KERNEL_TIMING' and e['core_type'] == 'AIC')
    trace = []
    for gid in range(32):
        for role, sub, offset in (('AIC', 0, 0), ('AIV', 0, 1), ('AIV', 1, 2)):
            trace.append(dict(name='thread_sort_index', ph='M', pid=0, tid=f'group_{gid}_{role.lower()}_{sub}', args={'sort_index': gid*3+offset}))
    for e in events:
        if e['name'] == 'KERNEL_TIMING' or any(s in e['name'] for s in ('WAIT', 'SYNC', 'DRAIN')):
            continue
        trace.append(dict(name=e['name'], ph='X', pid=0, tid=f"group_{e['group_id']}_{e['core_type'].lower()}_{e['sub_id']}",
            ts=(e['start_cycle']-origin)/1000, dur=(e['end_cycle']-e['start_cycle'])/1000,
            args={k: e[k] for k in ('group_id', 'core_type', 'sub_id')}))
    groups = []
    for gid in range(32):
        rows = [e for e in events if e['group_id'] == gid]
        sums = collections.defaultdict(float)
        for e in rows:
            if e['name'] != 'KERNEL_TIMING':
                sums[e['name']] += (e['end_cycle']-e['start_cycle'])/1000
        dispatch = [e for e in rows if e['name'] in ('AIV_DISPATCH', 'AIV_DISPATCH_ISSUE')]
        combine = [e for e in rows if e['name'] in ('AIV_COMBINE', 'AIV_COMBINE_ISSUE')]
        if not dispatch or not combine: raise ValueError(f'{method}: missing AIV1 group {gid}')
        groups.append(dict(group=gid, stages_us=dict(sums),
            last_dispatch_us=(max(e['end_cycle'] for e in dispatch)-origin)/1000,
            first_combine_us=(min(e['start_cycle'] for e in combine)-origin)/1000))
    if len({e['tid'] for e in trace if e['ph'] == 'X'}) != 96:
        raise ValueError(f'{method}: incomplete passive trace')
    (folder / 'timer_trace.json').write_text(json.dumps(trace, separators=(',', ':')))
    names = {key for g in groups for key in g['stages_us']}
    result['traces'][method] = dict(overhead=json.loads((folder / 'timer_overhead.json').read_text()),
        groups=groups, stage_mean_us={key: statistics.mean(g['stages_us'].get(key, 0) for g in groups) for key in sorted(names)},
        path=f'{method}-trace/timer_trace.json')

fmt = struct.Struct('=4HQ2BH2B3HL12Q')
cycle_events = ('0x1', '0x301', '0x501', '0x701', '0x202', '0x203', '0x714')
for method in ('control', 'async'):
    folder = root / f'pmu_{method}_PipeUtilization'
    paths = list(folder.glob('profiler/PROF_*/mindstudio_profiler_output/op_summary*.csv'))
    if len(paths) != 8:
        raise ValueError(f'{method}: expected eight rank PMU reports')
    matched = 0
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if any(n in (r['Op Name']+r['OP Type']).lower() for n in ('megamoe', 'mega_moe'))]
        if not rows or rows[0]['Device_id'] != '0':
            continue
        matched += 1
        rows.sort(key=lambda r: float(r['Task Start Time(us)']))
        if len(rows) < 20:
            raise ValueError(f'{method}: incomplete PMU calls')
        ids = {(int(r['Stream ID']), int(r['Task ID'])) for r in rows[-10:]}
        device = path.parent.parent / 'device_0'
        sample = json.loads((device / 'sample.json').read_text())
        with sqlite3.connect(f'file:{device / "sqlite/metric_summary.db"}?mode=ro', uri=True) as conn:
            conn.row_factory = sqlite3.Row
            timing = {(r['task_id'], r['core_type'], r['core_id']): r['duration']
                      for r in conn.execute('select * from V6BlockPmu') if (r['stream_id'], r['task_id']) in ids}
        records = []
        for rawpath in sorted((device / 'data').glob('ffts_profile.data.0.slice_*')):
            if rawpath.name.endswith('.done'): continue
            raw = rawpath.read_bytes()
            if len(raw) % fmt.size: raise ValueError(f'{method}: partial PMU record')
            for off in range(0, len(raw), fmt.size):
                f = fmt.unpack_from(raw, off)
                key = ((f[3] << 16) | f[2], f[8] & 1, f[9])
                if f[0] & 63 != 41 or key not in timing: continue
                names = sample['ai_core_profiling_events' if key[1] == 0 else 'aiv_profiling_events'].split(',')
                counters = dict(zip(names, f[14:24]))
                records.append(dict(task=key[0], role='AIC' if key[1] == 0 else f'AIV{f[11]}', block=f[12], physical_core=f[9],
                    duration_us=timing[key], total_cycles=f[4], events=counters,
                    equivalent_us={k: v/f[4]*timing[key] for k, v in counters.items() if k in cycle_events}))
        if len(records) != len(timing): raise ValueError(f'{method}: native/official PMU mismatch')
        roles = collections.defaultdict(list)
        for record in records: roles[record['role']].append(record)
        result['pmu'].append(dict(method=method, sample_config=sample, records=records,
            roles={role: dict(count=len(rs), blocks=sorted({r['block'] for r in rs}), duration_us=statistics.mean(r['duration_us'] for r in rs),
                equivalent_us={key: statistics.mean(r['equivalent_us'][key] for r in rs) for key in rs[0]['equivalent_us']}) for role, rs in roles.items()}))
    if matched != 1: raise ValueError(f'{method}: ambiguous device0 PMU')
(root / 'summary.json').write_text(json.dumps(result, indent=2))

# Only completed controllers are snapshotted. Keep traces and their exact CSVs together.
sources = set()
for base, prefix in ((root, ''),):
    selected = {p for p in base.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.rc', '.py', '.sh', '.cpp')}
    for folder in base.iterdir():
        if not folder.is_dir() or folder.name.endswith(('-source', '-package')) or folder.name == 'parser-check': continue
        selected.update(folder.glob('*.json'))
        for name in ('kernel_details.csv', 'kernel_samples.json', 'op_summary*.csv', 'sample.json', 'manifest.json', 'resources.json'):
            selected.update(folder.rglob(name))
        for device in folder.glob('profiler/PROF_*/device_0'):
            selected.update(device.glob('sqlite/metric_summary.db'))
            selected.update(device.glob('data/ffts_profile.data.0.slice_*'))
    for path in selected:
        rel = path.relative_to(base)
        if path.is_file(): sources.add((path, prefix+str(rel)))
for name in ('ours.py', 'check_label.py'):
    sources.add((root.parent / name, f'harness/{name}'))
for method, package in (('control', control), ('async', build / 'async-package/deep_ep_experimental/megamoe')):
    for path in [package / 'libmegamoe_kernel.so', *package.glob('_C*.so')]:
        sources.add((path, f'native-{method}/{path.name}'))
for attempt in ('aiv1-async-1020', 'aiv1-async-1020-v2', 'aiv1-async-1020-r1'):
    for name in ('controller.log', 'compile.log', 'compile-resume.log', 'cpu-preflight.log', 'campaign.rc',
                 'build-commands.json', 'identity.json', 'abi-probe.log', 'native/CMakeLists.txt', 'run_prepared_r2.sh',
                 'native-build/compile_commands.json'):
        path = root.parent / attempt / name
        if path.is_file(): sources.add((path, f'failed-builds/{attempt}/{name}'))
sources = {(p, name) for p, name in sources if name not in ('manifest.json', 'collect.log')}
manifest = [dict(path=name, size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path, name in sorted(sources, key=lambda item:item[1])]
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
with tarfile.open(root / 'evidence.tar.gz', 'w:gz') as archive:
    for path, name in sorted(sources, key=lambda item:item[1]): archive.add(path, arcname=name)
    archive.add(root / 'manifest.json', arcname='manifest.json')
print('COLLECT_COMPLETE', len(manifest), 'files', flush=True)
print('METHODS', json.dumps(result['methods']), 'REDUCTION', result['reduction_pct'], flush=True)
print('SKEW', json.dumps(result['skew_recheck']), flush=True)
print('PMU', json.dumps([dict(method=p['method'], roles=p['roles']) for p in result['pmu']]), flush=True)
print('STAGES', json.dumps({name: row['stage_mean_us'] for name,row in result['traces'].items()}), flush=True)
