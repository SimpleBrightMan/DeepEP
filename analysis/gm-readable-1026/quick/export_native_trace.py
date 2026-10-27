"""保留原生非零槽及完整wait，默认可视化隐藏wait；不增加设备同步。"""
import json
from pathlib import Path


def export(raw, directory, rank):
    import numpy as np
    from deep_ep_experimental.megamoe.timer import protocol as p
    from deep_ep_experimental.megamoe.timer.parse_timer_csv import core_info, validate_event_lane
    from deep_ep_experimental.megamoe.timer.trace_parser import (
        apply_wait_mode, event_family, first_start_cycle, trace_events, write_summary,
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if len(raw) != p.TOTAL_BUFFER_SIZE:
        raise ValueError(f'rank={rank}: native timer/parser length mismatch')
    # 只压缩零槽，不丢弃任何原生非零值；可用indices/values恢复完整int64 buffer。
    indices = np.flatnonzero(raw)
    np.savez_compressed(directory / 'native_sparse.npz', numel=np.int64(len(raw)),
                        indices=indices.astype(np.int64), values=raw[indices].astype('<i8'))
    events = []
    counts = []
    for core in range(p.N_CORE_COUNT):
        group, kind, sub = core_info(core)
        names = [(name, index) for index, name in enumerate(p.FIXED_NAMES)]
        row = []
        for family, name in enumerate(p.DYNAMIC_NAMES):
            count = int(raw[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family])
            if not 0 <= count <= p.MAX_DYNAMIC_ITER:
                raise ValueError(f'rank={rank} core={core} {name}: count={count}, overflow')
            row.append(count)
            names.extend((f'{name}_{i}', len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER + i)
                         for i in range(count))
        counts.append(row)
        for name, slot in names:
            offset = core * p.COUNTERS_PER_CORE + slot * 2
            start, end = int(raw[offset]), int(raw[offset + 1])
            if start == end == 0:
                continue
            if not 0 < start < end:
                raise ValueError(f'rank={rank} core={core} {name}: invalid interval {start}/{end}')
            validate_event_lane(name, core, kind, start, end)
            task = p.task_metadata(raw, core, slot)
            task = {k: int(v) if isinstance(v, np.integer) else v for k, v in task.items()}
            wave = int(raw[p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + slot])
            if wave < -1:
                raise ValueError(f'rank={rank} core={core} {name}: invalid wave={wave}')
            events.append(dict(name=name, rank=rank, group_id=group, core_type=kind, sub_id=sub,
                               start_cycle=start, end_cycle=end, task=task, wave_id=wave))
    # 同原解析器，在过滤wait之前确定每个wave/family的任务编号。
    groups = {}
    keys = []
    for event in events:
        family = event_family(event['name'])
        if event['task'].get('granularity') == 'tile' or (event['task'] and family in ('AIV_COMBINE', 'AIV_COMBINE_WAIT')):
            family = family.removesuffix('_WAIT')
        key = (event['group_id'], event['core_type'], event['sub_id'], family, event['wave_id'])
        suffix = event['name'].rsplit('_', 1)[-1]
        slot = int(suffix) if suffix.isdigit() else 0
        groups.setdefault(key, set()).add(slot)
        keys.append((key, slot))
    lookup = {k: {slot: i for i, slot in enumerate(sorted(v))} for k, v in groups.items()}
    for event, (key, slot) in zip(events, keys):
        event['wave_index'] = lookup[key][slot]
    if not events:
        raise ValueError(f'rank={rank}: empty timer')
    (directory / 'native_events.json').write_text(json.dumps(events, separators=(',', ':')))
    # 保留原生kernel包络用于分析，不在各泳道叠加一个覆盖整图的大矩形。
    visual_events = [e for e in events if e['name'] != 'KERNEL_TIMING']
    trace = trace_events(visual_events, 'keep', first_start_cycle(events), False, 'rank')
    (directory / 'timer_trace.json').write_text(json.dumps(trace, separators=(',', ':')))
    write_summary(apply_wait_mode(visual_events, False), directory / 'timer_summary.csv')
    metadata = dict(rank=rank, protocol='sparse_native_i64_v1', numel=len(raw),
                    nonzero_values=len(indices), default_view='wait_hidden_no_time_compression',
                    dynamic_groups=p.DYNAMIC_TRACE_GROUPS, dynamic_aiv0=p.DYNAMIC_TRACE_AIV0,
                    dynamic_names=p.DYNAMIC_NAMES, dynamic_counts=counts, event_count=len(events),
                    sparse_reconstruction='raw=zeros(numel,int64); raw[indices]=values')
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return dict(rank=rank, event_count=len(events), visible_events=len(trace))
