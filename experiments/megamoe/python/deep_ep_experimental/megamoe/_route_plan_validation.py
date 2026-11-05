"""Post-profile CPU oracle for route-only descriptor experiments.

Never imported or called by production forward. Build row identities with the
private independent oracle rather than calling the Device descriptor compiler.
"""
from types import SimpleNamespace
import numpy as np


FRAGMENT = np.dtype([
    ("expert", "<u4"), ("segmentBegin", "<u4"), ("rowBegin", "<u8"),
    ("rows", "<u4"), ("segmentCount", "<u4"), ("gmm1TileBegin", "<u8"), ("gmm2TileBegin", "<u8"),
], align=True)
SEGMENT = np.dtype([
    ("sourceRank", "<u4"), ("cv", "<u4"), ("receiveRow", "<u8"), ("returnRow", "<u8"),
    ("tileRow", "<u4"), ("rows", "<u4"), ("inputReadyIndex", "<u4"), ("reserved", "<u4"),
], align=True)
CHUNK = np.dtype([
    ("peer", "<u4"), ("cv", "<u4"), ("localRow", "<u8"), ("remoteRow", "<u8"),
    ("rows", "<u4"), ("fragmentBegin", "<u4"), ("fragmentCount", "<u4"), ("reserved", "<u4"),
], align=True)
BATCH = np.dtype([("fragmentBegin", "<u4"), ("fragmentCount", "<u4")], align=True)
TASK = np.dtype([("stage", "<u4"), ("batchIndex", "<u4")], align=True)
PACK = np.dtype([("chunks", CHUNK, (16,)), ("rows", "<u8")], align=True)
CONFIG = np.dtype([
    *((name, "<u4") for name in ("bs", "hidden", "intermediate", "topk", "localExperts", "cvRows",
       "rank", "world", "maxPeerChunks", "fragmentCount", "sendChunkCount", "recvChunkCount",
       "gmm1AheadMGroups", "computeTaskCount", "packRoundCount")),
    *((name, "<u8") for name in ("receiveRows", "gmm1Tiles", "gmm2Tiles", "epoch", "receiveXOffset",
       "receiveScaleOffset", "returnYOffset", "dispatchReadyOffset", "combineReadyOffset", "barrierOffset",
       "symmetricBytes", "selfSendRow", "selfRecvRow", "selfRows", "sourceRowCapacity", "routeInboxOffset",
       "routeEntryOffset", "routeReadyOffset")),
    ("routeSegmentCount", "<u4"), ("routeLinkCount", "<u4"),
], align=True)
assert FRAGMENT.itemsize == SEGMENT.itemsize == CHUNK.itemsize == 40
assert PACK.itemsize == 648 and BATCH.itemsize == TASK.itemsize == 8
assert CONFIG.itemsize == 216 and CONFIG.fields["sourceRowCapacity"][1] == 176


def _records(items, dtype):
    result = np.zeros(len(items), dtype=dtype)
    for index, item in enumerate(items):
        result[index] = tuple(item)
    return result


def expected_plan(source_counts, *, bs, topk, world, rank, max_bs, hidden, intermediate,
                  cv_rows, gmm1_ahead_mgroups=12):
    from ._urma_route_reference import expert_m_groups
    counts = np.asarray(source_counts, dtype=np.int64)
    if counts.shape != (world, 384):
        raise ValueError(f"reference count shape {counts.shape} != {(world,384)}")
    if np.any(counts < 0) or np.any(counts.sum(axis=1) != bs * topk):
        raise ValueError("invalid diagnostic source count table")
    local = 384 // world
    pitch = max_bs * topk
    max_cvs = (pitch + cv_rows - 1) // cv_rows
    routes = []
    for source in range(world):
        peer_counts = counts[source].reshape(world, local).sum(axis=1)
        routes.append(SimpleNamespace(
            world=world, num_experts=384,
            expert_offsets=np.r_[0, counts[source].cumsum()],
            peer_counts=peer_counts, peer_offsets=np.r_[0, peer_counts.cumsum()],
        ))
    compact_receive = np.r_[0, np.cumsum([route.peer_counts[rank] for route in routes])]
    groups = expert_m_groups(routes, rank, cv_rows)
    g1_per_group = (intermediate + (128 if bs * world * topk < 384 * 128 else 256) - 1) // (
        128 if bs * world * topk < 384 * 128 else 256)
    g2_per_group = (hidden + 255) // 256
    fragments, segments, chunk_groups = [], [], {}
    for group_id, group in enumerate(groups):
        segment_begin = len(segments)
        for segment in group.segments:
            source = segment.source_rank
            fixed_receive = source * pitch + segment.receive_row - int(compact_receive[source])
            segments.append((source, segment.cv_id, fixed_receive, segment.return_row,
                             segment.tile_row, segment.rows, source * max_cvs + segment.cv_id, 0))
            dependencies = chunk_groups.setdefault((source, segment.cv_id), [])
            if not dependencies or dependencies[-1] != group_id:
                dependencies.append(group_id)
        fragments.append((group.local_expert, segment_begin, group.row_begin, group.rows,
                          len(segments) - segment_begin, group_id * g1_per_group, group_id * g2_per_group))

    send_chunks, recv_chunks, links = [], [], []
    for peer in range(world):
        send_rows = int(routes[rank].peer_counts[peer])
        send_base = int(routes[rank].peer_offsets[peer])
        for cv, first in enumerate(range(0, send_rows, cv_rows)):
            send_chunks.append((peer, cv, send_base + first, rank * pitch + first,
                                min(cv_rows, send_rows - first), 0, 0, 0))
        receive_rows = int(routes[peer].peer_counts[rank])
        return_base = int(routes[peer].peer_offsets[rank])
        for cv, first in enumerate(range(0, receive_rows, cv_rows)):
            deps = chunk_groups[(peer, cv)]
            recv_chunks.append((peer, cv, peer * pitch + first, return_base + first,
                                min(cv_rows, receive_rows - first), len(links), len(deps), 0))
            links.extend(deps)

    rounds = max(((int(n) + cv_rows - 1) // cv_rows for n in routes[rank].peer_counts), default=0)
    pack = np.zeros(rounds, dtype=PACK)
    # Missing original CVs and all peer slots >= active EP must remain all-zero,
    # including their peer/cv/localRow fields, not merely rows=0.
    for chunk in send_chunks:
        peer, cv = chunk[:2]
        pack[cv]["chunks"][peer] = chunk
        pack[cv]["rows"] += int(chunk[4])

    batches, tasks = [], []
    if world == 4:
        for wave, first in enumerate(range(0, len(groups), 12)):
            batches.append((first, min(12, len(groups) - first)))
            tasks.extend(((1, wave), (2, wave)))
    else:
        group_cursor = 0
        for expert in range(local):
            n = sum(group.local_expert == expert for group in groups)
            batches.append((group_cursor, n)); group_cursor += n
        next_g1, outstanding = 0, 0
        for g2 in range(local):
            # Reproduce the public expert lookahead contract from batch sizes;
            # no Device-generated task/offset values are used as reference.
            while next_g1 < local and (next_g1 < min(local, g2 + 2) or outstanding < gmm1_ahead_mgroups):
                tasks.append((1, next_g1)); outstanding += batches[next_g1][1]; next_g1 += 1
            tasks.append((2, g2)); outstanding -= batches[g2][1]

    def align(value, step=512):
        return (int(value) + step - 1) // step * step
    receive_capacity = world * pitch
    receive_scale = align(receive_capacity * hidden)
    returned = align(receive_scale + receive_capacity * (hidden // 32))
    dispatch_ready = align(returned + pitch * hidden * 2)
    combine_ready = align(dispatch_ready + world * max_cvs * 128)
    barrier = align(combine_ready + world * max_cvs * 128)
    inbox = align(barrier + 4 * world * 128)
    entry = align(inbox + world * 512)
    ready = entry + world * 128
    config = dict(bs=bs, hidden=hidden, intermediate=intermediate, topk=topk,
        localExperts=local, cvRows=cv_rows, rank=rank, world=world, maxPeerChunks=max_cvs,
        fragmentCount=len(groups), sendChunkCount=len(send_chunks), recvChunkCount=len(recv_chunks),
        gmm1AheadMGroups=gmm1_ahead_mgroups, computeTaskCount=len(tasks), packRoundCount=rounds,
        receiveRows=int(compact_receive[-1]), gmm1Tiles=len(groups)*g1_per_group,
        gmm2Tiles=len(groups)*g2_per_group, receiveXOffset=0, receiveScaleOffset=receive_scale,
        returnYOffset=returned, dispatchReadyOffset=dispatch_ready, combineReadyOffset=combine_ready,
        barrierOffset=barrier, symmetricBytes=align(ready+world*128,2<<20),
        selfSendRow=int(routes[rank].peer_offsets[rank]), selfRecvRow=rank*pitch,
        selfRows=int(routes[rank].peer_counts[rank]), sourceRowCapacity=pitch,
        routeInboxOffset=inbox, routeEntryOffset=entry, routeReadyOffset=ready,
        routeSegmentCount=len(segments), routeLinkCount=len(links))
    arrays = dict(fragments=_records(fragments,FRAGMENT), segments=_records(segments,SEGMENT),
                  send_chunks=_records(send_chunks,CHUNK), recv_chunks=_records(recv_chunks,CHUNK),
                  compute_batches=_records(batches,BATCH), compute_tasks=_records(tasks,TASK),
                  chunk_group_indices=np.asarray(links,dtype="<u4"), pack_rounds=pack)
    return arrays, config


def _raw_bytes(tensor):
    if hasattr(tensor,"cpu"):
        tensor = tensor.cpu().numpy()
    return np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)


def _compare(actual, expected, path, checks, mismatches):
    if expected.dtype.names:
        for field in expected.dtype.names:
            _compare(actual[field], expected[field], f"{path}.{field}", checks, mismatches)
        return
    equal = actual == expected
    checks[path] = bool(np.all(equal))
    if not checks[path] and len(mismatches)<16:
        index = tuple(map(int,np.argwhere(~equal)[0]))
        mismatches.append(dict(field=path,index=list(index),actual=int(actual[index]),expected=int(expected[index])))


def validate_plan(native, source_counts, **dimensions):
    """Read Device only after measurement; return bounded JSON diagnostics."""
    checks, mismatches, sizes = {}, [], {}
    try:
        expected, config = expected_plan(source_counts, **dimensions)
        buffers = native.route_plan_tensors()
        for name, wanted in expected.items():
            raw = _raw_bytes(buffers[name])
            sizes[name] = dict(records=len(wanted),bytes=int(wanted.nbytes),capacity_bytes=int(raw.size))
            if raw.size < wanted.nbytes:
                checks[name+".capacity"] = False
                mismatches.append(dict(field=name+".capacity",actual=int(raw.size),expected=int(wanted.nbytes)))
                continue
            actual = np.frombuffer(raw,dtype=wanted.dtype,count=len(wanted))
            _compare(actual,wanted,name,checks,mismatches)
        raw = _raw_bytes(buffers["config"])
        if raw.size < CONFIG.itemsize:
            checks["config.capacity"] = False
        else:
            actual = np.frombuffer(raw,dtype=CONFIG,count=1)[0]
            for field, wanted in config.items():
                checks["config."+field] = int(actual[field]) == wanted
                if not checks["config."+field] and len(mismatches)<16:
                    mismatches.append(dict(field="config."+field,actual=int(actual[field]),expected=int(wanted)))
            checks["config.epoch_positive"] = int(actual["epoch"]) > 0
        return dict(passed=all(checks.values()),checks=checks,first_mismatches=mismatches,sizes=sizes,
                    oracle="CPU row identities -> M256 groups; fixed source pitch; field-by-field active-prefix comparison")
    except Exception as error:
        return dict(passed=False,checks=checks,first_mismatches=mismatches,error=f"{type(error).__name__}: {error}")
