# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU control-plane helpers and an independent payload-count oracle for EP16."""

import math
import statistics

import numpy as np


def validate_routes(ids, experts):
    ids = np.asarray(ids)
    if ids.ndim != 3 or ids.shape[0] != 16 or ids.shape[1] < 1 or ids.shape[2] < 1:
        raise ValueError("Routes must have shape [16,B,K] with positive B,K")
    if ids.dtype != np.int32 or experts < 16 or experts % 16:
        raise ValueError("Routes require int32 IDs and experts divisible by 16")
    if not 1 <= ids.shape[2] <= min(32, experts) or np.any(ids < 0) or np.any(ids >= experts):
        raise ValueError("Expert ID or top-k outside supported range")
    ordered = np.sort(ids, axis=-1)
    if np.any(ordered[..., 1:] == ordered[..., :-1]):
        raise ValueError("Duplicate experts within one token are unsupported")
    return ids


def gather_route_ids(group, local_ids, experts, chunk_elements=32768):
    """Chunk JSON control messages below StoreGroup's 1 MiB per-rank limit."""
    local_ids = np.asarray(local_ids)
    if group.size() != 16 or local_ids.ndim != 2 or local_ids.dtype != np.int32:
        raise ValueError("Expected EP16 and local int32 [B,K] routes")
    if not 1 <= chunk_elements <= 32768:
        raise ValueError("chunk_elements must be in [1,32768]")
    shapes = group.all_gather(list(local_ids.shape), "hierarchy route shapes")
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("Ranks must use identical B,K")
    flat = local_ids.reshape(-1)
    result = np.empty((16, flat.size), dtype=np.int32)
    for begin in range(0, flat.size, chunk_elements):
        end = min(begin + chunk_elements, flat.size)
        packets = group.all_gather(flat[begin:end].tolist(), f"hierarchy routes {begin}:{end}")
        if len(packets) != 16 or any(len(packet) != end - begin for packet in packets):
            raise ValueError("Incomplete route-control chunk")
        if any(type(value) is not int or not 0 <= value < experts for packet in packets for value in packet):
            raise ValueError("Invalid route-control expert ID")
        result[:, begin:end] = packets
    return validate_routes(result.reshape(16, *local_ids.shape), experts)


def expected_payload_bytes(ids, experts, hidden, rank):
    """Count GET payloads from logical routes, independently of native slot order.

    Dispatch counters belong to the receiving rank; Combine intra belongs to
    the group proxy and Combine cross to the original source rank. Same-card
    reads are reported separately, never charged as network traffic.
    """
    ids = validate_routes(ids, experts)
    if not 0 <= rank < 16 or hidden < 1 or hidden % 32:
        raise ValueError("Invalid rank or hidden size")
    destination = ids // (experts // 16)
    masks = np.zeros(ids.shape[:2], dtype=np.uint32)
    for k in range(ids.shape[-1]):
        masks |= np.left_shift(np.uint32(1), destination[..., k].astype(np.uint32))
    own_group, lane = divmod(rank, 4)
    group_bits = np.uint32(15 << (own_group * 4))
    dispatch_cross = intra = local = combine_intra = combine_cross = 0
    for source in range(16):
        needed = (masks[source] & np.uint32(1 << rank)) != 0
        upstream = source if source // 4 == own_group else own_group * 4 + source % 4
        if upstream == rank:
            local += int(np.count_nonzero(needed))
        else:
            intra += int(np.count_nonzero(needed))
        if source % 4 == lane:
            group_needed = (masks[source] & group_bits) != 0
            if source // 4 != own_group:
                dispatch_cross += int(np.count_nonzero(group_needed))
            for consumer in range(own_group * 4, own_group * 4 + 4):
                if consumer != rank:
                    combine_intra += int(np.count_nonzero(masks[source] & np.uint32(1 << consumer)))
    for group in range(4):
        if group != own_group:
            combine_cross += int(np.count_nonzero(masks[rank] & np.uint32(15 << (group * 4))))
    stride = ((hidden + 255) // 256) * 256 + ((hidden // 32 + 31) // 32) * 32
    return dict(
        dispatch_cross_bytes=dispatch_cross * stride,
        dispatch_intra_bytes=intra * stride,
        dispatch_local_bytes=local * stride,
        combine_intra_bytes=combine_intra * hidden * 4,
        combine_cross_bytes=combine_cross * hidden * 4,
        local_unique_tokens=intra + local,
    )


def validate_payload_metrics(metrics, expected):
    for key, value in expected.items():
        if not key.endswith("_bytes"):
            continue
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual != value:
            raise ValueError(f"Native payload mismatch for {key}: expected={value}, actual={actual}")


def summarize_measurements(per_rank):
    """All-rank operator spans, including the prototype's host/stage gaps."""
    if len(per_rank) != 16 or not per_rank[0] or any(len(rows) != len(per_rank[0]) for rows in per_rank):
        raise ValueError("Require 16 ranks with identical nonempty iteration counts")
    answer = {"ranks": 16, "rounds": len(per_rank[0])}
    for metric in ("acl_span_us", "wall_us"):
        values = [[row[metric] for row in rows] for rows in per_rank]
        if any(not math.isfinite(v) or v <= 0 for row in values for v in row):
            raise ValueError(f"Invalid {metric}")
        maxima = [max(round_values) for round_values in zip(*values)]
        answer[metric] = dict(
            rank_max_mean=statistics.mean(maxima),
            rank_max_min=min(maxima),
            rank_max_max=max(maxima),
            rank_mean=statistics.mean(v for row in values for v in row),
            per_rank_mean=[statistics.mean(row) for row in values],
            round_rank_max=maxima,
        )
    return answer


STAGE_BANDWIDTH = {
    "dispatch_cross": ("dispatch_cross_bytes", "last_dispatch_cross_ms"),
    "dispatch_intra": ("dispatch_intra_bytes", "last_dispatch_intra_ms"),
    "combine_intra": ("combine_intra_bytes", "last_group_reduce_ms"),
    "combine_cross": ("combine_cross_bytes", "last_source_reduce_ms"),
}


def stage_rates(metrics):
    """Payload per synchronous stage wall time; not pure MTE/link bandwidth."""
    rates = {}
    for name, (byte_field, time_field) in STAGE_BANDWIDTH.items():
        size, ms = metrics[byte_field], metrics[time_field]
        if not math.isfinite(size) or size < 0 or not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"Invalid stage metrics: {name}")
        rates[name] = size / (ms * 1e6)
    return rates
