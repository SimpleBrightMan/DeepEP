# Copyright (c) 2026, SimpleBright_Man

"""Independent, token-level oracle for the EP16 W4A8 communication experiment.

This module enumerates the benchmark's route stream and its per-peer queues.
It does not consume device slice counts, timer wave IDs, or the C++ planner.
"""

from collections import defaultdict
from typing import NamedTuple

WORLD = 16
CORES = 28
TILE_M = 256


class Transfer(NamedTuple):
    operation: str
    wave: int
    core: int
    expert: int
    row: int
    source_rank: int
    source_token: int
    topk_slot: int


def wave_budget(bs, hidden, intermediate, cores=CORES):
    """Derive the public W4A8 tiling budget from model dimensions."""

    def ceil(a, b):
        return (a + b - 1) // b

    if bs <= 256:
        return ceil(cores * 2, ceil(intermediate, 128))
    return max(ceil(cores * 4, ceil(2 * intermediate, 256)), ceil(cores, ceil(hidden, 256)))


def peer_order(rank, ranks_per_node):
    if ranks_per_node in (0, WORLD):
        return [(rank + offset) % WORLD for offset in range(WORLD)]
    if ranks_per_node <= 0 or WORLD % ranks_per_node:
        raise ValueError("ranks_per_node must divide 16")
    nodes = WORLD // ranks_per_node
    result = []
    for lane in range(ranks_per_node):
        offsets = [1, 0, *range(2, nodes)]
        for offset in offsets:
            node = (rank // ranks_per_node + offset) % nodes
            local = (rank % ranks_per_node + lane) % ranks_per_node
            result.append(node * ranks_per_node + local)
    return result


def route_queues(bs, topk, experts, rank):
    """Return [local expert][source] ordered (token, top-k slot) queues."""
    if not 0 <= rank < WORLD or experts <= 0 or experts % WORLD or not 0 < topk <= experts or bs <= 0:
        raise ValueError("Invalid EP16 balanced routing shape")
    local_experts = experts // WORLD
    queues = [[[] for _ in range(WORLD)] for _ in range(local_experts)]
    source_length = bs * topk
    for expert in range(local_experts):
        # A global route index with this residue targets this rank/expert.
        residue = expert * WORLD + rank
        for source in range(WORLD):
            base = source * source_length
            first = base + (residue - base) % experts
            queues[expert][source] = [
                divmod(route - base, topk) for route in range(first, base + source_length, experts)
            ]
    return queues


def _balanced_owners(rows, workers, rotation):
    """Assign a row sequence in quotient/remainder chunks to rotated workers."""
    result = []
    quotient, remainder = divmod(len(rows), len(workers))
    offset = 0
    for logical in range(len(workers)):
        count = quotient + (logical < remainder)
        owner = workers[(logical + rotation) % len(workers)]
        result.extend((owner, item) for item in rows[offset : offset + count])
        offset += count
    return result


def plan(bs, hidden, intermediate, experts, topk, rank, dispatch_intra=0, combine_intra=0, ranks_per_node=16):
    """Enumerate exact row/token identities and expected ownership in every wave.

    The peer-balanced layout first allocates a slice from round-robin peer
    queues, then lays each peer's quota contiguously in that original slice.
    This explicit enumeration is independent of the device prefix/cut formulas.
    """
    if any(type(n) is not int or not 0 <= n < CORES for n in (dispatch_intra, combine_intra)):
        raise ValueError("INTRA core counts must be 0 (original) or 1..27")
    queues = route_queues(bs, topk, experts, rank)
    budget = wave_budget(bs, hidden, intermediate)
    order = peer_order(rank, ranks_per_node)
    transfers, slices = [], []
    wave = used_groups = global_row = 0
    for expert, peer_queues in enumerate(queues):
        if bs > 256:
            allocation = [
                (source, ordinal)
                for ordinal in range(max(map(len, peer_queues), default=0))
                for source in order
                if ordinal < len(peer_queues[source])
            ]
        else:
            allocation = [(source, ordinal) for source in range(WORLD) for ordinal in range(len(peer_queues[source]))]
        row = 0
        while row < len(allocation):
            if used_groups == budget:
                wave += 1
                used_groups = 0
            count = min(len(allocation) - row, (budget - used_groups) * TILE_M)
            selected = allocation[row : row + count]
            if bs > 256:
                peer_quota = defaultdict(list)
                for source, ordinal in selected:
                    peer_quota[source].append(ordinal)
                selected = [(source, ordinal) for source in order for ordinal in peer_quota[source]]
            identities = [
                (row + index, source, ordinal, *peer_queues[source][ordinal])
                for index, (source, ordinal) in enumerate(selected)
            ]
            slices.append(dict(wave=wave, expert=expert, row_start=row, row_count=count, global_row_start=global_row))
            for operation, split in (("DISPATCH", dispatch_intra), ("COMBINE", combine_intra)):
                if not split:
                    owned = _balanced_owners(identities, list(range(CORES)), global_row + wave)
                else:
                    owned = []
                    for intra, workers in ((True, list(range(split))), (False, list(range(split, CORES)))):
                        rotation = global_row + wave
                        offsets = [0] if intra else [1, 2, 3]
                        for offset in offsets:
                            group = (rank // 4 - offset) % 4
                            group_rows = sorted(
                                (identity for identity in identities if identity[1] // 4 == group),
                                key=lambda identity: (identity[1], identity[2]),
                            )
                            owned.extend(_balanced_owners(group_rows, workers, rotation))
                            rotation += len(group_rows)
                transfers.extend(
                    Transfer(operation, wave, core, expert, local_row, source, token, slot)
                    for core, (local_row, source, _ordinal, token, slot) in owned
                )
            row += count
            global_row += count
            used_groups += (count + TILE_M - 1) // TILE_M
    return dict(m_groups_per_wave=budget, wave_count=wave + 1 if slices else 0, slices=slices, transfers=transfers)


def as_record(transfer, rank):
    return dict(
        rank=rank,
        operation=transfer.operation,
        wave=transfer.wave,
        core=transfer.core,
        expert_id=transfer.expert,
        row_start=transfer.row,
        row_count=1,
        source_rank=transfer.source_rank,
        source_token=transfer.source_token,
        topk_slot=transfer.topk_slot,
    )
