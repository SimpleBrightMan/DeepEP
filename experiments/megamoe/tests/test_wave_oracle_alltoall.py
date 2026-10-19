# Copyright (c) 2026, SimpleBright_Man

from collections import Counter, defaultdict

import pytest

from comm_report_fixtures import capture
from deep_ep_experimental.megamoe import comm_report, wave_oracle


def payload_key(transfer):
    return (
        transfer.operation,
        transfer.wave,
        transfer.expert,
        transfer.row,
        transfer.source_rank,
        transfer.source_token,
        transfer.topk_slot,
    )


def check_owners(schedule, rank, dispatch, combine):
    """Check explicit modulo owners, unchanged INTRA chunks and actual issue order."""
    slices = {(s["wave"], s["expert"]): s for s in schedule["slices"]}
    groups = defaultdict(list)
    for transfer in schedule["transfers"]:
        groups[transfer.operation, transfer.wave, transfer.expert].append(transfer)
    for (operation, wave, expert), transfers in groups.items():
        split = dispatch if operation == "DISPATCH" else combine
        if not split:
            continue
        rotation = slices[wave, expert]["global_row_start"] + wave
        intra = sorted((t for t in transfers if t.source_rank // 4 == rank // 4), key=lambda t: (t.source_rank, t.row))
        base, extra = divmod(len(intra), split)
        position = 0
        for logical in range(split):
            count = base + (logical < extra)
            assert all(t.core == (logical + rotation) % split for t in intra[position : position + count])
            position += count

        # Ascending cross-group offsets, with source-rank-relative lane order.
        peers = [((rank // 4 + phase) % 4) * 4 + (rank % 4 + lane) % 4 for phase in (1, 2, 3) for lane in range(4)]
        prefix = 0
        workers = 28 - split
        for peer in peers:
            rows = sorted((t for t in transfers if t.source_rank == peer), key=lambda t: t.row)
            by_core = defaultdict(list)
            for ordinal, transfer in enumerate(rows):
                assert transfer.core == split + (rotation + prefix + ordinal) % workers
                by_core[transfer.core].append(transfer.row)
            loads = [len(by_core[core]) for core in range(split, 28)]
            assert max(loads) - min(loads) <= 1
            for owned_rows in by_core.values():
                assert all(right - left == workers for left, right in zip(owned_rows, owned_rows[1:]))
            prefix += len(rows)
        cross_loads = Counter(t.core for t in transfers if t.source_rank // 4 != rank // 4)
        assert (
            max(cross_loads[core] for core in range(split, 28)) - min(cross_loads[core] for core in range(split, 28))
            <= 1
        )
        for core in range(split, 28):
            phases = [peers.index(t.source_rank) for t in transfers if t.core == core]
            assert phases == sorted(phases)


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("rank", range(16))
def test_alltoall_preserves_ep16_wave_peer_payload_ready_and_inverse(bs, rank):
    original = wave_oracle.plan(bs, 3584, 1280, 384, 8, rank)
    striped = wave_oracle.plan(bs, 3584, 1280, 384, 8, rank, 4, 6, group_order="alltoall")
    assert striped["slices"] == original["slices"]
    assert striped["wave_count"] == original["wave_count"]
    assert striped["m_groups_per_wave"] == original["m_groups_per_wave"]
    actual_payload = Counter(map(payload_key, striped["transfers"]))
    assert actual_payload == Counter(map(payload_key, original["transfers"]))
    assert set(actual_payload.values()) == {1}

    # Ready counters still receive exactly the original number of completed rows
    # in each M256 tile, regardless of the strided physical owner assignment.
    def ready_rows(schedule):
        return Counter((t.wave, t.expert, t.row // 256) for t in schedule["transfers"] if t.operation == "DISPATCH")

    assert ready_rows(striped) == ready_rows(original)
    inverse = {}
    for operation in ("DISPATCH", "COMBINE"):
        records = [t for t in striped["transfers"] if t.operation == operation]
        assert len(records) == bs * 8
        inverse[operation] = {(t.expert, t.row): (t.source_rank, t.source_token, t.topk_slot) for t in records}
        assert len(inverse[operation]) == len(records)
        assert len(set(inverse[operation].values())) == len(records)
        for t in records:
            # Check the repository's route equation directly, without reading
            # either schedule's route queues or device metadata.
            route = t.source_rank * bs * 8 + t.source_token * 8 + t.topk_slot
            assert route % 384 == t.expert * 16 + rank
    assert inverse["DISPATCH"] == inverse["COMBINE"]
    check_owners(striped, rank, 4, 6)


@pytest.mark.parametrize("split", range(1, 28))
def test_alltoall_every_split_handles_unequal_peer_counts_and_rotating_remainders(split):
    # A small non-divisible Prefill exercises every worker count without
    # multiplying the three large BS cases by 27 full token enumerations.
    rank = split % 16
    schedule = wave_oracle.plan(257, 3584, 1280, 384, 8, rank, split, 28 - split, group_order="alltoall")
    baseline = wave_oracle.plan(257, 3584, 1280, 384, 8, rank)
    assert Counter(map(payload_key, schedule["transfers"])) == Counter(map(payload_key, baseline["transfers"]))
    check_owners(schedule, rank, split, 28 - split)


@pytest.mark.parametrize("dispatch,combine", [(4, 0), (0, 6), (0, 0)])
def test_alltoall_keeps_unsplit_stage_original(dispatch, combine):
    original = wave_oracle.plan(257, 3584, 1280, 384, 8, 11, dispatch, combine)
    striped = wave_oracle.plan(257, 3584, 1280, 384, 8, 11, dispatch, combine, group_order="alltoall")
    assert striped["slices"] == original["slices"]
    for operation, split in (("DISPATCH", dispatch), ("COMBINE", combine)):
        if not split:
            assert [t for t in striped["transfers"] if t.operation == operation] == [
                t for t in original["transfers"] if t.operation == operation
            ]
    check_owners(striped, 11, dispatch, combine)


@pytest.mark.parametrize("dispatch,combine", [(4, 6), (4, 0), (0, 6)])
def test_alltoall_audit_uses_strided_owners_and_preserves_bandwidth_denominator(dispatch, combine):
    spec, measured, audit, _ = capture(
        bs=257, rank=11, dispatch_intra=dispatch, combine_intra=combine, group_order="alltoall"
    )
    events = comm_report.read_events(measured, 11)
    transfers = comm_report.read_transfers(audit, 11)
    report = comm_report.rank_report(events, transfers=transfers, **spec)
    assert report["validation"]["group_order"] == "alltoall"
    for operation in ("DISPATCH", "COMBINE"):
        result = report["operations"][operation]
        assert result["rows"] == 257 * 8
        assert result["gbps"] == result["bytes"] / (result["sum_core_us"] / 28) / 1000
    with pytest.raises(ValueError, match="fidelity mismatch"):
        comm_report.rank_report(events, transfers=transfers, **{**spec, "group_order": "reverse"})


def test_alltoall_small_sparse_routes_do_not_drop_empty_peer_boundaries():
    schedule = wave_oracle.plan(17, 3584, 1280, 384, 8, 7, 1, 27, group_order="alltoall")
    baseline = wave_oracle.plan(17, 3584, 1280, 384, 8, 7)
    assert Counter(map(payload_key, schedule["transfers"])) == Counter(map(payload_key, baseline["transfers"]))
    check_owners(schedule, 7, 1, 27)
