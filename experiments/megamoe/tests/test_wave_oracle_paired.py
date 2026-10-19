# Copyright (c) 2026, SimpleBright_Man

from collections import Counter, defaultdict

import pytest

from comm_report_fixtures import capture
from deep_ep_experimental.megamoe import comm_report, wave_oracle


# Independent, explicit perfect matchings of four groups in the three phases.
PARTNERS = ((1, 2, 3), (0, 3, 2), (3, 0, 1), (2, 1, 0))


@pytest.mark.parametrize("bs,rank", [(2048, 0), (2048, 4), (2048, 8), (2048, 12), (4096, 11), (8192, 11)])
def test_paired_order_preserves_all_wave_tokens_and_balances_remainders(bs, rank):
    args = (bs, 3584, 1280, 384, 8, rank, 12, 12)
    paired = wave_oracle.plan(*args, group_order="paired-xor")
    reverse = wave_oracle.plan(*args)
    assert paired["slices"] == reverse["slices"]
    assert paired["wave_count"] == reverse["wave_count"]

    def without_owner(t):
        return (t.operation, t.wave, t.expert, t.row, t.source_rank, t.source_token, t.topk_slot)

    assert Counter(map(without_owner, paired["transfers"])) == Counter(map(without_owner, reverse["transfers"]))
    assert [t[1:] for t in paired["transfers"] if t.operation == "DISPATCH"] == [
        t[1:] for t in paired["transfers"] if t.operation == "COMBINE"
    ]
    by_task = defaultdict(list)
    by_group = defaultdict(Counter)
    own = rank // 4
    for t in paired["transfers"]:
        group = t.source_rank // 4
        assert (t.core < 12) == (group == own)
        by_task[t.operation, t.wave, t.expert, t.core].append(group)
        by_group[t.operation, t.wave, t.expert, group][t.core] += 1
    for (*_, core), groups in by_task.items():
        phases = [0 if group == own else PARTNERS[own].index(group) + 1 for group in groups]
        assert phases == sorted(phases)
        assert core < 12 or all(group != own for group in groups)
    for (*_, group), counts in by_group.items():
        workers = range(12) if group == own else range(12, 28)
        loads = [counts[core] for core in workers]
        assert max(loads) - min(loads) <= 1
    for phase in range(3):
        destinations = [PARTNERS[group][phase] for group in range(4)]
        assert sorted(destinations) == list(range(4))
        assert all(PARTNERS[destinations[group]][phase] == group for group in range(4))


@pytest.mark.parametrize("dispatch,combine", [(0, 0), (0, 12), (12, 0), (10, 12), (12, 16)])
def test_paired_rejects_other_core_splits(dispatch, combine):
    with pytest.raises(ValueError, match="paired-xor requires EP16"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 0, dispatch, combine, group_order="paired-xor")


def test_paired_rejects_unknown_order_and_non_ep16_rank():
    with pytest.raises(ValueError, match="group_order must"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 0, 12, 12, group_order="other")
    with pytest.raises(ValueError, match="Invalid EP16"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 16, 12, 12, group_order="paired-xor")


def test_report_validates_selected_order_without_changing_bandwidth_formula():
    spec, measured, audit, _ = capture(bs=2048, rank=8, dispatch_intra=12, combine_intra=12, group_order="paired-xor")
    events = comm_report.read_events(measured, 8)
    transfers = comm_report.read_transfers(audit, 8)
    report = comm_report.rank_report(events, transfers=transfers, **spec)
    assert report["validation"]["group_order"] == "paired-xor"
    for operation in ("DISPATCH", "COMBINE"):
        result = report["operations"][operation]
        assert result["rows"] == 2048 * 8
        assert result["gbps"] == result["bytes"] / (result["sum_core_us"] / 28) / 1000
    with pytest.raises(ValueError, match="fidelity mismatch"):
        comm_report.rank_report(events, transfers=transfers, **{**spec, "group_order": "reverse"})
