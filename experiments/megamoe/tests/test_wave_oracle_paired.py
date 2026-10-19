# Copyright (c) 2026, SimpleBright_Man

from collections import Counter, defaultdict

import pytest

from comm_report_fixtures import capture
from deep_ep_experimental.megamoe import comm_report, wave_oracle


# Independent, explicit perfect matchings of four groups in the three phases.
PARTNERS = ((1, 2, 3), (0, 3, 2), (3, 0, 1), (2, 1, 0))


@pytest.mark.parametrize(
    "bs,rank,dispatch,combine",
    [
        (2048, 0, 4, 4),
        (2048, 4, 6, 6),
        (2048, 8, 12, 12),
        (2048, 12, 1, 27),
        (4096, 11, 4, 6),
        (8192, 11, 27, 1),
        (8192, 0, 6, 6),
    ],
)
def test_paired_order_preserves_all_wave_tokens_and_balances_remainders(bs, rank, dispatch, combine):
    args = (bs, 3584, 1280, 384, 8, rank, dispatch, combine)
    paired = wave_oracle.plan(*args, group_order="paired-xor")
    reverse = wave_oracle.plan(*args)
    assert paired["slices"] == reverse["slices"]
    assert paired["wave_count"] == reverse["wave_count"]

    def without_owner(t):
        return (t.operation, t.wave, t.expert, t.row, t.source_rank, t.source_token, t.topk_slot)

    assert Counter(map(without_owner, paired["transfers"])) == Counter(map(without_owner, reverse["transfers"]))
    if dispatch == combine:
        assert [t[1:] for t in paired["transfers"] if t.operation == "DISPATCH"] == [
            t[1:] for t in paired["transfers"] if t.operation == "COMBINE"
        ]
    by_task = defaultdict(list)
    by_group = defaultdict(Counter)
    own = rank // 4
    splits = dict(DISPATCH=dispatch, COMBINE=combine)
    for t in paired["transfers"]:
        group = t.source_rank // 4
        assert (t.core < splits[t.operation]) == (group == own)
        by_task[t.operation, t.wave, t.expert, t.core].append(group)
        by_group[t.operation, t.wave, t.expert, group][t.core] += 1
    for (operation, _wave, _expert, core), groups in by_task.items():
        phases = [0 if group == own else PARTNERS[own].index(group) + 1 for group in groups]
        assert phases == sorted(phases)
        assert core < splits[operation] or all(group != own for group in groups)
    for (operation, _wave, _expert, group), counts in by_group.items():
        split = splits[operation]
        workers = range(split) if group == own else range(split, 28)
        loads = [counts[core] for core in workers]
        assert max(loads) - min(loads) <= 1
    for phase in range(3):
        destinations = [PARTNERS[group][phase] for group in range(4)]
        assert sorted(destinations) == list(range(4))
        assert all(PARTNERS[destinations[group]][phase] == group for group in range(4))


@pytest.mark.parametrize("dispatch,combine", [(0, 0), (0, 12), (12, 0)])
def test_paired_rejects_unsplit_operations(dispatch, combine):
    with pytest.raises(ValueError, match="paired-xor requires EP16"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 0, dispatch, combine, group_order="paired-xor")


def test_paired_rejects_unknown_order_and_non_ep16_rank():
    with pytest.raises(ValueError, match="group_order must"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 0, 12, 12, group_order="other")
    with pytest.raises(ValueError, match="Invalid EP16"):
        wave_oracle.plan(2048, 3584, 1280, 384, 8, 16, 12, 12, group_order="paired-xor")


@pytest.mark.parametrize("dispatch,combine", [(4, 4), (6, 6), (12, 12), (1, 27), (27, 1), (4, 6)])
def test_report_validates_selected_order_without_changing_bandwidth_formula(dispatch, combine):
    spec, measured, audit, _ = capture(
        bs=2048,
        rank=8,
        dispatch_intra=dispatch,
        combine_intra=combine,
        group_order="paired-xor",
    )
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
