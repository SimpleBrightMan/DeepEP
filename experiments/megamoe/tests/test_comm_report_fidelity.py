# Copyright (c) 2026, SimpleBright_Man

from collections import Counter
from copy import deepcopy
import json

import pytest

from deep_ep_experimental.megamoe import comm_report, wave_oracle
from deep_ep_experimental.megamoe.timer import protocol as p
from comm_report_fixtures import capture


def report_capture(**kwargs):
    spec, measured, audit, plan = capture(**kwargs)
    events = comm_report.read_events(measured, spec["rank"])
    transfers = comm_report.read_transfers(audit, spec["rank"])
    return comm_report.rank_report(events, transfers=transfers, **spec), spec, events, transfers, plan


@pytest.mark.parametrize(
    "bs,waves,dispatch_bytes,combine_bytes",
    [
        (2048, 6, 60817408, 117440512),
        (4096, 12, 121634816, 234881024),
        (8192, 22, 243269632, 469762048),
    ],
)
@pytest.mark.parametrize("rank", [0, 11, 15])
def test_default_shapes_exact_wave_tasks_and_separate_mean_core_rates(bs, waves, dispatch_bytes, combine_bytes, rank):
    report, _, events, _, _ = report_capture(bs=bs, rank=rank)
    assert report["validation"]["wave_count"] == waves
    assert report["validation"]["m_groups_per_wave"] == 12
    assert report["validation"]["audited_transfers"] == bs * 8 * 2
    for operation, size in (("DISPATCH", dispatch_bytes), ("COMBINE", combine_bytes)):
        value = report["operations"][operation]
        duration = sum(
            (e["end_cycle"] - e["start_cycle"]) / 1000 for e in events if e["family"] == f"AIV_{operation}_COMM"
        )
        assert value["bytes"] == size
        assert value["rows"] == bs * 8
        assert value["sum_core_us"] == duration
        assert value["mean_core_us"] == duration / 28
        assert value["gbps"] == size / (duration / 28) / 1000
    for wave in report["waves"]:
        assert len(wave["cores"]) == 28
        assert wave["sum_core_us"] == sum(c["duration_us"] for c in wave["cores"])
        assert wave["gbps"] == wave["bytes"] / (wave["sum_core_us"] / 28) / 1000
        assert wave["rows"] == sum(c["rows"] for c in wave["cores"])


@pytest.mark.parametrize("dispatch,combine", [(0, 0), (1, 0), (10, 0), (12, 0), (27, 0), (0, 16), (10, 12)])
def test_independent_split_preserves_original_wave_identity(dispatch, combine):
    report, spec, _, _, plan = report_capture(bs=2048, rank=11, dispatch_intra=dispatch, combine_intra=combine)
    base = wave_oracle.plan(**{**spec, "dispatch_intra": 0, "combine_intra": 0})

    def ignore_core(row):
        return (row.operation, row.wave, row.expert, row.row, row.source_rank, row.source_token, row.topk_slot)

    assert Counter(map(ignore_core, plan["transfers"])) == Counter(map(ignore_core, base["transfers"]))
    assert report["validation"]["strict_wave_core"]
    if not combine:
        assert [r for r in plan["transfers"] if r.operation == "COMBINE"] == [
            r for r in base["transfers"] if r.operation == "COMBINE"
        ]


def test_wave_8192_crosses_expert_boundaries_and_original_core_rotation():
    plan = wave_oracle.plan(8192, 3584, 1280, 384, 8, 0)
    assert [(s["wave"], s["expert"], s["row_start"], s["row_count"]) for s in plan["slices"][:4]] == [
        (0, 0, 0, 2731),
        (0, 1, 0, 256),
        (1, 1, 256, 2475),
        (1, 2, 0, 512),
    ]
    first = wave_oracle.plan(2048, 3584, 1280, 384, 8, 0)
    rows = [r for r in first["transfers"] if r.operation == "DISPATCH" and r.wave == 0 and r.expert == 0]
    assert [r.row for r in rows if r.core == 0] == list(range(25))
    assert [r.source_rank for r in rows if r.core == 0] == [0] * 25
    assert [r.source_token for r in rows if r.core == 0] == list(range(0, 25 * 48, 48))


@pytest.mark.parametrize(
    "field", ["wave", "core", "expert_id", "row_start", "source_rank", "source_token", "topk_slot"]
)
def test_one_wrong_actual_identity_rejected_even_when_total_bytes_match(field):
    _, spec, events, transfers, _ = report_capture()
    wrong = deepcopy(transfers)
    wrong[0][field] += 1
    with pytest.raises(ValueError, match="fidelity mismatch"):
        comm_report.rank_report(events, transfers=wrong, **spec)


def test_missing_duplicate_audit_and_wrong_timer_budget_rejected():
    _, spec, events, transfers, _ = report_capture()
    for invalid in (transfers[:-1], [*transfers, transfers[0]], [*transfers[:-1], transfers[0]]):
        with pytest.raises(ValueError, match="fidelity mismatch"):
            comm_report.rank_report(events, transfers=invalid, **spec)
    events[0]["m_groups_per_wave"] += 1
    with pytest.raises(ValueError, match="wave budget"):
        comm_report.rank_report(events, transfers=transfers, **spec)


def test_wrong_measured_wave_core_assignment_and_overlap_rejected():
    _, spec, events, transfers, _ = report_capture()
    wrong = deepcopy(events)
    wrong[0]["row_count"] += 1
    with pytest.raises(ValueError):
        comm_report.rank_report(wrong, transfers=transfers, **spec)
    wrong = deepcopy(events)
    second = next(e for e in wrong[1:] if e["core"] == wrong[0]["core"])
    second["start_cycle"], second["end_cycle"] = wrong[0]["start_cycle"], wrong[0]["end_cycle"]
    with pytest.raises(ValueError, match="overlapping"):
        comm_report.rank_report(wrong, transfers=transfers, **spec)


def test_audit_instrumented_timing_cannot_contaminate_bandwidth():
    _, spec, events, transfers, _ = report_capture()
    events[0]["audit_enabled"] = True
    with pytest.raises(ValueError, match="timer-only"):
        comm_report.rank_report(events, transfers=transfers, **spec)


def test_native_memset_enabled_flag_is_nonzero_not_only_integer_one():
    _, _, audit, plan = capture()
    audit.values[p.COMM_AUDIT_ENABLE_OFFSET] = 0x0101010101010101
    assert len(comm_report.read_transfers(audit, 0)) == len(plan["transfers"])


def test_core_swap_with_unchanged_bytes_and_counts_rejected():
    _, spec, events, transfers, _ = report_capture()
    wrong = deepcopy(transfers)
    first = wrong[0]
    other = next(
        r
        for r in wrong
        if r["operation"] == first["operation"]
        and r["wave"] == first["wave"]
        and r["expert_id"] == first["expert_id"]
        and r["core"] != first["core"]
    )
    first["core"], other["core"] = other["core"], first["core"]
    with pytest.raises(ValueError, match="fidelity mismatch"):
        comm_report.rank_report(events, transfers=wrong, **spec)


def test_different_splits_have_identical_per_wave_byte_totals_and_logical_hash():
    base, *_ = report_capture(bs=4096)
    split, *_ = report_capture(bs=4096, dispatch_intra=10)
    assert base["validation"]["logical_wave_sha256"] == split["validation"]["logical_wave_sha256"]
    assert [(w["operation"], w["wave"], w["bytes"]) for w in base["waves"]] == [
        (w["operation"], w["wave"], w["bytes"]) for w in split["waves"]
    ]


def test_inactive_workers_are_zero_but_denominator_is_always_28(tmp_path):
    report, _, _, transfers, _ = report_capture(bs=1)
    wave = report["waves"][0]
    assert any(c["rows"] == 0 and c["duration_us"] == 0 for c in wave["cores"])
    assert wave["workers"] == 28
    compact = comm_report.write_rank_detail(report, tmp_path)
    assert len(json.dumps(compact)) < 10000
    assert "waves" not in compact
    data = json.loads((tmp_path / "rank_00_communication.json").read_text())
    assert data["waves"][0]["cores"] == wave["cores"]
    assert len((tmp_path / "rank_00_communication.csv").read_text().splitlines()) == len(transfers) + 1


def test_trace_overflow_and_legacy_only_timing_are_rejected():
    spec, measured, audit, _ = capture()
    audit.values[p.COMM_AUDIT_COUNT_OFFSET + 2] = p.COMM_AUDIT_MAX_RECORDS_PER_CORE + 1
    with pytest.raises(ValueError, match="overflow"):
        comm_report.read_transfers(audit, 0)
    with pytest.raises(ValueError, match="separate actual-transfer audit"):
        comm_report.rank_report(comm_report.read_events(measured, 0), **spec)
