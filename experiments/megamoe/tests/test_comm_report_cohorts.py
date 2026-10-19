# Copyright (c) 2026, SimpleBright_Man

from copy import deepcopy

import pytest

from comm_report_fixtures import capture
from deep_ep_experimental.megamoe import comm_report


def paired_capture():
    spec, measured, audit, _ = capture(bs=2048, rank=8, dispatch_intra=12, combine_intra=12, group_order="paired-xor")
    return spec, comm_report.read_events(measured, 8), comm_report.read_transfers(audit, 8)


def assert_cohorts(report, events, hidden):
    for operation in comm_report.OPERATIONS:
        selected = [event for event in events if event["family"] == f"AIV_{operation}_COMM"]
        cohort_metrics = report["operations"][operation]["cohorts"]
        for name, first, stop in (("INTRA", 0, 12), ("CROSS", 12, 28)):
            owned = [event for event in selected if first <= event["core"] < stop]
            rows = sum(event["row_count"] for event in owned)
            duration = sum((event["end_cycle"] - event["start_cycle"]) / 1000 for event in owned)
            size = rows * comm_report.payload_bytes(hidden, operation)
            result = cohort_metrics[name]
            assert result["workers"] == stop - first
            assert result["rows"] == rows
            assert result["bytes"] == size
            assert result["sum_core_us"] == duration
            assert result["mean_core_us"] == duration / (stop - first)
            assert result["gbps"] == size / (duration / (stop - first)) / 1000
            assert result["gbps"] != size / (duration / 28) / 1000
        assert cohort_metrics["INTRA"]["bytes"] != cohort_metrics["CROSS"]["bytes"]
        assert cohort_metrics["INTRA"]["mean_core_us"] != cohort_metrics["CROSS"]["mean_core_us"]
        for wave in [wave for wave in report["waves"] if wave["operation"] == operation]:
            for name, first, stop in (("INTRA", 0, 12), ("CROSS", 12, 28)):
                cores = [core for core in wave["cores"] if first <= core["core"] < stop]
                result = wave["cohorts"][name]
                assert result["rows"] == sum(core["rows"] for core in cores)
                assert result["bytes"] == sum(core["bytes"] for core in cores)
                assert result["sum_core_us"] == sum(core["duration_us"] for core in cores)
                assert result["mean_core_us"] == result["sum_core_us"] / (stop - first)


def test_strict_and_timer_cohorts_use_unequal_actual_rows_and_core_times():
    spec, events, transfers = paired_capture()
    strict = comm_report.rank_report(events, transfers=transfers, **spec)
    timer = comm_report.timer_rank_report(events, 8, spec["hidden"], dispatch_intra=12, combine_intra=12)
    assert_cohorts(strict, events, spec["hidden"])
    assert_cohorts(timer, events, spec["hidden"])
    for operation in comm_report.OPERATIONS:
        assert strict["operations"][operation]["cohorts"] == timer["operations"][operation]["cohorts"]

    # Alter the measured workload without changing the balanced-route oracle.
    # Performance-only cohort bytes must follow the recorded rows instead.
    changed = deepcopy(events)
    changed[0]["row_count"] += 37
    plain = comm_report.timer_rank_report(changed, 8, spec["hidden"])
    before = deepcopy(plain)
    result = comm_report.add_cohort_metrics(plain, 12, 12)
    assert result is plain
    assert_cohorts(result, changed, spec["hidden"])
    for operation in comm_report.OPERATIONS:
        assert {key: value for key, value in result["operations"][operation].items() if key != "cohorts"} == before[
            "operations"
        ][operation]
    assert all("cohorts" not in wave for wave in before["waves"])


def test_cohort_summary_aggregates_16_ranks_and_rejects_mixed_core_counts():
    spec, events, _ = paired_capture()
    records = []
    for rank in range(16):
        measured = [dict(event, rank=rank, row_count=event["row_count"] * (rank + 1)) for event in events]
        records.append(
            comm_report.timer_rank_report(measured, rank, spec["hidden"], dispatch_intra=12, combine_intra=12)
        )
    summary = comm_report.summarize(records, require_validation=False)
    for operation in comm_report.OPERATIONS:
        for name, workers in (("INTRA", 12), ("CROSS", 16)):
            base = records[0]["operations"][operation]["cohorts"][name]["gbps"]
            actual = summary["operations"][operation]["cohorts"][name]
            assert actual["workers"] == workers
            assert actual["gbps_per_card_mean"] == pytest.approx(base * 8.5)
            assert actual["gbps_per_card_min"] == pytest.approx(base)
            assert actual["gbps_per_card_max"] == pytest.approx(base * 16)
    invalid = deepcopy(records)
    invalid[7]["operations"]["COMBINE"]["cohorts"]["CROSS"]["workers"] = 15
    with pytest.raises(ValueError, match="same COMBINE/CROSS core count"):
        comm_report.summarize(invalid, require_validation=False)


def test_only_configured_operation_gets_cohort_metrics():
    spec, events, _ = paired_capture()
    report = comm_report.timer_rank_report(events, 8, spec["hidden"], dispatch_intra=12)
    assert "cohorts" in report["operations"]["DISPATCH"]
    assert "cohorts" not in report["operations"]["COMBINE"]
    with pytest.raises(ValueError, match="INTRA core counts"):
        comm_report.add_cohort_metrics(report, 12, 28)
