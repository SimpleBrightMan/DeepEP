# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU-only checks for native timing reduction, malformed output and dry-run launch."""

import copy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.dispatch_bandwidth import report, run  # noqa: E402


def brute_traffic_rows(rank, world, group_size, batch, experts, topk, scope):
    rows = dict(local=0, group=0, cross=0)
    for peer in range(world):
        traffic = "local" if peer == rank else "group" if peer // group_size == rank // group_size else "cross"
        for index in range(batch * topk):
            route = peer * batch * topk + index
            if route % world == rank and (scope == "all" or (route // world) % (experts // world) == 0):
                rows[traffic] += 1
    return rows


def native_report(rank=0, *, repeats=2, iterations=2, clock_offset=0, batch=5, topk=2):
    traffic_rows = brute_traffic_rows(rank, 4, 2, batch, 32, topk, "all")
    specs = [("local", 0, 100, 1100), ("group", 0, 300, 2300), ("cross", 0, 1000, 3000), ("cross", 1, 6000, 8000)]
    # Tiny fixture: independently distribute rows cyclically rather than calling the reporter's planner.
    allocation = {}
    for traffic, count in (("local", 1), ("group", 1), ("cross", 2)):
        allocation[traffic] = [0] * count
        for row in range(traffic_rows[traffic]):
            allocation[traffic][row % count] += 1
    rounds = []
    for iteration in range(iterations):
        base = clock_offset + iteration * 100000
        workers = [
            dict(
                group_id=group,
                traffic=traffic,
                rows=allocation[traffic][worker] * repeats,
                start_cycle=base + start,
                end_cycle=base + end,
            )
            for group, (traffic, worker, start, end) in enumerate(specs)
        ]
        rounds.append(dict(iteration=iteration, kernel_elapsed_us=12.0, workers=workers))
    return dict(
        schema_version=1,
        rank=rank,
        world=4,
        device=rank,
        cores=4,
        group_size=2,
        batch=batch,
        hidden=3584,
        intermediate=1280,
        experts=32,
        topk=topk,
        scope="all",
        row_bytes=3712,
        scale_elements=112,
        buffer_count=6,
        repeats=repeats,
        cycles_per_us=1000.0,
        traffic_rows=traffic_rows,
        worker_counts=dict(local=1, group=1, cross=2),
        validation=dict(passed=True, checked_rows=sum(traffic_rows.values())),
        rounds=rounds,
    )


def test_class_envelope_uses_all_worker_start_and_end_times():
    measured = report.rank_measurements(native_report())[0]["classes"]["cross"]
    # Two 2us workers with a 3us gap have a 7us class envelope, not 2us or 4us.
    assert measured["envelope_us"] == 7.0
    assert measured["bytes"] == 5 * 2 * 3712
    assert measured["gbps"] == pytest.approx(5 * 2 * 3712 / 7000)
    assert measured["kernel_window_gbps"] == pytest.approx(5 * 2 * 3712 / 12000)


def test_padding_and_repeats_are_in_the_payload_numerator():
    first = report.rank_measurements(native_report(repeats=2))[0]["classes"]["cross"]
    second = report.rank_measurements(native_report(repeats=3))[0]["classes"]["cross"]
    assert first["rows"] == 10
    assert first["bytes"] != 10 * (3584 + 112)  # Scale storage is padded to 128 bytes.
    assert second["gbps"] == pytest.approx(first["gbps"] * 1.5)
    malformed = native_report()
    malformed["row_bytes"] = 3584 + 112
    with pytest.raises(report.ReportError, match="padded"):
        report.validate_report(malformed)


def test_rank_clock_offsets_never_enter_cross_rank_bandwidth():
    # Also catches converting uint64 timestamps to imprecise floats before subtraction.
    raws = [native_report(rank, clock_offset=2**61 + rank * 1000000000000) for rank in range(4)]
    summary = report.summarize_reports(raws, expected_world=4, expected_iterations=2)
    cross = summary["classes"]["cross"]
    assert cross["rank_mean_gbps_per_card"]["min"] == pytest.approx(37120 / 7000)
    assert cross["rank_mean_gbps_per_card"]["max"] == pytest.approx(37120 / 7000)
    assert cross["envelope_us_per_card"]["mean"] == 7.0
    assert summary["successful_ranks"] == 4
    assert summary["successful_rank_rounds"] == 8


def test_rounds_keep_rank_min_mean_max_and_percentiles():
    raws = [native_report(rank) for rank in range(4)]
    # Rank 3 takes twice as long in round 1; rank starts remain unrelated.
    for worker in raws[3]["rounds"][1]["workers"]:
        worker["start_cycle"] *= 2
        worker["end_cycle"] *= 2
    raws[3]["rounds"][1]["kernel_elapsed_us"] *= 2
    summary = report.summarize_reports(raws)
    rates = summary["per_round"][1]["classes"]["cross"]["gbps_per_card"]
    baseline = 37120 / 7000
    assert rates["min"] == pytest.approx(baseline / 2)
    assert rates["max"] == pytest.approx(baseline)
    assert rates["mean"] == pytest.approx(baseline * 0.875)
    assert summary["classes"]["cross"]["round_rank_gbps"]["mean"]["p50"] == pytest.approx(baseline * 0.9375)
    assert summary["classes"]["cross"]["round_rank_gbps"]["mean"]["p95"] == pytest.approx(baseline * 0.99375)


def test_zero_traffic_has_no_artificial_bandwidth_observation():
    raws = [native_report(rank, batch=1, topk=1) for rank in range(4)]
    for raw in raws:
        for measured in raw["rounds"]:
            for worker in measured["workers"]:
                if worker["traffic"] == "cross":
                    worker.update(rows=0, start_cycle=0, end_cycle=0)
    summary = report.summarize_reports(raws)
    assert summary["classes"]["cross"]["gbps_per_card"] is None
    assert summary["classes"]["cross"]["envelope_us_per_card"] is None
    assert summary["per_round"][0]["classes"]["cross"]["active_ranks"] == 0


@pytest.mark.parametrize("scope", ["all", "first-expert"])
@pytest.mark.parametrize("world,group_size,experts", [(4, 2, 32), (16, 4, 384)])
@pytest.mark.parametrize("batch,topk", [(1, 1), (5, 2), (17, 8), (96, 6)])
def test_residue_counts_match_an_independent_route_enumerator(scope, world, group_size, experts, batch, topk):
    for rank in range(world):
        raw = dict(rank=rank, world=world, group_size=group_size, experts=experts, batch=batch, topk=topk, scope=scope)
        expected = brute_traffic_rows(rank, world, group_size, batch, experts, topk, scope)
        assert report.balanced_traffic_rows(raw) == expected
        if scope == "all":
            assert sum(expected.values()) == batch * topk


@pytest.mark.parametrize("batch,first", [(2048, (43, 128, 512)), (4096, (86, 256, 1024)), (8192, (171, 512, 2048))])
def test_default_ep16_counts_for_full_dispatch_and_first_expert(batch, first):
    raw = dict(rank=0, world=16, group_size=4, experts=384, batch=batch, topk=8, scope="first-expert")
    assert report.balanced_traffic_rows(raw) == dict(zip(report.TRAFFIC, first))
    raw["scope"] = "all"
    assert report.balanced_traffic_rows(raw) == dict(local=batch // 2, group=batch * 3 // 2, cross=batch * 6)


@pytest.mark.parametrize(
    "fault", ["self_consistent_wrong_count", "wrong_scope", "swapped_worker_class", "unbalanced_worker_rows"]
)
def test_report_must_match_routes_and_fixed_worker_partitions(fault):
    raw = native_report()
    if fault == "self_consistent_wrong_count":
        raw["traffic_rows"]["local"] += 1
        raw["validation"]["checked_rows"] += 1
        for measured in raw["rounds"]:
            measured["workers"][0]["rows"] += raw["repeats"]
    elif fault == "wrong_scope":
        raw["scope"] = "first-expert"
    elif fault == "swapped_worker_class":
        for measured in raw["rounds"]:
            first, second = measured["workers"][:2]
            first["traffic"], second["traffic"] = second["traffic"], first["traffic"]
            first["rows"], second["rows"] = second["rows"], first["rows"]
    else:
        for measured in raw["rounds"]:
            measured["workers"][2]["rows"] += raw["repeats"]
            measured["workers"][3]["rows"] -= raw["repeats"]
    with pytest.raises(report.ReportError, match="balanced routing|planned worker partition"):
        report.validate_report(raw)


def test_timer_acl_upper_bound_and_recorded_consistency_diagnostic():
    raw = native_report()
    measured = report.rank_measurements(raw)[0]
    assert measured["timer_envelope_us"] == 7.9
    assert measured["timer_to_acl_ratio"] == pytest.approx(7.9 / 12)
    summary = report.summarize_reports([native_report(rank) for rank in range(4)])
    assert summary["timing_validation"]["status"] == "consistent"
    assert summary["timing_validation"]["timer_to_acl_ratio"]["mean"] == pytest.approx(7.9 / 12)
    raw["rounds"][0]["kernel_elapsed_us"] = 6.0  # 7.9us <= 6us + 2us small-interval tolerance.
    report.validate_report(raw)
    raw["rounds"][0]["kernel_elapsed_us"] = 5.0
    with pytest.raises(report.ReportError, match="timer envelope.*ACL"):
        report.validate_report(raw)


def test_timer_acl_relative_tolerance_boundary_and_wrong_counter_conversion():
    raw = native_report()
    raw["rounds"][0]["kernel_elapsed_us"] = 100.0
    raw["rounds"][0]["workers"][3]["end_cycle"] = 110100
    report.validate_report(raw)  # Exactly 110us timer envelope, with 10us tolerance.
    raw["rounds"][0]["workers"][3]["end_cycle"] += 1000
    with pytest.raises(report.ReportError, match="counter conversion"):
        report.validate_report(raw)
    raw = native_report()
    raw["cycles_per_us"] = 50.0
    with pytest.raises(report.ReportError, match="timer envelope"):
        report.validate_report(raw)


@pytest.mark.parametrize(
    "fault", ["missing_rank", "duplicate_rank", "duplicate_device", "different_config", "partial_rounds"]
)
def test_incomplete_or_mismatched_rank_sets_are_rejected(fault):
    raws = [native_report(rank) for rank in range(4)]
    if fault == "missing_rank":
        raws.pop()
    elif fault == "duplicate_rank":
        raws[3] = copy.deepcopy(raws[2])
    elif fault == "duplicate_device":
        raws[3]["device"] = 0
    elif fault == "different_config":
        raws[3]["batch"] = 19
    else:
        raws[3]["rounds"].pop()
    with pytest.raises(report.ReportError):
        report.summarize_reports(raws, expected_world=4, expected_iterations=2)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_field",
        "invalid_schema",
        "failed_validation",
        "unchecked_rows",
        "missing_core",
        "duplicate_core",
        "bad_traffic",
        "wrong_rows",
        "bad_cycle",
        "duplicate_iteration",
        "missing_iteration",
        "bad_frequency",
        "nonfinite_kernel",
    ],
)
def test_malformed_native_records_fail_closed(fault):
    raw = native_report()
    if fault == "missing_field":
        del raw["row_bytes"]
    elif fault == "invalid_schema":
        raw["schema_version"] = True
    elif fault == "failed_validation":
        raw["validation"]["passed"] = False
    elif fault == "unchecked_rows":
        raw["validation"]["checked_rows"] = 1
    elif fault == "missing_core":
        raw["rounds"][0]["workers"].pop()
    elif fault == "duplicate_core":
        raw["rounds"][0]["workers"][3]["group_id"] = 2
    elif fault == "bad_traffic":
        raw["rounds"][0]["workers"][0]["traffic"] = "other"
    elif fault == "wrong_rows":
        raw["rounds"][0]["workers"][0]["rows"] += 1
    elif fault == "bad_cycle":
        raw["rounds"][0]["workers"][0]["end_cycle"] = 1
    elif fault == "duplicate_iteration":
        raw["rounds"][1]["iteration"] = 0
    elif fault == "missing_iteration":
        raw["rounds"][1]["iteration"] = 2
    elif fault == "bad_frequency":
        raw["cycles_per_us"] = 0
    else:
        raw["rounds"][0]["kernel_elapsed_us"] = float("nan")
    with pytest.raises(report.ReportError):
        report.validate_report(raw)


def test_load_reports_rejects_missing_bad_json_and_rank_identity(tmp_path):
    with pytest.raises(report.ReportError, match="Cannot read rank 0"):
        report.load_reports(tmp_path, 4)
    target = tmp_path / "rank_0.json"
    target.write_text("not json", encoding="utf-8")
    with pytest.raises(report.ReportError, match="Cannot read rank 0"):
        report.load_reports(tmp_path, 4)
    target.write_text(json.dumps(native_report(1)), encoding="utf-8")
    with pytest.raises(report.ReportError, match="identity"):
        report.load_reports(tmp_path, 4)


def test_exports_share_one_rank_origin_across_lanes_and_rounds(tmp_path):
    raw = native_report(clock_offset=2**61)
    csv_path, trace_path = report.export_timers(raw, tmp_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    assert min(event["ts"] for event in trace) == 0
    cross = next(event for event in trace if event["name"] == "round_0_CROSS" and event["tid"] == "AIV1_group_3")
    assert cross["ts"] == 5.9
    assert cross["dur"] == 2.0
    next_round = next(event for event in trace if event["name"] == "round_1_LOCAL")
    assert next_round["ts"] == 100.0
    with csv_path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 8
    assert int(records[0]["start_cycle"]) == 2**61 + 100


def test_plan_uses_no_devices_subprocess_ports_or_output_directory(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("--plan touched an execution-only operation")

    monkeypatch.setattr(run.subprocess, "run", forbidden)
    monkeypatch.setattr(run.subprocess, "Popen", forbidden)
    monkeypatch.setattr(run.socket, "socket", forbidden)
    output = tmp_path / "not-created"
    assert (
        run.main(
            [
                "--plan",
                "--binary",
                str(tmp_path / "missing-binary"),
                "--output-dir",
                str(output),
                "--devices",
                "0,1,2,3",
                "--group-size",
                "2",
                "--cores",
                "4",
                "--workers",
                "1,1,2",
                "--bs-values",
                "2048,4096,8192",
                "--timer",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert plan["mode"] == "concurrent_local_group_cross"
    assert len(plan["cases"]) == 3
    command = plan["cases"][0]["commands"][3]
    assert command[command.index("--rank") + 1] == "3"
    assert command[command.index("--device") + 1] == "3"
    assert command[command.index("--scope") + 1] == "all"
    assert "--timer" in command


@pytest.mark.parametrize(
    "arguments",
    [
        ["--workers", "2,5,20"],
        ["--bs-values", "4096,4096"],
        ["--devices", "0,0,1,2"],
        ["--repeat", "0"],
        ["--timeout", "nan"],
        ["--buffer-count", "7"],
        ["--group-size", "3"],
    ],
)
def test_invalid_launch_configuration_is_rejected(arguments):
    with pytest.raises(SystemExit):
        run.parse_args(["--plan", *arguments])


def test_cleanup_only_terminates_owned_live_process_objects():
    class Owned:
        def __init__(self, code=None):
            self.code, self.terminated = code, False

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated, self.code = True, -15

        def wait(self, timeout):
            return self.code

    completed, live = Owned(0), Owned()
    run._stop_owned([completed, live])
    assert live.terminated
    assert not completed.terminated
