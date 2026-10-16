# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU-only checks for concurrent INTRA/CROSS timers and staggered GET ordering."""

import builtins
import copy
import csv
from functools import lru_cache
import json
from pathlib import Path
import statistics
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.dispatch_bandwidth import group_report, run  # noqa: E402


@lru_cache(maxsize=None)
def brute_peer_rows(rank, batch, experts, topk, scope):
    """Enumerate source routes, independently of the report's residue formula."""
    counts = []
    for peer in range(16):
        count = 0
        for index in range(batch * topk):
            route = peer * batch * topk + index
            if route % 16 == rank and (scope == "all" or route // 16 % (experts // 16) == 0):
                count += 1
        counts.append(count)
    return tuple(counts)


def native_group_report(
    rank=0,
    *,
    batch=4096,
    experts=384,
    topk=8,
    scope="all",
    metadata_mode="dma",
    repeats=1,
    iterations=3,
    rotation_start=47,
    clock_offset=0,
    hidden=3584,
):
    counts = brute_peer_rows(rank, batch, experts, topk, scope)
    groups = [(rank // 4 - offset) % 4 for offset in range(4)]
    phases = []
    for group in groups:
        phase = dict(local=0, group=0, cross=0)
        for peer in range(group * 4, group * 4 + 4):
            traffic = "local" if peer == rank else "group" if peer // 4 == rank // 4 else "cross"
            phase[traffic] += counts[peer]
        phases.append(phase)
    traffic_rows = {traffic: sum(phase[traffic] for phase in phases) for traffic in ("local", "group", "cross")}
    total_rows = sum(counts)
    logical_counts = [0] * 28
    for row in range(sum(phases[0].values())):
        logical_counts[row % 12] += 1
    for row in range(sum(sum(phase.values()) for phase in phases[1:])):
        logical_counts[12 + row % 16] += 1
    rounds = []
    for iteration in range(iterations):
        rotation = (rotation_start + iteration) % 48
        workers = []
        for physical in range(28):
            logical = (physical + rotation) % 12 if physical < 12 else 12 + (physical - 12 + rotation) % 16
            amount = logical_counts[logical] * repeats
            start = clock_offset + 1_000_000 + iteration * 100_000_000 + physical * 13
            duration = amount * 2000 * (iteration + 1) + physical * 10 if amount else 0
            workers.append(
                dict(
                    group_id=physical,
                    traffic="intra" if physical < 12 else "cross",
                    rows=amount,
                    start_cycle=start,
                    end_cycle=start + duration,
                )
            )
        active = [worker for worker in workers if worker["rows"]]
        envelope = (
            max(worker["end_cycle"] for worker in active) - min(worker["start_cycle"] for worker in active)
        ) / 1000.0
        rounds.append(
            dict(
                iteration=iteration,
                worker_rotation=rotation,
                kernel_elapsed_us=envelope + 5.0,
                timer_envelope_us=envelope,
                workers=workers,
            )
        )
    scale = hidden // 32
    return dict(
        schema_version=5,
        mode="group_round_robin",
        data_kind="synthetic_test",
        rank=rank,
        device=rank,
        world=16,
        cores=28,
        worker_counts={"intra": 12, "cross": 16},
        group_size=4,
        batch=batch,
        hidden=hidden,
        intermediate=1280,
        experts=experts,
        topk=topk,
        scope=scope,
        metadata_mode=metadata_mode,
        row_bytes=(hidden + 255) // 256 * 256 + (scale + 31) // 32 * 32,
        scale_elements=scale,
        buffer_count=6,
        repeats=repeats,
        phase_count=4,
        phase_peer_groups=groups,
        phase_traffic_rows=phases,
        cycles_per_us=1000.0,
        traffic_rows=traffic_rows,
        validation=dict(passed=True, checked_rows=total_rows),
        rounds=rounds,
    )


@pytest.mark.parametrize("metadata_mode", ["scalar", "dma", "preload"])
@pytest.mark.parametrize("rotation_start", [0, 11, 15, 47])
def test_native_schema_modes_and_wrapping_rotation(metadata_mode, rotation_start):
    raw = native_group_report(metadata_mode=metadata_mode, rotation_start=rotation_start, repeats=2)
    assert group_report.validate_report(raw, expected_iterations=3) is raw
    assert raw["worker_counts"] == {"intra": 12, "cross": 16}
    assert [row["worker_rotation"] for row in raw["rounds"]] == [(rotation_start + i) % 48 for i in range(3)]
    for measured in raw["rounds"]:
        assert len(measured["workers"]) == 28
        for worker in measured["workers"]:
            physical, rotation = worker["group_id"], measured["worker_rotation"]
            logical = (physical + rotation) % 12 if physical < 12 else (physical - 12 + rotation) % 16
            expected = (682 + int(logical < 8)) if physical < 12 else 1536
            assert worker["rows"] == expected * 2
            assert worker["traffic"] == ("intra" if physical < 12 else "cross")
        assert sum(worker["rows"] for worker in measured["workers"]) == 32768 * 2


@pytest.mark.parametrize("batch", [2048, 4096, 8192])
@pytest.mark.parametrize("rank", [0, 4, 8, 12])
@pytest.mark.parametrize("scope", ["all", "first-expert"])
def test_source_prefixed_route_counts_match_independent_enumeration(batch, rank, scope):
    raw = native_group_report(rank, batch=batch, scope=scope)
    group_report.validate_report(raw)
    assert group_report.peer_rows(raw) == list(brute_peer_rows(rank, batch, 384, 8, scope))
    expected_total = batch * 8 if scope == "all" else {2048: 683, 4096: 1366, 8192: 2731}[batch]
    assert raw["validation"]["checked_rows"] == expected_total
    assert sum(raw["traffic_rows"].values()) == expected_total


@pytest.mark.parametrize("rank,groups", [(0, [0, 3, 2, 1]), (7, [1, 0, 3, 2]), (11, [2, 1, 0, 3]), (15, [3, 2, 1, 0])])
def test_group_order_has_get_direction_without_phase_timers(rank, groups):
    raw = native_group_report(rank)
    assert raw["phase_peer_groups"] == groups
    summary = group_report.summarize_reports([native_group_report(rank) for rank in range(16)])
    assert len(summary["phase_diagnostics"]) == 4
    for phase in summary["phase_diagnostics"]:
        assert set(phase) == {"phase", "bytes_per_round_per_card"}
        assert phase["bytes_per_round_per_card"]["mean"] == 8192 * 3712
    assert "phase_time" not in json.dumps(summary)


def test_class_envelopes_are_independent_and_total_contributions_exclude_metadata_reads():
    raw = native_group_report(repeats=2)
    for measured, reduced in zip(raw["rounds"], group_report.rank_measurements(raw)):
        envelope = (
            max(worker["end_cycle"] for worker in measured["workers"])
            - min(worker["start_cycle"] for worker in measured["workers"])
        ) / 1000
        assert reduced["envelope_us"] == envelope
        assert reduced["bytes"] == 32768 * 2 * 3712
        assert reduced["bytes"] != 32768 * 2 * (3712 + 32)
        assert reduced["gbps"] == pytest.approx(reduced["bytes"] / (envelope * 1000))
        assert sum(value["shared_timer_window_gbps"] for value in reduced["classes"].values()) == pytest.approx(
            reduced["gbps"]
        )
        assert sum(value["single_acl_window_gbps"] for value in reduced["classes"].values()) == pytest.approx(
            reduced["single_acl_window_gbps"]
        )
        for traffic, fraction in (("intra", 1 / 4), ("cross", 3 / 4)):
            assert reduced["classes"][traffic]["shared_timer_window_gbps"] == pytest.approx(reduced["gbps"] * fraction)
            workers = [worker for worker in measured["workers"] if worker["traffic"] == traffic]
            class_span = (
                max(worker["end_cycle"] for worker in workers) - min(worker["start_cycle"] for worker in workers)
            ) / 1000
            assert reduced["classes"][traffic]["envelope_us"] == class_span
            assert reduced["classes"][traffic]["gbps"] == pytest.approx(
                reduced["bytes"] * fraction / (class_span * 1000)
            )
        assert sum(value["gbps"] for value in reduced["classes"].values()) > reduced["gbps"]


def test_summary_averages_observed_rates_instead_of_inverting_mean_time():
    reports = [native_group_report(rank) for rank in range(16)]
    summary = group_report.summarize_reports(reports, expected_world=16, expected_iterations=3)
    per_rank_rates = []
    for raw in reports:
        samples = group_report.rank_measurements(raw)
        per_rank_rates.append(statistics.mean(sample["gbps"] for sample in samples))
    assert summary["total"]["rank_mean_gbps_per_card"]["mean"] == pytest.approx(statistics.mean(per_rank_rates))
    first = group_report.rank_measurements(reports[0])
    inverse_mean = first[0]["bytes"] / (statistics.mean(sample["envelope_us"] for sample in first) * 1000)
    assert per_rank_rates[0] != pytest.approx(inverse_mean)
    assert summary["successful_ranks"] == 16
    assert summary["successful_rank_rounds"] == 48
    assert summary["checked_rows"] == 16 * 32768
    assert summary["class_metric"] == "rank_local_class_payload_gbps"
    assert set(summary["classes"]) == {"intra", "cross"}
    for traffic in ("intra", "cross"):
        observed = [
            sample["classes"][traffic]["gbps"] for raw in reports for sample in group_report.rank_measurements(raw)
        ]
        assert summary["classes"][traffic]["rank_mean_gbps_per_card"]["mean"] == pytest.approx(
            statistics.mean(observed)
        )
        assert summary["classes"][traffic]["workers"] == (12 if traffic == "intra" else 16)


def test_moving_one_class_does_not_charge_its_idle_offset_to_its_bandwidth():
    raw = native_group_report(iterations=1)
    before = group_report.rank_measurements(raw)[0]
    measured = raw["rounds"][0]
    # Class envelopes measure their own active intervals. Moving all CROSS
    # intervals later grows TOTAL, but leaves both class durations unchanged.
    for worker in measured["workers"]:
        if worker["traffic"] == "cross":
            worker["start_cycle"] += 10_000_000
            worker["end_cycle"] += 10_000_000
    measured["timer_envelope_us"] += 10_000
    measured["kernel_elapsed_us"] += 10_000
    after = group_report.rank_measurements(raw)[0]
    assert after["gbps"] < before["gbps"]
    for traffic in ("intra", "cross"):
        assert after["classes"][traffic]["gbps"] == before["classes"][traffic]["gbps"]
        assert after["classes"][traffic]["envelope_us"] == before["classes"][traffic]["envelope_us"]
        assert (
            after["classes"][traffic]["shared_timer_window_gbps"]
            < before["classes"][traffic]["shared_timer_window_gbps"]
        )


def test_printed_classes_are_intra_cross_with_independent_class_envelopes(capsys):
    summary = group_report.summarize_reports([native_group_report(rank) for rank in range(16)])
    group_report.print_summary(summary)
    output = capsys.readouterr().out
    assert "workers=INTRA:12/CROSS:16" in output
    assert "TOTAL GB/s/card mean=" in output
    assert "INTRA GB/s/card mean=" in output and "workers=12" in output
    assert "CROSS GB/s/card mean=" in output and "workers=16" in output
    assert output.count("class-envelope-us mean=") == 2
    assert output.count("round-mean p50/p95=") == 2
    assert "LOCAL GB/s" not in output and "GROUP GB/s" not in output
    assert "shared-TOTAL contribution" not in output


def test_rank_clock_offsets_and_input_order_do_not_change_bandwidth():
    reports = [native_group_report(rank) for rank in range(16)]
    shifted = copy.deepcopy(reports)
    for raw in shifted:
        offset = 2**60 + raw["rank"] * 10**15
        for measured in raw["rounds"]:
            for worker in measured["workers"]:
                worker["start_cycle"] += offset
                worker["end_cycle"] += offset
    baseline = group_report.summarize_reports(reports)
    actual = group_report.summarize_reports(list(reversed(shifted)))
    assert actual["total"] == baseline["total"]
    assert actual["classes"] == baseline["classes"]
    assert actual["per_rank"] == baseline["per_rank"]


def test_zero_row_workers_and_classes_are_not_fabricated_measurements():
    raw = native_group_report(batch=1, experts=16, topk=1, iterations=1, rotation_start=0)
    measured = group_report.rank_measurements(raw)[0]
    assert sum(worker["rows"] > 0 for worker in raw["rounds"][0]["workers"]) == 1
    assert measured["classes"]["intra"]["rows"] == 1
    assert measured["classes"]["cross"]["gbps"] is None
    assert measured["classes"]["cross"]["envelope_us"] is None
    assert measured["classes"]["cross"]["single_acl_window_gbps"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(schema_version=2),
        lambda raw: raw.update(schema_version=4),
        lambda raw: raw.update(mode="concurrent_local_group_cross"),
        lambda raw: raw.update(cores=27),
        lambda raw: raw.update(buffer_count=5),
        lambda raw: raw.update(phase_count=3),
        lambda raw: raw.update(metadata_mode="unknown"),
        lambda raw: raw.pop("worker_counts"),
        lambda raw: raw.update(worker_counts={"intra": 16, "cross": 12}),
        lambda raw: raw.update(worker_counts={"intra": True, "cross": 16}),
        lambda raw: raw.update(worker_counts={"local": 2, "group": 5, "cross": 21}),
        lambda raw: raw.pop("phase_traffic_rows"),
        lambda raw: raw.update(phase_peer_groups=[0, 1, 2, 3]),
        lambda raw: raw["phase_traffic_rows"][0].update(local=2049),
        lambda raw: raw["traffic_rows"].update(cross=24575),
        lambda raw: raw["validation"].update(checked_rows=32767),
        lambda raw: raw["validation"].update(passed=False),
        lambda raw: raw["rounds"][0].update(worker_rotation=48),
        lambda raw: raw["rounds"][1].update(worker_rotation=47),
        lambda raw: raw["rounds"][1].update(iteration=0),
        lambda raw: raw["rounds"][0].update(timer_envelope_us=1),
        lambda raw: raw["rounds"][0].update(kernel_elapsed_us=1),
        lambda raw: raw["rounds"][0]["workers"].pop(),
        lambda raw: raw["rounds"][0]["workers"][0].update(group_id=1),
        lambda raw: raw["rounds"][0]["workers"][0].update(traffic="cross"),
        lambda raw: raw["rounds"][0]["workers"][12].update(traffic="intra"),
        lambda raw: raw["rounds"][0]["workers"][12].update(traffic="all"),
        lambda raw: raw["rounds"][0]["workers"][0].update(rows=1172),
        lambda raw: raw["rounds"][0]["workers"][0].update(start_cycle=-1),
        lambda raw: raw["rounds"][0]["workers"][0].update(end_cycle=0),
    ],
)
def test_malformed_native_reports_are_rejected(mutate):
    raw = native_group_report()
    mutate(raw)
    with pytest.raises(group_report.ReportError):
        group_report.validate_report(raw, expected_iterations=3)


def test_valid_rotation_with_wrong_physical_row_mapping_is_rejected():
    raw = native_group_report(rotation_start=0)
    for measured in raw["rounds"]:
        measured["worker_rotation"] = (measured["worker_rotation"] + 1) % 48
    with pytest.raises(group_report.ReportError, match="rotated"):
        group_report.validate_report(raw)


@pytest.mark.parametrize(
    "failure", ["missing_rank", "duplicate_rank", "duplicate_device", "mode_mismatch", "rotation_mismatch"]
)
def test_incomplete_or_inconsistent_rank_sets_are_rejected(failure):
    reports = [native_group_report(rank) for rank in range(16)]
    if failure == "missing_rank":
        reports.pop()
    elif failure == "duplicate_rank":
        reports[-1] = copy.deepcopy(reports[0])
    elif failure == "duplicate_device":
        reports[-1]["device"] = 0
    elif failure == "mode_mismatch":
        reports[-1]["metadata_mode"] = "scalar"
    else:
        reports[-1] = native_group_report(15, rotation_start=0)
    with pytest.raises(group_report.ReportError):
        group_report.summarize_reports(reports, expected_world=16, expected_iterations=3)


def test_timer_exports_contain_only_whole_worker_spans(tmp_path):
    raw = native_group_report(clock_offset=2**60)
    csv_path, trace_path = group_report.export_timers(raw, tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 28
    assert {row["traffic"] for row in rows} == {"intra", "cross"}
    assert {row["worker_rotation"] for row in rows} == {"47", "0", "1"}
    assert sum(row["traffic"] == "intra" for row in rows) == 3 * 12
    assert sum(row["traffic"] == "cross" for row in rows) == 3 * 16
    assert min(float(row["start_us_rank_origin"]) for row in rows) == 0.0
    assert all(int(row["bytes"]) == int(row["rows"]) * 3712 for row in rows)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(trace["traceEvents"]) == 3 * 28
    assert all(
        event["cat"] in {"intra", "cross"} and event["name"].endswith("_" + event["cat"].upper())
        for event in trace["traceEvents"]
    )
    assert all(
        event["args"]["source_group_order"] == ([0] if event["cat"] == "intra" else [3, 2, 1])
        for event in trace["traceEvents"]
    )
    assert all(event["args"]["time_origin"] == "rank_first_measured_start" for event in trace["traceEvents"])


@pytest.mark.parametrize("metadata_mode", ["scalar", "dma", "preload"])
def test_group_plan_has_no_npu_network_plot_or_filesystem_side_effects(tmp_path, monkeypatch, capsys, metadata_mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("--plan must not launch, query resources, reserve ports, or write files")

    for name in ("_endpoint", "_resource_snapshot", "_write_json", "run_case"):
        monkeypatch.setattr(run, name, forbidden)
    monkeypatch.setattr(run.subprocess, "Popen", forbidden)
    monkeypatch.setattr(run.subprocess, "run", forbidden)
    monkeypatch.setattr(run.socket, "socket", forbidden)
    original_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"matplotlib", "torch", "acl", "shmem"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only_import)
    output = tmp_path / "must_not_exist"
    assert (
        run.main(
            [
                "--schedule",
                "group-round-robin",
                "--metadata-mode",
                metadata_mode,
                "--bs-values",
                "2048,4096,8192",
                "--binary",
                str(tmp_path / "missing_binary"),
                "--output-dir",
                str(output),
                "--plan",
                "--timer",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert plan["mode"] == "group_round_robin" and plan["workers"] == {"intra": 12, "cross": 16}
    assert plan["buffer_count"] == 6
    assert len(plan["cases"]) == 3
    assert "without phase synchronization" in plan["note"]
    for case in plan["cases"]:
        assert len(case["commands"]) == 16
        for command in case["commands"]:
            assert command[command.index("--schedule") + 1] == "group-round-robin"
            assert command[command.index("--metadata-mode") + 1] == metadata_mode
            assert "--workers" not in command
