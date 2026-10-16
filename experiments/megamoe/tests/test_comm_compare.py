# Copyright (c) 2026, SimpleBright_Man

from copy import deepcopy
import csv
import json
import shutil

import pytest

from comm_report_fixtures import capture
from deep_ep_experimental.megamoe import comm_compare, comm_report


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def build_capture(directory):
    records = []
    for rank in range(16):
        spec, timer, audit, _ = capture(bs=16, rank=rank)
        report = comm_report.rank_report(
            comm_report.read_events(timer, rank), transfers=comm_report.read_transfers(audit, rank), **spec
        )
        records.append(comm_report.write_rank_detail(report, directory))
    write(directory / "communication.json", comm_report.summarize(records))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("full_capture")
    build_capture(directory)
    return directory


@pytest.fixture
def candidate(reference, tmp_path):
    directory = tmp_path / "communication_only_capture"
    shutil.copytree(reference, directory)
    return directory


def change_detail(directory, rank, change):
    """Keep aggregate accounting internally consistent after changing timings."""
    path = directory / f"rank_{rank:02d}_communication.json"
    detail = read(path)
    change(detail)
    for wave in detail["waves"]:
        wave["sum_core_us"] = sum(core["duration_us"] for core in wave["cores"])
        wave["mean_core_us"] = wave["sum_core_us"] / 28
        wave["gbps"] = wave["bytes"] / wave["mean_core_us"] / 1000 if wave["mean_core_us"] else None
    for operation, total in detail["operations"].items():
        total["sum_core_us"] = sum(wave["sum_core_us"] for wave in detail["waves"] if wave["operation"] == operation)
        total["mean_core_us"] = total["sum_core_us"] / 28
        total["gbps"] = total["bytes"] / total["mean_core_us"] / 1000
    write(path, detail)
    summary = read(directory / "communication.json")
    compact = summary["ranks"][rank]
    compact["operations"] = detail["operations"]
    compact["validation"] = detail["validation"]
    write(directory / "communication.json", summary)


def change_csv(directory, mutate):
    path = directory / "rank_00_communication.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


def test_identical_capture_and_relocated_details_pass(reference, candidate):
    # The copied compact report still names reference absolute paths. Adjacent
    # files must win so copied and remote-machine reports compare correctly.
    result = comm_compare.compare_reports(reference, candidate / "communication.json")
    assert result["workload_match"] and result["timing_match"] and result["passed"]
    for value in result["operations"].values():
        assert value["within_tolerance"] == 16 * 28
        assert value["max_abs_relative_error"] == 0
    assert len(result["ranks"][0]["operations"]["DISPATCH"]["waves"]) > 1


def test_dispatch_slowdown_does_not_hide_behind_combine_or_rank_average(reference, candidate):
    def slow_one_core(detail):
        for wave in detail["waves"]:
            if wave["operation"] == "DISPATCH":
                wave["cores"][0]["duration_us"] *= 1.4

    change_detail(candidate, 7, slow_one_core)
    result = comm_compare.compare_reports(reference, candidate)
    assert result["workload_match"] and not result["timing_match"]
    assert result["operations"]["DISPATCH"]["outside_tolerance"] == 1
    assert result["operations"]["COMBINE"]["outside_tolerance"] == 0
    core = result["ranks"][7]["operations"]["DISPATCH"]["cores"][0]
    assert core["ratio"] == pytest.approx(1.4)
    assert core["delta_us"] == pytest.approx(core["reference_us"] * 0.4)
    assert result["operations"]["DISPATCH"]["worst_cores"][0]["rank"] == 7


def test_gate_sums_per_core_over_waves_and_exposes_wave_differences(reference, candidate):
    def shift_time(detail):
        selected = [wave["cores"][0] for wave in detail["waves"] if wave["operation"] == "DISPATCH"]
        selected = [core for core in selected if core["duration_us"] > 0]
        delta = min(selected[0]["duration_us"], selected[1]["duration_us"]) * 0.5
        selected[0]["duration_us"] += delta
        selected[1]["duration_us"] -= delta

    change_detail(candidate, 0, shift_time)
    result = comm_compare.compare_reports(reference, candidate)
    assert result["timing_match"]
    assert not result["operations"]["DISPATCH"]["wave_timing_match"]
    assert result["operations"]["DISPATCH"]["wave_cores_outside_tolerance"] == 2


def test_same_byte_counts_and_logical_hash_cannot_mask_wrong_token(reference, candidate):
    def wrong_token(rows):
        rows[0]["source_token"] = str(int(rows[0]["source_token"]) + 999)

    change_csv(candidate, wrong_token)
    with pytest.raises(ValueError, match="exact wave/core/token workload mismatch"):
        comm_compare.compare_reports(reference, candidate)


def test_same_byte_counts_cannot_mask_wrong_physical_core(reference, candidate):
    def swap_core(rows):
        first = rows[0]
        second = next(
            row
            for row in rows
            if all(row[field] == first[field] for field in ("operation", "wave", "source_rank"))
            and row["core"] != first["core"]
        )
        first["core"], second["core"] = second["core"], first["core"]

    change_csv(candidate, swap_core)
    with pytest.raises(ValueError, match="exact wave/core/token workload mismatch"):
        comm_compare.compare_reports(reference, candidate)


def test_duplicate_trace_row_is_not_accepted(reference, candidate):
    change_csv(candidate, lambda rows: rows.__setitem__(1, deepcopy(rows[0])))
    with pytest.raises(ValueError, match="missing or duplicate"):
        comm_compare.compare_reports(reference, candidate)


def test_missing_candidate_audit_cannot_fall_back_to_reference_absolute_path(reference, candidate):
    (candidate / "rank_00_communication.csv").unlink()
    with pytest.raises(ValueError, match="missing csv detail"):
        comm_compare.compare_reports(reference, candidate)


def test_same_tasks_with_changed_timer_boundaries_rejected(reference, candidate):
    def change_boundary(detail):
        detail["waves"][0]["cores"][0]["spans"] += 1

    change_detail(candidate, 0, change_boundary)
    with pytest.raises(ValueError, match="timer-boundary mismatch"):
        comm_compare.compare_reports(reference, candidate)


def test_negative_duration_and_missing_rank_rejected(reference, candidate):
    path = candidate / "rank_00_communication.json"
    detail = read(path)
    detail["waves"][0]["cores"][0]["duration_us"] = -1
    write(path, detail)
    with pytest.raises(ValueError, match="finite nonnegative"):
        comm_compare.compare_reports(reference, candidate)
    summary = read(candidate / "communication.json")
    summary["ranks"].pop()
    write(candidate / "communication.json", summary)
    with pytest.raises(ValueError, match="exactly all 16"):
        comm_compare.compare_reports(reference, candidate)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(reference, candidate, value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        comm_compare.compare_reports(reference, candidate, timing_rtol=value)


def test_absolute_tolerance_and_cli_strict_gate(reference, candidate, tmp_path, capsys):
    def slow(detail):
        detail["waves"][0]["cores"][0]["duration_us"] += 2.0

    change_detail(candidate, 0, slow)
    assert comm_compare.compare_reports(reference, candidate, timing_rtol=0, timing_atol_us=2.0)["timing_match"]
    output = tmp_path / "comparison.json"
    args = ["--reference", str(reference), "--candidate", str(candidate), "--timing-rtol", "0", "--output", str(output)]
    result = comm_compare.main(args)
    assert not result["timing_match"]
    assert read(output)["ranks"][0]["operations"]["COMBINE"]["cores"]
    assert "[COMPARE] DISPATCH exact-workload=PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        comm_compare.main([*args, "--require-timing-match"])
    assert error.value.code == 1
