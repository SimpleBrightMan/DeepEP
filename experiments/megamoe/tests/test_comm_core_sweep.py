# Copyright (c) 2026, SimpleBright_Man

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import comm_report, output_check

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "comm_core_sweep", ROOT / "experiments/megamoe/scripts/comm_core_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_plan_does_not_touch_devices_or_create_output(tmp_path, capsys):
    target = tmp_path / "new"
    sweep.main(["--plan", "--output-dir", str(target), "--intra-cores", "2,4,6", "--bs-values", "2048"])
    planned = json.loads(capsys.readouterr().out)
    assert not target.exists()
    assert planned["total_case_launches"] == 1
    assert planned["mode"] == "baseline"
    assert planned["dispatch_metadata_mode"] == "dma"
    assert planned["combine_buffer_count"] == 2
    assert planned["group_order"] == "reverse"
    assert planned["combine_intra_candidates"] == []
    assert planned["model"]["num_topk"] == 8
    assert not planned["global_optimum_guaranteed"]


def test_paired_plan_is_one_fixed_pair_per_bs_without_baseline_or_selection():
    args = sweep.parse_args(["--mode", "paired-groups", "--no-validation"])
    planned = sweep.plan(args)
    assert planned["mode"] == "paired-groups" and planned["group_order"] == "paired-xor"
    assert planned["case_launches_per_bs"] == 1 and planned["total_case_launches"] == 3
    assert planned["fixed_pair"] == [12, 12] and planned["combine_fixed"]
    assert planned["dispatch_intra_candidates"] == planned["combine_intra_candidates"] == [12]
    assert planned["selection"] is None and len(planned["stages"]) == 1
    assert not any("baseline" in stage or "confirmation" in stage for stage in planned["stages"])
    assert planned["combine_buffer_count"] == 2 and planned["dispatch_metadata_mode"] == "dma"
    env = sweep.candidate_environment((12, 12), {sweep.ENV_GROUP_ORDER: "reverse"}, group_order="paired-xor")
    assert env[sweep.ENV_GROUP_ORDER] == "paired-xor"
    assert env[sweep.ENV_DISPATCH] == env[sweep.ENV_COMBINE] == "12"
    with pytest.raises(ValueError, match="fixed"):
        sweep.candidate_environment((10, 12), group_order="paired-xor")


@pytest.mark.parametrize(
    "extra",
    [
        ["--intra-cores", "10,12"],
        ["--dispatch-intra-cores", "12"],
        ["--combine-intra-cores", "12"],
        ["--combine-intra", "10"],
        ["--run-full-reference"],
        ["--full-reference-dir", "old"],
        ["--require-timing-match"],
    ],
)
def test_paired_groups_rejects_sweeps_and_original_ownership_reference(extra):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--mode", "paired-groups", *extra])


def test_paired_driver_launches_exactly_three_fixed_cases_and_no_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix", environ={sweep.ENV_GROUP_ORDER: "reverse"}))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    launches, reads, records, selections = [], [], [], []
    monkeypatch.setattr(sweep, "execute", lambda argv, **kwargs: launches.append((argv, kwargs)))
    monkeypatch.setattr(sweep, "print_case", records.append)
    monkeypatch.setattr(sweep, "write_summary", lambda path, plan, cases, chosen: selections.extend(chosen))

    def read(output, bs, pair, **options):
        reads.append((output, bs, pair, options))
        return dict(bs=bs, pair=list(pair), passed=True, validation="skipped", **options)

    monkeypatch.setattr(sweep, "read_case", read)
    sweep.main(
        [
            "--mode",
            "paired-groups",
            "--no-validation",
            "--combine-buffer-count",
            "6",
            "--case-gap-seconds",
            "0",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert len(launches) == len(reads) == 3 and not selections
    assert [bs for _, bs, _, _ in reads] == [2048, 4096, 8192]
    assert all(path.name == "paired_groups_d12_c12" and pair == (12, 12) for path, _, pair, _ in reads)
    assert all(item["stage"] == "paired_groups" for item in records)
    for argv, launch in launches:
        assert "deep_ep_experimental.megamoe.comm_benchmark" in argv and "--no-validation" in argv
        assert launch["environment"][sweep.ENV_GROUP_ORDER] == "paired-xor"
        assert launch["environment"][sweep.ENV_DISPATCH] == launch["environment"][sweep.ENV_COMBINE] == "12"
    for _, _, _, options in reads:
        assert options == dict(
            validate=False, dispatch_metadata_mode="dma", combine_buffer_count=6, group_order="paired-xor"
        )
    assert not list((tmp_path / "run").rglob("baseline"))


def test_staged_cached_case_and_grid_counts():
    args = sweep.parse_args(
        ["--mode", "staged", "--intra-cores", "7,12,21", "--bs-values", "4096", "--combine-intra", "12"]
    )
    assert sweep.plan(args)["case_launches_per_bs"] == 7
    args.mode = "fullgrid"
    assert sweep.plan(args)["case_launches_per_bs"] == 11


def test_full_reference_is_explicit_and_included_in_launch_plan():
    args = sweep.parse_args(["--run-full-reference", "--timing-rtol", "0.05", "--require-timing-match"])
    planned = sweep.plan(args)
    assert planned["workload"] == "communication_only"
    assert planned["total_case_launches"] == 6
    assert planned["full_reference"] == "run" and planned["timing_rtol"] == 0.05
    with pytest.raises(SystemExit):
        sweep.parse_args(["--require-timing-match"])
    with pytest.raises(SystemExit):
        sweep.parse_args(["--run-full-reference", "--full-reference-dir", "old"])


@pytest.mark.parametrize(
    "extra", [["--run-full-reference"], ["--full-reference-dir", "old"], ["--require-timing-match"]]
)
def test_performance_only_rejects_full_reference_and_timing_requirements(extra):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--no-validation", *extra])


def test_performance_only_plan_and_child_command(tmp_path, capsys):
    target = tmp_path / "performance"
    sweep.main(["--plan", "--no-validation", "--output-dir", str(target)])
    planned = json.loads(capsys.readouterr().out)
    assert not target.exists()
    assert planned["validation"] == "skipped"
    assert planned["total_case_launches"] == 3
    assert planned["full_reference"] is None
    assert "validation skipped" in planned["stages"][0]
    assert not any("checks Dispatch bytes" in note for note in planned["notes"])
    args = sweep.parse_args(["--no-validation"])
    command = sweep.command(args, tmp_path / "config", target, None, target)
    assert "--no-validation" in command
    assert "deep_ep_experimental.megamoe.comm_benchmark" in command
    assert command[command.index("--dispatch-metadata-mode") + 1] == "dma"
    assert command[command.index("--combine-buffer-count") + 1] == "2"


@pytest.mark.parametrize("count", [2, 4, 6])
def test_combine_buffer_count_reaches_child_and_never_changes_full_reference(tmp_path, count):
    args = sweep.parse_args(["--combine-buffer-count", str(count), "--no-validation"])
    planned = sweep.plan(args)
    assert planned["combine_buffer_count"] == count and planned["dispatch_metadata_mode"] == "dma"
    command = sweep.command(args, tmp_path / "config", tmp_path / "out", None, tmp_path)
    assert command[command.index("--combine-buffer-count") + 1] == str(count)
    assert "--no-validation" in command
    args.no_validation = False
    full = sweep.command(args, tmp_path / "config", tmp_path / "full", None, tmp_path, full_reference=True)
    assert "--combine-buffer-count" not in full
    assert "--dispatch-metadata-mode" not in full


def test_scalar_metadata_mode_is_explicit_and_does_not_change_full_reference(tmp_path):
    args = sweep.parse_args(["--dispatch-metadata-mode", "scalar"])
    assert sweep.plan(args)["dispatch_metadata_mode"] == "scalar"
    command = sweep.command(args, tmp_path / "config", tmp_path / "out", None, tmp_path)
    assert command[command.index("--dispatch-metadata-mode") + 1] == "scalar"
    for mode in ("dma", "scalar"):
        args.dispatch_metadata_mode = mode
        full = sweep.command(args, tmp_path / "config", tmp_path / "full", None, tmp_path, full_reference=True)
        assert "--dispatch-metadata-mode" not in full
        assert "deep_ep_experimental.megamoe.benchmark" in full


@pytest.mark.parametrize("different", ["intermediate_hidden", "alignment", "rank", "split", "group_order"])
def test_full_reference_rejects_incomparable_manifest_before_comparing_timers(tmp_path, different):
    args = sweep.parse_args([])
    reference, candidate = tmp_path / "full", tmp_path / "simulation"
    timer = reference / "bs_2048" / "timer"
    timer.mkdir(parents=True)
    (timer / "communication.json").write_text("{}")
    config = sweep.configuration(args, 2048)
    manifest = dict(
        api="fp8_fp4_mega_moe",
        rank=0,
        ep_size=16,
        mode="perf",
        input="fp8",
        routing="balanced",
        timer_ranks=list(range(16)),
        config=config,
        comm_core_split={sweep.ENV_DISPATCH: None, sweep.ENV_COMBINE: None},
    )
    simulated = dict(manifest, workload="communication_only", api="communication_forward")
    altered = copy.deepcopy(manifest)
    if different == "intermediate_hidden":
        altered["config"]["model"]["intermediate_hidden"] = 1536
    elif different == "alignment":
        altered["config"]["alignment"]["repeats"] += 1
    elif different == "rank":
        altered["rank"] = 1
    elif different == "group_order":
        altered["group_order"] = "paired-xor"
    else:
        altered["comm_core_split"][sweep.ENV_DISPATCH] = "12"
    for root, value in ((reference, altered), (candidate, simulated)):
        directory = root / "rank_0"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="matching model/BS/alignment"):
        sweep.compare_reference(args, reference, candidate, 2048, tmp_path / "comparison.json")


@pytest.mark.parametrize(
    "option,value",
    [
        ("--intra-cores", "0"),
        ("--intra-cores", "28"),
        ("--intra-cores", "2,2"),
        ("--dispatch-intra-cores", "x"),
        ("--bs-values", "0"),
        ("--bs-values", "256"),
        ("--bs-values", "2048,1024"),
        ("--timing-atol-us", "nan"),
        ("--initial-combine-intra", "28"),
        ("--combine-intra", "-1"),
        ("--combine-intra", "12"),  # baseline must preserve both original schedules
        ("--combine-intra-cores", "10,12"),
        ("--case-gap-seconds", "-1"),
        ("--case-gap-seconds", "nan"),
        ("--case-gap-seconds", "61"),
        ("--shmem-port-policy", "invalid"),
        ("--dispatch-metadata-mode", "invalid"),
        ("--combine-buffer-count", "0"),
        ("--combine-buffer-count", "3"),
        ("--combine-buffer-count", "8"),
    ],
)
def test_reject_invalid_scan(option, value):
    with pytest.raises(SystemExit):
        sweep.parse_args([option, value])


def test_original_environment_cleared_and_pair_command(tmp_path):
    inherited = {
        sweep.ENV_DISPATCH: "7",
        sweep.ENV_COMBINE: "8",
        sweep.ENV_GROUP_ORDER: "paired-xor",
        "LD_LIBRARY_PATH": "/sdk/lib",
    }
    baseline = sweep.candidate_environment(None, inherited)
    assert sweep.ENV_DISPATCH not in baseline and sweep.ENV_COMBINE not in baseline
    assert sweep.ENV_GROUP_ORDER not in baseline
    pair = sweep.candidate_environment((12, 7), inherited)
    assert pair[sweep.ENV_DISPATCH] == "12" and pair[sweep.ENV_COMBINE] == "7"
    assert pair["LD_LIBRARY_PATH"] == "/sdk/lib"
    assert pair["PYTHON_EXEC"] == sys.executable
    assert sweep.ENV_GROUP_ORDER not in pair
    args = sweep.parse_args([])
    command = sweep.command(args, tmp_path / "model.json", tmp_path / "test", (12, 7), tmp_path / "base")
    assert "--no-timer-raw" in command and "deep_ep_experimental.megamoe.comm_benchmark" in command
    assert "--save-output-reference" not in command
    original_command = sweep.command(args, tmp_path / "model.json", tmp_path / "base", None, tmp_path / "base")
    assert "--save-output-reference" not in original_command
    assert "deep_ep_experimental.megamoe.comm_benchmark" in original_command
    reference_command = sweep.command(
        args, tmp_path / "model.json", tmp_path / "reference", None, tmp_path / "base", full_reference=True
    )
    assert "deep_ep_experimental.megamoe.benchmark" in reference_command
    assert "--comm-report" in reference_command
    dispatch_only = sweep.candidate_environment((10, 0), inherited)
    assert dispatch_only[sweep.ENV_DISPATCH] == "10" and dispatch_only[sweep.ENV_COMBINE] == "0"


def test_selection_uses_slowest_rank_not_fastest():
    records = [
        dict(passed=True, pair=[7, 7], comm_kernel_rank_max_mean_us=110, comm_kernel_rank_min_mean_us=80),
        dict(passed=True, pair=[8, 8], comm_kernel_rank_max_mean_us=100, comm_kernel_rank_min_mean_us=90),
    ]
    assert sweep.choose(records)["pair"] == [8, 8]


def test_staged_search_retains_first_stage_winner_outside_combine_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    launches, gaps = [], []
    monkeypatch.setattr(sweep, "execute", lambda *a, **k: launches.append(k["environment"]))
    monkeypatch.setattr(sweep.time, "sleep", gaps.append)
    monkeypatch.setattr(sweep, "print_case", lambda *a: None)
    selections = []
    monkeypatch.setattr(sweep, "write_summary", lambda path, plan, records, chosen: selections.extend(chosen))
    durations = {None: 200, (2, 12): 90, (4, 12): 100, (2, 2): 150, (2, 4): 130}

    def read(
        output, bs, pair, *, validate=True, dispatch_metadata_mode="dma", combine_buffer_count=2, group_order="reverse"
    ):
        assert validate
        assert dispatch_metadata_mode == "dma"
        assert combine_buffer_count == 2
        assert group_order == "reverse"
        return dict(pair=list(pair) if pair else None, passed=True, comm_kernel_rank_max_mean_us=durations[pair])

    monkeypatch.setattr(sweep, "read_case", read)
    sweep.main(
        [
            "--mode",
            "staged",
            "--output-dir",
            str(tmp_path / "run"),
            "--bs-values",
            "2048",
            "--intra-cores",
            "2,4",
            "--combine-intra",
            "12",
        ]
    )
    assert selections[-1]["pair"] == [2, 12]
    assert selections[-1]["confirmation_us"] == 90
    assert len(launches) == 6 and gaps == [2.0] * 5
    assert len({env["DEEPEP_SHMEM_ENDPOINT"] for env in launches}) == len(launches)


def test_dispatch_only_keeps_combine_fixed_through_scan_and_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    launches, results, selections = [], [], []
    monkeypatch.setattr(sweep, "execute", lambda *a, **k: launches.append(k["environment"]))
    monkeypatch.setattr(sweep, "print_case", lambda item: results.append(item))
    monkeypatch.setattr(sweep, "write_summary", lambda path, plan, records, chosen: selections.extend(chosen))
    durations = {None: 200, (10, 7): 120, (12, 7): 110, (16, 7): 130, (20, 7): 125}

    def read(
        output, bs, pair, *, validate=True, dispatch_metadata_mode="dma", combine_buffer_count=2, group_order="reverse"
    ):
        assert validate
        assert dispatch_metadata_mode == "dma"
        assert combine_buffer_count == 2
        assert group_order == "reverse"
        return dict(pair=list(pair) if pair else None, passed=True, comm_kernel_rank_max_mean_us=durations[pair])

    monkeypatch.setattr(sweep, "read_case", read)
    sweep.main(
        [
            "--mode",
            "dispatch",
            "--output-dir",
            str(tmp_path / "run"),
            "--bs-values",
            "2048",
            "--intra-cores",
            "10,12,16,20",
            "--combine-intra",
            "7",
            "--case-gap-seconds",
            "0",
        ]
    )
    assert [item["stage"] for item in results] == ["baseline"] + ["dispatch"] * 4 + ["confirm"]
    assert [item["pair"] for item in results] == [None, [10, 7], [12, 7], [16, 7], [20, 7], [12, 7]]
    assert all(env[sweep.ENV_COMBINE] == "7" for env in launches[1:])
    assert selections[-1]["pair"] == [12, 7]


def test_dispatch_only_plan_counts_and_legacy_fixed_combine_alias():
    args = sweep.parse_args(["--mode", "dispatch", "--intra-cores", "10,12,16,20"])
    plan = sweep.plan(args)
    assert plan["case_launches_per_bs"] == 6 and plan["total_case_launches"] == 18
    assert args.combine_cores == [0]
    args = sweep.parse_args(["--mode", "dispatch", "--initial-combine-intra", "8"])
    assert args.combine_cores == [8] and args.initial_combine_intra == 8


def test_default_baseline_runs_each_bs_once_without_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sweep, "os", SimpleNamespace(name="posix", environ={sweep.ENV_DISPATCH: "10", sweep.ENV_COMBINE: "12"})
    )
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    launches, results, selections = [], [], []
    monkeypatch.setattr(sweep, "execute", lambda argv, **kwargs: launches.append((argv, kwargs["environment"])))
    monkeypatch.setattr(sweep, "print_case", results.append)
    monkeypatch.setattr(sweep, "write_summary", lambda path, plan, records, chosen: selections.extend(chosen))
    monkeypatch.setattr(sweep, "read_case", lambda output, bs, pair, **kwargs: dict(bs=bs, pair=pair, passed=True))
    sweep.main(["--output-dir", str(tmp_path / "run"), "--case-gap-seconds", "0"])
    assert [item["bs"] for item in results] == [2048, 4096, 8192]
    assert all(item["pair"] is None and item["stage"] == "baseline" for item in results)
    assert not selections and len(launches) == 3
    assert all("deep_ep_experimental.megamoe.comm_benchmark" in argv for argv, env in launches)
    assert all(sweep.ENV_DISPATCH not in env and sweep.ENV_COMBINE not in env for argv, env in launches)


def test_fresh_endpoint_preserves_host_and_skips_previous_ports(monkeypatch):
    ports, bound = iter([41001, 41001, 41002]), []

    class Reservation:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            bound.append(address)

        def getsockname(self):
            return "127.0.0.1", next(ports)

    monkeypatch.setattr(sweep.socket, "socket", lambda *args: Reservation())
    used = set()
    env = {"DEEPEP_SHMEM_ENDPOINT": "tcp://127.0.0.1:19091"}
    assert sweep.fresh_endpoint(env, used) == "tcp://127.0.0.1:41001"
    assert sweep.fresh_endpoint(env, used) == "tcp://127.0.0.1:41002"
    assert bound == [("127.0.0.1", 0)] * 3


@pytest.mark.skipif(sys.platform != "linux", reason="Isolated process groups require Linux")
@pytest.mark.parametrize("status", [0, 3])
def test_execute_waits_and_records_exit_without_masking_child_failure(tmp_path, status):
    log = tmp_path / "case.log"
    argv = [sys.executable, "-c", f"raise SystemExit({status})"]
    kwargs = dict(cwd=tmp_path, environment=sweep.candidate_environment(None), logfile=log, timeout=10)
    if status:
        with pytest.raises(RuntimeError, match="status 3"):
            sweep.execute(argv, **kwargs)
    else:
        sweep.execute(argv, **kwargs)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]["event"] == "torchrun_exit" and records[-1]["returncode"] == status


def write_case_fixture(
    tmp_path, pair, *, validate=True, metadata_mode="dma", combine_buffer_count=2, group_order="reverse"
):
    bs = 2048
    for rank in range(16):
        directory = tmp_path / f"rank_{rank}"
        directory.mkdir()
        manifest = dict(
            rank=rank,
            ep_size=16,
            mode="perf",
            workload="communication_only",
            routing="balanced",
            input="fp8",
            comm_core_split={
                sweep.ENV_DISPATCH: str(pair[0]) if pair else None,
                sweep.ENV_COMBINE: str(pair[1]) if pair else None,
            },
            config=dict(bs_values=[bs]),
            build={"test": True},
        )
        if not validate:
            manifest["validation"] = "skipped"
        if metadata_mode is not None:
            manifest["dispatch_metadata_mode"] = metadata_mode
        if combine_buffer_count is not None:
            manifest["combine_buffer_count"] = combine_buffer_count
        if group_order is not None:
            manifest["group_order"] = group_order
        (directory / "manifest.json").write_text(json.dumps(manifest))
    rank0 = tmp_path / "rank_0" / f"bs_{bs}"
    rank0.mkdir()
    profile = dict(
        timer_pass=False,
        metric="communication_only_kernel_duration_us",
        per_rank_rounds_us=[[100.0 + rank, 102.0 + rank] for rank in range(16)],
    )
    (rank0 / "profiling.json").write_text(json.dumps(profile))
    checks = [dict(rank=rank, passed=True) for rank in range(16)]
    if validate:
        (rank0 / "payload_accuracy.json").write_text(json.dumps(checks))
        (rank0 / "timer_accuracy.json").write_text(json.dumps(checks))
        (rank0 / "communication_audit_accuracy.json").write_text(json.dumps(checks))
    timer = tmp_path / f"bs_{bs}" / "timer"
    timer.mkdir(parents=True)
    communication = dict(
        schema_version=2,
        ranks=[dict(rank=rank, validation=dict(strict_wave_core=True)) for rank in range(16)],
        operations={
            operation: dict(gbps_per_card_mean=80.0, gbps_per_card_min=75.0, gbps_per_card_max=85.0)
            for operation in ("DISPATCH", "COMBINE")
        },
    )
    if metadata_mode is not None:
        communication["dispatch_metadata_mode"] = metadata_mode
    if combine_buffer_count is not None:
        communication["combine_buffer_count"] = combine_buffer_count
    if group_order is not None:
        communication["group_order"] = group_order
    if group_order == "paired-xor":
        for operation in communication["operations"].values():
            operation["cohorts"] = {
                name: dict(workers=workers, gbps_per_card_mean=60.0, gbps_per_card_min=55.0, gbps_per_card_max=65.0)
                for name, workers in (("INTRA", 12), ("CROSS", 16))
            }
    if not validate:
        communication["validation"] = dict(strict_wave_core=False, status="skipped")
        for record in communication["ranks"]:
            record["validation"] = dict(strict_wave_core=False, status="skipped", accounting="timer_row_counts")
    (timer / "communication.json").write_text(json.dumps(communication))
    return rank0, timer, communication, checks


@pytest.mark.parametrize("recorded", ["dma", "scalar", None])
def test_case_reader_checks_requested_metadata_mode_and_legacy_means_scalar(tmp_path, recorded):
    write_case_fixture(tmp_path, None, validate=False, metadata_mode=recorded)
    actual = recorded or "scalar"
    report = sweep.read_case(tmp_path, 2048, None, validate=False, dispatch_metadata_mode=actual)
    assert report["dispatch_metadata_mode"] == actual
    opposite = "scalar" if actual == "dma" else "dma"
    with pytest.raises(ValueError, match="manifest"):
        sweep.read_case(tmp_path, 2048, None, validate=False, dispatch_metadata_mode=opposite)


def test_case_reader_rejects_metadata_mode_mismatch_in_summary(tmp_path):
    _, timer, communication, _ = write_case_fixture(tmp_path, None, validate=False)
    communication["dispatch_metadata_mode"] = "scalar"
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="report metadata mode"):
        sweep.read_case(tmp_path, 2048, None, validate=False)


@pytest.mark.parametrize("recorded", [2, 4, 6, None])
def test_case_reader_checks_requested_combine_slots_and_legacy_means_two(tmp_path, recorded):
    write_case_fixture(tmp_path, None, validate=False, combine_buffer_count=recorded)
    actual = recorded or 2
    report = sweep.read_case(tmp_path, 2048, None, validate=False, combine_buffer_count=actual)
    assert report["combine_buffer_count"] == actual
    other = 4 if actual == 2 else 2
    with pytest.raises(ValueError, match="manifest"):
        sweep.read_case(tmp_path, 2048, None, validate=False, combine_buffer_count=other)


def test_case_reader_rejects_combine_slot_mismatch_in_summary(tmp_path):
    _, timer, communication, _ = write_case_fixture(tmp_path, None, validate=False, combine_buffer_count=4)
    communication["combine_buffer_count"] = 6
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="report Combine buffer count"):
        sweep.read_case(tmp_path, 2048, None, validate=False, combine_buffer_count=4)


@pytest.mark.parametrize("recorded", ["reverse", "paired-xor", None])
def test_case_reader_checks_requested_group_order_and_legacy_means_reverse(tmp_path, recorded):
    write_case_fixture(tmp_path, (12, 12), validate=False, group_order=recorded)
    actual = recorded or "reverse"
    report = sweep.read_case(tmp_path, 2048, (12, 12), validate=False, group_order=actual)
    assert report["group_order"] == actual
    other = "paired-xor" if actual == "reverse" else "reverse"
    with pytest.raises(ValueError, match="manifest"):
        sweep.read_case(tmp_path, 2048, (12, 12), validate=False, group_order=other)


def test_case_reader_rejects_group_order_mismatch_in_summary(tmp_path):
    _, timer, communication, _ = write_case_fixture(tmp_path, (12, 12), validate=False, group_order="paired-xor")
    communication["group_order"] = "reverse"
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="report group order"):
        sweep.read_case(tmp_path, 2048, (12, 12), validate=False, group_order="paired-xor")


@pytest.mark.parametrize("corruption", ["missing", "workers", "bandwidth"])
def test_paired_case_reader_requires_complete_cohort_metrics(tmp_path, corruption):
    _, timer, communication, _ = write_case_fixture(tmp_path, (12, 12), validate=False, group_order="paired-xor")
    dispatch = communication["operations"]["DISPATCH"]
    if corruption == "missing":
        dispatch.pop("cohorts")
    elif corruption == "workers":
        dispatch["cohorts"]["CROSS"]["workers"] = 28
    else:
        dispatch["cohorts"]["INTRA"]["gbps_per_card_mean"] = 0
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="cohort"):
        sweep.read_case(tmp_path, 2048, (12, 12), validate=False, group_order="paired-xor")


def test_paired_cohort_console_csv_and_total_metrics_are_separate(tmp_path, capsys):
    write_case_fixture(tmp_path, (12, 12), validate=False, group_order="paired-xor")
    report = sweep.read_case(tmp_path, 2048, (12, 12), validate=False, group_order="paired-xor")
    report["stage"] = "paired_groups"
    sweep.print_case(report)
    sweep.write_summary(tmp_path, {}, [report], [])
    printed = capsys.readouterr().out
    for operation in ("DISPATCH", "COMBINE"):
        assert f"[BW] {operation} per-card mean=80.000" in printed
        for name in ("INTRA", "CROSS"):
            assert f"[BW] {operation} {name} per-card mean=60.000" in printed
    assert "workers=12 (cohort payload / mean of cohort core communication times)" in printed
    assert "workers=16 (cohort payload / mean of cohort core communication times)" in printed
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    for operation in ("DISPATCH", "COMBINE"):
        assert row[f"{operation}_GBps"] == "80.0"
        assert row[f"{operation}_INTRA_GBps"] == row[f"{operation}_CROSS_GBps"] == "60.0"


@pytest.mark.parametrize("pair", [None, (12, 0), (12, 12)])
def test_case_reader_accepts_complete_data_and_rejects_incomplete_ranks(tmp_path, pair):
    bs = 2048
    rank0, timer, communication, checks = write_case_fixture(tmp_path, pair)
    report = sweep.read_case(tmp_path, bs, pair)
    assert report["comm_kernel_rank_max_mean_us"] == 116.0
    assert report["comm_kernel_rank_min_mean_us"] == 101.0
    (rank0 / "timer_accuracy.json").write_text(json.dumps(checks[:-1]))
    with pytest.raises(ValueError, match="timer-on/off"):
        sweep.read_case(tmp_path, bs, pair)
    (rank0 / "timer_accuracy.json").write_text(json.dumps(checks))
    (rank0 / "communication_audit_accuracy.json").write_text(json.dumps(checks[:-1]))
    with pytest.raises(ValueError, match="audit output"):
        sweep.read_case(tmp_path, bs, pair)
    (rank0 / "communication_audit_accuracy.json").write_text(json.dumps(checks))
    communication["ranks"][4]["validation"]["strict_wave_core"] = False
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="per-wave/core"):
        sweep.read_case(tmp_path, bs, pair)


def test_performance_only_reader_never_requires_accuracy_files_or_audit(tmp_path, capsys):
    rank0, timer, communication, checks = write_case_fixture(tmp_path, None, validate=False)
    assert not list(rank0.glob("*accuracy.json"))
    report = sweep.read_case(tmp_path, 2048, None, validate=False)
    assert report["validation"] == "skipped"
    assert report["comm_kernel_rank_max_mean_us"] == 116.0
    sweep.print_case(report)
    output = capsys.readouterr().out
    assert "validation skipped" in output
    assert "passed" not in output
    assert "JSON/CSV" not in output
    assert "[BW] DISPATCH" in output and "[BW] COMBINE" in output
    # A performance-only read must not silently accept a partially strict report.
    communication["ranks"][4]["validation"]["strict_wave_core"] = True
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError, match="validation-skipped"):
        sweep.read_case(tmp_path, 2048, None, validate=False)


def test_performance_reader_accepts_real_timer_accounting_summary(tmp_path):
    _, timer, _, _ = write_case_fixture(tmp_path, None, validate=False)
    records = []
    for rank in range(16):
        events = [
            dict(
                rank=rank,
                core=core,
                wave=wave,
                family=f"AIV_{operation}_COMM",
                row_count=80 + rank + core,
                start_cycle=1000 + wave * 100000,
                end_cycle=5000 + wave * 100000 + core * 100,
                audit_enabled=False,
            )
            for operation in ("DISPATCH", "COMBINE")
            for wave in range(2)
            for core in range(28)
        ]
        measured = comm_report.timer_rank_report(events, rank, 3584)
        records.append(comm_report.write_rank_detail(measured, timer))
    actual = comm_report.summarize(records, require_validation=False)
    actual["dispatch_metadata_mode"] = "dma"
    (timer / "communication.json").write_text(json.dumps(actual))
    loaded = sweep.read_case(tmp_path, 2048, None, validate=False)
    assert loaded["communication"]["operations"] == actual["operations"]
    assert loaded["validation"] == "skipped"
    assert loaded["communication"]["ranks"][0]["detail_files"].keys() == {"json"}


@pytest.mark.parametrize("validate", [False, True])
def test_case_reader_rejects_validation_mode_mismatch(tmp_path, validate):
    write_case_fixture(tmp_path, None, validate=not validate)
    with pytest.raises(ValueError, match="manifest"):
        sweep.read_case(tmp_path, 2048, None, validate=validate)


@pytest.mark.parametrize("corruption", ["rank", "profile", "bandwidth", "schema"])
def test_performance_only_still_rejects_incomplete_or_invalid_measurements(tmp_path, corruption):
    rank0, timer, communication, checks = write_case_fixture(tmp_path, None, validate=False)
    if corruption == "rank":
        communication["ranks"].pop()
    elif corruption == "profile":
        path = rank0 / "profiling.json"
        profile = json.loads(path.read_text())
        profile["per_rank_rounds_us"][2] = []
        path.write_text(json.dumps(profile))
    elif corruption == "bandwidth":
        communication["operations"]["DISPATCH"]["gbps_per_card_mean"] = 0.0
    else:
        communication["schema_version"] = 1
    (timer / "communication.json").write_text(json.dumps(communication))
    with pytest.raises(ValueError):
        sweep.read_case(tmp_path, 2048, None, validate=False)


def test_performance_only_runs_all_bs_without_reference_or_validation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    launches, reads = [], []
    monkeypatch.setattr(sweep, "execute", lambda argv, **kwargs: launches.append(argv))
    monkeypatch.setattr(sweep, "print_case", lambda item: None)
    monkeypatch.setattr(sweep, "write_summary", lambda *a: None)

    def read(
        output, bs, pair, *, validate=True, dispatch_metadata_mode="dma", combine_buffer_count=2, group_order="reverse"
    ):
        assert dispatch_metadata_mode == "dma"
        assert combine_buffer_count == 6
        assert group_order == "reverse"
        reads.append((bs, pair, validate))
        return dict(bs=bs, pair=pair, passed=True, validation="skipped")

    monkeypatch.setattr(sweep, "read_case", read)
    sweep.main(
        [
            "--no-validation",
            "--output-dir",
            str(tmp_path / "run"),
            "--case-gap-seconds",
            "0",
            "--combine-buffer-count",
            "6",
        ]
    )
    assert reads == [(2048, None, False), (4096, None, False), (8192, None, False)]
    assert len(launches) == 3 and all("--no-validation" in command for command in launches)
    assert all(command[command.index("--combine-buffer-count") + 1] == "6" for command in launches)
    assert all("deep_ep_experimental.megamoe.benchmark" not in command for command in launches)
    assert "COMPLETED_VALIDATION_SKIPPED" in capsys.readouterr().out


def test_output_reference_is_full_finite_exact_and_signature_bound(tmp_path):
    config = dict(model=dict(hidden=64, num_experts=32, num_topk=8), seed=42)
    signature = output_check.signature(config, 0, 16, 2, "fp8", "balanced")
    actual = np.ones((2, 64), dtype=np.float32)
    counts = output_check.expected_counts(signature)
    target = tmp_path / "rank_0" / "bs_2"
    target.mkdir(parents=True)
    output_check.save(target / "output_reference.npz", actual, counts, signature)
    assert output_check.verify(actual, counts, signature, tmp_path)["passed"]
    changed = actual.copy()
    changed[1, 63] += 1
    assert not output_check.verify(changed, counts, signature, tmp_path)["passed"]
    changed[1, 63] = np.nan
    assert not output_check.verify(changed, counts, signature, tmp_path)["passed"]
    assert not output_check.verify(actual, counts + 1, signature, tmp_path)["passed"]
    wrong = copy.deepcopy(signature)
    wrong["seed"] += 1
    with pytest.raises(ValueError, match="signature"):
        output_check.verify(actual, counts, wrong, tmp_path)


def test_summary_and_console_keep_dispatch_combine_independent(tmp_path, capsys):
    operation_values = {
        "DISPATCH": dict(gbps_per_card_mean=70.0, gbps_per_card_min=65.0, gbps_per_card_max=75.0),
        "COMBINE": dict(gbps_per_card_mean=130.0, gbps_per_card_min=125.0, gbps_per_card_max=135.0),
    }
    cases = [
        dict(
            bs=2048,
            stage="baseline" if pair is None else "dispatch",
            dispatch_metadata_mode="dma",
            combine_buffer_count=4,
            group_order="reverse",
            pair=pair,
            output_dir=str(tmp_path / ("baseline" if pair is None else "candidate")),
            comm_kernel_rank_max_mean_us=2400.0,
            comm_kernel_rank_min_mean_us=2350.0,
            comm_kernel_all_ranks_mean_us=2375.0,
            communication=dict(operations=operation_values),
        )
        for pair in (None, [10, 0])
    ]
    sweep.write_summary(tmp_path, {}, cases, [])
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["dispatch_metadata_mode"] == "dma" for row in rows)
    assert all(row["combine_buffer_count"] == "4" for row in rows)
    assert all(row["group_order"] == "reverse" for row in rows)
    assert all(
        row[f"{op}_{cohort}_GBps"] == ""
        for row in rows
        for op in ("DISPATCH", "COMBINE")
        for cohort in ("INTRA", "CROSS")
    )
    assert all(row["DISPATCH_GBps"] == "70.0" and row["COMBINE_GBps"] == "130.0" for row in rows)
    assert rows[0]["dispatch_intra"] == "0" and rows[0]["dispatch_cross"] == ""
    assert rows[1]["combine_intra"] == "0" and rows[1]["combine_cross"] == ""
    sweep.print_case(cases[1])
    printed = capsys.readouterr().out
    assert "D=10/18 C=original" in printed
    assert "dispatch_metadata_mode=dma" in printed
    assert "combine_buffer_count=4" in printed
    assert "group_order=reverse" in printed
    assert "[BW] DISPATCH" in printed and "[BW] COMBINE" in printed
    assert "all 28 core communication times / 28" in printed
    assert "wave envelopes" not in printed
