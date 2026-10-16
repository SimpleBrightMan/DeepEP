# Copyright (c) 2026, SimpleBright_Man

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import comm_report, output_check
from deep_ep_experimental.megamoe.timer import protocol as p

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
    assert planned["total_case_launches"] == 8  # baseline + 3D + 3C + confirmation
    assert planned["model"]["num_topk"] == 8
    assert not planned["global_optimum_guaranteed"]


def test_staged_cached_case_and_grid_counts():
    args = sweep.parse_args(["--intra-cores", "7,12,21", "--bs-values", "4096"])
    assert sweep.plan(args)["case_launches_per_bs"] == 7
    args.mode = "fullgrid"
    assert sweep.plan(args)["case_launches_per_bs"] == 11


@pytest.mark.parametrize(
    "option,value",
    [
        ("--intra-cores", "0"),
        ("--intra-cores", "28"),
        ("--intra-cores", "2,2"),
        ("--dispatch-intra-cores", "x"),
        ("--bs-values", "0"),
        ("--comparison-atol", "nan"),
        ("--initial-combine-intra", "28"),
    ],
)
def test_reject_invalid_scan(option, value):
    with pytest.raises(SystemExit):
        sweep.parse_args([option, value])


def test_original_environment_cleared_and_pair_command(tmp_path):
    inherited = {sweep.ENV_DISPATCH: "7", sweep.ENV_COMBINE: "8", "LD_LIBRARY_PATH": "/sdk/lib"}
    baseline = sweep.candidate_environment(None, inherited)
    assert sweep.ENV_DISPATCH not in baseline and sweep.ENV_COMBINE not in baseline
    pair = sweep.candidate_environment((12, 7), inherited)
    assert pair[sweep.ENV_DISPATCH] == "12" and pair[sweep.ENV_COMBINE] == "7"
    assert pair["LD_LIBRARY_PATH"] == "/sdk/lib"
    args = sweep.parse_args([])
    command = sweep.command(args, tmp_path / "model.json", tmp_path / "test", (12, 7), tmp_path / "base")
    assert "--no-timer-raw" in command and "--comm-report" in command and "0-15" in command
    assert "--save-output-reference" not in command


def test_selection_uses_slowest_rank_not_fastest():
    records = [
        dict(passed=True, pair=[7, 7], operator_rank_max_mean_us=110, operator_rank_min_mean_us=80),
        dict(passed=True, pair=[8, 8], operator_rank_max_mean_us=100, operator_rank_min_mean_us=90),
    ]
    assert sweep.choose(records)["pair"] == [8, 8]


def test_staged_search_retains_first_stage_winner_outside_combine_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    monkeypatch.setattr(sweep, "execute", lambda *a, **k: None)
    monkeypatch.setattr(sweep, "print_case", lambda *a: None)
    selections = []
    monkeypatch.setattr(sweep, "write_summary", lambda path, plan, records, chosen: selections.extend(chosen))
    durations = {None: 200, (2, 12): 90, (4, 12): 100, (2, 2): 150, (2, 4): 130}

    def read(output, bs, pair):
        return dict(pair=list(pair) if pair else None, passed=True, operator_rank_max_mean_us=durations[pair])

    monkeypatch.setattr(sweep, "read_case", read)
    sweep.main(["--output-dir", str(tmp_path / "run"), "--bs-values", "2048", "--intra-cores", "2,4"])
    assert selections[-1]["pair"] == [2, 12]
    assert selections[-1]["confirmation_us"] == 90


def test_case_reader_accepts_complete_data_and_rejects_incomplete_ranks(tmp_path):
    pair, bs = (12, 12), 16
    for rank in range(16):
        directory = tmp_path / f"rank_{rank}"
        directory.mkdir()
        manifest = dict(
            rank=rank,
            ep_size=16,
            mode="perf",
            routing="balanced",
            input="fp8",
            comm_core_split={sweep.ENV_DISPATCH: "12", sweep.ENV_COMBINE: "12"},
            config=dict(bs_values=[bs]),
            build={"test": True},
        )
        (directory / "manifest.json").write_text(json.dumps(manifest))
    rank0 = tmp_path / "rank_0" / "bs_16"
    rank0.mkdir()
    profile = dict(timer_pass=False, per_rank_rounds_us=[[100.0 + rank, 102.0 + rank] for rank in range(16)])
    (rank0 / "profiling.json").write_text(json.dumps(profile))
    checks = [dict(rank=rank, passed=True) for rank in range(16)]
    (rank0 / "output_comparison.json").write_text(json.dumps(checks))
    (rank0 / "timer_accuracy.json").write_text(json.dumps(checks))
    timer = tmp_path / "bs_16" / "timer"
    timer.mkdir(parents=True)
    reports = [comm_report.rank_report(events(rank), rank, bs, 3584, 8, 12, 12) for rank in range(16)]
    (timer / "communication.json").write_text(json.dumps(comm_report.summarize(reports)))
    report = sweep.read_case(tmp_path, bs, pair)
    assert report["operator_rank_max_mean_us"] == 116.0
    assert report["operator_rank_min_mean_us"] == 101.0
    (rank0 / "timer_accuracy.json").write_text(json.dumps(checks[:-1]))
    with pytest.raises(ValueError, match="timer-on/off"):
        sweep.read_case(tmp_path, bs, pair)


@pytest.mark.parametrize("hidden,expected", [(3584, 3712), (3712, 3968), (4096, 4224), (6144, 6336), (8192, 8448)])
def test_payload_alignment(hidden, expected):
    assert comm_report.payload_bytes(hidden, "DISPATCH") == expected
    assert comm_report.payload_bytes(hidden, "COMBINE") == hidden * 2


@pytest.mark.parametrize("bs", [1, 3, 2048, 4096, 8192])
@pytest.mark.parametrize("topk", [1, 6, 8, 17])
def test_balanced_counts_independent_enumeration(bs, topk):
    length = bs * topk
    for rank in (0, 3, 7, 15):
        oracle = {"INTRA": 0, "CROSS": 0}
        for source in range(16):
            routes = np.arange(source * length, (source + 1) * length)
            key = "INTRA" if source // 4 == rank // 4 else "CROSS"
            oracle[key] += int(np.count_nonzero(routes % 16 == rank))
        assert comm_report.expected_rows(bs, topk, rank) == oracle


def events(rank=0):
    # Two waves. One core per cohort; other cores may legitimately have no work.
    result = []
    for operation in ("DISPATCH", "COMBINE"):
        for kind, rows, core in (("INTRA", 16, 0), ("CROSS", 48, 12)):
            for wave in (0, 1):
                result.append(
                    dict(
                        rank=rank,
                        core=core,
                        family=f"AIV_{operation}_{kind}",
                        wave=wave,
                        row_count=rows,
                        start_cycle=1000 + 100000 * wave,
                        end_cycle=11000 + 100000 * wave,
                    )
                )
    return result


def test_wave_envelope_not_whole_operator_or_sum_of_cores():
    record = comm_report.rank_report(events(), 0, 16, 3584, 8, 12, 12)
    value = record["classes"]["DISPATCH_INTRA"]
    assert value["rows"] == 32
    assert value["summed_wave_envelopes_us"] == 20  # excludes the gap between waves
    assert value["gbps"] == pytest.approx(32 * 3712 / 20 / 1000)
    records = [comm_report.rank_report(events(rank), rank, 16, 3584, 8, 12, 12) for rank in range(16)]
    assert comm_report.summarize(records)["classes"]["DISPATCH_INTRA"]["gbps_per_card_mean"] == value["gbps"]


@pytest.mark.parametrize("mutation", ["missing", "row_count", "wrong_cohort", "wave", "duration"])
def test_reject_missing_or_corrupt_timer_rows(mutation):
    sample = events()
    if mutation == "missing":
        sample.pop()
    elif mutation == "row_count":
        sample[0]["row_count"] += 1
    elif mutation == "wrong_cohort":
        sample[0]["core"] = 12
    elif mutation == "wave":
        sample[0]["wave"] = -1
    else:
        sample[0]["end_cycle"] = sample[0]["start_cycle"]
    with pytest.raises(ValueError):
        comm_report.rank_report(sample, 0, 16, 3584, 8, 12, 12)


def test_sparse_raw_protocol_extraction():
    class SparseRaw:
        def __init__(self):
            self.values = {}

        def __len__(self):
            return p.TOTAL_BUFFER_SIZE

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.values.get(i, 0) for i in range(index.start, index.stop)]
            return self.values.get(index, 0)

    raw = SparseRaw()
    core, family = 2, p.DYNAMIC_NAMES.index("AIV_DISPATCH_INTRA")
    logical = len(p.FIXED_NAMES) + family * p.MAX_DYNAMIC_ITER
    raw.values[p.N_TIMING_COUNTER + core * p.DYNAMIC_ITER_PER_CORE + family] = 1
    raw.values[core * p.COUNTERS_PER_CORE + logical * 2] = 1000
    raw.values[core * p.COUNTERS_PER_CORE + logical * 2 + 1] = 2000
    offset = (
        p.TASK_METADATA_OFFSET + (core * len(p.DYNAMIC_NAMES) * p.MAX_DYNAMIC_ITER + family * p.MAX_DYNAMIC_ITER) * 2
    )
    raw.values[offset], raw.values[offset + 1] = 1 << 32, 7
    raw.values[p.WAVE_ID_OFFSET + core * p.WAVE_ID_PER_CORE + logical] = 3
    event = comm_report.read_events(raw, 4)[0]
    assert event["row_count"] == 7 and event["wave"] == 3 and event["core"] == 0
    assert event["granularity"] == "cohort_slice"


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
