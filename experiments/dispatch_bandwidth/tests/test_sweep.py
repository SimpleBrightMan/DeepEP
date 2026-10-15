# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""CPU-only sweep checks; every generated timing fixture is synthetic test data."""

import builtins
import copy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.dispatch_bandwidth import plot_sweep, report, run, sweep  # noqa: E402


def synthetic_native(args, batch, rank):
    """Invent timestamps only inside tests, keeping real routing and padding contracts."""
    world, cores = len(args.devices), args.cores
    items = batch * args.topk
    rows = 0
    for peer in range(world):
        if peer // args.group_size != rank // args.group_size:
            first = peer * items + (rank - peer * items) % world
            rows += len(range(first, (peer + 1) * items, world))
    row_bytes = (args.hidden + 255) // 256 * 256 + (args.hidden // 32 + 31) // 32 * 32
    rounds = []
    for iteration in range(args.iters):
        # Widely separated per-NPU clocks must not change bandwidth or the rank band.
        origin = 2**60 + rank * 10**12 + iteration * 10**9
        duration = int(rows * args.repeat * row_bytes / (15 + cores + rank / 4 + iteration / 2))
        base, extra = divmod(rows, cores)
        workers = [
            dict(
                group_id=core,
                traffic="cross",
                rows=(base + int(core < extra)) * args.repeat,
                start_cycle=origin + core * 10,
                end_cycle=origin + duration,
            )
            for core in range(cores)
        ]
        rounds.append(dict(iteration=iteration, kernel_elapsed_us=duration / 1000 + 10, workers=workers))
    return dict(
        schema_version=3,
        rank=rank,
        world=world,
        device=args.devices[rank],
        cores=cores,
        group_size=args.group_size,
        batch=batch,
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        topk=args.topk,
        scope=args.scope,
        traffic_selection=args.traffic,
        metadata_mode=args.metadata_mode,
        row_bytes=row_bytes,
        scale_elements=args.hidden // 32,
        buffer_count=args.buffer_count,
        repeats=args.repeat,
        cycles_per_us=1000.0,
        traffic_rows=dict(local=0, group=0, cross=rows),
        worker_counts=dict(local=0, group=0, cross=cores),
        validation=dict(passed=True, checked_rows=rows),
        rounds=rounds,
    )


def synthetic_runner(args, batch, directory):
    """Stand in for one fresh native process group without querying or opening an NPU."""
    directory.mkdir(exist_ok=False)
    raws = [synthetic_native(args, batch, rank) for rank in range(len(args.devices))]
    for raw in raws:
        (directory / f"rank_{raw['rank']}.json").write_text(json.dumps(raw), encoding="utf-8")
        (directory / f"rank_{raw['rank']}.log").write_text("SYNTHETIC TEST DATA\n", encoding="utf-8")
    summary = report.summarize_reports(raws, expected_world=len(args.devices), expected_iterations=args.iters)
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


@pytest.fixture
def collected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_resource_snapshot", lambda directory: None)
    args = sweep.parse_args(
        [
            "--hidden-values",
            "3712,8192",
            "--core-values",
            "2,12",
            "--buffer-count",
            "6",
            "--iters",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    calls = []

    def runner(case, batch, directory):
        calls.append((case.hidden, case.cores, directory))
        return synthetic_runner(case, batch, directory)

    path = sweep.collect_sweep(args, runner=runner, data_kind="synthetic_test")
    return args, path, calls


def test_default_plan_has_sixty_exact_cases_without_side_effects(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("A plan must not launch workers, query resources, or import plotting components")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("matplotlib") or name.endswith("plot_sweep"):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(run, "run_case", forbidden)
    monkeypatch.setattr(run, "_resource_snapshot", forbidden)
    output = tmp_path / "must-not-exist"
    assert sweep.main(["--plan", "--output-dir", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert data["expected_cases"] == 60 and data["buffer_counts"] == [1, 2, 3, 4, 5, 6]
    assert len(data["slot_sweeps"]) == 6
    for item in data["slot_sweeps"]:
        plan = item["plan"]
        assert len(plan["cases"]) == plan["expected_cases"] == 10
        assert plan["expected"]["devices"] == list(range(16))
        assert plan["expected"]["buffer_count"] == item["buffer_count"]
        assert plan["expected"]["hidden_values"] == [3584]
        assert [(case["hidden"], case["cores"]) for case in plan["cases"]] == [
            (3584, core) for core in sweep.DEFAULT_CORES
        ]
        for case in plan["cases"]:
            options = run.parse_args(case["command"][3:])
            assert options.traffic == "cross" and options.workers == [0, 0, case["cores"]]
            assert options.metadata_mode == "dma" and options.timer
            assert options.buffer_count == item["buffer_count"]
            assert options.hidden == 3584 and options.bs_values == [4096]


def test_single_slot_option_preserves_original_plan_shape():
    args = sweep.parse_args(["--buffer-count", "3", "--plan"])
    data = sweep.make_scan_plan(args)
    assert args.buffer_counts == [3] and data["expected"]["buffer_count"] == 3
    assert data["mode"] == "cross_bandwidth_sweep" and data["expected_cases"] == 10


def test_custom_slot_and_hidden_matrices_are_preserved_in_order():
    args = sweep.parse_args(["--buffer-counts", "6,1,3", "--hidden-values", "3584,4096", "--plan"])
    data = sweep.make_scan_plan(args)
    assert data["expected_cases"] == 60
    assert [item["buffer_count"] for item in data["slot_sweeps"]] == [6, 1, 3]
    assert all(item["plan"]["expected"]["hidden_values"] == [3584, 4096] for item in data["slot_sweeps"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--devices", "0,1,2,3"],
        ["--devices", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,14"],
        ["--target-gbps", "nan"],
        ["--core-values", "2,2"],
        ["--core-values", "29"],
        ["--core-values", "32"],
        ["--buffer-count", "0"],
        ["--buffer-counts", "1,7"],
        ["--buffer-counts", "2,2"],
        ["--buffer-count", "2", "--buffer-counts", "3,4"],
    ],
)
def test_invalid_plan_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        sweep.parse_args(argv + ["--plan"])


@pytest.mark.parametrize("cores", [29, 32])
def test_saved_sweep_rejects_active_workers_above_fixed_launch_count(cores):
    expected = sweep.expected_configuration(sweep.parse_args(["--plan"]))
    expected["core_values"] = [cores]
    data = dict(
        schema_version=1,
        mode="cross_bandwidth_sweep",
        status="passed",
        data_kind="synthetic_test",
        expected=expected,
        cases=[],
    )
    with pytest.raises(sweep.SweepError, match="Invalid expected core count"):
        sweep.validate_sweep(data)


def test_collection_is_sequential_complete_and_keeps_rank_artifacts(collected):
    args, path, calls = collected
    data = sweep.load_sweep(path)
    assert data["data_kind"] == "synthetic_test" and data["status"] == "passed"
    assert [(hidden, cores) for hidden, cores, _ in calls] == [(3712, 2), (3712, 12), (8192, 2), (8192, 12)]
    assert len({directory for _, _, directory in calls}) == 4
    for _, _, directory in calls:
        assert len(list(directory.glob("rank_*.json"))) == 16
        assert (directory / "rank_15.log").read_text(encoding="utf-8").startswith("SYNTHETIC")
    assert data["cases"][0]["summary"]["configuration"]["row_bytes"] == 3968
    assert data["cases"][0]["summary"]["classes"]["cross"]["rows_per_round_per_card"]["mean"] == 24576
    with path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {row["sweep_status"] for row in rows} == {"passed"}
    assert {row["metadata_mode"] for row in rows} == {"dma"}
    assert all(
        float(row["rank_min_gbps_card"]) <= float(row["mean_gbps_card"]) <= float(row["rank_max_gbps_card"])
        for row in rows
    )
    assert sweep.make_plan(args)["expected"] == data["expected"]
    plot_sweep.verify_raw_reports(path, data)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_case",
        "duplicate_case",
        "missing_rank",
        "failed_validation",
        "nan",
        "wrong_mode",
        "wrong_bytes",
        "wrong_rank_mean",
        "missing_round",
        "wrong_core",
        "missing_world",
        "malformed_case",
    ],
)
def test_malformed_or_partial_results_never_become_a_successful_plot(collected, damage):
    _, path, _ = collected
    data = copy.deepcopy(sweep.load_sweep(path))
    case = data["cases"][0]
    summary = case["summary"]
    if damage == "missing_case":
        data["cases"].pop()
    elif damage == "duplicate_case":
        data["cases"][-1] = copy.deepcopy(case)
    elif damage == "missing_rank":
        summary["per_rank"].pop()
    elif damage == "failed_validation":
        summary["per_rank"][0]["validation"]["passed"] = False
    elif damage == "nan":
        summary["per_rank"][0]["classes"]["cross"]["mean"] = float("nan")
    elif damage == "wrong_mode":
        summary["configuration"]["metadata_mode"] = "preload"
    elif damage == "wrong_bytes":
        summary["configuration"]["row_bytes"] = 3712 + 116
    elif damage == "wrong_rank_mean":
        summary["classes"]["cross"]["rank_mean_gbps_per_card"]["mean"] += 1
    elif damage == "missing_round":
        summary["per_round"].pop()
    elif damage == "wrong_core":
        summary["configuration"]["cores"] = 28
    elif damage == "missing_world":
        del data["expected"]["world"]
    elif damage == "malformed_case":
        data["cases"][0] = None
    with pytest.raises(sweep.SweepError):
        sweep.validate_sweep(data)


def test_failed_case_preserves_partial_results_without_missing_point_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_resource_snapshot", lambda directory: None)
    args = sweep.parse_args(
        ["--hidden-values", "3712", "--core-values", "2,4,6", "--iters", "1", "--output-dir", str(tmp_path)]
    )

    def fail_second(case, batch, directory):
        if case.cores == 4:
            raise RuntimeError("synthetic test failure")
        return synthetic_runner(case, batch, directory)

    with pytest.raises(sweep.SweepError, match="partial results retained"):
        sweep.collect_sweep(args, runner=fail_second, data_kind="synthetic_test")
    paths = list(tmp_path.glob("cross_*/sweep.json"))
    assert len(paths) == 1
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert [(case["cores"], case["status"]) for case in data["cases"]] == [(2, "passed"), (4, "failed")]
    with paths[0].with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["cores"] == "2" and rows[0]["sweep_status"] == "failed"
    with pytest.raises(sweep.SweepError):
        sweep.load_sweep(paths[0])


def test_raw_replay_rejects_changed_native_report(collected):
    _, path, _ = collected
    data = sweep.load_sweep(path)
    raw_path = path.parent / data["cases"][0]["case_dir"] / "rank_15.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["metadata_mode"] = "scalar"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(report.ReportError):
        plot_sweep.verify_raw_reports(path, data)


def test_main_passes_custom_matrix_to_real_plot_and_exports_nonempty_pngs(collected, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    args, path, _ = collected
    axes = []
    original_subplots = plt.subplots

    def record_axes(*args, **kwargs):
        figure, axis = original_subplots(*args, **kwargs)
        axes.append(axis)
        return figure, axis

    monkeypatch.setattr(plt, "subplots", record_axes)
    monkeypatch.setattr(sweep, "collect_sweep", lambda requested: path)
    # The standalone default plot requires the full matrix rather than silently calling this complete.
    with pytest.raises(sweep.SweepError, match="matrix"):
        plot_sweep.plot_sweep(path)
    assert (
        sweep.main(
            [
                "--hidden-values",
                ",".join(map(str, args.hidden_values)),
                "--core-values",
                "2,12",
                "--iters",
                "2",
                "--buffer-count",
                "6",
                "--output-dir",
                str(args.output_dir),
                "--svg",
            ]
        )
        == 0
    )
    data = sweep.load_sweep(path)
    outputs = [path.parent / relative for relative in data["plots"]]
    assert len(outputs) == 4 and sum(output.suffix == ".png" for output in outputs) == 2
    for output in outputs:
        assert output.stat().st_size > 1000
        if output.suffix == ".png":
            assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(axes) == 2 and axes[0].get_ylim() == axes[1].get_ylim()
    for axis in axes:
        assert list(axis.lines[0].get_xdata()) == [2, 12]
        assert list(axis.get_xticks()) == [2, 12]
        assert list(axis.lines[1].get_ydata()) == [134, 134]
        assert "SYNTHETIC TEST DATA" in axis.get_title()
        assert "metadata=dma" in axis.get_title() and "EP=16" in axis.get_title()
        assert "group=4" in axis.get_title() and "slots=6" in axis.get_title()
    assert "record=3968 B" in axes[0].get_title()


def test_missing_plotting_components_fail_before_collection(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("synthetic missing plotting dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setattr(sweep, "collect_sweep", lambda args: pytest.fail("Must check plotting before measurement"))
    assert sweep.main(["--hidden-values", "3712", "--core-values", "2", "--output-dir", str(tmp_path / "absent")]) == 1
    assert not (tmp_path / "absent").exists()


def test_multiple_slots_keep_independent_summaries_csvs_and_six_plots(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(run, "_resource_snapshot", lambda directory: None)
    calls = []

    def runner(args, batch, directory):
        calls.append((args.buffer_count, args.hidden, args.cores))
        return synthetic_runner(args, batch, directory)

    original_collect = sweep.collect_sweep
    monkeypatch.setattr(
        sweep,
        "collect_sweep",
        lambda args: original_collect(args, runner=runner, data_kind="synthetic_test"),
    )
    assert sweep.main(["--core-values", "2,12", "--iters", "1", "--output-dir", str(tmp_path)]) == 0
    assert calls == [(slots, 3584, cores) for slots in range(1, 7) for cores in (2, 12)]
    paths = list(tmp_path.glob("cross_slots_*/slot_sweeps.json"))
    assert len(paths) == 1
    index_path = paths[0]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["status"] == "passed" and index["expected_cases"] == 12
    assert len(index["results"]) == 6
    for item in index["results"]:
        assert item["status"] == "passed" and len(item["plots"]) == 1
        path = index_path.parent / item["sweep_json"]
        assert path.is_relative_to(index_path.parent / f"slots_{item['buffer_count']}")
        data = sweep.load_sweep(path)
        assert data["expected"]["buffer_count"] == item["buffer_count"]
        assert data["expected"]["hidden_values"] == [3584]
        plot_sweep.verify_raw_reports(path, data)
        with path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 and {int(row["buffer_count"]) for row in rows} == {item["buffer_count"]}
        plot = index_path.parent / item["plots"][0]
        assert plot.stat().st_size > 1000 and plot.name == "cross_bandwidth_h3584.png"


def test_multi_slot_failure_stops_and_indexes_partial_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_resource_snapshot", lambda directory: None)
    args = sweep.parse_args(
        ["--buffer-counts", "2,3,4", "--core-values", "2", "--iters", "1", "--output-dir", str(tmp_path)]
    )
    calls = []

    def collect(selected):
        calls.append(selected.buffer_count)
        if selected.buffer_count == 3:
            raise sweep.SweepError("synthetic test failure")
        return selected.output_dir / "cross_synthetic/sweep.json", []

    monkeypatch.setattr(sweep, "collect_and_plot", lambda selected, plotter: collect(selected))
    with pytest.raises(sweep.SweepError, match="partial results retained"):
        sweep.collect_slot_sweeps(args, None)
    assert calls == [2, 3]
    index = json.loads(next(tmp_path.glob("cross_slots_*/slot_sweeps.json")).read_text(encoding="utf-8"))
    assert index["status"] == "failed"
    assert [(item["buffer_count"], item["status"]) for item in index["results"]] == [(2, "passed"), (3, "failed")]
