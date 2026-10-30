"""CPU-only contracts for exact URMA profiler capture and rank aggregation."""

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.urma_timing import (
    measure_profiled_forward,
    read_profiled_kernels,
    summarize_profile_rounds,
)


def write_samples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "Stream ID", "Device_id"])
        writer.writerows(rows)


def kernel(start, duration, name="mega_moe_urma_kernel", stream=3, device=0):
    return [name, "AI_MIX_AIC", start, duration, stream, device]


def test_only_exact_urma_device_tasks_are_selected(tmp_path):
    write_samples(
        tmp_path / "kernel_details.csv",
        [
            kernel(3, 3, "void MegaMoe::mega_moe_urma_kernel(Arguments)"),
            kernel(1, 1),
            kernel(2, 2, "mega_moe_urma_kernel<true>"),
            *[
                kernel(0, 99, name)
                for name in (
                    "Exp",
                    "HcclAllReduce",
                    "mega_moe_kernel",
                    "mega_moe_urma_kernel_do",
                    "my_mega_moe_urma_kernel",
                    "_Z23mega_moe_urma_kernel_doN11MegaMoeUrma9ArgumentsE",
                    "_Z26my_mega_moe_urma_kernelN11MegaMoeUrma9ArgumentsE",
                )
            ],
        ],
    )
    samples = read_profiled_kernels(tmp_path, 3)
    assert [sample["duration_us"] for sample in samples] == [1, 2, 3]
    assert all(sample["stream_id"] == "3" and sample["device_id"] == "0" for sample in samples)
    json.dumps(samples, allow_nan=False)


def test_cann_mangled_kernel_name_from_ubx_capture(tmp_path, monkeypatch):
    directory, log = tmp_path / "profile", []
    name = "_Z20mega_moe_urma_kernelN11MegaMoeUrma9ArgumentsE"
    rows = [kernel(i, i + 1, name) for i in range(41)]
    fake_profiler(monkeypatch, directory, rows, log)
    guard = SimpleNamespace(begin=lambda: None, end=lambda: None, before_forward=lambda: None)
    result = measure_profiled_forward(
        lambda: None, warmup=20, iters=20, guard=guard, directory=directory, last_forward=lambda: None
    )
    assert len(result["samples"]) == 41
    assert result["durations_us"] == list(range(21, 41))
    assert result["diagnostic_sample"]["duration_us"] == 41
    assert result["diagnostic_sample"]["phase"] == "diagnostic"
    assert all(sample["name"] == name for sample in result["samples"])


def test_decimal_timestamp_order_survives_float_rounding_across_shards(tmp_path):
    write_samples(tmp_path / "a/kernel_details.csv", [kernel("123456789012345678.002", 20)])
    write_samples(tmp_path / "b/kernel_details_1.csv", [kernel("123456789012345678.001", 10)])
    samples = read_profiled_kernels(tmp_path, 2)
    assert samples[0]["start_us"] == samples[1]["start_us"]
    assert [sample["duration_us"] for sample in samples] == [10, 20]
    assert samples[0]["start_us_decimal"].endswith(".001")


@pytest.mark.parametrize(
    "rows,expected,message",
    [
        ([], 1, "expected 1"),
        ([kernel(1, 2)], 2, "expected 2"),
        ([kernel(1, 2), kernel("1.0", 2)], 2, "Duplicate"),
        ([kernel(1, 2), kernel(2, 2, stream=4)], 2, "Multiple"),
        ([kernel(1, 2), kernel(2, 2, device=1)], 2, "Multiple"),
        ([kernel("bad", 2)], 1, "Invalid"),
        ([kernel(1, 0)], 1, "Nonfinite"),
        ([kernel(1, -2)], 1, "Nonfinite"),
        ([kernel(-1, 2)], 1, "Nonfinite"),
        ([kernel("NaN", 2)], 1, "Nonfinite"),
        ([kernel(1, "Infinity")], 1, "Nonfinite"),
        ([kernel(1, "1e1000")], 1, "numeric range"),
    ],
)
def test_invalid_profiler_rows_fail_closed(tmp_path, rows, expected, message):
    write_samples(tmp_path / "kernel_details.csv", rows)
    with pytest.raises(ValueError, match=message):
        read_profiled_kernels(tmp_path, expected)


def fake_profiler(monkeypatch, directory, rows, log):
    config = {}

    class Profile:
        def __enter__(self):
            log.append("profile-enter")
            return self

        def step(self):
            log.append("profile-step")

        def __exit__(self, *args):
            log.append("profile-export")
            write_samples(directory / "kernel_details.csv", rows)

    def profile(**kwargs):
        config.update(kwargs)
        return Profile()

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: kwargs,
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level0="level0"),
        AiCMetrics=SimpleNamespace(AiCoreNone="none"),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        schedule=lambda **kwargs: kwargs,
        tensorboard_trace_handler=lambda path: path,
        profile=profile,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    return config


@pytest.mark.parametrize("diagnostic", [False, True])
def test_one_continuous_profile_block_preserves_alignment_and_last_trace_only(tmp_path, monkeypatch, diagnostic):
    directory, log = tmp_path / "profile", []
    rows = [kernel(i, duration) for i, duration in enumerate([99, 100, 5, 6] + ([88] if diagnostic else []))]
    config = fake_profiler(monkeypatch, directory, rows, log)
    # No torch/Event attribute exists: profiling must neither instantiate an
    # Event nor perform extra synchronization apart from the guard's one drain.
    guard = SimpleNamespace(
        begin=lambda: log.extend(["barrier", "exp"]),
        before_forward=lambda: log.append("all-reduce"),
        end=lambda: log.extend(["barrier", "synchronize"]),
    )
    result = measure_profiled_forward(
        lambda: log.append("forward"),
        warmup=2,
        iters=2,
        guard=guard,
        directory=directory,
        last_forward=(lambda: log.append("diagnostic-with-timer-trace")) if diagnostic else None,
    )
    assert log == [
        "profile-enter",
        "barrier",
        "exp",
        *(["all-reduce", "forward"] * 4),
        *(["all-reduce", "diagnostic-with-timer-trace"] if diagnostic else []),
        "barrier",
        "synchronize",
        "profile-step",
        "profile-export",
    ]
    assert result["durations_us"] == [5, 6]
    assert [sample["phase"] for sample in result["samples"]] == ["warmup", "warmup", "formal", "formal"] + (
        ["diagnostic"] if diagnostic else []
    )
    assert result["diagnostic_sample"] == (result["samples"][-1] if diagnostic else None)
    assert result["profile_dir"] == str(directory)
    assert config["schedule"] == dict(wait=0, warmup=0, active=1, repeat=1)
    assert config["experimental_config"] == dict(
        export_type=["text"], profiler_level="level0", aic_metrics="none", data_simplification=False
    )
    assert all(
        config[field] is False
        for field in ("record_shapes", "profile_memory", "with_stack", "with_flops", "with_modules")
    )
    json.dumps(result, allow_nan=False)


def test_fresh_profile_directory_rejects_old_capture(tmp_path, monkeypatch):
    log = []
    fake_profiler(monkeypatch, tmp_path, [], log)
    with pytest.raises(FileExistsError):
        measure_profiled_forward(lambda: None, warmup=0, iters=1, guard=None, directory=tmp_path)
    assert log == []


@pytest.mark.parametrize("warmup,iters", [(-1, 2), (True, 2), (0, 0), (0, 1.5)])
def test_iteration_counts_are_validated_before_launch(tmp_path, monkeypatch, warmup, iters):
    log = []
    fake_profiler(monkeypatch, tmp_path, [], log)
    with pytest.raises(ValueError, match="warmup"):
        measure_profiled_forward(lambda: None, warmup=warmup, iters=iters, guard=None, directory=tmp_path)
    assert log == []


def test_all_formal_rounds_aggregate_extrema_before_average():
    rows = [[100, 1, 20], [2, 90, 30], [3, 4, 40], [4, 5, 50]]
    report = summarize_profile_rounds(rows, ranks_per_node=2)
    assert report["rank_max_rounds_us"] == [100, 90, 50]
    assert report["rank_min_rounds_us"] == [2, 1, 20]
    assert report["rank_max_mean_us"] == 80
    assert report["rank_min_mean_us"] == pytest.approx(23 / 3)
    assert report["rank_max_mean_us"] != max(sum(row) / 3 for row in rows)
    assert report["rounds"] == report["recorded_rounds"] == 3
    assert report["first_selected_iteration"] == 0
    assert report["iteration_selection"] == "all_formal_rounds"
    assert report["raw_per_rank_rounds_us"] == rows
    assert [node["rank_start"] for node in report["per_node"]] == [0, 2]
    assert [node["rank_max_mean_us"] for node in report["per_node"]] == [pytest.approx(220 / 3), pytest.approx(59 / 3)]
    report["raw_per_rank_rounds_us"][0][0] = 0
    assert rows[0][0] == 100


@pytest.mark.parametrize("rows", [[], [[]], [[1], [2, 3]], [[float("nan")]], [[0]], [[-1]], [[True]]])
def test_invalid_formal_samples_cannot_produce_a_report(rows):
    with pytest.raises(ValueError):
        summarize_profile_rounds(rows, ranks_per_node=1)


def test_node_size_must_divide_world():
    with pytest.raises(ValueError, match="divide"):
        summarize_profile_rounds([[1], [2]], ranks_per_node=3)
