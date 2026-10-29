"""CPU integration contracts for the URMA profiler-only benchmark runner."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import urma_benchmark as benchmark
from deep_ep_experimental.megamoe import api, reference, urma, urma_timer, urma_timing, urma_trace_merge, validation


@pytest.mark.parametrize("diagnostic_option", (None, "--timer", "--trace"))
def test_run_profiles_all_formal_rounds_and_only_instruments_final_call(
    monkeypatch, tmp_path, capsys, diagnostic_option
):
    """Run the real BS loop while replacing hardware and profiler I/O only."""
    calls = []
    profile_durations = [10.0, 80.0, 20.0]
    per_rank_durations = [profile_durations, [90.0, 5.0, 30.0]] + [[40.0, 40.0, 40.0] for _ in range(14)]
    config = dict(
        seed=7,
        bs_values=[2048, 4096, 8192],
        warmup=2,
        iters=3,
        model=dict(hidden=2, intermediate_hidden=4, num_experts=384, num_topk=8, num_shared_experts=0),
    )
    argv = ["--output-dir", str(tmp_path), "--exp-shape", "2", "3", "--exp-repeat", "1"]
    if diagnostic_option:
        argv.append(diagnostic_option)
    args = benchmark.parse_args(argv)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def npu(self):
            return self

        def view(self, dtype):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def numel(self):
            return self.data.size

    def forbidden_event(*unused_args, **unused_kwargs):
        pytest.fail("The benchmark must not construct timing Events")

    torch = ModuleType("torch")
    torch.float8_e4m3fn = "fp8"
    torch.float8_e8m0fnu = "scale"
    torch.bfloat16 = "bf16"
    torch.from_numpy = Tensor
    torch.empty = lambda shape, **unused: Tensor(np.zeros(shape))
    torch.npu = SimpleNamespace(Event=forbidden_event, synchronize=lambda: calls.append(("synchronize",)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        benchmark,
        "inputs",
        lambda config, rank, bs, routing, world: (
            np.zeros((bs, 2)),
            np.zeros((bs, 8), dtype=np.int64),
            np.ones((bs, 8)),
        ),
    )
    monkeypatch.setattr(reference, "quantize_input", lambda data: (data, np.ones((len(data), 1))))
    monkeypatch.setattr(
        reference,
        "make_weights",
        lambda *unused: ((np.zeros(1), np.ones(1)), (np.zeros(1), np.ones(1))),
    )
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda *pairs: pairs)

    class Group:
        def rank(self):
            return 0

        def size(self):
            return 16

        def all_gather(self, value, stage):
            calls.append(("gather", stage))
            if stage.startswith("URMA profile samples"):
                assert value == profile_durations
                return per_rank_durations
            return [value] * 16

        def barrier(self, stage):
            calls.append(("barrier", stage))

        def close(self):
            calls.append(("group-close",))

    group = Group()

    def collective_check(group, stage, error, *unused):
        assert error is None
        calls.append(("check", stage))

    monkeypatch.setattr(validation, "collective_check", collective_check)

    class Buffer:
        timer_enabled = True
        max_recv_rows = 200000
        symmetric_buffer_bytes = 1234

        def __init__(self, group, **kwargs):
            calls.append(("owner-create", kwargs))
            self.prepared = None
            self.ordinal = 0

        def prepare_routes(self, ids):
            bs = len(ids.data)
            calls.append(("prepare", bs))
            self.prepared = SimpleNamespace(
                bs=bs, received_rows=bs * 8, native=SimpleNamespace(gmm1_startup_mgroups=12)
            )
            self.ordinal = 0
            return self.prepared

        def allocate_timer(self):
            calls.append(("allocate-timer", self.prepared.bs))
            return Tensor(np.zeros(1, dtype=np.int64))

        def allocate_trace(self, capacity):
            calls.append(("allocate-trace", self.prepared.bs, capacity))
            return Tensor(np.zeros(1, dtype=np.int64))

        def forward_prepared(self, x, topk_weights, *weights_and_prepared, timer=None, trace=None, out=None):
            assert weights_and_prepared[-1] is self.prepared
            assert out is not None
            self.ordinal += 1
            calls.append(("forward", self.prepared.bs, self.ordinal, timer is not None, trace is not None))
            for buffer in (timer, trace):
                if buffer is not None:
                    buffer.data[0] = self.ordinal
            return out

        def destroy(self):
            calls.append(("owner-destroy",))

    extension = SimpleNamespace(
        urma_api_protocol=1,
        urma_compute_schedule="expert-m256-gateup128-lookahead-v2",
        urma_local_pipeline="multi-slot",
        urma_tail_schedule="combine-ready-scan-reduce-world-v1",
        urma_timer_schema=1,
    )
    monkeypatch.setattr(urma, "UrmaBuffer", Buffer)
    monkeypatch.setattr(urma, "_extension", lambda: extension)

    class Guard:
        def __init__(self, group, shape, repeat, timeout, *, backend="hccl"):
            calls.append(("guard-create", tuple(shape), repeat, timeout, backend))

        def describe(self):
            return {"allocated_bytes_per_rank": 48}

        def begin(self):
            calls.append(("guard-begin",))

        def end(self):
            calls.append(("guard-end",))

        def close(self):
            calls.append(("guard-close",))

    def profile(forward, *, warmup, iters, guard, directory, last_forward=None):
        assert (warmup, iters) == (2, 3)
        bs = int(directory.parent.name.removeprefix("bs_"))
        calls.append(("profile-begin", bs))
        guard.begin()
        for _ in range(warmup + iters):
            forward()
        if last_forward is not None:
            last_forward()
        guard.end()
        calls.append(("profile-end", bs))
        durations = [9999.0, 4999.0, *profile_durations] + ([7777.0] if last_forward is not None else [])
        samples = [dict(iteration=index, duration_us=value) for index, value in enumerate(durations)]
        return dict(
            samples=samples,
            durations_us=profile_durations,
            diagnostic_sample=samples[-1] if last_forward is not None else None,
            profile_dir=str(directory),
        )

    monkeypatch.setattr(urma_timing, "UrmaLaunchGuard", Guard)
    monkeypatch.setattr(urma_timing, "measure_profiled_forward", profile)

    def bandwidth(raw, rank, *, world, clock_mhz):
        assert world == group.size()
        calls.append(("timer-export", int(raw[0])))
        assert int(raw[0]) == 6
        return {"rank": rank}

    monkeypatch.setattr(urma_timer, "communication_bandwidth", bandwidth)
    monkeypatch.setattr(urma_timer, "summarize_bandwidth", lambda records: [])

    def export_trace(group, raw, case, *, bs, clock_mhz, include_waits=False):
        calls.append(("trace-export", bs, int(raw[0])))
        assert int(raw[0]) == 6
        return [{"rank": rank, "chrome_path": str(case / f"rank{rank}.json")} for rank in range(16)]

    def merge_trace(group, records, path, *, bs, include_waits=False):
        calls.append(("trace-merge", bs, len(records), str(path)))
        return {"path": str(path), "ranks": 16}

    monkeypatch.setattr(benchmark, "export_trace_collective", export_trace)
    monkeypatch.setattr(urma_trace_merge, "merge_trace_collective", merge_trace)

    benchmark.run(args, config, group)

    has_timer = diagnostic_option is not None
    has_trace = diagnostic_option == "--trace"
    for bs in config["bs_values"]:
        assert calls.count(("prepare", bs)) == 1
        expected_forwards = [("forward", bs, ordinal, False, False) for ordinal in range(1, 6)]
        if has_timer:
            expected_forwards.append(("forward", bs, 6, True, has_trace))
        actual_forwards = [call for call in calls if call[0] == "forward" and call[1] == bs]
        assert actual_forwards == expected_forwards
        begin = calls.index(("profile-begin", bs))
        end = calls.index(("profile-end", bs))
        assert [call[0] for call in calls[begin + 1 : end]] == [
            "guard-begin",
            *(["forward"] * (5 + int(has_timer))),
            "guard-end",
        ]
        assert calls.count(("allocate-timer", bs)) == int(has_timer)
        assert calls.count(("allocate-trace", bs, args.trace_capacity)) == int(has_trace)
        case = tmp_path / "rank_0" / f"bs_{bs}"
        summary = json.loads((case / "summary.json").read_text())
        report = summary["profile"]
        assert report["raw_per_rank_rounds_us"] == per_rank_durations
        assert report["rank_max_rounds_us"] == [90.0, 80.0, 40.0]
        assert report["rank_min_rounds_us"] == [10.0, 5.0, 20.0]
        assert report["rank_max_mean_us"] == pytest.approx(70.0)
        assert report["rank_min_mean_us"] == pytest.approx(35.0 / 3)
        assert report["rounds"] == 3
        assert report["continuous_iterations"] == 5 + int(has_timer)
        assert report["diagnostic_iterations"] == int(has_timer)
        assert len(report["per_node"]) == 2
        assert (summary["diagnostic_sample"] is not None) == has_timer
        assert (case / "urma_timer.bin").exists() == has_timer
        if has_timer:
            assert np.fromfile(case / "urma_timer.bin", dtype="<i8").tolist() == [6]
        if has_trace:
            assert calls.count(("trace-export", bs, 6)) == 1
            assert calls.count(("trace-merge", bs, 16, str(tmp_path / f"bs_{bs}" / "allrank.json"))) == 1
            assert len(summary["trace_per_rank"]) == 16
        else:
            assert "trace_per_rank" not in summary
    assert calls.count(("guard-begin",)) == calls.count(("guard-end",)) == 3
    assert [call[0] for call in calls[-3:]] == ["owner-destroy", "guard-close", "group-close"]
    output = capsys.readouterr().out
    assert output.count("[URMA PROFILE]") == 3
    assert output.count("round-max-mean=70.000us round-min-mean=11.667us rounds=3") == 3
    for obsolete_label in ("ROUTE_PREP", "CACHED_FORWARD", "FULL_E2E"):
        assert obsolete_label not in output
    summaries = json.loads((tmp_path / "summary.json").read_text())
    assert [row["bs"] for row in summaries] == config["bs_values"]
