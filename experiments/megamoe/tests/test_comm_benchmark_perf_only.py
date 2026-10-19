# Copyright (c) 2026, SimpleBright_Man

"""Run the performance-only benchmark control flow with no NPU or giant buffers."""

from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api, comm_benchmark as bench, comm_report, validation, wave_oracle
from deep_ep_experimental.megamoe.timer import breakdown


@pytest.fixture
def mocked_run(monkeypatch, tmp_path):
    state = SimpleNamespace(forbidden=[], timer_allocations=[], cpu_reads=[], calls=[], destroyed=False, closed=False)

    def forbidden(name):
        def fail(*_args, **_kwargs):
            state.forbidden.append(name)
            raise AssertionError(f"Performance-only run called {name}")

        return fail

    class Tensor:
        def __init__(self, data, label="device"):
            self.data, self.label = np.asarray(data), label

        def __getitem__(self, item):
            return Tensor(self.data[item], self.label)

        def copy_(self, other):
            self.data[...] = other.data
            return self

        def fill_(self, value):
            self.data.fill(value)
            return self

        def npu(self):
            return self

        def view(self, _dtype):
            return self

        def float(self):
            return forbidden("output float conversion")()

        def cpu(self):
            state.cpu_reads.append(self.label)
            if self.label != "timer":
                return forbidden("payload/output/count D2H")()
            return self

        def numpy(self):
            return self.data

    torch = ModuleType("torch")
    for dtype in ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16", "float32", "uint8"):
        setattr(torch, dtype, dtype)
    torch.npu = SimpleNamespace(synchronize=lambda: None)
    torch.from_numpy = Tensor
    torch.empty = lambda shape, **_kwargs: Tensor(np.zeros(shape, dtype=np.float32))
    torch.zeros = torch.empty
    torch.exp = lambda _prefix, *, out: out
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Buffer:
        timer_enabled = True

        def __init__(self, _group, *, num_max_tokens_per_rank, **model):
            bs, hidden, topk = num_max_tokens_per_rank, model["hidden"], model["num_topk"]
            self.x = Tensor(np.zeros((bs, hidden), dtype=np.uint8))
            self.x_sf = Tensor(np.zeros((bs, hidden // 32), dtype=np.uint8))
            self.topk_idx = Tensor(np.zeros((bs, topk), dtype=np.int32))
            self.topk_weights = Tensor(np.zeros((bs, topk), dtype=np.float32))

        def prepare_communication(self, bs, *, combine_payload, dispatch_metadata_mode, combine_buffer_count):
            state.prepared_rows = combine_payload.data.shape[0]
            state.dispatch_metadata_mode = dispatch_metadata_mode
            state.combine_buffer_count = combine_buffer_count
            assert bs == 8

        def allocate_timer(self, *, audit=False):
            state.timer_allocations.append(audit)
            if audit:
                return forbidden("audit allocation")()
            return Tensor(np.zeros(4, dtype=np.int64), "timer")

        communication_snapshot = forbidden("communication snapshot")

        def destroy(self):
            state.destroyed = True

    class Group:
        def rank(self):
            return 0

        def size(self):
            return 16

        def barrier(self, _stage):
            pass

        def all_gather(self, value, _stage):
            records = [deepcopy(value) for _ in range(16)]
            for rank, record in enumerate(records):
                if isinstance(record, dict) and "rank" in record:
                    record["rank"] = rank
            return records

        def close(self):
            state.closed = True

    def forward(_buffer, *, bs, timer, out):
        state.calls.append(timer is not None)
        return out, Tensor(np.zeros(2, dtype=np.int32), "counts")

    def profile(call, _group, _torch, config, _prefix, _directory, _bs):
        for _ in range(config["iters"]):
            call()
        return [dict(start_us=100.0, duration_us=17.0), dict(start_us=200.0, duration_us=19.0)]

    events = [
        dict(
            rank=0,
            core=core,
            family=f"AIV_{operation}_COMM",
            wave=0,
            row_count=3 if core < 8 else 2,
            audit_enabled=False,
            start_cycle=1000,
            end_cycle=1000 + duration,
        )
        for operation, duration in (("DISPATCH", 1000), ("COMBINE", 2000))
        for core in range(28)
    ]
    monkeypatch.setattr(api, "SymmBuffer", Buffer)
    monkeypatch.setattr(api, "communication_forward", forward)
    monkeypatch.setattr(bench, "collect_profile", profile)
    monkeypatch.setattr(comm_report, "read_events", lambda _raw, _rank: deepcopy(events))
    monkeypatch.setattr(breakdown, "compact_events", lambda _raw, _rank: [])
    monkeypatch.setattr(validation, "collective_check", lambda _group, _stage, error, *_args: error is None)
    for module, name in (
        (bench, "expected_output"),
        (bench, "expected_layout"),
        (bench, "validate_payload"),
        (wave_oracle, "plan"),
        (wave_oracle, "route_queues"),
        (comm_report, "read_transfers"),
        (comm_report, "rank_report"),
    ):
        monkeypatch.setattr(module, name, forbidden(name))
    (tmp_path / "_build_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bench, "files", lambda _package: tmp_path)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("DEEPEP_MEGAMOE_COMM_GROUP_ORDER", raising=False)
    monkeypatch.delenv("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", raising=False)
    monkeypatch.delenv("DEEPEP_MEGAMOE_COMBINE_INTRA_CORES", raising=False)
    args = SimpleNamespace(
        output_dir=tmp_path / "run",
        no_timer_raw=False,
        no_validation=True,
        dispatch_metadata_mode="dma",
        combine_buffer_count=2,
    )
    config = dict(
        model=dict(hidden=64, intermediate_hidden=128, num_experts=32, num_topk=8, num_shared_experts=0),
        bs_values=[8],
        warmup=1,
        iters=2,
        alignment=dict(rows=1, cols=1, repeats=1),
    )
    return SimpleNamespace(args=args, config=config, group=Group(), state=state)


@pytest.mark.parametrize("metadata_mode", ["dma", "scalar"])
@pytest.mark.parametrize("combine_buffer_count", [2, 4, 6, 8])
@pytest.mark.parametrize("group_order", ["reverse", "paired-xor"])
def test_performance_only_run_skips_all_validation_and_writes_real_reports(
    mocked_run,
    capsys,
    metadata_mode,
    combine_buffer_count,
    group_order,
    monkeypatch,
):
    run = mocked_run
    run.args.dispatch_metadata_mode = metadata_mode
    run.args.combine_buffer_count = combine_buffer_count
    monkeypatch.setenv("DEEPEP_MEGAMOE_COMM_GROUP_ORDER", group_order)
    if group_order == "paired-xor":
        monkeypatch.setenv("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", "12")
        monkeypatch.setenv("DEEPEP_MEGAMOE_COMBINE_INTRA_CORES", "12")
    bench.run(run.args, run.config, run.group)
    state, output = run.state, run.args.output_dir
    assert state.forbidden == []
    assert state.timer_allocations == [False]
    assert state.cpu_reads == ["timer"]
    assert state.calls == [False, False, False, False, False, True]
    assert state.prepared_rows == 64
    assert state.dispatch_metadata_mode == metadata_mode
    assert state.combine_buffer_count == combine_buffer_count
    assert state.destroyed and state.closed
    assert not list(output.rglob("*accuracy*"))
    assert not list(output.rglob("*audit*"))
    assert not list(output.rglob("*.csv"))

    manifest = json.loads((output / "rank_0/manifest.json").read_text())
    assert manifest["workload"] == "communication_only" and manifest["validation"] == "skipped"
    assert manifest["dispatch_metadata_mode"] == metadata_mode
    assert manifest["combine_buffer_count"] == combine_buffer_count
    assert manifest["group_order"] == group_order
    profile = json.loads((output / "rank_0/bs_8/profiling.json").read_text())
    assert profile["metric"] == "communication_only_kernel_duration_us"
    assert profile["rank_max_mean_us"] == 19.0
    assert len(json.loads((output / "rank_0/bs_8/kernel_samples.json").read_text())) == 2
    timer_dir = output / "bs_8/timer"
    assert (timer_dir / "rank_0_npu_time.bin").stat().st_size == 4 * 8
    assert (timer_dir / "timer_breakdown.json").is_file()
    detail = json.loads((timer_dir / "rank_00_communication.json").read_text())
    assert "_transfers" not in detail
    assert detail["validation"]["status"] == "skipped"
    assert detail["dispatch_metadata_mode"] == metadata_mode
    assert detail["combine_buffer_count"] == combine_buffer_count
    assert detail["group_order"] == group_order
    assert detail["validation"]["accounting"] == "timer_row_counts"
    assert [wave["operation"] for wave in detail["waves"]] == ["DISPATCH", "COMBINE"]
    assert all(len(wave["cores"]) == 28 for wave in detail["waves"])
    summary = json.loads((timer_dir / "communication.json").read_text())
    assert summary["validation"] == dict(strict_wave_core=False, status="skipped")
    assert summary["dispatch_metadata_mode"] == metadata_mode
    assert summary["combine_buffer_count"] == combine_buffer_count
    assert summary["group_order"] == group_order
    assert len(summary["ranks"]) == 16
    for operation, mean_core_us in (("DISPATCH", 1.0), ("COMBINE", 2.0)):
        measured = detail["operations"][operation]
        expected_bytes = 64 * comm_report.payload_bytes(64, operation)
        assert measured["rows"] == 64 and measured["bytes"] == expected_bytes
        assert measured["mean_core_us"] == mean_core_us
        assert summary["operations"][operation]["gbps_per_card_mean"] == expected_bytes / mean_core_us / 1000
        if group_order == "paired-xor":
            for name, workers in (("INTRA", 12), ("CROSS", 16)):
                cohort = measured["cohorts"][name]
                assert cohort["workers"] == workers and cohort["rows"] == 32
                assert cohort["mean_core_us"] == mean_core_us
                assert summary["operations"][operation]["cohorts"][name]["gbps_per_card_mean"] == cohort["gbps"]
        else:
            assert "cohorts" not in measured and "cohorts" not in summary["operations"][operation]
    printed = capsys.readouterr().out
    assert "validation skipped; no GMM executed" in printed
    assert f"combine_buffer_count={combine_buffer_count}" in printed
    assert f"group_order={group_order}" in printed


def test_strict_run_still_calls_oracle_before_timing(mocked_run):
    run = mocked_run
    run.args.no_validation = False
    with pytest.raises(AssertionError, match="Performance-only run called expected_output"):
        bench.run(run.args, run.config, run.group)
    assert run.state.forbidden == ["expected_output"]
    assert run.state.timer_allocations == []


@pytest.mark.parametrize("dispatch,combine", [(4, 4), (6, 6), (1, 27), (4, 6)])
def test_strict_run_passes_actual_paired_split_to_payload_oracle(mocked_run, monkeypatch, dispatch, combine):
    run = mocked_run
    run.args.no_validation = False
    monkeypatch.setenv("DEEPEP_MEGAMOE_COMM_GROUP_ORDER", "paired-xor")
    monkeypatch.setenv("DEEPEP_MEGAMOE_DISPATCH_INTRA_CORES", str(dispatch))
    monkeypatch.setenv("DEEPEP_MEGAMOE_COMBINE_INTRA_CORES", str(combine))
    observed = []

    def expected(_config, _rank, _bs, **schedule):
        observed.append(schedule)
        raise RuntimeError("payload oracle reached")

    monkeypatch.setattr(bench, "expected_output", expected)
    with pytest.raises(RuntimeError, match="payload oracle reached"):
        bench.run(run.args, run.config, run.group)
    assert observed == [dict(group_order="paired-xor", dispatch_intra=dispatch, combine_intra=combine)]
    assert run.state.timer_allocations == []
